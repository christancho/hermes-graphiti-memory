"""Graphiti memory provider for Hermes Agent.

Talks to a self-hosted Graphiti MCP server (FalkorDB or Neo4j behind it) over
Streamable HTTP. The graph database and the extraction LLM key stay on the
server; Hermes only holds the URL (and an optional bearer token).

- recall: before each turn, search the profile's group + shared groups and
  inject the top facts (time-boxed, skipped for trivial prompts)
- capture: one episode per session at session end (default), per turn, or off
- mirror: built-in MEMORY.md / USER.md writes are copied into the graph
- tools: graphiti_search, graphiti_remember (no delete tools by design)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread
from agent.secret_scope import get_secret  # profile-scoped secrets (never read os.environ directly)

from .mcp_client import GraphitiMCPClient, MCPError

logger = logging.getLogger(__name__)

PROVIDER_NAME = "graphiti"
CONFIG_FILE = "graphiti.json"
URL_ENV, TOKEN_ENV = "GRAPHITI_MCP_URL", "GRAPHITI_MCP_TOKEN"
CAPTURE_MODES = ("session_end", "turn", "off")

DEFAULTS: dict[str, Any] = {
    "url": "",                 # e.g. https://memory.example.com/mcp
    "group_id": "",            # "" -> derived from the profile name
    "shared_groups": [],       # opt-in: groups every profile reads (e.g. ["main"]); empty = profiles stay separate
    "recall": True,
    "recall_max_facts": 8,
    "recall_budget_seconds": 2.0,
    "capture": "session_end",
    "capture_min_user_turns": 2,
    "capture_max_chars": 8000,
    "mirror_builtin": True,
    "use_env_proxy": True,
    "timeout_seconds": 30,
}


def load_config(hermes_home: Path) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    path = Path(hermes_home) / CONFIG_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
        except (OSError, ValueError) as e:
            logger.warning("graphiti: ignoring unreadable %s: %s", path, e)
    if isinstance(cfg["shared_groups"], str):
        cfg["shared_groups"] = [g.strip() for g in cfg["shared_groups"].split(",") if g.strip()]
    if cfg["capture"] not in CAPTURE_MODES:
        cfg["capture"] = "session_end"
    return cfg


def _hermes_home_from_context() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def default_group_id(hermes_home: Path, agent_identity: str = "") -> str:
    """Profile name as the group (``default`` profile -> ``hermes``); Graphiti wants [A-Za-z0-9_-]."""
    name = agent_identity or Path(hermes_home).name
    if name in ("", ".hermes", "default"):
        name = "hermes"
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)


def format_facts(facts: list[dict], limit: int) -> str:
    lines = []
    for f in facts[:limit]:
        text = (f.get("fact") or "").strip()
        if not text:
            continue
        when = (f.get("valid_at") or "")[:10]
        lines.append(f"- {text}" + (f" (since {when})" if when else ""))
    if not lines:
        return ""
    return ("## Recalled from long-term memory (Graphiti)\n"
            "Hints from earlier sessions; verify live state before acting on them.\n" + "\n".join(lines))


def transcript_text(messages: list[dict], max_chars: int) -> tuple[str, int]:
    """User/assistant prose only (no tool output), newest kept when trimming. Returns (text, user_turns)."""
    rows, users = [], 0
    for m in messages or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, list):  # multimodal parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = (content or "").strip()
        if not content:
            continue
        if role == "user":
            users += 1
        rows.append(f"{role}: {content[:2000]}")
    text = "\n".join(rows)
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = text[text.find("\n") + 1:] if "\n" in text else text
    return text, users


class GraphitiMemoryProvider(MemoryProvider):
    """Per-profile instance; every binding is made in initialize() (multiplexing contract)."""

    def __init__(self) -> None:
        self._client: Optional[GraphitiMCPClient] = None
        self._cfg: dict[str, Any] = dict(DEFAULTS)
        self._group = "hermes"
        self._session_id = ""
        self._platform = ""
        self._primary = True
        self._lock = threading.Lock()
        self._recall: dict[str, str] = {}          # query -> formatted block
        self._recall_thread: Optional[threading.Thread] = None
        self._last_count = 0
        self._saved: set[str] = set()              # digests of episodes already written this process
        self._writers: list[threading.Thread] = []

    # -- identity / availability ---------------------------------------------------
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        if get_secret(URL_ENV):
            return True
        return bool(load_config(_hermes_home_from_context()).get("url"))

    def unavailable_reason(self) -> str:
        return (f"Set the Graphiti MCP URL: `hermes memory setup` or {URL_ENV}, "
                f"or \"url\" in $HERMES_HOME/{CONFIG_FILE}.")

    def initialize(self, session_id: str, **kwargs) -> None:
        home = Path(kwargs.get("hermes_home") or _hermes_home_from_context())
        self._cfg = load_config(home)
        url = get_secret(URL_ENV) or self._cfg["url"]
        self._group = self._cfg["group_id"] or default_group_id(home, kwargs.get("agent_identity") or "")
        self._session_id = session_id
        self._platform = kwargs.get("platform") or ""
        self._primary = (kwargs.get("agent_context") or "primary") == "primary"
        self._client = GraphitiMCPClient(url, token=get_secret(TOKEN_ENV) or "",
                                         timeout=float(self._cfg["timeout_seconds"]),
                                         use_env_proxy=bool(self._cfg["use_env_proxy"]))

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        self._session_id = new_session_id
        if reset:
            with self._lock:
                self._recall.clear()

    # -- prompt / recall ---------------------------------------------------------------
    def _groups(self) -> list[str]:
        return [self._group] + [g for g in self._cfg["shared_groups"] if g != self._group]

    def system_prompt_block(self) -> str:
        shared = [g for g in self._cfg["shared_groups"] if g != self._group]
        scope = (f"Shared groups (also read by other profiles): {', '.join(shared)}." if shared
                 else "Memory is private to this profile.")
        return ("# Long-term memory (Graphiti)\n"
                f"Your group: `{self._group}`. {scope}\n"
                "Relevant facts are recalled automatically before each turn. Use `graphiti_search` for "
                "more, and `graphiti_remember` for durable facts, decisions, preferences and root causes. "
                "Never store secrets, tokens or credentials.")

    def _search_block(self, query: str) -> str:
        facts = self._client.search_facts(query[:400], self._groups(),
                                          int(self._cfg["recall_max_facts"]),
                                          timeout=max(float(self._cfg["recall_budget_seconds"]) * 3, 5.0))
        return format_facts(facts, int(self._cfg["recall_max_facts"]))

    def _start_recall(self, query: str) -> None:
        if not (self._client and self._cfg["recall"] and self._primary) or is_trivial_prompt(query):
            return
        with self._lock:
            if query in self._recall or (self._recall_thread and self._recall_thread.is_alive()):
                return

        def job() -> None:
            try:
                block = self._search_block(query)
            except MCPError as e:
                logger.debug("graphiti recall failed: %s", e)
                block = ""
            with self._lock:
                self._recall = {query: block}

        self._recall_thread = spawn_context_thread(job, name="graphiti-recall")
        self._recall_thread.start()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_recall(message or "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._start_recall(query or "")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_count = 0
        if not (self._client and self._cfg["recall"] and self._primary) or is_trivial_prompt(query):
            return ""
        self._start_recall(query)
        t = self._recall_thread
        if t and t.is_alive():
            t.join(timeout=float(self._cfg["recall_budget_seconds"]))
        with self._lock:
            block = self._recall.get(query, "")
        self._last_count = block.count("\n- ")
        return block

    def recall_status(self) -> Optional[RecallStatus]:
        return RecallStatus("Graphiti", self._last_count) if self._last_count else None

    # -- writes --------------------------------------------------------------------
    def _write(self, *, name: str, body: str, group: str, source: str, description: str) -> None:
        """Queue one episode in the background; identical content is written once per process."""
        if not self._client or not body.strip():
            return
        digest = hashlib.sha256(f"{group}\n{body}".encode()).hexdigest()
        with self._lock:
            if digest in self._saved:
                return
            self._saved.add(digest)
            self._writers = [t for t in self._writers if t.is_alive()]

        def job() -> None:
            try:
                self._client.add_memory(name=name, body=body, group_id=group, source=source,
                                        source_description=description)
            except MCPError as e:
                with self._lock:
                    self._saved.discard(digest)  # allow a retry on the next trigger
                logger.warning("graphiti: add_memory failed (%s): %s", name, e)

        t = spawn_context_thread(job, name="graphiti-write")
        self._writers.append(t)
        t.start()

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[list[dict[str, Any]]] = None) -> None:
        if self._cfg["capture"] != "turn" or not self._primary or is_trivial_prompt(user_content):
            return
        body = f"user: {(user_content or '').strip()[:3000]}\nassistant: {(assistant_content or '').strip()[:3000]}"
        self._write(name=f"turn {self._stamp()}", body=body, group=self._group, source="message",
                    description=f"hermes turn ({self._platform or 'unknown'}) session {session_id or self._session_id}")

    def _capture_transcript(self, messages: list[dict[str, Any]], label: str) -> None:
        if self._cfg["capture"] != "session_end" or not self._primary:
            return
        text, users = transcript_text(messages, int(self._cfg["capture_max_chars"]))
        if users < int(self._cfg["capture_min_user_turns"]):
            return
        self._write(name=f"{label} {self._stamp()}", body=text, group=self._group, source="message",
                    description=f"hermes {label} ({self._platform or 'unknown'}) session {self._session_id}")

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._capture_transcript(messages, "session")

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        # Long sessions compress before they end: save the part that is about to be summarized.
        self._capture_transcript(messages, "session-part")
        return ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[dict[str, Any]] = None) -> None:
        if not self._cfg["mirror_builtin"] or action not in ("add", "replace") or not (content or "").strip():
            return
        shared = [g for g in self._cfg["shared_groups"] if g != self._group]
        group = shared[0] if (target == "user" and shared) else self._group
        kind = "user profile" if target == "user" else "agent notes"
        self._write(name=f"builtin {kind} {action}", body=content.strip(), group=group, source="text",
                    description=f"mirrored from Hermes built-in {kind} ({self._group})")

    def shutdown(self) -> None:
        for t in list(self._writers):
            t.join(timeout=10.0)  # let queued episodes reach the server

    # -- tools -----------------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": "graphiti_search",
             "description": "Search long-term memory (Graphiti knowledge graph). Returns facts (relationships "
                            "with validity dates) or entities. Searches your own group plus shared groups.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string", "description": "What to look for, in natural language."},
                 "kind": {"type": "string", "enum": ["facts", "entities"], "default": "facts"},
                 "scope": {"type": "string", "enum": ["all", "own", "shared"], "default": "all"},
                 "max_results": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10}},
                 "required": ["query"]}},
            {"name": "graphiti_remember",
             "description": "Store a durable fact, decision, preference or root cause in long-term memory. "
                            "Write plain self-contained sentences with names and dates. Processing takes "
                            "10-60 s. Never store secrets, tokens or credentials.",
             "parameters": {"type": "object", "properties": {
                 "content": {"type": "string"},
                 "name": {"type": "string", "description": "Short title for the episode."},
                 "shared": {"type": "boolean", "default": False,
                            "description": "Write to the first shared group, if one is configured "
                                           "(otherwise the profile's own group)."}},
                 "required": ["content"]}},
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._client:
            return json.dumps({"error": "graphiti provider is not initialized"})
        try:
            if tool_name == "graphiti_search":
                scope = args.get("scope") or "all"
                shared = [g for g in self._cfg["shared_groups"] if g != self._group]
                groups = {"own": [self._group], "shared": shared or [self._group]}.get(scope, self._groups())
                n = max(1, min(int(args.get("max_results") or 10), 30))
                if (args.get("kind") or "facts") == "entities":
                    nodes = self._client.search_nodes(args["query"], groups, n)
                    return json.dumps({"groups": groups, "entities": [
                        {"name": x.get("name"), "summary": x.get("summary"), "group": x.get("group_id")}
                        for x in nodes]})
                facts = self._client.search_facts(args["query"], groups, n)
                return json.dumps({"groups": groups, "facts": [
                    {"fact": x.get("fact"), "valid_at": x.get("valid_at"), "invalid_at": x.get("invalid_at"),
                     "group": x.get("group_id")} for x in facts]})
            if tool_name == "graphiti_remember":
                content = (args.get("content") or "").strip()
                if not content:
                    return json.dumps({"error": "content is empty"})
                shared = [g for g in self._cfg["shared_groups"] if g != self._group]
                group = shared[0] if (args.get("shared") and shared) else self._group
                msg = self._client.add_memory(name=(args.get("name") or "note")[:120], body=content,
                                              group_id=group, source="text",
                                              source_description=f"graphiti_remember ({self._group})")
                return json.dumps({"ok": True, "group": group, "message": msg})
        except MCPError as e:
            return json.dumps({"error": str(e)})
        except (KeyError, ValueError, TypeError) as e:
            return json.dumps({"error": f"bad arguments: {e}"})
        return json.dumps({"error": f"unknown tool {tool_name}"})

    # -- setup ----------------------------------------------------------------------
    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "url", "description": "Graphiti MCP endpoint (Streamable HTTP), e.g. https://memory.example.com/mcp",
             "required": True},
            {"key": "group_id", "description": "This profile's group ('' = profile name, default profile -> hermes)",
             "default": ""},
            {"key": "shared_groups", "description": "Optional comma-separated groups shared by several profiles "
                                                    "(read by each; USER.md mirrors go to the first). Empty keeps "
                                                    "profiles separate", "default": ""},
            {"key": "capture", "description": "What to store automatically", "default": "session_end",
             "choices": list(CAPTURE_MODES)},
            {"key": "token", "description": "Bearer token if your proxy requires one (optional)", "secret": True,
             "required": False, "env_var": TOKEN_ENV},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / CONFIG_FILE
        current: dict[str, Any] = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError):
                current = {}
        for k, v in values.items():
            if k in DEFAULTS:
                current[k] = [g.strip() for g in v.split(",") if g.strip()] if k == "shared_groups" and isinstance(v, str) else v
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def register(ctx) -> None:
    ctx.register_memory_provider(GraphitiMemoryProvider())  # new instance per load (multiplexing)
