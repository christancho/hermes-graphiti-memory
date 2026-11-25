"""Shared test helpers: load the plugin as a package, fake Graphiti MCP server, Hermes stubs."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERMES_AGENT_DIR = Path(os.environ.get("HERMES_AGENT_DIR", Path.home() / ".hermes" / "hermes-agent"))


def _install_stubs() -> None:
    """Minimal stand-ins for agent.memory_provider / agent.secret_scope when Hermes isn't installed (CI)."""
    import re
    from abc import ABC, abstractmethod
    from dataclasses import dataclass

    agent = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    @dataclass(frozen=True)
    class RecallStatus:
        provider_label: str
        count: int
        glyph: str = "🧠"

    trivial = re.compile(r"^(yes|no|ok|okay|thanks|hi|hey|hello|done|continue|go ahead|k)[\s!?.]*$", re.I)

    def is_trivial_prompt(text):
        s = (text or "").strip()
        return not s or s.startswith("/") or bool(trivial.match(s))

    def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
        return threading.Thread(target=target, args=args, kwargs=kwargs or {}, name=name, daemon=daemon)

    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self): ...

        @abstractmethod
        def is_available(self): ...

        @abstractmethod
        def initialize(self, session_id, **kwargs): ...

        @abstractmethod
        def get_tool_schemas(self): ...

    mp.MemoryProvider, mp.RecallStatus = MemoryProvider, RecallStatus
    mp.is_trivial_prompt, mp.spawn_context_thread = is_trivial_prompt, spawn_context_thread
    ss = types.ModuleType("agent.secret_scope")
    ss.get_secret = lambda name, default=None: os.environ.get(name, default)
    sys.modules.update({"agent": agent, "agent.memory_provider": mp, "agent.secret_scope": ss})


def hermes_available() -> bool:
    return (HERMES_AGENT_DIR / "agent" / "memory_provider.py").exists() and os.environ.get("GRAPHITI_TEST_STUBS") != "1"


if hermes_available():
    sys.path.insert(0, str(HERMES_AGENT_DIR))
else:
    _install_stubs()


def load_plugin():
    name = "graphiti_plugin_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "__init__.py",
                                                  submodule_search_locations=[str(REPO)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeGraphiti:
    """In-memory Graphiti MCP server: facts are stored episodes; search = substring match."""

    def __init__(self, sse: bool = True):
        self.sse = sse
        self.episodes: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.sessions: set[str] = set()
        self.expire_next = False
        self.headers_seen: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.headers_seen.append(dict(self.headers))
                method, mid = body.get("method"), body.get("id")
                sid = self.headers.get("Mcp-Session-Id")
                if method == "initialize":
                    new = f"s{len(fake.sessions) + 1}"
                    fake.sessions.add(new)
                    return self._reply(mid, {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "serverInfo": {"name": "fake"}}, {"mcp-session-id": new})
                if fake.expire_next or sid not in fake.sessions:
                    fake.expire_next = False
                    fake.sessions.discard(sid)
                    self.send_response(404)
                    self.end_headers()
                    return
                if mid is None:  # notification
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == "tools/call":
                    name, args = body["params"]["name"], body["params"]["arguments"]
                    fake.calls.append((name, args))
                    return self._reply(mid, {"structuredContent": {"result": fake.tool(name, args)}, "content": []})
                self._reply(mid, None, error={"code": -32601, "message": "nope"})

            def _reply(self, mid, result, extra=None, error=None):
                msg = {"jsonrpc": "2.0", "id": mid}
                msg.update({"error": error} if error else {"result": result})
                data = (f"event: message\ndata: {json.dumps(msg)}\n\n" if fake.sse else json.dumps(msg)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if fake.sse else "application/json")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tool(self, name: str, args: dict):
        if name == "get_status":
            return {"status": "ok", "message": "fake graphiti"}
        if name == "add_memory":
            self.episodes.append(args)
            return {"message": f"Episode '{args['name']}' queued for processing in group '{args['group_id']}'"}
        if name == "search_memory_facts":
            q = args["query"].lower()
            facts = [{"fact": e["episode_body"], "group_id": e["group_id"], "valid_at": "2026-11-02T00:00:00Z"}
                     for e in self.episodes if e["group_id"] in args["group_ids"]
                     and any(w in e["episode_body"].lower() for w in q.split() if len(w) > 3)]
            return {"message": "ok", "facts": facts[: args.get("max_facts", 10)]}
        if name == "search_nodes":
            return {"message": "ok", "nodes": [{"name": "FalkorDB", "summary": "graph db", "group_id": "main"}]}
        return {"error": f"unknown tool {name}"}

    def close(self):
        self.server.shutdown()
