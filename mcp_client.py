"""Minimal MCP (Streamable HTTP) client for a Graphiti MCP server. Stdlib only.

Speaks JSON-RPC over HTTP POST, accepts JSON or SSE responses, keeps the
``Mcp-Session-Id`` and re-initializes once when the server drops the session.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    """Transport, protocol or tool error from the MCP server."""


def _origin(url: str) -> tuple[str, str, int]:
    p = urllib.parse.urlsplit(url)
    scheme = (p.scheme or "").lower()
    return scheme, (p.hostname or "").lower(), p.port or (443 if scheme == "https" else 80)


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only within the same scheme, host and port.

    urllib's default handler follows cross-host redirects and copies every request header,
    including ``Authorization``, to the new host. A misconfigured or compromised server could
    then send the bearer token anywhere (or downgrade https to http). Anything that leaves the
    original origin is refused and surfaces as an HTTPError instead.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(newurl) != _origin(req.full_url):
            raise urllib.error.HTTPError(
                req.full_url, code, f"refused cross-origin redirect to {_origin(newurl)[1] or newurl}",
                headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class GraphitiMCPClient:
    def __init__(self, url: str, *, token: str = "", timeout: float = 30.0,
                 use_env_proxy: bool = True, client_name: str = "hermes-graphiti-memory",
                 client_version: str = "0.1.2"):
        self.url = url.rstrip("/")  # Graphiti answers /mcp; /mcp/ 307-redirects to plain http
        self.token = token
        self.timeout = timeout
        self._client_info = {"name": client_name, "version": client_version}
        handlers: list = [SameOriginRedirectHandler()]
        if not use_env_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)
        self._session_id: Optional[str] = None
        self._lock = threading.Lock()  # one in-flight session handshake at a time
        self._next_id = 0

    # -- transport -----------------------------------------------------------------
    def _post(self, body: dict, timeout: float) -> tuple[Optional[dict], dict]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
        with self._opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return _parse_body(raw, body.get("id")), resp_headers

    def _ensure_session(self, timeout: float) -> None:
        if self._session_id:
            return
        with self._lock:
            if self._session_id:
                return
            msg, hdrs = self._post({"jsonrpc": "2.0", "id": self._new_id(), "method": "initialize",
                                    "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                               "clientInfo": self._client_info}}, timeout)
            if not msg or "result" not in msg:
                raise MCPError(f"initialize failed: {msg}")
            self._session_id = hdrs.get("mcp-session-id") or ""
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout)

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _rpc(self, method: str, params: dict, timeout: Optional[float] = None) -> dict:
        timeout = timeout or self.timeout
        for attempt in (1, 2):
            try:
                self._ensure_session(timeout)
                msg, _ = self._post({"jsonrpc": "2.0", "id": self._new_id(), "method": method, "params": params}, timeout)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404) and attempt == 1 and self._session_id:
                    self._session_id = None  # server restarted / session expired: re-handshake once
                    continue
                raise MCPError(f"HTTP {e.code} from {self.url}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise MCPError(f"cannot reach {self.url}: {e}") from e
            if msg is None:
                raise MCPError(f"empty response to {method}")
            if "error" in msg:
                raise MCPError(f"{method}: {msg['error'].get('message', msg['error'])}")
            return msg.get("result") or {}
        raise MCPError(f"{method}: session could not be re-established")

    # -- tools ---------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict, timeout: Optional[float] = None) -> Any:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)
        if result.get("isError"):
            text = " ".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
            raise MCPError(f"{name}: {text or 'tool error'}")
        payload = result.get("structuredContent")
        if payload is None:  # older servers only send text content holding JSON
            for c in result.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    try:
                        payload = json.loads(c["text"])
                    except (ValueError, KeyError):
                        payload = {"message": c.get("text", "")}
                    break
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]  # Graphiti wraps tool output as {"result": {...}}
        if isinstance(payload, dict) and payload.get("error"):
            raise MCPError(f"{name}: {payload['error']}")
        return payload

    def status(self, timeout: Optional[float] = None) -> dict:
        return self.call_tool("get_status", {}, timeout) or {}

    def search_facts(self, query: str, group_ids: list[str], max_facts: int = 8,
                     timeout: Optional[float] = None) -> list[dict]:
        out = self.call_tool("search_memory_facts", {"query": query, "group_ids": group_ids,
                                                     "max_facts": max_facts}, timeout)
        return list((out or {}).get("facts") or [])

    def search_nodes(self, query: str, group_ids: list[str], max_nodes: int = 8,
                     timeout: Optional[float] = None) -> list[dict]:
        out = self.call_tool("search_nodes", {"query": query, "group_ids": group_ids,
                                              "max_nodes": max_nodes}, timeout)
        return list((out or {}).get("nodes") or [])

    def add_memory(self, *, name: str, body: str, group_id: str, source: str = "text",
                   source_description: str = "", timeout: Optional[float] = None) -> str:
        out = self.call_tool("add_memory", {"name": name, "episode_body": body, "group_id": group_id,
                                            "source": source, "source_description": source_description}, timeout)
        return str((out or {}).get("message") or "queued")


def _parse_body(raw: str, want_id: Any) -> Optional[dict]:
    """Return the JSON-RPC message for ``want_id`` from a JSON or SSE body (None for 202/empty)."""
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        return json.loads(raw)
    found = None
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                msg = json.loads(line[5:].strip())
            except ValueError:
                continue
            if want_id is None or msg.get("id") == want_id:
                found = msg
    return found
