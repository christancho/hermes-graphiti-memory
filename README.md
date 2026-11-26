# hermes-graphiti-memory

Long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent), stored in a **self-hosted [Graphiti](https://github.com/getzep/graphiti) knowledge graph** and reached through Graphiti's MCP server.

The graph database (FalkorDB or Neo4j) and the extraction LLM key stay on your server. Hermes only needs the MCP URL. The plugin has **no Python dependencies**: it uses the standard library only.

```
Hermes profile ──HTTP (MCP)──▶ Graphiti MCP server ──▶ FalkorDB / Neo4j
  recall / capture / tools         entity + fact extraction (LLM)
```

## What it does

| Hook | Behavior |
|---|---|
| **Recall** (before each turn) | Searches the profile's group plus shared groups for facts related to your message and injects the top results. Time-boxed (default 2 s) and skipped for trivial prompts ("ok", "thanks", slash commands). |
| **Capture** (default: end of session) | Saves one episode per session: user and assistant text only, never tool output. Long sessions also save the part that is about to be compressed. Sessions with fewer than 2 user turns, subagents and cron runs are skipped. |
| **Mirror** | Writes to Hermes's built-in memory (`MEMORY.md` / `USER.md`) are copied into the graph: `user` entries go to the first shared group, `memory` entries to the profile's group. |
| **Tools** | `graphiti_search` (facts or entities, scope own / shared / all) and `graphiti_remember`. There are **no delete tools**, so a model can't wipe memory. |

### Groups

Each Hermes profile gets its own Graphiti group, named after the profile (the `default` profile maps to `hermes`). Every profile also **reads** the shared groups (default `main`) and writes to them only on purpose (`graphiti_remember` with `shared: true`, or a mirrored `USER.md` entry). That gives several bots one common memory of you plus their own working knowledge.

## Requirements

- Hermes Agent 0.23.4 or newer (declared as `requires_hermes` in `plugin.yaml`).
- A Graphiti MCP server reachable over Streamable HTTP, e.g. `mcp_server/docker/docker-compose-falkordb.yml` from the Graphiti repo. Use an image that persists data (FalkorDB with `appendonly yes`): see [getzep/graphiti#2117](https://github.com/getzep/graphiti/issues/2117).

## Install

```bash
hermes plugins install christancho/hermes-graphiti-memory
hermes memory setup          # pick "graphiti", enter the MCP URL
hermes memory status         # Provider: graphiti, Status: available ✓
```

Per profile: `hermes -p <profile> memory setup`. Restart the gateway (or start a new session) to activate it.

If you already added the same server under `mcp_servers:`, remove that entry (`hermes mcp remove <name>`). Otherwise the model sees two sets of memory tools.

> **Don't name an MCP server `memory`.** That name collides with Hermes's built-in `memory` toolset. If the server is disabled, Hermes drops the whole `memory` toolset, which removes the built-in memory tool **and** this provider's tools (the log says `the 'memory' toolset is gated off`). Remove the entry, or name it e.g. `graphiti`.

Session capture runs when a session really ends for Hermes, not when a client disconnects. For gateway-hosted chats (Desktop, TUI, messaging) that happens when the session closes or is evicted, reset or compressed.

## Configuration

`$HERMES_HOME/graphiti.json` (written by `hermes memory setup`; edit for the advanced keys):

```json
{
  "url": "https://memory.example.com/mcp",
  "group_id": "",
  "shared_groups": ["main"],
  "recall": true,
  "recall_max_facts": 8,
  "recall_budget_seconds": 2.0,
  "capture": "session_end",
  "capture_min_user_turns": 2,
  "capture_max_chars": 8000,
  "mirror_builtin": true,
  "use_env_proxy": true,
  "timeout_seconds": 30
}
```

| Key | Default | Meaning |
|---|---|---|
| `url` | — | Graphiti MCP endpoint. Use `/mcp` **without** a trailing slash: Graphiti redirects `/mcp/` to plain http. `GRAPHITI_MCP_URL` overrides it. |
| `group_id` | profile name | This profile's group (`default` profile → `hermes`). |
| `shared_groups` | `["main"]` | Groups every profile reads. |
| `capture` | `session_end` | `session_end`, `turn` (one episode per turn: more recall, more LLM cost), or `off`. |
| `recall_budget_seconds` | `2.0` | Maximum time a turn waits for recall before going on without it. |
| `mirror_builtin` | `true` | Copy built-in memory writes into the graph. |
| `use_env_proxy` | `true` | Honor `HTTPS_PROXY` (e.g. a credential-injecting proxy). Set `false` for direct LAN access. |

Secrets: `GRAPHITI_MCP_TOKEN` (optional), sent as `Authorization: Bearer …`, goes in `$HERMES_HOME/.env`. Graphiti's MCP server has no authentication of its own, so put a reverse proxy that checks a token (or an identity-aware proxy) in front of it if it's reachable by anything other than your agents.

## Privacy and cost

**Disclosure.** The plugin talks only to the Graphiti MCP server URL you configure (and through `HTTPS_PROXY` if `use_env_proxy` is on). It runs short-lived background threads for recall and writes; nothing keeps running after a session ends. It reads only its own `$HERMES_HOME/graphiti.json`; the optional bearer token is read through Hermes's secret route (`GRAPHITI_MCP_TOKEN`), never from a file. It runs no shell commands, stores no credentials and sends no telemetry.

- **What leaves Hermes:** user and assistant text from captured sessions, `graphiti_remember` content and mirrored built-in memory entries. Tool calls and tool output are never sent. Hermes masks recognizable secrets before handing anything to a memory provider.
- **What leaves your server:** Graphiti sends every episode to its configured LLM and embedding provider for extraction. Use a local model (e.g. via an OpenAI-compatible endpoint) if memories must not leave your network.
- **Cost:** each episode triggers several LLM calls on the server. `session_end` capture keeps that to one episode per conversation; `turn` multiplies it.

## Development

```bash
python -m unittest                              # stubs Hermes if it isn't installed
HERMES_AGENT_DIR=~/.hermes/hermes-agent <hermes-venv-python> -m unittest   # also runs the MemoryManager test
GRAPHITI_MCP_URL=https://memory.example.com/mcp python -m tests.live_check  # real server, throwaway group
```

The tests start a fake Graphiti MCP server (JSON and SSE replies, session expiry) and cover recall, capture, mirroring, tools, config and Hermes's `MemoryManager` lifecycle.

## License

MIT
