"""Live check against a real Graphiti MCP server. Usage:
    GRAPHITI_MCP_URL=https://memory.example.com/mcp python -m tests.live_check
Writes to a throwaway group (graphiti-plugin-livecheck) and clears it at the end.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tests.helpers import load_plugin

plugin = load_plugin()
from graphiti_plugin_under_test.mcp_client import GraphitiMCPClient  # noqa: E402

URL = os.environ.get("GRAPHITI_MCP_URL") or sys.exit("set GRAPHITI_MCP_URL")
GROUP = "graphiti-plugin-livecheck"

with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp)
    (home / "graphiti.json").write_text(json.dumps({"url": URL, "group_id": GROUP, "shared_groups": [],
                                                    "use_env_proxy": os.environ.get("USE_ENV_PROXY", "1") == "1"}))
    p = plugin.GraphitiMemoryProvider()
    p.initialize("live-1", hermes_home=str(home), platform="cli")
    print("status:", p._client.status().get("message"))
    print("remember:", json.loads(p.handle_tool_call("graphiti_remember", {
        "name": "livecheck", "content": "The livecheck service for the Graphiti plugin runs on host orion since November 24, 2026."})))
    p.on_session_end([{"role": "user", "content": "Move the backups to the NAS named vega"},
                      {"role": "assistant", "content": "Done: nightly backups now go to vega at 02:00."},
                      {"role": "user", "content": "And keep 14 days of history"},
                      {"role": "assistant", "content": "Retention on vega set to 14 days."}])
    p.shutdown()
    t0, block = time.time(), ""
    while time.time() - t0 < 150 and not ("orion" in block and "vega" in block):
        time.sleep(10)
        block = p.prefetch(f"where do the livecheck service and backups run? ({int(time.time())})")
    print(f"recall after {int(time.time() - t0)}s (count={p.recall_status().count if p.recall_status() else 0}):\n{block}")
    ok = "orion" in block and "vega" in block
    c = GraphitiMCPClient(URL, use_env_proxy=os.environ.get("USE_ENV_PROXY", "1") == "1")
    print("cleanup:", c.call_tool("clear_graph", {"group_ids": [GROUP]}))
    print("LIVE CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
