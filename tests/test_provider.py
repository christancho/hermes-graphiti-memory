import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import FakeGraphiti, hermes_available, load_plugin

plugin = load_plugin()
from graphiti_plugin_under_test.mcp_client import GraphitiMCPClient, MCPError, _parse_body  # noqa: E402


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class ClientTests(unittest.TestCase):
    def test_parse_sse_and_json(self):
        self.assertEqual(_parse_body('{"id":1,"result":{}}', 1)["id"], 1)
        sse = 'event: message\ndata: {"id":2,"result":{"x":1}}\n\n'
        self.assertEqual(_parse_body(sse, 2)["result"], {"x": 1})
        self.assertIsNone(_parse_body("", 3))

    def test_roundtrip_json_and_sse(self):
        for sse in (True, False):
            fake = FakeGraphiti(sse=sse)
            try:
                c = GraphitiMCPClient(fake.url, use_env_proxy=False)
                self.assertEqual(c.status()["status"], "ok")
                c.add_memory(name="n", body="Chris prefers FalkorDB", group_id="main")
                self.assertEqual(c.search_facts("prefers falkordb", ["main"])[0]["group_id"], "main")
            finally:
                fake.close()

    def test_trailing_slash_is_stripped(self):
        self.assertTrue(GraphitiMCPClient("https://x/mcp/").url.endswith("/mcp"))

    def test_reinitializes_after_session_expiry(self):
        fake = FakeGraphiti()
        try:
            c = GraphitiMCPClient(fake.url, use_env_proxy=False)
            c.status()
            fake.expire_next = True
            self.assertEqual(c.status()["status"], "ok")
            self.assertEqual(len(fake.sessions), 1)  # old dropped, new one created
        finally:
            fake.close()

    def test_bearer_token_sent(self):
        fake = FakeGraphiti()
        try:
            GraphitiMCPClient(fake.url, token="t0k", use_env_proxy=False).status()
            self.assertEqual(fake.headers_seen[-1].get("Authorization"), "Bearer t0k")
        finally:
            fake.close()

    def test_unreachable_raises_mcperror(self):
        with self.assertRaises(MCPError):
            GraphitiMCPClient("http://127.0.0.1:9/mcp", timeout=1, use_env_proxy=False).status()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGraphiti()
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "profiles" / "devops"
        self.home.mkdir(parents=True)
        (self.home / "graphiti.json").write_text(json.dumps({"url": self.fake.url, "use_env_proxy": False}))
        self.p = plugin.GraphitiMemoryProvider()
        self.p.initialize("sess-1", hermes_home=str(self.home), platform="cli", agent_context="primary")

    def tearDown(self):
        self.p.shutdown()
        self.fake.close()
        self.tmp.cleanup()

    def test_group_from_profile_name(self):
        self.assertEqual(self.p._group, "devops")
        self.assertEqual(plugin.default_group_id(Path("/x/.hermes")), "hermes")
        self.assertEqual(plugin.default_group_id(Path("/x"), "default"), "hermes")

    def test_available_from_config(self):
        os.environ["HERMES_HOME"] = str(self.home)
        try:
            if not hermes_available():
                self.assertTrue(self.p.is_available())
        finally:
            os.environ.pop("HERMES_HOME", None)

    def test_remember_and_search_tools(self):
        out = json.loads(self.p.handle_tool_call("graphiti_remember", {"content": "Grafana runs on host-b", "shared": True}))
        self.assertEqual(out["group"], "main")
        out = json.loads(self.p.handle_tool_call("graphiti_search", {"query": "where does grafana run"}))
        self.assertEqual(out["groups"], ["devops", "main"])
        self.assertIn("Grafana runs on host-b", out["facts"][0]["fact"])
        out = json.loads(self.p.handle_tool_call("graphiti_search", {"query": "x", "kind": "entities", "scope": "own"}))
        self.assertEqual(out["groups"], ["devops"])
        self.assertIn("error", json.loads(self.p.handle_tool_call("graphiti_remember", {"content": "  "})))
        self.assertIn("error", json.loads(self.p.handle_tool_call("nope", {})))

    def test_prefetch_recalls_facts_and_skips_trivial(self):
        self.fake.tool("add_memory", {"name": "a", "episode_body": "Portainer stack agentic-memory runs on atlas",
                                      "group_id": "main"})
        block = self.p.prefetch("which host runs the agentic-memory stack?")
        self.assertIn("Recalled from long-term memory", block)
        self.assertIn("atlas", block)
        self.assertEqual(self.p.recall_status().count, 1)
        self.assertEqual(self.p.prefetch("thanks!"), "")
        self.assertIsNone(self.p.recall_status())

    def test_prefetch_survives_server_down(self):
        self.fake.close()
        start = time.time()
        self.assertEqual(self.p.prefetch("anything about the homelab network"), "")
        self.assertLess(time.time() - start, 3.0)  # bounded by recall_budget_seconds

    def test_session_end_writes_one_transcript_episode(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "deploy graphiti"},
                {"role": "assistant", "content": "done on atlas"},
                {"role": "tool", "content": "SECRET tool output"},
                {"role": "user", "content": "now test restart"},
                {"role": "assistant", "content": "data survived"}]
        self.p.on_session_end(msgs)
        self.p.on_session_end(msgs)  # duplicate content is written once
        self.assertTrue(wait_for(lambda: len(self.fake.episodes) == 1))
        time.sleep(0.2)
        ep = self.fake.episodes[0]
        self.assertEqual((len(self.fake.episodes), ep["group_id"], ep["source"]), (1, "devops", "message"))
        self.assertNotIn("SECRET", ep["episode_body"])
        self.assertTrue(ep["episode_body"].startswith("user: deploy graphiti"))

    def test_short_sessions_and_subagents_are_not_captured(self):
        self.p.on_session_end([{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello"}])
        sub = plugin.GraphitiMemoryProvider()
        sub.initialize("sess-2", hermes_home=str(self.home), agent_context="subagent")
        sub.on_session_end([{"role": "user", "content": f"q{i}"} for i in range(5)])
        self.assertEqual(sub.prefetch("tell me about the homelab"), "")
        time.sleep(0.3)
        self.assertEqual(self.fake.episodes, [])

    def test_turn_capture_mode(self):
        (self.home / "graphiti.json").write_text(json.dumps({"url": self.fake.url, "use_env_proxy": False,
                                                             "capture": "turn"}))
        p = plugin.GraphitiMemoryProvider()
        p.initialize("s", hermes_home=str(self.home))
        p.sync_turn("ok", "fine")  # trivial: skipped
        p.sync_turn("switch the db to falkordb", "switched")
        p.on_session_end([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])  # not in turn mode
        self.assertTrue(wait_for(lambda: len(self.fake.episodes) == 1))
        time.sleep(0.2)
        self.assertEqual(len(self.fake.episodes), 1)

    def test_builtin_mirror_routes_user_to_shared(self):
        self.p.on_memory_write("add", "user", "Chris prefers short replies")
        self.p.on_memory_write("add", "memory", "Docker host atlas runs Portainer")
        self.p.on_memory_write("remove", "memory", "whatever")
        self.assertTrue(wait_for(lambda: len(self.fake.episodes) == 2))
        groups = sorted(e["group_id"] for e in self.fake.episodes)
        self.assertEqual(groups, ["devops", "main"])

    def test_save_config_merges(self):
        self.p.save_config({"shared_groups": "main, family", "capture": "off", "token": "x"}, str(self.home))
        data = json.loads((self.home / "graphiti.json").read_text())
        self.assertEqual(data["shared_groups"], ["main", "family"])
        self.assertEqual(data["capture"], "off")
        self.assertNotIn("token", data)  # secrets go to .env, never the json
        self.assertEqual(data["url"], self.fake.url)


@unittest.skipUnless(hermes_available(), "Hermes Agent not installed")
class HermesMemoryManagerTests(unittest.TestCase):
    """Run the provider through Hermes's real MemoryManager (tool routing + lifecycle)."""

    def test_manager_lifecycle(self):
        from agent.memory_manager import MemoryManager
        fake = FakeGraphiti()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "graphiti.json").write_text(json.dumps({"url": fake.url, "use_env_proxy": False}))
            old_home = os.environ.get("HERMES_HOME")
            os.environ["HERMES_HOME"] = str(home)  # the caller's profile must be the one the provider is bound to
            try:
                mgr = MemoryManager()
                mgr.add_provider(plugin.GraphitiMemoryProvider())
                mgr.initialize_all(session_id="t-1", platform="cli", hermes_home=str(home))
                names = {s.get("name") or s.get("function", {}).get("name") for s in mgr.get_all_tool_schemas()}
                self.assertTrue({"graphiti_search", "graphiti_remember"} <= names)
                out = json.loads(mgr.handle_tool_call("graphiti_remember", {"content": "Loki listens on 3100"}))
                self.assertTrue(out.get("ok"), out)
                mgr.on_session_end([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                                    {"role": "user", "content": "u2"}])
                mgr.shutdown_all()
                self.assertTrue(wait_for(lambda: len(fake.episodes) == 2))
            finally:
                fake.close()
                if old_home is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = old_home


if __name__ == "__main__":
    unittest.main()
