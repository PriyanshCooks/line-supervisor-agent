"""The per-session agent runtime: one MCP subprocess for many calls, cheap graphs, clean teardown. No LLM is called."""
import json

import langchain_mcp_adapters.sessions as sessions
from langgraph.checkpoint.memory import MemorySaver

from app.session_runtime import SessionRuntime


def _text(res) -> str:
    if isinstance(res, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in res)
    return str(res)


def test_many_tool_calls_and_a_report_share_one_mcp_subprocess(mini_db, monkeypatch):
    spawns = {"n": 0}
    real = sessions.stdio_client

    def counted(*a, **k):
        spawns["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(sessions, "stdio_client", counted)

    rt = SessionRuntime().start()
    try:
        assert rt.alive and {"find_bottleneck", "list_data_sources"} <= set(rt._rt.tools)
        for _ in range(3):                                            # three "turns" worth of tool calls
            out = json.loads(_text(rt._submit(rt._rt.tools["get_line_summary"].ainvoke({"shift": "W1", "line": "L0"}))))
            assert out["line_output"] == 3 and out["window"] == "W1"
        rep = rt.report("W1", "L0", None, use_llm=False)              # the report fetches its facts through the same tools
        assert "time window 1" in rep["whatsapp"]
        assert spawns["n"] == 1                                       # was: two for setup + one per tool call, every turn
    finally:
        rt.close()
    assert rt.closed and not rt.alive and rt._rt._task.done()


def test_graph_is_rebuilt_only_for_a_new_checkpointer(mini_db):
    rt = SessionRuntime().start()
    try:
        a, b = MemorySaver(), MemorySaver()
        g1 = rt._graph(a)
        assert rt._graph(a) is g1                                     # same conversation context: reuse
        g2 = rt._graph(b)
        assert g2 is not g1 and rt._graph(b) is g2                    # new context (line/window switch): fresh, empty conversation
    finally:
        rt.close()


def test_close_is_idempotent_and_stops_the_loop(mini_db):
    rt = SessionRuntime().start()
    rt.close()
    rt.close()
    rt._thread.join(timeout=5)
    assert not rt._thread.is_alive() and not rt.alive
