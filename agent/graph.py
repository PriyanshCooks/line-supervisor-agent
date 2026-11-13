"""LangGraph supervisor agent.

agent -> (tool calls?) -> tools -> agent ...
Read-only metric tools come from the MCP server. `propose_action` is the only side-effecting
capability, and it is gated by a human-in-the-loop `interrupt()`: nothing executes until a
person approves.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from agent import policy, tracing
from agent.llm import get_llm

ROOT = Path(__file__).resolve().parents[1]
ACTION_LOG = ROOT / "data" / "actions.log"

SYSTEM = """You are a manufacturing line-supervisor assistant. Production data is only reachable through your tools.

DATA CATALOG (live, from the tools):
{catalog}

Rules:
- Ground every number in tool results. Never guess metrics; call a tool. Shift ids and line ids come from the catalog above.
- Sources: 'bosch' = real production data; 'synthetic' = a Bosch-format development sample (say so if you use it); 'vision' = a short camera clip.
  If the user does not name a source, use bosch if it is listed, otherwise synthetic. Questions about the camera clip use source='vision'.
- Time units: never convert them to hours, minutes, days or seconds, even if the user asks: say Bosch timestamps are anonymised time units that cannot be converted, then give the value in time units (and ticks). Never write a number followed by "hours". Quote values with their unit, e.g. '0.02 time units'.
- Statistic: every number you report must name its statistic, taken from the tool's `stat` field and field names (average queue, throughput, median dwell in ticks, failure rate, count). Never call a mean or p90 a "median", and never mix them up.
- Windows: ids like W5 are 'time window 5' (an equal slice of the anonymised timeline), never weeks or days. Use the tool's `window` and `window_label` fields.
- Single-station numbers (median dwell in ticks, queue, throughput) come from get_station_metrics, not cycle_time_trend.
- Counts: when asked for a count (line output, visits, throughput), state the exact integer from the tool, using the tool's `line_output_text` / `count_text` wording, in the first sentence.
- Bottleneck = the station with the longest average queue (time-averaged parts waiting between entering the station and entering the next one); report throughput (parts per window) and median dwell in ticks (1 tick = 0.01 time units) alongside. Mean dwell is not the ranking key because Bosch timestamps have 0.01 resolution.
- 'dwell' = time until a unit reaches its next recorded station (includes waiting). Do not call it a machine cycle time.
- If find_bottleneck returns refused=true, say no bottleneck can be determined, give the reason, and do NOT name a station.
- For the camera clip, use get_station_events(source='vision'): report the unit leaving one station and entering another, and
  say there are no complete cycles, so no cycle time or bottleneck. Do not invent more.
- QC failure results are associations (Response=1 for the whole part), not proven causes.
- You reason about STATIONS and the LINE only. Never discuss or rank individual workers.
- If a tool returns an error, read valid_values / hint, fix your arguments and retry.
- Call propose_action ONLY when the user explicitly asks you to alert someone or flag/escalate a station. For questions about metrics, causes or what to rebalance, answer in text and mention suggested actions in words; do not call propose_action. When the user DOES ask you to alert/flag/escalate, you MUST call propose_action (if they refer to "the bottleneck", call find_bottleneck first, then call propose_action with that station id); never write that you alerted, flagged or proposed something unless you called propose_action. A human must confirm; never claim an action was done unless the tool says it was approved and executed.
- Be concise: lead with the answer, then 2-3 pieces of evidence, then a suggested action."""


def catalog_text(cat: dict) -> str:
    rows = []
    for s in cat.get("sources", []):
        st = s["stations"]
        rows.append(f"- source={s['source']} line={s['line']}: {len(st)} stations ({st[0]}..{st[-1]}), shifts {', '.join(s['shifts'])}, "
                    f"time unit '{s['time_unit']}', {s['visits']} visits" + (f". {s['notes'][0]}" if s["notes"] else ""))
    return "\n".join(rows) or "- (no data loaded)"


@tool
def propose_action(action: Literal["alert_line_lead", "flag_for_rebalancing"], station_id: str, reason: str) -> str:
    """Propose an operational action. Requires human approval before it is executed."""
    return "pending"  # executed in the gate node after approval


def _tool_text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def mcp_client() -> MultiServerMCPClient:
    """Spawns the MCP server over stdio. Full env is passed so LINE_DB selects the dataset."""
    return MultiServerMCPClient({"metrics": {
        "command": sys.executable, "args": ["-m", "mcp_server.server"],
        "transport": "stdio", "cwd": str(ROOT), "env": dict(os.environ)}})


async def build_agent(checkpointer=None):
    client = mcp_client()
    mcp_tools = await client.get_tools()
    by_name = {t.name: t for t in mcp_tools}
    cat = json.loads(_tool_text(await by_name["list_data_sources"].ainvoke({})))   # injected so the model needn't spend a call
    system = SYSTEM.format(catalog=catalog_text(cat))
    llm = get_llm().bind_tools([*mcp_tools, propose_action])

    async def agent_node(state: MessagesState):
        msgs = state["messages"]
        resp = await llm.ainvoke([SystemMessage(system), *msgs])
        # Action policy (rules and phrasing live in agent/policy.py; both directions found by the real-data eval):
        #   force: "Alert the line lead about station 15" was once answered with a metrics analysis and no proposal
        #   block: "Which station should I rebalance...?" once made the 4B model call propose_action unprompted (1 of 16)
        turn = msgs[max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=0):]
        text = _tool_text(turn[0].content) if turn else ""
        called = any(c["name"] == "propose_action" for m in turn if isinstance(m, AIMessage) for c in m.tool_calls)
        names = [c["name"] for c in (getattr(resp, "tool_calls", None) or [])]
        if policy.should_force_action(text, called, bool(names)):
            resp = await llm.ainvoke([SystemMessage(system), *msgs, SystemMessage(policy.force_nudge(text))])
        elif policy.should_block_action(text, names):
            resp = await llm.ainvoke([SystemMessage(system), *msgs, SystemMessage(policy.BLOCK_NUDGE)])
            if any(c["name"] == "propose_action" for c in (getattr(resp, "tool_calls", None) or [])):    # still insists: drop the call
                keep = [c for c in resp.tool_calls if c["name"] != "propose_action"]
                resp = AIMessage(content=resp.content or policy.BLOCK_FALLBACK, tool_calls=keep)
        return {"messages": [resp]}

    async def tools_node(state: MessagesState):
        last: AIMessage = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            if call["name"] == "propose_action":
                a = call["args"]
                decision = interrupt({"type": "approval_request", **a})
                if decision.get("approved"):
                    ACTION_LOG.parent.mkdir(exist_ok=True)
                    with ACTION_LOG.open("a") as f:
                        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **a}) + "\n")
                    msg = f"APPROVED and executed: {a['action']} for {a['station_id']}."
                else:
                    msg = "REJECTED by human supervisor. Do not retry; acknowledge and offer an alternative."
            elif call["name"] in by_name:
                msg = _tool_text(await by_name[call["name"]].ainvoke(call["args"]))
            else:
                msg = json.dumps({"error": f"unknown tool {call['name']}", "available": list(by_name)})
            out.append(ToolMessage(content=msg, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    def route(state: MessagesState):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())


async def ask(graph, question: str | None, thread: str = "default", approve=None, resume: bool | None = None) -> dict:
    """Run one turn. `approve(request:dict)->bool` is called if the agent proposes an action.
    Returns {answer, tool_calls, actions}.  With approve=None, pending actions are returned unexecuted
    (the graph stays paused in its checkpoint); call again with question=None, resume=True/False to decide."""
    cfg = {"configurable": {"thread_id": thread}, "callbacks": tracing.callbacks()}
    if resume is not None:
        result = await graph.ainvoke(Command(resume={"approved": resume}), cfg)
    else:
        result = await graph.ainvoke({"messages": [("user", question)]}, cfg)
    actions: list[dict] = []
    while result.get("__interrupt__"):
        req = result["__interrupt__"][0].value
        if approve is None:
            actions.append({**req, "status": "pending"})
            break
        ok = approve(req)
        actions.append({**req, "status": "approved" if ok else "rejected"})
        result = await graph.ainvoke(Command(resume={"approved": ok}), cfg)
    msgs = result["messages"]
    # only THIS turn: the thread keeps earlier turns, and a paused (pending-approval) turn has no text of its own yet, so
    # reading the whole thread showed the previous question's answer and tool calls under the new question
    start = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=0)
    msgs = msgs[start:]
    calls = [{"name": c["name"], "args": c["args"]} for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls]
    final = next((_tool_text(m.content) for m in reversed(msgs) if isinstance(m, AIMessage) and m.content), "")
    return {"answer": final, "tool_calls": calls, "actions": actions}
