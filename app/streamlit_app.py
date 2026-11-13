"""Demo UI. Tab 1: real line data (Bosch) + the supervisor agent. Tab 2: the camera -> events adapter."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402  (ships with streamlit)
import streamlit as st  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.session_runtime import SessionRuntime  # noqa: E402
from mcp_server import queries  # noqa: E402

st.set_page_config(page_title="Line Supervisor Agent", layout="wide")


def runtime() -> SessionRuntime:
    """One persistent agent runtime per browser session: a single MCP subprocess, built once, reused by every chat turn and the report."""
    rt = st.session_state.get("rt")
    if rt is None or not rt.alive:                      # first use, or the subprocess died: (re)start
        if rt is not None:
            rt.close()
        with st.spinner("Starting the agent (MCP server + data catalog)..."):
            rt = st.session_state.rt = SessionRuntime().start()
    return rt


def _num(s: str) -> int:
    return int("".join(ch for ch in s if ch.isdigit()) or 0)


with st.sidebar:
    st.header("Line Supervisor Agent")
    st.caption(f"LLM: {os.environ.get('LLM_PROVIDER', 'ollama')} / {os.environ.get('LLM_MODEL', 'qwen3.5:4b')}")
    st.caption("Shared event schema -> MCP tools -> agent. Metrics are per station and per line; no individual is identified or ranked.")

cat = queries.catalog().sources
tab_bosch, tab_vision = st.tabs(["Real line data (Bosch)", "Vision adapter (camera → events)"])

# ============================================================================== TAB 1
with tab_bosch:
    srcs = [s for s in ("bosch", "synthetic") if any(c.source == s for c in cat)]
    if not srcs:
        st.error("No line data loaded. Download the Bosch files (see docs/DATA.md) and run `python tasks.py ingest`, "
                 "or generate the Bosch-format dev sample with `python tasks.py sample`.")
        st.stop()
    src = srcs[0]
    if src == "synthetic":
        st.warning("**No real Bosch data is loaded.** Showing the Bosch-FORMAT development sample (source = synthetic). "
                   "Numbers below are not Bosch results. See docs/DATA.md.")
    infos = [c for c in cat if c.source == src]
    c1, c2, c3 = st.columns([1, 1, 3])
    line = c1.selectbox("Line", [i.line for i in infos])
    info = next(i for i in infos if i.line == line)
    shift = c2.selectbox("Window (shift)", info.shifts, index=min(4, len(info.shifts) - 1))
    c3.caption(f"Timestamps are anonymised **{info.time_unit.replace('_', ' ')}**, not clock time. {(info.notes or [''])[0]}. "
               "'Dwell' = time until the unit reaches its next recorded station (includes waiting).")

    if st.session_state.get("ctx") != (src, line, shift):            # new context -> fresh conversation, no stale answers
        st.session_state.update(ctx=(src, line, shift), saver=MemorySaver(), chat=[], thread=str(uuid.uuid4()), pending=None, report=None)

    summ = queries.line_summary(shift, line, src)
    bn = queries.find_bottleneck(shift, line, src)
    rk = queries.qc_failure_ranking(line, None, src, top_n=8, min_parts=50)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Line output (units)", f"{summ.line_output:,}")
    m2.metric("Bottleneck", "none (refused)" if bn.refused else bn.station)
    m3.metric("Avg parts in queue at bottleneck", "-" if bn.refused else f"{bn.avg_queue:.3f}")
    m4.metric("Line QC failure baseline", "-" if rk.line_baseline_rate is None else f"{100 * rk.line_baseline_rate:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Station metrics")
        df = pd.DataFrame([s.model_dump() for s in summ.stations]).set_index("station")
        show = df[["throughput", "avg_queue", "peak_queue", "median_dwell_ticks", "complete_cycles", "qc_fail_rate"]]
        st.dataframe(show.style.format({"avg_queue": "{:.3f}", "peak_queue": "{:.0f}", "median_dwell_ticks": "{:g}", "qc_fail_rate": "{:.1%}"}, na_rep="-").apply(
            lambda r: ["background-color:#ffd6d6;color:#000"] * len(r) if (not bn.refused and r.name == bn.station) else [""] * len(r), axis=1),
            width="stretch", height=300)
        if bn.refused:
            st.info(bn.reason)
        else:
            st.error(f"Bottleneck: {bn.station} ({bn.avg_queue} parts in queue on average, peak {bn.peak_queue}; throughput {bn.throughput} parts in {bn.window_label})")
            for e in bn.evidence:
                st.caption("• " + e)
        st.caption("Average parts in queue by station (ranking key). Throughput, peak queue and median dwell in ticks of 0.01 are in the table; "
                   "per-station dwell is only 0-3 ticks, so it is not used for ranking.")
        st.bar_chart(df["avg_queue"].fillna(0).rename("avg parts in queue").reset_index(), x="station", y="avg parts in queue", sort=False, height=200,
                     x_label="station (route order)", y_label="avg parts in queue")
    with right:
        st.subheader("QC failure by station")
        if rk.stations:
            q = pd.DataFrame([{"station": r.station, "failure rate": r.failure_rate, "units": r.parts, "lift vs line": r.lift} for r in rk.stations]).set_index("station")
            st.bar_chart(q["failure rate"].reset_index(), x="station", y="failure rate", sort="-failure rate", height=240,
                         x_label="station (highest failure rate first)", y_label="failure rate")
            st.dataframe(q.style.format({"failure rate": "{:.1%}", "lift vs line": "{:.2f}x"}), width="stretch")
            st.caption("Failure = the whole part's Response=1, so this shows association with passing a station, not proof of cause. "
                       f"Line baseline {100 * (rk.line_baseline_rate or 0):.1f}%.")
        else:
            st.info("No QC labels available for this line.")
        if not bn.refused:
            tp = queries.throughput_trend(line, None, src)
            if tp.points:
                st.caption("Line output over the whole timeline (units entering the last station per bucket)")
                st.line_chart(pd.DataFrame({"time (units)": [round(p.window_start) for p in tp.points], "units": [p.units for p in tp.points]}),
                              x="time (units)", y="units", height=180)

    st.divider()
    if st.button("Generate shift report", type="primary"):
        with st.spinner("Generating report (LLM writes the prose; numbers come from the tools)..."):
            st.session_state.report = runtime().report(shift, line, src)
    if st.session_state.get("report"):
        r1, r2 = st.tabs(["WhatsApp", "Markdown"])
        r1.code(st.session_state.report["whatsapp"], language=None)
        r2.markdown(st.session_state.report["markdown"])

    st.divider()
    st.subheader("Ask the supervisor agent")
    st.caption("Try: *Which station is the bottleneck?*  ·  *Which stations correlate with QC failures?*  ·  *Compare throughput between windows W2 and W6*  ·  *Alert the line lead about the bottleneck* (needs your approval)")

    def _ask(q=None, resume=None):
        return runtime().ask(q, st.session_state.saver, st.session_state.thread, resume=resume)

    def _handle(out: dict) -> None:
        st.session_state.pending = next((a for a in out["actions"] if a["status"] == "pending"), None)
        if out["answer"] or not st.session_state.pending:
            st.session_state.chat.append(("assistant", out["answer"], [c["name"] for c in out["tool_calls"]]))

    for role, text, *tools in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)
            if tools and tools[0]:
                st.caption("tools called: " + ", ".join(tools[0]))

    p = st.session_state.pending
    if p:
        st.warning(f"**Human approval required:** {p['action']} -> {p['station_id']}\n\n{p['reason']}")
        a1, a2, _ = st.columns([1, 1, 6])
        for col, label, val in ((a1, "Approve", True), (a2, "Reject", False)):
            if col.button(label, key=f"{label}-{len(st.session_state.chat)}"):
                with st.spinner("Resuming..."):
                    _handle(_ask(resume=val))
                st.rerun()

    if qtext := st.chat_input("Ask about this line, e.g. Which station is the bottleneck?", key="chat_bosch"):
        st.session_state.chat.append(("user", qtext, []))
        with st.spinner("Thinking..."):
            _handle(_ask(q=f"[Context: source {src}, line {line}, window {shift}] {qtext}"))
        st.rerun()

# ============================================================================== TAB 2
with tab_vision:
    vcat = [c for c in cat if c.source == "vision"]
    st.subheader("Camera footage → the same event schema")
    st.markdown(
        "A working CV pipeline (YOLO person detection used **only to blur people**, plus per-zone presence and motion) turns "
        "footage into `enter` / `leave` events written to the **same table** the Bosch data uses (`source = 'vision'`). "
        "Public fixed-camera factory footage is scarce, so this runs on a **17 s stable window** of a stock clip. "
        "It detects a station hand-off, and it **refuses to report cycle times** from incomplete data.")
    vid = ROOT / "data" / "out" / "annotated.mp4"
    cl, cr = st.columns([3, 2])
    with cl:
        if vid.exists():
            st.video(str(vid), loop=True, autoplay=True, muted=True)
            st.caption("People are blurred in all outputs. Zones S1-S3 are the three nearest conveyor fixtures; the legend shows live enter/leave counts.")
        else:
            st.info("No annotated clip yet. Run `python tasks.py vision` (needs data/input_stable.mp4).")
    with cr:
        if vcat:
            ev = queries.station_events(source="vision", limit=100)
            st.metric("Events written", ev.total_events)
            st.metric("Complete visits (enter + leave)", ev.complete_visits)
            c_a, c_b = st.columns(2)
            c_a.metric("Unit already inside at start", ev.unmatched_leaves)
            c_b.metric("Unit still inside at end", ev.unmatched_enters)
            tbl = pd.DataFrame([e.model_dump() for e in ev.events])[["event_type", "station", "ts", "visit_id", "shift"]]
            tbl = tbl.rename(columns={"ts": "t (s)"})
            st.dataframe(tbl, width="stretch", hide_index=True)
            b = queries.find_bottleneck("CLIP", "CAM1", "vision")
            st.info("find_bottleneck on this clip -> **refused**: " + (b.reason or "") if b.refused else f"bottleneck {b.station}")
        else:
            st.info("No vision events in the shared table yet. Run `python tasks.py vision`.")
    st.markdown("**What this proves:** the adapter emits the same schema as the production data, so the same MCP tools and agent work on camera "
                "events. **What it does not prove:** throughput or cycle times: one hand-off in 17 s gives no complete cycle. On a fixed CCTV "
                "feed the same code would emit continuous events. Ask the agent from tab 1: *What happened at the stations in the camera clip?*")
