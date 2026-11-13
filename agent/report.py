"""End-of-shift report (WhatsApp-style short text + markdown).

Every number, the table and the structure are built deterministically from MCP tool results. The LLM writes only two
short prose fields (likely cause, suggested action); any number in them that is not already in the facts causes the
whole prose to be rejected and the rule-based wording to be used instead. (An earlier version let the LLM write the
entire report and it misstated the last station and invented a reason to exclude S9.) If the LLM is unavailable or returns
bad JSON, the rule-based template is used, so the button always works. The report never names a bottleneck when
find_bottleneck refuses, and never converts time units.
"""
from __future__ import annotations

import json
import re

from agent.graph import _tool_text, mcp_client
from agent.llm import get_llm


async def _facts(shift: str, line: str | None, source: str | None) -> dict:
    tools = {t.name: t for t in await mcp_client().get_tools()}

    async def call(name, **kw):
        return json.loads(_tool_text(await tools[name].ainvoke({k: v for k, v in kw.items() if v is not None})))

    cat = await call("list_data_sources")
    summary = await call("get_line_summary", shift=shift, line=line, source=source)
    if "error" in summary:
        return {"error": summary}
    bn = await call("find_bottleneck", shift=shift, line=summary["line"], source=summary["source"])
    facts = {"summary": summary, "bottleneck": bn, "catalog": cat}
    if not bn["refused"]:
        facts["trend"] = await call("cycle_time_trend", station_id=bn["station"], shift=shift, line=summary["line"], source=summary["source"])
    if summary["source"] != "vision":
        facts["qc"] = await call("qc_failure_ranking", line=summary["line"], shift=shift, source=summary["source"], top_n=3)
    return facts


def _template(f: dict, cause_override: str | None = None, action_override: str | None = None) -> tuple[str, str]:
    s, b = f["summary"], f["bottleneck"]
    unit = s["time_unit"].replace("_", " ")
    t, qc = f.get("trend"), f.get("qc")
    head = f"Line {s['line']} | {s['window_label']} ({s['shift']}) | source {s['source']}" + (" (development sample, not real Bosch data)" if s["source"] == "synthetic" else "")
    if b["refused"]:
        bn_line = "Bottleneck: cannot be determined; no station has enough complete enter-to-leave cycles in this data."
        cause = "No cause can be stated from this data."
        action = "Collect more complete station cycles before rebalancing anything."
    else:
        bn_line = (f"Bottleneck: {b['station']} ({b['avg_queue']} parts in queue on average, peak {b['peak_queue']}; throughput {b['throughput']} parts; "
                   f"median dwell {b['median_dwell_ticks']} ticks of 0.01 {unit}).")
        cause = (f"{b['station']} held the longest average queue on the line in this window; at 0.01 timestamp resolution per-station dwell is "
                 "only a few ticks, so the data cannot say why.")
        action = f"Inspect {b['station']} (fixtures, staffing, upstream batching) and consider moving work to a faster station."
    cause, action = cause_override or cause, action_override or action      # LLM prose (already number-checked) replaces the defaults
    qc_line = ""
    if qc and qc["stations"]:
        top = qc["stations"][0]
        qc_line = (f"QC: highest failure rate at {top['station']} {100 * top['failure_rate']:.1f}% "
                   f"(line baseline {100 * qc['line_baseline_rate']:.1f}%, association only).")
    out_line = "" if s["source"] == "vision" else f"Output: {s['line_output']} units reached the last station in {s['window_label']} ({s['shift']})."
    wa = "\n".join(x for x in [f"*{head}*", out_line, bn_line, qc_line,
                               f"Likely cause: {cause}", f"Suggested: {action} (needs line-lead confirmation)"] if x)
    def pct(v):
        return "-" if v is None else f"{100 * v:.1f}%"

    def num(v):
        return "-" if v is None else v

    rows = "\n".join(f"| {x['station']} | {x['throughput']} | {num(x['avg_queue'])} | {num(x['peak_queue'])} | {num(x['median_dwell_ticks'])} | {pct(x['qc_fail_rate'])} |"
                     for x in s["stations"][:30])
    md = (f"# {head}\n\n" + (f"**Line output:** {s['line_output']} units reached the last station in {s['window_label']} ({s['shift']}).\n\n" if out_line else "") +
          f"| Station | Throughput (parts) | Avg queue (parts) | Peak queue | Median dwell (ticks) | QC fail rate |\n|---|---|---|---|---|---|\n{rows}\n\n"
          f"## Bottleneck\n{bn_line}\n" + ("\n".join(f"- {e}" for e in b["evidence"]) if not b["refused"] else f"- {b['reason']}") +
          (f"\n\n## QC\n{qc_line}\n" if qc_line else "\n") + f"\n## Likely cause\n{cause}\n\n## Suggested action\n{action}\n\n"
          f"## Notes\n" + "\n".join(f"- {c}" for c in s["caveats"]) + "\n\n*Any action requires confirmation by the line lead.*\n")
    return wa, md


async def generate_report(shift: str, line: str | None = None, source: str | None = None, use_llm: bool = True) -> dict:
    f = await _facts(shift, line, source)
    if "error" in f:
        return {"whatsapp": f"Cannot build report: {f['error']}", "markdown": json.dumps(f["error"], indent=1), "facts": f}
    wa, md = _template(f)
    llm_used = False
    if use_llm:
        try:
            facts_json = json.dumps({k: v for k, v in f.items() if k != "catalog"})
            prompt = ("You are writing two short sentences for an end-of-shift report, from these FACTS ONLY. Return JSON with keys "
                      "likely_cause (max 35 words: the most likely reason for the bottleneck, as an association not a proof) and "
                      "suggested_action (max 30 words, a recommendation a line lead must confirm). Rules: use ONLY numbers that appear "
                      "in the facts; do not convert time units; if bottleneck.refused is true say no cause can be stated and do not "
                      "name a station; never mention individual workers.\nFACTS: " + facts_json)
            resp = await get_llm().ainvoke(prompt)
            text = _tool_text(resp.content)
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            cause, action = str(data["likely_cause"]).strip(), str(data["suggested_action"]).strip()
            # guard: every number in the prose must already be in the facts (substring check on the rounded forms)
            nums = re.findall(r"\d+(?:\.\d+)?", cause + " " + action)
            if all(n in facts_json for n in nums) and 0 < len(cause) < 400 and 0 < len(action) < 400:
                wa, md = _template(f, cause, action)
                llm_used = True
        except Exception:  # LLM down / bad JSON / unsupported numbers -> deterministic template
            pass
    return {"whatsapp": wa, "markdown": md, "facts": f, "llm_prose_used": llm_used}

