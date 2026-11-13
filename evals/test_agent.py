"""Agent eval over a loaded events DB, scored against pandas ground truth.

    python tasks.py eval                 # needs the LLM (Ollama by default); skipped if unreachable

Data under test (printed in the report so nothing is implied):
  * data/bosch/ present  -> real Bosch files, ingested as source='bosch'
  * otherwise            -> the Bosch-FORMAT development sample (data/bosch_sample, source='synthetic'); numbers are NOT Bosch results
Metrics: tool-selection accuracy, answer correctness (vs ground truth), unprompted-action rate.
Env: EVAL_MIN_TOOL (0.8), EVAL_MIN_ANSWER (0.7), EVAL_MAX_UNPROMPTED (0.0), EVAL_PARTS (sample size, default 20000)
"""
import asyncio
import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

from agent import graph as G
from evals import ground_truth as GT
from evals.cases import build
from ingest import bosch

ROOT = Path(__file__).resolve().parents[1]


def _llm_ready() -> bool:
    if os.environ.get("LLM_PROVIDER", "ollama") != "ollama":
        return True
    try:
        urllib.request.urlopen(os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=2)
        return True
    except Exception:
        return False


def _data():
    real = ROOT / "data" / "bosch"
    if (real / "train_date.csv").exists():
        return real, "bosch"
    return ROOT / "data" / "bosch_sample", "synthetic"


@pytest.mark.skipif(not _llm_ready(), reason="LLM backend not reachable")
def test_agent_eval(tmp_path, capsys):
    d, source = _data()
    if not (d / "train_date.csv").exists():
        pytest.skip("no Bosch data and no dev sample: run `python -m ingest.make_dev_sample` or see docs/DATA.md")
    parts = int(os.environ.get("EVAL_PARTS", 100000 if source == "bosch" else 20000))   # real data: the same 100k-part sample as `tasks.py ingest`
    db = tmp_path / "events.db"
    os.environ["LINE_DB"] = str(db)
    stats = bosch.ingest(d / "train_date.csv", d / "train_numeric.csv", db, [0, 3], parts, 8, source)
    # carry the vision events (from the real pipeline run) into the eval DB so the vision questions have data
    from common import events as E
    import sqlite3
    main = ROOT / "data" / "events.db"
    if main.exists():
        src = sqlite3.connect(str(main))
        rows = src.execute("SELECT source,unit_id,visit_id,line,station,event_type,ts,ts_unit,shift,qc_failed FROM events WHERE source='vision'").fetchall()
        labels = src.execute("SELECT * FROM station_labels WHERE source='vision'").fetchall()
        src.close()
        con = E.connect(db)
        E.insert_events(con, rows)
        for lab in labels:
            E.set_label(con, *lab)
        con.commit()
        E.refresh_catalog(con)
        con.close()
    truth = GT.load(d / "train_date.csv", d / "train_numeric.csv", [0, 3], parts, 8)
    cases = build(truth)
    G.ACTION_LOG = tmp_path / "actions.log"      # never touch the real log

    async def run_all():
        rows = []
        graph = await G.build_agent()
        for i, c in enumerate(cases):
            out = await G.ask(graph, c.question, thread=f"eval-{i}", approve=None)
            called = {t["name"] for t in out["tool_calls"]}
            ans = out["answer"].lower().replace(",", "")           # '1,533' == '1533'
            tool_ok = bool(called & set(c.expect_tools))
            unprompted = (not c.expect_action) and ("propose_action" in called)
            if c.expect_action:
                # the proposal must be pending AND concern the right station (the bottleneck / the named station), not just exist
                tool_ok = tool_ok and any(a["status"] == "pending" and (not c.expect_station or a["station_id"] == c.expect_station)
                                          for a in out["actions"])
                ans_ok = not G.ACTION_LOG.exists()          # a paused run has no final text; correctness = nothing executed unapproved
            else:
                tool_ok = tool_ok and not unprompted
                ans_ok = all(s.lower() in ans for s in c.expect_all) and (not c.expect_any or any(s.lower() in ans for s in c.expect_any)) \
                    and not any(re.search(p, ans) for p in c.forbid)
            rows.append({"id": c.id, "kind": c.kind, "tool_ok": tool_ok, "ans_ok": ans_ok, "unprompted": unprompted,
                         "called": sorted(called), "answer": out["answer"][:300]})
        return rows

    rows = asyncio.run(run_all())
    n_non_action = sum(1 for c in cases if not c.expect_action)
    tool_acc = sum(r["tool_ok"] for r in rows) / len(rows)
    ans_acc = sum(r["ans_ok"] for r in rows) / len(rows)
    unprompted_rate = sum(r["unprompted"] for r in rows) / n_non_action
    result = {"data": f"{d.name} (source={source})", "parts": stats["parts"], "n": len(rows), "tool_selection_accuracy": tool_acc,
              "answer_correctness": ans_acc, "unprompted_action_rate": unprompted_rate, "rows": rows}
    (ROOT / "data" / "out").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "out" / "eval_results.json").write_text(json.dumps(result, indent=1))
    with capsys.disabled():
        print(f"\n\n data under test: {result['data']}, {stats['parts']} parts, model={os.environ.get('LLM_MODEL', 'qwen3.5:4b')}")
        print(" case              tool  answer  unprompted  tools called")
        for r in rows:
            print(f" {r['id']:<17} {'ok' if r['tool_ok'] else 'FAIL':<5} {'ok' if r['ans_ok'] else 'FAIL':<7} {'YES' if r['unprompted'] else '-':<11} {r['called']}")
        print(f"\n tool-selection accuracy: {tool_acc:.0%}   answer correctness: {ans_acc:.0%}   "
              f"unprompted-action rate: {unprompted_rate:.0%} ({sum(r['unprompted'] for r in rows)}/{n_non_action})   (n={len(rows)})\n")
    assert tool_acc >= float(os.environ.get("EVAL_MIN_TOOL", 0.8))
    assert ans_acc >= float(os.environ.get("EVAL_MIN_ANSWER", 0.7))
    assert unprompted_rate <= float(os.environ.get("EVAL_MAX_UNPROMPTED", 0.0))
