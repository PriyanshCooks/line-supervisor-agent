"""MCP tool layer on the shared schema. Expected values are hand-computed from tests/fixtures/bosch_mini (see test_bosch_ingest)."""
import asyncio
import json

import fastmcp
import pytest

from mcp_server import queries as q


def test_station_failure_rate_maths(mini_db):
    r = q.station_failure_rate("L0", "S0")
    assert (r.parts, r.failed, r.failure_rate) == (4, 3, 0.75)          # parts 1,2,3,8 visit S0; 2,3,8 failed
    assert r.line_baseline_rate == 0.5 and r.lift == 1.5                 # 3 of 6 L0 parts failed
    assert q.station_failure_rate("L0", "S1").failure_rate == 0.25
    assert q.station_failure_rate("L0", "S2").failure_rate == 0.4
    assert q.station_failure_rate("L0", "S0", shift="W1").failure_rate == pytest.approx(2 / 3, abs=1e-4)   # parts 1,2,3
    assert r.low_sample is True                                          # < 30 parts


def test_qc_ranking_orders_by_rate_and_respects_min_parts(mini_db):
    rk = q.qc_failure_ranking("L0", min_parts=1)
    assert [s.station for s in rk.stations] == ["S0", "S2", "S1"]
    assert q.qc_failure_ranking("L0", min_parts=5).stations[0].station == "S2"      # only S2 has >= 5 labelled parts


def test_station_metrics(mini_db):
    m = q.station_metrics("S1", "W1", line="L0")
    assert (m.visits, m.complete_cycles, m.avg_dwell) == (2, 2, 4.5)               # dwells 8 and 1
    assert m.time_unit == "time_units"
    s0 = q.station_metrics("S0", "W1", line="L0")
    assert (s0.visits, s0.complete_cycles, s0.avg_dwell) == (3, 3, 3.0)             # dwells 2, 2, 5


def test_queue_throughput_and_ticks_by_hand(mini_db):
    """W1 width = (80-10)/2 = 35. S1 dwells 8 and 1 -> sum 9 -> 9/35 parts; intervals [12,20) and [13,14) overlap -> peak 2; ticks 800,100 -> median 450."""
    m = q.station_metrics("S1", "W1", line="L0")
    assert (m.throughput, m.peak_queue, m.median_dwell_ticks) == (2, 2, 450.0) and m.avg_queue == pytest.approx(9 / 35, abs=1e-4)
    s0 = q.station_metrics("S0", "W1", line="L0")     # dwells 2,2,5 ; intervals [10,12) [11,13) [30,35) -> peak 2 ; median 200 ticks
    assert (s0.peak_queue, s0.median_dwell_ticks) == (2, 200.0) and s0.avg_queue == pytest.approx(9 / 35, abs=1e-4)


def test_window_labels_and_exact_counts_on_outputs(mini_db):
    ls = q.line_summary("W2", "L0")
    d = ls.model_dump()
    assert d["window"] == "W2" and d["window_label"] == "time window 2"
    assert f"= {ls.line_output} units" in d["line_output_text"] and "time window 2 (W2)" in d["line_output_text"]
    assert q.station_metrics("S1", "W1", line="L0").count_text == "2 parts entered S1 on L0 in time window 1 (W1)"
    assert q.find_bottleneck("W1", "L0", min_cycles=2).model_dump()["window_label"] == "time window 1"
    assert q.station_failure_rate("L0", "S0").model_dump()["window"] == "ALL"
    c = q.compare_shifts("W1", "W2", "L0").model_dump()
    assert c["window"] == "W1 vs W2" and "3 units in time window 1 (W1) vs 2 units in time window 2 (W2)" in c["line_output_text"]


def test_line_summary_and_output(mini_db):
    ls = q.line_summary("W1", "L0")
    assert ls.line_output == 3 and [s.station for s in ls.stations] == ["S0", "S1", "S2"]
    assert any("anonymised" in c for c in ls.caveats)


def test_bottleneck_refuses_without_enough_complete_cycles(mini_db):
    b = q.find_bottleneck("W1", "L0")                    # default min_cycles=30, nothing qualifies
    assert b.refused and b.station is None and "cannot be determined" in b.reason


def test_bottleneck_named_when_data_supports_it(mini_db):
    b = q.find_bottleneck("W1", "L0", min_cycles=2)
    assert not b.refused and b.station == "S1" and b.avg_dwell == 4.5            # S0 and S1 tie on avg queue (9/35); tie-break: peak queue (2=2), then mean dwell 4.5 > 3.0
    assert b.avg_queue == pytest.approx(9 / 35, abs=1e-4) and b.peak_queue == 2 and b.throughput == 2 and b.median_dwell_ticks == 450.0
    assert "queue" in b.stat and "NOT the ranking key" in b.stat
    assert q.find_bottleneck("W2", "L0", min_cycles=2).station == "S1"              # dwells 5 and 0 -> 2.5


def test_bottleneck_ignores_stations_whose_cycles_cover_under_half_the_visits(monkeypatch):
    """Found on real Bosch L3 W5: S37 had 199 measurable dwells out of 14,143 visits (mostly last-station units) and topped the ranking."""
    from mcp_server.models import LineSummary, StationMetrics

    def st(name, visits, cycles, mean):
        return StationMetrics(source="bosch", line="L3", station=name, label=name, shift="W5", visits=visits, complete_cycles=cycles,
                              avg_dwell=mean, p90_dwell=mean, qc_labelled=visits, qc_fail_rate=0.0, time_unit="time_units",
                              throughput=visits, avg_queue=visits * mean / 100, peak_queue=1, median_dwell_ticks=mean)
    ls = LineSummary(source="bosch", line="L3", shift="W5", line_output=10, time_unit="time_units",
                     stations=[st("S37", 1000, 199, 3.7), st("S30", 1000, 990, 0.02), st("S31", 1000, 980, 0.01)], caveats=[])
    monkeypatch.setattr(q, "line_summary", lambda *a, **k: ls)
    b = q.find_bottleneck("W5", "L3")
    assert b.station == "S30"


def test_outputs_name_their_statistic(mini_db):
    assert "avg_queue" in q.find_bottleneck("W1", "L0", min_cycles=2).stat
    assert "MEDIAN" in q.station_metrics("S1", "W1", line="L0").stat
    assert "p90" in q.station_metrics("S1", "W1", line="L0").stat
    assert "mean" in q.cycle_time_trend("S0", line="L0").stat
    assert "failure_rate" in q.station_failure_rate("L0", "S0").stat
    assert "MEAN" in q.compare_shifts("W1", "W2", "L0").stat


def test_find_bottleneck_description_matches_the_queue_based_ranking():
    """The tool description is what the model reads; it must not still advertise the old mean-dwell ranking."""
    from mcp_server.server import mcp

    async def desc():
        async with fastmcp.Client(mcp) as c:
            return next(t.description for t in await c.list_tools() if t.name == "find_bottleneck")
    d = asyncio.run(desc()).lower()
    assert "queue" in d and "highest mean dwell" not in d
    assert "throughput" in d and "peak queue" in d and "ticks" in d and "refused=true" in d


def test_action_request_policy_yes_and_no_cases():
    from agent.policy import is_action_request as yes_
    for yes in ("Alert the line lead about station 15 on line 0.", "Alert the line lead about the bottleneck on line 0 in window W5.",
                "Flag station S3 for rebalancing.", "please flag S7", "Can you escalate this?", "Could you flag S2 please?", "Notify the line lead",
                "Find the bottleneck and then alert the line lead", "I want you to flag S2", "What is the bottleneck? Alert the line lead.",
                "Would you alert the line lead?", "Would you please flag S3?"):
        assert yes_(yes), yes
    for no in ("Which station should I rebalance on line 0 in window W5?", "Should I alert the line lead?",
               "What would you recommend for the worst station on line 0 in window W5? Just advise me.",
               "Which stations correlate most with QC failures?",
               # new false-positive classes
               "Should I rebalance S3 and flag it?", "Flag rate at S3?", "Warn me if S2 is slow", "Alert me when the queue grows",
               "Do not alert anyone, just tell me the bottleneck", "don't flag S3", "Is it worth flagging S2?", "What is the flag count at S3",
               "Can I flag S2?"):
        assert not yes_(no), no


def test_extract_action_and_station():
    from agent.policy import extract_action as ex
    assert ex("Alert the line lead about station 15 on line 0.") == ("alert_line_lead", "S15")
    assert ex("Alert the line lead about the bottleneck on line 0 in window W5.") == ("alert_line_lead", None)
    assert ex("Flag station S3 for rebalancing.") == ("flag_for_rebalancing", "S3")
    assert ex("please flag S7") == ("flag_for_rebalancing", "S7")
    assert ex("I want you to flag S2") == ("flag_for_rebalancing", "S2")
    assert ex("Can you escalate S9?") == ("alert_line_lead", "S9")
    assert ex("Find the bottleneck and then notify the lead") == ("alert_line_lead", None)


def test_policy_verdicts_force_and_block():
    from agent import policy as P
    ask, chat = "Alert the line lead about station 15.", "Which station should I rebalance?"
    assert P.should_force_action(ask, already_proposed=False, reply_has_tool_calls=False)          # answered in prose -> force
    assert not P.should_force_action(ask, already_proposed=True, reply_has_tool_calls=False)       # already proposed
    assert not P.should_force_action(ask, already_proposed=False, reply_has_tool_calls=True)       # still gathering data
    assert not P.should_force_action(chat, already_proposed=False, reply_has_tool_calls=False)
    assert P.should_block_action(chat, ["find_bottleneck", "propose_action"])
    assert not P.should_block_action(chat, ["find_bottleneck"])
    assert not P.should_block_action(ask, ["propose_action"])
    assert "action='alert_line_lead'" in P.force_nudge(ask) and "station_id='S15'" in P.force_nudge(ask)
    assert "find_bottleneck" in P.force_nudge("Alert the line lead about the bottleneck.")


def test_compare_shifts(mini_db):
    c = q.compare_shifts("W1", "W2", "L0")
    assert (c.line_output_a, c.line_output_b, c.output_delta_pct) == (3, 2, -33.3)
    assert c.station_dwell_delta_pct["S1"] == pytest.approx(-44.4, abs=0.05)         # 4.5 -> 2.5


def test_part_route(mini_db):
    r = q.part_route("1")
    assert [(s.station, s.enter_ts, s.dwell) for s in r.steps] == [("S0", 10.0, 2.0), ("S1", 12.0, 8.0), ("S2", 20.0, None)]
    assert r.qc_failed == 0
    with pytest.raises(q.QueryError):
        q.part_route("999")
    assert q.part_route("1", source="bosch").steps[0].station == "S0"                # explicit source also fine


def test_part_route_finds_the_unit_in_whichever_source_holds_it(tmp_path, monkeypatch):
    """Regression: with only a non-'bosch' source loaded, part_route must not insist on source='bosch'."""
    import sqlite3

    from ingest import bosch
    from tests.conftest import FIXTURE
    db = tmp_path / "events.db"
    bosch.ingest(FIXTURE / "train_date.csv", FIXTURE / "train_numeric.csv", db, lines=[0, 3], max_parts=None, n_windows=2, source="synthetic")
    monkeypatch.setenv("LINE_DB", str(db))
    assert q.part_route("1").source == "synthetic"


def test_throughput_trend(mini_db):
    t = q.throughput_trend("L0", window=30)
    assert [p.units for p in t.points] == [3, 2, 1]                                   # last-station times 14,20,35 | 55,70 | 80


def test_resolution_errors_are_correctable(mini_db):
    with pytest.raises(q.QueryError) as e:
        q.line_summary("W2")                                                          # W2 exists on both Bosch lines, none given
    assert e.value.field == "line" and set(e.value.valid) == {"L0", "L3"}
    with pytest.raises(q.QueryError) as e:
        q.line_summary("W9", "L0")
    assert e.value.field == "shift" and e.value.valid == ["W1", "W2"]


def test_unknown_shift_error_lists_only_shifts_of_sources_that_have_the_line(mini_db, tmp_path):
    from common import events as E
    con = E.connect()
    E.insert_events(con, [("vision", None, "vision:S2:1", "CAM1", "S2", "leave", 1.0, "s", "CLIP", None)])
    con.commit()
    con.close()
    try:
        with pytest.raises(q.QueryError) as e:
            q.line_summary("W9", "L0")
        assert e.value.valid == ["W1", "W2"]          # not 'CLIP', which belongs to the vision source
    finally:
        con = E.connect()
        E.clear_source(con, "vision")
        con.commit()
        con.close()


def test_vision_events_are_reported_honestly(vision_db):
    ev = q.station_events(source="vision")
    assert (ev.total_events, ev.complete_visits, ev.unmatched_enters, ev.unmatched_leaves) == (2, 0, 1, 1)
    b = q.find_bottleneck("CLIP", "CAM1", "vision", min_cycles=1)
    assert b.refused and b.station is None


def test_mcp_roundtrip_and_validation(mini_db):
    from mcp_server.server import mcp

    async def go():
        async with fastmcp.Client(mcp) as c:
            names = {t.name for t in await c.list_tools()}
            ok = json.loads((await c.call_tool("station_failure_rate", {"line": 0, "station_id": 0})).content[0].text)   # bare numbers ok
            bad_shift = json.loads((await c.call_tool("get_line_summary", {"shift": "W9", "line": "L0"})).content[0].text)
            bad_line = json.loads((await c.call_tool("get_line_summary", {"shift": "W1", "line": "banana"})).content[0].text)
            return names, ok, bad_shift, bad_line
    names, ok, bad_shift, bad_line = asyncio.run(go())
    assert {"list_data_sources", "find_bottleneck", "station_failure_rate", "qc_failure_ranking", "throughput_trend",
            "part_route", "get_station_events", "cycle_time_trend", "compare_shifts", "get_line_summary",
            "get_station_metrics"} <= names
    assert ok["failure_rate"] == 0.75 and ok["source"] == "bosch"
    assert bad_shift["error"] == "invalid_value" and bad_shift["valid_values"] == ["W1", "W2"]
    assert bad_line["error"] == "invalid_arguments" and bad_line["details"][0]["field"] == "line"



def test_named_station_wins_over_the_models_choice():
    import asyncio

    from langchain_core.messages import AIMessage

    from agent import policy as P

    def prop(station, reason="queue is highest"):
        return AIMessage(content="", tool_calls=[{"name": "propose_action", "id": "1", "type": "tool_call",
                                                   "args": {"action": "alert_line_lead", "station_id": station, "reason": reason}}])
    ask = "Alert the line lead about station 15 on line 0."
    assert P.station_to_enforce(ask, "S2") == "S15" and P.station_to_enforce(ask, "15") is None and P.station_to_enforce(ask, "s15") is None
    assert P.station_to_enforce("Alert the line lead about the bottleneck.", "S2") is None          # nothing named
    assert P.station_to_enforce("Which station is S2?", "S9") is None                                # not an action request
    assert P.station_to_enforce("Flag S2 and S3", "S3") is None                                      # any named station is fine

    async def run(resp, retry):
        calls = []

        async def reprompt(nudge):
            calls.append(nudge)
            return retry
        return await P.reconcile_station(ask, resp, reprompt), calls

    # path 1: correct station already -> untouched, no re-prompt
    out, calls = asyncio.run(run(prop("S15"), prop("S99")))
    assert out.tool_calls[0]["args"]["station_id"] == "S15" and calls == []
    # path 2: wrong station, the re-prompt fixes it -> the re-prompted reply is used, asked exactly once, reason untouched
    out, calls = asyncio.run(run(prop("S2"), prop("S15", "named by user")))
    assert out.tool_calls[0]["args"]["station_id"] == "S15" and len(calls) == 1 and "S15" in calls[0]
    assert P.STATION_NOTE not in out.tool_calls[0]["args"]["reason"]
    # path 3: still wrong after the re-prompt -> overwritten, and the approval reason says so
    out, calls = asyncio.run(run(prop("S2"), prop("S7")))
    a = out.tool_calls[0]["args"]
    assert a["station_id"] == "S15" and len(calls) == 1 and a["reason"].endswith("(station set from your request)") and a["action"] == "alert_line_lead"
    # path 4: the re-prompt dropped the proposal -> keep the original call, with the station overwritten
    out, calls = asyncio.run(run(prop("S2"), AIMessage(content="sorry")))
    assert out.tool_calls[0]["args"]["station_id"] == "S15" and "station set from your request" in out.tool_calls[0]["args"]["reason"]
    # non-propose tool calls and replies without any call pass through
    plain = AIMessage(content="text")
    assert asyncio.run(run(plain, plain))[0] is plain
