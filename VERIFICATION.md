# VERIFICATION

Everything below was actually run on this machine (Windows 11, Python 3.11.0, Ollama 0.40.16, model `qwen3.5:4b`
on the integrated AMD GPU via Vulkan, ~19 tokens/s). Output is pasted from the real runs; long blocks are trimmed where marked.

## 0. The most important caveat

**This file has two kinds of evidence; each section says which.** The first version of this repo was verified on a Bosch-**format**
development sample (`source: synthetic`, 20,000 planted-structure parts from `ingest/make_dev_sample.py`) because the Kaggle files were
not available. The real files (`train_date.csv`, `train_numeric.csv`, `source: bosch`) have since been ingested and re-verified:
sections **2b, 4b, 6b** are real-data runs. Sections marked "dev sample" are the earlier synthetic runs, kept for the record; they say nothing about Bosch.
**Re-verified on real data in the latest round (0b):** the Streamlit chat/report/approve/reject buttons and the LLM shift report (one real run each, in the browser). **Still only on the dev sample:** the hand-run agent Q&As in section 5.

## 0b. LATEST ROUND (read this first): what changed after the first real-data run, and the new evidence

Everything in this section was run on the real Bosch sample (`source=bosch`, 100,000 parts, lines 0 and 3). Where an older section below contradicts this one, this one wins.

**1. Bottleneck now ranks by queue and throughput, not mean dwell.** Bosch timestamps have 0.01 resolution (1 tick), so per-station dwells are 0-3 ticks and a mean is mostly rounding noise (earlier finding: L0 W5 S15 had mean 0.031 but p90 0.03). `find_bottleneck` ranks eligible stations (>= 30 complete cycles covering >= 50% of visits) by **average parts in queue** = sum of dwell over complete visits / window width; each station also reports `throughput` (parts entering per window), `peak_queue` (max simultaneous parts between entering a station and its next one) and `median_dwell_ticks`. Mean/p90 dwell are still returned, named, and never used for ranking. Ties: higher peak queue, then mean dwell, then lower station number. Real results:

```
line window bottleneck avg_queue peak throughput median_dwell_ticks
L0   W2     S2         0.5591    14   4018       2.0
L0   W5     S2         0.4725    17   3544       2.0     (was S15 under mean dwell)
L0   W8     S3         0.6976    17   3606       2.0
L3   W2     S30        1.0528    25   11956      1.0
L3   W5     S30        1.2559    26   14134      1.0
L3   W8     S30        0.7498    24   8837       1.0
```

I interpreted "rank by throughput and parts in queue" as: rank by queue, report throughput beside it (not a second sort key). The winner's median dwell is only 1-2 ticks, so the tool appends a warning that the ranking is weak evidence at this resolution. Ground truth (`evals/ground_truth.py`) re-implements queue, peak queue and median ticks independently in pandas; unit tests have hand-computed values (fixture: S1 W1 dwells 8 and 1, window width 35 -> avg queue 9/35, peak 2, median 450 ticks). **Real-data cross-check: 1,296 comparisons (throughput, avg queue, peak queue, median ticks per station, bottleneck, line output; 2 lines x 8 windows) - 0 mismatches.**

**2. Window labels and exact counts on every tool output.** Each window-scoped output now has `"window": "W5"` and `"window_label": "time window 5"` (`W5` is never "5 weeks"), plus ready-to-quote sentences: `line_output_text` ("Line output for L0 in time window 5 (W5) = 465 units ..."), `count_text` per station. `bucket_width` replaces the old numeric `window` field in trend outputs (name clash). Prompt rule: quote the exact count in the first sentence. Real chat answer in the browser: "Line output for L0 in window W5 was 465 units. The bottleneck is S2, with an average queue of 0.4725 parts (peak 17), throughput of 3,544 parts per window, and median dwell of 2 ticks".

**3. Streamlit exercised in a real browser on Bosch data** (built-in browser pane, qwen3.5:4b, Ollama on the GPU):
- "Generate shift report": 43 s and 52 s (two runs); numbers match the tools (465, S2, 0.4725, peak 17, 3544, 2.0 ticks).
- Chat: "What was the line output ... and which station is the bottleneck?" -> correct, named statistics, 99 s.
- Approve: "Alert the line lead about the bottleneck." -> approval card `alert_line_lead -> S2` after 32 s; `data/actions.log` did not exist before the click and held exactly one line after it.
- Reject: "Flag station S3 for rebalancing." -> card `flag_for_rebalancing -> S3`; after Reject no log was written and the agent acknowledged and offered alternatives.
- Screenshots: `docs/img/tab1_bosch.jpg`, `report_and_chat.jpg`, `chat_approval_pending.jpg`, `chat_rejected.jpg`.
- **Bugs this found (none were caught by unit tests, which do not import the app):** (a) my own edit left an `IndentationError` in `app/streamlit_app.py` and `agent/report.py` (the app showed a script error until fixed); (b) the approval card showed the *previous* turn's answer and tool calls, because `ask()` read the whole thread - fixed by slicing to the current turn; (c) `peak_queue` rendered as `6.000000` in the table; (d) each tool call started a fresh MCP server process that rebuilt the 600k-row frame (~14 s), so a report took minutes - added an on-disk pickle cache keyed to the DB mtime (cold query 10 s -> 0.2 s, report 43 s). The user also clicked around in the pane during the run (switched tabs), which I handled; this was not a clean hands-off run.

**4. People blur confirmed (measured, not just eyeballed).** YOLOv8n at the pipeline's own settings (conf 0.12, imgsz 1280) finds 8 / 4 / 3 people in the *original* clip at 0.5 s / 8.5 s / 16.8 s. In `data/out/annotated.mp4` at those times, sharpness (Laplacian variance) inside those boxes is at most 0.155x the original (median 0.018-0.034x), and the detector finds **0** people in the output at conf 0.12 and 0.25. Same for `data/out/frames/{start,middle,end}.jpg` (max 0.154x) and for the three panels of `docs/img/vision_start_middle_end.jpg` (max 0.135x / 0.056x / 0.044x). Across the whole video (425 frames, every 10th checked = 43 frames) 0 frames had a residual detection. Limits: not every frame was checked; a person the detector misses in the original would also be missed here; the silhouette is blurred, not removed. I viewed the composite image too. No regeneration was needed.

**5. Pre-push check (nothing committed).** `.gitignore` now also excludes every `*.mp4/*.avi/*.mov`, `*.db/*.sqlite`, `yolov8n.pt`, `*.pkl` caches, `*.csv.zip`, `data/events.db`, `data/bosch/`, `data/bosch_sample/`, `.venv/`, `.env`; `git check-ignore` confirmed each of 9 sample paths is ignored. Final `git add -A` (staged, **not committed**): 46 files, 905,526 bytes (884.3 KB); largest is `docs/img/vision_start_middle_end.jpg` (537 KB); the only CSVs tracked are the two hand-written fixture files; no `.mp4/.db/.pt/.pkl/.zip` tracked. A regex scan of tracked text files for API-key/password patterns found nothing. Not verified: git history (there is none yet); the Kaggle zips remain in `~/Downloads`, outside the repo.

**6. Final eval (real data, n=18):** tool selection 100%, answer correctness 94% (17/18), unprompted actions 0/16, 12 min 11 s. Only failure: `fail_station` (agent added `shift='W1'` unprompted, reported 0.78% instead of the overall rate). Honest history of this eval on the same data: run 1 94%/94%/0% (`hitl` no proposal; `line_output` rambled); run 2 100%/94%/0% (`line_output` still failing); run 3 94%/89%/0% (`median_ticks` wrong tool; `units` **converted time units to hours**); run 4 94%/94%/**6% (1/16 unprompted `propose_action`)** and the eval test FAILED its own gate; run 5 (final) 100%/94%/0%. Fixes made between runs (all designed after seeing the failure on these same questions, so this is a regression guard, not out-of-sample): window labels + exact-count sentences; tool docstrings pointing single-station numbers/medians to `get_station_metrics`; a stricter "never convert time units, never write a number followed by hours" rule (spot-checked: the "in hours" question is now refused and answered in time units/ticks); a code guard that re-prompts once when an imperative alert/flag request gets no `propose_action`, with the action and station taken from the request (a first version proposed the wrong action/station - the eval now also requires the pending proposal to name the bottleneck station); and a policy that blocks `propose_action` unless the current user message is an imperative alert/flag request. Unit tests: 31 pass. Raw results of earlier runs are kept in `data/out/eval_results_real_run*.json`.

**7. LLM on the RTX 3050 (4 GB).** Ollama's automatic split put only ~50% of qwen3.5:4b on the GPU (`ollama ps`: 48%/52% CPU/GPU, ~10 tokens/s); CPU only: 6.4 tokens/s; forcing all layers (`num_gpu=99`, now the default in `agent/llm.py`, env `OLLAMA_NUM_GPU`) with `num_ctx=8192` (default 4096 can truncate tool output): `ollama ps` 100% GPU, **~19 tokens/s, i.e. the same as the AMD iGPU earlier**, so the 3050 gave ~2x over Ollama's default but no gain over the iGPU because its 4 GB barely holds the model plus vision encoder. Caveat: `nvidia-smi` showed only ~870 MiB used and low utilisation during these runs while Ollama reported 100% GPU; I could not reconcile that (Windows hybrid-GPU reporting?), so GPU placement rests on Ollama's own report and the speed change. Chat answers still take 30-100 s because each question is 2-4 LLM calls over ~1-3k-token prompts.

**Still not verified / not done:** the demo video and GIF (placeholders in the README); Docker Compose; Langfuse; Anthropic/OpenAI providers; the dev-sample sections below were not regenerated and use the old mean-dwell definition.

## 1. Unit tests

```
$ python tasks.py test        (final run, after all changes)
...
======================== 25 passed, 1 warning in 4.64s ========================
```

Per-test list from the earlier full verbose run (24 tests; the 25th, `test_unknown_shift_error_lists_only_shifts_of_sources_that_have_the_line`, was added afterwards for a bug found below):

```
tests/test_bosch_ingest.py::test_counts PASSED
tests/test_bosch_ingest.py::test_route_ordering_and_station_entry_is_min_of_date_columns PASSED
tests/test_bosch_ingest.py::test_tie_is_broken_by_station_number PASSED
tests/test_bosch_ingest.py::test_leave_is_next_station_entry_and_last_station_has_no_leave PASSED
tests/test_bosch_ingest.py::test_lines_are_independent_routes PASSED
tests/test_bosch_ingest.py::test_windows PASSED
tests/test_bosch_ingest.py::test_qc_label_attached_to_every_event_of_the_part PASSED
tests/test_bosch_ingest.py::test_time_units_are_never_relabelled PASSED
tests/test_bosch_ingest.py::test_schema_rejects_bad_values PASSED
tests/test_crosscheck.py::test_pandas_ground_truth_matches_hand_values PASSED
tests/test_crosscheck.py::test_tools_agree_with_ground_truth PASSED
tests/test_tools.py::test_station_failure_rate_maths PASSED
tests/test_tools.py::test_qc_ranking_orders_by_rate_and_respects_min_parts PASSED
tests/test_tools.py::test_station_metrics PASSED
tests/test_tools.py::test_line_summary_and_output PASSED
tests/test_tools.py::test_bottleneck_refuses_without_enough_complete_cycles PASSED
tests/test_tools.py::test_bottleneck_named_when_data_supports_it PASSED
tests/test_tools.py::test_compare_shifts PASSED
tests/test_tools.py::test_part_route PASSED
tests/test_tools.py::test_part_route_finds_the_unit_in_whichever_source_holds_it PASSED
tests/test_tools.py::test_throughput_trend PASSED
tests/test_tools.py::test_resolution_errors_are_correctable PASSED
tests/test_tools.py::test_vision_events_are_reported_honestly PASSED
tests/test_tools.py::test_mcp_roundtrip_and_validation PASSED
======================== 24 passed, 1 warning in 4.88s ========================
```

The fixture (`tests/fixtures/bosch_mini`) has 8 hand-written parts; the expected values (failure rates 0.75 / 0.25 / 0.4, mean dwells 3.0 / 3.5, baseline 0.5, windows, tie-break) were computed by hand before running. `test_tools_agree_with_ground_truth` also checks every station/shift against an independent pandas computation (`evals/ground_truth.py`).

## 2b. Ingest: REAL Bosch data

Files unzipped from Kaggle's `train_date.csv.zip` (2.89 GB) and `train_numeric.csv.zip` (2.14 GB) into `data/bosch/` (zip entry names checked first).

```
$ python tasks.py ingest        (92 s)
{'parts': 100000, 'visits': 1168003, 'enter': 1168003, 'leave': 989868, 'tmin': 0.01, 'tmax': 1718.48, 'window_width': 214.8088, 'n_windows': 8, 'fail_rate_overall': 0.0057, 'events': 2157871}
```

Per line: L0 1,016,491 events / 24 stations; L3 1,141,380 events / 22 stations. The dev-sample rows were then removed from the live DB so the catalog contains only `bosch` (+ the `vision` clip events). Tool results vs an independent pandas computation straight from the CSVs: **507 comparisons (output, bottleneck, visits, cycles, mean dwell per station for L0/L3 x W2/W5/W8; failure counts/rates for every station; top-ranked failure station) and 6 part routes: 0 mismatches** (one-off script; the same check on the small fixture is `tests/test_crosscheck.py`).

## 2. Ingest (Bosch-format dev sample, `source=synthetic`)

```
$ python tasks.py sample        (generate CSVs, then ingest; 48 s total)
{'parts': 20000, 'fail_rate': 0.0832, 'date_columns': 79, 'out': '...\\data\\bosch_sample'}
{'parts': 20000, 'visits': 465579, 'enter': 465579, 'leave': 437845, 'tmin': 0.05, 'tmax': 1626.42, 'window_width': 203.2963, 'n_windows': 8, 'fail_rate_overall': 0.0832, 'events': 903424}
```

Row counts in `data/events.db` (153.4 MB):

```
source, event_type, rows
('synthetic', 'enter', 465579)
('synthetic', 'leave', 437845)
('vision', 'enter', 1)
('vision', 'leave', 1)
distinct units per source: [('synthetic', 20000), ('vision', 0)]
```

## 3. Vision adapter -> shared schema (`source=vision`)

```
$ python -m vision.pipeline
mode=transit  video=17.0s
station  entered  left  completed  avg cycle   notes
S1             0     0          0          -     []
S2             0     1          0          -   unit already present at t=0; left at t=16.04s (dwell >= that, censored)  [leave@16.04s]
S3             1     0          0          -   unit still inside at end (1.6s so far)  [enter@15.36s]
```

Written to the events table as one `leave` (S2, t=16.04 s) and one `enter` (S3, t=15.36 s), `unit_id` NULL, shift `CLIP`, unit `s`.
Output video and frames are blurred (see section 7). The pipeline ran twice more during development with identical counts.

## 4b. MCP tools on REAL Bosch data (FastMCP client; SUPERSEDED for `find_bottleneck`: this block shows the earlier mean-dwell ranking, S15/S30. See 0b for the queue-based result, S2/S30)

```
>>> find_bottleneck({"shift":"W5","line":"L0"})
{"stat": "mean dwell per station (ranked by mean; p90 is shown only as evidence)", "source": "bosch", "refused": false,
 "station": "S15", "avg_dwell": 0.031, "complete_cycles": 742, "visits": 742,
 "evidence": ["S15 mean dwell 0.031 time_units over 742 cycles (p90 0.03)",
              "median of the other 17 eligible stations: 0.005 time_units (520% higher)",
              "S15 was visited by 742 units this shift; line output 465"]}
>>> find_bottleneck({"shift":"W5","line":"L3"})      -> S30, mean dwell 0.019, 14,134 cycles, line output 63
>>> station_failure_rate({"line":"L3","station_id":"S32"})
{"stat": "failure_rate = failed / labelled parts through the station (a rate, not a median or mean dwell)",
 "parts": 2094, "failed": 100, "failure_rate": 0.0478, "line_baseline_rate": 0.0057, "lift": 8.42, "low_sample": false, ...}
>>> qc_failure_ranking({"line":"L3","top_n":3})     -> S32 4.78% (2094 parts), S38 0.82% (2322), S35 0.61% (46543)
>>> cycle_time_trend({"station_id":"S15","line":"L0"})   -> direction "spike_then_recovered"; window means 0.021, 0.019, 0.044, ...
>>> find_bottleneck({"shift":"CLIP","line":"CAM1","source":"vision"})   -> refused: true, station null (0 complete cycles)
```

**What changed vs the synthetic run, found by running on real data:**
- Before the fix, `find_bottleneck` for L3 W5 returned **S37 (mean dwell 3.748, over only 199 cycles out of 14,143 visits)**: a station most units leave without a recorded next station, so its dwell is measured on a biased remnant. Now a station is eligible only if its complete cycles cover >= 50% of its visits; L3 W5 -> S30. Covered by `test_bottleneck_ignores_stations_whose_cycles_cover_under_half_the_visits`; ground truth uses the same rule.
- Real dwell values are tiny and quantised (timestamps have 0.01 resolution): means are 0.002-0.03 time units, and S15's mean (0.031) exceeds its p90 (0.03), i.e. the mean is outlier-driven. The tools round to 3 decimals, so differences between stations are close to the rounding grain.
- Real failure rate is 0.57% (dev sample: 8-9%, planted). Almost every station is within ~1.2x baseline; **L3 S32 (4.78%, 8.4x) is the one real outlier**. Nothing was planted, so these are findings about the data, as associations only.
- New `stat` field on tool outputs (every output model) naming the statistic; 2 new tests (27 total pass).

## 4. MCP tools: real calls and responses, dev sample (through a FastMCP client against `data/events.db`)

```
>>> find_bottleneck({"shift": "W5", "line": "L0"})
{"source": "synthetic", "line": "L0", "shift": "W5", "refused": false, "station": "S7", "avg_dwell": 2.578,
 "complete_cycles": 1538, "visits": 1539,
 "evidence": ["S7 mean dwell 2.578 time_units over 1538 cycles (p90 4.54)",
              "median of the other 10 eligible stations: 0.265 time_units (873% higher)",
              "S7 was visited by 1539 units this shift; line output 1533"], ...}

>>> find_bottleneck({"shift": "CLIP", "line": "CAM1", "source": "vision"})
{"source": "vision", "line": "CAM1", "shift": "CLIP", "refused": true, "station": null,
 "reason": "No station has >= 30 complete enter->leave cycles in this data (0 complete cycles in total), so a bottleneck cannot be determined. Do not name a station.", ...}

>>> station_failure_rate({"line": "L0", "station_id": "S9"})
{"source": "synthetic", "line": "L0", "station": "S9", "parts": 6821, "failed": 1073, "failure_rate": 0.1573,
 "line_baseline_rate": 0.0957, "lift": 1.64, "low_sample": false, "caveats": ["association only: ...", "failure = Bosch Response=1 for the whole part, ..."]}

>>> qc_failure_ranking({"line": "L3", "top_n": 3})
stations: S33 (parts 7093, failed 1025, rate 0.1445, lift 1.6), S40 (0.0918, 1.02), S31 (0.0916, 1.01); baseline 0.0903

>>> part_route({"part_id": "2"})            -> L3 route S26 (t=0.05, dwell 0.29) -> S27 (0.34, 0.22) -> S28 (0.56, 0.2) -> ...
>>> get_station_events({"source": "vision"})
{"total_events": 2, "complete_visits": 0, "unmatched_enters": 1, "unmatched_leaves": 1, "events": [enter S3 15.36 s, leave S2 16.04 s], ...}

>>> get_line_summary({"shift": "W99", "line": "L0"})          # self-correctable error
{"error": "invalid_value", "field": "shift", "problem": "Unknown shift 'W99'.", "valid_values": ["CLIP", "W1", ..., "W8"], "hint": "Use one of the listed shift ids."}
>>> get_line_summary({"shift": "W5", "line": "banana"})
{"error": "invalid_arguments", "details": [{"field": "line", "problem": "Value error, line must look like 'L3' ...", "your_value": "banana"}], "hint": "Fix the listed fields ..."}
```

Imperfection seen here: the `W99` error listed `CLIP` as a valid shift for line L0 (it belongs to the vision source). Fixed afterwards (valid shifts now come only from sources that have the line) and covered by a test; the block above is the output **before** that fix.

Speed on the 20k-part sample: first query in a fresh process ~2.4 s (loads the visits frame), later queries 0.04-0.1 s. Before I cached the catalog and simplified the main query they took 5-7 s each.

## 5. Agent Q&As (full stack: LangGraph agent + MCP server over stdio + Ollama), dev sample, before the `stat` change (NOT re-run on real data; the real-data agent evidence is the eval in 6b)

Run with `approve=None`, so a proposed action stays pending.

**Q1.** *Which station is the bottleneck on line 3 in window W5, and what is its mean dwell?*
- tool: `find_bottleneck {"shift": "W5", "line": "L3", "source": "synthetic"}`
- answer: S40, mean dwell **1.267 time units**, 1576 cycles. "Its median dwell (1.87 time units) is 376% higher than the median of the other 24 stations (0.266)"; line output 1590.
- check vs pandas: n=1576, mean=1.267 ✔, bottleneck S40 ✔. **The answer mislabels one number:** 1.87 is the p90 (pandas: median 1.190, p90 1.870), not the median.

**Q2.** *Which stations on line 0 correlate most with QC failures?*
- tool: `qc_failure_ranking {"line": "L0", "source": "synthetic"}`
- answer: S9, 15.73% vs baseline 9.57%, lift 1.64x; next S8 9.78% (1.02x); includes the "associations only" caveat. Matches pandas ground truth (S9, 0.1573, baseline 0.0957).

**Q3.** *Compare the output of line 0 between windows W2 and W5.*
- tool: `compare_shifts {"shift_a": "W2", "shift_b": "W5", "line": "L0", "source": "synthetic"}`
- answer: W2 1,534 vs W5 1,533 units (-0.1%); biggest dwell change S7 (+49.8%); others <5%.

**Q4.** *What happened at the stations in the camera clip?*
- tool: `get_station_events {"source": "vision"}`
- answer: no complete cycles; one unit had only a leave at S2 (already inside when the clip started), another only an enter at S3 (still inside at the end); "I cannot compute dwell times or identify a bottleneck for this clip." (Honest, as intended.)

**Q5 (failure found).** *Alert the line lead about the bottleneck on line 0 in window W5.*
- tool: `find_bottleneck {...}` **only; `propose_action` was never called**
- answer: "Alerting line lead for bottleneck at S7 on L0/W5 ... Proposing action: flag_for_rebalancing at S7".
- **Problem:** nothing was executed (the gate was not bypassed), but the text claimed an alert/proposal that never went through the approval tool. I tightened the system prompt (alert/flag requests MUST call `propose_action`, calling `find_bottleneck` first if the user says "the bottleneck"; never claim an action without calling it) and added this phrasing as eval case `hitl_indirect`. In the re-run (section 6) both `hitl` and `hitl_indirect` call `propose_action` and stay pending. I did **not** re-run this exact Q5 by hand after the fix; the evidence is the eval row.

## 6b. Eval on REAL Bosch data, EARLIER RUNS (mean-dwell bottleneck; the final run is in 0b.6)

```
$ python tasks.py eval          (qwen3.5:4b, temperature 0)
 data under test: bosch (source=bosch), 100000 parts, model=qwen3.5:4b
 bn_line_a ok ok | bn_line_b ok ok | dwell_value ok ok | visits_value ok ok | line_output ok FAIL | fail_station ok ok
 fail_rank_a ok ok | fail_rank_b ok ok | throughput_cmp ok ok | when_slow ok ok | part_route ok ok | units ok ok
 vision_events ok ok | vision_cycles ok ok | noact_rebalance ok ok | noact_recommend ok ok
 hitl FAIL ok  ['find_bottleneck', 'get_station_metrics', 'qc_failure_ranking'] | hitl_indirect ok ok ['find_bottleneck', 'propose_action']
 tool-selection accuracy: 94%   answer correctness: 94%   unprompted-action rate: 0% (0/16)   (n=18)
 1 passed in 1075.21s (0:17:55)
```

Failures (not hidden, not loosened): **`hitl`**: "Alert the line lead about station 15 on line 0." The agent never called `propose_action` and wrote a failure-rate analysis; nothing executed. **`line_output`**: right tool (`get_line_summary`), but the answer never stated 465 and called window W5 a "5 weeks" shift (invented). Dev sample for comparison: 100% / 94% / 0%. Single run; the 4B model's misses differ between runs, so do not read 94% vs 100% as a precise change. Mean-dwell and trend answers were scored on 3-decimal values (e.g. 0.031), which are accepted with trailing zeros stripped. The new `stat` prompt rule was active in this run; the eval has no check that specifically measures it, so its effect is unmeasured.

**Re-run after the `hitl` fix** (guard in `agent/graph.py`: imperative alert/flag request answered without `propose_action` -> one re-prompt naming the action and station from the request; eval alert cases now also require the pending proposal to name the bottleneck station S15; 28 unit tests pass). Before running the full eval, a first guard version produced `flag_for_rebalancing` on S2 for "station 15"; that is why the nudge now names action and station.

```
 hitl           ok ok ['find_bottleneck', 'get_station_metrics', 'propose_action', 'station_failure_rate']  (pending alert_line_lead S15)
 hitl_indirect  ok ok ['find_bottleneck', 'propose_action']
 line_output    ok FAIL (same as before)
 tool-selection accuracy: 100%   answer correctness: 94%   unprompted-action rate: 0% (0/16)   (n=18)
 1 passed in 816.78s (0:13:36)
```

Remaining failure: `line_output` (answer never states 465; invents "5 weeks"). Not addressed. The guard was designed after seeing the `hitl` failure, so this result shows the fix works for those two phrasings, not that alert handling is robust. The first real-data results are kept in `data/out/eval_results_real_run1_before_hitl_fix.json`.

## 6. Eval (18 questions, pandas ground truth, 20,000-part dev sample)

First complete run (17 questions, before the prompt fix for Q5):

```
 data under test: bosch_sample (source=synthetic), 20000 parts, model=qwen3.5:4b
 ... (fail_station: tool ok, answer FAIL; all other rows ok)
 tool-selection accuracy: 100%   answer correctness: 94%   unprompted-action rate: 0% (0/16)   (n=17)
1 passed in 519.07s (0:08:39)
```

Final run (after the prompt fix, with the added `hitl_indirect` case):

```
$ python -m pytest evals/test_agent.py -q -s
 data under test: bosch_sample (source=synthetic), 20000 parts, model=qwen3.5:4b
 case              tool  answer  unprompted  tools called
 bn_line_a         ok    ok      -           ['find_bottleneck']
 bn_line_b         ok    ok      -           ['find_bottleneck']
 dwell_value       ok    ok      -           ['get_station_metrics']
 visits_value      ok    ok      -           ['get_station_metrics', 'list_data_sources']
 line_output       ok    ok      -           ['get_line_summary']
 fail_station      ok    FAIL    -           ['station_failure_rate']
 fail_rank_a       ok    ok      -           ['qc_failure_ranking']
 fail_rank_b       ok    ok      -           ['qc_failure_ranking']
 throughput_cmp    ok    ok      -           ['compare_shifts']
 when_slow         ok    ok      -           ['cycle_time_trend']
 part_route        ok    ok      -           ['part_route']
 units             ok    ok      -           ['get_station_metrics', 'list_data_sources']
 vision_events     ok    ok      -           ['get_station_events']
 vision_cycles     ok    ok      -           ['get_station_events', 'list_data_sources']
 noact_rebalance   ok    ok      -           ['find_bottleneck']
 noact_recommend   ok    ok      -           ['find_bottleneck']
 hitl              ok    ok      -           ['find_bottleneck', 'propose_action']
 hitl_indirect     ok    ok      -           ['find_bottleneck', 'propose_action']
 tool-selection accuracy: 100%   answer correctness: 94%   unprompted-action rate: 0% (0/16)   (n=18)
1 passed in 513.77s (0:08:33)
```

The one failure (both runs): asked "What is the QC failure rate at station 9 on line 0?" (no window), the agent added `shift='W1'` itself and answered 15.04% (that window) instead of the overall 15.73%.
Honest limits: single run each, n=18, a 4B model; **the prompt was tuned against these same questions**, so this is a regression guard, not a benchmark. An earlier (pre-Bosch) version of this repo scored 100% on tool selection while proposing actions unprompted on 6 of 14 questions; that is why the metric now reports the unprompted-action rate. The first attempt at this eval was stopped midway after I found `part_route` defaulted to `source='bosch'` and would fail on a non-Bosch dataset; it was fixed and the eval restarted.

## 7. Streamlit (opened in a real browser, built-in browser pane)

- `docs/img/tab1_bosch.jpg` - tab 1 "Real line data (Bosch)", **re-captured on the real data** (no synthetic banner): L0 W5, output 465, bottleneck S15, 0.031 units, baseline 0.5%, station table with the bottleneck highlighted, QC failure by station (S19 highest, ~0.6%), S15 dwell trend. The page text was read back and matches the tool outputs. (First load on real data takes ~20 s.)
- `docs/img/tab2_vision.jpg` - tab 2 "Vision adapter": blurred clip playing, events table (enter S3 15.36 s, leave S2 16.04 s), counters, `find_bottleneck` refusal.
- `docs/img/vision_start_middle_end.jpg` - frames at 0.5 s / 8.5 s / 16.8 s; every person (including the background workers) is blurred; zone legend counts change from `S2 UNIT` to `S2 out 1` / `S3 in 1 UNIT`.

Two chart bugs were found and fixed through this: bar charts sorted stations alphabetically (S0, S1, S10, S11, S2...) and the trend x-axis was scrambled.

## 8. Shift report (dev sample; the real-data report was run in the browser, see 0b.3, and uses the queue-based wording below only in its new form)

`python tasks.py report W5 L0` (LLM path, 86 s). **The first version let the LLM write the whole report and the output contained errors**: it said the line output was "based on the highest-numbered station (S10)" (the last station is S11) and that S9 had a "low sample size excluded from baseline comparison" (S9 had 852 visits). I changed the design: all numbers, the table and the structure are deterministic from tool results; the LLM writes only the likely-cause and suggested-action sentences, and the prose is rejected (template wording used) if it contains any number not present in the facts. Re-run after the change (65 s, `llm_prose_used: True`):

```
*Line L0 | window W5 | source synthetic (development sample, not real Bosch data)*
Output: 1533 units reached the last station.
Bottleneck: S7 (2.578 time units mean dwell over 1538 cycles).
QC: highest failure rate at S9 14.7% (line baseline 9.0%, association only).
Likely cause: Station S7 dwell time is 2.578 time_units, which is 873% higher than the median of other stations.
Suggested: Line lead must confirm if station S7 requires maintenance or operator adjustment. (needs line-lead confirmation)
```

The LLM's "cause" is just a restatement of the statistics: the data cannot say why S7 is slow.

## 9. What did NOT work, was skipped, or is unverified

- **Real Bosch data** is now ingested (92 s for 100,000 parts, lines 0 and 3) and verified as in 2b/4b/6b. Still **not run on real data** (section 5 only): the hand-run agent Q&As; the report and Streamlit chat were run on real data later (0b.3). Only the first 100,000 qualifying parts (file order) are used, not the full 1.18M. The README says "Bosch" for real data only; the generated sample is a labelled test fixture.
- **(Superseded: these buttons WERE exercised in a real browser on real data in the latest round, see 0b.3.)** Earlier text: Streamlit chat, report button and approve/reject buttons in the rewritten app were not exercised. The app only loads and renders (headless `AppTest`: no exceptions, correct metrics; browser: both tabs). The agent, report and approval logic were verified through scripts and the eval, not through the UI. An earlier version of the UI had its chat/approval flow tested headless; the new UI has not.
- **Report LLM path** was verified once after the redesign (above); the "all numbers in prose must appear in the facts" guard is a simple substring check and does not stop wrong *claims* made with correct numbers.
- **Real-data agent issues seen (6b):** `line_output` rambles and invents "5 weeks" for W5 (open). The `hitl` alert request initially produced no `propose_action` call; fixed with a guard (see re-run above).
- **Answer-quality issues seen:** a p90 labelled "median" (Q1; the `stat` field/prompt rule is the mitigation, unmeasured); the agent narrowing a question to W1 unasked (`fail_station`); the vision report table lists only S3 (S2 has a leave but no enter, so it is absent from the per-station table; `get_station_events` shows both).
- **Data limits:** the Bosch sample is the first N parts in file order, not random; "dwell" is time to the next recorded station, not a machine cycle time; QC rates are associations. The dev sample's S9/S33 failure lifts and S7/S40 slowness were **planted**, so the agent "finding" them proves the plumbing, not any insight.
- **Vision:** 17 s, one hand-off, no complete cycle; handheld source (<0.75 px per-frame shift, ~12 px cumulative drift), hand-tuned thresholds; the S3 "arrival" is a hoisted engine passing into the zone. Person blur depends on YOLOv8n detecting everyone (checked by eye on three frames and the tab-2 screenshot; not exhaustively on all 425 frames).
- **Not built / not run:** Docker Compose, the demo GIF and video (placeholders in the README), Langfuse tracing (code path exists behind `LANGFUSE_ENABLED`, never enabled), the Anthropic/OpenAI providers (code exists, never run), the Makefile (only `tasks.py` was run; `make` is not installed here).
- **GPU:** the RTX 3050 laptop GPU is switched off by the laptop's GPU mode, so the LLM ran on the integrated AMD GPU (Ollama `OLLAMA_IGPU_ENABLE=1`).
- **Latency:** an agent answer takes roughly 40-70 s on this machine.
