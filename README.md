# line-supervisor-agent

- **What it is:** an AI supervisor layer for production lines: one shared event schema fed by real Bosch data and a camera adapter, MCP tools on top, and a LangGraph agent that answers supervisor questions and writes the shift report.
- **Real-data finding:** on Bosch line 3, station S32 has a QC failure rate of 4.78% (100 of 2,094 parts), 8.4x the 0.57% baseline: an association, not a cause.
- **Safety:** actions (alert the line lead, flag a station) only execute after explicit human approval.

**Vision gives you events. This repo is the AI supervisor layer that turns line events into decisions.**

## What it looks like (real Bosch data, captured from the running app)

| | |
|---|---|
| ![Bosch tab](docs/img/tab1_bosch.jpg) **Bosch tab:** line output, bottleneck (average queue), throughput / queue / median-dwell-in-ticks table, QC failure by station | ![QC outlier on L3](docs/img/qc_outlier_l3_s32.jpg) Real Bosch data: L3 S32 fails QC at 4.78%, 8.4x the line baseline (an association, not a cause). |
| ![Report and chat](docs/img/report_and_chat.jpg) **Shift report + chat** (numbers from the tools; LLM writes two sentences) | ![Approval](docs/img/chat_approval_pending.jpg) **Human approval:** `alert_line_lead -> S2` waits for Approve/Reject; nothing is logged until approved |
| ![Rejected](docs/img/chat_rejected.jpg) **Rejected:** nothing executed, the agent offers alternatives | ![Vision tab](docs/img/tab2_vision.jpg) **Vision tab:** camera clip -> same event schema, people blurred |
| ![Frames](docs/img/vision_start_middle_end.jpg) Start / middle / end frames (blurred) | |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && ollama pull qwen3.5:4b   # agent, MCP tools, Streamlit app, tests (no torch / OpenCV)
pip install -r requirements-vision.txt                      # OPTIONAL: camera adapter only (adds torch, ultralytics, OpenCV, ffmpeg)
python tasks.py ingest        # real Bosch data in data/bosch/ (see docs/DATA.md); or: python tasks.py sample (dev sample)
python tasks.py vision        # optional (needs requirements-vision.txt): camera clip -> events + blurred annotated video
python tasks.py app           # Streamlit demo.  Also: python tasks.py test | eval | report W5 L0   (or make <task>)
```

`tasks.py` is a cross-platform runner (Windows has no `make`); the `Makefile` wraps it.

The Bosch tab, the MCP tools, the agent, the evals and the unit tests need only `requirements.txt`; the Vision tab just reads events and a finished video from disk, so it also loads without the extras (it then shows a hint instead of the clip). The vision pipeline itself (`python tasks.py vision`) needs `requirements-vision.txt` and says so if it is missing.

## Architecture

```mermaid
flowchart LR
    B[Bosch production data<br/>train_date + Response] --> BI[ingest.bosch]
    C[Camera footage] --> V[vision pipeline<br/>zones + presence + motion<br/>people blurred]
    BI --> S[(Shared event schema<br/>events: source, unit, line, station,<br/>enter/leave/pass, ts, shift, qc_failed)]
    V --> S
    S --> M[FastMCP tools<br/>Pydantic-validated]
    M -->|MCP| A[LangGraph supervisor agent]
    A -->|propose_action| H{{Human approval}}
    H -->|approved only| X[Alert / flag station]
    A --> R[Shift report<br/>WhatsApp + markdown]
    A --> UI[Streamlit: Bosch tab + Vision tab]
```

| Dir | What |
|---|---|
| `common/` | the shared event schema (`events.py`) |
| `ingest/` | Bosch ingest, plus the Bosch-format dev-sample generator |
| `vision/` | camera pipeline: zones, unit presence, person blur, writes `source='vision'` events |
| `mcp_server/` | metric queries, Pydantic models, 11 FastMCP tools (all read only the events table) |
| `agent/` | LangGraph agent, LLM factory (Ollama default, Anthropic/OpenAI via env), HITL gate, report |
| `evals/` | pandas ground truth, 18-question agent eval, `summarize.py` -> `RESULTS.md` (latest run, per question) |
| `tests/` | ingest, schema, tool and cross-check unit tests on a hand-computed fixture |
| `app/` | Streamlit demo |

## Data provenance (what comes from where)

| Number you see | Source | Notes |
|---|---|---|
| Line output, station dwell, p90, throughput trend, shift/window comparisons | **Bosch** (`source='bosch'`) | Derived from the date columns; time is anonymised *time units*, never hours. Windows `W1..Wn` are fixed slices of the timeline used as shifts. |
| QC failure rate / ranking by station | **Bosch** (`Response`) | Failure is per *part*, so station rates are associations, not causes. |
| Station enter/leave events from the camera clip | **Vision** (`source='vision'`) | 17 s window of a stock clip; seconds from clip start; no unit identity. |
| Anything shown while `data/bosch/` is missing | **Synthetic (test fixture only)** | `python tasks.py sample` generates a Bosch-*format* sample with planted structure (`source='synthetic'`) so the app and tests run without the 5 GB download. It is **not** Bosch data, the app and agent label it, and no number in this README comes from it. Unit tests use a tiny hand-written fixture. |

"Bosch" in this README means the real Kaggle *Bosch Production Line Performance* files (`train_date.csv`, `train_numeric.csv`): the first 100,000 parts (file order) that visit lines 0 or 3, 2,157,871 events, overall failure rate 0.57%.

## The vision adapter

A working CV pipeline converts footage into the same event schema the Bosch data uses. Public fixed-camera factory footage is scarce, so it is shown on a **17 s stable window of a stock clip**: it detects a station hand-off (a unit leaves S2, one arrives at S3), and it **refuses to report cycle times** because no station has a complete enter-to-leave cycle. On a fixed CCTV feed the same code would emit continuous events. Details: YOLO person detection runs only to apply a Gaussian blur to every person (no boxes drawn, no IDs, nothing stored); unit presence per zone comes from dark-engine pixels confirmed by motion; zones are padded 15 px against slow camera drift (no stabilisation). The source clip is not committed.

## Evals

`python tasks.py eval` asks the agent 18 questions and scores it against **ground truth computed directly from the CSVs in pandas** (independent of the ingest/SQL/tool code, which is itself cross-checked against it in `tests/test_crosscheck.py`): 11 Bosch-data questions, 2 camera questions, 2 "advise me but do not act" questions, 2 requests that *should* produce an approval (one direct, one phrased "the bottleneck"), and 1 unit-conversion trap. Metrics: **tool-selection accuracy**, **answer correctness**, **unprompted-action rate**.

**Last two real-data runs** (`source=bosch`, 100,000 parts, lines 0 and 3; `qwen3.5:4b` via Ollama on the RTX 3050, temperature 0, n=18 each). Ground truth was recomputed with pandas from the real CSVs, using the queue/throughput bottleneck definition below.

| Metric | Result (range over the last two runs) |
|---|---|
| Tool-selection accuracy | **100%** (18/18 in both runs) |
| Answer correctness vs pandas ground truth | **94-100%** (17/18 and 18/18; n=18) |
| Unprompted-action rate (non-action questions that called `propose_action`) | **0%** (0/16 in both runs) |

The one miss that comes and goes is `fail_station` ("QC failure rate at station 19 on line 0", no window): the model sometimes adds a window it wasn't asked for and reports that window's rate. That is run-to-run variance on a 4B model, not a tool bug (the tool returns the overall rate when no window is passed). These scores are a **regression guard, not a benchmark**: n=18, and the prompt, tool descriptions and action policy were all tuned after failures on these same questions. Per-question results of the latest run, with tools called and a reason for any failure: [evals/RESULTS.md](evals/RESULTS.md) (regenerate with `python tasks.py eval && python -m evals.summarize`).

**What the eval caught** (what broke → how it was caught → fix):

- **Unprompted action proposals.** A plain "what is the bottleneck?" got an "alert the line lead" proposal on 6 of 14 questions while tool selection scored 100% → the metric only checked that an expected tool was called → added the unprompted-action rate and a "propose only when asked" rule.
- **"Alerting…" written without calling the gated tool.** The agent said it alerted the line lead but never called `propose_action` → found in a hand-run Q&A → added the `hitl` and `hitl_indirect` eval cases, a stricter prompt, and a guard that re-prompts once with the action and station.
- **Bottleneck ranked on too few complete cycles.** L3's "bottleneck" S37 rested on 199 of its 14,143 visits → seen on the first real-data run → a station needs complete cycles on at least 50% of its visits (regression test).
- **Mean dwell was rounding noise.** At 0.01 resolution dwells are 0-3 ticks (L0 W5 S15: mean 0.031 above its p90 of 0.03) → noticed on the real data → rank by average parts in queue, and report throughput, peak queue and median dwell in ticks.
- **A p90 called a "median".** → caught checking answers against pandas → every tool output has an explicit `stat` field and the agent must name the statistic.
- **Action detector too eager, and a named station ignored.** The first regex would read "Flag rate at S3?" as an alert request, and "alert… station 15" was proposed for S2 → found reviewing the regex and in a live check → `agent/policy.py` (questions, noun use, "warn me…" and negations are not requests) plus a named-station override that notes "station set from your request" in the approval reason.

## Design decisions

- **Why MCP between the data and the agent.** The agent never sees SQL, CSVs or CV code, only a small typed tool surface. Metric logic lives in one testable place (and is cross-checked against pandas), any MCP client can use it, and moving from SQLite to a production store changes nothing agent-side. Tool errors come back as structured `{field, valid_values, hint}`, so a small model can self-correct. `find_bottleneck` *refuses* rather than guesses when the data has no complete cycles.
- **Why human-in-the-loop on actions.** Reading metrics is safe; telling a line lead to act, or flagging a station, costs something when wrong. The only side-effecting tool, `propose_action`, pauses the graph (`interrupt`) and executes only on explicit approval; a rejection is fed back to the model.
- **Why station-level, not person-level.** The unit of analysis is the station and the line; bottlenecks and balance are what a supervisor can act on, and monitoring individuals is an ethical and adoption problem on a real floor. No face detection, no person IDs, and **people are blurred in all video and image outputs** (a missed detection would stay visible, so spot-check new footage).
- **Agent built once per session.** The Streamlit app keeps one background event loop and one MCP server process per browser session instead of rebuilding the agent every turn. Per-turn latency on the same question went from 24.0 / 14.3 / 15.0 s to 11.7 / 4.0 / 4.0 s (MCP server starts 7 → 1; the shift report 43-52 s → 20 s).
- **One schema, many sources.** Adding a source means writing `enter`/`leave` events; every tool, the agent, the report and the evals work unchanged.

## Limitations

- **Bosch dwell is not a cycle time.** Bosch gives per-station timestamps, so "dwell" is the time until a part reaches its next recorded station (waiting + processing + transport). The first station of a route and parts that skip stations make some dwells long for reasons unrelated to the station.
- **Anonymised time.** Bosch time units are not clock time; the agent is told not to convert them, and one eval case checks that.
- **Why the bottleneck is ranked by queue and throughput, not dwell.** Bosch timestamps have 0.01 resolution ("1 tick"), so per-station dwells are 0-3 ticks (means of 0.002-0.03 time units) and a *mean* dwell is mostly rounding noise (L0 W5 S15 had mean 0.031 yet p90 0.03). Counts survive that quantisation; averages of 0-3 do not. So `find_bottleneck` ranks eligible stations by **average parts in queue** (time-averaged number of parts between entering the station and entering their next one = sum of dwell / window width, Little's law), reports **throughput** (parts entering per window) and **peak queue** beside it, and reports dwell as a **median in ticks**. Mean dwell is still returned (and named as a mean) but never used to rank. Eligibility is unchanged (>= 30 complete cycles covering >= 50% of visits). Caveat: queue length is still built from the same quantised gaps, so on this data it is weak evidence of a constraint (the tool says so when the winner's median dwell is <= 1 tick). Ranked by queue; throughput is reported alongside.
- **Failure signal is weak.** Overall failure rate is 0.57%; most stations sit within ~1.2x of baseline. The one clear outlier is L3 S32 (4.78% over 2,094 parts, 8.4x baseline), still an association.
- **Sampling.** The ingest takes the *first N parts in file order* that visit the chosen lines (default 100,000, lines 0 and 3), not a random sample; windows are equal slices of that sample's timeline.
- **QC association.** `Response` is a property of the whole part; station failure rates measure association and can reflect which parts reach a station.
- **Vision.** 17 s, one hand-off, no complete cycle: it proves the adapter and the schema, not production accuracy. The camera in the source clip is handheld (per-frame shift <0.75 px but ~12 px cumulative drift in the chosen window), the presence signal is a hand-tuned heuristic, and the S3 "arrival" is a hoisted engine passing into the zone.
- **Evals** are small (n=18), single run, on a 4B local model; the system prompt was tuned against the same questions, so treat the score as a regression guard, not a benchmark. The numbers above are from real Bosch data; the 4B model's misses vary between runs.
- No auth, no streaming ingestion, Docker Compose not included. Local-LLM latency on this machine (RTX 3050 4 GB, all layers on the GPU, ~19 tokens/s) is roughly 4-18 s per chat turn (the first turn of a session also starts the MCP server), and the first load of the real data in the app takes ~15 s (later loads use an on-disk frame cache).

## What I'd build next

1. **Replace the presence heuristic with a trained part/action detector**, so camera events carry real unit identity and SOP-step labels, and join them with line-level data by `unit_id`.
2. **Continuous camera ingestion** into the same schema, per-line baselines, and drift alerts instead of window comparisons.
3. **Context beyond one line:** multi-line flow (the supervisor says "line 3 is starved because line 2's output dropped"), shift calendars, maintenance events.
4. **Alerting that earns trust:** proactive WhatsApp alerts with approval flows and an audit log of what was suggested, approved and what happened next; an eval set built from real supervisor questions.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Priyansh Bhatt. The Bosch data is not part of this repository and remains under Kaggle's competition terms.
