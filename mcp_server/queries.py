"""Metric computations over the shared `events` table. No MCP/LLM code here, so evals and tests call it directly.

Definitions used everywhere (and repeated in tool `caveats`):
  visit        a unit entering a station (one `enter` event)
  cycle        a visit that also has a `leave` event; dwell = leave.ts - enter.ts, in the source's native time unit
  Bosch dwell  = time until the part reaches its next recorded station: queueing + processing + transport,
               not a machine cycle time. Bosch time units are anonymised, never hours.
"""
from __future__ import annotations

import os
import re
import statistics
from functools import lru_cache

import numpy as np
import pandas as pd

from common import events as E
from mcp_server.models import (Bottleneck, CycleTrend, DataCatalog, EventRow, FailureRanking, FailureRate,
                               LineSummary, PartRoute, RouteStep, ShiftComparison, SourceInfo, StationEvents,
                               StationMetrics, ThroughputPoint, ThroughputTrend, TrendPoint)

PRIORITY = ["bosch", "vision", "synthetic"]


class QueryError(Exception):
    """A user-correctable problem; the server returns it as structured JSON so the agent can retry."""

    def __init__(self, field: str, problem: str, valid=None, hint: str | None = None):
        super().__init__(problem)
        self.field, self.problem, self.valid, self.hint = field, problem, valid, hint


def _con():
    return E.connect()


def _num(s: str) -> int:
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


def _caveats(source: str, unit: str) -> list[str]:
    c = ["dwell = time from entering a station until the unit enters its next recorded station "
         "(includes waiting and transport); it is not a machine cycle time"] if source != "synthetic" else []
    if unit == "time_units":
        c.append("timestamps are anonymised 'time units', not clock time; do not convert them to hours or minutes")
    if source == "vision":
        c.append("camera data covers a short clip; units are not tracked between stations")
    return c


# ----------------------------------------------------------------------------- catalog / resolution
def catalog() -> DataCatalog:
    return _catalog(_db_key())


@lru_cache(maxsize=8)
def _catalog(key: tuple) -> DataCatalog:       # cached per DB state: the GROUP BY scans every event
    con = _con()
    E.ensure_catalog(con)
    out: list[SourceInfo] = []
    agg: dict[tuple, dict] = {}
    for g in con.execute("SELECT source, line, station, shift, ts_unit, event_type, n FROM catalog_agg"):
        a = agg.setdefault((g["source"], g["line"]), {"st": set(), "sh": set(), "n": 0, "v": 0, "unit": g["ts_unit"]})
        a["st"].add(g["station"]); a["sh"].add(g["shift"]); a["n"] += g["n"]
        a["v"] += g["n"] if g["event_type"] == "enter" else 0
    for (src, ln), a in sorted(agg.items(), key=lambda kv: (PRIORITY.index(kv[0][0]), _num(kv[0][1]))):
        r = {"source": src, "line": ln, "ts_unit": a["unit"], "n": a["n"], "u": a["v"]}
        st = sorted(a["st"], key=_num)
        sh = sorted(a["sh"], key=lambda s: (_num(s), s))
        meta = {m["key"]: m["value"] for m in con.execute("SELECT key, value FROM meta WHERE source=?", (r["source"],))}
        notes = []
        if r["source"] == "bosch" or meta.get("ingest") == "ingest.bosch":
            notes.append(f"{meta.get('n_windows', '?')} equal time windows act as shifts; window width "
                         f"{float(meta.get('window_width', 0)):.2f} time units")
        out.append(SourceInfo(source=r["source"], line=r["line"], stations=st, shifts=sh, time_unit=r["ts_unit"],
                              events=r["n"], visits=r["u"], notes=notes))
    con.close()
    return DataCatalog(sources=out)


def _resolve(source: str | None, line: str | None, station: str | None = None, shift: str | None = None,
             need_line: bool = True) -> tuple[str, str | None]:
    """Pick (source, line). Auto-picks bosch > vision > synthetic; raises QueryError with the valid options."""
    cat = catalog().sources
    if not cat:
        raise QueryError("source", "No events are loaded.", hint="Run the Bosch ingest (docs/DATA.md) or the vision pipeline.")
    src_shifts = {s: {x for c in cat if c.source == s for x in c.shifts} for s in {c.source for c in cat}}
    # a shift is valid if it exists anywhere in the source; a line with no events in it simply returns empty metrics
    cands = [c for c in cat if (source is None or c.source == source) and (line is None or c.line == line)
             and (station is None or station in c.stations) and (shift is None or shift in src_shifts[c.source])]
    if not cands:
        if source and all(c.source != source for c in cat):
            raise QueryError("source", f"No data for source '{source}'.", valid=sorted({c.source for c in cat}))
        if line and all(c.line != line for c in cat if source is None or c.source == source):
            raise QueryError("line", f"Unknown line '{line}'.", valid=sorted({c.line for c in cat if source is None or c.source == source}))
        if station:
            raise QueryError("station_id", f"Station '{station}' not found" + (f" on line {line}" if line else "") + ".",
                             valid=sorted({s for c in cat if (line is None or c.line == line) and (source is None or c.source == source) for s in c.stations}, key=_num),
                             hint="Call list_data_sources to see stations per line.")
        srcs = {c.source for c in cat if (line is None or c.line == line) and (source is None or c.source == source)}
        shifts = sorted({s for c in cat if c.source in srcs for s in c.shifts}, key=_num)   # only shifts of sources that have this line
        raise QueryError("shift", f"Unknown shift '{shift}'.", valid=shifts, hint="Use one of the listed shift ids.")
    cands.sort(key=lambda c: (PRIORITY.index(c.source), _num(c.line)))
    best_src = cands[0].source
    same = [c for c in cands if c.source == best_src]
    if need_line and line is None and len(same) > 1:
        raise QueryError("line", f"Several lines in source '{best_src}'; specify one.", valid=[c.line for c in same])
    return best_src, (same[0].line if line is None else line)


# ----------------------------------------------------------------------------- core frames (cached per DB state)
def _db_key() -> tuple:
    p = E.db_path()
    return (str(p), p.stat().st_mtime_ns if p.exists() else 0)


@lru_cache(maxsize=64)
def _visits_df(key: tuple, source: str, line: str) -> pd.DataFrame:
    """One row per visit: unit, station, shift, enter_ts, leave_ts (NaN if none), qc_failed (NaN if unknown).

    The MCP client starts a fresh server process per tool call, and building this frame from 1M+ events takes ~14 s on the real
    Bosch sample, so it is cached on disk next to the DB, keyed by the DB's mtime (a re-ingest invalidates it)."""
    cache = E.db_path().with_name(f"{E.db_path().stem}.{source}.{line}.{key[1]}.visits.pkl")
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except Exception:
            pass                                            # corrupt/partial file: rebuild
    df = _build_visits_df(source, line)
    try:
        for old in cache.parent.glob(f"{E.db_path().stem}.{source}.{line}.*.visits.pkl"):
            old.unlink(missing_ok=True)                     # drop caches of earlier DB states
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        df.to_pickle(tmp)
        tmp.replace(cache)                                  # atomic: concurrent servers never read a half-written file
    except OSError:
        pass
    return df


def _build_visits_df(source: str, line: str) -> pd.DataFrame:
    con = _con()
    df = pd.read_sql_query("SELECT unit_id, visit_id, station, shift, ts AS enter_ts, qc_failed, ts_unit FROM events "
                           "WHERE source=? AND line=? AND event_type='enter'", con, params=(source, line))
    lv = pd.read_sql_query("SELECT visit_id, MIN(ts) AS leave_ts FROM events "
                           "WHERE source=? AND line=? AND event_type='leave' GROUP BY visit_id", con, params=(source, line))
    con.close()
    df = df.merge(lv, on="visit_id", how="left")
    df["dwell"] = df["leave_ts"] - df["enter_ts"]
    df["seq"] = df["station"].map(_num)
    return df


def _visits(source: str, line: str, shift: str | None = None) -> pd.DataFrame:
    df = _visits_df(_db_key(), source, line)
    return df if shift is None else df[df["shift"] == shift]


def _unit(df: pd.DataFrame) -> str:
    return df["ts_unit"].iloc[0] if len(df) else "time_units"


def _label(source: str, line: str, station: str) -> str:
    con = _con()
    r = con.execute("SELECT label FROM station_labels WHERE source=? AND line=? AND station=?", (source, line, station)).fetchone()
    con.close()
    return r["label"] if r else station


TICK = 0.01        # Bosch timestamp resolution: 1 tick = 0.01 time units. Dwells of 0.002-0.03 are 0-3 ticks, so means are mostly rounding noise.


def _window_width(source: str, df: pd.DataFrame) -> float | None:
    """Width of one window in time units: from the ingest metadata, else the span of the data in hand."""
    con = _con()
    r = con.execute("SELECT value FROM meta WHERE source=? AND key='window_width'", (source,)).fetchone()
    con.close()
    if r:
        return float(r["value"])
    return float(df["enter_ts"].max() - df["enter_ts"].min()) or None if len(df) else None


def _queue(d: pd.DataFrame, width: float | None) -> tuple[float | None, int | None]:
    """(time-averaged parts in queue, peak simultaneous parts) over complete visits: a part is 'in queue' from entering
    this station until entering its next recorded station. avg = sum(dwell) / window width (Little's law)."""
    c = d.dropna(subset=["dwell"])
    if not len(c) or not width:
        return None, None
    ts = np.concatenate([c["enter_ts"].to_numpy(), c["leave_ts"].to_numpy()])
    delta = np.concatenate([np.ones(len(c)), -np.ones(len(c))])
    order = np.lexsort((delta, ts))                    # by time; at equal times leaves (-1) before enters (+1)
    return round(float(c["dwell"].sum()) / width, 4), int(np.cumsum(delta[order]).max())


def _metrics(df: pd.DataFrame, source: str, line: str, station: str, shift: str, unit: str, width: float | None = None) -> StationMetrics:
    d = df[df["station"] == station]
    cyc = d["dwell"].dropna()
    lab = d["qc_failed"].dropna()
    aq, pq = _queue(d, width)
    ticks = round(float(np.median(np.round(cyc / TICK))), 1) if len(cyc) and unit == "time_units" else None
    return StationMetrics(
        source=source, line=line, station=station, label=_label(source, line, station), shift=shift,
        visits=len(d), complete_cycles=len(cyc), throughput=len(d), avg_queue=aq, peak_queue=pq, median_dwell_ticks=ticks,
        avg_dwell=round(float(cyc.mean()), 3) if len(cyc) else None,
        p90_dwell=round(float(cyc.quantile(0.9)), 3) if len(cyc) else None,
        qc_labelled=len(lab), qc_fail_rate=round(float(lab.mean()), 4) if len(lab) else None, time_unit=unit)


# ----------------------------------------------------------------------------- tools
def line_summary(shift: str, line: str | None = None, source: str | None = None, **_) -> LineSummary:
    source, line = _resolve(source, line, shift=shift)
    df = _visits(source, line, shift)
    unit = _unit(_visits(source, line))
    stations = sorted(df["station"].unique(), key=_num)
    width = _window_width(source, _visits(source, line))
    sts = [_metrics(df, source, line, s, shift, unit, width) for s in stations]
    out = sts[-1].visits if sts else 0
    return LineSummary(source=source, line=line, shift=shift, line_output=out, time_unit=unit, stations=sts,
                       caveats=_caveats(source, unit) + ["line_output = units entering the highest-numbered station seen in this shift"])


def station_metrics(station_id: str, shift: str, line: str | None = None, source: str | None = None, **_) -> StationMetrics:
    source, line = _resolve(source, line, station=station_id, shift=shift, need_line=False)
    df = _visits(source, line, shift)
    return _metrics(df, source, line, station_id, shift, _unit(_visits(source, line)), _window_width(source, _visits(source, line)))


MIN_COVERAGE = 0.5          # share of a station's visits that must have a measurable dwell to be bottleneck-eligible


def find_bottleneck(shift: str, line: str | None = None, source: str | None = None, min_cycles: int = 30, **_) -> Bottleneck:
    ls = line_summary(shift, line, source)
    cav = ls.caveats
    # eligible = enough complete cycles AND cycles cover at least half the visits; otherwise a station whose units
    # mostly have no recorded next station (e.g. the last station of a route) is ranked on a biased remnant.
    elig = [s for s in ls.stations if s.complete_cycles >= min_cycles and s.avg_dwell is not None
            and s.complete_cycles >= MIN_COVERAGE * s.visits]
    if not elig:
        have = sum(s.complete_cycles for s in ls.stations)
        return Bottleneck(source=ls.source, line=ls.line, shift=shift, refused=True, time_unit=ls.time_unit,
                          reason=(f"No station has >= {min_cycles} complete enter->leave cycles in this data "
                                  f"({have} complete cycles in total), so a bottleneck cannot be determined. "
                                  "Do not name a station."),
                          visits=sum(s.visits for s in ls.stations), caveats=cav)
    # Ranking key = time-averaged parts in queue (WIP). Mean dwell is NOT used: Bosch timestamps have 0.01 resolution, so
    # per-station dwells are 0-3 ticks and their means are mostly rounding noise; queue length and throughput are counts.
    # Ties: higher peak queue, then higher mean dwell, then lower station number.
    elig = [s for s in elig if s.avg_queue is not None]
    worst = max(elig, key=lambda s: (s.avg_queue, s.peak_queue or 0, s.avg_dwell, -_num(s.station)))
    others = [s.avg_queue for s in elig if s is not worst]
    tk = f", median dwell {worst.median_dwell_ticks:g} tick(s) (1 tick = {TICK} {ls.time_unit})" if worst.median_dwell_ticks is not None else ""
    ev = [f"{worst.station} averaged {worst.avg_queue} parts in queue (peak {worst.peak_queue}) in {worst.window_label} ({shift}); "
          f"throughput {worst.throughput} parts{tk}"]
    if others:
        med = statistics.median(others)
        ev.append(f"median avg queue of the other {len(others)} eligible stations: {med:.4f} parts"
                  + (f" ({worst.avg_queue / med:.1f}x)" if med else ""))
    if worst.median_dwell_ticks is not None and worst.median_dwell_ticks <= 1:
        ev.append(f"median dwell at {worst.station} is only {worst.median_dwell_ticks:g} tick(s): at this timestamp resolution "
                  "per-station dwell is barely measurable, so treat the ranking as weak evidence")
    ev.append(f"line output {ls.line_output} units in {worst.window_label} ({shift})")
    return Bottleneck(source=ls.source, line=ls.line, shift=shift, refused=False, station=worst.station,
                      avg_dwell=worst.avg_dwell, avg_queue=worst.avg_queue, peak_queue=worst.peak_queue, throughput=worst.throughput,
                      median_dwell_ticks=worst.median_dwell_ticks, complete_cycles=worst.complete_cycles, visits=worst.visits,
                      evidence=ev, time_unit=ls.time_unit, caveats=cav)


def cycle_time_trend(station_id: str, shift: str | None = None, line: str | None = None, source: str | None = None,
                     window: float | None = None, **_) -> CycleTrend:
    source, line = _resolve(source, line, station=station_id, shift=shift, need_line=False)
    df = _visits(source, line, shift)
    d = df[(df["station"] == station_id) & df["dwell"].notna()]
    unit = _unit(_visits(source, line))
    if len(d) < 2:
        return CycleTrend(source=source, line=line, station=station_id, shift=shift, bucket_width=window or 0.0, time_unit=unit,
                          points=[], direction="insufficient_data", caveats=_caveats(source, unit) + ["fewer than 2 complete cycles"])
    t0, t1 = float(d["enter_ts"].min()), float(d["enter_ts"].max())
    w = window or max((t1 - t0) / 8, 1e-9)
    b = ((d["enter_ts"] - t0) // w).astype(int)
    pts = [TrendPoint(window_start=round(t0 + k * w, 3), window_end=round(t0 + (k + 1) * w, 3), cycles=len(g),
                      avg_dwell=round(float(g["dwell"].mean()), 3)) for k, g in d.groupby(b)]
    if len(pts) < 3:
        direction = "insufficient_data"
    else:
        first, last, peak = pts[0].avg_dwell, pts[-1].avg_dwell, max(p.avg_dwell for p in pts)
        direction = ("degrading" if last > first * 1.10 else "spike_then_recovered" if peak > first * 1.10
                     else "improving" if last < first * 0.9 else "stable")
    return CycleTrend(source=source, line=line, station=station_id, shift=shift, bucket_width=round(w, 3), time_unit=unit,
                      points=pts, direction=direction, caveats=_caveats(source, unit))


def compare_shifts(shift_a: str, shift_b: str, line: str | None = None, source: str | None = None, **_) -> ShiftComparison:
    a = line_summary(shift_a, line, source)
    b = line_summary(shift_b, a.line, a.source)
    ia = {s.station: s for s in a.stations}
    deltas = {s.station: round(100 * (s.avg_dwell / ia[s.station].avg_dwell - 1), 1) for s in b.stations
              if s.station in ia and s.avg_dwell and ia[s.station].avg_dwell}
    return ShiftComparison(
        source=a.source, line=a.line, shift_a=shift_a, shift_b=shift_b, line_output_a=a.line_output, line_output_b=b.line_output,
        output_delta_pct=round(100 * (b.line_output / a.line_output - 1), 1) if a.line_output else None,
        station_dwell_delta_pct=deltas,
        biggest_dwell_change_station=max(deltas, key=lambda k: abs(deltas[k])) if deltas else None,
        time_unit=a.time_unit, caveats=a.caveats)


def _failure(df: pd.DataFrame, source: str, line: str, station: str, shift: str | None, base: float | None) -> FailureRate:
    d = df[df["station"] == station].dropna(subset=["qc_failed"])
    d = d.drop_duplicates("unit_id") if d["unit_id"].notna().any() else d
    n, f = len(d), int(d["qc_failed"].sum())
    rate = round(f / n, 4) if n else None
    return FailureRate(source=source, line=line, station=station, shift=shift, parts=n, failed=f, failure_rate=rate,
                       line_baseline_rate=round(base, 4) if base is not None else None,
                       lift=round(rate / base, 2) if (rate is not None and base) else None, low_sample=n < 30,
                       caveats=["association only: units that pass a station differ in many ways; this is not proof of cause",
                                "failure = Bosch Response=1 for the whole part, not a failure measured at this station"])


def _baseline(df: pd.DataFrame) -> float | None:
    u = df.dropna(subset=["qc_failed"]).drop_duplicates("unit_id")
    return float(u["qc_failed"].mean()) if len(u) else None


def station_failure_rate(line: str, station_id: str, shift: str | None = None, source: str | None = None, **_) -> FailureRate:
    source, line = _resolve(source, line, station=station_id, shift=shift)
    df = _visits(source, line, shift)
    return _failure(df, source, line, station_id, shift, _baseline(df))


def qc_failure_ranking(line: str, shift: str | None = None, source: str | None = None, top_n: int = 5,
                       min_parts: int = 50, **_) -> FailureRanking:
    source, line = _resolve(source, line, shift=shift)
    df = _visits(source, line, shift)
    base = _baseline(df)
    rows = [_failure(df, source, line, s, shift, base) for s in df["station"].unique()]
    rows = [r for r in rows if r.parts >= min_parts and r.failure_rate is not None]
    rows.sort(key=lambda r: (-r.failure_rate, _num(r.station)))
    return FailureRanking(source=source, line=line, shift=shift, line_baseline_rate=round(base, 4) if base is not None else None,
                          stations=rows[:top_n],
                          caveats=["association only, not causation", f"stations with fewer than {min_parts} labelled units are excluded"])


def throughput_trend(line: str, window: float | None = None, source: str | None = None, **_) -> ThroughputTrend:
    source, line = _resolve(source, line)
    df = _visits(source, line)
    unit = _unit(df)
    if df.empty:
        return ThroughputTrend(source=source, line=line, bucket_width=window or 0.0, time_unit=unit, points=[], caveats=["no data"])
    key = df["unit_id"].fillna(df["visit_id"])
    last = df.assign(k=key).groupby("k")["enter_ts"].max()          # when each unit reached its last station on the line
    t0, t1 = float(last.min()), float(last.max())
    w = window or max((t1 - t0) / 8, 1e-9)
    b = ((last - t0) // w).astype(int)
    pts = [ThroughputPoint(window_start=round(t0 + k * w, 3), window_end=round(t0 + (k + 1) * w, 3), units=int(n))
           for k, n in b.value_counts().sort_index().items()]
    return ThroughputTrend(source=source, line=line, bucket_width=round(w, 3), time_unit=unit, points=pts,
                           caveats=["throughput = units whose last station visit on this line falls in the window"]
                                   + _caveats(source, unit)[1:])


def part_route(part_id: str, source: str | None = None, **_) -> PartRoute:
    con = _con()
    if source is None:                       # find the unit in whichever source holds it (bosch > vision > synthetic)
        have = {r[0] for r in con.execute("SELECT DISTINCT source FROM events WHERE unit_id=?", (part_id,))}
        source = next((s for s in PRIORITY if s in have), None)
        if source is None:
            srcs = [r[0] for r in con.execute("SELECT DISTINCT source FROM events WHERE unit_id IS NOT NULL")]
            ex = [r[0] for r in con.execute("SELECT DISTINCT unit_id FROM events WHERE unit_id IS NOT NULL LIMIT 5")]
            con.close()
            raise QueryError("part_id", f"No unit '{part_id}' in any source ({', '.join(srcs) or 'none'}).", valid=ex,
                             hint="valid = a few example ids")
    rows = con.execute(
        "SELECT line, station, ts, ts_unit, qc_failed, visit_id FROM events WHERE source=? AND unit_id=? AND event_type='enter' "
        "ORDER BY ts, line, station", (source, part_id)).fetchall()
    if not rows:
        ex = [r[0] for r in con.execute("SELECT DISTINCT unit_id FROM events WHERE source=? AND unit_id IS NOT NULL LIMIT 5", (source,))]
        con.close()
        raise QueryError("part_id", f"No unit '{part_id}' in source '{source}'.", valid=ex, hint="valid = a few example ids")
    leaves = {r["visit_id"]: r["ts"] for r in con.execute(
        "SELECT visit_id, ts FROM events WHERE source=? AND unit_id=? AND event_type='leave'", (source, part_id))}
    con.close()
    steps = [RouteStep(line=r["line"], station=r["station"], enter_ts=r["ts"], next_station_ts=leaves.get(r["visit_id"]),
                       dwell=round(leaves[r["visit_id"]] - r["ts"], 3) if r["visit_id"] in leaves else None) for r in rows]
    qc = rows[0]["qc_failed"]
    unit = rows[0]["ts_unit"]
    return PartRoute(source=source, part_id=part_id, qc_failed=qc, time_unit=unit, steps=steps,
                     caveats=["route = stations ordered by entry time; ties broken by station number"] + _caveats(source, unit)[1:])


def station_events(station_id: str | None = None, line: str | None = None, shift: str | None = None,
                   source: str | None = None, limit: int = 50, **_) -> StationEvents:
    source, line = _resolve(source, line, station=station_id, shift=shift, need_line=False)
    con = _con()
    where, par = "source=? AND line=?", [source, line]
    for col, v in (("station", station_id), ("shift", shift)):
        if v:
            where += f" AND {col}=?"
            par.append(v)
    rows = con.execute(f"SELECT * FROM events WHERE {where} ORDER BY ts, station, event_type", par).fetchall()
    con.close()
    enters = {r["visit_id"] for r in rows if r["event_type"] == "enter"}
    leaves = {r["visit_id"] for r in rows if r["event_type"] == "leave"}
    unit = rows[0]["ts_unit"] if rows else "s"
    return StationEvents(
        source=source, time_unit=unit, total_events=len(rows), complete_visits=len(enters & leaves),
        unmatched_enters=len(enters - leaves), unmatched_leaves=len(leaves - enters),
        events=[EventRow(event_type=r["event_type"], line=r["line"], station=r["station"], ts=round(r["ts"], 2), shift=r["shift"],
                         visit_id=r["visit_id"], unit_id=r["unit_id"]) for r in rows[:limit]],
        caveats=_caveats(source, unit) + (["a unit with only a leave was already inside when the data began; one with only an "
                                           "enter was still inside when it ended: neither gives a cycle time"] if source == "vision" else []))

