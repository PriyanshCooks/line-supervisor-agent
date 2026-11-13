"""FastMCP server: the only way the agent touches the data. Every tool reads the shared `events` table.

Inputs are validated by Pydantic. Bad inputs come back as structured, self-correctable JSON
(`error`, per-field `details`, the `valid` values and a `hint`) rather than a stack trace.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from fastmcp import FastMCP
from pydantic import BaseModel, ValidationError

from mcp_server import queries
from mcp_server.models import (CompareArgs, EventsArgs, FailureArgs, LineArgs, RankArgs, RouteArgs, StationArgs,
                               ThroughputArgs, TrendArgs, _Base)

mcp = FastMCP("line-supervisor-metrics")


def _run(model: type[BaseModel], raw: dict[str, Any], fn: Callable[..., BaseModel]) -> str:
    try:
        args = model(**{k: v for k, v in raw.items() if v is not None})
    except ValidationError as e:
        errs = [{"field": ".".join(map(str, x["loc"])), "problem": x["msg"], "your_value": x.get("input")} for x in e.errors()]
        return json.dumps({"error": "invalid_arguments", "details": errs,
                           "hint": "Fix the listed fields and call the tool again. Call list_data_sources for valid sources, lines, stations and shifts."})
    try:
        return fn(**args.model_dump()).model_dump_json()
    except queries.QueryError as e:
        return json.dumps({"error": "invalid_value", "field": e.field, "problem": e.problem,
                           "valid_values": e.valid, "hint": e.hint or "Retry with one of the valid values."})


@mcp.tool
def list_data_sources() -> str:
    """List the loaded data: sources ('bosch' real production data, 'vision' camera clip), their lines, stations, shift ids and
    time units. Call this first if you don't know valid line/station/shift ids."""
    return queries.catalog().model_dump_json()


@mcp.tool
def get_line_summary(shift: str | int, line: str | int | None = None, source: str | None = None) -> str:
    """All stations of one line in one window: throughput (parts), avg/peak queue, MEDIAN dwell in ticks, mean/p90 dwell, QC failure rate, plus the exact line output (`line_output_text`)."""
    return _run(LineArgs, {"shift": shift, "line": line, "source": source}, queries.line_summary)


@mcp.tool
def get_station_metrics(station_id: str | int, shift: str | int, line: str | int | None = None, source: str | None = None) -> str:
    """Metrics for ONE station in one window: throughput (parts entering), avg/peak queue, MEDIAN dwell in ticks (`median_dwell_ticks`, 1 tick = 0.01 time units), mean/p90 dwell, QC rate. Use this for any single-station number, including medians."""
    return _run(StationArgs, {"station_id": station_id, "shift": shift, "line": line, "source": source}, queries.station_metrics)


@mcp.tool
def find_bottleneck(shift: str | int, line: str | int | None = None, source: str | None = None) -> str:
    """The station with the highest mean dwell among stations that have enough complete cycles, with evidence.
    Needs >= 30 complete cycles per station. Returns refused=true (and no station) when the data lacks complete cycles: never name a station in that case."""
    return _run(LineArgs, {"shift": shift, "line": line, "source": source}, queries.find_bottleneck)


@mcp.tool
def cycle_time_trend(station_id: str | int, shift: str | int | None = None, line: str | int | None = None, source: str | None = None,
                     window: float | None = None) -> str:
    """MEAN dwell per time bucket across the timeline, to show WHEN a station slowed. Not for medians, ticks or queue lengths: use get_station_metrics for those."""
    return _run(TrendArgs, {"station_id": station_id, "shift": shift, "line": line, "source": source, "window": window},
                queries.cycle_time_trend)


@mcp.tool
def compare_shifts(shift_a: str | int, shift_b: str | int, line: str | int | None = None, source: str | None = None) -> str:
    """Compare two shifts/windows: line output and per-station dwell change (B vs A)."""
    return _run(CompareArgs, {"shift_a": shift_a, "shift_b": shift_b, "line": line, "source": source}, queries.compare_shifts)


@mcp.tool
def station_failure_rate(line: str | int, station_id: str | int, shift: str | int | None = None, source: str | None = None) -> str:
    """Fraction of units through this station that failed QC (Bosch Response=1), vs the line baseline. Association, not cause."""
    return _run(FailureArgs, {"line": line, "station_id": station_id, "shift": shift, "source": source}, queries.station_failure_rate)


@mcp.tool
def qc_failure_ranking(line: str | int, shift: str | int | None = None, source: str | None = None, top_n: int = 5, min_parts: int = 50) -> str:
    """Stations on a line ranked by QC failure rate (with lift over the line baseline). Use for 'which stations correlate with failures'."""
    return _run(RankArgs, {"line": line, "shift": shift, "source": source, "top_n": top_n, "min_parts": min_parts},
                queries.qc_failure_ranking)


@mcp.tool
def throughput_trend(line: str | int, window: float | None = None, source: str | None = None) -> str:
    """Units finishing the line per time bucket (window in native time units; omit for ~8 buckets)."""
    return _run(ThroughputArgs, {"line": line, "window": window, "source": source}, queries.throughput_trend)


@mcp.tool
def part_route(part_id: str | int, source: str | None = None) -> str:
    """The ordered stations one part visited, with entry times, time to next station, and its QC outcome."""
    return _run(RouteArgs, {"part_id": part_id, "source": source}, queries.part_route)


@mcp.tool
def get_station_events(station_id: str | int | None = None, line: str | int | None = None, shift: str | int | None = None,
                       source: str | None = None, limit: int = 50) -> str:
    """Raw enter/leave events plus how many visits are complete. Use source='vision' for 'what happened at the camera stations'."""
    return _run(EventsArgs, {"station_id": station_id, "line": line, "shift": shift, "source": source, "limit": limit},
                queries.station_events)


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio by default


