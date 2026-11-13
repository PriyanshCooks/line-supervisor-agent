"""Pydantic input + output models for every MCP tool. Inputs are normalised ('3' -> 'L3', '29' -> 'S29')."""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_serializer, field_validator

Source = Literal["bosch", "vision", "synthetic"]


def _line(v):
    s = str(v).strip().upper().replace("LINE", "").strip()
    if s.isdigit():
        s = "L" + s
    if not re.fullmatch(r"L\d+|CAM\d+", s):
        raise ValueError("line must look like 'L3' (Bosch lines L0-L3) or 'CAM1' (camera); a bare number is fine")
    return s


def _station(v):
    s = str(v).strip().upper()
    if s.isdigit():
        s = "S" + s
    if not re.fullmatch(r"S\d+", s):
        raise ValueError("station must look like 'S29'; a bare number is fine")
    return s


def _shift(v):
    s = str(v).strip().upper()
    return "W" + s if s.isdigit() else s


class _Base(BaseModel):
    source: Optional[Source] = Field(None, description="'bosch' (real), 'vision' (camera clip) or 'synthetic'. Omit to auto-pick (bosch first).")


class LineArgs(_Base):
    line: Optional[str] = Field(None, description="Line id, e.g. 'L3' or just 3. Required when the source has several lines.")
    shift: str = Field(description="Shift/window id such as 'W3' (Bosch) or 'CLIP' (camera). See list_data_sources.")
    min_cycles: int = Field(30, ge=1, description="Minimum complete enter->leave cycles a station needs to be eligible (find_bottleneck)")
    v_line = field_validator("line", mode="before")(lambda cls, v: None if v is None else _line(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: _shift(v))


class StationArgs(_Base):
    station_id: str = Field(description="Station id, e.g. 'S29' or just 29")
    shift: str
    line: Optional[str] = None
    v_line = field_validator("line", mode="before")(lambda cls, v: None if v is None else _line(v))
    v_station = field_validator("station_id", mode="before")(lambda cls, v: _station(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: _shift(v))


class TrendArgs(_Base):
    station_id: str
    shift: Optional[str] = Field(None, description="Limit to one shift/window; omit for the whole timeline")
    line: Optional[str] = None
    window: Optional[float] = Field(None, gt=0, description="Bucket size in the data's native time unit; omit for ~8 buckets")
    v_line = field_validator("line", mode="before")(lambda cls, v: None if v is None else _line(v))
    v_station = field_validator("station_id", mode="before")(lambda cls, v: _station(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: None if v is None else _shift(v))


class CompareArgs(_Base):
    shift_a: str
    shift_b: str
    line: Optional[str] = None
    v_line = field_validator("line", mode="before")(lambda cls, v: None if v is None else _line(v))
    v_shifts = field_validator("shift_a", "shift_b", mode="before")(lambda cls, v: _shift(v))


class FailureArgs(_Base):
    line: str = Field(description="Line id, e.g. 'L3' or 3")
    station_id: str = Field(description="Station id, e.g. 'S29' or 29")
    shift: Optional[str] = None
    v_line = field_validator("line", mode="before")(lambda cls, v: _line(v))
    v_station = field_validator("station_id", mode="before")(lambda cls, v: _station(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: None if v is None else _shift(v))


class RankArgs(_Base):
    line: str
    shift: Optional[str] = None
    top_n: int = Field(5, ge=1, le=25)
    min_parts: int = Field(50, ge=1, description="Ignore stations visited by fewer units than this")
    v_line = field_validator("line", mode="before")(lambda cls, v: _line(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: None if v is None else _shift(v))


class ThroughputArgs(_Base):
    line: str
    window: Optional[float] = Field(None, gt=0, description="Bucket size in native time units; omit for ~8 buckets")
    v_line = field_validator("line", mode="before")(lambda cls, v: _line(v))


class RouteArgs(_Base):
    part_id: str = Field(description="Part/unit id, e.g. '4' (Bosch Id)")


class EventsArgs(_Base):
    station_id: Optional[str] = None
    line: Optional[str] = None
    shift: Optional[str] = None
    limit: int = Field(50, ge=1, le=500)
    v_line = field_validator("line", mode="before")(lambda cls, v: None if v is None else _line(v))
    v_station = field_validator("station_id", mode="before")(lambda cls, v: None if v is None else _station(v))
    v_shift = field_validator("shift", mode="before")(lambda cls, v: None if v is None else _shift(v))




def window_label(shift) -> str:
    """'W5' -> 'time window 5'. Spelled out so a model never reads W5 as '5 weeks'."""
    if shift is None or shift == "ALL":
        return "whole timeline"
    m = re.fullmatch(r"W(\d+)", str(shift))
    return f"time window {m.group(1)}" if m else ("camera clip" if shift == "CLIP" else str(shift))


class _Windowed(BaseModel):
    """Every output that is scoped to a window carries its id and a spelled-out label."""

    @computed_field(description="Window id, e.g. 'W5' (one of 8 equal slices of the Bosch timeline); 'ALL' = whole timeline")
    @property
    def window(self) -> str:
        return getattr(self, "shift", None) or "ALL"

    @computed_field(description="Human label of the window, e.g. 'time window 5'. Windows are slices of anonymised time, not weeks")
    @property
    def window_label(self) -> str:
        return window_label(getattr(self, "shift", None))


# ----------------------------------------------------------------------------- outputs
class StationMetrics(_Windowed):
    stat: str = Field("throughput = parts entering the station in the window; avg_queue = time-averaged parts in queue; peak_queue = max simultaneous parts; median_dwell_ticks = MEDIAN dwell in ticks of 0.01; avg_dwell = mean; p90_dwell = 90th percentile; qc_fail_rate = share of labelled visits whose unit failed", description="Names the statistic behind each number; always quote it")
    source: str
    line: str
    station: str
    label: str
    shift: str
    visits: int = Field(description="Units that entered the station")
    complete_cycles: int = Field(description="Visits with both an enter and a leave event")
    avg_dwell: Optional[float] = Field(None, description="Mean (leave - enter) over complete cycles, in `time_unit`; None if none")
    p90_dwell: Optional[float] = None
    qc_labelled: int = Field(description="Visits whose unit has a known QC outcome")
    qc_fail_rate: Optional[float] = Field(None, description="Share of labelled visits whose unit failed QC")
    time_unit: str
    throughput: int = Field(0, description="Parts that entered this station in the window (parts per window); same number as `visits`")
    avg_queue: Optional[float] = Field(None, description="Time-averaged parts in queue = sum of dwell over complete visits / window width (Little's law)")
    peak_queue: Optional[int] = Field(None, description="Most parts simultaneously between entering this station and entering their next station")
    median_dwell_ticks: Optional[float] = Field(None, description="Median dwell in ticks (1 tick = 0.01 time units, the Bosch timestamp resolution); None if no tick")

    @computed_field(description="Ready-to-quote sentence with the exact count and window")
    @property
    def count_text(self) -> str:
        return f"{self.visits} parts entered {self.station} on {self.line} in {self.window_label} ({self.window})"


class LineSummary(_Windowed):
    source: str
    line: str
    shift: str
    line_output: int = Field(description="Units that entered the line's highest-numbered station in this shift")
    time_unit: str
    stations: list[StationMetrics]
    caveats: list[str]

    stat: str = Field(StationMetrics.model_fields["stat"].default, description="Names the statistic behind each station column; always quote it")

    @field_serializer("stations")
    def _slim_stations(self, v):
        """Each station repeats source/line/window/stat/count_text otherwise: ~3x more tokens for the same information."""
        drop = {"stat", "window", "window_label", "count_text", "source", "line", "shift", "label", "qc_labelled", "time_unit"}
        return [x.model_dump(exclude=drop) for x in v]

    @computed_field(description="Ready-to-quote sentence with the exact line output and window")
    @property
    def line_output_text(self) -> str:
        return (f"Line output for {self.line} in {self.window_label} ({self.window}) = {self.line_output} units "
                "(units that entered the line's last station in that window)")


class Bottleneck(_Windowed):
    stat: str = Field("ranked by avg_queue = time-averaged parts in queue (sum of dwell / window width); throughput (parts per window), peak_queue and MEDIAN dwell in ticks (0.01 time units) are shown alongside; mean dwell is NOT the ranking key", description="The statistic used; always quote it")
    source: str
    line: str
    shift: str
    refused: bool
    station: Optional[str] = None
    reason: Optional[str] = None
    avg_dwell: Optional[float] = None
    avg_queue: Optional[float] = Field(None, description="The ranking key: time-averaged parts in queue at the station")
    peak_queue: Optional[int] = None
    throughput: int = Field(0, description="Parts entering the station in the window")
    median_dwell_ticks: Optional[float] = Field(None, description="Median dwell in ticks of 0.01 time units")
    complete_cycles: int = 0
    visits: int = 0
    evidence: list[str] = []
    time_unit: str
    caveats: list[str] = []


class TrendPoint(BaseModel):
    window_start: float
    window_end: float
    cycles: int
    avg_dwell: float


class CycleTrend(_Windowed):
    stat: str = Field("mean dwell per time window", description="The statistic used; always quote it")
    source: str
    line: str
    station: str
    shift: Optional[str]
    bucket_width: float = Field(description='Width of each trend bucket, in time_unit')
    time_unit: str
    points: list[TrendPoint]
    direction: Literal["improving", "stable", "degrading", "spike_then_recovered", "insufficient_data"]
    caveats: list[str] = []


class ShiftComparison(BaseModel):
    stat: str = Field("line_output = count of units; station_dwell_delta_pct = % change in MEAN dwell, B vs A", description="The statistic used; always quote it")
    source: str
    line: str
    shift_a: str
    shift_b: str
    line_output_a: int
    line_output_b: int
    output_delta_pct: Optional[float]
    station_dwell_delta_pct: dict[str, float] = Field(description="Avg dwell change B vs A per station, only where both have cycles")
    biggest_dwell_change_station: Optional[str]
    time_unit: str
    caveats: list[str] = []

    @computed_field(description="Ids of the two windows compared")
    @property
    def window(self) -> str:
        return f"{self.shift_a} vs {self.shift_b}"

    @computed_field(description="Labels of the two windows compared")
    @property
    def window_label(self) -> str:
        return f"{window_label(self.shift_a)} vs {window_label(self.shift_b)}"

    @computed_field(description="Ready-to-quote sentence with the exact outputs")
    @property
    def line_output_text(self) -> str:
        return (f"Line output for {self.line}: {self.line_output_a} units in {window_label(self.shift_a)} ({self.shift_a}) vs "
                f"{self.line_output_b} units in {window_label(self.shift_b)} ({self.shift_b})")


class FailureRate(_Windowed):
    stat: str = Field("failure_rate = failed / labelled parts through the station (a rate, not a median or mean dwell)", description="The statistic used; always quote it")
    source: str
    line: str
    station: str
    shift: Optional[str]
    parts: int = Field(description="Distinct units through the station with a known QC outcome")
    failed: int
    failure_rate: Optional[float]
    line_baseline_rate: Optional[float] = Field(description="Failure rate over all labelled units on the line")
    lift: Optional[float] = Field(description="failure_rate / line_baseline_rate")
    low_sample: bool
    caveats: list[str]


class FailureRanking(_Windowed):
    stat: str = Field("ranked by failure rate = failed / labelled parts through the station", description="The statistic used; always quote it")
    source: str
    line: str
    shift: Optional[str]
    line_baseline_rate: Optional[float]
    stations: list[FailureRate]
    caveats: list[str]

    @field_serializer("stations")
    def _slim_stations(self, v):
        drop = {"stat", "window", "window_label", "source", "line", "shift", "caveats"}
        return [x.model_dump(exclude=drop) for x in v]


class ThroughputPoint(BaseModel):
    window_start: float
    window_end: float
    units: int


class ThroughputTrend(_Windowed):
    stat: str = Field("count of units entering the last station per time window", description="The statistic used; always quote it")
    source: str
    line: str
    bucket_width: float = Field(description='Width of each throughput bucket, in time_unit')
    time_unit: str
    points: list[ThroughputPoint]
    caveats: list[str]


class RouteStep(BaseModel):
    line: str
    station: str
    enter_ts: float
    next_station_ts: Optional[float] = Field(None, description="When the unit entered its next station on this line (the 'leave' event)")
    dwell: Optional[float] = None


class PartRoute(BaseModel):
    source: str
    part_id: str
    qc_failed: Optional[int]
    time_unit: str
    steps: list[RouteStep]
    caveats: list[str]


class EventRow(BaseModel):
    event_type: str
    line: str
    station: str
    ts: float
    shift: str
    visit_id: str
    unit_id: Optional[str]


class StationEvents(BaseModel):
    source: str
    time_unit: str
    total_events: int
    complete_visits: int = Field(description="Visits that have both enter and leave")
    unmatched_enters: int = Field(description="Enter with no leave: the unit was still inside when data ended")
    unmatched_leaves: int = Field(description="Leave with no enter: the unit was already inside when data began")
    events: list[EventRow]
    caveats: list[str]


class SourceInfo(BaseModel):
    source: str
    line: str
    stations: list[str]
    shifts: list[str]
    time_unit: str
    events: int
    visits: int = Field(description="Station visits (enter events)")
    notes: list[str]


class DataCatalog(BaseModel):
    sources: list[SourceInfo]


