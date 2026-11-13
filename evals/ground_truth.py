"""Ground truth computed DIRECTLY from the Bosch CSVs with pandas, independently of the ingest + SQL + tool code.

Used by the evals (expected answers) and by tests/test_crosscheck.py (tools must agree with this).
Same definitions as the tools: entry time at a station = min over its date columns; route = stations ordered by entry time
(ties by station number); dwell = next station's entry time - this station's entry time (same line only); bottleneck = largest average queue (sum of dwell / window width) among stations with >=30 complete cycles covering >=50% of visits;
shift/window = equal-width windows over [min entry, max entry] of the sampled parts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = re.compile(r"^L(\d+)_S(\d+)_D(\d+)$")


@dataclass
class Truth:
    visits: pd.DataFrame          # Id, line, station, t, next_t, dwell, shift, qc
    n_windows: int
    tmin: float
    width: float

    # ---- helpers
    def line(self, line: int) -> pd.DataFrame:
        return self.visits[self.visits["line"] == line]

    def stations(self, line: int) -> list[int]:
        return sorted(self.line(line)["station"].unique())

    def shifts(self) -> list[str]:
        return [f"W{i}" for i in range(1, self.n_windows + 1)]

    # ---- answers
    def mean_dwell(self, line: int, station: int, shift: str | None = None) -> float:
        d = self.line(line)
        d = d[(d["station"] == station) & d["dwell"].notna()]
        d = d if shift is None else d[d["shift"] == shift]
        return float(d["dwell"].mean())

    def cycles(self, line: int, station: int, shift: str | None = None) -> int:
        d = self.line(line)
        d = d[(d["station"] == station) & d["dwell"].notna()]
        return len(d if shift is None else d[d["shift"] == shift])

    def _cyc(self, line: int, station: int, shift: str | None) -> pd.DataFrame:
        d = self.line(line)
        d = d[(d["station"] == station) & d["dwell"].notna()]
        return d if shift is None else d[d["shift"] == shift]

    def avg_queue(self, line: int, station: int, shift: str) -> float:
        """Time-averaged parts in queue = sum(dwell over complete visits in the window) / window width."""
        return float(self._cyc(line, station, shift)["dwell"].sum() / self.width)

    def peak_queue(self, line: int, station: int, shift: str) -> int:
        """Max simultaneous parts between entering the station and entering their next one (leaves before enters at equal times)."""
        d = self._cyc(line, station, shift)
        ev = sorted([(t, 1) for t in d["t"]] + [(t, -1) for t in d["next_t"]])
        cur = best = 0
        for _, dl in ev:
            cur += dl
            best = max(best, cur)
        return best

    def median_ticks(self, line: int, station: int, shift: str | None = None, tick: float = 0.01) -> float:
        return float(np.median(np.round(self._cyc(line, station, shift)["dwell"] / tick)))

    def bottleneck(self, line: int, shift: str, min_cycles: int = 30) -> int | None:
        """Station with the largest average queue among stations with >=30 complete cycles covering >=50% of visits.
        Ties: higher peak queue, higher mean dwell, lower station number (same rule as the tool)."""
        best, arg = None, None
        for s in self.stations(line):
            n = self.cycles(line, s, shift)
            if n >= min_cycles and n >= 0.5 * self.visits_at(line, s, shift):
                key = (round(self.avg_queue(line, s, shift), 4), self.peak_queue(line, s, shift), round(self.mean_dwell(line, s, shift), 3), -s)
                if best is None or key > best:
                    best, arg = key, s
        return arg

    def trend_peak(self, line: int, station: int, n_buckets: int = 8) -> float:
        """Highest bucket-mean dwell when the station's complete cycles are split into equal time buckets (same rule as the tool)."""
        d = self.line(line)
        d = d[(d["station"] == station) & d["dwell"].notna()]
        t0, t1 = float(d["t"].min()), float(d["t"].max())
        w = max((t1 - t0) / n_buckets, 1e-9)
        return float(d.groupby(((d["t"] - t0) // w).astype(int))["dwell"].mean().max())

    def visits_at(self, line: int, station: int, shift: str | None = None) -> int:
        d = self.line(line)
        d = d[d["station"] == station]
        return len(d if shift is None else d[d["shift"] == shift])

    def fail_rate(self, line: int, station: int) -> tuple[int, int, float]:
        d = self.line(line)
        d = d[d["station"] == station].dropna(subset=["qc"]).drop_duplicates("Id")
        return len(d), int(d["qc"].sum()), float(d["qc"].mean())

    def baseline(self, line: int) -> float:
        d = self.line(line).dropna(subset=["qc"]).drop_duplicates("Id")
        return float(d["qc"].mean())

    def fail_ranking(self, line: int, min_parts: int = 50) -> list[tuple[int, float]]:
        out = []
        for s in self.stations(line):
            n, f, r = self.fail_rate(line, s)
            if n >= min_parts:
                out.append((s, r))
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def line_output(self, line: int, shift: str) -> int:
        d = self.line(line)
        d = d[d["shift"] == shift]
        return len(d[d["station"] == d["station"].max()])

    def route(self, part_id: int) -> list[tuple[int, int, float]]:
        d = self.visits[self.visits["Id"] == part_id].sort_values(["t", "station"])
        return [(int(r.line), int(r.station), float(r.t)) for r in d.itertuples()]


def load(date_csv: Path, numeric_csv: Path, lines: list[int], max_parts: int | None, n_windows: int = 8) -> Truth:
    cols = pd.read_csv(date_csv, nrows=0).columns
    use = ["Id"] + [c for c in cols if (m := DATE_COL.match(c)) and int(m.group(1)) in lines]
    dc = [c for c in use if c != "Id"]
    parts, kept = [], 0
    # chunked: stop as soon as max_parts qualifying rows (>=1 date on a chosen line) have been seen; per-chunk melt keeps memory small
    for chunk in pd.read_csv(date_csv, usecols=use, chunksize=20000):
        chunk = chunk.dropna(subset=dc, how="all")
        if max_parts is not None:
            chunk = chunk.iloc[: max_parts - kept]
        kept += len(chunk)
        long = chunk.melt(id_vars="Id", value_vars=dc, var_name="col", value_name="t").dropna(subset=["t"])
        g = long["col"].str.extract(DATE_COL)
        long["line"], long["station"] = g[0].astype(int), g[1].astype(int)
        parts.append(long.groupby(["Id", "line", "station"], as_index=False)["t"].min())   # entry = min over the station's columns
        if max_parts is not None and kept >= max_parts:
            break
    v = pd.concat(parts, ignore_index=True)
    v = v.sort_values(["Id", "line", "t", "station"]).reset_index(drop=True)
    v["next_t"] = v.groupby(["Id", "line"])["t"].shift(-1)
    v["dwell"] = v["next_t"] - v["t"]
    tmin, tmax = float(v["t"].min()), float(v["t"].max())
    width = (tmax - tmin) / n_windows if tmax > tmin else 1.0
    v["shift"] = "W" + (np.minimum(((v["t"] - tmin) // width).astype(int), n_windows - 1) + 1).astype(str)
    ids = set(v["Id"].unique())
    resp = pd.concat([c[c["Id"].isin(ids)] for c in pd.read_csv(numeric_csv, usecols=["Id", "Response"], chunksize=100000)]).set_index("Id")["Response"]
    v["qc"] = v["Id"].map(resp)
    return Truth(v, n_windows, tmin, width)
