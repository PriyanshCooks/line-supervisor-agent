"""Bosch Production Line Performance -> shared events table.

    python -m ingest.bosch --lines 0 3 --max-parts 100000          # real data in data/bosch/
    python -m ingest.bosch --data-dir data/bosch_sample --source synthetic   # format-compatible dev sample

Inputs (see docs/DATA.md): train_date.csv (Id + L{line}_S{station}_D{feature} timestamps) and
train_numeric.csv (only `Id` and `Response` are read).

Derivation, per part and per line:
  * entry time at a station = MIN over that station's date columns (the first time the part was measured there)
  * the route = stations ordered by entry time (ties broken by station number)
  * one `enter` event per station visit at its entry time
  * one `leave` event per visit that has a next station on the same line, at the NEXT station's entry time.
    So `leave - enter` is "time until the part reaches its next recorded station" (queueing + processing +
    transport). It is NOT a measured machine cycle time. The last station of a route has no `leave`.
  * qc_failed = Response (1 = failed QC), attached to every event of the part

Timestamps are Bosch's anonymised time units and stay 'time_units'; they are never converted to hours.
The timeline is split into `--windows` equal-width windows 'W1'..'Wn' (by enter time) that the tools treat as shifts.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import events as E

DATE_COL = re.compile(r"^L(\d+)_S(\d+)_D(\d+)$")
ROOT = Path(__file__).resolve().parents[1]


def station_columns(date_csv: Path, lines: list[int] | None) -> dict[tuple[int, int], list[str]]:
    cols = pd.read_csv(date_csv, nrows=0).columns
    out: dict[tuple[int, int], list[str]] = {}
    for c in cols:
        m = DATE_COL.match(c)
        if m and (lines is None or int(m.group(1)) in lines):
            out.setdefault((int(m.group(1)), int(m.group(2))), []).append(c)
    if not out:
        raise SystemExit(f"No date columns found for lines={lines} in {date_csv}")
    return out


def read_routes(date_csv: Path, lines: list[int] | None, max_parts: int | None, chunksize: int = 50_000) -> pd.DataFrame:
    """Long table: Id, line, station, t (entry time). Only the first `max_parts` parts that visit the chosen lines."""
    groups = station_columns(date_csv, lines)
    use = ["Id"] + [c for cols in groups.values() for c in cols]
    parts: list[pd.DataFrame] = []
    n = 0
    for chunk in pd.read_csv(date_csv, usecols=use, chunksize=chunksize):
        wide = pd.DataFrame({f"{ln}|{st}": chunk[cols].min(axis=1) for (ln, st), cols in groups.items()})
        wide.index = chunk["Id"].to_numpy()
        wide = wide.dropna(how="all")
        if max_parts is not None and n + len(wide) > max_parts:
            wide = wide.iloc[: max_parts - n]
        n += len(wide)
        long = wide.stack().dropna().rename("t").reset_index()
        long.columns = ["Id", "key", "t"]
        parts.append(long)
        if max_parts is not None and n >= max_parts:
            break
    long = pd.concat(parts, ignore_index=True)
    ls = long["key"].str.split("|", expand=True).astype(int)
    long["line"], long["station"] = ls[0].to_numpy(), ls[1].to_numpy()
    return long.drop(columns="key")


def build_events(long: pd.DataFrame, response: pd.Series, n_windows: int, source: str) -> tuple[pd.DataFrame, dict]:
    long = long.sort_values(["Id", "line", "t", "station"], kind="stable").reset_index(drop=True)
    long["next_t"] = long.groupby(["Id", "line"])["t"].shift(-1)
    tmin, tmax = float(long["t"].min()), float(long["t"].max())
    width = (tmax - tmin) / n_windows if tmax > tmin else 1.0
    win = np.minimum(((long["t"] - tmin) // width).astype(int), n_windows - 1) + 1
    long["shift"] = "W" + win.astype(str)
    long["qc_failed"] = pd.Series(response.reindex(long["Id"]).to_numpy(), index=long.index).astype("Int64")  # NaN -> NULL
    base = pd.DataFrame({
        "source": source,
        "unit_id": long["Id"].astype(str),
        "visit_id": long["Id"].astype(str) + ":S" + long["station"].astype(str),
        "line": "L" + long["line"].astype(str),
        "station": "S" + long["station"].astype(str),
        "ts_unit": "time_units",
        "shift": long["shift"],
        "qc_failed": long["qc_failed"],
    })
    enter = base.assign(event_type="enter", ts=long["t"])
    has_next = long["next_t"].notna()
    leave = base[has_next].assign(event_type="leave", ts=long.loc[has_next, "next_t"])
    ev = pd.concat([enter, leave], ignore_index=True)[E.COLUMNS]
    stats = {"parts": int(long["Id"].nunique()), "visits": len(long), "enter": len(enter), "leave": len(leave),
             "tmin": tmin, "tmax": tmax, "window_width": width, "n_windows": n_windows}
    return ev, stats


def ingest(date_csv: Path, numeric_csv: Path, db: Path | None = None, lines: list[int] | None = None,
           max_parts: int | None = 100_000, n_windows: int = 8, source: str = "bosch") -> dict:
    for p in (date_csv, numeric_csv):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}. See docs/DATA.md for how to download the Bosch files.")
    resp = pd.read_csv(numeric_csv, usecols=["Id", "Response"]).set_index("Id")["Response"]
    long = read_routes(Path(date_csv), lines, max_parts)
    ev, stats = build_events(long, resp, n_windows, source)
    con = E.connect(db)
    E.clear_source(con, source)
    ev.to_sql("events", con, if_exists="append", index=False, chunksize=50_000)
    for (ln, st) in ev[["line", "station"]].drop_duplicates().itertuples(index=False):
        E.set_label(con, source, ln, st, f"{ln} {st}")
    E.set_meta(con, source, ts_unit="time_units", ingest="ingest.bosch", date_csv=Path(date_csv).name,
               lines=",".join(map(str, lines)) if lines else "all", max_parts=max_parts, **stats,
               sample_note="first N parts in file order that visit the chosen lines")
    con.commit()
    E.refresh_catalog(con)
    stats["fail_rate_overall"] = float(resp.reindex(long["Id"].unique()).mean())
    stats["events"] = len(ev)
    con.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "bosch"))
    ap.add_argument("--lines", type=int, nargs="*", default=[0, 3], help="Bosch line numbers (default: 0 3)")
    ap.add_argument("--max-parts", type=int, default=100_000)
    ap.add_argument("--windows", type=int, default=8, help="number of equal time windows used as 'shifts'")
    ap.add_argument("--source", default="bosch", choices=list(E.SOURCES),
                    help="label written to events.source; use 'synthetic' for the dev sample, never for real Bosch data")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    d = Path(a.data_dir)
    s = ingest(d / "train_date.csv", d / "train_numeric.csv", Path(a.db) if a.db else None,
               a.lines or None, a.max_parts, a.windows, a.source)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, file=sys.stdout)
