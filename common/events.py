"""The ONE event schema every source writes and every MCP tool reads.

    events(source, unit_id, visit_id, line, station, event_type, ts, ts_unit, shift, qc_failed)

* source      'bosch' | 'vision' | 'synthetic'      (provenance, always reported by the tools)
* unit_id     the part/unit, or NULL when identity is unknown (the camera adapter does not track identity)
* visit_id    links the enter and leave events of ONE unit at ONE station (this is how cycles are paired)
* line/station text ids, e.g. 'L3'/'S29' for Bosch, 'CAM1'/'S2' for the camera
* event_type  'enter' | 'leave' | 'pass'   ('pass' = point observation with no dwell; reserved, no ingest emits it yet)
* ts, ts_unit time in the source's NATIVE unit: 'time_units' (Bosch, anonymised, NOT clock time) or 's'
* shift       a fixed time window the tools treat as a shift: 'W1'..'Wn' (Bosch), 'CLIP' (vision)
* qc_failed   1 = the unit ended up failing QC (Bosch Response=1), 0 = passed, NULL = unknown
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "events.db"
SOURCES = ("bosch", "vision", "synthetic")
EVENT_TYPES = ("enter", "leave", "pass")
TS_UNITS = ("time_units", "s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL CHECK (source IN ('bosch','vision','synthetic')),
    unit_id    TEXT,
    visit_id   TEXT NOT NULL,
    line       TEXT NOT NULL,
    station    TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('enter','leave','pass')),
    ts         REAL NOT NULL,
    ts_unit    TEXT NOT NULL CHECK (ts_unit IN ('time_units','s')),
    shift      TEXT NOT NULL,
    qc_failed  INTEGER CHECK (qc_failed IN (0,1))
);
CREATE INDEX IF NOT EXISTS ix_ev_main  ON events(source, line, station, shift, event_type);
CREATE INDEX IF NOT EXISTS ix_ev_visit ON events(source, visit_id);
CREATE INDEX IF NOT EXISTS ix_ev_unit  ON events(source, unit_id);
CREATE TABLE IF NOT EXISTS station_labels (
    source TEXT NOT NULL, line TEXT NOT NULL, station TEXT NOT NULL, label TEXT NOT NULL,
    PRIMARY KEY (source, line, station)
);
CREATE TABLE IF NOT EXISTS catalog_agg (   -- tiny summary of `events`, rebuilt by refresh_catalog() at the end of every ingest
    source TEXT, line TEXT, station TEXT, shift TEXT, ts_unit TEXT, event_type TEXT, n INTEGER
);
CREATE TABLE IF NOT EXISTS meta (          -- per-source ingest facts (window width, sample size, files...)
    source TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (source, key)
);
"""

COLUMNS = ["source", "unit_id", "visit_id", "line", "station", "event_type", "ts", "ts_unit", "shift", "qc_failed"]


def db_path() -> Path:
    return Path(os.environ.get("LINE_DB", DEFAULT_DB))


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def refresh_catalog(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM catalog_agg")
    con.execute("INSERT INTO catalog_agg SELECT source, line, station, shift, ts_unit, event_type, COUNT(*) FROM events "
                "GROUP BY source, line, station, shift, ts_unit, event_type")
    con.commit()


def ensure_catalog(con: sqlite3.Connection) -> None:
    """Self-heal: if events exist but the summary is empty (e.g. rows inserted by hand), rebuild it once."""
    if con.execute("SELECT COUNT(*) FROM catalog_agg").fetchone()[0] == 0 and \
       con.execute("SELECT 1 FROM events LIMIT 1").fetchone():
        refresh_catalog(con)


def clear_source(con: sqlite3.Connection, source: str) -> None:
    for t in ("events", "station_labels", "meta"):
        con.execute(f"DELETE FROM {t} WHERE source=?", (source,))


def insert_events(con: sqlite3.Connection, rows) -> int:
    """rows: iterable of tuples in COLUMNS order."""
    con.executemany(f"INSERT INTO events({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})", rows)
    return con.execute("SELECT changes()").fetchone()[0]


def set_meta(con: sqlite3.Connection, source: str, **kv) -> None:
    for k, v in kv.items():
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?)", (source, k, str(v)))


def set_label(con: sqlite3.Connection, source: str, line: str, station: str, label: str) -> None:
    con.execute("INSERT OR REPLACE INTO station_labels VALUES (?,?,?,?)", (source, line, station, label))
