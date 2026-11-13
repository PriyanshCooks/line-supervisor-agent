import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = Path(__file__).parent / "fixtures" / "bosch_mini"


@pytest.fixture(scope="session")
def _mini_built(tmp_path_factory):
    from ingest import bosch
    db = tmp_path_factory.mktemp("db") / "events.db"
    stats = bosch.ingest(FIXTURE / "train_date.csv", FIXTURE / "train_numeric.csv", db, lines=[0, 3],
                         max_parts=None, n_windows=2, source="bosch")
    return db, stats


@pytest.fixture()
def mini_db(_mini_built, monkeypatch):
    """Hand-written Bosch-format fixture ingested as source='bosch' into a temp DB (2 windows). Sets LINE_DB per test."""
    monkeypatch.setenv("LINE_DB", str(_mini_built[0]))
    return _mini_built


@pytest.fixture()
def vision_db(tmp_path, monkeypatch):
    """A DB holding only camera-style events: S2 has a leave with no enter, S3 has an enter with no leave."""
    from common import events as E
    db = tmp_path / "events.db"
    con = E.connect(db)
    rows = [("vision", None, "vision:S2:1", "CAM1", "S2", "leave", 16.04, "s", "CLIP", None),
            ("vision", None, "vision:S3:1", "CAM1", "S3", "enter", 15.36, "s", "CLIP", None)]
    E.insert_events(con, rows)
    E.set_label(con, "vision", "CAM1", "S2", "Fixture 2 (worker)")
    con.commit()
    con.close()
    monkeypatch.setenv("LINE_DB", str(db))
    return db
