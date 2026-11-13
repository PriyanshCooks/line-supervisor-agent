"""Bosch ingest on a hand-computed fixture (tests/fixtures/bosch_mini). Expected numbers are worked out by hand."""
import sqlite3

import pytest


def rows(db, sql, *a):
    con = sqlite3.connect(str(db))
    try:
        return con.execute(sql, a).fetchall()
    finally:
        con.close()


def test_counts(mini_db):
    db, stats = mini_db
    assert stats["parts"] == 7           # part 7 has no dates at all and is excluded
    # visits per part: p1=3 p2=3 p3=2 p4=2 p5=2 p6=2 p8=1 -> 15 enters; leaves (visits with a next station): 2+2+1+1+1+1+0 = 8
    assert stats["enter"] == 15
    assert stats["leave"] == 8
    assert rows(db, "SELECT COUNT(*) FROM events")[0][0] == 23


def test_route_ordering_and_station_entry_is_min_of_date_columns(mini_db):
    db, _ = mini_db
    route = rows(db, "SELECT station, ts FROM events WHERE unit_id='1' AND line='L0' AND event_type='enter' ORDER BY ts")
    assert route == [("S0", 10.0), ("S1", 12.0), ("S2", 20.0)]       # S0 entry = min(10.0, 10.5)
    # part 3 skips S1; part 3's S0 date comes from D3 only (D1 is empty)
    route3 = rows(db, "SELECT station, ts FROM events WHERE unit_id='3' AND event_type='enter' ORDER BY ts")
    assert route3 == [("S0", 30.0), ("S2", 35.0)]


def test_tie_is_broken_by_station_number(mini_db):
    db, _ = mini_db
    # part 6 enters S1 and S2 at the same time 70.0 -> S1 first, so S1 'leaves' at 70.0 (dwell 0), S2 has no leave
    leaves = rows(db, "SELECT station, ts FROM events WHERE unit_id='6' AND event_type='leave'")
    assert leaves == [("S1", 70.0)]


def test_leave_is_next_station_entry_and_last_station_has_no_leave(mini_db):
    db, _ = mini_db
    d = rows(db, """SELECT e.station, l.ts - e.ts FROM events e JOIN events l ON l.visit_id=e.visit_id
                    WHERE e.event_type='enter' AND l.event_type='leave' AND e.unit_id='1' ORDER BY e.ts""")
    assert d == [("S0", 2.0), ("S1", 8.0)]
    assert rows(db, "SELECT COUNT(*) FROM events WHERE unit_id='8' AND event_type='leave'")[0][0] == 0   # single visit
    assert rows(db, "SELECT COUNT(*) FROM events WHERE unit_id='1' AND station='S2' AND event_type='leave'")[0][0] == 0


def test_lines_are_independent_routes(mini_db):
    db, _ = mini_db
    got = set(rows(db, "SELECT line, station, event_type, ts FROM events WHERE unit_id='5'"))
    assert got == {("L3", "S29", "enter", 60.0), ("L3", "S29", "leave", 65.0), ("L3", "S30", "enter", 65.0)}


def test_windows(mini_db):
    db, _ = mini_db
    # tmin=10, tmax=80, 2 windows -> width 35: W1 = [10,45), W2 = [45,80]
    w = dict(rows(db, "SELECT unit_id, MIN(shift) FROM events WHERE station='S0' AND line='L0' AND event_type='enter' GROUP BY unit_id"))
    assert w == {"1": "W1", "2": "W1", "3": "W1", "8": "W2"}
    # enter and leave of one visit always share a window
    assert rows(db, """SELECT COUNT(*) FROM events e JOIN events l ON l.visit_id=e.visit_id
                       WHERE e.event_type='enter' AND l.event_type='leave' AND e.shift<>l.shift""")[0][0] == 0


def test_qc_label_attached_to_every_event_of_the_part(mini_db):
    db, _ = mini_db
    assert rows(db, "SELECT DISTINCT qc_failed FROM events WHERE unit_id='2'") == [(1,)]
    assert rows(db, "SELECT DISTINCT qc_failed FROM events WHERE unit_id='1'") == [(0,)]


def test_time_units_are_never_relabelled(mini_db):
    db, _ = mini_db
    assert rows(db, "SELECT DISTINCT ts_unit, source FROM events") == [("time_units", "bosch")]


def test_schema_rejects_bad_values(mini_db):
    db, _ = mini_db
    con = sqlite3.connect(str(db))
    for bad in ("('oops','1','v','L0','S0','enter',1.0,'time_units','W1',0)",
                "('bosch','1','v','L0','S0','teleport',1.0,'time_units','W1',0)",
                "('bosch','1','v','L0','S0','enter',1.0,'hours','W1',0)",
                "('bosch','1','v','L0','S0','enter',1.0,'time_units','W1',7)"):
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO events(source,unit_id,visit_id,line,station,event_type,ts,ts_unit,shift,qc_failed) VALUES " + bad)
    con.close()
