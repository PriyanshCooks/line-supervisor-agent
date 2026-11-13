"""The ingest + SQL + tool path must agree with an independent pandas computation straight from the CSVs."""
import pytest

from evals import ground_truth as GT
from tests.conftest import FIXTURE


@pytest.fixture()
def truth():
    return GT.load(FIXTURE / "train_date.csv", FIXTURE / "train_numeric.csv", [0, 3], None, n_windows=2)


def test_pandas_ground_truth_matches_hand_values(truth):
    assert truth.fail_rate(0, 0) == (4, 3, 0.75)
    assert truth.mean_dwell(0, 1) == 3.5 and truth.cycles(0, 1) == 4
    assert truth.route(1) == [(0, 0, 10.0), (0, 1, 12.0), (0, 2, 20.0)]


def test_tools_agree_with_ground_truth(mini_db, truth):
    from mcp_server import queries as q
    for line in (0, 3):
        for st in truth.stations(line):
            for shift in truth.shifts():
                m = q.station_metrics(f"S{st}", shift, line=f"L{line}")
                assert m.visits == truth.visits_at(line, st, shift)
                assert m.complete_cycles == truth.cycles(line, st, shift)
                if m.complete_cycles:
                    assert m.avg_dwell == pytest.approx(truth.mean_dwell(line, st, shift), abs=1e-3)
                    assert m.avg_queue == pytest.approx(truth.avg_queue(line, st, shift), abs=1e-4)
                    assert m.peak_queue == truth.peak_queue(line, st, shift)
                    assert m.median_dwell_ticks == pytest.approx(truth.median_ticks(line, st, shift), abs=0.05)
                assert m.throughput == truth.visits_at(line, st, shift)
            n, f, r = truth.fail_rate(line, st)
            fr = q.station_failure_rate(f"L{line}", f"S{st}")
            assert (fr.parts, fr.failed) == (n, f)


def test_bottleneck_agrees_with_ground_truth(mini_db, truth):
    from mcp_server import queries as q
    for shift in truth.shifts():
        b = q.find_bottleneck(shift, "L0", min_cycles=2)
        assert b.station == f"S{truth.bottleneck(0, shift, min_cycles=2)}"
