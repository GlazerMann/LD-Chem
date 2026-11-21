import numpy as np
import pytest

import ld_chem.run as run_module
from ld_chem.run import _make_time_grid, _parcel_end_time


def test_time_grid_reaches_end_with_short_final_interval():
    times = _make_time_grid(0.0, 2.5, 1.0)

    np.testing.assert_allclose(times, [0.0, 1.0, 2.0, 2.5])
    np.testing.assert_allclose(np.diff(times), [1.0, 1.0, 0.5])


def test_time_grid_handles_duration_shorter_than_dt():
    np.testing.assert_allclose(_make_time_grid(4.0, 4.25, 1.0), [4.0, 4.25])


def test_time_grid_handles_zero_duration():
    np.testing.assert_allclose(_make_time_grid(4.0, 4.0, 1.0), [4.0])


@pytest.mark.parametrize("dt", [0.0, -1.0, np.inf, np.nan])
def test_time_grid_rejects_nonpositive_or_nonfinite_dt(dt):
    with pytest.raises(ValueError, match="dt must be a finite positive number"):
        _make_time_grid(0.0, 1.0, dt)


def test_time_grid_rejects_end_before_start():
    with pytest.raises(ValueError, match="end time must not precede start time"):
        _make_time_grid(2.0, 1.0, 1.0)

@pytest.mark.parametrize(
    ("t_start", "t_end"),
    [(np.nan, 1.0), (0.0, np.nan), (np.inf, 1.0), (0.0, -np.inf)],
)
def test_time_grid_rejects_nonfinite_bounds(t_start, t_end):
    with pytest.raises(ValueError, match="start and end times must be finite"):
        _make_time_grid(t_start, t_end, 1.0)


def test_time_grid_rejects_nonfinite_duration():
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="duration must be finite"):
            _make_time_grid(-1e308, 1e308, 1.0)


def test_parcel_end_time_uses_height_difference_and_updraft_velocity():
    assert _parcel_end_time(10.0, 20.0, 2.0) == 5.0


@pytest.mark.parametrize("updraft_velocity", [0.0, np.inf, -np.inf, np.nan])
def test_parcel_end_time_rejects_zero_or_nonfinite_updraft(updraft_velocity):
    with pytest.raises(ValueError, match="updraft_velocity"):
        _parcel_end_time(0.0, 1000.0, updraft_velocity)

def test_simulate_parcel_rejects_zero_updraft_before_scenario_setup(monkeypatch):
    def unexpected_scenario_creation(**kwargs):
        pytest.fail("scenario creation should not run for zero updraft")

    monkeypatch.setattr(
        run_module, "create_parcel_scenario", unexpected_scenario_creation
    )
    with pytest.raises(ValueError, match="updraft_velocity must be nonzero"):
        run_module.simulate_parcel([], [], [], [], updraft_velocity=0.0)


@pytest.mark.parametrize(
    ("t_start", "t_end", "dt"),
    [
        (0.0, 0.1 + 0.2, 0.1),
        (0.0, 3 * 0.1, 0.1),
        (0.1, 0.1 + 0.1 + 0.1, 0.1),
        (1.0e16, 1.0e16 + 4.0, 0.5),
    ],
)
def test_positive_duration_has_strictly_increasing_times(t_start, t_end, dt):
    times = _make_time_grid(t_start, t_end, dt)

    assert times[0] == t_start
    assert times[-1] == t_end
    assert np.all(np.diff(times) > 0.0)
    assert np.count_nonzero(times == t_end) == 1
