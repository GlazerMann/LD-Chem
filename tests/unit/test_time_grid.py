import numpy as np
import pytest

from ld_chem.run import _make_time_grid


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
