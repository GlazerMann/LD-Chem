"""Reject nonphysical or singular vapor-pressure inputs in S_to_wv."""
import numpy as np
import pytest

from ld_chem.processes.air_thermo import S_to_wv


@pytest.mark.parametrize('S,T,P', [
    (1.0, 373.15, 101325.0),  # Saturation pressure exceeds ambient pressure.
    (0.8, 298.15, 0.0),
    (-0.1, 298.15, 101325.0),
    (0.8, -1.0, 101325.0),
    (np.nan, 298.15, 101325.0),
    (0.8, np.inf, 101325.0),
])
def test_invalid_vapor_pressure_domain_raises(S, T, P):
    with pytest.raises(ValueError):
        S_to_wv(S, T, P)


def test_regular_conditions_remain_finite():
    assert np.isfinite(S_to_wv(0.85, 298.15, 101325.0))
