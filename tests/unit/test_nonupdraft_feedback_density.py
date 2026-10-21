"""The zero-updraft condensation path must use mass-specific heating."""
from copy import copy

import numpy as np
import pytest

from ld_chem import constants
from ld_chem.processes.air_thermo import compute_thermo_props
from ld_chem.systems import Feedbacks, Processes, update_air


class StationaryState:
    z = 0.0
    T = 298.15
    P = 101325.0
    S = 0.85
    w = 0.0
    gas = None
    wv = None

    def clone_detached(self):
        return copy(self)


@pytest.mark.parametrize('water_change', [-1.0e-5, 1.0e-5])
def test_no_updraft_uses_air_density_for_latent_heating(water_change):
    before = StationaryState()
    _, rho_air, _ = compute_thermo_props(before.T, before.P, before.S)
    after = update_air(
        t2=0.5,
        ParcelState_0=before,
        processes=Processes(condensation=True),
        feedbacks=Feedbacks(dwv_dt=water_change),
        dt=0.5,
    )
    expected = before.T - constants.L * water_change / (constants.Cp * rho_air)
    assert after.T == pytest.approx(expected, abs=1e-10)
    assert np.isfinite(after.S)
