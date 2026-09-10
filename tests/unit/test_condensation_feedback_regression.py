@ -0,0 +1,63 @@
"""Regression tests for condensation feedback timestep scaling."""

from __future__ import annotations

import numpy as np

import ld_chem.systems as systems
from ld_chem.systems import Feedbacks, Processes


class _ParcelState:
    def __init__(self):
        self.z = 0.0
        self.T = 298.0
        self.P = 101325.0
        self.S = 1.0
        # Keep the updraft branch active while making its thermodynamic
        # contribution negligible relative to condensation.
        self.w = 1.0e-12
        self.wv = None
        self.gas = None

    def clone_detached(self):
        clone = _ParcelState()
        clone.z = self.z
        clone.T = self.T
        clone.P = self.P
        clone.S = self.S
        clone.w = self.w
        clone.wv = self.wv
        clone.gas = self.gas
        return clone


def _integrate_feedback(dt: float, condensed_water_change: float):
    return systems.update_air(
        t2=dt,
        ParcelState_0=_ParcelState(),
        processes=Processes(condensation=True),
        feedbacks=Feedbacks(dwc_dt=condensed_water_change),
        dt=dt,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_equivalent_condensed_mass_change_is_timestep_invariant():
    """The air response should depend on mass change, not arbitrary dt."""
    condensed_water_change = 1.0e-5  # kg/m^3 over the completed timestep

    one_second = _integrate_feedback(1.0, condensed_water_change)
    quarter_second = _integrate_feedback(0.25, condensed_water_change)

    # Condensation must produce a nontrivial response.
    assert one_second.T > 298.0
    assert one_second.S < 1.0

    np.testing.assert_allclose(
        [one_second.T, one_second.S, one_second.wv],
        [quarter_second.T, quarter_second.S, quarter_second.wv],
        rtol=1.0e-9,
        atol=1.0e-11,
    )
