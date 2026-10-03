"""Regression tests for condensation feedback timestep scaling."""

from __future__ import annotations

import numpy as np
import pytest

import ld_chem.constants as constants
import ld_chem.processes.air_thermo as air_thermo
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


def test_condensation_rate_matches_independent_mass_and_energy_tendencies():
    """dstate_dt must interpret its condensation input as kg m^-3 s^-1."""
    temperature = 298.0
    pressure = 101325.0
    saturation = 1.0
    state = np.array(
        [
            0.0,
            temperature,
            pressure,
            saturation,
            air_thermo.S_to_wv(saturation, temperature, pressure),
        ]
    )
    condensed_water_rate = 2.0e-6

    derivative = air_thermo.dstate_dt(
        state,
        0.0,
        condensed_water_rate,
    )

    _, rho_air, _ = air_thermo.compute_thermo_props(
        temperature,
        pressure,
        saturation,
    )
    expected_mixing_ratio_rate = condensed_water_rate / rho_air

    # With zero updraft, the independently derived vapor and latent-heating
    # tendencies contain only the supplied condensation rate.
    assert derivative[0] == pytest.approx(0.0)
    assert derivative[2] == pytest.approx(0.0)
    assert derivative[4] == pytest.approx(-expected_mixing_ratio_rate)
    assert derivative[1] == pytest.approx(
        constants.L * expected_mixing_ratio_rate / constants.Cp
    )

