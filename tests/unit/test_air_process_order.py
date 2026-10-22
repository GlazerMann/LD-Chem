"""Gas chemistry must preserve the result of preceding air processes."""
from copy import deepcopy

import numpy as np
import pytest

from ld_chem.gases import GasSpecies, TraceGasPopulation
from ld_chem.processes.air_thermo import S_to_wv
from ld_chem.processes.cocondensation import GasFeedback
from ld_chem.reactions import GasReaction, GasReactions
from ld_chem.systems import Feedbacks, Processes, update_air


class State:
    def __init__(self):
        self.z, self.T, self.P, self.S, self.w = 0.0, 298.15, 101325.0, 0.85, 1e-12
        self.wv = S_to_wv(self.S, self.T, self.P)
        names = ("N2", "O2", "A", "B", "HO2", "OH")
        masses = (0.028, 0.032, 0.028, 0.028, 0.033, 0.017)
        self.gas = TraceGasPopulation(
            gases=tuple(GasSpecies(name, 0.0, mass, 0.0, 0.0)
                        for name, mass in zip(names, masses)),
            concs=np.array([7.8e8, 2.1e8, 1000.0, 0.0, 10.0, 10.0]),
            ids=tuple(range(len(names))),
        )

    def clone_detached(self):
        return deepcopy(self)


def advance(state, processes, feedback, reactions=(), dt=0.5):
    return update_air(dt, state, processes, feedback, dt,
                      gas_reactions=GasReactions(tuple(reactions), tuple(range(len(reactions)))),
                      rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("dt", [0.5, 1.0])
def test_identity_gas_chemistry_preserves_condensation_and_heating(dt):
    state = State()
    feedback = Feedbacks(dwc_dt=1e-5, dwv_dt=-1e-5)
    air = advance(state, Processes(condensation=True), feedback, dt=dt)
    combined = advance(state, Processes(condensation=True, gas_chemistry=True),
                       feedback, dt=dt)
    assert air.S < state.S
    assert air.T > state.T
    np.testing.assert_allclose([combined.T, combined.P, combined.S, combined.wv],
                               [air.T, air.P, air.S, air.wv], rtol=1e-12, atol=0.0)


def test_identity_gas_chemistry_preserves_cocondensation_feedback():
    state = State()
    original = state.gas.concs.copy()
    feedback = Feedbacks(gases=GasFeedback(("A",), np.array([-100.0])))
    result = advance(state, Processes(condensation=False, cocondensation=True,
                                     gas_chemistry=True), feedback)
    assert result.gas.concs[2] == pytest.approx(900.0)
    np.testing.assert_array_equal(state.gas.concs, original)


def test_nonempty_balanced_chemistry_uses_post_feedback_state():
    state = State()
    dt, rate = 0.5, 0.2
    feedback = Feedbacks(dwc_dt=1e-5, dwv_dt=-1e-5,
                        gases=GasFeedback(("A",), np.array([-100.0])))
    reaction = GasReaction(["A"], ["B"], rate, 0.0, 0.0, "exp")
    air = advance(state, Processes(condensation=True), feedback, dt=dt)
    result = advance(state, Processes(condensation=True, cocondensation=True,
                                     gas_chemistry=True), feedback, [reaction], dt=dt)
    expected_a = 900.0*np.exp(-rate*dt)
    assert result.gas.concs[2] == pytest.approx(expected_a, rel=1e-6)
    assert result.gas.concs[3] == pytest.approx(900.0-expected_a, rel=1e-6)
    assert result.S == pytest.approx(air.S, rel=1e-12)
    assert result.T == pytest.approx(air.T, rel=1e-12)


def test_chemistry_water_production_updates_vapor_mixing_ratio():
    state = State()
    feedback = Feedbacks(dwc_dt=1e-5, dwv_dt=-1e-5)
    reaction = GasReaction(["HO2", "OH"], ["O2", "H2O"],
                           2.89e7, 0.0, 0.0, "exp")
    air = advance(state, Processes(condensation=True), feedback)
    result = advance(state, Processes(condensation=True, gas_chemistry=True),
                     feedback, [reaction])
    assert result.S > air.S
    assert result.wv == pytest.approx(S_to_wv(result.S, result.T, result.P), rel=1e-12)
