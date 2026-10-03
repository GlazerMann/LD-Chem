import pytest
import numpy as np
import warnings
import ld_chem.systems as systems
from ld_chem.systems import Processes, Feedbacks


def test_processes_creation():
    """Test Processes dataclass creation."""
    # Test default values
    processes = Processes()
    assert processes.condensation == True
    assert processes.cocondensation == False
    assert processes.aq_chemistry == False
    assert processes.gas_chemistry == False

    # Test custom values
    processes_custom = Processes(
        condensation=False,
        cocondensation=True,
        aq_chemistry=True,
        gas_chemistry=True
    )
    assert processes_custom.condensation == False
    assert processes_custom.cocondensation == True
    assert processes_custom.aq_chemistry == True
    assert processes_custom.gas_chemistry == True


def test_feedbacks_creation():
    """Test Feedbacks dataclass creation."""
    # Test default values
    feedbacks = Feedbacks()
    assert feedbacks.dwc_dt == 0.0
    assert feedbacks.dwv_dt == 0.0
    assert feedbacks.gases is None

    # Test custom values
    feedbacks_custom = Feedbacks(
        dwc_dt=1.5e-6,
        dwv_dt=-2.3e-7,
        gases=None
    )
    assert feedbacks_custom.dwc_dt == 1.5e-6
    assert feedbacks_custom.dwv_dt == -2.3e-7

class _DummyParcelState:
    def __init__(self):
        self.z = 0.0
        self.T = 298.0
        self.P = 101325.0
        self.S = 1.0
        self.w = 1.0
        self.wv = None
        self.gas = None

    def clone_detached(self):
        clone = _DummyParcelState()
        clone.z = self.z
        clone.T = self.T
        clone.P = self.P
        clone.S = self.S
        clone.w = self.w
        clone.wv = self.wv
        clone.gas = self.gas
        return clone


class _DummyOde:
    def __init__(self, rhs):
        self.rhs = rhs
        self.t = 0.0
        self.state = None

    def set_integrator(self, *args, **kwargs):
        return self

    def set_initial_value(self, state, t):
        self.state = state
        self.t = t
        return self

    def integrate(self, target):
        # Exercise the RHS once; returning the unchanged state is sufficient
        # for testing that update_air passes the correct feedback rate.
        self.rhs(self.t, self.state)
        self.t = target
        return self.state

def test_update_air_converts_condensed_mass_to_rate(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        systems.air_thermo, "S_to_wv", lambda S, T, P: 0.01)

    def fake_dstate_dt(state, velocity, condensed_water_rate):
        captured["condensed_water_rate"] = condensed_water_rate
        return np.zeros_like(state)

    monkeypatch.setattr(systems.air_thermo, "dstate_dt", fake_dstate_dt)
    monkeypatch.setattr(systems, "ode", _DummyOde)

    systems.update_air(
        t2=2.0,
        ParcelState_0=_DummyParcelState(),
        processes=Processes(condensation=True),
        feedbacks=Feedbacks(dwc_dt=4.0e-6),
        dt=2.0,
    )

    assert captured["condensed_water_rate"] == pytest.approx(2.0e-6)


def test_update_air_rejects_nonpositive_dt_for_condensation_feedback():
    with pytest.raises(ValueError, match="dt must be positive"):
        systems.update_air(
            t2=0.0,
            ParcelState_0=_DummyParcelState(),
            processes=Processes(condensation=True),
            feedbacks=Feedbacks(dwc_dt=1.0e-6),
            dt=0.0,
        )
