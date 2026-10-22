"""Humidity and total-air collider slots cannot also be tracked gases."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ld_chem.scenario as scenario
from ld_chem.reactions import GasReactions
from ld_chem.systems import Feedbacks, Processes, update_air


DATA = str(Path(__file__).resolve().parents[2]/"src"/"ld_chem"/"species_data")+"/"


@pytest.mark.parametrize("reserved", ["H2O", "M"])
def test_parcel_rejects_explicit_reserved_gas_before_species_loading(monkeypatch, reserved):
    monkeypatch.setattr(scenario, "make_GasReactions", lambda **kwargs: GasReactions((), ()))
    with pytest.raises(ValueError, match="H2O or M"):
        scenario.create_parcel_scenario(gas_names=[reserved], gas_concs=[1.0],
                                        gas_chemistry=True, specdata_path=DATA)


@pytest.mark.parametrize("reserved", ["H2O", "M"])
def test_les_rejects_explicit_reserved_trajectory_gas(monkeypatch, reserved):
    monkeypatch.setattr(scenario, "make_GasReactions", lambda **kwargs: GasReactions((), ()))
    trajectory = {"t": np.array([0., 1.]), "T": np.array([298., 298.]),
                  "P": np.array([101325., 101325.]), "s": np.array([.85, .85]),
                  "gas": {reserved: np.array([1., 1.])}}
    with pytest.raises(ValueError, match="H2O or M"):
        scenario.create_les_scenario(trajectory_data=trajectory, gas_chemistry=True,
                                     specdata_path=DATA)


class State(SimpleNamespace):
    def clone_detached(self):
        return deepcopy(self)


@pytest.mark.parametrize("names", [("H2O",), ("M",), ("A", "A")])
def test_direct_solver_rejects_reserved_or_duplicate_slots(names):
    state = State(T=298., P=101325., S=.85, z=0., w=0.,
                  gas=SimpleNamespace(gases=[SimpleNamespace(name=n) for n in names],
                                      concs=np.ones(len(names))))
    with pytest.raises(ValueError, match="H2O or M|unique"):
        update_air(1., state, Processes(condensation=False, gas_chemistry=True),
                   Feedbacks(), 1., gas_reactions=GasReactions((), ()))
