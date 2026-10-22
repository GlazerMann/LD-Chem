"""Conservation checks must reject invalid states instead of accepting NaN."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from ld_chem.utilities import check_gas_chemistry


def state():
    return SimpleNamespace(T=298.15, P=101325.0, S=0.85,
        gas=SimpleNamespace(
            gases=[SimpleNamespace(name="A", molar_mass=0.028),
                   SimpleNamespace(name="N2", molar_mass=0.028)],
            concs=np.array([10.0, 7.8e8])))


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["concentration", "S", "T", "P", "molar_mass"])
def test_nonfinite_inputs_are_rejected(which, bad, field):
    states = [state(), state()]
    target = states[which]
    if field == "concentration":
        target.gas.concs[0] = bad
    elif field == "molar_mass":
        target.gas.gases[0].molar_mass = bad
    else:
        setattr(target, field, bad)
    with pytest.raises(ValueError, match="finite"):
        check_gas_chemistry(*states)


@pytest.mark.parametrize("bad", [0.0, -1.0])
@pytest.mark.parametrize("field", ["T", "P", "molar_mass"])
def test_nonpositive_physical_reference_values_are_rejected(field, bad):
    before, after = state(), state()
    if field == "molar_mass":
        before.gas.gases[0].molar_mass = bad
    else:
        setattr(before, field, bad)
    with pytest.raises(ValueError, match="positive"):
        check_gas_chemistry(before, after)


@pytest.mark.parametrize("concs", [[10.0], [10.0, 7.8e8, 1.0], [[10.0, 7.8e8]],
                                    ["10", "780000000"], [1+0j, 7.8e8]])
def test_species_and_concentration_schema_cannot_be_truncated(concs):
    before, after = state(), state()
    after.gas.concs = concs
    with pytest.raises(ValueError, match="vector matching species"):
        check_gas_chemistry(before, after)


def test_reordered_or_changed_species_are_rejected():
    before, after = state(), state()
    after.gas.gases.reverse()
    after.gas.concs = after.gas.concs[::-1]
    with pytest.raises(ValueError, match="order"):
        check_gas_chemistry(before, after)
    after = deepcopy(before)
    after.gas.gases[0].molar_mass = 0.029
    with pytest.raises(ValueError, match="molar masses"):
        check_gas_chemistry(before, after)


@pytest.mark.parametrize("names", [("A", "A"), ("H2O", "N2"), ("M", "N2")])
def test_ambiguous_or_reserved_species_are_rejected(names):
    before, after = state(), state()
    for gas, name in zip(after.gas.gases, names):
        gas.name = name
    with pytest.raises(ValueError, match="unique|H2O or M"):
        check_gas_chemistry(before, after)


def test_finite_inputs_cannot_overflow_into_a_successful_check():
    before, after = state(), state()
    before.gas.concs[0] = 1e308
    before.gas.gases[0].molar_mass = 1e308
    after = deepcopy(before)
    with pytest.raises(ValueError, match="finite"):
        check_gas_chemistry(before, after)


@pytest.mark.parametrize("saturation", [-0.1, 1e8])
def test_invalid_water_vapor_domain_is_rejected(saturation):
    before, after = state(), state()
    after.S = saturation
    with pytest.raises(ValueError, match="saturation ratio|vapor pressure"):
        check_gas_chemistry(before, after)


def test_unchanged_large_background_cannot_erase_trace_loss():
    before, after = state(), state()
    before.gas.concs[0] = 0.01
    after.gas.concs[0] = 0.0
    before.gas.concs[1] = after.gas.concs[1] = 1e30
    with pytest.raises(ValueError, match="Mass not conserved"):
        check_gas_chemistry(before, after)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_finite_numpy_thermodynamic_scalars_keep_an_unchanged_budget(dtype):
    before = state()
    before.T, before.P, before.S = dtype(before.T), dtype(before.P), dtype(before.S)
    check_gas_chemistry(before, deepcopy(before))
