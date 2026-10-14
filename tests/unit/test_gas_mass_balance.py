from types import SimpleNamespace

import pytest

from ld_chem.processes.air_thermo import es
from ld_chem.utilities import check_gas_chemistry


def _state(gas_ppb, saturation_ratio, T=298.15, P=101325.0):
    names_and_molar_masses = (
        ("HO2", 0.033), ("OH", 0.017), ("O2", 0.032),
        ("N2", 0.028),
    )
    gases = [SimpleNamespace(name=name, molar_mass=molar_mass)
             for name, molar_mass in names_and_molar_masses]
    return SimpleNamespace(
        gas=SimpleNamespace(gases=gases, concs=gas_ppb),
        S=saturation_ratio, T=T, P=P,
    )


def test_gas_mass_check_includes_water_produced_by_reaction():
    water_ppb = 10.0
    saturation_ratio = water_ppb * 1e-9 * 101325.0 / es(25.0)
    before = _state([10.0, 10.0, 0.0], 0.0)
    after = _state([0.0, 0.0, 10.0], saturation_ratio)

    # HO2 + OH -> O2 + H2O conserves the tracked molecular mass.
    check_gas_chemistry(before, after)


def test_gas_mass_check_still_rejects_unbalanced_change():
    before = _state([10.0, 10.0, 0.0], 0.0)
    after = _state([0.0, 0.0, 10.0], 0.0)

    with pytest.raises(ValueError, match="Mass not conserved"):
        check_gas_chemistry(before, after)


def test_gas_mass_check_uses_same_thermodynamic_reference_for_both_states():
    gases = [10.0, 10.0, 0.0, 7.8e8]
    before = _state(gases, 0.85)
    after = _state(gases, 0.85, T=302.0, P=99000.0)

    check_gas_chemistry(before, after)


def test_bulk_water_and_background_gases_do_not_hide_trace_mass_loss():
    before = _state([10.0, 0.0, 0.0, 7.8e8], 0.85)
    after = _state([0.0, 0.0, 0.0, 7.8e8], 0.85)

    with pytest.raises(ValueError, match="Mass not conserved"):
        check_gas_chemistry(before, after)
