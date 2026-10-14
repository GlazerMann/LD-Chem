import numpy as np
import pytest
from types import SimpleNamespace

from ld_chem.systems import (
    account_for_implicit_aqueous_water,
    check_mass_balance,
)


def test_implicit_aqueous_water_exchange_conserves_particle_mass():
    # SO2 -> HSO3 + H+ increases the represented solute mass by 17 g/mol;
    # the aqueous solvent supplies that mass without changing the rate law.
    before = np.array([[0.064, 0.0, 0.0, 0.100]])
    after = np.array([[0.0, 0.080, 0.001, 0.100]])

    balanced = account_for_implicit_aqueous_water(before, after, water_idx=3)

    assert np.isclose(balanced[0, 3], 0.083)
    check_mass_balance(
        SimpleNamespace(particles=SimpleNamespace(spec_masses=before)),
        SimpleNamespace(particles=SimpleNamespace(spec_masses=balanced)),
    )


def test_implicit_aqueous_water_exchange_rejects_negative_water():
    before = np.array([[0.0, 0.0, 0.0, 0.010]])
    after = np.array([[0.0, 0.020, 0.0, 0.010]])

    with pytest.raises(ValueError, match="more water than is available"):
        account_for_implicit_aqueous_water(before, after, water_idx=3)
