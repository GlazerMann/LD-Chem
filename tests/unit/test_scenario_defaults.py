"""Regression coverage for scenario constructor defaults."""

import warnings
from pathlib import Path

from ld_chem.scenario import create_parcel_scenario


def test_create_parcel_scenario_default_species_are_supported_and_2d():
    root = Path(__file__).parent.parent.parent
    species_data = root / "src" / "ld_chem" / "species_data"
    mechanisms = root / "src" / "ld_chem" / "mechanisms"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        state, aq_reactions, gas_reactions = create_parcel_scenario(
            z_end=1.0,
            specdata_path=str(species_data) + "/",
            mechanism_data_path=str(mechanisms) + "/",
            aq_chemistry=None,
            gas_chemistry=False,
        )

    names = [species.name for species in state.particles.species]
    assert names[:2] == ["SO4", "H2O"]
    assert state.particles.spec_masses.ndim == 2
    assert state.particles.spec_masses.shape[0] == 1
    assert aq_reactions is None
    assert gas_reactions is None
