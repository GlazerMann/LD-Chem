import pytest
import numpy as np
from pathlib import Path
from ld_chem.particles import AerosolSpecies, retrieve_one_species


def test_retrieve_one_species_unknown_species(tmp_path):
    """Unknown aerosol species should raise a clear ValueError."""
    (tmp_path / "aero_data.dat").write_text(
        "\n"
        "SO4 1800 2 0.096 0.65\n"
        "OC 1200 1 0.200 0.10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown aerosol species 'NOPE'"):
        retrieve_one_species(
            "NOPE",
            specdata_path=str(tmp_path) + "/",
        )


def test_retrieve_one_species_handles_blank_lines(tmp_path):
    """Blank lines in aero_data.dat should be ignored safely."""
    (tmp_path / "aero_data.dat").write_text(
        "\n"
        "\n"
        "SO4 1800 2 0.096 0.65\n"
        "\n",
        encoding="utf-8",
    )

    species = retrieve_one_species(
        "SO4",
        specdata_path=str(tmp_path) + "/",
    )

    assert species.name == "SO4"
    assert species.density == pytest.approx(1800.0)
    assert species.molar_mass == pytest.approx(0.096)
    assert species.kappa == pytest.approx(0.65)


def test_retrieve_one_species_rejects_malformed_matching_row(tmp_path):
    """A malformed matching species row should fail immediately."""
    (tmp_path / "aero_data.dat").write_text(
        "SO4 1800 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        retrieve_one_species(
            "SO4",
            specdata_path=str(tmp_path) + "/",
        )

def test_aerosol_species_creation():
    """Test AerosolSpecies dataclass creation."""
    species = AerosolSpecies(
        name="SO4",
        density=1800.0,
        kappa=0.65,
        molar_mass=0.096,
        surface_tension=0.072
    )

    assert species.name == "SO4"
    assert species.density == 1800.0
    assert species.kappa == 0.65
    assert species.molar_mass == 0.096
    assert species.surface_tension == 0.072


def test_retrieve_one_species():
    """Test aerosol species data retrieval from file."""
    # Path to species_data directory from test file location
    # tests/objects/test_particles.py -> ../../src/ld_chem/species_data/
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    # Test with a known species from aero_data.dat
    species = retrieve_one_species("SO4", specdata_path=str(species_data_path) + "/")

    assert species.name == "SO4"
    assert isinstance(species.density, float)
    assert isinstance(species.kappa, float)
    assert isinstance(species.molar_mass, float)
    assert species.surface_tension == 0.072  # default value

    # Test with another species
    species_oc = retrieve_one_species("OC", specdata_path=str(species_data_path) + "/")
    assert species_oc.name == "OC"
    assert species_oc.density == 1200.0  # from aero_data.dat
    assert species_oc.kappa == 0.1


def test_retrieve_one_species_custom_surface_tension():
    """Test aerosol species retrieval with custom surface tension."""
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"
    custom_surface_tension = 0.08
    species = retrieve_one_species(
        "SO4",
        specdata_path=str(species_data_path) + "/",
        surface_tension=custom_surface_tension
    )
    assert species.name == "SO4"
    assert species.surface_tension == custom_surface_tension
