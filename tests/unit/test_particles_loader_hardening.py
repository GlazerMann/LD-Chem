"""Additional aerosol species loader regressions."""

from pathlib import Path

import pytest

from ld_chem.particles import retrieve_one_species


def test_retrieve_one_species_accepts_path_without_trailing_separator(tmp_path: Path):
    (tmp_path / "aero_data.dat").write_text(
        "SO4 1800 0 96d-3 0.65\n",
        encoding="utf-8",
    )

    species = retrieve_one_species("SO4", specdata_path=tmp_path)

    assert species.name == "SO4"
    assert species.molar_mass == pytest.approx(0.096)


def test_retrieve_one_species_reports_malformed_line_context(tmp_path: Path):
    (tmp_path / "aero_data.dat").write_text(
        "SO4 1800 0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Malformed aerosol species entry.*line 1.*expected 5 fields",
    ):
        retrieve_one_species("SO4", specdata_path=tmp_path)


def test_retrieve_one_species_reports_invalid_numeric_context(tmp_path: Path):
    (tmp_path / "aero_data.dat").write_text(
        "SO4 not-a-density 0 96d-3 0.65\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value in aerosol species 'SO4'.*:1",
    ):
        retrieve_one_species("SO4", specdata_path=tmp_path)
