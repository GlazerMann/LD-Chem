"""Additional gas species loader regressions."""

from pathlib import Path

import pytest

from ld_chem.gases import retrieve_gas_species


def test_retrieve_gas_species_accepts_path_without_trailing_separator(tmp_path: Path):
    (tmp_path / "gas_data.dat").write_text(
        "SO2 0.11 64d-3 1.4d0 2.9d3\n",
        encoding="utf-8",
    )

    species = retrieve_gas_species("SO2", specdata_path=tmp_path)

    assert species.name == "SO2"
    assert species.molar_mass == pytest.approx(0.064)


def test_retrieve_gas_species_reports_invalid_numeric_context(tmp_path: Path):
    (tmp_path / "gas_data.dat").write_text(
        "SO2 invalid 64d-3 1.4d0 2.9d3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid numeric value in gas species 'SO2'.*:1",
    ):
        retrieve_gas_species("SO2", specdata_path=tmp_path)
