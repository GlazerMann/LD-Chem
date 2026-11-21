"""Verify gas species range checks and Fortran exponent parsing."""
import pytest

from ld_chem.gases import retrieve_gas_species


def _load(tmp_path, alpha='1.1d-1', mass='64D-3'):
    (tmp_path / 'gas_data.dat').write_text(
        f'SO2 {alpha} {mass} 1.4d0 2.9D3\n', encoding='utf-8'
    )
    return retrieve_gas_species('SO2', specdata_path=tmp_path)


def test_gas_loader_parses_fortran_exponents_in_all_fields(tmp_path):
    gas = _load(tmp_path)
    assert gas.alpha == pytest.approx(0.11)
    assert gas.molar_mass == pytest.approx(0.064)
    assert gas.H0 == pytest.approx(1.4)
    assert gas.H_exp == pytest.approx(2900.0)


@pytest.mark.parametrize('alpha,mass', [
    ('-1d-2', '64d-3'),
    ('1.01', '64d-3'),
    ('0.1', '0'),
    ('0.1', '-64d-3'),
])
def test_gas_loader_rejects_invalid_ranges(tmp_path, alpha, mass):
    with pytest.raises(ValueError, match='Invalid numeric value'):
        _load(tmp_path, alpha, mass)
