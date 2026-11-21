"""Exercise the numeric contract without rejecting zero-density solutes."""
import pytest

from ld_chem.particles import retrieve_one_species


def _load(tmp_path, density, mass='96D-3', kappa='6.5d-1'):
    (tmp_path / 'aero_data.dat').write_text(
        f'SO4 {density} 0 {mass} {kappa}\n', encoding='utf-8'
    )
    return retrieve_one_species('SO4', specdata_path=tmp_path)


def test_aerosol_loader_accepts_fortran_exponents_in_all_numeric_fields(tmp_path):
    value = _load(tmp_path, '1.8D3')
    assert value.density == pytest.approx(1800)
    assert value.kappa == pytest.approx(0.65)
    assert value.molar_mass == pytest.approx(0.096)


def test_aerosol_loader_preserves_zero_density_marker(tmp_path):
    assert _load(tmp_path, '0').density == 0


@pytest.mark.parametrize('density,mass,kappa', [
    ('-1', '96d-3', '0.65'),
    ('1800', '0', '0.65'),
    ('1800', '-1d-3', '0.65'),
    ('1800', '96d-3', '-0.1'),
])
def test_aerosol_loader_rejects_unphysical_values(tmp_path, density, mass, kappa):
    with pytest.raises(ValueError, match='Invalid numeric value'):
        _load(tmp_path, density, mass, kappa)
