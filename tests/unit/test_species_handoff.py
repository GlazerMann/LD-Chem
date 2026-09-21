"""Regression tests for the part2pop-to-LD-Chem species handoff."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from part2pop.population import build_population

import ld_chem.run as run_module
from ld_chem.particles import AerosolSpecies, retrieve_one_species
from ld_chem.scenario import (
    _prepare_initial_species,
    create_les_scenario,
    create_parcel_scenario,
)


ROOT = Path(__file__).parent.parent.parent
SPECIES_DATA = str(ROOT / "src" / "ld_chem" / "species_data") + "/"


def _prepare(*args):
    return _prepare_initial_species(*args, specdata_path=SPECIES_DATA)


def _build_source_population():
    """Build part2pop input whose OC properties differ from LD-Chem."""
    return build_population(
        {
            "type": "binned_lognormals",
            "N": [1.0e9, 2.0e9],
            "GMD": [100e-9, 200e-9],
            "GSD": [1.4, 1.6],
            "aero_spec_names": [["SO4", "OC"], ["OC"]],
            "aero_spec_fracs": [[0.2, 0.8], [1.0]],
            "N_bins": [3, 3],
            "N_sigmas": 3,
            "species_modifications": {
                "OC": {
                    "density": 1000.0,
                    "kappa": 0.001,
                    # These intentionally differ too; LD-Chem must NOT import
                    # them. Molar mass affects chemistry/conversions, while
                    # per-species surface tension is outside this handoff and is
                    # not consistently consumed by current part2pop particles.
                    "molar_mass": 0.012,
                    "surface_tension": 0.071,
                }
            },
        }
    )


def _build_readme_population():
    return build_population(
        {
            "type": "binned_lognormals",
            "N": [1.0e9],
            "GMD": [150e-9],
            "GSD": [1.6],
            "aero_spec_names": [["SO4", "OC"]],
            "aero_spec_fracs": [[0.2, 0.8]],
            "N_bins": 2,
            "N_sigmas": 3,
        }
    )


def _handoff_arrays(population):
    return (
        np.array([species.name for species in population.species]),
        np.array(population.spec_masses, copy=True),
        np.array(population.num_concs, copy=True),
        tuple(population.species),
    )


def _trajectory():
    return {
        "t": np.array([0.0, 1.0]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
        "z": np.array([100.0, 101.0]),
        "T": np.array([298.0, 298.0]),
        "P": np.array([101325.0, 101325.0]),
        "s": np.array([0.85, 0.85]),
    }


def _assert_source_dry_physics_preserved(state, population):
    np.testing.assert_allclose(
        state.particles.get_particle_var("dry_diameter"),
        population.get_particle_var("dry_diameter"),
        rtol=1e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.particles.get_particle_var("tkappa"),
        population.get_particle_var("tkappa"),
        rtol=1e-10,
        atol=0.0,
    )


def test_parcel_preserves_part2pop_density_and_kappa_only():
    population = _build_source_population()
    names, masses, num_concs, species = _handoff_arrays(population)

    state, _, _ = create_parcel_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    _assert_source_dry_physics_preserved(state, population)

    source_oc = next(spec for spec in species if spec.name == "OC")
    expected_oc = retrieve_one_species("OC", specdata_path=SPECIES_DATA)
    actual_oc = state.particles.species[state.particles.get_species_idx("OC")]
    assert isinstance(actual_oc, AerosolSpecies)
    assert actual_oc.density == pytest.approx(source_oc.density)
    assert actual_oc.kappa == pytest.approx(source_oc.kappa)
    assert actual_oc.molar_mass == pytest.approx(expected_oc.molar_mass)
    assert actual_oc.surface_tension == pytest.approx(expected_oc.surface_tension)
    assert actual_oc.molar_mass != pytest.approx(source_oc.molar_mass)
    assert actual_oc.surface_tension != pytest.approx(source_oc.surface_tension)


def test_les_preserves_part2pop_dry_diameter_and_kappa():
    population = _build_source_population()
    names, masses, num_concs, species = _handoff_arrays(population)

    state, _, _, _ = create_les_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        trajectory_data=_trajectory(),
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    _assert_source_dry_physics_preserved(state, population)


def test_readme_population_preserves_dry_physics_without_importing_out_of_scope_properties():
    population = _build_readme_population()
    names, masses, num_concs, species = _handoff_arrays(population)

    state, _, _ = create_parcel_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    _assert_source_dry_physics_preserved(state, population)
    expected_oc = retrieve_one_species("OC", specdata_path=SPECIES_DATA)
    actual_oc = state.particles.species[state.particles.get_species_idx("OC")]
    assert actual_oc.molar_mass == pytest.approx(expected_oc.molar_mass)


def test_h2o_definition_stays_ld_chem_owned_and_mass_is_reequilibrated():
    population = _build_source_population()
    names, masses, num_concs, species = _handoff_arrays(population)
    h2o_idx = int(np.where(names == "H2O")[0][0])
    masses[:, h2o_idx] = 1.0e-9
    supplied_h2o_mass = masses[:, h2o_idx].copy()

    # Even a source-side H2O definition that differs is not imported.
    source_h2o = next(spec for spec in species if spec.name == "H2O")
    source_h2o.density = 900.0
    source_h2o.kappa = 0.5
    source_h2o.molar_mass = 0.020
    source_h2o.surface_tension = 0.071

    state, _, _ = create_parcel_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    expected_h2o = retrieve_one_species("H2O", specdata_path=SPECIES_DATA)
    actual_h2o = state.particles.species[state.particles.get_species_idx("H2O")]
    assert actual_h2o == expected_h2o
    assert not np.allclose(
        state.particles.spec_masses[:, h2o_idx], supplied_h2o_mass,
        rtol=1e-6, atol=0.0,
    )


def test_les_h2o_definition_stays_ld_chem_owned_and_mass_is_reequilibrated():
    population = _build_source_population()
    names, masses, num_concs, species = _handoff_arrays(population)
    h2o_idx = int(np.where(names == "H2O")[0][0])
    masses[:, h2o_idx] = 1.0e-9
    supplied_h2o_mass = masses[:, h2o_idx].copy()

    source_h2o = next(spec for spec in species if spec.name == "H2O")
    source_h2o.density = 900.0
    source_h2o.kappa = 0.5
    source_h2o.molar_mass = 0.020
    source_h2o.surface_tension = 0.071

    state, _, _, _ = create_les_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        trajectory_data=_trajectory(),
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    expected_h2o = retrieve_one_species("H2O", specdata_path=SPECIES_DATA)
    actual_h2o = state.particles.species[state.particles.get_species_idx("H2O")]
    assert actual_h2o == expected_h2o
    assert not np.allclose(
        state.particles.spec_masses[:, h2o_idx], supplied_h2o_mass,
        rtol=1e-6, atol=0.0,
    )


def test_name_only_calls_keep_existing_ld_chem_lookup_behavior():
    state, _, _ = create_parcel_scenario(
        num_concs=np.array([1.0e6]),
        pHs=np.array([4.5]),
        species_names=np.array(["OC", "H2O"]),
        species_masses=np.array([[1.0e-18, 0.0]]),
        specdata_path=SPECIES_DATA,
    )

    expected = retrieve_one_species("OC", specdata_path=SPECIES_DATA)
    actual = state.particles.species[state.particles.get_species_idx("OC")]
    assert actual == expected


def test_ld_chem_added_species_still_use_ld_chem_data():
    population = _build_source_population()
    names, masses, num_concs, species = _handoff_arrays(population)

    state, _, _ = create_parcel_scenario(
        num_concs=num_concs,
        pHs=np.full(num_concs.shape, 4.5),
        species_names=names,
        species_masses=masses,
        specdata_path=SPECIES_DATA,
        aero_species=species,
    )

    for name in ("H+", "OH-"):
        expected = retrieve_one_species(name, specdata_path=SPECIES_DATA)
        actual = state.particles.species[state.particles.get_species_idx(name)]
        assert actual == expected


@pytest.mark.parametrize(
    "names",
    [
        np.array([["OC", "H2O"]]),
        np.array([["OC"], ["H2O"]]),
    ],
)
def test_singleton_2d_name_axes_are_accepted(names):
    source = (
        SimpleNamespace(name="OC", density=1000.0, kappa=0.001),
        SimpleNamespace(name="H2O"),
    )
    normalized, normalized_masses, snapshots = _prepare(
        names, [[0.0, 0.0], [0.0, 0.0]], source
    )
    assert list(normalized) == ["OC", "H2O"]
    assert normalized_masses.shape == (2, 2)
    assert list(snapshots) == ["OC", "H2O"]


def test_ragged_species_names_are_rejected_with_contract_error():
    with pytest.raises(ValueError, match="species_names must be 1-D"):
        _prepare(
            [["OC"], ["H2O", "SO4"]],
            np.zeros((1, 3)),
            (),
        )


def test_genuine_2d_name_grid_is_rejected():
    with pytest.raises(ValueError, match="single-row/single-column"):
        _prepare(
            np.array([["OC", "H2O"], ["SO4", "NH4"]]),
            np.zeros((1, 4)),
            (),
        )


@pytest.mark.parametrize(
    "masses",
    [
        np.zeros(2),
        [[[0.0, 0.0]]],
        [[0.0], [0.0, 0.0]],
    ],
)
def test_mass_matrix_must_be_rectangular_2d(masses):
    source = (
        SimpleNamespace(name="OC", density=1000.0, kappa=0.001),
        SimpleNamespace(name="H2O"),
    )
    with pytest.raises(ValueError, match="species_masses"):
        _prepare(np.array(["OC", "H2O"]), masses, source)


def test_mass_columns_must_match_species_names():
    source = (SimpleNamespace(name="OC", density=1000.0, kappa=0.001),)
    with pytest.raises(
        ValueError, match="species_names must describe exactly one entry per"
    ):
        _prepare(np.array(["OC"]), np.zeros((1, 2)), source)


def test_aero_species_must_be_iterable():
    source = SimpleNamespace(name="OC", density=1000.0, kappa=0.001)
    with pytest.raises(TypeError, match="aero_species must be an iterable"):
        _prepare(np.array(["OC"]), np.zeros((1, 1)), source)


@pytest.mark.parametrize(
    ("names", "source", "message"),
    [
        (
            np.array(["OC", "H2O"]),
            (SimpleNamespace(name="OC", density=1000.0, kappa=0.001),),
            "exactly one species object",
        ),
        (
            np.array(["OC", "H2O"]),
            (SimpleNamespace(name="H2O"), SimpleNamespace(name="OC", density=1000.0, kappa=0.001)),
            "same order",
        ),
        (
            np.array(["OC", "OC"]),
            (SimpleNamespace(name="OC", density=1000.0, kappa=0.001),) * 2,
            "unique",
        ),
    ],
)
def test_species_count_order_and_uniqueness_are_enforced(names, source, message):
    with pytest.raises(ValueError, match=message):
        _prepare(names, np.zeros((1, len(names))), source)


def test_species_name_attribute_is_required():
    with pytest.raises(TypeError, match="required attributes: name"):
        _prepare(np.array(["OC"]), np.zeros((1, 1)), (SimpleNamespace(),))


@pytest.mark.parametrize("missing", ["density", "kappa"])
def test_particulate_density_and_kappa_attributes_are_required(missing):
    values = {"name": "OC", "density": 1000.0, "kappa": 0.001}
    del values[missing]
    with pytest.raises(TypeError, match=rf"required attributes:.*{missing}"):
        _prepare(
            np.array(["OC"]), np.zeros((1, 1)),
            (SimpleNamespace(**values),),
        )


@pytest.mark.parametrize(
    ("attribute", "value", "expected_exception"),
    [
        ("density", [1000.0], TypeError),
        ("kappa", np.array([0.001]), TypeError),
        ("density", "not-a-number", TypeError),
        ("kappa", np.nan, ValueError),
        ("density", np.inf, ValueError),
    ],
)
def test_particulate_density_and_kappa_reject_invalid_values(
        attribute, value, expected_exception):
    values = {"name": "OC", "density": 1000.0, "kappa": 0.001}
    values[attribute] = value
    with pytest.raises(expected_exception):
        _prepare(
            np.array(["OC"]), np.zeros((1, 1)),
            (SimpleNamespace(**values),),
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("density", "1000.0"),
        ("kappa", "0.001"),
        ("density", True),
        ("kappa", False),
        ("density", 1000.0 + 0j),
        ("kappa", 0.001 + 0j),
    ],
)
def test_particulate_density_and_kappa_reject_non_real_numeric_scalar_types(
        attribute, value):
    values = {"name": "OC", "density": 1000.0, "kappa": 0.001}
    values[attribute] = value
    with pytest.raises(TypeError, match="scalar real numeric"):
        _prepare(
            np.array(["OC"]), np.zeros((1, 1)),
            (SimpleNamespace(**values),),
        )


@pytest.mark.parametrize("density", [0.0, -1.0])
def test_particulate_density_must_remain_positive(density):
    source = SimpleNamespace(name="OC", density=density, kappa=0.001)
    with pytest.raises(ValueError, match="density must be > 0"):
        _prepare(np.array(["OC"]), np.zeros((1, 1)), (source,))


def test_particulate_kappa_must_be_nonnegative():
    source = SimpleNamespace(name="OC", density=1000.0, kappa=-0.001)
    with pytest.raises(ValueError, match="kappa must be >= 0"):
        _prepare(np.array(["OC"]), np.zeros((1, 1)), (source,))


def test_unknown_species_is_rejected_with_old_or_hardened_loader():
    source = SimpleNamespace(name="OCC", density=1000.0, kappa=0.001)
    # The historical loader falls through with UnboundLocalError and the helper
    # translates it; hardened loaders may raise ValueError directly. The public
    # contract is rejection either way.
    with pytest.raises(ValueError):
        _prepare(np.array(["OCC"]), np.zeros((1, 1)), (source,))


@pytest.mark.parametrize("name", ["NaCl", "biological", "IEPOX_SOA"])
def test_part2pop_only_species_remain_rejected_by_default_ld_chem_data(name):
    population = build_population(
        {
            "type": "binned_lognormals",
            "N": [1.0e8],
            "GMD": [100e-9],
            "GSD": [1.4],
            "aero_spec_names": [[name]],
            "aero_spec_fracs": [[1.0]],
            "N_bins": 2,
            "N_sigmas": 3,
        }
    )
    names, masses, _, species = _handoff_arrays(population)
    with pytest.raises(ValueError):
        _prepare(names, masses, species)


def test_part2pop_alias_is_canonicalized_before_ld_chem_validation():
    population = build_population(
        {
            "type": "binned_lognormals",
            "N": [1.0e8],
            "GMD": [100e-9],
            "GSD": [1.4],
            "aero_spec_names": [["org"]],
            "aero_spec_fracs": [[1.0]],
            "N_bins": 2,
            "N_sigmas": 3,
        }
    )
    names, masses, _, species = _handoff_arrays(population)
    normalized, _, snapshots = _prepare(names, masses, species)
    assert "org" not in normalized
    assert "OC" in normalized
    assert "OC" in snapshots


def test_custom_ld_chem_table_can_admit_name_without_importing_out_of_scope_metadata(tmp_path):
    (tmp_path / "aero_data.dat").write_text(
        "CUSTOM 1000 0 50d-3 0.1\n", encoding="utf-8"
    )
    source = SimpleNamespace(
        name="CUSTOM",
        density=900.0,
        kappa=0.02,
        molar_mass=0.040,
        surface_tension=0.070,
    )

    _, _, snapshots = _prepare_initial_species(
        np.array(["CUSTOM"]), np.zeros((1, 1)), (source,),
        specdata_path=str(tmp_path) + "/",
    )

    actual = snapshots["CUSTOM"]
    assert actual.density == pytest.approx(900.0)
    assert actual.kappa == pytest.approx(0.02)
    assert actual.molar_mass == pytest.approx(0.050)
    assert actual.surface_tension == pytest.approx(0.072)


def test_zero_density_species_keep_complete_ld_chem_definition():
    source = SimpleNamespace(name="H+", density=1000.0, kappa=0.9, molar_mass=99.0)
    _, _, snapshots = _prepare(np.array(["H+"]), np.zeros((1, 1)), (source,))
    assert snapshots["H+"] == retrieve_one_species("H+", specdata_path=SPECIES_DATA)


def test_snapshot_is_isolated_from_later_source_mutation():
    source = SimpleNamespace(name="OC", density=950.0, kappa=0.017)
    _, _, snapshots = _prepare(np.array(["OC"]), np.zeros((1, 1)), (source,))
    snapshot = snapshots["OC"]

    source.density = 2000.0
    source.kappa = 0.5

    assert snapshot.density == pytest.approx(950.0)
    assert snapshot.kappa == pytest.approx(0.017)


def test_public_parcel_driver_preserves_readme_dry_physics_in_output(tmp_path):
    population = _build_readme_population()
    names, masses, num_concs, species = _handoff_arrays(population)
    expected_dry_diameter = population.get_particle_var("dry_diameter")
    expected_kappa = population.get_particle_var("tkappa")

    output_filename = tmp_path / "trajectory.pkl"
    run_module.simulate_parcel(
        names,
        masses,
        num_concs,
        np.full(num_concs.shape, 4.5),
        z_start=0.0,
        z_end=1.0,
        dt=1.0,
        updraft_velocity=1.0,
        condensation=False,
        cocondensation=False,
        aq_chemistry=None,
        gas_chemistry=False,
        aero_species=species,
        mechanism_data_path=str(ROOT / "src" / "ld_chem" / "mechanisms") + "/",
        specdata_path=SPECIES_DATA,
        output_filename=str(output_filename),
        restart_filename=str(tmp_path / "restart.pkl"),
        status_filename=str(tmp_path / "status"),
        progress_filename=str(tmp_path / "progress.out"),
        write_every=10.0,
        print_to_screen=False,
    )

    with output_filename.open("rb") as handle:
        output = pickle.load(handle)

    particle_fields = list(output["particle species"])
    dry_idx = particle_fields.index("Ddry")
    kappa_idx = particle_fields.index("kappa")
    np.testing.assert_allclose(
        output["particles"][0, :, dry_idx],
        expected_dry_diameter,
        rtol=1e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        output["particles"][0, :, kappa_idx],
        expected_kappa,
        rtol=1e-10,
        atol=0.0,
    )


class _StopAfterScenario(Exception):
    pass


def test_simulate_parcel_forwards_aero_species(monkeypatch):
    supplied_species = (object(),)
    captured = {}

    def fake_create_parcel_scenario(**kwargs):
        captured["aero_species"] = kwargs["aero_species"]
        raise _StopAfterScenario

    monkeypatch.setattr(run_module, "create_parcel_scenario", fake_create_parcel_scenario)

    with pytest.raises(_StopAfterScenario):
        run_module.simulate_parcel(
            np.array(["OC"]),
            np.array([[1.0e-16]]),
            np.array([1.0e6]),
            np.array([7.0]),
            aero_species=supplied_species,
            mechanism_data_path="unused-mechanisms/",
            specdata_path="unused-species/",
            print_to_screen=False,
        )

    assert captured["aero_species"] is supplied_species


def test_simulate_les_trajectory_forwards_aero_species(monkeypatch):
    supplied_species = (object(),)
    captured = {}

    def fake_create_les_scenario(**kwargs):
        captured["aero_species"] = kwargs["aero_species"]
        raise _StopAfterScenario

    monkeypatch.setattr(run_module, "create_les_scenario", fake_create_les_scenario)

    with pytest.raises(_StopAfterScenario):
        run_module.simulate_les_trajectory(
            np.array(["OC"]),
            np.array([[1.0e-16]]),
            np.array([1.0e6]),
            np.array([7.0]),
            _trajectory(),
            aero_species=supplied_species,
            mechanism_data_path="unused-mechanisms/",
            specdata_path="unused-species/",
            print_to_screen=False,
        )

    assert captured["aero_species"] is supplied_species
