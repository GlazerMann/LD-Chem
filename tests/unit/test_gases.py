import pytest, warnings
import numpy as np
from pathlib import Path
from ld_chem.particles import retrieve_one_species
from ld_chem.gases import make_TraceGasPopulation
from part2pop.population import ParticlePopulation
from ld_chem.gases import (
    GasSpecies, TraceGasPopulation, retrieve_gas_species,
    equilibrate_gases, make_TraceGasPopulation
)

def test_gas_species_creation():
    """Test GasSpecies dataclass creation."""
    gas = GasSpecies(
        name="SO2",
        alpha=0.11,
        molar_mass=0.064,
        H0=1.4,
        H_exp=2900.0
    )
    assert gas.name == "SO2"
    assert gas.alpha == 0.11
    assert gas.molar_mass == 0.064
    assert gas.H0 == 1.4
    assert gas.H_exp == 2900.0


def test_gas_species_get_heff():
    """Test Henry's law coefficient calculation."""
    gas = GasSpecies(
        name="SO2",
        alpha=0.11,
        molar_mass=0.064,
        H0=1.4,
        H_exp=2900.0
    )
    T = 298  # 25°C
    Heff = gas.get_Heff(T)
    
    # Should be positive
    assert Heff > 0
    # At reference temperature, should equal H0 converted to mol/m^3/Pa
    expected_at_ref = (1000/101325) * gas.H0
    assert np.isclose(Heff, expected_at_ref)

    # Should change with temperature
    T_higher = 308.15  # 35°C
    Heff_higher = gas.get_Heff(T_higher)
    assert Heff_higher != Heff


def test_trace_gas_population_creation():
    """Test TraceGasPopulation dataclass creation."""
    gas1 = GasSpecies("SO2", 0.11, 0.064, 1.4, 2900.0)
    gas2 = GasSpecies("O3", 0.05, 0.048, 0.011, 2300.0)
    gases = (gas1, gas2)
    concs = (1e-6, 2e-6)
    ids = (0, 1)
    pop = TraceGasPopulation(gases=gases, concs=concs, ids=ids)
    assert len(pop.gases) == 2
    assert len(pop.concs) == 2
    assert len(pop.ids) == 2
    assert pop.gases[0].name == "SO2"
    assert pop.gases[1].name == "O3"


def test_trace_gas_population_get_species_idx():
    """Test species index lookup."""
    gas1 = GasSpecies("SO2", 0.11, 0.064, 1.4, 2900.0)
    gas2 = GasSpecies("O3", 0.05, 0.048, 0.011, 2300.0)
    gases = (gas1, gas2)
    concs = (1e-6, 2e-6)
    ids = (0, 1)
    pop = TraceGasPopulation(gases=gases, concs=concs, ids=ids)
    assert pop.get_species_idx("SO2") == 0
    assert pop.get_species_idx("O3") == 1
    assert pop.get_species_idx("NO2") is None


def test_trace_gas_population_clone_detached():
    """Test detached cloning of TraceGasPopulation."""
    gas1 = GasSpecies("SO2", 0.11, 0.064, 1.4, 2900.0)
    gas2 = GasSpecies("O3", 0.05, 0.048, 0.011, 2300.0)

    gases = (gas1, gas2)
    concs = np.array([1e-6, 2e-6])
    ids = (0, 1)

    pop = TraceGasPopulation(gases=gases, concs=concs, ids=ids)
    cloned = pop.clone_detached()

    # Check that it's a different object
    assert cloned is not pop

    # Check that immutable data is shared
    assert cloned.gases is pop.gases
    assert cloned.ids is pop.ids

    # Check that mutable data is detached
    assert cloned.concs is not pop.concs
    assert np.array_equal(cloned.concs, pop.concs)

    # Modify original and check clone is unaffected
    pop.concs[0] = 999
    assert cloned.concs[0] == 1e-6


def test_retrieve_gas_species():
    """Test gas species data retrieval from file."""
    # Path to species_data directory from test file location
    # tests/objects/test_gases.py -> ../../src/ld_chem/species_data/
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"
    
    # Test with a known species from gas_data.dat
    gas = retrieve_gas_species("SO2", specdata_path=str(species_data_path) + "/")

    assert gas.name == "SO2"
    assert isinstance(gas.alpha, float)
    assert isinstance(gas.molar_mass, float)
    assert isinstance(gas.H0, float)
    assert isinstance(gas.H_exp, float)

    # Test with another species
    gas_o3 = retrieve_gas_species("O3", specdata_path=str(species_data_path) + "/")
    assert gas_o3.name == "O3"

def test_retrieve_gas_species_unknown_species(tmp_path):
    (tmp_path / "gas_data.dat").write_text(
        "\nSO2 0.11 0.064 1.4 2900\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown gas species 'NOPE'"):
        retrieve_gas_species("NOPE", specdata_path=str(tmp_path) + "/")


def test_retrieve_gas_species_reports_malformed_matching_row(tmp_path):
    (tmp_path / "gas_data.dat").write_text(
        "SO2 0.11 0.064\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Malformed gas species entry.*line 1.*expected 5 fields",
    ):
        retrieve_gas_species("SO2", specdata_path=str(tmp_path) + "/")


def test_make_trace_gas_population():
    """Test creation of TraceGasPopulation from names and concentrations."""
    # Path to species_data directory from test file location
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"
    
    gas_names = ["SO2", "O3"]
    gas_concs = [1e-6, 2e-6]

    pop = make_TraceGasPopulation(gas_names, gas_concs, specdata_path=str(species_data_path) + "/")

    assert len(pop.gases) == 2
    assert len(pop.concs) == 2
    assert len(pop.ids) == 2

    assert pop.gases[0].name == "SO2"
    assert pop.gases[1].name == "O3"
    assert pop.concs[0] == 1e-6
    assert pop.concs[1] == 2e-6

def test_empty_trace_gas_population():
    """Test TraceGasPopulation creation with empty inputs."""
    gas_names = []
    gas_concs = []
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"
    pop = make_TraceGasPopulation(gas_names, gas_concs, specdata_path=str(species_data_path) + "/")
    assert pop.gases == None
    assert pop.concs == None
    assert pop.ids == None

def test_equilibrate_gases():
    """Test gas-aerosol equilibration."""
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"
    aero_spec_names = np.array(["OC", "H2O", "SO2"])
    species_masses = np.array([[1e-10, 1e-10, 0.0]])
    num_concs = np.array([1e6])
    ids = [ii for ii in range(len(species_masses))]
    aero_specs = []
    for spec in aero_spec_names:
        aero_specs.append(retrieve_one_species(spec, specdata_path=str(species_data_path) + "/"))
    aerosol_population = ParticlePopulation(species=aero_specs, spec_masses=species_masses, num_concs=num_concs, ids=ids)
    gas_names = ["SO2"]
    gas_concs = [1.0]
    gas_population = make_TraceGasPopulation(gas_names, gas_concs, specdata_path=str(species_data_path) + "/")

    # Suppress divide by zero warnings from including SO2 in the particles at this point
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
        radii = 0.5*aerosol_population.get_particle_var('wet_diameter')
        dry_radii = 0.5*aerosol_population.get_particle_var('dry_diameter')
        aerosol_population=equilibrate_gases(aerosol_population,gas_population,298,101325)
    
    # concentration should be close to the equilibrium value
    water_volumes = (4.0/3.0)*np.pi*(radii**3-dry_radii**3)
    SO2_conc = (aerosol_population.spec_masses[:,-1]/aerosol_population.species[-1].molar_mass)/water_volumes # mol/m^3
    equilibrium_SO2_conc = gas_population.gases[0].get_Heff(298)*1e-9*gas_population.concs[0]*101325
    
    assert np.isclose(SO2_conc, equilibrium_SO2_conc)

