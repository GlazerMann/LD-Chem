import pytest
import numpy as np
import warnings
from pathlib import Path
from ld_chem.scenario import (
    LagrangianElement, LagrangianElementDriver,
    create_parcel_scenario, create_les_scenario
)
from ld_chem.particles import AerosolSpecies
from ld_chem.gases import GasSpecies, TraceGasPopulation


def test_lagrangian_element_creation():
    """Test LagrangianElement dataclass creation."""
    # Create mock particle population
    species = AerosolSpecies("SO4", 1800.0, 0.65, 0.096, 0.072)
    particle_pop = type('MockParticlePop', (), {
        'spec_masses': np.array([[1e-25]]),
        'num_concs': np.array([1e6]),
        'get_particle_var': lambda self, var: np.array([1e-6]) if 'diameter' in var else np.array([0.1])
    })()

    # Create mock gas population
    gas = GasSpecies("SO2", 0.11, 0.064, 1.4, 2900.0)
    gas_pop = TraceGasPopulation(
        gases=(gas,),
        concs=(1e-6,),
        ids=(0,)
    )
    element = LagrangianElement(
        particles=particle_pop,
        gas=gas_pop,
        x=0.0, y=0.0, z=100.0,
        u=0.0, v=0.0, w=1.0,
        S=0.85, P=101325.0, T=298.15
    )
    assert element.x == 0.0
    assert element.z == 100.0
    assert element.T == 298.15
    assert element.particles is particle_pop
    assert element.gas is gas_pop


def test_lagrangian_element_clone_detached():
    """Test detached cloning of LagrangianElement."""
    # Create mock particle population with clone_detached method
    particle_pop = type('MockParticlePop', (), {
        'spec_masses': np.array([[1e-25]]),
        'num_concs': np.array([1e6]),
        'clone_detached': lambda self: type('MockParticlePop', (), {'spec_masses': np.array([[1e-25]]), 'num_concs': np.array([1e6])})()
    })()

    # Create mock gas population with clone_detached method
    gas_pop = type('MockGasPop', (), {
        'gases': ('gas',),
        'concs': (1e-6,),
        'ids': (0,),
        'clone_detached': lambda self: type('MockGasPop', (), {'gases': ('gas',), 'concs': (1e-6,), 'ids': (0,)})()
    })()

    element = LagrangianElement(
        particles=particle_pop,
        gas=gas_pop,
        x=0.0, y=0.0, z=100.0,
        u=0.0, v=0.0, w=1.0,
        S=0.85, P=101325.0, T=298.15
    )

    cloned = element.clone_detached()

    # Check that it's a different object
    assert cloned is not element

    # Check that primitive values are copied
    assert cloned.x == element.x
    assert cloned.T == element.T

    # Check that complex objects are detached clones
    assert cloned.particles is not element.particles
    assert cloned.gas is not element.gas


def test_lagrangian_element_driver_creation():
    """Test LagrangianElementDriver dataclass creation."""
    driver = LagrangianElementDriver(
        t_data=np.array([0.0, 1.0, 2.0]),
        z_data=np.array([0.0, 100.0, 200.0]),
        T_data=np.array([298.0, 299.0, 300.0]),
        P_data=np.array([101325.0, 101320.0, 101315.0])
    )

    assert np.array_equal(driver.t_data, [0.0, 1.0, 2.0])
    assert np.array_equal(driver.z_data, [0.0, 100.0, 200.0])
    assert np.array_equal(driver.T_data, [298.0, 299.0, 300.0])


def test_create_parcel_scenario_basic():
    """Test basic parcel scenario creation."""
    # Set up minimal test data
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array(['SO4','H2O'])
    species_masses = np.array([[1e-16, 1e-25]])

    # Get paths
    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    # Test basic scenario creation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        parcel_state, aq_reactions, gas_reactions = create_parcel_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            z_end=10.0,  # Very short simulation
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            aq_chemistry=None,
            gas_chemistry=False
        )
        parcel_state.particles.spec_masses[:,parcel_state.particles.get_species_idx("H2O")]*=10000.0

        # Check that we got a LagrangianElement
        assert isinstance(parcel_state, LagrangianElement)
        assert parcel_state.particles is not None
        assert parcel_state.gas is None  # No gas chemistry
        assert aq_reactions is None  # No aqueous chemistry
        assert parcel_state.get_activated_fraction() > 0.0  # Should have some activation


def test_create_parcel_scenario_gas_chemistry_uses_package_data_defaults(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    # Isolate package-data lookup from the constructor's aerosol defaults.
    parcel_state, _, gas_reactions = create_parcel_scenario(
        gas_chemistry=True,
        updraft_velocity=0.0,
        species_names=np.array(["SO4", "H2O"]),
        species_masses=np.array([[1e-25, 1e-25]]),
    )

    assert parcel_state.gas is not None
    assert gas_reactions is not None


def test_create_parcel_scenario_with_cocondensation():
    """Test parcel scenario creation with gas chemistry."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array(['SO4','H2O'])
    species_masses = np.array([[1e-25, 1e-25]])
    gas_names = ['SO2']
    gas_concs = [1e-6]

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore") # ignore divide by zero warnings
        parcel_state, aq_reactions, gas_reactions = create_parcel_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            gas_names=gas_names,
            gas_concs=gas_concs,
            z_end=10.0,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            gas_chemistry=False,
            cocondensation=True
        )

    assert isinstance(parcel_state, LagrangianElement)
    assert parcel_state.gas is not None
    assert gas_reactions is None


def test_create_parcel_scenario_with_gas_chemistry():
    """Test parcel scenario creation with gas chemistry."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array([['SO4','H2O']])
    species_masses = np.array([[1e-25, 1e-25]])
    gas_names = ['SO2']
    gas_concs = [1e-6]

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore") # ignore divide by zero warnings
        parcel_state, aq_reactions, gas_reactions = create_parcel_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            gas_names=gas_names,
            gas_concs=gas_concs,
            z_end=10.0,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            gas_chemistry=True,
            cocondensation=False
        )

    assert isinstance(parcel_state, LagrangianElement)
    assert parcel_state.gas is not None
    assert gas_reactions is not None
    assert parcel_state.particles.spec_masses.shape[1]==len(species_names[0])+2 # make sure it doesn't add aby species


def test_create_parcel_scenario_with_aqueous_chemistry():
    """Test parcel scenario creation with gas chemistry."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array(['SO4','H2O'])
    species_masses = np.array([[1e-25, 1e-25]])
    gas_names = None
    gas_concs = None

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore") # ignore divide by zero warnings
        parcel_state, aq_reactions, gas_reactions = create_parcel_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            gas_names=gas_names,
            gas_concs=gas_concs,
            z_end=10.0,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            gas_chemistry=False,
            cocondensation=False,
            aq_chemistry=['sulfate', 'nitrate']
        )

    assert isinstance(parcel_state, LagrangianElement)
    assert parcel_state.gas is None
    assert gas_reactions is None
    assert aq_reactions is not None
    assert parcel_state.particles.get_species_idx("H2SO4") is not None


def test_create_les_scenario_basic():
    """Test basic LES scenario creation."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array([['SO4','H2O']])
    species_masses = np.array([[1e-25, 1e-25]])

    # Mock trajectory data
    trajectory_data = {
        't': np.array([0.0, 1.0]),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0, 1.0]),
        'z': np.array([0.0, 1.0]),
        'T': np.array([298.0, 298.0]),
        'P': np.array([101325.0, 101325.0]),
        's': np.array([0.85, 0.85])
    }

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        parcel_state, driver, aq_reactions, gas_reactions = create_les_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            trajectory_data=trajectory_data,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            aq_chemistry=None,
            gas_chemistry=False,
            cocondensation=False
        )

    assert isinstance(parcel_state, LagrangianElement)
    assert isinstance(driver, LagrangianElementDriver)
    assert parcel_state.particles is not None


def test_create_les_scenario_with_cocondensation():
    """Test basic LES scenario creation."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array([['SO4','H2O']])
    species_masses = np.array([[1e-25, 1e-25]])

    # Mock trajectory data
    trajectory_data = {
        't': np.array([0.0, 1.0]),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0, 1.0]),
        'z': np.array([0.0, 1.0]),
        'T': np.array([298.0, 298.0]),
        'P': np.array([101325.0, 101325.0]),
        's': np.array([0.85, 0.85]),
        'gas': {'SO2': np.array([1.0, 1.0])}
    }

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        parcel_state, driver, aq_reactions, gas_reactions = create_les_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            trajectory_data=trajectory_data,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            aq_chemistry=None,
            gas_chemistry=False,
            cocondensation=True
        )

    assert parcel_state.gas is not None
    assert parcel_state.particles.get_species_idx('SO2') is not None


def test_create_les_scenario_with_gas_chemistry():
    """Test basic LES scenario creation."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array([['SO4','H2O']])
    species_masses = np.array([[1e-25, 1e-25]])

    # Mock trajectory data
    trajectory_data = {
        't': np.array([0.0, 1.0]),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0, 1.0]),
        'z': np.array([0.0, 1.0]),
        'T': np.array([298.0, 298.0]),
        'P': np.array([101325.0, 101325.0]),
        's': np.array([0.85, 0.85]),
        'gas': None
    }

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        parcel_state, driver, aq_reactions, gas_reactions = create_les_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            trajectory_data=trajectory_data,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            aq_chemistry=None,
            gas_chemistry=True,
            cocondensation=False
        )

    assert parcel_state.gas is not None
    assert parcel_state.particles.spec_masses.shape[1] == len(species_names[0])+2 


def test_create_les_scenario_with_aqueous_chemistry():
    """Test basic LES scenario creation."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array([['SO4','H2O']])
    species_masses = np.array([[1e-25, 1e-25]])

    # Mock trajectory data
    trajectory_data = {
        't': np.array([0.0, 1.0]),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0, 1.0]),
        'z': np.array([0.0, 1.0]),
        'T': np.array([298.0, 298.0]),
        'P': np.array([101325.0, 101325.0]),
        's': np.array([0.85, 0.85]),
        'gas': None
    }

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        parcel_state, driver, aq_reactions, gas_reactions = create_les_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            trajectory_data=trajectory_data,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            aq_chemistry=['sulfate','nitrate'],
            gas_chemistry=False,
            cocondensation=False
        )

    assert parcel_state.gas is None
    assert gas_reactions is None
    assert aq_reactions is not None
    assert parcel_state.particles.get_species_idx("H2SO4") is not None
