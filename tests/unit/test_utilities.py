import pytest
import numpy as np
from ld_chem.utilities import (
    check_gas_condensation, check_water_condensation,
    check_mass_balance, check_gas_chemistry
)


def create_mock_parcel_state(T=298.15, P=101325.0, S=0.85):
    """Create a mock parcel state for testing."""
    # Mock species
    species_h2o = type('MockSpecies', (), {
        'name': 'H2O',
        'molar_mass': 0.018,
        'density': 1000.0
    })()

    species_so2 = type('MockSpecies', (), {
        'name': 'SO2',
        'molar_mass': 0.064
    })()

    # Mock particle population
    particles = type('MockParticles', (), {
        'species': [species_h2o, species_so2],
        'spec_masses': np.array([[1e-25, 0.0]]),  # kg per particle
        'num_concs': np.array([1e6]),  # particles/m^3
        'get_species_idx': lambda self, name: 0 if name == 'H2O' else 1
    })()

    # Mock gas population
    gas_so2 = type('MockGas', (), {
        'name': 'SO2',
        'molar_mass': 0.064
    })()

    gas = type('MockGasPop', (), {
        'gases': [gas_so2],
        'concs': [1e-6]  # ppb
    })()

    # Mock parcel state
    parcel_state = type('MockParcelState', (), {
        'T': T,
        'P': P,
        'S': S,
        'particles': particles,
        'gas': gas
    })()

    return parcel_state


def test_check_water_condensation_pass():
    """Test water condensation check with balanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Set up balanced water masses
    water_idx = 0
    parcel_0.particles.spec_masses[0, water_idx] = 1e-25  # kg
    parcel_next.particles.spec_masses[0, water_idx] = 1.1e-25  # kg (slightly more)

    dwc_dt = 1e6 * (1.1e-25 - 1e-25)  # kg/m^3/s

    # Should not raise an exception
    result = check_water_condensation(parcel_0, parcel_next, dwc_dt)
    assert result is None


def test_check_water_condensation_fail():
    """Test water condensation check with imbalanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Set up imbalanced water masses
    water_idx = 0
    parcel_0.particles.spec_masses[0, water_idx] = 1e-25
    parcel_next.particles.spec_masses[0, water_idx] = 2e-25  # Much more
    dwc_dt = 1e6*(1.2e-25+2e-25) # Wrong dwc_dt
    
    # Should raise an exception
    with pytest.raises(ValueError):
        check_water_condensation(parcel_0, parcel_next, dwc_dt)


def test_check_mass_balance_pass():
    """Test mass balance check with balanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Set up balanced masses
    parcel_0.particles.spec_masses = np.array([[1e-25, 2e-26]])
    parcel_next.particles.spec_masses = np.array([[1e-25, 2e-26]])  # Same total mass

    # Should not raise an exception
    result = check_mass_balance(parcel_0, parcel_next)
    assert result is None


def test_check_mass_balance_fail():
    """Test mass balance check with imbalanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Set up imbalanced masses
    parcel_0.particles.spec_masses = np.array([[1e-25, 2e-26]])  # Total: 1.2e-25
    parcel_next.particles.spec_masses = np.array([[1e-25, 3e-26]])  # Total: 1.3e-25

    # Should raise ValueError
    with pytest.raises(ValueError):
        check_mass_balance(parcel_0, parcel_next)


def test_check_gas_chemistry_pass():
    """Test gas chemistry check with balanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Set up balanced gas concentrations
    parcel_0.gas.concs = [10.0]  # ppb
    parcel_next.gas.concs = [10.0]  # Same concentration

    # Should not raise an exception
    result = check_gas_chemistry(parcel_0, parcel_next)
    assert result is None


def test_check_gas_chemistry_fail():
    """Test gas chemistry check with imbalanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # A 1 ppb change is above the mass check's 1e-15 kg/m^3 noise floor.
    # The former 1e-7 ppb change was below that floor after unit conversion.
    parcel_0.gas.concs = [10.0]
    parcel_next.gas.concs = [11.0]

    # Should raise ValueError
    with pytest.raises(ValueError, match="Mass not conserved in gas chemistry"):
        check_gas_chemistry(parcel_0, parcel_next)


def test_check_gas_condensation_pass():
    """Test gas condensation check with balanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Mock gas feedback
    gas_feedback = type('MockGasFeedback', (), {
        'names': ['SO2'],
        'dc_dts': [-1e-9]  # ppb/s
    })()

    # Set up balanced particle masses
    so2_idx = 1
    parcel_0.particles.spec_masses[0, so2_idx] = 0.0
    parcel_next.particles.spec_masses[0, so2_idx] = 1e-27  # Small amount condensed

    # Should not raise an exception and return the feedback
    result = check_gas_condensation(parcel_0, parcel_next, gas_feedback)
    assert result is gas_feedback


def test_check_gas_condensation_correction():
    """Test gas condensation check with correction of imbalanced data."""
    parcel_0 = create_mock_parcel_state()
    parcel_next = create_mock_parcel_state()

    # Mock gas feedback with wrong dc_dt
    gas_feedback = type('MockGasFeedback', (), {
        'names': ['SO2'],
        'dc_dts': [-1e-9]  # Wrong value
    })()

    # Set up particle masses that don't match the feedback
    so2_idx = 1
    parcel_0.particles.spec_masses[0, so2_idx] = 0.0
    parcel_next.particles.spec_masses[0, so2_idx] = 2e-27  # Different amount

    # Should correct the feedback value
    result = check_gas_condensation(parcel_0, parcel_next, gas_feedback)
    
    print(result.dc_dts)
    
    assert result is gas_feedback
    # The dc_dt should have been corrected
    assert result.dc_dts[0] != -1e-9
