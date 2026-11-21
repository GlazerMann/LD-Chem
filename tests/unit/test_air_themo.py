import pytest
import numpy as np
from ld_chem.processes.air_thermo import (
    es, S_to_wv, H2O_gas_conc, H2O_mole_fraction, compute_thermo_props, dstate_dt
)


def test_es():
    """Test saturation vapor pressure calculation."""
    # Test at 0°C (273.15 K)
    T_c = 0.0
    expected_es = 611.2  # Pa
    assert np.isclose(es(T_c), expected_es, rtol=1e-3)

    # Test at 25°C (298.15 K)
    T_c = 25.0
    expected_es = 3168.0  # Approximate value
    assert np.isclose(es(T_c), expected_es, rtol=1e-2)
    
    # Test at negative temperatures
    T_c = -10.0
    es_cold = es(T_c)
    assert es_cold > 0
    
    # Test at high temperatures
    T_c = 40.0
    es_hot = es(T_c)
    assert es_hot > es(25.0)


def test_S_to_wv():
    """Test conversion from saturation ratio to water vapor mixing ratio."""
    S = 1.0  # Saturated
    T = 298.15  # 25°C
    P = 101325  # 1 atm
    wv = S_to_wv(S, T, P)
    assert wv > 0
    assert wv < 1  # Mixing ratio should be less than 1

    # Test with S=0
    wv_dry = S_to_wv(0.0, T, P)
    assert wv_dry == 0.0
    
    # Test with S>1 (supersaturated)
    wv_super = S_to_wv(1.1, T, P)
    assert wv_super > wv
    
    # Test at different pressures
    wv_high_p = S_to_wv(S, T, P * 2)
    assert wv_high_p > 0
    
    # Test at different temperatures
    wv_cold = S_to_wv(S, 280.0, P)
    assert wv_cold > 0


@pytest.mark.parametrize("S, T, P", [(0.2, 298.15, 101325.0),
                                    (0.7, 280.0, 80000.0),
                                    (1.1, 305.0, 95000.0)])
def test_S_to_wv_uses_actual_vapor_pressure_in_denominator(S, T, P):
    T_C = T - 273.15
    e_sat_hPa = 6.1094 * np.exp(17.625 * T_C / (243.04 + T_C))
    e_hPa = S * e_sat_hPa
    expected = 0.622 * e_hPa / (P / 100.0 - e_hPa)

    assert np.isclose(S_to_wv(S, T, P), expected, rtol=1e-12)


def test_H2O_gas_conc():
    """Test water vapor gas concentration calculation."""
    S = 1.0
    T = 298.15
    P = 101325
    conc = H2O_gas_conc(S, T, P)
    assert conc > 0
    # Should be approximately P_H2O / (R*T)
    Psat = es(T - 273.15)
    expected = (S * Psat) / (8.314 * T)
    assert np.isclose(conc, expected, rtol=1e-3)
    
    # Test with S=0
    conc_dry = H2O_gas_conc(0.0, T, P)
    assert conc_dry == 0.0
    
    # Test at different temperatures
    conc_hot = H2O_gas_conc(S, 310.0, P)
    conc_cold = H2O_gas_conc(S, 280.0, P)
    assert conc_hot > 0 and conc_cold > 0


def test_H2O_mole_fraction():
    """Test water vapor mole fraction calculation."""
    S = 1.0
    T = 298.15
    P = 101325
    mole_frac = H2O_mole_fraction(S, T, P)
    assert mole_frac > 0
    assert mole_frac < 1

    # At saturation, mole fraction should be Psat/P
    Psat = es(T - 273.15)
    expected = Psat / P
    assert np.isclose(mole_frac, expected, rtol=1e-3)
    
    # Test with S=0
    mole_frac_dry = H2O_mole_fraction(0.0, T, P)
    assert mole_frac_dry == 0.0
    
    # Test with S>1
    mole_frac_super = H2O_mole_fraction(1.1, T, P)
    assert mole_frac_super > mole_frac


def test_compute_thermo_props():
    """Test computation of thermodynamic properties."""
    T = 298.15
    P = 101325
    S = 1.0

    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)

    assert pv_sat > 0
    assert rho_air > 0
    assert rho_air_dry > 0
    assert rho_air_dry < rho_air  # Dry air density should be less than moist air

    # Test with S=0 (dry air)
    pv_sat_dry, rho_air_dry_only, rho_air_dry_dry = compute_thermo_props(T, P, 0.0)
    assert rho_air_dry_only == rho_air_dry_dry
    
    # Test at different conditions
    pv_sat_cold, rho_cold, rho_cold_dry = compute_thermo_props(280.0, 100000, 0.95)
    assert pv_sat_cold > 0 and rho_cold > 0
    
    # Test at high pressure
    pv_sat_hp, rho_hp, rho_hp_dry = compute_thermo_props(T, 150000, S)
    assert rho_hp > rho_air  # Higher pressure should increase density


def test_dstate_dt():
    """Test state vector time derivatives calculation."""
    # Define initial parcel state: [z, T, P, S, wv]
    X0 = np.array([0.0, 298.15, 101325.0, 0.95, 0.01])
    V = 1.0  # updraft velocity in m/s
    dwc_dt = 0.0  # no condensation initially
    
    dX_dt = dstate_dt(X0, V, dwc_dt)
    
    # Check that output shape is correct
    assert dX_dt.shape == X0.shape
    assert len(dX_dt) == 5
    
    # Check that derivatives are finite (no NaNs)
    assert np.all(np.isfinite(dX_dt))
    
    # dz/dt should equal updraft velocity
    assert np.isclose(dX_dt[0], V)
    
    # dP/dt should be negative (pressure decreases with altitude)
    assert dX_dt[2] < 0
    
    # dT/dt should be negative (adiabatic cooling with upward motion)
    assert dX_dt[1] < 0


def test_dstate_dt_with_condensation():
    """Test state derivatives with condensation."""
    X0 = np.array([100.0, 298.15, 101000.0, 1.0, 0.012])
    V = 2.0
    dwc_dt = 0.001  # condensation occurring
    
    dX_dt = dstate_dt(X0, V, dwc_dt)
    
    # Check all derivatives are finite
    assert np.all(np.isfinite(dX_dt))
    
    # dz/dt should equal updraft velocity
    assert np.isclose(dX_dt[0], V)
    
    # With condensation, water vapor should decrease
    assert dX_dt[4] < 0


def test_dstate_dt_different_velocities():
    """Test state derivatives at different updraft velocities."""
    X0 = np.array([500.0, 298.15, 95000.0, 0.99, 0.011])
    dwc_dt = 0.0
    
    # Test with different velocities
    for V in [0.5, 1.0, 2.0, 5.0]:
        dX_dt = dstate_dt(X0, V, dwc_dt)
        assert np.isclose(dX_dt[0], V)  # dz/dt should equal V
        assert np.all(np.isfinite(dX_dt))
    
    # Higher updraft should cause greater cooling (more negative dT/dt)
    dX_dt_fast = dstate_dt(X0, 5.0, dwc_dt)
    dX_dt_slow = dstate_dt(X0, 0.5, dwc_dt)
    assert dX_dt_fast[1] < dX_dt_slow[1]  # Faster cooling at higher V


def test_dstate_dt_edge_cases():
    """Test state derivatives at edge cases."""
    # Test with very low updraft
    X0 = np.array([100.0, 298.15, 101325.0, 1.0, 0.01])
    dX_dt = dstate_dt(X0, 0.1, 0.0)
    assert np.all(np.isfinite(dX_dt))
    
    # Test at different altitudes
    X0_high = np.array([5000.0, 280.0, 54000.0, 0.9, 0.005])
    dX_dt_high = dstate_dt(X0_high, 1.5, 0.0)
    assert np.all(np.isfinite(dX_dt_high))
