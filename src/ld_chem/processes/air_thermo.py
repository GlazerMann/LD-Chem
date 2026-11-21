#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:03:39 2024

@author: Laura Fierce

"""

import numba as nb
import numpy as np
from numba.pycc import CC
import ld_chem.constants as c

# AOT/numba stuff
auxcc = CC("parcel_aux_numba")
auxcc.verbose = True

@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    return 611.2 * np.exp(17.67 * T / (T + 243.5))

## RHS Derivative callback function
@nb.njit()
def dstate_dt(X0, V, dwc_dt):
    """Calculate the instantaneous thermodynamic parcel-state derivative.

    Parameters
    ----------
    X0 : array_like
        Current parcel state, ordered as:
            * X0[0] = altitude, m
            * X0[1] = temperature, K
            * X0[2] = pressure, Pa
            * X0[3] = saturation ratio
            * X0[4] = water-vapor mass mixing ratio, kg/kg
    V : float
        Updraft velocity, m/s.

    dwc_dt : float
        Condensed-water mass concentration **rate**, kg m^-3 s^-1. This is a
        rate, not the total condensed-water mass change over a model timestep.

    Returns
    -------
    dX_dt : ndarray
        Five-element derivative corresponding to ``X0``.

    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """    
    z=X0[0]
    T=X0[1]
    P=X0[2]
    S=X0[3]
    wv=X0[4]
    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)
    dwc_dt/=rho_air # convert from kg/m^to kg/kg

    ## Begin computing tendencies
    dP_dt = -1.0 * rho_air * c.g * V
    
    # No freezing implemented yet
    dwi_dt = 0.0

    ## MASS BALANCE CONSTRAINT
    dwv_dt = -1.0 * (dwc_dt + dwi_dt)
    
    ## ADIABATIC COOLING
    dT_dt = -c.g * V / c.Cp - c.L * dwv_dt / c.Cp
    dz_dt = V
    
    """ Alternative methods for calculation supersaturation tendency
    # Used eq 12.28 from Pruppacher and Klett in stead of (9) from Nenes et al, 2001
    #cdef double S_a, S_b, S_c, dS_dt
    #cdef double S_b_old, S_c_old, dS_dt_old
    #S_a = (S+1.0)

    ## NENES (2001)
    #S_b_old = dT_dt*wv_sat*(17.67*243.5)/((243.5+(Tv-273.15))**2.)
    #S_c_old = (rho_air*g*V)*(wv_sat/P)*((0.622*L)/(Cp*Tv) - 1.0)
    #dS_dt_old = (1./wv_sat)*(dwv_dt - S_a*(S_b_old-S_c_old))

    ## PRUPPACHER (PK 1997)
    #S_b = dT_dt*0.622*L/(Rd*T**2.)
    #S_c = g*V/(Rd*T)
    #dS_dt = P*dwv_dt/(0.622*es(T-273.15)) - S_a*(S_b + S_c)

    ## SEINFELD (SP 1998)
    #S_b = L*Mw*dT_dt/(R*T**2.)
    #S_c = V*g*Ma/(R*T)
    #dS_dt = dwv_dt*(Ma*P)/(Mw*es(T-273.15)) - S_a*(S_b + S_c)
    """
    
    ## GHAN (2011)
    alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2))
    alpha -= (c.g * c.Ma) / (c.R * T)
    gamma = (P * c.Ma) / (c.Mw * pv_sat)
    gamma += (c.Mw * c.L * c.L) / (c.Cp * c.R * T * T)
    dS_dt = alpha * V - gamma * dwc_dt

    dX_dt = np.zeros(X0.shape)
    dX_dt[0] = dz_dt
    dX_dt[1] = dT_dt
    dX_dt[2] = dP_dt
    dX_dt[3] = dS_dt
    dX_dt[4] = dwv_dt
    
    return dX_dt

@nb.njit()
def compute_thermo_props(T, P, S):
    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    wv = S_to_wv(S,T,P)
    
    Tv = (1.0 + 0.61 * wv) * T
    e = S * pv_sat  # water vapor pressure
    
    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    rho_air_dry = (P - e) / c.Rd / T
    return pv_sat, rho_air, rho_air_dry

@nb.njit()
def S_to_wv(S,T,P):
    # saturation ratio to mixing ratio [mols h2o/mol air]
    
    p_hPa = P/100. # convert Pa to hPa
    T_C = T-273.15 # Convert from kelvin to degrees C
    
    #Saturation vapor pressure (T)
    e_sat = 6.1094*np.exp(17.625*T_C/(243.04+T_C)) #(hPa)
    
    # Vapor pressure from saturation ratio
    e = S*e_sat
    wv = e*0.622/(p_hPa - e_sat) 
    
    return wv

def H2O_gas_conc(S,T,P):
    Psat = es(T-273.15) # Pa
    P_H2O = S*Psat # Pa    
    return P_H2O/(c.R*T) # mol/m^3

def H2O_mole_fraction(S,T,P):
    Psat = es(T-273.15) 
    return (S*Psat)/P 
