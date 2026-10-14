#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:38:52 2024

@author: fier887
"""
import numpy as np
import sys
from ld_chem.processes import water_uptake
from ld_chem.processes.air_thermo import H2O_gas_conc
import ld_chem.constants as c

def check_gas_condensation(ParcelState_0, ParcelState_Next, gas_feedback):
    for ii, (name, dc_dt) in enumerate(zip(gas_feedback.names, gas_feedback.dc_dts)):
        idx = ParcelState_0.particles.get_species_idx(name)
        moles=ParcelState_0.particles.spec_masses[:,idx]/ParcelState_0.particles.species[idx].molar_mass # mol
        Cpart_0=1e9*np.sum(moles*ParcelState_0.particles.num_concs*((8.314*ParcelState_0.T)/ParcelState_0.P)) # ppb
        moles=ParcelState_Next.particles.spec_masses[:,idx]/ParcelState_Next.particles.species[idx].molar_mass # mol
        Cpart_next=1e9*np.sum(moles*ParcelState_Next.particles.num_concs*((8.314*ParcelState_Next.T)/ParcelState_Next.P)) # ppb
        check = np.isclose(-1.0*dc_dt, Cpart_next-Cpart_0, atol=10**(np.round(np.log10(abs(dc_dt))-5.0, 0)))
        if not check:
            gas_feedback.dc_dts[ii]=-1.0*(Cpart_next-Cpart_0)
    return gas_feedback


def check_water_condensation(ParcelState_0, ParcelState_Next, dwc_dt):
    water_idx=ParcelState_0.particles.get_species_idx("H2O")
    masses_0=ParcelState_0.particles.num_concs*ParcelState_0.particles.spec_masses[:,water_idx]
    water_idx=ParcelState_Next.particles.get_species_idx("H2O")
    masses_next=ParcelState_Next.particles.num_concs*ParcelState_Next.particles.spec_masses[:,water_idx]
    check = np.isclose(np.sum(masses_next-masses_0), dwc_dt, atol=10**(np.round(np.log10(abs(dwc_dt))-5.0, 0)))
    if not check:
        raise ValueError('Moles of condensing water does not match the change in particles!')
    return

def check_mass_balance(ParcelState_0, ParcelState_Next):
    Mass_0 = np.sum(ParcelState_0.particles.spec_masses)
    Mass_next = np.sum(ParcelState_Next.particles.spec_masses)    
    check = np.isclose(Mass_0, Mass_next, atol=10**(np.round(np.log10(abs(Mass_0))-5.0, 0)))
    if not check:
        raise ValueError('Mass not conserved in aqueous chemistry module!')
    return


def check_gas_chemistry(ParcelState_0, ParcelState_Next):
    # Trace gases are stored in ppb, while H2O is tracked separately through
    # saturation ratio. Convert both to mass concentration so reactions that
    # produce or consume water are included in the mass check.
    Mass_0 = 0.0
    air_molar_conc_0 = ParcelState_0.P/(c.R*ParcelState_0.T)
    for gas, conc in zip(ParcelState_0.gas.gases, ParcelState_0.gas.concs):
        Mass_0 += 1e-9*conc*air_molar_conc_0*gas.molar_mass
    Mass_0 += H2O_gas_conc(
        ParcelState_0.S, ParcelState_0.T, ParcelState_0.P
    ) * c.Mw

    Mass_next = 0.0
    air_molar_conc_next = ParcelState_Next.P/(c.R*ParcelState_Next.T)
    for gas, conc in zip(ParcelState_Next.gas.gases, ParcelState_Next.gas.concs):
        Mass_next += 1e-9*conc*air_molar_conc_next*gas.molar_mass
    Mass_next += H2O_gas_conc(
        ParcelState_Next.S, ParcelState_Next.T, ParcelState_Next.P
    ) * c.Mw

    check = np.isclose(Mass_0, Mass_next, atol=10**(np.round(np.log10(abs(Mass_0))-5.0, 0)))
    if not check:
        raise ValueError('Mass not conserved in gas chemistry module!')
    return

        
    
    
    
    


