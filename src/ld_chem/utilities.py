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


def _finite_chemistry_scalar(value, name, positive=False):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(array)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _chemistry_gas_inputs(state, label):
    from .gases import _validate_chemistry_gas_names

    gases = tuple(state.gas.gases)
    names = tuple(gas.name for gas in gases)
    _validate_chemistry_gas_names(names)
    concs = np.asarray(state.gas.concs)
    if (concs.ndim != 1 or len(concs) != len(gases)
            or concs.dtype.kind not in "iuf" or not np.all(np.isfinite(concs))):
        raise ValueError(f"{label} gas concentrations must be a finite real vector matching species")
    molar_masses = np.array([
        _finite_chemistry_scalar(gas.molar_mass, f"{label} molar mass", positive=True)
        for gas in gases
    ])
    return names, molar_masses, concs


def check_gas_chemistry(ParcelState_0, ParcelState_Next):
    """Check tracked gas-plus-water mass at the chemistry input's fixed T/P.

    Concentrations are ppb relative to total moist air. H2O and M must not
    occur in the tracked species: humidity and the total-air collider own
    those solver slots. This checks mass only, not elemental/charge balance
    or a mechanism's implicit reservoirs; see docs/gas-chemistry-accounting.md.
    """
    validated_states = []
    for label, state in (("reference", ParcelState_0), ("updated", ParcelState_Next)):
        temperature = _finite_chemistry_scalar(state.T, f"{label} temperature", positive=True)
        pressure = _finite_chemistry_scalar(state.P, f"{label} pressure", positive=True)
        saturation = _finite_chemistry_scalar(state.S, f"{label} saturation ratio")
        if saturation < 0.0:
            raise ValueError("Gas chemistry saturation ratio must be nonnegative")
        validated_states.append((temperature, pressure, saturation))

    names_0, molar_0, concs_0 = _chemistry_gas_inputs(ParcelState_0, "reference")
    names_next, molar_next, concs_next = _chemistry_gas_inputs(ParcelState_Next, "updated")
    if names_0 != names_next or not np.array_equal(molar_0, molar_next):
        raise ValueError("Gas chemistry species order and molar masses must match")

    # Use the normalized scalars, not the original objects: es() is a Numba
    # function and cannot accept every NumPy extended-precision scalar type.
    reference_T, reference_P, _ = validated_states[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        air_molar_conc = reference_P / (c.R * reference_T)
        mass_0 = 1e-9 * concs_0 * air_molar_conc * molar_0
        mass_next = 1e-9 * concs_next * air_molar_conc * molar_next
        water_concs = np.array([
            H2O_gas_conc(saturation, reference_T, reference_P)
            for _, _, saturation in validated_states
        ])
        water_masses = water_concs * c.Mw
    if (not np.isfinite(air_molar_conc) or air_molar_conc <= 0.0
            or not np.all(np.isfinite(mass_0))
            or not np.all(np.isfinite(mass_next))
            or not np.all(np.isfinite(water_masses))):
        raise ValueError("Gas chemistry masses and reference concentration must remain finite")
    if np.any(water_concs >= air_molar_conc):
        raise ValueError("Gas chemistry water vapor pressure must be below total pressure")

    # Difference species before summing so unchanged background mass does not
    # erase trace changes. Both states use the immediate chemistry input T/P.
    with np.errstate(over="ignore", invalid="ignore"):
        gas_changes = mass_next - mass_0
        water_change = water_masses[1] - water_masses[0]
        mass_change = np.sum(gas_changes) + water_change
        changed_mass = np.sum(np.abs(gas_changes)) + abs(water_change)
        tolerance = max(1e-15, 1e-6 * changed_mass)
    if not np.all(np.isfinite([mass_change, changed_mass, tolerance])):
        raise ValueError("Gas chemistry mass change and tolerance must remain finite")
    if abs(mass_change) > tolerance:
        raise ValueError("Mass not conserved in gas chemistry module!")
