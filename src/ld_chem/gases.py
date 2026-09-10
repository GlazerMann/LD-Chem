#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:39:14 2024

@author: beel083
"""
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import sys

R = 8.314

@dataclass(frozen=True)
class GasSpecies:
    """GasSpecies: the definition of a gaseous species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    alpha: float
    molar_mass: float
    H0: float
    H_exp: float
    
    def get_Heff(self, T):
        # returns Henry's Law coefficient at given temperature in mol/m^3*Pa
        return (1000/101325)*self.H0*np.exp(self.H_exp*((1/T)-(1/298))) 

@dataclass
class TraceGasPopulation:
    """TraceGasPopulation: the definition of gas composition
    in terms of the amounts of different constituent species """
    gases: Tuple[GasSpecies, ...]
    concs: Tuple[float, ...]
    ids: Tuple[int, ...]
    
    def get_species_idx(self, species):
        idx = None
        for ii in range(len(self.gases)):
            if self.gases[ii].name == species:
                return ii
        return idx
        
    def clone_detached(self):
        """Return a copy that shares immutable data but has detached numeric arrays."""
        return TraceGasPopulation(
            gases=self.gases,                    # shared: typically immutable metadata
            concs=self.concs.copy(),             # detached: new NumPy array
            ids=tuple(self.ids),                 # safe shallow copy
        )  

def retrieve_gas_species(name, specdata_path='../species_data/'):
    gas_datafile = Path(specdata_path) / "gas_data.dat"
    with gas_datafile.open() as data_file:
        for line_number, line in enumerate(data_file, start=1):
            fields = line.split()
            if not fields or fields[0] != name:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"Malformed gas species entry for '{name}' in "
                    f"{gas_datafile} at line {line_number}: expected 5 fields, "
                    f"found {len(fields)}."
                )
            _, alpha, molar_mass, H0, H_exp = fields
            try:
                return GasSpecies(
                    name=name,
                    alpha=float(alpha),
                    molar_mass=float(
                        molar_mass.replace("D", "E").replace("d", "e")
                    ),
                    H0=float(H0.replace("D", "E").replace("d", "e")),
                    H_exp=float(H_exp.replace("D", "E").replace("d", "e")),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in gas species '{name}' at "
                    f"{gas_datafile}:{line_number}."
                ) from exc

    raise ValueError(
        f"Unknown gas species '{name}': no matching entry found in "
        f"{gas_datafile}. Add it to the species data file or check for "
        f"a typo in gas_names.")


def equilibrate_gases(aerosol_population,TraceGas_population,T,P):
    radii = 0.5*aerosol_population.get_particle_var('wet_diameter')
    dry_radii = 0.5*aerosol_population.get_particle_var('dry_diameter')
    water_volumes = (4.0/3.0)*np.pi*(radii**3-dry_radii**3)
    for gas, gas_conc in zip(TraceGas_population.gases, TraceGas_population.concs):
        if gas.name != 'IEPOX':           
            idx = aerosol_population.get_species_idx(gas.name)
            Caq_x = gas_conc*1e-9*P*gas.get_Heff(T)  # mol/m^3
            aerosol_population.spec_masses[:,idx]=water_volumes*Caq_x*aerosol_population.species[idx].molar_mass
    return aerosol_population

def make_TraceGasPopulation(gas_names, gas_conc, specdata_path='species_data/'):
    if len(gas_names)>0:
        gases = []
        concs = []
        ids = []
        for ii in range(0,len(gas_names)):
            OneGas = retrieve_gas_species(gas_names[ii], specdata_path=specdata_path)
            gases.append(OneGas)
            concs.append(gas_conc[ii])
            ids.append(ii) 
        concs = np.array(concs)
    else:
        gases = None
        concs = None
        ids = None
    return TraceGasPopulation(gases=gases, concs=concs, ids=ids)
