""" scenario - types and functions supporting the creation of aerosol particles
and particle populations

@author: Laura Fierce
"""

import sys, warnings
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable
from pathlib import Path
from part2pop.aerosol_particle import compute_Dwet, compute_mass_h2o

warnings.filterwarnings('ignore')

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    density: float
    kappa: float
    molar_mass: float
    surface_tension: float

def retrieve_one_species(name, specdata_path='species_data/', surface_tension=0.072):
    aero_datafile = Path(specdata_path) / "aero_data.dat"
    with aero_datafile.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            fields = line.split()
            if not fields or fields[0] != name:
                continue

            if len(fields) != 5:
                raise ValueError(
                    f"Malformed aerosol species entry for '{name}' in "
                    f"{aero_datafile} at line {line_number}: expected 5 fields, "
                    f"found {len(fields)}."
                )

            name_in_file, density, _ions_in_solution, molar_mass, kappa = fields
            try:
                return AerosolSpecies(
                    name=name_in_file,
                    density=float(density),
                    kappa=float(kappa),
                    molar_mass=float(
                        molar_mass.replace("D", "E").replace("d", "e")
                    ),
                    surface_tension=surface_tension,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in aerosol species '{name}' at "
                    f"{aero_datafile}:{line_number}."
                ) from exc

    raise ValueError(
        f"Unknown aerosol species '{name}': no matching entry found in "
        f"{aero_datafile}. Add it to the species data file or check for "
        f"a typo in aero_spec_names."
    )
'''
def equilibrate_h2o(species_names, species_masses, S, T, P, specdata_path='species_data/', sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
    spec_volumes = np.zeros(species_masses.shape)
    spec_kappas = np.zeros(species_masses.shape)
    for ii, (name) in enumerate(species_names):
        if name!="H2O":
            spec=retrieve_one_species(name, specdata_path=specdata_path)
            spec_volumes[:,ii]=species_masses[:,ii]/spec.density
            spec_kappas[:,ii]=spec.kappa
    particle_kappas = np.average(spec_kappas, weights=spec_volumes, axis=1)
    Ddrys = 2.0*((3.0/(4.0*np.pi))*np.sum(spec_volumes, axis=1))**(1.0/3.0)
    masses_h2o=np.zeros(Ddrys.shape)
    for ii, (Ddry, kappa) in enumerate(zip(Ddrys, particle_kappas)):
        Dwet=compute_Dwet(Ddry, kappa, S, T, sigma_h2o=sigma_h2o, rho_h2o=rho_h2o, MW_h2o=MW_h2o)
        masses_h2o[ii]=compute_mass_h2o(Ddry,Dwet,rho_h2o=rho_h2o)
    idx=int(np.where(species_names=="H2O")[0])
    species_masses[:,idx]=masses_h2o
    return species_masses

def get_particle_concentrations(aerosol_population):
    """Get particle concentrations (mol/m^3) from an aerosol population"""    
    spec_masses = aerosol_population.spec_masses
    species = aerosol_population.species
    water_volumes = spec_masses[:,aerosol_population.get_species_idx("H2O")]/aerosol_population.species[aerosol_population.get_species_idx("H2O")].density # m^3
    concs = np.zeros(spec_masses.shape)
    for ii, spec in enumerate(species):
        concs[:,ii] = (spec_masses[:,ii]/spec.molar_mass)/water_volumes # mol/m^3
    return concs
'''
