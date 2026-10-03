""" scenario - types and functions supporting the creation of single
simulation scenarios.

@author: Laura Fierce
"""

from dataclasses import dataclass
import numpy as np
from .reactions import make_AqReactions, make_GasReactions
from .particles import retrieve_one_species
from part2pop.population import ParticlePopulation
from .gases import TraceGasPopulation, make_TraceGasPopulation
from scipy.optimize import fminbound
from .processes.air_thermo import H2O_mole_fraction
import ld_chem.constants as c
from typing import Optional

@dataclass
class LagrangianElement: 
    particles: ParticlePopulation
    gas: TraceGasPopulation 
    
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    
    u: Optional[float]
    v: Optional[float]
    w: Optional[float]
    
    S: Optional[float]
    P: Optional[float]
    T: Optional[float]

    def get_activated_fraction(self):
        
        def Seq(r, r_dry, T, kappa):
            """ Saturation ratio over the aqueous droplet. From pyrcel. """
            a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
            sigma_w=0.0761 - (1.55e-4) * (T - 273.15)
            Seq = a_w*np.exp((2.0*sigma_w*(18.0 / 1e3))/(8.314*T*1000*r))
            return Seq
                
        T = self.T
        N_active = 0
        radii=0.5*self.particles.get_particle_var('wet_diameter')
        dry_radii=0.5*self.particles.get_particle_var('dry_diameter')
        kappas=self.particles.get_particle_var('tkappa')
        
        for r,r_dry,kappa,num_conc in zip(radii,dry_radii,kappas,self.particles.num_concs):
            neg_Seq = lambda r: -1.0 * Seq(r, r_dry, T, kappa)
            out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
            r_crit, s_crit = out[:2]
            s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
            if r>=r_crit:
                N_active+=num_conc
        
        return N_active/np.sum(self.particles.num_concs)
    
    def clone_detached(self):
        return LagrangianElement(
            x=self.x, y=self.y, z=self.z,
            u=self.u, v=self.v, w=self.w,
            S=self.S, P=self.P, T=self.T,
            particles=self.particles.clone_detached(),
            gas=(
                None if self.gas is None
                else self.gas.clone_detached())
        )
    
@dataclass
class LagrangianElementDriver:
    t_data: Optional[float] = None
    x_data: Optional[float] = None
    y_data: Optional[float] = None
    z_data: Optional[float] = None
    
    u_data: Optional[float] = None
    v_data: Optional[float] = None
    w_data: Optional[float] = None
    
    S_data: Optional[float] = None
    P_data: Optional[float] = None
    T_data: Optional[float] = None
    TraceGas_data: Optional[float] = None


def create_parcel_scenario(
        num_concs = np.array([1.0e6]), pHs=np.array([7.0]),
        species_names=np.array(['SO4', 'H2O']),
        species_masses=np.array([[2.4e-25, 0.0]]),
        updraft_velocity=1.0, S0=0.85, P0=101325, T0=298,
        z_start=0.0,z_end=1000, gas_names=None, gas_concs=None, 
        dt=1.0, specdata_path='species_data/',
        mechanism_data_path='mechamisms/', aq_chemistry=None, 
        cocondensation=False, gas_chemistry=False):

    # load in the gas reactions
    if gas_chemistry:
        gas_reactions = make_GasReactions(mechanism_data_path=mechanism_data_path)
    else:
        gas_reactions = None
    
    # make sure all species involved in gas reactions are included in gas_concs and gas_names
    if cocondensation or gas_chemistry:
        if gas_names:
            gas_names = list(gas_names)
            gas_concs = list(gas_concs)
        else:
            gas_names= []
            gas_concs = []
        if gas_reactions:
            for reaction in gas_reactions.reactions:
                for reactant in reaction.reactants:
                    if reactant not in gas_names and reactant not in ['H2O', 'O2', 'N2', 'M']:
                        gas_names.append(reactant)
                        gas_concs.append(0.0)
                for product in reaction.products:
                    if product not in gas_names and product not in ['H2O', 'O2', 'N2', 'M']:
                        gas_names.append(product)
                        gas_concs.append(0.0)
        H2O_x=H2O_mole_fraction(S0,T0,P0) # mol/m^3
        if 'O2' not in gas_names:
            gas_names.append('O2')
            gas_concs.append(1e9*0.2095*(1-H2O_x))
        if 'N2' not in gas_names:
            gas_names.append('N2')
            gas_concs.append(1e9*0.7808*(1-H2O_x))
        TraceGas_population = make_TraceGasPopulation(gas_names, gas_concs, specdata_path=specdata_path)
    else:
        TraceGas_population = None
    
    # load in the aqueous reactions    
    if aq_chemistry:
        aq_reactions = make_AqReactions(chemistry=aq_chemistry, mechanism_data_path=mechanism_data_path)

        # make sure all species involved in reactions are included in species_names and species_masses
        for reaction in aq_reactions.reactions:
            for reactant in reaction.reactants:
                if reactant not in species_names:
                    if reactant == 'S(IV)':
                        for subreactant in ['SO2', 'HSO3', 'SO3']:
                            if subreactant not in species_names:
                                species_names=np.append(species_names, subreactant)
                                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
                    else:
                        species_names=np.append(species_names, reactant)
                        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
            for product in reaction.products:
                if product not in species_names:
                    if product == 'S(VI)':
                        for subproduct in ['H2SO4', 'HSO4', 'SO4']:
                            if subproduct not in species_names:
                                species_names=np.append(species_names, subproduct)
                                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
                    else:
                            species_names=np.append(species_names, product)
                            species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1)))) 
    else:
        aq_reactions=None   
    
    # make sure that soluble gases are included in species_names and species_masses
    if TraceGas_population and cocondensation:
        for gas in TraceGas_population.gases:
            if gas.name not in species_names and gas.H0 > 0:
                species_names=np.append(species_names, gas.name)
                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))

    # also make sure that H+ and OH- are included in species_names and species_masses
    if 'H+' not in species_names:
        species_names=np.append(species_names, 'H+')
        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
    if 'OH-' not in species_names:
        species_names=np.append(species_names, 'OH-')
        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))

    # check the input shapes
    assert len(num_concs) == len(species_masses)
    assert len(species_names) == species_masses.shape[1]
    assert len(pHs) == len(species_masses)
    
    # turn the species names and masses into particles
    ids = [ii for ii in range(len(species_masses))]
    aero_specs = []
    for spec in species_names:
        aero_specs.append(retrieve_one_species(spec, specdata_path=specdata_path))
    aerosol_population = ParticlePopulation(species=aero_specs, spec_masses=species_masses, num_concs=num_concs, ids=ids)
    
    # equilibrate the particles with water at the initial conditions
    aerosol_population._equilibrate_h2o(S0, T0)

    # set the masses of H+ and OH- based on the input pHs
    water_volumes = 1000*(aerosol_population.get_particle_var("vol_tot") - aerosol_population.get_particle_var("vol_dry")) # L
    Hplus_concs = 10**(-1.0*pHs) # mol/L
    OH_concs = 10**(-14.0+pHs) # mol/L
    aerosol_population.spec_masses[:,aerosol_population.get_species_idx("H+")]=water_volumes*Hplus_concs*aerosol_population.species[aerosol_population.get_species_idx("H+")].molar_mass
    aerosol_population.spec_masses[:,aerosol_population.get_species_idx("OH-")]=water_volumes*OH_concs*aerosol_population.species[aerosol_population.get_species_idx("OH-")].molar_mass    
    
    # equilibrate the sulfate and nitrate systems if needed
    if aq_chemistry:
        concs = aerosol_population.get_particle_var('concentrations')
        if 'sulfate' in aq_chemistry:
            SO4_concs=0.001*concs[:,aerosol_population.get_species_idx('SO4')] # mol/L
            Hplus_concs=0.001*concs[:,aerosol_population.get_species_idx('H+')]
            HSO4_concs=(SO4_concs*Hplus_concs)/0.01
            H2SO4_concs=(HSO4_concs*Hplus_concs)/1000.0
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('HSO4')]=HSO4_concs*aerosol_population.species[aerosol_population.get_species_idx('HSO4')].molar_mass*water_volumes
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('H2SO4')]=H2SO4_concs*aerosol_population.species[aerosol_population.get_species_idx('H2SO4')].molar_mass*water_volumes
        if 'nitrate' in aq_chemistry:
            NO3_concs=0.001*concs[:,aerosol_population.get_species_idx('NO3')] # mol/L
            HNO3_concs=(NO3_concs*Hplus_concs)/15.625
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('HNO3')]=HNO3_concs*aerosol_population.species[aerosol_population.get_species_idx('HNO3')].molar_mass*water_volumes

    element = LagrangianElement(
        particles=aerosol_population,
        gas=TraceGas_population,
        x=None, y=None, z=z_start,
        u=None, v=None, w=updraft_velocity,
        S=S0, P=P0, T=T0)
    
    return element, aq_reactions, gas_reactions


def create_les_scenario(num_concs=np.array([1e6]),
            pHs=np.array([7.0]),species_names=np.array(['SO4', 'H2O']),
            species_masses=np.array([[2.4e-25, 0.0]]),
            trajectory_data=None,
            specdata_path='species_data/',
            mechanism_data_path='mechanisms/',
            condensation=True, cocondensation=False, 
            aq_chemistry=None, gas_chemistry=None):
    
    # load the gas data
    try:
        gas_data = trajectory_data['gas']
    except:
        gas_data = None

    # load in the gas reactions
    if gas_chemistry:
        gas_reactions = make_GasReactions(mechanism_data_path=mechanism_data_path)
    else:
        gas_reactions = None

    # make sure all species involved in gas reactions are included in gas_concs and gas_names
    if cocondensation or gas_chemistry:
        gas_names= []
        gas_concs = []
        if gas_data is not None:
            for gas in gas_data.keys():
                gas_names.append(gas)
                gas_concs.append(gas_data[gas][0]) # initial concentration
        if gas_reactions:
            for reaction in gas_reactions.reactions:
                for reactant in reaction.reactants:
                    if reactant not in gas_names and reactant not in ['H2O', 'O2', 'N2', 'M']:
                        gas_names.append(reactant)
                        gas_concs.append(0.0)
                for product in reaction.products:
                    if product not in gas_names and product not in ['H2O', 'O2', 'N2', 'M']:
                        gas_names.append(product)
                        gas_concs.append(0.0)
        H2O_x=H2O_mole_fraction(trajectory_data['s'][0],trajectory_data['T'][0],trajectory_data['P'][0]) # mol/m^3
        if 'O2' not in gas_names:
            gas_names.append('O2')
            gas_concs.append(1e9*0.2095*(1-H2O_x))
        if 'N2' not in gas_names:
            gas_names.append('N2')
            gas_concs.append(1e9*0.7808*(1-H2O_x))
        TraceGas_population = make_TraceGasPopulation(gas_names, gas_concs, specdata_path=specdata_path)
    else:
        TraceGas_population = None
    
    # load in the aqueous reactions    
    if aq_chemistry:
        aq_reactions = make_AqReactions(chemistry=aq_chemistry, mechanism_data_path=mechanism_data_path)

        # make sure all species involved in reactions are included in species_names and species_masses
        for reaction in aq_reactions.reactions:
            for reactant in reaction.reactants:
                if reactant not in species_names:
                    if reactant == 'S(IV)':
                        for subreactant in ['SO2', 'HSO3', 'SO3']:
                            if subreactant not in species_names:
                                species_names=np.append(species_names, subreactant)
                                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
                    else:
                        species_names=np.append(species_names, reactant)
                        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
            for product in reaction.products:
                if product not in species_names:
                    if product == 'S(VI)':
                        for subproduct in ['H2SO4', 'HSO4', 'SO4']:
                            if subproduct not in species_names:
                                species_names=np.append(species_names, subproduct)
                                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
                    else:
                            species_names=np.append(species_names, product)
                            species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1)))) 
    else:
        aq_reactions=None   

    # make sure that soluble gases are included in species_names and species_masses
    if TraceGas_population and cocondensation:
        for gas in TraceGas_population.gases:
            if gas.name not in species_names and gas.H0 > 0:
                species_names=np.append(species_names, gas.name)
                species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))

    # also make sure that H+ and OH- are included in species_names and species_masses
    if 'H+' not in species_names:
        species_names=np.append(species_names, 'H+')
        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))
    if 'OH-' not in species_names:
        species_names=np.append(species_names, 'OH-')
        species_masses=np.hstack((species_masses, np.zeros((len(num_concs),1))))

    # check the input shapes
    assert len(num_concs) == len(species_masses)
    assert len(species_names) == species_masses.shape[1]
    assert len(pHs) == len(species_masses)

    # turn the species names and masses into particles
    ids = [ii for ii in range(len(species_masses))]
    aero_specs = []
    for spec in species_names:
        aero_specs.append(retrieve_one_species(spec, specdata_path=specdata_path))
    aerosol_population = ParticlePopulation(species=aero_specs, spec_masses=species_masses, num_concs=num_concs, ids=ids)

    # equilibrate the particles with water at the initial conditions
    S0=trajectory_data['s'][0]
    T0=trajectory_data['T'][0]
    P0=trajectory_data['P'][0]
    aerosol_population._equilibrate_h2o(S0, T0)

    # set the masses of H+ and OH- based on the input pHs
    water_volumes = 1000*(aerosol_population.get_particle_var("vol_tot") - aerosol_population.get_particle_var("vol_dry")) # L
    Hplus_concs = 10**(-1.0*pHs) # mol/L
    OH_concs = 10**(-14.0+pHs) # mol/L
    aerosol_population.spec_masses[:,aerosol_population.get_species_idx("H+")]=water_volumes*Hplus_concs*aerosol_population.species[aerosol_population.get_species_idx("H+")].molar_mass
    aerosol_population.spec_masses[:,aerosol_population.get_species_idx("OH-")]=water_volumes*OH_concs*aerosol_population.species[aerosol_population.get_species_idx("OH-")].molar_mass    

    # equilibrate the sulfate and nitrate systems if needed
    if aq_chemistry:
        concs = aerosol_population.get_particle_var('concentrations')
        if 'sulfate' in aq_chemistry:
            SO4_concs=0.001*concs[:,aerosol_population.get_species_idx('SO4')] # mol/L
            Hplus_concs=0.001*concs[:,aerosol_population.get_species_idx('H+')]
            HSO4_concs=(SO4_concs*Hplus_concs)/0.01
            H2SO4_concs=(HSO4_concs*Hplus_concs)/1000.0
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('HSO4')]=HSO4_concs*aerosol_population.species[aerosol_population.get_species_idx('HSO4')].molar_mass*water_volumes
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('H2SO4')]=H2SO4_concs*aerosol_population.species[aerosol_population.get_species_idx('H2SO4')].molar_mass*water_volumes
        if 'nitrate' in aq_chemistry:
            NO3_concs=0.001*concs[:,aerosol_population.get_species_idx('NO3')] # mol/L
            HNO3_concs=(NO3_concs*Hplus_concs)/15.625
            aerosol_population.spec_masses[:,aerosol_population.get_species_idx('HNO3')]=HNO3_concs*aerosol_population.species[aerosol_population.get_species_idx('HNO3')].molar_mass*water_volumes

    element = LagrangianElement(
        particles=aerosol_population,
        gas=TraceGas_population,
        x=trajectory_data['x'][0], y=trajectory_data['y'][0], z=trajectory_data['z'][0],
        u=None, v=None, w=None,
        S=S0, P=P0, T=T0)
    driver = LagrangianElementDriver(
        t_data=trajectory_data['t'], 
        x_data=trajectory_data['x'], y_data=trajectory_data['y'], z_data=trajectory_data['z'],
        u_data=None, v_data=None, w_data=None,
        S_data=trajectory_data['s'], P_data=trajectory_data['P'], T_data=trajectory_data['T'], 
        TraceGas_data=gas_data)
    return element, driver, aq_reactions, gas_reactions
