""" systems - types and functions used to create different model systems. 

@author: Laura Fierce
"""

import numpy as np
from .utilities import check_water_condensation, check_gas_condensation, check_gas_chemistry, check_mass_balance
from dataclasses import dataclass
from numba.typed import Dict
from numba import types
from .processes import water_uptake, air_thermo, cocondensation, gas_chemistry, aqueous_chemistry
from .processes.air_thermo import S_to_wv, H2O_mole_fraction #, compute_thermo_props, wv_to_S, 
from .processes.cocondensation import GasFeedback
import ld_chem.constants as c
from scipy.integrate import ode
from typing import Tuple

@dataclass
class Processes:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    condensation: bool = True
    cocondensation: bool = False
    aq_chemistry: bool = False
    gas_chemistry: bool = False

@dataclass
class Feedbacks:
    # Historical names: these fields store water-mass changes accumulated over
    # one model timestep (kg/m^3), not time derivatives. update_air() converts
    # dwc_dt to kg/m^3/s before passing it to air_thermo.dstate_dt(), whose
    # output is an instantaneous derivative.
    dwc_dt: float = 0.
    # Corresponding vapor-water mass change over the same timestep (kg/m^3).
    dwv_dt: float = 0.
    # dwc_dt_next: float = 0.
    # dwc: float = 0. # change in water mass, kg water / m^3 air
    # dwi: float = 0. # change in ice mass, kg ice / m^3 air
    gases: Tuple[GasFeedback, ...] = None  # change trace gases, ppb

def update_state(t1, t2, ParcelState_0, processes, dt,
        accom=1.,radius_scale='lin', mechanism_data_path='../mechanisms/', 
        aq_reactions=None, gas_reactions=None, rtol=1e-10, atol=1e-10):
    
    ParcelState_1, feedbacks_1 = update_particle_population(
        ParcelState_0, processes, dt, 
        radius_scale=radius_scale,
        accom=accom, mechanism_data_path=mechanism_data_path,
        aq_reactions=aq_reactions, rtol=rtol, atol=atol)
    
    ParcelState_2 = update_air(t2,
        ParcelState_1, processes, feedbacks_1, dt, 
        gas_reactions=gas_reactions, rtol=rtol, atol=atol)
    
    return ParcelState_2

def update_particle_population(
        ParcelState_0, processes, dt, 
        radius_scale='lin', accom=1.0,
        mechanism_data_path='mechanisms/', 
        aq_reactions=None, rtol=1e-10, atol=1e-10): 
    
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?
    wv0 = S_to_wv(S0,T0,P0)
    dwc_dt = 0.
    ParcelState_Next = ParcelState_0.clone_detached()

    # water condensation   
    if processes.condensation:
        r_drys=0.5*ParcelState_0.particles.get_particle_var('dry_diameter')
        tkappas=ParcelState_0.particles.get_particle_var('tkappa')
        r0s=0.5*ParcelState_0.particles.get_particle_var('wet_diameter')
        if radius_scale == 'log':
            lnr0s=np.log(r0s)
            rhs=lambda t, lnr: water_uptake.dlnr_dt(lnr, r_drys, tkappas, P0, T0, S0, accom)  
            ode15s=ode(rhs).set_integrator('lsoda', method='bdf', rtol=1E-6, atol=1E-12, nsteps=5000)
            ode15s.set_initial_value(lnr0s, 0.0)
            r_nexts=np.exp(ode15s.integrate(ode15s.t+dt))
        elif radius_scale == 'lin':
            rhs=lambda t, r: water_uptake.dr_dt(r, r_drys, tkappas, P0, T0, S0, accom=accom)
            ode15s=ode(rhs).set_integrator('lsoda', method='bdf', rtol=1E-6, atol=1E-12, nsteps=5000)
            ode15s.set_initial_value(r0s, 0.0)
            r_nexts=ode15s.integrate(ode15s.t+dt)        
        water_idx=ParcelState_0.particles.get_species_idx("H2O")
        water_volumes=(4.0/3.0)*np.pi*(r_nexts**3-r_drys**3)
        ParcelState_Next.particles.spec_masses[:,water_idx]=ParcelState_0.particles.species[water_idx].density*water_volumes
        dwc_dt=np.sum(ParcelState_0.particles.num_concs*(ParcelState_Next.particles.spec_masses[:,water_idx] - ParcelState_0.particles.spec_masses[:,water_idx])) # mass of water to particle phase, kg/m^3
    else:
        dwc_dt = 0
    check_water_condensation(ParcelState_0, ParcelState_Next, dwc_dt)

    # gas condensation
    if processes.cocondensation:
        particles_next, gas_feedback=cocondensation.cocondensation_solver(
            ParcelState_Next.particles, 
            ParcelState_Next.gas, 
            P0, T0, S0, dt=dt,
            rtol=rtol, atol=atol)
        ParcelState_Next.particles.spec_masses=particles_next.spec_masses
        gas_feedback = check_gas_condensation(ParcelState_0, ParcelState_Next, gas_feedback)        
    else:
        gas_feedback=None

    # aqueous chemistry
    if processes.aq_chemistry:
        
        # save a copy before chemistry for mass balance later
        ParcelState_PreChem = ParcelState_Next.clone_detached()

        # set up the initial aqueous concentrations
        X0 = np.zeros(ParcelState_Next.particles.spec_masses.shape)
        radii = 0.5*ParcelState_Next.particles.get_particle_var('wet_diameter')
        dry_radii = 0.5*ParcelState_Next.particles.get_particle_var('dry_diameter')
        water_volumes = (4.0/3.0)*np.pi*(radii**3-dry_radii**3) # m^3
        aq_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
        for ii, (species) in enumerate(ParcelState_0.particles.species):
            aq_names[species.name]=ii
            X0[:,ii]=(ParcelState_Next.particles.spec_masses[:,ii]/species.molar_mass)/water_volumes # mol/m^3        
            
        # set up an array of reactants and products that
        # can be passed into njit function
        reactants = Dict.empty(key_type=types.int32, value_type=types.string)
        products = Dict.empty(key_type=types.int32, value_type=types.string)
        rates = np.empty(0)
        for ii, (reaction) in enumerate(aq_reactions.reactions):
            temp=str(reaction.reactants)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            reactants[ii]=temp
            temp=str(reaction.products)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            products[ii]=temp
            rates=np.append(rates, reaction.get_rate(T0)) 
        
        # define function
        rhs = lambda t, X: aqueous_chemistry.dCaq_dt(
            X, reactants, products, rates, aq_names, T0) # mol/m^3*s

        # solve one particle at a time
        X_next = np.zeros(ParcelState_Next.particles.spec_masses.shape)
        ode15s = ode(rhs).set_integrator('lsoda', method='bdf', rtol=rtol, atol=atol, nsteps=5000)
        for ii in range(ParcelState_Next.particles.spec_masses.shape[0]):
            ode15s.set_initial_value(X0[ii], 0.0)
            X_next[ii] = ode15s.integrate(ode15s.t+dt)  # mol/m^3
        
        # adjust the OH concentration based on the pH
        Hplus_concs_next=0.001*X_next[:,np.where(np.array(aq_names)=='H+')[0][0]] # mol/L
        pHs_next=-1.0*np.log10(Hplus_concs_next)
        OH_concs = 10**(-14.0+pHs_next) # mol/L
        X_next[:,np.where(np.array(aq_names)=='OH-')[0][0]]=1000*OH_concs
        
        # convert to masses
        for ii, (species) in enumerate(ParcelState_Next.particles.species):
            molar_mass=species.molar_mass
            ParcelState_Next.particles.spec_masses[:,ii] = X_next[:,ii]*water_volumes*molar_mass # kg
        
        # check mass balance
        check_mass_balance(ParcelState_PreChem, ParcelState_Next)

    feedbacks = Feedbacks(dwc_dt=dwc_dt, dwv_dt=-1.0*dwc_dt, gases=gas_feedback)        
    
    return ParcelState_Next, feedbacks

def update_air(
    t2,
    ParcelState_0,
    processes,
    feedbacks,
    dt,
    gas_reactions=None,
    rtol=1e-10,
    atol=1e-10,
):
    
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S
    z0 = ParcelState_0.z
    wv0 = air_thermo.S_to_wv(S0, T0, P0)
    
    dz_dt = 0
    dT_dt = 0
    dP_dt = 0
    dS_dt = 0
    dwv_dt = 0
    ParcelState_Next = ParcelState_0.clone_detached()

    if processes.condensation:
        state0 = np.array([z0, T0, P0, S0, wv0])

        if ParcelState_0.w:
            if dt <= 0:
                raise ValueError(
                    "dt must be positive when condensation feedback is enabled"
                )

            # feedbacks.dwc_dt is the total condensed-water mass change
            # accumulated over the preceding timestep (kg/m^3).
            # air_thermo.dstate_dt() requires a rate (kg/m^3/s).
            condensed_water_rate = feedbacks.dwc_dt / dt

            rhs = lambda t, state: air_thermo.dstate_dt(
                state,
                ParcelState_0.w,
                condensed_water_rate,
            )

            ode15s = ode(rhs).set_integrator(
                "lsoda",
                method="bdf",
                rtol=rtol,
                atol=atol,
                nsteps=5000,
            )
            ode15s.set_initial_value(state0, 0.0)
            state_next = ode15s.integrate(ode15s.t + dt)

            ParcelState_Next.z = state_next[0]
            ParcelState_Next.T = state_next[1]
            ParcelState_Next.P = state_next[2]
            ParcelState_Next.S = state_next[3]
            ParcelState_Next.wv = state_next[4]

        else:
            # assumes that pressure is constant over time step
            X0 = (S0 * water_uptake.es(T0 - 273.15)) / (c.R * T0)
            X_next = X0 + (feedbacks.dwv_dt / c.Mw)
            Ph2o_next = X_next * c.R * T0
            dT_dt = -1.0 * (c.L / c.Cp) * feedbacks.dwv_dt
            S_next = Ph2o_next / water_uptake.es((T0 + dT_dt) - 273.15)

            ParcelState_Next.T = T0 + dT_dt
            ParcelState_Next.S = S_next
            ParcelState_Next.wv = air_thermo.S_to_wv(
                S_next,
                T0 + dT_dt,
                P0,
            )
    
    if processes.cocondensation:
        if ParcelState_Next.gas:
            for ii in range(len(feedbacks.gases.names)):
                gas = feedbacks.gases.names[ii]
                dppb_dt = feedbacks.gases.dc_dts[ii]
                TraceGas_idx = ParcelState_Next.gas.get_species_idx(gas)
                ParcelState_Next.gas.concs[TraceGas_idx] += dppb_dt
    
    if processes.gas_chemistry:
        P = ParcelState_0.P
        T = ParcelState_0.T
        S = ParcelState_0.S

        # set up the initial gas concentrations
        X0 = (ParcelState_0.gas.concs*1e-9*P)/(c.R*T) # mol/m^3
        gas_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
        for ii, (gas) in enumerate(ParcelState_0.gas.gases):
            gas_names[gas.name]=ii
        X0=np.append(X0, air_thermo.H2O_gas_conc(S,T,P))
        gas_names['H2O']=ii+1
        X0=np.append(X0, P/(c.R*T))
        gas_names['M']=ii+2

        # set up an array of reactants and products that
        # can be passed into njit function
        reactants = Dict.empty(key_type=types.int32, value_type=types.string)
        products = Dict.empty(key_type=types.int32, value_type=types.string)
        rates = np.empty(0)
        for ii, (reaction) in enumerate(gas_reactions.reactions):
            temp=str(reaction.reactants)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            reactants[ii]=temp
            temp=str(reaction.products)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            products[ii]=temp
            rates=np.append(rates, reaction.get_rate(S, T, P))

        # define function    
        rhs = lambda t, X: gas_chemistry.dCgas_dt(X, reactants, products, rates, gas_names, T, P) 
        
        # solve
        ode15s = ode(rhs).set_integrator('lsoda', method='bdf', rtol=rtol, atol=atol, nsteps=5000)
        ode15s.set_initial_value(X0, 0.0)
        X_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3
         
        # convert to ppb
        ParcelState_Next.gas.concs=X_next[:-2]*1e9*((c.R*T)/P)
        ParcelState_Next.S = (X_next[-2]*c.R*T)/water_uptake.es(T-273.15) # S change from chemistry
        
        # do mass balance
        check_gas_chemistry(ParcelState_0, ParcelState_Next)

    return ParcelState_Next


def air_from_les(ParcelState_0, processes, t2, 
                 relaxation_time, dt, driver, 
                 atol=1e-10, rtol=1e-10):
    
    ParcelState_Next=ParcelState_0.clone_detached()
    t0=0.0
    if relaxation_time>0.0:        
        # update the parcel location, temp, and pressure
        ParcelState_Next.z = np.interp(t2, driver.t_data, driver.z_data)
        ParcelState_Next.x = np.interp(t2, driver.t_data, driver.x_data)
        ParcelState_Next.y = np.interp(t2, driver.t_data, driver.y_data)
        ParcelState_Next.P = np.interp(t2, driver.t_data, driver.P_data)
        ParcelState_Next.T = np.interp(t2, driver.t_data, driver.T_data)
        
        # update the saturation ratio
        S_env = np.interp(t2, driver.t_data, driver.S_data)
        X_h2o_env = (S_env*water_uptake.es(ParcelState_Next.T-273.15))/(c.R*ParcelState_Next.T) # mol/m^3
        X0_h2o_parcel = (ParcelState_0.S*water_uptake.es(ParcelState_Next.T-273.15))/(c.R*ParcelState_Next.T) # mol/m^3   
        rhs = lambda t, X_parcel: (1/relaxation_time)*(X_h2o_env-X_parcel)        
        ode15s = ode(rhs).set_integrator('lsoda', method='bdf', rtol=rtol, atol=atol, nsteps=5000)
        ode15s.set_initial_value(X0_h2o_parcel, t0)
        X_h2o_next = ode15s.integrate(ode15s.t+dt) # mol/m^3
        P_h2o_next = X_h2o_next[-1]*c.R*ParcelState_Next.T
        ParcelState_Next.S = P_h2o_next/water_uptake.es(ParcelState_Next.T-273.15)

        # update the gas concentrations
        if ParcelState_0.gas:
            X0 = ParcelState_0.gas.concs
            X_env = np.zeros(X0.shape)
            H2O_x=H2O_mole_fraction(ParcelState_Next.S,ParcelState_Next.T,ParcelState_Next.P)
            if driver.TraceGas_data is not None:
                for ii, (gas) in enumerate(ParcelState_0.gas.gases):
                    if gas.name in driver.TraceGas_data.keys():
                        X_env[ii]=np.interp(t2, driver.t_data, driver.TraceGas_data[gas.name])
                    elif gas.name == 'N2':
                        X_env[ii]=1e9*0.7808*(1-H2O_x)
                    elif gas.name == 'O2':
                        X_env[ii]=1e9*0.2095*(1-H2O_x)
                    else:
                        X_env[ii]=0.0
            else:
                for ii, (gas) in enumerate(ParcelState_0.gas.gases):
                    if gas.name == 'N2':
                        X_env[ii]=1e9*0.7808*(1-H2O_x)
                    elif gas.name == 'O2':
                        X_env[ii]=1e9*0.2095*(1-H2O_x)
                    else:
                        X_env[ii]=0.0


            rhs = lambda t, X_parcel: (1/relaxation_time)*(X_env-X_parcel)
            ode15s = ode(rhs).set_integrator('lsoda', method='bdf', rtol=rtol, atol=atol, nsteps=5000)
            ode15s.set_initial_value(X0, t0)
            X_next = ode15s.integrate(ode15s.t+dt)
            ParcelState_Next.gas.concs = X_next     
    else:
        # transport between background and parcel is instantaneous
        ParcelState_Next.z = np.interp(t2, driver.t_data, driver.z_data)
        ParcelState_Next.x = np.interp(t2, driver.t_data, driver.x_data)
        ParcelState_Next.y = np.interp(t2, driver.t_data, driver.y_data)
        ParcelState_Next.P = np.interp(t2, driver.t_data, driver.P_data)
        ParcelState_Next.S = np.interp(t2, driver.t_data, driver.S_data)
        ParcelState_Next.T = np.interp(t2, driver.t_data, driver.T_data)
                    
        # update the gas concentrations
        if ParcelState_0.gas:
            for ii, (gas) in enumerate(ParcelState_0.gas.gases):
                if gas.name in driver.TraceGas_data.keys():
                    ParcelState_Next.gas.concs[ii] = np.interp(t2, driver.t_data, driver.TraceGas_data[gas.name])
    return ParcelState_Next


