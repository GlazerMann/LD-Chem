#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the Lagrangian Droplets with Chemistry model.

@author: fier887
"""
from .systems import Processes
from .scenario import create_les_scenario, create_parcel_scenario
import numpy as np
import warnings, time, pickle
from pathlib import Path
from .write_files import write_original, overwrite
from .systems import update_state, air_from_les

def simulate_les_trajectory(
    aero_spec_names, aero_spec_masses, num_concs, pHs, 
    trajectory_data, mechanism_data_path=None, 
    specdata_path=None, dt=1.0, restart_filename='trajectory_restart.pkl',
    output_filename='trajectory_output.pkl', status_filename='trajectory_status',
    progress_filename='RUN_PROGRESS.out', write_every=60.0, print_to_screen=True,
    radius_scale='lin', accom=1.0, 
    condensation=True, cocondensation=False, aq_chemistry=False,
    gas_chemistry=False, relaxation_time=None):

    if not mechanism_data_path:
        if print_to_screen:
            print('WARNING: No mechanism path specified; using default mechanisms in '+str(Path(__file__).resolve().parent)+"/mechanisms/")
        else:
            with open(progress_filename, 'a') as f:
                print('WARNING: No mechanism path specified; using default mechanisms in '+str(Path(__file__).resolve().parent)+"/mechanisms/", file=f)
                f.close()
        warnings.warn('No mechanism path specified; using default mechanisms.', UserWarning)
        mechanism_data_path = str(Path(__file__).resolve().parent)+"/mechanisms/"

    if not specdata_path:
        if print_to_screen:
            print('WARNING: No species data path specified; using default values in '+str(Path(__file__).resolve().parent)+"/species_data/")
        else:
            with open(progress_filename, 'a') as f:
                print('WARNING: No species data path specified; using default values in '+str(Path(__file__).resolve().parent)+"/species_data/", file=f)
                f.close()
        warnings.warn('No species data path specified; using default values.', UserWarning)
        specdata_path = str(Path(__file__).resolve().parent)+"/species_data/"
  
    ParcelState_0, ParcelState_driver, aq_reactions, gas_reactions = create_les_scenario(
        num_concs=num_concs,
        pHs=pHs,species_names=aero_spec_names,
        species_masses=aero_spec_masses,
        trajectory_data=trajectory_data,
        specdata_path=specdata_path,
        mechanism_data_path=mechanism_data_path,
        condensation=condensation, cocondensation=cocondensation, 
        aq_chemistry=aq_chemistry, gas_chemistry=gas_chemistry)   
    
    processes = Processes(
        condensation = condensation, 
        cocondensation = cocondensation, 
        aq_chemistry = aq_chemistry, 
        gas_chemistry = gas_chemistry)

    runtime0 = time.time()
    t_start=ParcelState_driver.t_data[0]
    t_end=ParcelState_driver.t_data[-1]
    Ntimes = int((t_end - t_start)/dt + 1)
    t_eval = np.linspace(t_start, t_end, Ntimes)
    last_written=t_start
    counter=0

    write_original(t_start, ParcelState_0, output_filename, specdata_path=specdata_path)
    f = open(status_filename, 'w')
    f.write('in progress')
    f.close()
    if print_to_screen:
        print('')
        print('Running trajectory', len(ParcelState_0.particles.num_concs),'particles...')
    else:
        with open(progress_filename, 'a') as f:
            print('', file=f)
            print('Running trajectory', len(ParcelState_0.particles.num_concs),'particles...', file=f)
            f.close()
    
    for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
        steptime0 = time.time()

        ParcelState_Next = update_state(t1, t2,
            ParcelState_0, processes, dt,
            radius_scale=radius_scale,
            accom=accom, mechanism_data_path=mechanism_data_path,
            aq_reactions=aq_reactions, gas_reactions=gas_reactions,
            rtol=1e-4, atol=1e-8)     
        
        # get new air state from LES
        ParcelState_Next=air_from_les(
            ParcelState_Next, processes, t2,
            relaxation_time, dt, ParcelState_driver, 
            rtol=1e-4, atol=1e-8)

        # adjust the number concentration based on the new temperature and pressure
        ParcelState_Next.particles.num_concs*=((ParcelState_Next.P*ParcelState_0.T)/(ParcelState_0.P*ParcelState_Next.T))
        
        # check for NaNs
        total_mass = np.sum(ParcelState_Next.particles.num_concs*np.sum(ParcelState_Next.particles.spec_masses, axis=1))

        # kill the program if there is a NaN
        if np.isnan(np.sum(total_mass)):
            with open(status_filename, 'w') as f:
                print('killed (NaNs)', file=f)
                f.close()
            raise ValueError('killed (NaNs)')

        # print timestep and time for timestep
        counter+=1
        if print_to_screen:
            print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')
            print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))
            print()
        else:
            with open(progress_filename, 'a') as f:
                print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it', file=f)
                print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))), file=f)
                print('', file=f)
                f.close()

        # update parcel state
        ParcelState_0=ParcelState_Next
    
        # write backup files
        if t2-last_written>=write_every:
            f = open(status_filename, 'w')
            f.write('in progress')
            f.close()
            overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
            ParcelState_dict = {
                'time': t2, 'parcel state': ParcelState_Next, 'dt': dt, 'accom': accom, 
                'radius_scale': radius_scale, 'specdata_path': specdata_path, 
                'mechanism_data_path': mechanism_data_path, 'processes': processes, 
                'write_every': write_every, 'driver': ParcelState_driver, 'aq_reactions': aq_reactions, 
                'gas_reactions': gas_reactions, 'relaxation_time': relaxation_time}
            pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
            last_written=t2
    
    # write final state
    overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
    f = open(status_filename, 'w')
    f.write('complete')
    f.close()
    
    # show total run time
    if print_to_screen:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
    else:
        with open(progress_filename, 'a') as f:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds', file=f)
            f.close()
    return
    

def simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    z_start=0., z_end=1000., dt=1., updraft_velocity=1.0,
    S0=0.85, P0=101325, T0=298, accom=1.0, radius_scale='log',
    gas_names=None, gas_concs=None, specdata_path=None, 
    restart_filename='trajectory_restart.pkl',
    output_filename='trajectory_output.pkl', status_filename='trajectory_status',
    progress_filename='RUN_PROGRESS.out', write_every=60.0, print_to_screen=True,
    mechanism_data_path=None, condensation = True, 
    cocondensation = False, aq_chemistry = False, gas_chemistry = False):

    if not mechanism_data_path:
        if print_to_screen:
            print('WARNING: No mechanism path specified; using default mechanisms in '+str(Path(__file__).resolve().parent)+"/mechanisms/")
        else:
            with open(progress_filename, 'a') as f:
                print('WARNING: No mechanism path specified; using default mechanisms in '+str(Path(__file__).resolve().parent)+"/mechanisms/", file=f)
                f.close()
        warnings.warn('No mechanism path specified; using default mechanisms.', UserWarning)
        mechanism_data_path = str(Path(__file__).resolve().parent)+"/mechanisms/"

    if not specdata_path:
        if print_to_screen:
            print('WARNING: No species data path specified; using default values in '+str(Path(__file__).resolve().parent)+"/species_data/")
        else:
            with open(progress_filename, 'a') as f:
                print('WARNING: No species data path specified; using default values in '+str(Path(__file__).resolve().parent)+"/species_data/", file=f)
                f.close()
        warnings.warn('No species data path specified; using default values.', UserWarning)
        specdata_path = str(Path(__file__).resolve().parent)+"/species_data/"
    
    ParcelState_0, aq_reactions, gas_reactions = create_parcel_scenario(
        num_concs=num_concs, pHs=pHs,
        species_names=aero_spec_names, species_masses=aero_spec_masses,
        updraft_velocity=updraft_velocity, S0=S0, P0=P0, T0=T0,
        z_start=z_start, z_end=z_end, gas_names=gas_names, gas_concs=gas_concs, 
        dt=dt, specdata_path=specdata_path,
        mechanism_data_path=mechanism_data_path, aq_chemistry=aq_chemistry, 
        cocondensation=cocondensation, gas_chemistry=gas_chemistry)
    
    processes = Processes(
        condensation = condensation, 
        cocondensation = cocondensation, 
        aq_chemistry = aq_chemistry, 
        gas_chemistry = gas_chemistry)

    runtime0 = time.time()
    t_start=0.0
    t_end=(z_end-z_start)/updraft_velocity
    Ntimes = int((t_end - t_start)/dt + 1)
    t_eval = np.linspace(t_start, t_end, Ntimes)
    last_written=t_start
    counter=0
    write_original(t_start, ParcelState_0, output_filename, specdata_path=specdata_path)
    f = open(status_filename, 'w')
    f.write('in progress')
    f.close()
    if print_to_screen:
        print('Running trajectory', len(ParcelState_0.particles.num_concs),'particles...')
    else:
        with open(progress_filename, 'a') as f:
            print('Running trajectory', len(ParcelState_0.particles.num_concs),'particles...', file=f)
            f.close()   
    
    for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):

        steptime0 = time.time()

        ParcelState_Next = update_state(t1, t2,
            ParcelState_0, processes, dt,
            radius_scale=radius_scale,
            accom=accom, mechanism_data_path=mechanism_data_path,
            aq_reactions=aq_reactions, gas_reactions=gas_reactions,
            rtol=1e-4, atol=1e-8)     

        # adjust the number concentration based on the new temperature and pressure
        ParcelState_Next.particles.num_concs*=((ParcelState_Next.P*ParcelState_0.T)/(ParcelState_0.P*ParcelState_Next.T))
        
        # check for NaNs
        total_mass = np.sum(ParcelState_Next.particles.num_concs*np.sum(ParcelState_Next.particles.spec_masses, axis=1))

        # kill the program if there is a NaN
        if np.isnan(np.sum(total_mass)):
            with open(status_filename, 'w') as f:
                print('killed (NaNs)', file=f)
                f.close()
            raise ValueError('killed (NaNs)')

        # print timestep and time for timestep
        counter+=1
        if print_to_screen:
            print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')
            print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))
            print()
        else:
            with open(progress_filename, 'a') as f:
                print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it', file=f)
                print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))), file=f)
                print('', file=f)
                f.close()

        # update parcel state
        ParcelState_0=ParcelState_Next
    
        # write backup files
        if t2-last_written>=write_every:
            f = open(status_filename, 'w')
            f.write('in progress')
            f.close()
            overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
            ParcelState_dict = {
                'time': t2, 'parcel state': ParcelState_Next, 'dt': dt, 
                'z_end': z_end, 'z_start': z_start, 'updraft_velocity': updraft_velocity, 
                'accom': accom, 'radius_scale': radius_scale, 'specdata_path': specdata_path, 
                'mechanism_data_path': mechanism_data_path, 'processes': processes, 
                'write_every': write_every, 'aq_reactions': aq_reactions, 
                'gas_reactions': gas_reactions}
            pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
            last_written=t2
    
    # write final state
    overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
    f = open(status_filename, 'w')
    f.write('complete')
    f.close()
    
    # show total run time
    if print_to_screen:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
    else:
        with open(progress_filename, 'a') as f:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds', file=f)
            f.close()
    return



def restart_trajectory(trajectory_filename='trajectory_output.pkl',
                       print_to_screen=True, progress_filename='RUN_PROGRESS.out', 
                       restart_filename='trajectory_restart.pkl',
                       status_filename='trajectory_status'):
    
    data = pickle.load(open(restart_filename, 'rb'))
    ParcelState_0 = data['parcel state']
    t_start=data['time']
    dt=data['dt']
    accom=data['accom']
    radius_scale=data['radius_scale']
    specdata_path=data['specdata_path']
    mechanism_data_path=data['mechanism_data_path']
    processes=data['processes']
    write_every=data['write_every']
    aq_reactions=data['aq_reactions']
    gas_reactions=data['gas_reactions']
    try:
        ParcelState_driver=data['driver']
        t_end=data['driver'].t_data[-1]
        relaxation_time=data['relaxation_time']
        trajectory_type='les'
    except:
        z_start=data['z_start']
        z_end=data['z_end']
        updraft_velocity=data['updraft_velocity']
        t_end=(z_end-z_start)/updraft_velocity
        trajectory_type='parcel'

    f = open(status_filename, 'w')
    f.write('in progress')
    f.close()
    if print_to_screen:
        print('Restarting '+trajectory_filename+',', len(ParcelState_0.particles.spec_masses),'particles...')
        print('')
    else:
        with open(progress_filename, 'w') as f:
            print('Restarting '+trajectory_filename+',', len(ParcelState_0.particles.spec_masses),'particles...', file=f)
            print('', file=f)

    Ntimes = int((t_end - t_start)/dt + 1)
    t_eval = np.linspace(t_start, t_end, Ntimes)
    last_written=t_start
    runtime0 = time.time()
    counter=0

    for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
        steptime0 = time.time()

        ParcelState_Next = update_state(t1, t2,
            ParcelState_0, processes, dt,
            radius_scale=radius_scale,
            accom=accom, mechanism_data_path=mechanism_data_path,
            aq_reactions=aq_reactions, gas_reactions=gas_reactions,
            rtol=1e-4, atol=1e-8)            
        
        # get new air state from LES
        if trajectory_type=='les':
            ParcelState_Next=air_from_les(
                ParcelState_Next, processes, t2,
                relaxation_time, dt, ParcelState_driver, 
                rtol=1e-4, atol=1e-8)

        # adjust the number concentration based on the new temperature and pressure
        ParcelState_Next.particles.num_concs*=((ParcelState_Next.P*ParcelState_0.T)/(ParcelState_0.P*ParcelState_Next.T))
        
        # check for NaNs
        total_mass = np.sum(ParcelState_Next.particles.num_concs*np.sum(ParcelState_Next.particles.spec_masses, axis=1))

        # kill the program if there is a NaN
        if np.isnan(np.sum(total_mass)):
            with open(status_filename, 'w') as f:
                print('killed (NaNs)', file=f)
                f.close()
            raise ValueError('killed (NaNs)')

        # print timestep and time for timestep
        counter+=1
        if print_to_screen:
            print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')
            print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))
            print()
        else:
            with open(progress_filename, 'a') as f:
                print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it', file=f)
                print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))), file=f)
                print('', file=f)
                f.close()

        # update parcel state
        ParcelState_0=ParcelState_Next
    
        # write backup files
        if t2-last_written>=write_every:
            f = open(status_filename, 'w')
            f.write('in progress')
            f.close()
            overwrite(t2, ParcelState_0, trajectory_filename, specdata_path=specdata_path)
            if trajectory_type=='les':
                ParcelState_dict = {
                    'time': t2, 'parcel state': ParcelState_Next, 'dt': dt, 'accom': accom, 
                    'radius_scale': radius_scale, 'specdata_path': specdata_path, 
                    'mechanism_data_path': mechanism_data_path, 'processes': processes, 
                    'write_every': write_every, 'driver': ParcelState_driver, 'aq_reactions': aq_reactions, 
                    'gas_reactions': gas_reactions, 'relaxation_time': relaxation_time}
            else:
                ParcelState_dict = {
                'time': t2, 'parcel state': ParcelState_Next, 'dt': dt, 
                'z_end': z_end, 'z_start': z_start, 'updraft_velocity': updraft_velocity, 
                'accom': accom, 'radius_scale': radius_scale, 'specdata_path': specdata_path, 
                'mechanism_data_path': mechanism_data_path, 'processes': processes, 
                'write_every': write_every, 'aq_reactions': aq_reactions, 
                'gas_reactions': gas_reactions}
            pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
            last_written=t2
        
    # write final state
    overwrite(t2, ParcelState_0, trajectory_filename, specdata_path=specdata_path)
    f = open(status_filename, 'w')
    f.write('complete')
    f.close()
    
    # show total run time
    if print_to_screen:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
    else:
        with open(progress_filename, 'a') as f:
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds', file=f)
            f.close()
    return
