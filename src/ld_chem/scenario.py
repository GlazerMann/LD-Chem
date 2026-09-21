""" scenario - types and functions supporting the creation of single
simulation scenarios.

@author: Laura Fierce
"""

from dataclasses import dataclass
import numpy as np
from .reactions import make_AqReactions, make_GasReactions
from .particles import AerosolSpecies, retrieve_one_species
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


def _prepare_initial_species(
        species_names, species_masses, aero_species, specdata_path):
    """Validate and snapshot part2pop density/kappa for initial species.

    ``species_masses`` is column-oriented, so ``aero_species`` must contain one
    uniquely named definition per input mass column, in exactly the same order
    and with exactly the same spelling as ``species_names``. Names may be a 1-D
    sequence or a single-row/single-column 2-D array; genuinely two-dimensional
    name grids are rejected instead of being silently flattened.

    LD-Chem cannot verify provenance: callers must pass species definitions from
    the same population that produced ``species_masses``. For particulate
    initial species, only density and kappa are taken from the supplied objects.
    Those are the properties that determine the dry volume and effective kappa
    of a part2pop particle. LD-Chem keeps its own molar mass so this handoff
    does not silently change aqueous-chemistry or gas/particle conversion
    semantics. LD-Chem also retains its existing surface-tension default
    (0.072 N/m); the part2pop per-species surface-tension value is not imported
    because part2pop's current particle implementation does not consistently
    consume it.

    Supplying definitions does not expand LD-Chem's accepted species namespace.
    Every supplied initial name must still resolve through the selected LD-Chem
    species-data table and use the exact spelling expected by enabled chemistry.
    H2O and LD-Chem species with zero reference density keep their complete
    LD-Chem definitions. Both scenario constructors still re-equilibrate H2O at
    their initial saturation ratio and temperature, so the handoff preserves
    dry-particle interpretation rather than an arbitrary incoming wet state.

    Caller object identity, subclasses, methods, and unrelated attributes are
    deliberately not imported into model state.
    """
    # Leave existing callers untouched when the explicit handoff is not used.
    if aero_species is None:
        return species_names, species_masses, {}

    name_array = np.asarray(species_names)
    if name_array.ndim == 1:
        normalized_names = name_array
    # Existing LD-Chem callers commonly use (1, N) name arrays. Accept a
    # singleton axis, but never flatten a genuine 2-D species grid because
    # that could silently rebind definitions to the wrong mass columns.
    elif name_array.ndim == 2 and 1 in name_array.shape:
        normalized_names = name_array.reshape(-1)
    else:
        raise ValueError(
            "species_names must be 1-D or a single-row/single-column 2-D array "
            "when aero_species is provided"
        )

    try:
        mass_array = np.asarray(species_masses)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "species_masses must be a rectangular 2-D particle-by-species array "
            "when aero_species is provided"
        ) from exc
    if mass_array.ndim != 2:
        raise ValueError(
            "species_masses must be a 2-D particle-by-species array when "
            "aero_species is provided"
        )
    if mass_array.shape[1] != len(normalized_names):
        raise ValueError(
            "species_names and aero_species must describe exactly one entry per "
            "species_masses column"
        )

    supplied_species = tuple(aero_species)
    if len(supplied_species) != len(normalized_names):
        raise ValueError(
            "aero_species must contain exactly one species object per "
            "species_names entry"
        )

    expected_names = [str(name) for name in normalized_names]
    if len(set(expected_names)) != len(expected_names):
        raise ValueError(
            "species_names must be unique when aero_species is provided; "
            "duplicate names cannot be mapped unambiguously to mass columns"
        )

    for index, species in enumerate(supplied_species):
        if getattr(species, "name", None) is None:
            raise TypeError(
                f"aero_species[{index}] is missing required attributes: name"
            )

    supplied_names = [str(species.name) for species in supplied_species]
    if supplied_names != expected_names:
        raise ValueError(
            "aero_species names must exactly match species_names in the same order"
        )

    # Validate names with the same LD-Chem table used by the original path.
    # Translate the historical missing-row UnboundLocalError; hardened loaders
    # can raise ValueError directly and that error should pass through unchanged.
    reference_species = {}
    for name in supplied_names:
        try:
            reference_species[name] = retrieve_one_species(
                name, specdata_path=specdata_path
            )
        except UnboundLocalError as exc:
            raise ValueError(
                f"supplied initial species {name!r} is not defined in "
                f"LD-Chem species data at {specdata_path!r}"
            ) from exc

    snapshots = {}
    for index, source in enumerate(supplied_species):
        name = supplied_names[index]
        reference = reference_species[name]

        # Water is immediately re-equilibrated, and zero-density species encode
        # dissolved/gas semantics in LD-Chem. Keep those definitions entirely
        # local instead of importing part2pop values that LD-Chem would not use
        # consistently.
        if name == "H2O" or reference.density == 0:
            snapshots[name] = reference
            continue

        missing = [
            attribute
            for attribute in ("density", "kappa")
            if getattr(source, attribute, None) is None
        ]
        if missing:
            raise TypeError(
                f"aero_species[{index}] is missing required attributes: "
                + ", ".join(missing)
            )

        converted = {}
        for attribute in ("density", "kappa"):
            value = getattr(source, attribute)
            # Require true scalars so one-element arrays cannot be silently
            # coerced and mistaken for a valid species definition.
            try:
                value_array = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"aero_species[{index}] {attribute} must be a scalar "
                    "numeric value"
                ) from exc
            if value_array.ndim != 0 or value_array.dtype.kind not in "iuf":
                raise TypeError(
                    f"aero_species[{index}] {attribute} must be a scalar "
                    "numeric value"
                )
            try:
                converted[attribute] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"aero_species[{index}] {attribute} must be a scalar "
                    "numeric value"
                ) from exc

        density = converted["density"]
        kappa = converted["kappa"]
        if not np.isfinite([density, kappa]).all():
            raise ValueError(
                f"aero_species[{index}] density and kappa must be finite"
            )
        if density <= 0:
            raise ValueError(
                f"aero_species[{index}] density must be > 0 for particulate "
                f"LD-Chem species {name!r}"
            )
        if kappa < 0:
            raise ValueError(
                f"aero_species[{index}] kappa must be >= 0 for particulate "
                f"LD-Chem species {name!r}"
            )

        snapshots[name] = AerosolSpecies(
            name=reference.name,
            density=density,
            kappa=kappa,
            molar_mass=reference.molar_mass,
            surface_tension=reference.surface_tension,
        )

    return normalized_names, mass_array, snapshots


def create_parcel_scenario(
        num_concs = np.array([1.0e6]), pHs=np.array([7.0]),
        species_names=np.array(['NaCl']), species_masses=np.array([2.4e-25]),
        updraft_velocity=1.0, S0=0.85, P0=101325, T0=298,
        z_start=0.0,z_end=1000, gas_names=None, gas_concs=None, 
        dt=1.0, specdata_path='species_data/',
        mechanism_data_path='mechamisms/', aq_chemistry=None, 
        cocondensation=False, gas_chemistry=False, aero_species=None):
    """Create the initial state for an adiabatic parcel simulation.

    When provided, ``aero_species`` must contain one species definition per
    ``species_masses`` column, with unique names matching ``species_names`` in
    exact order and spelling. ``species_names`` may be 1-D or have one singleton
    axis; ``species_masses`` must be a rectangular 2-D particle-by-species
    array. LD-Chem imports particulate density and kappa only. Molar mass
    remains LD-Chem-owned, surface tension uses LD-Chem's existing 0.072 N/m
    default, and H2O and zero-density species retain their complete LD-Chem
    definitions.
    """

    species_names, species_masses, initial_species = _prepare_initial_species(
        species_names, species_masses, aero_species, specdata_path
    )

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
    # Preserve definitions for original mass columns. Species appended above by
    # LD-Chem chemistry/cocondensation are absent from initial_species and must
    # continue to come from LD-Chem's local species table.
    for spec_name in species_names:
        spec_name = str(spec_name)
        if spec_name in initial_species:
            aero_specs.append(initial_species[spec_name])
        else:
            aero_specs.append(
                retrieve_one_species(spec_name, specdata_path=specdata_path)
            )
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
            pHs=np.array([7.0]),species_names=np.array(['NaCl']),
            species_masses=np.array([2.4e-25]),
            trajectory_data=None,
            specdata_path='species_data/',
            mechanism_data_path='mechanisms/',
            condensation=True, cocondensation=False, 
            aq_chemistry=None, gas_chemistry=None, aero_species=None):
    """Create the initial state for a trajectory-driven LES simulation.

    When provided, ``aero_species`` must contain one species definition per
    ``species_masses`` column, with unique names matching ``species_names`` in
    exact order and spelling. ``species_names`` may be 1-D or have one singleton
    axis; ``species_masses`` must be a rectangular 2-D particle-by-species
    array. LD-Chem imports particulate density and kappa only. Molar mass
    remains LD-Chem-owned, surface tension uses LD-Chem's existing 0.072 N/m
    default, and H2O and zero-density species retain their complete LD-Chem
    definitions.
    """

    species_names, species_masses, initial_species = _prepare_initial_species(
        species_names, species_masses, aero_species, specdata_path
    )
    
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
    # Preserve definitions for original mass columns. Species appended above by
    # LD-Chem chemistry/cocondensation are absent from initial_species and must
    # continue to come from LD-Chem's local species table.
    for spec_name in species_names:
        spec_name = str(spec_name)
        if spec_name in initial_species:
            aero_specs.append(initial_species[spec_name])
        else:
            aero_specs.append(
                retrieve_one_species(spec_name, specdata_path=specdata_path)
            )
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
