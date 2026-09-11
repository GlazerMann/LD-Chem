# Lagrangian Droplets with Chemistry package (ld_chem)

The `ld_chem` package is the core module of the Lagrangian Droplets with Chemistry Model, providing a comprehensive framework for simulating aerosol-cloud microphysics and aqueous-phase chemistry.

## Purpose

This module enables detailed process-level simulations of:
- Aerosol to cloud droplet activation
- Condensation and co-condensation of water and semi-volatile organic compounds
- Aqueous-phase and gas-phase chemistry
- Gas-particle mass transfer and equilibrium
- Particle microphysical evolution in adiabatic parcel or large eddy simulation (LES) conditions

## Module Contents

### Core Modules

- **`scenario.py`** - Scenario setup and initialization functions for creating simulation configurations
- **`run.py`** - Main driver functions for executing parcel and LES simulations
- **`systems.py`** - State management and solver integration for updating particle and air properties
- **`particles.py`** - Particle population definition and management
- **`reactions.py`** - Chemical reaction definitions and rate calculations
- **`gases.py`** - Trace gas definition and properties
- **`constants.py`** - Physical and chemical constants
- **`utilities.py`** - Helper functions for mass balance and calculations
- **`write_files.py`** - Output file writing and checkpoint management

### Subdirectories

- **`mechanisms/`** - Mechanism definition files for aqueous and gas-phase chemistry
  - `aq_reactions.dat` - Default aqueous-phase reaction definitions
  - `gas_reactions.dat` - Default gas-phase reaction definitions
  
- **`processes/`** - Differential equation definitions for physical and chemical processes
  - `air_thermo.py` - Thermodynamic calculations for air parcel
  - `gas_chemistry.py` - Gas-phase chemistry rate equations
  - `aqueous_chemistry.py` - Aqueous-phase chemistry rate equations
  - `cocondensation.py` - Gas-aqueous mass transfer rate equations
  - `water_uptake.py` - Water vaopr condensation rate equations

- **`species_data/`** - Configuration data for aerosol and gas species properties

## Exports

The following are the primary public interfaces of the `ld_chem` module:

### Scenario Creation
- **`create_les_scenario()`** - Create a simulation scenario for trajectory-driven mode
- **`create_parcel_scenario()`** - Create a simulation scenario for parcel mode
- **`make_AqReactions()`** - Load and configure aqueous-phase chemistry mechanisms
- **`make_GasReactions()`** - Load and configure gas-phase chemistry mechanisms

### Core Classes
- **`ParcelState`** - State representation of an particle-containing air parcel and its thermodynamic properties
- **`Processes`** - Configuration object specifying which physical and chemical processes to simulate
- **`ParticlePopulation`** - Representation of the aerosol population

## Usage Example

```python
from ld_chem import create_les_scenario

# Create a simulation scenario
element, driver, aq_reactions, gas_reactions = create_les_scenario(
    num_concs=num_concs,
    pHs=pHs,
    species_names=species_names,
    species_masses=species_masses,
    trajectory_data=trajectory_data,
    mechanism_data_path="src/ld_chem/mechanisms/",
    aq_chemistry=["sulfate"],
    gas_chemistry=True,
)
```

`aq_chemistry` and `gas_chemistry` are scenario configuration arguments, not
pre-built reaction objects:

- `aq_chemistry` is an iterable of aqueous reaction group names, such as
  `["sulfate"]`. The scenario loader passes those names to
  `make_AqReactions()`.
- `gas_chemistry` enables gas-phase chemistry when truthy. Gas reaction files
  do not contain a group column, so the scenario loader loads all rows from the
  selected `gas_reactions.dat`.

The reaction loaders can also be used directly when a caller needs the parsed
reaction definitions:

```python
from ld_chem import make_AqReactions, make_GasReactions

mechanism_data_path = "src/ld_chem/mechanisms/"

aq_reactions = make_AqReactions(
    chemistry=["sulfate"],
    mechanism_data_path=mechanism_data_path,
)
gas_reactions = make_GasReactions(
    mechanism_data_path=mechanism_data_path,
)
```

Passing a non-`None` `chemistry` argument directly to `make_GasReactions()` is
not supported and raises `NotImplementedError`. Use `make_GasReactions()`
without a chemistry selector to load all gas-phase reactions from the selected
mechanism directory.

The default loader path, `"mechanisms/"`, is resolved relative to the current
working directory. When running from the repository root, use
`"src/ld_chem/mechanisms/"` explicitly, as shown above. Installed applications
should pass the directory that contains their selected `aq_reactions.dat` and
`gas_reactions.dat` files rather than relying on the current working directory.

## Documentation

For detailed usage information, custom mechanism definitions, and API documentation, see the main [README.md](../../README.md) in the repository root.
