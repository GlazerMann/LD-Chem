[![CI](https://github.com/pnnl/LD-Chem/actions/workflows/ci.yml/badge.svg)](https://github.com/pnnl/LD-Chem/actions/workflows/ci.yml)

# Lagrangian Droplets with Chemistry Model (LD-Chem)

LD-Chem is a Python package for simulating aqueous chemistry and cloud-aerosol
processes in individual particles. It supports adiabatic parcel simulations and
trajectory-driven simulations using time series of position, saturation ratio,
temperature, pressure, and trace gas concentrations when available.

The LD-Chem model is an extension of the Lagrangian Droplets model:
https://github.com/pnnl/lagrangian-droplets.

## Features

- Adiabatic parcel and trajectory-driven simulation modes
- Particle activation and condensational growth
- Gas-particle mass transfer
- Configurable aqueous-phase and gas-phase chemistry
- Aerosol population definitions through `part2pop`
- Numba-accelerated process calculations

## Installation

LD-Chem can be installed from a local clone of the repository. We recommend using the provided conda environment to create a consistent Python environment before installing the package.

```bash
git clone https://github.com/pnnl/LD-Chem.git
cd LD-Chem

conda env create -f environment.yml
conda activate ld-chem

python -m pip install -e .
python -m pytest
```

The editable install keeps the local source tree connected to the installed package, so updates to the code are available without reinstalling.

## Quick Start

This simple test builds a population with part2pop and runs an adiabatic parcel simulation. It is intended to demonstrate
the package interface with a small synthetic aerosol population, not to
reproduce a full publication simulation.


### Build particle population and run parcel case

```python
import numpy as np
from part2pop.population.builder import build_population

from ld_chem import simulate_parcel

pop_cfg = {
    "type": "binned_lognormals",
    "N": [1e9],
    "GMD": [150e-9],
    "GSD": [1.6],
    "aero_spec_names": [["SO4","OC"]],
    "aero_spec_fracs": [[0.2, 0.8]],
    "N_bins": 20,
    "N_sigmas": 5,
}

pop = build_population(pop_cfg)
aero_spec_names = np.array([species.name for species in pop.species])
aero_spec_masses = np.array(pop.spec_masses)
num_concs = np.array(pop.num_concs)
pHs = np.full(num_concs.shape[0], 4.5)

simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    z_start=0.0, z_end=1000.0, dt=1.0, updraft_velocity=0.5,
    S0=0.85, P0=101325.0, T0=298.0, radius_scale='log',
    gas_names=None, gas_concs=None, condensation = True, 
    cocondensation = False, aq_chemistry = None, 
    gas_chemistry = False, output_filename='trajectory.pkl')

```

### Analyze and plot results
```python
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("trajectory.pkl", "rb") as f:
    data = pickle.load(f)

plt.plot(data['S'], data['z'])
plt.xlim(1.0,)
plt.ylabel('altitude [m]')
plt.xlabel('saturation ratio')
plt.show()

spec_idx = np.where(data['particle species']=='Dwet')[0][0]
plt.plot(data['particles'][:,:,spec_idx], data['z'], '-r')
plt.xscale('log')
plt.xlabel('wet diameter [m]')
plt.ylabel('altitude [m]')
plt.show()

```

Runnable cases are available under `cases/`.

## Repository Structure

```text
src/ld_chem/
    constants.py             # Physical constants
    gases.py                 # Trace gas representation
    particles.py             # Particle species helpers
    reactions.py             # Gas and aqueous reaction definitions
    run.py                   # Simulation entry points
    scenario.py              # Simulation setup
    systems.py               # Solvers and state updates
    utilities.py             # Runtime consistency checks
    write_files.py           # Output writers
    mechanisms/              # Default gas and aqueous reactions
    processes/               # Differential equation definitions
    species_data/            # Aerosol and gas species definitions

cases/
    basic_examples/          # Small instructional cases
    hiscale_20160425/        # HISCALE April 25, 2016 research case scaffold
```

`cases/` contains runnable LD-Chem cases, including small basic examples and larger research workflows.

LD-Chem was designed with flexible gas- and aqueous-phase chemistry definitions. Both aqueous and gas-phase mechanisms are defined in simple data format files, allowing easy customization and extension.

### Aqueous-Phase Reactions

The default aqueous-phase reactions are defined in `src/ld_chem/mechanisms/aq_reactions.dat` with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `reactants` | Comma-separated list of reactant species | - |
| `products` | Comma-separated list of product species | - |
| `rate` | Pre-exponential rate constant | mol/m³^(1-n)/s |
| `-Ea/R` | Negative activation energy divided by gas constant | K |
| `group` | Reaction classification/mechanism name | - |

The temperature-dependent reaction rate is calculated as:

$$k(T) = k_0 \cdot \exp\left(\frac{-E_a}{R}\left(\frac{1}{T} - \frac{1}{298}\right)\right)$$

Default aqueous chemistry groups include:
- `sulfate` - SO₂ and sulfuric acid chemistry
- `ammonium` - Ammonia and ammonium equilibria
- `nitrate` - Nitrogen oxide chemistry
- `IEPOX` - IEPOX secondary organic aerosol reactions

### Gas-Phase Reactions

Gas-phase reactions are defined in `src/ld_chem/mechanisms/gas_reactions.dat` with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `reactants` | Comma-separated list of reactant species | - |
| `products` | Comma-separated list of product species | - |
| `rate` | Pre-exponential rate constant | (molec/cm³)^(1-n)/s |
| `high_P_limit` | High pressure limit (Troe reactions only) | (molec/cm³)^(1-n)/s |
| `T_dependence` | Temperature dependence parameter; equal to $-E_a/R$ for Arrhenius-type and $n$ for power-law type | K or dimensionless |
| `form` | Functional form of temperature dependence | - |

Four temperature-dependent rate forms are supported:

- **`exp`**: Arrhenius-like temperature dependence 

$$k(T) = k_0 \cdot \exp\left(\frac{-E_a}{RT}\right)$$

- **`power`**: Power-law temperature dependence

  $$k(T) = k_0 \cdot \left(\frac{T}{300}\right)^n$$

- **`troe`**: Troe Three-Parameter Fall-off Region (pressure-dependent bimolecular reactions)
  Accounts for N₂/H₂O collision broadening and high-pressure limits

- **`HO2_water_enhancement`**: Water-enhanced HO₂ self-reaction specific to atmospheric conditions

  $$k = (k_1 + k_2) \cdot \left(1 + 8.4 \times 10^{-4} [H_2O] \exp(2200/T)\right)$$

### Custom Mechanisms

There are two ways to define custom aqueous- or gas-phase chemical mechanisms:

**Option 1: Edit the default mechanism files**

Modify the default mechanism files in `src/ld_chem/mechanisms/`:
1. Edit `aq_reactions.dat` and/or `gas_reactions.dat` directly.
2. For aqueous reactions, add or remove rows and assign them to group names.
3. Select aqueous groups with `aq_chemistry`. Gas reactions have no group column; `gas_chemistry=True` enables all rows in the selected `gas_reactions.dat`.

The scenario APIs perform this loading internally. Pass aqueous group names to
`aq_chemistry` and use `gas_chemistry=True` to enable gas chemistry; do not pass
objects returned by `make_AqReactions()` or `make_GasReactions()` to those
scenario arguments.

Direct calls to `make_GasReactions()` do not support a chemistry/group
selector. Supplying a non-`None` `chemistry` argument raises
`NotImplementedError`.

`mechanism_data_path` is interpreted relative to the process working
directory. From a source checkout at the repository root, the bundled mechanism
directory is `src/ld_chem/mechanisms/`. Callers that use the reaction loaders
directly should pass the mechanism directory explicitly instead of assuming
that `"mechanisms/"` refers to the package's bundled data.

**Option 2: Create custom mechanism files and specify the path**

Create your own mechanism files and point the model to them:
1. Create new `aq_reactions.dat` and/or `gas_reactions.dat` files with your custom reactions
2. Pass the directory path to the `mechanism_data_path` argument when running a simulation
3. This allows you to maintain multiple mechanism configurations without modifying the source code. It should be noted that `aq_reactions.dat` and `gas_reactions.dat` are read using the same `mechanism_data_path`.

Example using Option 1:
```python
from ld_chem import simulate_parcel

# define aerosol properties (single organic particle)
aero_spec_names=[['OC']]
aero_spec_masses=[[1.0]]
num_concs=[1e6]
pHs=[3.0]

# Run a parcel simulation with the default sulfate mechanisms
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    aq_chemistry=['sulfate'],
    gas_chemistry=True,
    output_filename='trajectory.pkl'
)
```

Example using Option 2:
```python
from ld_chem import simulate_parcel

# define aerosol properties (single organic particle)
aero_spec_names=[['OC']]
aero_spec_masses=[[1.0]]
num_concs=[1e6]
pHs=[3.0]

# Point to custom mechanism directory
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    mechanism_data_path='/path/to/custom/mechanisms/',
    aq_chemistry=['custom_group_1'],
    gas_chemistry=True,
    output_filename='trajectory.pkl'
)
``` 

## Reproducibility and Data

This repository bundles source code, tests, cases, default mechanisms, and
small species/mechanism data files needed by the package. It does not bundle an unpublished paper-specific dataset.

For a publication release, paper-specific input datasets and large generated
outputs should be archived separately and cited through the manuscript Data
Availability statement. Final DOI or accession values should be added to the
manuscript and release notes only after they are minted by the selected archive.

## Versioning

LD-Chem uses calendar-based release tags of the form `vYYYY.N` for citable
research-software snapshots. These release numbers do not imply
semantic-versioning compatibility guarantees. Changes affecting scientific
results, inputs, outputs, dependencies, or public APIs are documented in the
release notes.

## Citation

See `CITATION.cff` for software citation metadata.

## License

LD-Chem is distributed under the BSD-2-Clause license. See `LICENSE.txt`.
