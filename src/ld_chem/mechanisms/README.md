# Chemical Mechanisms

This directory contains default chemical reaction mechanism definition files for the aqueous and gas phases in LD-Chem simulations. These simple text-based files define reactions, rate constants, and thermodynamic parameters.

## Contents

### `aq_reactions.dat` - Aqueous-Phase Reaction Mechanisms

Defines all aqueous-phase chemical reactions available for simulation.

**Columns:**

| Column | Description | Example | Notes |
|--------|-------------|---------|-------|
| `reactants` | Comma-separated list of reactant species (no spaces) | `SO2`, `S(IV),H2O2` | Must match species in `aero_data.dat` and `gas_data.dat` |
| `products` | Comma-separated list of product species (no spaces) | `HSO3,H+`, `S(VI)` | Space-separated when multiple products |
| `rate` | Pre-exponential rate constant | `5.0E5`, `-999` | Units: mol/m³^(1-n)/s; `-999` indicates special handling |
| `-Ea/R` | Negative activation energy divided by R | `1960`, `0` | Units: K; temperature dependence parameter |
| `group` | Reaction classification/mechanism group | `sulfate`, `IEPOX` | Used to select subsets of reactions |

**Temperature-Dependent Rate:**

$$k(T) = k_0 \cdot \exp\left(\frac{-E_a}{R}\left(\frac{1}{T} - \frac{1}{298}\right)\right)$$

**Default Reaction Groups:**
- `sulfate` - S(IV) oxidation and equilibria of sulfur-containing species
- `ammonium` - Ammonia uptake and equilibria  
- `nitrate` - Nitrogen oxide chemistry and equilibria
- `IEPOX` - Isoprene epoxide-derived secondary organic aerosol (SOA)

### `gas_reactions.dat` - Gas-Phase Reaction Mechanism

Defines all gas-phase chemical reactions.

**Columns:**

| Column | Description | Example | Notes |
|--------|-------------|---------|-------|
| `reactants` | Comma-separated list of reactant species (no spaces) | `H2O2`, `O1D,H2O` | Species from `gas_data.dat` |
| `products` | Comma-separated list of product species (no spaces) | `OH,OH`, `O3P,N2` | Space-separated when multiple |
| `rate` | Pre-exponential rate constant | `7.66e-6`, `2.18E2` | Units: (molec/cm³)^(1-n)/s |
| `high_P_limit` | High-pressure limit rate | `0`, `1.20E6` | For Troe reactions; 0 for others |
| `T_dependence` | negative activation energy divided by R for `exp` form; power-law exponent for `power` or `troe` forms | `0`, `-940` | K for `exp` form; dimensionless for `power` |
| `form` | Temperature dependence functional form | `exp`, `power`, `troe` | See reaction rate forms below |

**Temperature-Dependent Rate Forms:**

1. **`exp`** - Arrhenius-like (exponential temperature dependence)
   $$k(T) = k_0 \cdot \exp\left(\frac{-E_a}{RT}\right)$$


2. **`power`** - Power-law temperature dependence
   $$k(T) = k_0 \cdot \left(\frac{T}{300}\right)^n$$
   
3. **`troe`** - Troe three-body recombination fall-off
   - Used for pressure-dependent termolecular reactions
   - Accounts for N₂ and H₂O collision broadening
   - `high_P_limit` specifies $k_\infty$

4. **`HO2_water_enhancement`** - Water-enhanced HO₂ self-reaction, accounting for enhanced collision rates with water vapor

## Usage

### Loading Mechanisms in Simulations

**Option 1: Use default mechanisms**

```python
from ld_chem import make_AqReactions, make_GasReactions

# Load all default aqueous reactions
aq_reactions = make_AqReactions()

# Load specific reaction groups
aq_reactions = make_AqReactions(chemistry=['sulfate', 'IEPOX'])

# Load all default gas reactions
gas_reactions = make_GasReactions()
```

**Option 2: Use custom mechanism path**

```python
from ld_chem import simulate_parcel

simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    mechanism_data_path='/path/to/custom/mechanisms/',
    aq_chemistry=['sulfate'],
    gas_chemistry=True
)
```

### Creating Custom Mechanisms

**To add a new reaction:**

1. Add a line to `aq_reactions.dat` or `gas_reactions.dat`.
2. Assign aqueous reactions to a group name (existing or new). Gas reactions do not have a group column.
3. Ensure all reactants and products exist in species data files.
4. Reference aqueous groups through `aq_chemistry`. Setting `gas_chemistry=True` loads all rows from `gas_reactions.dat`.

Gas reaction selection is file-level rather than group-level. The
`make_GasReactions()` loader therefore does not support gas reaction group
selection: calling it with a non-`None` `chemistry` argument raises
`NotImplementedError`. To use a different set of gas reactions, provide a
different `gas_reactions.dat` through `mechanism_data_path`.

**Example: Add new aqueous reaction**

```
# Add to aq_reactions.dat
HNO3                NO3,H+                         5.0e5                   0            nitrate
```

Then load it:
```python
aq_reactions = make_AqReactions(chemistry=['nitrate'])
```

**Example: Add new gas reaction**

```
# Add to gas_reactions.dat
CO,OH                CO2,HO2            9.03E4                      0                    0                      exp
```

### Creating a Custom Mechanism File

To maintain separate mechanism sets without modifying defaults:

1. Copy `aq_reactions.dat` and/or `gas_reactions.dat` to a new directory
2. Edit your copies with custom reactions
3. Pass the directory path via `mechanism_data_path`:

```python
simulate_parcel(
    ...,
    mechanism_data_path='/path/to/my_mechanisms/',
    aq_chemistry=['sulfate', 'custom_group'],
    gas_chemistry=True
)
```

## Format Guidelines

### Creating New Reaction Files

- **Header row:** Start with `#` describing columns
- **One reaction per line:** Whitespace-separated columns
- **Whitespace:** Multiple spaces/tabs are interchangeable
- **Scientific notation:** Use format `1.5E3` or `1.5e3` or `1.5E+03` or `1.5d3`
- **Groups:** Choose meaningful names; hyphenated for multi-word groups

### Reaction Definition Rules

**Aqueous reactions:**
- Reactants and products: Comma-separated, no spaces around commas
- Rate units must be consistent with reaction order (n = number of reactants)
  - Unimolecular: mol/m³·s
  - Bimolecular: mol/m⁶·s
  - Termolecular: mol/m⁹·s
- `-Ea/R` must be numeric (not `-999`), except for special reactions with custom rate laws

**Gas reactions:**
- Comma-separated reactants/products; single spaces between multiple products
- `M` indicates third body (N₂, O₂, or other collision partner)
- High-pressure limit: 0 for non-Troe reactions
- Temperature dependence: 0 if not applicable
- Form must be one of: `exp`, `power`, `troe`, `HO2_water_enhancement`

## Special Reaction Rate Handling

Some reactions (marked with `-999` rate) use custom rate law functions:

**Aqueous phase:**
- `S(IV),O3` - O₃-sulfur oxidation with pH dependence
- `S(IV),H2O2` - H₂O₂-sulfur oxidation with pH dependence
- `S(IV),NO2` - NO₂-sulfur oxidation with pH/water dependence
- `S(IV),HNO2` - HNO₂-sulfur oxidation with pH dependence
- `S(IV),O2` - O₂-sulfur oxidation with catalysis

These are handled by specific functions in `aqueous_chemistry.py` that account for intermediate speciation and pH effects.

## References

Please refer to (xxx) for a complete list of references for the model formulation.
