# Gas chemistry: concentration basis, state order, and reservoirs

Gas chemistry converts tracked ppb using total moist-air molar concentration
`P/(R T)`. The input state's T and P are held fixed during the chemistry substep.
Confirm that external trajectory concentrations use this same basis.

Water has one authoritative representation: saturation ratio S. The solver
derives its H2O concentration from S/T/P and derives the total-air collider M
from T/P. Explicit tracked H2O or M and duplicate tracked names are rejected.
Supply parcel water through S0 and LES water through the trajectory's `s`
values; remove H2O entries from gas_names or trajectory gas mappings. The code
does not silently reconcile conflicting humidity and explicit water inputs.

Coupled-process integration must apply gas chemistry to the output of air
thermodynamics and gas/particle feedback. The chemistry-only mass comparison
must use the snapshot immediately preceding gas chemistry. Comparing with the
outer step's initial state can incorrectly charge condensation or heating to
the gas chemistry budget. Validate the complete process order together.

The mass checker rejects nonfinite values, invalid thermodynamic domains,
inconsistent species/concentration dimensions, mismatched species order or
molar masses, reserved names, and numerical overflow. It differences species
before summing and retains the tolerance
`max(1e-15 kg/m^3, 1e-6 * sum(abs(component mass changes)))`.
It checks represented total mass, not elemental or charge conservation.

## Unresolved oxygen accounting

The shipped rows `CO,OH -> CO2,HO2` and `ISOP,OH -> ISOPOO` each gain 32 g/mol
using the current species table, consistent with implicit O2 participation.
The strict closed-system check rejects this unaccounted mass. An isolated
one-second CO/OH calculation with CO=100 ppb and OH=1 ppb reproduces that error.

Keep the conservation PR draft until chemistry reviewers decide whether O2 is
explicitly consumed or is an external buffered reservoir, and specify the
corresponding budget. Do not add O2 to kinetic reactants without checking the
meaning and units of the effective rate coefficient. No rate or stoichiometry
change, reservoir exemption, or looser tolerance is introduced here.

Synthetic balanced HO2/OH tests validate arithmetic. They cannot validate the
full shipped mechanism. Obtain reviewed calculations for real mechanisms and
complete coupled trajectories before scientific acceptance.
