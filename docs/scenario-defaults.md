# Scenario default composition and size

The SO4/H2O scenario defaults contain 2.4e-25 kg sulfate per represented
particle, with a two-dimensional particle-by-species mass array. At the
shipped SO4 density of 1800 kg/m^3, the spherical dry diameter is
`(6*m/(pi*rho))^(1/3) = 0.6338406 nm`. Using the table molar mass of
96 g/mol, this is about 1.5055 sulfate formula units per represented particle.

The intended numerical use and physical relevance of this retained default
need maintainer confirmation. It is not a validated continuum aerosol example.
Supply masses appropriate to the scientific application explicitly. This
change documents the existing value; it does not infer a unit mistake,
choose a new default size, or modify the mass by an assumed scale factor.

The default species names and array shape are usable inputs, but successful
construction and passing smoke tests do not validate the physical population.
