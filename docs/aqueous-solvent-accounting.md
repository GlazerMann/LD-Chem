# Aqueous solvent accounting: scientific review gate

The implicit-water correction preserves represented particle mass by assigning
the negative of the net represented mass change to liquid H2O. Its numerical
checks do not establish that the underlying exchange was solvent water.
This convention requires mechanism-level chemistry review before production use.

The HSO3 table entry is 81 g/mol. Consequently,
`SO2 + implicit H2O -> HSO3 + H+` needs an 18 g/mol solvent contribution.
That supported bookkeeping example does not validate unrelated mechanism rows.

Review at least these additional cases:

| Current row or implementation | Unresolved accounting |
| --- | --- |
| `NO2,NO2 -> NO3,H+,H+` | Two N atoms become one; listed species masses total 92 -> 64 g/mol. A 28 g/mol water credit conceals the represented nitrogen loss. |
| `IEPOX,SO4,H+ -> IEPOX_OS,H+` | Table masses total 215 -> 216 g/mol. Confirm the organosulfate definition and whether the proton is consumed or catalytic. |
| `HNO2_sulfur_oxidation_rate()` for `S(IV),HNO2` | The implemented derivative consumes HNO2 and produces sulfur(VI), without returning a nitrogen-containing product. Specify the missing product or external reservoir. |

The special and lumped rows must be reviewed against their derivative functions,
not only their abbreviated mechanism-file products. Review all active rows with
an accepted species elemental-composition and charge registry, and document
intentional reservoirs and lumped products explicitly. Solvent adjustment cannot
repair nitrogen or charge imbalance.

Keep kinetic rate laws and reaction order unchanged until their coefficient units
and the intended mechanism have been confirmed. Do not insert H2O or O2 kinetic
concentration factors solely to make a mass check pass. Do not unconditionally
clamp negative water: decide the acceptable roundoff policy separately.

The cancellation-safe condensation check shared with PR #11 is also required
for the small-water-change LES regressions in PR #23. This numerical dependency
does not resolve the scientific review gate above.
