# Local pair sampling performance design

## Decision

Keep WFOMC unchanged. `c2_wms.PairSampler` uses three local paths, in order:

1. restore atoms directly when the projected mask fixes every binary predicate;
2. enumerate complete pair assignments with PySAT after conditioning on the two
   endpoint cells and projected mask;
3. compile an SDD lazily only when conditioned enumeration exceeds its bounded
   model limit.

Every cached distribution is keyed by `(left_cell, right_cell,
projection_mask)`. Its exact total must equal WFOMC's materialized pair mass.
Assignment weights remain in the WFOMC arithmetic context, and degree-specific
alias tables use exact coefficients without floating-point normalization.

## Boundaries

- WFOMC remains the owner of normalization, cells, counting transitions,
  arithmetic, and aggregate pair masses.
- `c2_wms` owns reconstruction of concrete pair atoms.
- The direct path never constructs pair CNF. General paths construct it lazily.
  SDD construction occurs only after conditioned SAT enumeration exceeds the
  fixed limit.
- SAT enumeration blocks only original atom variables, so Tseitin auxiliary
  assignments cannot duplicate a source pair assignment.
- Binary atoms absent from the QF formula are registered as free SAT variables
  and sampled with their exact predicate weights.
- A sampler closes every SDD circuit that was actually constructed.

## Verification

- Direct projected predicates preserve both orientations of the mask.
- General counting bodies satisfy their projected definitions.
- Free binary atoms follow their exact weighted distribution.
- Forced enumeration overflow exercises the lazy SDD fallback.
- `stmu_4_20` and `stmu_5_12` compile and sample without constructing an SDD.
