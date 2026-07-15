# General C2 Sampling Implementation Plan

**Goal:** Build an exact, reusable weighted model sampler for the general C2
fragment accepted by WFOMC's `incremental3` compiler.

**Architecture:** Pin WFOMC as a uv Git dependency and centralize its compiler
policy in one adapter. Re-run only the numeric incremental3 recursion with
compact value traces, lazily materialize exact alias tables during traceback,
and reconstruct local pair models and concrete domain labels at the edge.

**Tech stack:** Python 3.11, uv, WFOMC, python-flint, PySAT/PySDD, pytest.

## Tasks

1. Bootstrap the uv package and assert the pinned WFOMC input contract.
2. Implement exact integer alias sampling and sparse polynomial degree masses.
3. Implement the traceable incremental3 value kernel and conditioned traceback.
4. Sample the global branch/component/root mixture and cardinality budgets.
5. Reconstruct local two-tables, evidence-aware labels, and public structures.
6. Add API/CLI, exhaustive small-domain tests, benchmarks, and documentation.

## Hard gates

- Trace root mass equals the pinned WFOMC count exactly.
- Pair-sampler conditioned masses equal WFOMC `PairFactor` projections exactly.
- Every sample emitted by the semantic test corpus is checked with an
  independent property-specific oracle for counting, evidence, cardinality,
  nullary definitions, and linear order.
- Warm sampling is `O(n^2)` and samples stream instead of accumulating in RAM.
