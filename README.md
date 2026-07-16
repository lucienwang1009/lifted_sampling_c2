# c2-wms

`c2-wms` is an exact lifted weighted model sampler for the general C2 fragment
supported by WFOMC's `incremental3` algorithm. It accepts arbitrary C2 counting
bodies—not only atomic forms such as `exists_=k Y: R(X,Y)`—and supports all
WFOMC row/global count comparators, modulo counts, exact non-negative weights,
cardinality constraints, unary evidence, nullary predicates, and `LEQ`.

WFOMC is pinned by Git commit in both `pyproject.toml` and `uv.lock`:
`481230d668dd34051161f2ca41fa21f2f008af84` from the upstream `devel` branch.

## Setup

The repository is managed entirely with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

### Ganak performance dependency

This repository relies on
[`meelgroup/ganak@82a1d1fb6f0d6fb4a46b825f84b29567728ae483`](https://github.com/meelgroup/ganak/commit/82a1d1fb6f0d6fb4a46b825f84b29567728ae483),
the exact revision pinned by the WFOMC dependency. Ganak is an optional native
executable used by WFOMC while computing pair factors. WFOMC falls back to
PySDD when it is absent, so Ganak is not a hard installation or correctness
dependency for small models. It is, however, a practical performance
dependency for stable-roommates and other complex pair theories: the PySDD
fallback can be extremely slow and memory-intensive.

Install and verify the WFOMC-pinned Ganak build with:

```bash
uv run wfomc-install-ganak
uv run python -c "from wfomc.ganak import find_ganak; print(find_ganak())"
```

Alternatively, put a `ganak` executable on `PATH` or set `GANAK` to its path.
When supplying a binary this way, it must be compatible with the pinned commit;
`wfoms` does not introspect an arbitrary binary to verify its Git revision.
`wfoms` checks once per process and emits a warning naming the required revision
when Ganak is unavailable; sampling continues through WFOMC's fallback backend.

## Python API

Compile once and reuse the sampler. `sample_many()` is an iterator, so it does
not retain earlier structures.

```python
from c2_wms import compile_sampler, validate_structure
from wfomc import parse_problem

problem = parse_problem(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & (S(Y,X) | T(X))))
domain = 20
2 1 R
""")

with compile_sampler(problem, seed=7) as sampler:
    for structure in sampler.sample_many(100):
        validate_structure(problem, structure)  # opt-in diagnostic check
        print(structure.true_atoms())
```

`SampledStructure.relation(name, arity)` returns the true tuples for one source
predicate. Reduction predicates introduced by WFOMC are never exposed.
`validate_structure(problem, structure)` directly checks the original formula,
domain, evidence, cardinality constraints, nullary predicates, and `LEQ`. It is
intended for tests and diagnostics, not the warm sampling path.

Sampling requires concrete, exact, non-negative weights. This repository only
supports WFOMC's exact arithmetic path; rounded arithmetic is deliberately out
of scope and is not exposed as a sampler option. Symbolic output weights must
be substituted before compilation because they do not define a probability
measure. Unsatisfiable inputs raise `UnsatisfiableProblemError`.

## CLI

```bash
uv run wfoms --input model.wfomcs --samples 10 --seed 7
uv run wfoms --input model.wfomcs --samples 10 --seed 7 --output samples.jsonl
uv run wfoms --input model.wfomcs --samples 10 --validate --output samples.jsonl
uv run wfoms --input model.wfomcs --samples 10 -v --output samples.jsonl
uv run wfoms --input model.wfomcs --samples 1 -vv 2>debug.log
```

Without `--output`, the CLI writes one JSON object per sampled structure to
standard output. With `--output/-o`, it saves the same JSON Lines stream to the
given file. Each record contains the complete domain and every source relation,
including empty and nullary relations:

```json
{"domain":["a","b"],"relations":[{"predicate":"P","arity":1,"tuples":[["a"]]}]}
```

One model per line keeps large runs streamable and allows processing with tools
such as `jq`, Python, Polars, or DuckDB without loading the full sample set.
`--validate` checks every structure against the original formula and problem
metadata before writing it. It is disabled by default because formula
interpretation and the cubic `LEQ` transitivity check are diagnostic work.

Diagnostic logs use `WARNING` by default. `-v` enables INFO messages for parse,
WFOMC preparation, trace compilation, batch sampling, and validation timing.
`-vv` enables DEBUG messages for root and degree choices, traceback/label alias
creation, pair backend cache misses, relation sizes, and per-sample timing.
Additional `v` flags remain at DEBUG. Logs always go to standard error, so
standard output remains a valid JSON Lines stream. Redirect stderr to retain a
debug log without changing `--output`. Python API users can enable the same
library logs with `logging.basicConfig(level=logging.DEBUG)` before calling
`compile_sampler()`.

## Architecture and performance

The expensive lifted DP is compiled once. A warm sample has two conceptual
steps:

1. sample an anonymous lifted structure through an exact root draw and
   incremental3 traceback;
2. materialize it through conditioned local pair sampling, evidence-aware
   labels, and source-vocabulary projection.

All discrete choices use rational-to-integer alias tables; no probability is
rounded through `float`. Cardinality markers are sampled as exact sparse FLINT
degree budgets. Pair atoms are restored directly when the projected mask fully
determines them; otherwise conditioned distributions and aliases are built
lazily and reused. Per-sample storage is `O(n²)`, matching the worst-case output
size of a binary structure. See [the architecture notes](docs/architecture.md)
for module boundaries and validation invariants.

The test suite includes mass equality checks against WFOMC, pair-factor
equality checks, general-count semantic checks, weighted-distribution tests,
and a warm-sampling regression:

```bash
uv run pytest tests/unit tests/integration
uv run pytest -m performance tests/performance
```

## Model corpus

[`models/`](models/README.md) contains 81 deduplicated `.wfomcs` inputs from
WFOMC and `lifted_sampling_fo2` that successfully complete parse, compile,
sampling, and diagnostic validation with the current implementation. The
corpus README documents its categories, provenance, selection rule, and the
command for rerunning the full compatibility check.
