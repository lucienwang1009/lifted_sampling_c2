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

## Python API

Compile once and reuse the sampler. `sample_many()` is an iterator, so it does
not retain earlier structures.

```python
from c2_wms import compile_sampler
from wfomc import parse_problem

problem = parse_problem(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & (S(Y,X) | T(X))))
domain = 20
2 1 R
""")

with compile_sampler(problem, seed=7) as sampler:
    for structure in sampler.sample_many(100):
        print(structure.true_atoms())
```

`SampledStructure.relation(name, arity)` returns the true tuples for one source
predicate. Reduction predicates introduced by WFOMC are never exposed.

Sampling requires concrete, exact, non-negative weights. This repository only
supports WFOMC's exact arithmetic path; rounded arithmetic is deliberately out
of scope and is not exposed as a sampler option. Symbolic output weights must
be substituted before compilation because they do not define a probability
measure. Unsatisfiable inputs raise `UnsatisfiableProblemError`.

## CLI

```bash
uv run wfoms --input model.wfomcs --samples 10 --seed 7
```

The CLI writes one JSON object per sampled structure, which makes large runs
streamable.

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
