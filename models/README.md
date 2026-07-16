# wfoms model corpus

This directory contains the deduplicated `.wfomcs` inputs that successfully
completed the following pipeline with the current repository revision:

```text
parse -> compile -> sample one structure -> validate_structure
```

The corpus contains 81 distinct model contents. Models that are unsatisfiable
with their current parameters, require unsupported PREDk/CircularPred features,
or use unsupported negative weights are intentionally excluded.

## Layout

| Directory | Models | Coverage |
| --- | ---: | --- |
| repository root | 14 | baseline C2, graph coloring, functions, permutations |
| `counting_quantifiers/` | 28 | row/global comparators and unary/nullary definitions |
| `linear_order/` | 5 | LEQ, cardinality, and unary evidence |
| `MATH/` | 1 | compatible benchmark instance |
| `modk/` | 3 | modulo-counting graph constraints |
| `regular_graphs/` | 12 | directed, undirected, colored, and SC2 instances |
| `stable_roommates/` | 14 | STMU scalability instances |
| `unary_evidence/` | 4 | evidence profiles and overlapping profiles |

The two upstream `regular_graphs/2-regular-graph-sc2.wfomcs` files have
different domain sizes. They are preserved as `-n6` and `-n7` instead of
silently choosing one.

## Sources

- `yuanhong-wang/WFOMC`, local commit `84949b2d0e3a2ea1fbbaff7f4029e26561a551b9`
- `lucienwang1009/lifted_sampling_fo2`, local commit
  `b49ef85478cb91adaf0ea38bdebf1b81f26c21a1`

Identical files from both sources are stored once. The original category
layout is retained where possible; equivalent files found under different
source paths use the WFOMC layout.

## Run the corpus

From the repository root:

```bash
find models -name '*.wfomcs' -exec sh -c '
  for model do
    echo "$model"
    uv run wfoms --input "$model" --samples 1 --seed 1 \
      --validate --output /dev/null || exit 1
  done
' sh {} +
```

Validation is diagnostic and makes this command slower than normal sampling,
especially for the larger stable-roommates instances.

The stable-roommates models require the repository's pinned Ganak revision
`82a1d1fb6f0d6fb4a46b825f84b29567728ae483` for practical preprocessing
performance. Without Ganak, WFOMC falls back to PySDD, which can be extremely
slow and memory-intensive. See the root README's Ganak setup instructions.
