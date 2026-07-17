# WFOMCS preprocessing and sampling comparison

Run date: 2026-07-17

Environment: Apple M4, 16 GiB RAM, macOS 15.7.4, Python 3.11.7. Each case ran
in a fresh worker process with a 100-second wall timeout and a 4 GiB aggregate
RSS limit. A successful case produced one separately timed first sample followed
by 20 warm samples.

Two runs are retained locally: the large-domain run at
`benchmarks/results/large-comparison.{json,csv}` and the original small-domain
baseline at `benchmarks/results/comparison.{json,csv}`. The results directory is
ignored because timings are machine-specific.

## Large-domain follow-up

The adjustable models were rerun at domain sizes 20, 40, 60, 80, and 100. Of 40
matrix entries, 33 completed, two timed out, and five larger entries were skipped
after the first timeout. No worker exceeded the 4 GiB RSS limit.

| Model | Largest completed n | Compile | Warm P50 | Peak RSS | Limit reached |
|---|---:|---:|---:|---:|---|
| 2-colored-graph | 100 | 34.4 ms | 4.703 ms | 57.8 MiB | none |
| friends-smokes | 20 | 922.2 ms | 0.730 ms | 160.1 MiB | n=40 timeout |
| nonisolated_graph | 100 | 10.67 s | 9.426 ms | 2006.6 MiB | none |
| permutation-no-fix-sc2 | 100 | 5.5 ms | 4.147 ms | 48.8 MiB | none |
| function-no-fix-sc2 | 100 | 315.8 ms | 8.364 ms | 133.0 MiB | none |
| 2-regular-graph-sc2 | 100 | 29.3 ms | 3.575 ms | 55.9 MiB | none |
| 2-regular-directed-graph | 40 | 14.58 s | 5.227 ms | 2493.0 MiB | n=60 timeout |
| 3-regular-2-colored-graph | 100 | 54.42 s | 9.596 ms | 3888.0 MiB | near 4 GiB |

The larger range changes the conclusion materially. Three models now show clear
preprocessing state explosion:

- `friends-smokes` does not finish n=40 in 100 seconds, despite n=20 compiling
  in 0.92 seconds. Its timeout worker peaked around 1.5 GiB, so time—not memory—
  was the first limit.
- `2-regular-directed-graph` compiles n=40 in 14.6 seconds and times out at n=60;
  the timed-out worker reached about 3.6 GiB RSS.
- `3-regular-2-colored-graph` finishes n=100 in 54.4 seconds at 3.89 GiB RSS,
  leaving very little headroom below the configured memory ceiling.

`nonisolated_graph` also grows substantially but remains within the limits at
n=100: 10.7 seconds compile and 2.0 GiB RSS. In contrast, permutation,
undirected 2-regular SC2, and ordinary 2-colored graph compilation remains cheap.
Sampling latency scales much more gently than preprocessing; every completed
large-domain case remains below 10 ms warm P50.

## Small-domain baseline

The original n=6,8,10,12,16 run completed all 54 cases without timeout, OOM, or
crash. Its stable-roommates rows are still used below because those models keep
their file-defined domains.

## Small-domain adjustable models

The main table shows compile time and warm-sample P50 latency at the smallest
and largest tested sizes. Parsing is approximately 20-28 ms for these files and
is therefore omitted from the compile columns.

| Model | Compile n=6 | Compile n=16 | Growth | Warm P50 n=6 | Warm P50 n=16 | Peak RSS n=16 |
|---|---:|---:|---:|---:|---:|---:|
| 2-colored-graph | 4.47 ms | 3.79 ms | noise-level | 0.034 ms | 0.147 ms | 47.8 MiB |
| friends-smokes | 4.55 ms | 223.00 ms | 49.0x | 0.166 ms | 0.503 ms | 81.3 MiB |
| nonisolated_graph | 1.79 ms | 15.21 ms | 8.5x | 0.087 ms | 0.339 ms | 49.3 MiB |
| permutation-no-fix-sc2 | 1.81 ms | 1.88 ms | 1.0x | 0.041 ms | 0.151 ms | 46.9 MiB |
| function-no-fix-sc2 | 1.28 ms | 3.54 ms | 2.8x | 0.052 ms | 0.265 ms | 47.7 MiB |
| 2-regular-graph-sc2 | 1.49 ms | 2.02 ms | 1.4x | 0.032 ms | 0.142 ms | 46.8 MiB |
| 2-regular-directed-graph | 3.63 ms | 210.96 ms | 58.1x | 0.108 ms | 1.121 ms | 88.0 MiB |
| 3-regular-2-colored-graph | 2.55 ms | 19.63 ms | 7.7x | 0.034 ms | 0.231 ms | 48.7 MiB |

Compile-time progression for the two fastest-growing adjustable models:

| Model | n=6 | n=8 | n=10 | n=12 | n=16 |
|---|---:|---:|---:|---:|---:|
| friends-smokes | 4.55 ms | 11.04 ms | 21.92 ms | 47.25 ms | 223.00 ms |
| 2-regular-directed-graph | 3.63 ms | 11.38 ms | 25.76 ms | 55.61 ms | 210.96 ms |

The dominant scaling cost for these two models is preprocessing rather than
warm sampling. At n=16, compilation is roughly 200 ms, while a warm sample is
about 0.5-1.1 ms median. For repeated sampling, the compile cost is amortized
after a few hundred samples. For one-off use, parsing plus compilation dominates.

The permutation and undirected 2-regular SC2 models have nearly domain-invariant
compile time in this range. Their runtime growth appears in traceback and output
construction instead: warm latency rises by roughly 3.7-4.4x from n=6 to n=16,
while memory remains near the 47 MiB process baseline.

## Stable roommates

These source files retain their explicit domains and evidence. They are distinct
fixed instances, not textually identical theories with only the domain changed,
so adjacent rows should be treated as workload progression rather than a clean
single-variable scaling experiment.

| Instance | Domain | Parse | Compile | First sample | Warm P50 | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| stmu_4_20 | 20 | 129.8 ms | 40.5 ms | 1.06 ms | 0.201 ms | 61.6 MiB |
| stmu_4_24 | 24 | 137.9 ms | 55.9 ms | 1.38 ms | 0.268 ms | 61.8 MiB |
| stmu_4_28 | 28 | 132.7 ms | 89.5 ms | 2.11 ms | 0.374 ms | 61.5 MiB |
| stmu_4_32 | 32 | 153.8 ms | 120.0 ms | 1.98 ms | 0.434 ms | 63.0 MiB |
| stmu_4_36 | 36 | 139.5 ms | 165.7 ms | 12.07 ms | 0.525 ms | 71.6 MiB |
| stmu_4_40 | 40 | 146.5 ms | 216.5 ms | 2.26 ms | 0.601 ms | 79.5 MiB |
| stmu_4_44 | 44 | 141.2 ms | 337.5 ms | 2.75 ms | 0.712 ms | 97.2 MiB |
| stmu_4_48 | 48 | 142.7 ms | 478.1 ms | 2.83 ms | 0.797 ms | 109.8 MiB |
| stmu_5_12 | 12 | 216.6 ms | 307.3 ms | 0.92 ms | 0.113 ms | 83.4 MiB |
| stmu_5_16 | 16 | 220.5 ms | 215.6 ms | 1.18 ms | 0.168 ms | 84.2 MiB |
| stmu_5_20 | 20 | 222.8 ms | 540.4 ms | 1.57 ms | 0.221 ms | 118.8 MiB |
| stmu_5_24 | 24 | 222.1 ms | 1476.1 ms | 1.90 ms | 0.285 ms | 257.9 MiB |
| stmu_5_28 | 28 | 243.1 ms | 3429.6 ms | 2.09 ms | 0.362 ms | 517.0 MiB |
| stmu_5_32 | 32 | 224.7 ms | 8088.1 ms | 2.52 ms | 0.466 ms | 919.8 MiB |

The five-profile family is the clearest preprocessing and memory hotspot. From
stmu_5_20 to stmu_5_32, compile time grows from 0.54 s to 8.09 s and peak RSS
from 119 MiB to 920 MiB, while warm sampling remains below 0.5 ms median. This
points to compiled trace/state volume, not sample materialization, as the primary
optimization target for these instances.

The stmu_4 family is much gentler: from domain 20 to 48, compilation grows about
11.8x, warm P50 about 4.0x, and peak RSS reaches 110 MiB. The isolated 12 ms first
sample for stmu_4_36 did not repeat in its warm timings and should be treated as
a cold-path or scheduling outlier until a repeated-case benchmark confirms it.

## Conclusions

1. Warm sampling remains under 10 ms P50 for every completed n=100 case; all
   stable-roommates P50 values remain below 0.8 ms.
2. Optimization should prioritize compilation/state construction for
   friends-smokes, directed regular graphs, nonisolated graphs, and especially
   the five-profile stable-roommates family.
3. At n=100, `3-regular-2-colored-graph` is the closest to both configured
   limits: 54.4 seconds compile and 3.89 GiB RSS.
4. Sub-millisecond sampling measurements are sensitive to scheduler noise. For
   regression gating, use repeated whole-case runs or increase `--samples`; do
   not set thresholds from a single P95 observation in this 20-sample run.
