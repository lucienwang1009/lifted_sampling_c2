# Preprocessing and sampling benchmarks

`run_benchmarks.py` compares compilation and sampling for the selected WFOMCS
models. Every case runs in a fresh process with a wall-clock timeout and an
address-space limit, so compile caches and memory do not leak between cases.

The default run uses domain sizes `20,40,60,80,100`, 20 warm samples, a 100-second
timeout, and a 4 GiB memory limit:

```bash
uv run python benchmarks/run_benchmarks.py
```

Stable-roommates inputs keep the explicit domains from their files. The other
models have their single numeric domain declaration replaced in memory; source
model files are never modified. The requested `2-regular-graph-sc2.wfomcs`
does not exist in the repository, so the runner uses
`2-regular-graph-sc2-n6.wfomcs` as its template. The n6 and n7 files differ only
in their domain declaration.

Useful options:

```bash
# Inspect the matrix without running it.
uv run python benchmarks/run_benchmarks.py --list

# Quick smoke run for one model.
uv run python benchmarks/run_benchmarks.py \
  --include function-no-fix --sizes 6,8 --samples 3 --timeout 30

# Customize the full experiment.
uv run python benchmarks/run_benchmarks.py \
  --sizes 20,40,60,80,100 --samples 20 --timeout 100 --memory-gib 4 \
  --output benchmarks/results/comparison.json
```

Each successful row reports:

- `parse_ms`: WFOMCS parsing only.
- `compile_ms`: `compile_sampler` only.
- `preprocess_ms`: parse plus compile.
- `first_sample_ms`: first sample, including lazy runtime cache construction.
- `warm_*`: subsequent sample latency and throughput.
- `peak_rss_mib`: peak resident set size reported by the worker process.

JSON metadata and raw results are written after every case. A flat CSV file is
written beside the JSON file. After timeout, OOM, or worker crash, larger domain
sizes for the same adjustable model are skipped by default; use
`--keep-going-sizes` to disable that behavior.
