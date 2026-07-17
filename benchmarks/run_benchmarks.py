#!/usr/bin/env python3
"""Benchmark preprocessing and exact sampling in isolated worker processes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (20, 40, 60, 80, 100)
DOMAIN_LINE = re.compile(r"(?m)^(\s*[A-Za-z_]\w*\s*=\s*)\d+(\s*)$")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    path: Path
    adjustable: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    model: str
    path: Path
    domain_size: int | None
    adjustable: bool

    @property
    def case_id(self) -> str:
        suffix = str(self.domain_size) if self.domain_size is not None else "fixed"
        return f"{self.model}:n={suffix}"


ADJUSTABLE_MODELS = (
    ModelSpec("2-colored-graph", REPOSITORY / "models/2-colored-graph.wfomcs"),
    ModelSpec("friends-smokes", REPOSITORY / "models/friends-smokes.wfomcs"),
    ModelSpec("nonisolated_graph", REPOSITORY / "models/nonisolated_graph.wfomcs"),
    ModelSpec(
        "permutation-no-fix-sc2",
        REPOSITORY / "models/permutation-no-fix-sc2.wfomcs",
    ),
    ModelSpec("function-no-fix-sc2", REPOSITORY / "models/function-no-fix-sc2.wfomcs"),
    # The repository has n6/n7 copies whose only difference is the domain line.
    ModelSpec(
        "2-regular-graph-sc2",
        REPOSITORY / "models/regular_graphs/2-regular-graph-sc2-n6.wfomcs",
    ),
    ModelSpec(
        "2-regular-directed-graph",
        REPOSITORY / "models/regular_graphs/2-regular-directed-graph.wfomcs",
    ),
    ModelSpec(
        "3-regular-2-colored-graph",
        REPOSITORY / "models/regular_graphs/3-regular-2-colored-graph.wfomcs",
    ),
)


def _stable_roommates() -> tuple[ModelSpec, ...]:
    directory = REPOSITORY / "models/stable_roommates"
    return tuple(
        ModelSpec(path.stem, path, adjustable=False) for path in sorted(directory.glob("*.wfomcs"))
    )


def _sizes(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    return values


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--samples", type=_positive_int, default=20)
    parser.add_argument("--timeout", type=_positive_float, default=100.0)
    parser.add_argument("--memory-gib", type=_positive_float, default=4.0)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="TEXT",
        help="run only model names containing TEXT; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; CSV is written alongside it",
    )
    parser.add_argument(
        "--keep-going-sizes",
        action="store_true",
        help="continue larger domains after a timeout or memory failure",
    )
    parser.add_argument("--list", action="store_true", help="list selected cases and exit")

    worker = parser.add_argument_group("internal worker options")
    worker.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--model", help=argparse.SUPPRESS)
    worker.add_argument("--path", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--domain-size", type=int, help=argparse.SUPPRESS)
    return parser


def _selected_specs(includes: list[str]) -> tuple[ModelSpec, ...]:
    specs = ADJUSTABLE_MODELS + _stable_roommates()
    if not includes:
        return specs
    lowered = tuple(value.lower() for value in includes)
    return tuple(spec for spec in specs if any(value in spec.name.lower() for value in lowered))


def _cases(specs: tuple[ModelSpec, ...], sizes: tuple[int, ...]) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for spec in specs:
        domains = sizes if spec.adjustable else (None,)
        cases.extend(BenchmarkCase(spec.name, spec.path, size, spec.adjustable) for size in domains)
    return tuple(cases)


def _replace_domain(source: str, domain_size: int) -> str:
    matches = tuple(DOMAIN_LINE.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"expected one numeric domain declaration, found {len(matches)}")
    match = matches[0]
    replacement = f"{match.group(1)}{domain_size}{match.group(2)}"
    return source[: match.start()] + replacement + source[match.end() :]


def _set_memory_limit(memory_gib: float) -> None:
    if sys.platform == "darwin":
        # macOS processes reserve enormous virtual address spaces, so RLIMIT_AS
        # is not a useful proxy for resident memory. The parent monitors RSS.
        return
    limit = int(memory_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _peak_rss_mib() -> float:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix platforms report KiB.
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return maximum / divisor


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _worker(args: argparse.Namespace) -> int:
    if not args.model or args.path is None:
        raise SystemExit("worker requires --model and --path")
    started = perf_counter()
    result: dict[str, Any] = {
        "model": args.model,
        "path": str(args.path.relative_to(REPOSITORY)),
        "requested_domain_size": args.domain_size,
        "samples": args.samples,
    }
    try:
        _set_memory_limit(args.memory_gib)
        source = args.path.read_text(encoding="utf-8")
        if args.domain_size is not None:
            source = _replace_domain(source, args.domain_size)

        from wfomc import parse_problem

        from c2_wms import compile_sampler

        parse_started = perf_counter()
        problem = parse_problem(source)
        result["parse_ms"] = (perf_counter() - parse_started) * 1000
        result["domain_size"] = len(problem.domain)
        if args.domain_size is not None and len(problem.domain) != args.domain_size:
            raise ValueError(
                f"parsed domain size {len(problem.domain)} does not match {args.domain_size}"
            )

        compile_started = perf_counter()
        sampler = compile_sampler(problem, seed=42)
        result["compile_ms"] = (perf_counter() - compile_started) * 1000
        result["preprocess_ms"] = result["parse_ms"] + result["compile_ms"]

        with sampler:
            first_started = perf_counter()
            sampler.sample()
            result["first_sample_ms"] = (perf_counter() - first_started) * 1000

            warm_ms: list[float] = []
            for _ in range(args.samples):
                sample_started = perf_counter()
                sampler.sample()
                warm_ms.append((perf_counter() - sample_started) * 1000)

        result.update(
            {
                "status": "ok",
                "warm_total_ms": sum(warm_ms),
                "warm_mean_ms": statistics.fmean(warm_ms),
                "warm_p50_ms": statistics.median(warm_ms),
                "warm_p95_ms": _percentile(warm_ms, 0.95),
                "warm_min_ms": min(warm_ms),
                "warm_max_ms": max(warm_ms),
                "warm_samples_per_second": 1000 / statistics.fmean(warm_ms),
            }
        )
    except MemoryError as exc:
        result.update(status="oom", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - benchmark must record each failed case.
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["wall_ms"] = (perf_counter() - started) * 1000
        result["peak_rss_mib"] = _peak_rss_mib()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 1


def _worker_command(case: BenchmarkCase, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        case.model,
        "--path",
        str(case.path),
        "--samples",
        str(args.samples),
        "--memory-gib",
        str(args.memory_gib),
    ]
    if case.domain_size is not None:
        command.extend(("--domain-size", str(case.domain_size)))
    return command


def _run_case(case: BenchmarkCase, args: argparse.Namespace) -> dict[str, Any]:
    import psutil

    started = perf_counter()
    process = subprocess.Popen(
        _worker_command(case, args),
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    monitored = psutil.Process(process.pid)
    memory_limit = args.memory_gib * 1024**3
    peak_rss = 0
    rss = 0
    limit_status: str | None = None
    while process.poll() is None:
        try:
            processes = (monitored, *monitored.children(recursive=True))
            rss = sum(child.memory_info().rss for child in processes if child.is_running())
            peak_rss = max(peak_rss, rss)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        elapsed = perf_counter() - started
        if rss > memory_limit:
            limit_status = "oom"
            break
        if elapsed > args.timeout:
            limit_status = "timeout"
            break
        time.sleep(0.05)

    if limit_status is not None:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {
            "model": case.model,
            "path": str(case.path.relative_to(REPOSITORY)),
            "requested_domain_size": case.domain_size,
            "domain_size": case.domain_size,
            "samples": args.samples,
            "status": limit_status,
            "wall_ms": (perf_counter() - started) * 1000,
            "peak_rss_mib": peak_rss / 1024**2,
            "error": (
                f"exceeded {args.timeout:g}s"
                if limit_status == "timeout"
                else f"exceeded {args.memory_gib:g} GiB RSS"
            ),
            "stderr": stderr.strip(),
        }

    stdout, stderr = process.communicate()

    lines = tuple(line for line in stdout.splitlines() if line.strip())
    if lines:
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError:
            result = {}
    else:
        result = {}
    if not result:
        result = {
            "model": case.model,
            "path": str(case.path.relative_to(REPOSITORY)),
            "requested_domain_size": case.domain_size,
            "domain_size": case.domain_size,
            "samples": args.samples,
            "status": "crashed",
            "wall_ms": (perf_counter() - started) * 1000,
            "error": f"worker exited with code {process.returncode}",
        }
    if stderr.strip():
        result["stderr"] = stderr.strip()
    result["peak_rss_mib"] = max(result.get("peak_rss_mib", 0), peak_rss / 1024**2)
    return result


def _output_path(path: Path | None) -> Path:
    if path is not None:
        return path if path.is_absolute() else REPOSITORY / path
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY / "benchmarks/results" / f"benchmark-{timestamp}.json"


def _write_results(
    path: Path,
    args: argparse.Namespace,
    cases: tuple[BenchmarkCase, ...],
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": sys.platform,
            "timeout_seconds": args.timeout,
            "memory_gib": args.memory_gib,
            "warm_samples": args.samples,
            "sizes": args.sizes,
            "cases": [
                {**asdict(case), "path": str(case.path.relative_to(REPOSITORY))} for case in cases
            ],
        },
        "results": results,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    fields = (
        "model",
        "path",
        "domain_size",
        "requested_domain_size",
        "status",
        "parse_ms",
        "compile_ms",
        "preprocess_ms",
        "first_sample_ms",
        "warm_mean_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "warm_samples_per_second",
        "peak_rss_mib",
        "wall_ms",
        "samples",
        "error",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _main(args: argparse.Namespace) -> int:
    specs = _selected_specs(args.include)
    cases = _cases(specs, args.sizes)
    if args.list:
        for case in cases:
            print(f"{case.case_id}\t{case.path.relative_to(REPOSITORY)}")
        return 0
    if not cases:
        raise SystemExit("no benchmark cases matched --include")

    output = _output_path(args.output)
    results: list[dict[str, Any]] = []
    blocked_models: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if case.adjustable and case.model in blocked_models:
            result = {
                "model": case.model,
                "path": str(case.path.relative_to(REPOSITORY)),
                "requested_domain_size": case.domain_size,
                "domain_size": case.domain_size,
                "samples": args.samples,
                "status": "skipped_after_limit",
                "error": "an earlier domain hit timeout or memory limit",
            }
        else:
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            result = _run_case(case, args)
            if (
                case.adjustable
                and not args.keep_going_sizes
                and result["status"] in {"timeout", "oom", "crashed"}
            ):
                blocked_models.add(case.model)
        results.append(result)
        metric = result.get("compile_ms")
        detail = f" compile={metric:.1f}ms" if metric is not None else ""
        print(f"  {result['status']}{detail}", flush=True)
        _write_results(output, args, cases, results)

    print(f"JSON: {output.relative_to(REPOSITORY)}")
    print(f"CSV:  {output.with_suffix('.csv').relative_to(REPOSITORY)}")
    return 0


if __name__ == "__main__":
    namespace = _parser().parse_args()
    raise SystemExit(_worker(namespace) if namespace.worker else _main(namespace))
