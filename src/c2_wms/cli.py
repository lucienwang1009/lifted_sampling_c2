"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"


def _configure_logging(verbosity: int) -> None:
    """Configure application logging without changing JSON output streams."""

    levels = (logging.WARNING, logging.INFO, logging.DEBUG)
    numeric_level = levels[min(verbosity, 2)]
    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Test runners and embedding applications may already have installed a
    # root handler, in which case basicConfig intentionally does nothing.
    logging.getLogger().setLevel(numeric_level)


def _model_record(sample) -> dict[str, object]:
    return {
        "domain": [str(element) for element in sample.domain],
        "relations": [
            {
                "predicate": predicate.name,
                "arity": predicate.arity,
                "tuples": [
                    [str(term) for term in terms]
                    for terms in sorted(tuples, key=lambda values: tuple(map(str, values)))
                ],
            }
            for predicate, tuples in sample.relations
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact lifted sampling for general C2")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--samples", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase logging verbosity; use -v for INFO and -vv for DEBUG",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate every sampled structure before writing it",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write sampled models as JSON Lines instead of stdout",
    )
    return parser


def main() -> None:
    from wfomc import parse_input

    from .errors import StructureValidationError
    from .sampler import compile_sampler
    from .validation import validate_structure

    parser = _parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    started = perf_counter()
    logger.info(
        "wfoms started input=%s samples=%d seed=%r validate=%s output=%s",
        args.input,
        args.samples,
        args.seed,
        args.validate,
        args.output or "stdout",
    )

    parse_started = perf_counter()
    problem = parse_input(args.input)
    logger.info(
        "Parsed problem domain=%d predicates=%d elapsed_ms=%.3f",
        len(problem.domain),
        len(problem.declared_predicate_names()),
        (perf_counter() - parse_started) * 1000,
    )
    output = (
        nullcontext(sys.stdout) if args.output is None else args.output.open("w", encoding="utf-8")
    )
    with output as stream, compile_sampler(problem, seed=args.seed) as sampler:
        for index, sample in enumerate(sampler.sample_many(args.samples), start=1):
            if args.validate:
                try:
                    validate_structure(problem, sample)
                except StructureValidationError as exc:
                    logger.error("Sample validation failed index=%d error=%s", index, exc)
                    parser.exit(1, f"wfoms: sample {index} failed validation: {exc}\n")
            print(
                json.dumps(_model_record(sample), ensure_ascii=False, separators=(",", ":")),
                file=stream,
            )
            logger.debug(
                "Wrote sample index=%d relations=%d true_tuples=%d",
                index,
                len(sample.relations),
                sum(len(tuples) for _, tuples in sample.relations),
            )
    logger.info(
        "wfoms completed samples=%d output=%s elapsed_ms=%.3f",
        args.samples,
        args.output or "stdout",
        (perf_counter() - started) * 1000,
    )
