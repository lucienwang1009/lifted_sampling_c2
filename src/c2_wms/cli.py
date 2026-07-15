"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path


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
        "--output",
        "-o",
        type=Path,
        help="write sampled models as JSON Lines instead of stdout",
    )
    return parser


def main() -> None:
    from wfomc import parse_input

    from .sampler import compile_sampler

    args = _parser().parse_args()
    output = (
        nullcontext(sys.stdout) if args.output is None else args.output.open("w", encoding="utf-8")
    )
    with output as stream, compile_sampler(parse_input(args.input), seed=args.seed) as sampler:
        for sample in sampler.sample_many(args.samples):
            print(
                json.dumps(_model_record(sample), ensure_ascii=False, separators=(",", ":")),
                file=stream,
            )
