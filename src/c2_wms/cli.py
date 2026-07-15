"""Command-line entry point."""

from __future__ import annotations

import argparse
import json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact lifted sampling for general C2")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--samples", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    from wfomc import parse_input

    from .sampler import compile_sampler

    args = _parser().parse_args()
    sampler = compile_sampler(parse_input(args.input), seed=args.seed)
    for sample in sampler.sample_many(args.samples):
        print(json.dumps({"atoms": sample.true_atoms()}, ensure_ascii=False))
