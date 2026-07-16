"""The single compatibility boundary around WFOMC's algorithm-owned inputs."""

from __future__ import annotations

import logging
from fractions import Fraction
from time import perf_counter

from wfomc import AlgoName, compile_problem
from wfomc.algo import AlgoOptions, EvidenceStrategy, ExistentialStrategy
from wfomc.algo.incremental3.input import CountingDPInput
from wfomc.weights import WeightOptions

from .errors import UnsupportedSamplingInput, WfomcCompatibilityError

PINNED_WFOMC_REVISION = "481230d668dd34051161f2ca41fa21f2f008af84"

logger = logging.getLogger(__name__)


def _validate_source_weights(problem) -> None:
    for predicate, (positive, negative) in problem.weights.items():
        for polarity, value in (("positive", positive), ("negative", negative)):
            try:
                rational = Fraction(value)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise UnsupportedSamplingInput(
                    f"symbolic {polarity} weight for {predicate} requires a "
                    "numeric substitution before sampling"
                ) from exc
            if rational < 0:
                raise UnsupportedSamplingInput(
                    f"{polarity} weight for {predicate} must be non-negative"
                )


def compile_incremental3(problem):
    """Compile through WFOMC with sampling-safe strategy choices."""

    started = perf_counter()
    logger.debug(
        "Preparing WFOMC incremental3 input domain=%d weights=%d unary_evidence=%d "
        "binary_evidence=%d cardinality_constraints=%d",
        len(problem.domain),
        len(problem.weights),
        len(problem.evidence.unary.literals),
        len(problem.evidence.binary.literals),
        len(problem.cardinality_constraints.constraints),
    )
    _validate_source_weights(problem)
    artifacts = compile_problem(
        problem,
        algo=AlgoName.INCREMENTAL3,
        options=AlgoOptions(
            evidence_strategy=EvidenceStrategy.LIFTED_PROFILES,
            existential_strategy=ExistentialStrategy.COUNTING,
            weight_options=WeightOptions(precision="exact"),
        ),
    )
    if not artifacts.algo_inputs:
        raise WfomcCompatibilityError("WFOMC produced no incremental3 inputs")
    for algo_input in artifacts.algo_inputs:
        if not isinstance(algo_input, CountingDPInput):
            raise WfomcCompatibilityError(
                f"expected CountingDPInput, got {type(algo_input).__name__}"
            )
        if algo_input.arithmetic.output_symbols:
            raise UnsupportedSamplingInput(
                "symbolic output weights require a numeric substitution before sampling"
            )
    logger.info(
        "Prepared WFOMC incremental3 input branches=%d components=%d cells=%d elapsed_ms=%.3f",
        len(artifacts.algo_inputs),
        sum(len(algo_input.components) for algo_input in artifacts.algo_inputs),
        sum(
            len(component.cells)
            for algo_input in artifacts.algo_inputs
            for component in algo_input.components
        ),
        (perf_counter() - started) * 1000,
    )
    return artifacts


__all__ = ["PINNED_WFOMC_REVISION", "compile_incremental3"]
