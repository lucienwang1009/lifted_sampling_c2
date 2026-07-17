"""Compile and repeatedly sample exact general-C2 structures."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

from flint import fmpq

from ._wfomc_adapter import compile_incremental3
from .arithmetic import CoefficientCache, Degree
from .discrete_sampling import ExactAliasTable, RandomSource
from .errors import (
    SamplingError,
    UnsatisfiableProblemError,
    WfomcCompatibilityError,
)
from .label_sampling import LabelSampler
from .options import SamplerOptions
from .pair_sampling import PairSampler
from .projection import project_structure, projection_metadata, source_predicate_keys
from .trace import (
    AnonymousSample,
    ComponentTrace,
    TracebackSampler,
    compile_component_trace,
)

if TYPE_CHECKING:
    from wfomc.problem import Problem

    from .structure import SampledStructure


logger = logging.getLogger(__name__)


def _cardinality_marker_map(problem) -> dict[object, str]:
    predicates = sorted(
        {
            term.predicate
            for constraint in problem.cardinality_constraints.constraints
            for term in constraint.terms
        },
        key=str,
    )
    return {predicate: f"__wfomc_cardinality_{index}" for index, predicate in enumerate(predicates)}


def _valid_cardinality_degree(
    problem,
    symbolic_variables: tuple[str, ...],
    degree: Degree,
) -> bool:
    if len(symbolic_variables) != len(degree):
        return False
    by_symbol = dict(zip(symbolic_variables, degree, strict=True))
    marker_map = _cardinality_marker_map(problem)
    return all(
        constraint.accepts(
            sum(
                term.coefficient * by_symbol.get(marker_map[term.predicate], 0)
                for term in constraint.terms
            )
        )
        for constraint in problem.cardinality_constraints.constraints
    )


def _compile_traces(
    artifacts,
    *,
    max_trace_states: int | None,
) -> tuple[ComponentTrace, ...]:
    if artifacts.reduced_problem is None:
        raise WfomcCompatibilityError("WFOMC did not retain reduced problems")
    reduced = tuple(item.problem for item in artifacts.reduced_problem.problems)
    if len(reduced) != len(artifacts.algo_inputs):
        raise WfomcCompatibilityError("WFOMC branch metadata is misaligned")

    traces = []
    for branch_index, (reduced_problem, algo_input) in enumerate(
        zip(reduced, artifacts.algo_inputs, strict=True)
    ):
        for component_index, component in enumerate(algo_input.components):
            started = perf_counter()
            logger.info(
                "Compiling sampling trace branch=%d component=%d domain=%d cells=%d",
                branch_index,
                component_index,
                algo_input.domain_size,
                len(component.cells),
            )
            trace = compile_component_trace(
                algo_input,
                component,
                reduced_problem=reduced_problem,
                max_trace_states=max_trace_states,
            )
            traces.append(trace)
            logger.info(
                "Compiled sampling trace branch=%d component=%d roots=%d states=%d elapsed_ms=%.3f",
                branch_index,
                component_index,
                len(trace.root_terms),
                len(trace.domain_values),
                (perf_counter() - started) * 1000,
            )
    if not traces:
        raise UnsatisfiableProblemError("WFOMC produced no satisfiable components")
    return tuple(traces)


class CompiledSampler:
    """Compiled root mixture plus the caches needed for repeated sampling."""

    def __init__(self, problem, traces, rng, options):
        self.problem = problem
        self.traces = traces
        self.rng = rng
        self.options = options
        self._trace_samplers = tuple(TracebackSampler(trace, rng) for trace in traces)
        self._coefficients = tuple(
            CoefficientCache(len(trace.arithmetic.symbolic_variables)) for trace in traces
        )
        self._root_alias, mixture_mass = self._build_root_mixture()
        order_factor = (
            math.factorial(len(problem.domain))
            if any(trace.has_linear_order for trace in traces)
            else 1
        )
        self._total_weight = mixture_mass * order_factor
        self._split_aliases: dict[object, ExactAliasTable] = {}
        self._label_sampler = LabelSampler(problem, rng)
        self._source_predicates = source_predicate_keys(problem)
        self._projection_metadata = {
            id(trace): projection_metadata(trace, self._source_predicates) for trace in traces
        }
        self._closed = False
        self._sample_count = 0
        self._pair_samplers = {
            id(trace): PairSampler(
                trace,
                rng,
                self._source_predicates,
                validate_masses=options.validate_masses,
            )
            for trace in traces
        }
        logger.info(
            "Initialized compiled sampler traces=%d root_choices=%d source_predicates=%d "
            "linear_order=%s",
            len(self.traces),
            len(self._root_alias.choices),
            len(self._source_predicates),
            any(trace.has_linear_order for trace in traces),
        )

    def _build_root_mixture(self):
        choices = []
        weights = []
        for trace_index, trace in enumerate(self.traces):
            cache = self._coefficients[trace_index]
            symbols = tuple(trace.arithmetic.symbolic_variables)
            for root_index, root in enumerate(trace.root_terms):
                for degree, coefficient in cache.terms(root.mass).items():
                    if not _valid_cardinality_degree(self.problem, symbols, degree):
                        continue
                    if coefficient < 0:
                        raise SamplingError("compiled root mixture has negative mass")
                    if coefficient > 0:
                        choices.append((trace_index, root_index, degree))
                        weights.append(coefficient)
        if not choices:
            raise UnsatisfiableProblemError("the problem has zero total weight")
        logger.debug(
            "Built root mixture choices=%d traces=%d symbolic_dimensions=%s",
            len(choices),
            len(self.traces),
            tuple(len(trace.arithmetic.symbolic_variables) for trace in self.traces),
        )
        return ExactAliasTable(choices, weights), sum(weights, fmpq(0))

    def _split_root(
        self,
        trace_index: int,
        root_index: int,
        degree: Degree,
    ):
        key = (trace_index, root_index, degree)
        table = self._split_aliases.get(key)
        if table is None:
            root = self.traces[trace_index].root_terms[root_index]
            cache = self._coefficients[trace_index]
            choices = []
            weights = []
            for base_degree, domain_degree, weight in cache.product_splits(
                root.base_weight, root.domain_weight, degree
            ):
                if weight < 0:
                    raise SamplingError("compiled root split has negative mass")
                choices.append((base_degree, domain_degree))
                weights.append(weight)
            table = ExactAliasTable(choices, weights)
            self._split_aliases[key] = table
            logger.debug(
                "Built root split alias trace=%d root=%d degree=%s choices=%d cache_entries=%d",
                trace_index,
                root_index,
                degree,
                len(choices),
                len(self._split_aliases),
            )
        return table.sample(self.rng)

    def _sample_anonymous(self) -> AnonymousSample:
        trace_index, root_index, degree = self._root_alias.sample(self.rng)
        _base_degree, domain_degree = self._split_root(trace_index, root_index, degree)
        trace = self.traces[trace_index]
        root = trace.root_terms[root_index]
        logger.debug(
            "Selected root trace=%d root=%d degree=%s domain_degree=%s cell_config=%s",
            trace_index,
            root_index,
            degree,
            domain_degree,
            root.cell_config,
        )
        return self._trace_samplers[trace_index].sample(
            root,
            domain_degree,
            self._pair_samplers[id(trace)].sample_condition,
        )

    def _materialize(
        self,
        anonymous: AnonymousSample,
    ) -> SampledStructure:
        labels = self._label_sampler.sample(anonymous)
        pair_sampler = self._pair_samplers[id(anonymous.trace)]
        return project_structure(
            self.problem,
            anonymous,
            labels,
            pair_sampler,
            self._projection_metadata[id(anonymous.trace)],
            source_keys=self._source_predicates,
        )

    def sample(self) -> SampledStructure:
        if self._closed:
            raise RuntimeError("sampler is closed")
        self._sample_count += 1
        if not logger.isEnabledFor(logging.DEBUG):
            return self._materialize(self._sample_anonymous())

        started = perf_counter()
        anonymous_started = perf_counter()
        anonymous = self._sample_anonymous()
        anonymous_ms = (perf_counter() - anonymous_started) * 1000
        materialize_started = perf_counter()
        sample = self._materialize(anonymous)
        materialize_ms = (perf_counter() - materialize_started) * 1000
        logger.debug(
            "Sampled structure index=%d elements=%d pair_requests=%d relations=%d "
            "true_tuples=%d anonymous_ms=%.3f materialize_ms=%.3f elapsed_ms=%.3f",
            self._sample_count,
            len(sample.domain),
            len(anonymous.pair_requests),
            len(sample.relations),
            sum(len(tuples) for _, tuples in sample.relations),
            anonymous_ms,
            materialize_ms,
            (perf_counter() - started) * 1000,
        )
        return sample

    def sample_many(self, count: int) -> Iterator[SampledStructure]:
        if count < 0:
            raise ValueError("count must be non-negative")
        started = perf_counter()
        produced = 0
        logger.info("Sampling batch started requested=%d", count)
        try:
            for _ in range(count):
                sample = self.sample()
                produced += 1
                yield sample
        finally:
            logger.info(
                "Sampling batch completed requested=%d produced=%d elapsed_ms=%.3f",
                count,
                produced,
                (perf_counter() - started) * 1000,
            )

    @property
    def total_weight(self):
        """Exact accepted source-level model weight."""

        return self._total_weight

    def close(self) -> None:
        if not self._closed:
            for pair_sampler in self._pair_samplers.values():
                pair_sampler.close()
            self._closed = True
            logger.info(
                "Closed compiled sampler samples=%d root_split_aliases=%d",
                self._sample_count,
                len(self._split_aliases),
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def compile_sampler(
    problem: Problem,
    *,
    seed: int | None = None,
    options: SamplerOptions | None = None,
) -> CompiledSampler:
    """Compile *problem* into a reusable exact sampler."""

    if options is None:
        options = SamplerOptions(seed=seed)
    elif seed is not None:
        if options.seed is not None and options.seed != seed:
            raise ValueError("seed and options.seed disagree")
        options = replace(options, seed=seed)
    started = perf_counter()
    logger.info(
        "Compiling sampler domain=%d seed=%r validate_masses=%s max_trace_states=%r",
        len(problem.domain),
        options.seed,
        options.validate_masses,
        options.max_trace_states,
    )
    rng = RandomSource(options.seed)
    artifacts = compile_incremental3(problem)
    traces = _compile_traces(
        artifacts,
        max_trace_states=options.max_trace_states,
    )
    sampler = CompiledSampler(problem, traces, rng, options)
    logger.info(
        "Compiled sampler domain=%d traces=%d elapsed_ms=%.3f",
        len(problem.domain),
        len(traces),
        (perf_counter() - started) * 1000,
    )
    return sampler
