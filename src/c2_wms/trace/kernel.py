"""Trace-producing variant of WFOMC's incremental3 value recursion."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from itertools import product
from time import perf_counter

from wfomc.algo.incremental3.counting_kernel import (
    Config,
    ConfigSpace,
    State,
    _build_elimination_orders,
)
from wfomc.algo.incremental3.input import CountingCellGraphComponent, CountingDPInput
from wfomc.cell_graph import CellConfigCoefficientBasis, CellEvidenceAllocation
from wfomc.multinomial import MultinomialCoefficients

from c2_wms.errors import WfomcCompatibilityError

from .model import (
    ComponentTrace,
    DomainTraceNode,
    HTrace,
    RelationChoice,
    RootTerm,
    TargetTrace,
)

logger = logging.getLogger(__name__)


def _add_configs(left: Config, right: Config) -> Config:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _advance(counter_state, delta, counting_state):
    updated = []
    for value, observed, counter in zip(
        counter_state,
        delta,
        counting_state.row_counters,
        strict=True,
    ):
        next_value = counter.true_transitions[value] if observed else value
        if next_value is None:
            return None
        updated.append(next_value)
    return tuple(updated)


def projection_mask(forward, reverse, counting_state) -> int:
    forward_by_projection: dict[int, int] = {}
    reverse_by_projection: dict[int, int] = {}
    for counter_index, projection_index in enumerate(counting_state.counter_projection_indices):
        fwd = int(forward[counter_index])
        rev = int(reverse[counter_index])
        old_fwd = forward_by_projection.setdefault(projection_index, fwd)
        old_rev = reverse_by_projection.setdefault(projection_index, rev)
        if old_fwd != fwd or old_rev != rev:
            raise WfomcCompatibilityError(
                "duplicated counters disagree on their projected predicate"
            )
    mask = 0
    for projection_index in range(len(counting_state.projected_predicates)):
        if reverse_by_projection.get(projection_index, 0):
            mask |= 1 << (2 * projection_index)
        if forward_by_projection.get(projection_index, 0):
            mask |= 1 << (2 * projection_index + 1)
    return mask


def _build_transition_tables(component, counting_state, arithmetic):
    n_cells = len(component.cells)
    t_update = defaultdict(lambda: defaultdict(arithmetic.zero))
    choices = defaultdict(lambda: defaultdict(list))
    all_states = tuple(
        product(*(range(counter.state_count) for counter in counting_state.row_counters))
    )
    relation_rows = component.counting_binary_relation_weights
    if relation_rows is None:
        raise WfomcCompatibilityError("counting relation weights are not materialized")

    for left_index in range(n_cells):
        for right_index in range(n_cells):
            for left_counter in all_states:
                for right_counter in all_states:
                    for forward, reverse, weight in relation_rows[left_index][right_index]:
                        left_new = _advance(left_counter, forward, counting_state)
                        right_new = _advance(right_counter, reverse, counting_state)
                        if left_new is None or right_new is None:
                            continue
                        left = (left_index,) + left_counter
                        right = (right_index,) + right_counter
                        outcome = ((left_index,) + left_new, (right_index,) + right_new)
                        pair = (left, right)
                        t_update[pair][outcome] = arithmetic.add(t_update[pair][outcome], weight)
                        choices[pair][outcome].append(
                            RelationChoice(
                                projection_mask(forward, reverse, counting_state),
                                weight,
                            )
                        )
    return t_update, choices


class _TracedUpdater:
    def __init__(self, t_update_dict, space: ConfigSpace, arithmetic):
        self.t_update_dict = t_update_dict
        self.space = space
        self.arithmetic = arithmetic
        self.traces: dict[tuple[State, State], HTrace] = {}

    def get(self, target: State, other: State, count: int):
        key = (target, other)
        trace = self.traces.get(key)
        if trace is None:
            trace = HTrace([{(target, self.space.zero): self.arithmetic.one()}])
            self.traces[key] = trace
        while len(trace.layers) <= count:
            previous = trace.layers[-1]
            current = defaultdict(self.arithmetic.zero)
            for (old_target, old_config), prefix_weight in previous.items():
                for (new_target, new_other), relation_weight in self.t_update_dict[
                    (old_target, other)
                ].items():
                    new_config = self.space.inc(old_config, new_other)
                    outcome = (new_target, new_config)
                    current[outcome] = self.arithmetic.add_product(
                        current[outcome], prefix_weight, relation_weight
                    )
            trace.layers.append(dict(current))
        return trace.layers[count]


class _TraceKernel:
    def __init__(
        self,
        t_update_dict,
        space,
        accepting_states,
        has_linear_order,
        arithmetic,
        max_trace_states,
    ):
        self.t_update_dict = t_update_dict
        self.space = space
        self.accepting_states = accepting_states
        self.has_linear_order = has_linear_order
        self.arithmetic = arithmetic
        self.max_trace_states = max_trace_states
        self.updater = _TracedUpdater(t_update_dict, space, arithmetic)
        self.values = {space.zero: arithmetic.one()}
        self.nodes = {}
        self._progress_enabled = logger.isEnabledFor(logging.DEBUG)
        self._next_progress = 50_000
        target_order, other_orders = _build_elimination_orders(t_update_dict, space.offset_to_state)
        self.target_rank = {state: rank for rank, state in enumerate(target_order)}
        self.other_ranks = {
            target: {state: rank for rank, state in enumerate(order)}
            for target, order in other_orders.items()
        }

    def _accepts(self, state: State) -> bool:
        cell_index = state[0]
        return all(
            value in accepted
            for value, accepted in zip(state[1:], self.accepting_states[cell_index], strict=True)
        )

    def evaluate(self, config: Config):
        cached = self.values.get(config)
        if cached is not None:
            return cached
        if self.max_trace_states is not None and len(self.values) >= self.max_trace_states:
            raise RuntimeError(f"trace state limit exceeded ({self.max_trace_states})")

        nonzero = self.space.nonzero_states(config)
        targets = (
            nonzero if self.has_linear_order else (min(nonzero, key=self.target_rank.__getitem__),)
        )
        total = self.arithmetic.zero()
        target_traces = []
        for target in targets:
            reduced = self.space.dec(config, target)
            other_states = tuple(
                sorted(
                    self.space.nonzero_states(reduced),
                    key=self.other_ranks[target].__getitem__,
                )
            )
            layers = [{(target, self.space.zero): self.arithmetic.one()}]
            for other in other_states:
                count = self.space.count(reduced, other)
                current = defaultdict(self.arithmetic.zero)
                for (old_target, old_config), prefix_weight in layers[-1].items():
                    for (new_target, h_config), h_weight in self.updater.get(
                        old_target, other, count
                    ).items():
                        adjusted = h_weight
                        if self.has_linear_order:
                            denominator = 1
                            for value in h_config:
                                if value > 1:
                                    denominator *= math.factorial(value)
                            adjusted = self.arithmetic.multiply(
                                adjusted,
                                self.arithmetic.from_fraction(
                                    1, math.factorial(count) // denominator
                                ),
                            )
                        outcome = (new_target, _add_configs(old_config, h_config))
                        current[outcome] = self.arithmetic.add_product(
                            current[outcome], prefix_weight, adjusted
                        )
                layers.append(dict(current))
                if not current:
                    break

            terminal = defaultdict(self.arithmetic.zero)
            for (final_target, next_config), weight in layers[-1].items():
                if self._accepts(final_target):
                    terminal[next_config] = self.arithmetic.add(terminal[next_config], weight)
            for next_config, prefix_weight in terminal.items():
                total = self.arithmetic.add_product(
                    total, prefix_weight, self.evaluate(next_config)
                )
            target_traces.append(
                TargetTrace(
                    target=target,
                    other_states=other_states[: max(0, len(layers) - 1)],
                    g_layers=tuple(layers),
                    terminal_weights=dict(terminal),
                )
            )

        self.values[config] = total
        self.nodes[config] = DomainTraceNode(config, tuple(target_traces))
        state_count = len(self.values)
        if self._progress_enabled and state_count >= self._next_progress:
            logger.debug(
                "Trace DP progress states=%d nodes=%d h_traces=%d",
                state_count,
                len(self.nodes),
                len(self.updater.traces),
            )
            while self._next_progress <= state_count:
                self._next_progress += 50_000
        return total


def compile_component_trace(
    algo_input: CountingDPInput,
    component: CountingCellGraphComponent,
    *,
    reduced_problem: object | None = None,
    max_trace_states: int | None = None,
) -> ComponentTrace:
    """Compile one materialized WFOMC component into a traceback plan."""

    started = perf_counter()
    counting_state = algo_input.counting_state
    unary_masks = algo_input.unary_cardinality_masks
    if counting_state is None or unary_masks is None:
        raise WfomcCompatibilityError("incremental3 counting metadata is missing")
    arithmetic = algo_input.arithmetic
    MultinomialCoefficients.setup(algo_input.domain_size)
    t_update, transition_choices = _build_transition_tables(component, counting_state, arithmetic)
    space = ConfigSpace((len(component.cells),) + counting_state.c_type_shape)
    logger.debug(
        "Built trace transition tables domain=%d cells=%d counter_shape=%s state_types=%d "
        "transition_pairs=%d transition_outcomes=%d",
        algo_input.domain_size,
        len(component.cells),
        counting_state.c_type_shape,
        len(space.offset_to_state),
        len(t_update),
        sum(len(outcomes) for outcomes in t_update.values()),
    )
    kernel = _TraceKernel(
        t_update,
        space,
        component.counting_accepting_states,
        algo_input.has_linear_order,
        arithmetic,
        max_trace_states,
    )

    allocation = component.cell_evidence_allocation
    if allocation is None:
        allocation = CellEvidenceAllocation.unconstrained(
            len(component.cells), algo_input.domain_size
        )
    basis = (
        CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL
        if algo_input.has_linear_order
        else CellConfigCoefficientBasis.ABSOLUTE
    )
    unary_mask = unary_masks.build_mask(tuple(component.cells), component.nullary_assignments)
    roots = []
    total = arithmetic.zero()
    for cell_config, coefficient in allocation.iter_config_coefficients(arithmetic, basis):
        if unary_masks.check(cell_config, unary_mask):
            continue
        init = list(space.zero)
        cell_weight = arithmetic.one()
        valid = True
        for cell_index, count in enumerate(cell_config):
            if count == 0:
                continue
            initial = component.counting_initial_states[cell_index]
            if initial is None:
                valid = False
                break
            init[space.offset((cell_index,) + tuple(initial))] = count
            cell_weight = arithmetic.multiply(
                cell_weight,
                arithmetic.power(component.cell_weights[cell_index], count),
            )
        if not valid:
            continue
        init_config = tuple(init)
        domain_weight = kernel.evaluate(init_config)
        base_weight = arithmetic.multiply(component.graph_weight, coefficient)
        base_weight = arithmetic.multiply(base_weight, cell_weight)
        mass = arithmetic.multiply(base_weight, domain_weight)
        if arithmetic.is_zero(mass):
            continue
        roots.append(
            RootTerm(
                cell_config,
                init_config,
                base_weight,
                domain_weight,
                mass,
            )
        )
        total = arithmetic.add(total, mass)
    trace = ComponentTrace(
        component=component,
        reduced_problem=reduced_problem,
        space=space,
        arithmetic=arithmetic,
        counting_state=counting_state,
        has_linear_order=algo_input.has_linear_order,
        t_update_dict=t_update,
        transition_choices=transition_choices,
        h_traces=kernel.updater.traces,
        domain_values=kernel.values,
        domain_nodes=kernel.nodes,
        root_terms=tuple(roots),
        total_mass=total,
    )
    if logger.isEnabledFor(logging.DEBUG):
        target_traces = sum(len(node.targets) for node in kernel.nodes.values())
        g_layers = sum(
            len(target.g_layers) for node in kernel.nodes.values() for target in node.targets
        )
        g_entries = sum(
            len(layer)
            for node in kernel.nodes.values()
            for target in node.targets
            for layer in target.g_layers
        )
        h_layers = sum(len(h_trace.layers) for h_trace in kernel.updater.traces.values())
        h_entries = sum(
            len(layer) for h_trace in kernel.updater.traces.values() for layer in h_trace.layers
        )
        logger.debug(
            "Compiled component trace roots=%d states=%d nodes=%d target_traces=%d "
            "g_layers=%d g_entries=%d h_traces=%d h_layers=%d h_entries=%d "
            "elapsed_ms=%.3f",
            len(roots),
            len(kernel.values),
            len(kernel.nodes),
            target_traces,
            g_layers,
            g_entries,
            len(kernel.updater.traces),
            h_layers,
            h_entries,
            (perf_counter() - started) * 1000,
        )
    return trace
