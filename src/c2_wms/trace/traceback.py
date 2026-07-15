"""Conditioned traceback through compact incremental3 value traces."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from c2_wms.arithmetic import CoefficientCache, Degree
from c2_wms.discrete_sampling import ExactAliasTable, RandRange
from c2_wms.errors import SamplingError

from .model import ComponentTrace, RelationChoice, RootTerm


def _subtract_configs(total, part):
    result = tuple(a - b for a, b in zip(total, part, strict=True))
    return None if any(value < 0 for value in result) else result


@dataclass(slots=True)
class _Element:
    identifier: int
    cell_index: int
    entry_state: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PairRequest:
    left: int
    right: int
    left_cell: int
    right_cell: int
    projection_mask: int
    degree: Degree


@dataclass(frozen=True, slots=True)
class AnonymousSample:
    trace: ComponentTrace
    cell_config: tuple[int, ...]
    cell_indices: tuple[int, ...]
    pair_requests: tuple[PairRequest, ...]


class TracebackSampler:
    def __init__(self, trace: ComponentTrace, rng: RandRange):
        self.trace = trace
        self.rng = rng
        dimension = len(trace.arithmetic.symbolic_variables)
        self.coefficients = CoefficientCache(dimension)
        self._aliases: dict[object, ExactAliasTable] = {}
        self._next_identifier = 0
        self._pairs: list[PairRequest] = []

    def _value_choice(self, key, pairs, degree):
        cache_key = ("value", key, degree)
        table = self._aliases.get(cache_key)
        if table is None:
            choices = []
            weights = []
            for choice, value in pairs:
                coefficient = self.coefficients.coefficient(value, degree)
                if coefficient > 0:
                    choices.append(choice)
                    weights.append(coefficient)
                elif coefficient < 0:
                    raise SamplingError("trace contains a negative coefficient")
            table = ExactAliasTable(choices, weights)
            self._aliases[cache_key] = table
        return table.sample(self.rng)

    def _product_choice(self, key, triples, degree):
        cache_key = ("product", key, degree)
        table = self._aliases.get(cache_key)
        if table is None:
            choices = []
            weights = []
            for choice, left, right in triples:
                for left_degree, right_degree, weight in self.coefficients.product_splits(
                    left, right, degree
                ):
                    if weight < 0:
                        raise SamplingError("trace contains a negative coefficient")
                    choices.append((choice, left_degree, right_degree))
                    weights.append(weight)
            table = ExactAliasTable(choices, weights)
            self._aliases[cache_key] = table
        return table.sample(self.rng)

    def sample(self, root: RootTerm, degree: Degree) -> AnonymousSample:
        self._next_identifier = 0
        self._pairs = []
        elements = self._sample_domain(root.init_config, degree)
        elements.sort(key=lambda element: element.identifier)
        return AnonymousSample(
            self.trace,
            root.cell_config,
            tuple(element.cell_index for element in elements),
            tuple(self._pairs),
        )

    def _sample_domain(self, config, degree):
        if sum(config) == 0:
            if (
                self.coefficients.coefficient(
                    self.trace.domain_values[self.trace.space.zero], degree
                )
                <= 0
            ):
                raise SamplingError("invalid terminal degree budget")
            return []

        node = self.trace.domain_nodes[config]
        domain_choices = []
        for target_index, target_trace in enumerate(node.targets):
            for next_config, terminal_weight in target_trace.terminal_weights.items():
                domain_choices.append(
                    (
                        (target_index, next_config),
                        terminal_weight,
                        self.trace.domain_values[next_config],
                    )
                )
        (target_index, next_config), terminal_degree, next_degree = self._product_choice(
            ("domain", config), domain_choices, degree
        )
        target_trace = node.targets[target_index]
        remaining = self._sample_domain(next_config, next_degree)

        final_pairs = [
            ((target_state, g_config), weight)
            for (target_state, g_config), weight in target_trace.g_layers[-1].items()
            if g_config == next_config and self._accepts(target_state)
        ]
        current_target, current_g = self._value_choice(
            ("terminal", config, target_index, next_config),
            final_pairs,
            terminal_degree,
        )

        remaining_by_state = defaultdict(list)
        for element in remaining:
            remaining_by_state[element.entry_state].append(element)

        current_degree = terminal_degree
        for layer_index in range(len(target_trace.other_states), 0, -1):
            other = target_trace.other_states[layer_index - 1]
            count = self.trace.space.count(self.trace.space.dec(config, target_trace.target), other)
            previous_layer = target_trace.g_layers[layer_index - 1]
            triples = []
            for (old_target, old_g), prefix_weight in previous_layer.items():
                h_trace = self.trace.h_traces.get((old_target, other))
                if h_trace is None or count >= len(h_trace.layers):
                    continue
                for (new_target, h_config), h_weight in h_trace.layers[count].items():
                    if new_target != current_target:
                        continue
                    if tuple(a + b for a, b in zip(old_g, h_config, strict=True)) != current_g:
                        continue
                    adjusted = h_weight
                    if self.trace.has_linear_order:
                        denominator = 1
                        for value in h_config:
                            if value > 1:
                                denominator *= math.factorial(value)
                        adjusted = self.trace.arithmetic.multiply(
                            adjusted,
                            self.trace.arithmetic.from_fraction(
                                1, math.factorial(count) // denominator
                            ),
                        )
                    triples.append(
                        (
                            (old_target, old_g, h_config),
                            prefix_weight,
                            adjusted,
                        )
                    )
            (
                (
                    old_target,
                    old_g,
                    h_config,
                ),
                prefix_degree,
                h_degree,
            ) = self._product_choice(
                ("g", config, target_index, layer_index, current_target, current_g),
                triples,
                current_degree,
            )
            self._reverse_h(
                old_target,
                other,
                count,
                current_target,
                h_config,
                h_degree,
                remaining_by_state,
                config,
                target_index,
                layer_index,
            )
            current_target = old_target
            current_g = old_g
            current_degree = prefix_degree

        if current_target != target_trace.target or current_g != self.trace.space.zero:
            raise SamplingError("G traceback did not reach its initial state")
        if any(remaining_by_state.values()):
            raise SamplingError("traceback left unassigned remaining elements")

        target = _Element(
            self._next_identifier,
            target_trace.target[0],
            target_trace.target,
        )
        self._next_identifier += 1
        # Pair requests were temporarily recorded with a sentinel target id.
        for index, request in enumerate(self._pairs):
            if request.left == -1:
                self._pairs[index] = PairRequest(
                    target.identifier,
                    request.right,
                    request.left_cell,
                    request.right_cell,
                    request.projection_mask,
                    request.degree,
                )
        return [*remaining, target]

    def _reverse_h(
        self,
        initial_target,
        other,
        count,
        final_target,
        final_config,
        degree,
        remaining_by_state,
        domain_config,
        target_index,
        layer_index,
    ):
        trace = self.trace.h_traces[(initial_target, other)]
        current_target = final_target
        current_config = final_config
        current_degree = degree
        for step in range(count, 0, -1):
            triples = []
            for (old_target, old_config), prefix_weight in trace.layers[step - 1].items():
                for (new_target, new_other), transition_weight in self.trace.t_update_dict[
                    (old_target, other)
                ].items():
                    if new_target != current_target:
                        continue
                    expected = self.trace.space.inc(old_config, new_other)
                    if expected != current_config:
                        continue
                    triples.append(
                        (
                            (old_target, old_config, new_other),
                            prefix_weight,
                            transition_weight,
                        )
                    )
            (
                (
                    old_target,
                    old_config,
                    new_other,
                ),
                prefix_degree,
                relation_degree,
            ) = self._product_choice(
                (
                    "h",
                    domain_config,
                    target_index,
                    layer_index,
                    initial_target,
                    other,
                    step,
                    current_target,
                    current_config,
                ),
                triples,
                current_degree,
            )
            relation_choices: list[RelationChoice] = self.trace.transition_choices[
                (old_target, other)
            ][(current_target, new_other)]
            relation = self._value_choice(
                (
                    "relation",
                    old_target,
                    other,
                    current_target,
                    new_other,
                ),
                [(choice, choice.weight) for choice in relation_choices],
                relation_degree,
            )
            bucket = remaining_by_state[new_other]
            if not bucket:
                raise SamplingError(f"no remaining element in state {new_other!r}")
            element = bucket.pop()
            element.entry_state = other
            self._pairs.append(
                PairRequest(
                    -1,
                    element.identifier,
                    old_target[0],
                    other[0],
                    relation.mask,
                    relation_degree,
                )
            )
            current_target = old_target
            current_config = old_config
            current_degree = prefix_degree
        if current_target != initial_target or current_config != self.trace.space.zero:
            raise SamplingError(
                "H traceback did not reach its initial state: "
                f"target={current_target!r}, expected={initial_target!r}, "
                f"config={current_config!r}"
            )

    def _accepts(self, state):
        cell_index = state[0]
        return all(
            value in accepted
            for value, accepted in zip(
                state[1:],
                self.trace.component.counting_accepting_states[cell_index],
                strict=True,
            )
        )
