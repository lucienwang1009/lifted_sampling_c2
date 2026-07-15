"""Uniformly restore concrete domain labels from lifted cell profiles."""

from __future__ import annotations

import math

from wfomc.cell_graph import CellEvidenceAllocation
from wfomc.multinomial import multinomial

from c2_wms.discrete_sampling import ExactAliasTable, RandRange
from c2_wms.errors import SamplingError, WfomcCompatibilityError


def _predicate_key(predicate) -> tuple[str, int]:
    if isinstance(predicate, str):
        return predicate, 1
    return predicate.name, predicate.arity


def _profile_key(literals) -> frozenset[tuple[str, int, bool]]:
    return frozenset((*_predicate_key(literal.predicate), literal.positive) for literal in literals)


def _multinomial_coefficient(distribution: tuple[int, ...]) -> int:
    result = math.factorial(sum(distribution))
    for count in distribution:
        result //= math.factorial(count)
    return result


class LabelSampler:
    """Sample profile-to-cell allocations and bind real domain constants."""

    def __init__(self, problem, rng: RandRange):
        self.problem = problem
        self.rng = rng
        self.domain = tuple(sorted(problem.domain, key=str))
        by_constant: dict[object, set[tuple[str, int, bool]]] = {
            constant: set() for constant in self.domain
        }
        for literal in problem.evidence.unary.literals:
            by_constant[literal.constant].add(
                (*_predicate_key(literal.predicate), literal.positive)
            )
        self.constants_by_profile: dict[frozenset[tuple[str, int, bool]], tuple[object, ...]] = {}
        mutable: dict[frozenset[tuple[str, int, bool]], list[object]] = {}
        for constant in self.domain:
            key = frozenset(by_constant[constant])
            mutable.setdefault(key, []).append(constant)
        self.constants_by_profile = {key: tuple(values) for key, values in mutable.items()}
        self._masses: dict[tuple[int, int, tuple[int, ...]], int] = {}
        self._transitions: dict[
            tuple[int, int, tuple[int, ...]],
            tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...],
        ] = {}
        self._aliases: dict[object, ExactAliasTable] = {}
        self._allocations: dict[int, CellEvidenceAllocation] = {}
        self._profile_constant_cache: dict[int, tuple[tuple[object, ...], ...]] = {}

    def _allocation(self, trace) -> CellEvidenceAllocation:
        allocation = trace.component.cell_evidence_allocation
        if allocation is None:
            key = id(trace.component)
            allocation = self._allocations.get(key)
            if allocation is None:
                allocation = CellEvidenceAllocation.unconstrained(
                    len(trace.component.cells), len(self.domain)
                )
                self._allocations[key] = allocation
        return allocation

    def _profile_constants(self, allocation):
        cached = self._profile_constant_cache.get(id(allocation))
        if cached is not None:
            return cached
        if not allocation.profile_literals:
            result = (self.domain,)
            self._profile_constant_cache[id(allocation)] = result
            return result
        result = tuple(
            self.constants_by_profile.get(_profile_key(literals), ())
            for literals in allocation.profile_literals
        )
        actual_sizes = tuple(map(len, result))
        if actual_sizes != allocation.evidence_profile_sizes:
            raise WfomcCompatibilityError(
                "source unary-evidence profiles do not match WFOMC allocation: "
                f"observed={actual_sizes}, "
                f"expected={allocation.evidence_profile_sizes}"
            )
        self._profile_constant_cache[id(allocation)] = result
        return result

    def _mass(self, allocation, profile_index: int, remaining: tuple[int, ...]) -> int:
        key = (id(allocation), profile_index, remaining)
        cached = self._masses.get(key)
        if cached is not None:
            return cached
        if profile_index == len(allocation.evidence_profile_sizes):
            result = int(not any(remaining))
            self._masses[key] = result
            self._transitions[key] = ()
            return result

        size = allocation.evidence_profile_sizes[profile_index]
        compatible = allocation.compatible_cells_by_evidence_profile[profile_index]
        transitions = []
        total = 0
        for distribution in multinomial(len(compatible), size):
            if any(
                count > remaining[cell_index]
                for cell_index, count in zip(compatible, distribution, strict=True)
            ):
                continue
            next_remaining = list(remaining)
            for cell_index, count in zip(compatible, distribution, strict=True):
                next_remaining[cell_index] -= count
            next_tuple = tuple(next_remaining)
            suffix = self._mass(allocation, profile_index + 1, next_tuple)
            if suffix == 0:
                continue
            weight = _multinomial_coefficient(distribution) * suffix
            transitions.append((distribution, next_tuple, weight))
            total += weight
        self._masses[key] = total
        self._transitions[key] = tuple(transitions)
        return total

    def _choose(self, allocation, profile_index, remaining):
        key = (id(allocation), profile_index, remaining)
        if self._mass(allocation, profile_index, remaining) == 0:
            raise SamplingError("cell configuration cannot realize evidence profiles")
        table = self._aliases.get(key)
        if table is None:
            transitions = self._transitions[key]
            table = ExactAliasTable(
                [(distribution, next_remaining) for distribution, next_remaining, _ in transitions],
                [weight for _, _, weight in transitions],
            )
            self._aliases[key] = table
        return table.sample(self.rng)

    def sample(self, anonymous) -> tuple[object, ...]:
        """Return labels aligned with the anonymous identifiers in a trace."""

        trace = anonymous.trace
        cell_config = anonymous.cell_config
        cell_indices = anonymous.cell_indices
        if len(cell_indices) != len(self.domain):
            raise SamplingError("trace/domain size mismatch")
        allocation = self._allocation(trace)
        profiles = self._profile_constants(allocation)
        remaining = cell_config
        buckets: list[list[object]] = [[] for _ in range(len(trace.component.cells))]
        for profile_index, constants in enumerate(profiles):
            distribution, remaining = self._choose(allocation, profile_index, remaining)
            shuffled = list(constants)
            self.rng.shuffle(shuffled)
            offset = 0
            compatible = allocation.compatible_cells_by_evidence_profile[profile_index]
            for cell_index, count in zip(compatible, distribution, strict=True):
                buckets[cell_index].extend(shuffled[offset : offset + count])
                offset += count
        if any(remaining):
            raise SamplingError("evidence allocation left unfilled cell positions")

        identifiers_by_cell: list[list[int]] = [[] for _ in range(len(trace.component.cells))]
        for identifier, cell_index in enumerate(cell_indices):
            identifiers_by_cell[cell_index].append(identifier)
        labels: list[object | None] = [None] * len(cell_indices)
        for cell_index, identifiers in enumerate(identifiers_by_cell):
            values = buckets[cell_index]
            if len(values) != len(identifiers):
                raise SamplingError("evidence allocation produced a wrong cell size")
            self.rng.shuffle(values)
            for identifier, value in zip(identifiers, values, strict=True):
                labels[identifier] = value
        if any(value is None for value in labels):
            raise SamplingError("not every anonymous element received a label")
        return tuple(labels)


__all__ = ["LabelSampler"]
