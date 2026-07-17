"""Project a reduced WFOMC trace model back to the source vocabulary."""

from __future__ import annotations

from wfomc.fol import Predicate, predicates

from c2_wms.errors import UnsupportedSamplingInput
from c2_wms.structure import PredicateKey, SampledStructure


def projection_metadata(
    trace,
    source_keys: tuple[PredicateKey, ...],
) -> tuple[
    tuple[tuple[tuple[PredicateKey, int], ...], ...],
    tuple[PredicateKey, ...],
    tuple[tuple[PredicateKey, bool] | None, ...],
]:
    """Precompute source-level cell, nullary, and pair projection actions."""

    source_key_set = frozenset(source_keys)
    cell_actions = []
    for cell in trace.component.cells:
        actions = []
        for predicate in cell.preds:
            key = PredicateKey(predicate.name, predicate.arity)
            if key in source_key_set and cell.is_positive(predicate):
                actions.append((key, predicate.arity))
        cell_actions.append(tuple(actions))
    nullary_keys = []
    for predicate, positive in trace.component.nullary_assignments:
        key = PredicateKey(predicate.name, predicate.arity)
        if positive and key in source_key_set:
            nullary_keys.append(key)
    actions: list[tuple[PredicateKey, bool] | None] = [None] * (
        2 * len(trace.counting_state.projected_predicates)
    )
    for index, predicate in enumerate(trace.counting_state.projected_predicates):
        key = PredicateKey(predicate.name, predicate.arity)
        if predicate.arity != 2 or key not in source_key_set:
            continue
        actions[2 * index] = (key, True)
        actions[2 * index + 1] = (key, False)
    return tuple(cell_actions), tuple(nullary_keys), tuple(actions)


def source_predicate_keys(problem) -> tuple[PredicateKey, ...]:
    keys = {
        PredicateKey(predicate.name, predicate.arity) for predicate in predicates(problem.sentence)
    }

    def add(predicate, arity: int | None = None) -> None:
        if isinstance(predicate, Predicate):
            keys.add(PredicateKey(predicate.name, predicate.arity))
        elif arity is not None:
            keys.add(PredicateKey(str(predicate), arity))
        else:
            matches = {key.arity for key in keys if key.name == str(predicate)}
            if len(matches) != 1:
                raise UnsupportedSamplingInput(
                    f"cannot infer the arity of source predicate {predicate!r}"
                )

    for predicate in problem.weights:
        add(predicate)
    for literal in problem.evidence.unary.literals:
        add(literal.predicate, 1)
    for literal in problem.evidence.binary.literals:
        add(literal.predicate, 2)
    for constraint in problem.cardinality_constraints.constraints:
        for term in constraint.terms:
            add(term.predicate)
    return tuple(sorted(keys))


def project_structure(
    problem,
    anonymous,
    labels,
    pair_sampler,
    metadata,
    *,
    source_keys: tuple[PredicateKey, ...] | None = None,
) -> SampledStructure:
    keys = source_predicate_keys(problem) if source_keys is None else source_keys
    relations: dict[PredicateKey, set[tuple[object, ...]]] = {key: set() for key in keys}
    cell_actions, nullary_keys, direct_bits = metadata
    for label, cell_index in zip(labels, anonymous.cell_indices, strict=True):
        for key, arity in cell_actions[cell_index]:
            relations[key].add((label,) * arity)

    for key in nullary_keys:
        relations[key].add(())

    source_actions = pair_sampler.source_actions
    for request in anonymous.pair_requests:
        if request.source_mask is not None:
            mask = request.source_mask
            actions = source_actions
        elif pair_sampler.is_direct:
            mask = request.projection_mask
            actions = direct_bits
        else:
            mask = pair_sampler.sample_mask(request)
            actions = source_actions
        while mask:
            bit = mask & -mask
            action = actions[bit.bit_length() - 1]
            if action is not None:
                key, reverse = action
                left, right = request.left, request.right
                terms = (labels[right], labels[left]) if reverse else (labels[left], labels[right])
                relations[key].add(terms)
            mask ^= bit
    return SampledStructure(
        tuple(labels),
        tuple((key, frozenset(relations[key])) for key in keys),
    )


__all__ = ["project_structure", "projection_metadata", "source_predicate_keys"]
