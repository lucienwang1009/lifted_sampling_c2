"""Project a reduced WFOMC trace model back to the source vocabulary."""

from __future__ import annotations

from wfomc.fol import Predicate, a, b, predicates

from c2_wms.errors import SamplingError, UnsupportedSamplingInput
from c2_wms.structure import PredicateKey, SampledStructure


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
    sampled_pairs,
    *,
    source_keys: tuple[PredicateKey, ...] | None = None,
) -> SampledStructure:
    keys = source_predicate_keys(problem) if source_keys is None else source_keys
    relations: dict[PredicateKey, set[tuple[object, ...]]] = {key: set() for key in keys}
    key_set = frozenset(keys)
    for label, cell_index in zip(labels, anonymous.cell_indices, strict=True):
        cell = anonymous.trace.component.cells[cell_index]
        for predicate in cell.preds:
            key = PredicateKey(predicate.name, predicate.arity)
            if key in key_set and cell.is_positive(predicate):
                relations[key].add((label,) * predicate.arity)

    for predicate, positive in anonymous.trace.component.nullary_assignments:
        key = PredicateKey(predicate.name, predicate.arity)
        if positive and key in key_set:
            relations[key].add(())

    for request, atoms in sampled_pairs:
        for atom in atoms:
            key = PredicateKey(atom.predicate.name, atom.predicate.arity)
            if key not in key_set:
                continue
            if atom.terms == (a, b):
                terms = (labels[request.left], labels[request.right])
            elif atom.terms == (b, a):
                terms = (labels[request.right], labels[request.left])
            else:
                raise SamplingError(f"unexpected pair atom orientation: {atom}")
            relations[key].add(terms)
    return SampledStructure.from_mapping(labels, relations)


__all__ = ["project_structure", "source_predicate_keys"]
