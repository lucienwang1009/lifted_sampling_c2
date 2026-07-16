"""Direct semantic validation of one sampled source-level structure."""

from __future__ import annotations

import logging
from itertools import product
from time import perf_counter

from wfomc.fol import (
    And,
    Atom,
    BoolConst,
    Constant,
    CountingQuantifier,
    Eq,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
    Variable,
)

from .errors import StructureValidationError
from .projection import source_predicate_keys
from .structure import PredicateKey, SampledStructure

logger = logging.getLogger(__name__)


def _assignment_text(assignment: dict[object, object]) -> str:
    if not assignment:
        return "{}"
    pairs = sorted((str(variable), str(value)) for variable, value in assignment.items())
    return "{" + ", ".join(f"{variable}={value}" for variable, value in pairs) + "}"


def _resolve_term(term, assignment: dict[object, object]):
    if isinstance(term, Variable):
        try:
            return assignment[term]
        except KeyError as exc:
            raise StructureValidationError(f"formula has an unbound variable: {term}") from exc
    if isinstance(term, Constant):
        return term
    return term


def _predicate_key(predicate, arity: int | None = None) -> PredicateKey:
    name = getattr(predicate, "name", str(predicate))
    actual_arity = getattr(predicate, "arity", arity)
    if actual_arity is None:
        raise StructureValidationError(f"cannot infer arity for predicate {name}")
    return PredicateKey(name, actual_arity)


def _quantified_domain(variable, domain: tuple[object, ...]) -> tuple[object, ...]:
    sort = getattr(variable, "sort", None)
    if sort is None:
        return domain
    return tuple(value for value in domain if getattr(value, "sort", None) == sort)


def _count_accepts(observed: int, comparator: str, target) -> bool:
    if comparator == "=":
        return observed == target
    if comparator == "!=":
        return observed != target
    if comparator == "<":
        return observed < target
    if comparator == "<=":
        return observed <= target
    if comparator == ">":
        return observed > target
    if comparator == ">=":
        return observed >= target
    if comparator == "mod":
        if hasattr(target, "remainder") and hasattr(target, "modulus"):
            remainder, modulus = target.remainder, target.modulus
        else:
            remainder, modulus = target
        return observed % modulus == remainder % modulus
    raise StructureValidationError(f"unsupported counting comparator: {comparator}")


def _evaluate(
    formula,
    domain: tuple[object, ...],
    relations: dict[PredicateKey, frozenset[tuple[object, ...]]],
    assignment: dict[object, object],
) -> tuple[bool, str | None]:
    if isinstance(formula, BoolConst):
        return formula.value, None if formula.value else "formula contains false"

    if isinstance(formula, Atom):
        key = _predicate_key(formula.predicate)
        terms = tuple(_resolve_term(term, assignment) for term in formula.terms)
        valid = terms in relations.get(key, frozenset())
        message = (
            None if valid else f"false atom {key.name}{terms} under {_assignment_text(assignment)}"
        )
        return valid, message

    if isinstance(formula, Eq):
        left = _resolve_term(formula.left, assignment)
        right = _resolve_term(formula.right, assignment)
        valid = left == right
        return valid, None if valid else f"equality is false: {left} != {right}"

    if isinstance(formula, Not):
        body_valid, _ = _evaluate(formula.body, domain, relations, assignment)
        valid = not body_valid
        return (
            valid,
            None if valid else f"negation is false under {_assignment_text(assignment)}: {formula}",
        )

    if isinstance(formula, And):
        for argument in formula.args:
            valid, message = _evaluate(argument, domain, relations, assignment)
            if not valid:
                return False, message
        return True, None

    if isinstance(formula, Or):
        for argument in formula.args:
            valid, _ = _evaluate(argument, domain, relations, assignment)
            if valid:
                return True, None
        return False, f"disjunction is false under {_assignment_text(assignment)}: {formula}"

    if isinstance(formula, Implies):
        left, _ = _evaluate(formula.left, domain, relations, assignment)
        right, _ = _evaluate(formula.right, domain, relations, assignment)
        valid = not left or right
        return valid, None if valid else f"implication is false: {formula}"

    if isinstance(formula, Iff):
        left, _ = _evaluate(formula.left, domain, relations, assignment)
        right, _ = _evaluate(formula.right, domain, relations, assignment)
        valid = left == right
        return valid, None if valid else f"equivalence has different truth values: {formula}"

    if isinstance(formula, Quantifier):
        domains = tuple(_quantified_domain(variable, domain) for variable in formula.variables)
        values = product(*domains)
        if formula.kind == QuantifierKind.FORALL:
            for combination in values:
                extended = dict(assignment)
                extended.update(zip(formula.variables, combination, strict=True))
                valid, message = _evaluate(formula.body, domain, relations, extended)
                if not valid:
                    return (
                        False,
                        f"universal counterexample {_assignment_text(extended)}: {message}",
                    )
            return True, None
        if formula.kind == QuantifierKind.EXISTS:
            for combination in values:
                extended = dict(assignment)
                extended.update(zip(formula.variables, combination, strict=True))
                valid, _ = _evaluate(formula.body, domain, relations, extended)
                if valid:
                    return True, None
            return False, f"existential quantifier has no witness: {formula}"
        raise StructureValidationError(f"unsupported quantifier kind: {formula.kind}")

    if isinstance(formula, CountingQuantifier):
        values = _quantified_domain(formula.variable, domain)
        observed = 0
        for value in values:
            extended = dict(assignment)
            extended[formula.variable] = value
            valid, _ = _evaluate(formula.body, domain, relations, extended)
            observed += int(valid)
        valid = _count_accepts(observed, formula.comparator, formula.count)
        message = None
        if not valid:
            message = (
                f"counting constraint is false under {_assignment_text(assignment)}: "
                f"observed={observed}, comparator={formula.comparator}, "
                f"target={formula.count}"
            )
        return valid, message

    if isinstance(formula, Formula):
        raise StructureValidationError(f"unsupported formula node: {type(formula).__name__}")
    raise StructureValidationError(f"expected a formula node, got {type(formula).__name__}")


def _relation_map(problem, sample: SampledStructure):
    if len(set(sample.domain)) != len(sample.domain):
        raise StructureValidationError("domain contains duplicate elements")
    actual_domain = frozenset(sample.domain)
    expected_domain = frozenset(problem.domain)
    if actual_domain != expected_domain:
        missing = sorted(map(str, expected_domain - actual_domain))
        extra = sorted(map(str, actual_domain - expected_domain))
        raise StructureValidationError(
            f"domain differs from problem: missing={missing}, extra={extra}"
        )

    source_keys = frozenset(source_predicate_keys(problem))
    relations: dict[PredicateKey, frozenset[tuple[object, ...]]] = {}
    for key, tuples in sample.relations:
        if key in relations:
            raise StructureValidationError(f"relation occurs more than once: {key}")
        if key not in source_keys:
            raise StructureValidationError(f"unexpected non-source relation: {key}")
        for terms in tuples:
            if len(terms) != key.arity:
                raise StructureValidationError(
                    f"relation {key.name}/{key.arity} contains tuple with arity {len(terms)}"
                )
            outside = tuple(term for term in terms if term not in actual_domain)
            if outside:
                raise StructureValidationError(
                    f"relation {key.name}/{key.arity} contains values outside domain: {outside}"
                )
        relations[key] = tuples
    return tuple(sample.domain), source_keys, relations


def _validate_evidence(problem, relations) -> None:
    for literal in problem.evidence.unary.literals:
        key = _predicate_key(literal.predicate, 1)
        observed = (literal.constant,) in relations.get(key, frozenset())
        if observed != literal.positive:
            raise StructureValidationError(
                f"unary evidence is false: predicate={key.name}, "
                f"constant={literal.constant}, expected={literal.positive}"
            )
    for literal in problem.evidence.binary.literals:
        key = _predicate_key(literal.predicate, 2)
        observed = (literal.left, literal.right) in relations.get(key, frozenset())
        if observed != literal.positive:
            raise StructureValidationError(
                f"binary evidence is false: predicate={key.name}, "
                f"terms=({literal.left}, {literal.right}), expected={literal.positive}"
            )


def _validate_cardinality(problem, source_keys, relations) -> None:
    for constraint in problem.cardinality_constraints.constraints:
        observed = 0
        for term in constraint.terms:
            predicate = term.predicate
            arity = getattr(predicate, "arity", None)
            if arity is None:
                matches = tuple(key for key in source_keys if key.name == str(predicate))
                if len(matches) != 1:
                    raise StructureValidationError(
                        f"cannot infer cardinality predicate arity: {predicate}"
                    )
                key = matches[0]
            else:
                key = _predicate_key(predicate)
            observed += term.coefficient * len(relations.get(key, frozenset()))
        if not constraint.accepts(observed):
            raise StructureValidationError(
                f"cardinality constraint is false: observed={observed}, "
                f"comparator={constraint.comparator}, rhs={constraint.rhs}, "
                f"modulus={constraint.modulus}"
            )


def _validate_leq(domain, source_keys, relations) -> None:
    key = PredicateKey("LEQ", 2)
    if key not in source_keys:
        return
    leq = relations.get(key, frozenset())
    for left in domain:
        if (left, left) not in leq:
            raise StructureValidationError(f"LEQ is not reflexive at {left}")
        for right in domain:
            if left == right:
                continue
            forward = (left, right) in leq
            reverse = (right, left) in leq
            if forward == reverse:
                raise StructureValidationError(
                    f"LEQ is not a total antisymmetric order for {left}, {right}"
                )
    for left in domain:
        for middle in domain:
            if (left, middle) not in leq:
                continue
            for right in domain:
                if (middle, right) in leq and (left, right) not in leq:
                    raise StructureValidationError(
                        f"LEQ is not transitive: {left} <= {middle} <= {right}"
                    )


def validate_structure(problem, sample: SampledStructure) -> None:
    """Raise ``StructureValidationError`` unless *sample* models *problem*.

    This is an explicit diagnostic operation. It is not called during normal
    sampling because direct formula evaluation can be quadratic in the domain,
    and the LEQ transitivity check is cubic.
    """

    started = perf_counter()
    domain, source_keys, relations = _relation_map(problem, sample)
    logger.debug(
        "Structure validation started domain=%d predicates=%d true_tuples=%d",
        len(domain),
        len(source_keys),
        sum(len(tuples) for tuples in relations.values()),
    )
    free_variables = problem.sentence.free_vars()
    if free_variables:
        names = ", ".join(sorted(map(str, free_variables)))
        raise StructureValidationError(f"problem sentence has free variables: {names}")
    formula_started = perf_counter()
    valid, message = _evaluate(problem.sentence, domain, relations, {})
    if not valid:
        raise StructureValidationError(f"formula is false: {message}")
    formula_ms = (perf_counter() - formula_started) * 1000

    evidence_started = perf_counter()
    _validate_evidence(problem, relations)
    evidence_ms = (perf_counter() - evidence_started) * 1000

    cardinality_started = perf_counter()
    _validate_cardinality(problem, source_keys, relations)
    cardinality_ms = (perf_counter() - cardinality_started) * 1000

    leq_started = perf_counter()
    _validate_leq(domain, source_keys, relations)
    leq_ms = (perf_counter() - leq_started) * 1000
    logger.debug(
        "Structure validation completed formula_ms=%.3f evidence_ms=%.3f "
        "cardinality_ms=%.3f leq_ms=%.3f elapsed_ms=%.3f",
        formula_ms,
        evidence_ms,
        cardinality_ms,
        leq_ms,
        (perf_counter() - started) * 1000,
    )


__all__ = ["validate_structure"]
