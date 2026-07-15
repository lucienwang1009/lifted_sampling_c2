import pytest
from wfomc import parse_problem

from c2_wms import (
    PredicateKey,
    SampledStructure,
    StructureValidationError,
    compile_sampler,
    validate_structure,
)


@pytest.mark.parametrize(
    "source",
    (
        r"""
\forall X: (\exists_=1 Y: (R(X,Y) & S(Y,X))) &
\forall X: (P(X) | ~P(X))
domain = 3
|R| = 3
P(domain0)
""",
        r"""
A <-> (\exists_{1mod2} X: U(X))
domain = 3
2 3 A
""",
        r"""
\forall X: (\forall Y: (LEQ(X,Y) | ~LEQ(X,Y)))
domain = 3
""",
    ),
)
def test_sampled_structures_validate_against_source_problem(source):
    problem = parse_problem(source)
    with compile_sampler(problem, seed=23) as sampler:
        for sample in sampler.sample_many(10):
            assert validate_structure(problem, sample) is None


def test_validation_reports_formula_counterexample():
    problem = parse_problem(r"""
\forall X: P(X)
domain = 2
""")
    domain = tuple(sorted(problem.domain, key=str))
    sample = SampledStructure.from_mapping(
        domain,
        {PredicateKey("P", 1): {(domain[0],)}},
    )

    with pytest.raises(StructureValidationError, match="universal counterexample"):
        validate_structure(problem, sample)


@pytest.mark.parametrize(
    ("quantifier", "positive_count"),
    (
        ("=2", 2),
        ("!=1", 2),
        ("<3", 2),
        ("<=2", 2),
        (">1", 2),
        (">=2", 2),
        ("1mod2", 1),
    ),
)
def test_validation_interprets_every_count_comparator(quantifier, positive_count):
    problem = parse_problem(
        rf"""
\exists_{{{quantifier}}} X: P(X)
domain = 3
"""
    )
    domain = tuple(sorted(problem.domain, key=str))
    sample = SampledStructure.from_mapping(
        domain,
        {PredicateKey("P", 1): {(element,) for element in domain[:positive_count]}},
    )

    assert validate_structure(problem, sample) is None


def test_validation_checks_evidence_and_cardinality_metadata():
    evidence_problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 2
P(domain0)
""")
    evidence_sample = SampledStructure.from_mapping(
        sorted(evidence_problem.domain, key=str),
        {PredicateKey("P", 1): set()},
    )
    with pytest.raises(StructureValidationError, match="unary evidence"):
        validate_structure(evidence_problem, evidence_sample)

    cardinality_problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 2
|P| = 1
""")
    cardinality_sample = SampledStructure.from_mapping(
        sorted(cardinality_problem.domain, key=str),
        {PredicateKey("P", 1): set()},
    )
    with pytest.raises(StructureValidationError, match="cardinality constraint"):
        validate_structure(cardinality_problem, cardinality_sample)


def test_validation_rejects_non_transitive_leq():
    problem = parse_problem(r"""
\forall X: (\forall Y: (LEQ(X,Y) | ~LEQ(X,Y)))
domain = 3
""")
    left, middle, right = sorted(problem.domain, key=str)
    relation = {
        (left, left),
        (middle, middle),
        (right, right),
        (left, middle),
        (middle, right),
        (right, left),
    }
    sample = SampledStructure.from_mapping(
        (left, middle, right),
        {PredicateKey("LEQ", 2): relation},
    )

    with pytest.raises(StructureValidationError, match="not transitive"):
        validate_structure(problem, sample)


def test_validation_rejects_invalid_structure_shape_and_auxiliary_relation():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 1
""")
    (element,) = tuple(problem.domain)
    wrong_domain = SampledStructure.from_mapping(
        ("outside",),
        {PredicateKey("P", 1): {("outside",)}},
    )
    with pytest.raises(StructureValidationError, match="domain differs"):
        validate_structure(problem, wrong_domain)

    auxiliary = SampledStructure.from_mapping(
        (element,),
        {
            PredicateKey("P", 1): set(),
            PredicateKey("@aux", 1): {(element,)},
        },
    )
    with pytest.raises(StructureValidationError, match="non-source"):
        validate_structure(problem, auxiliary)
