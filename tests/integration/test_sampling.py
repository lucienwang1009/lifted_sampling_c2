import pytest
from wfomc import parse_problem

from c2_wms import compile_sampler


def test_general_c2_counting_body_projects_only_source_vocabulary():
    problem = parse_problem(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & S(Y,X)))
domain = 3
""")
    sampler = compile_sampler(problem, seed=31)

    for _ in range(20):
        sample = sampler.sample()
        for element in sample.domain:
            witnesses = sum(
                (element, other) in sample.relation("R", 2)
                and (other, element) in sample.relation("S", 2)
                for other in sample.domain
            )
            assert witnesses == 1
        assert all(not key.name.startswith("@c2_") for key, _ in sample.relations)


def test_seeded_sampling_is_reproducible_and_respects_evidence():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 3
P(domain0)
""")
    left = compile_sampler(problem, seed=8)
    right = compile_sampler(problem, seed=8)

    left_samples = tuple(left.sample() for _ in range(10))
    right_samples = tuple(right.sample() for _ in range(10))
    assert left_samples == right_samples
    assert all(
        any(str(terms[0]) == "domain0" for terms in sample.relation("P", 1))
        for sample in left_samples
    )


def test_cardinality_constraint_holds_in_every_sample():
    sampler = compile_sampler(
        parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 5
|P| = 2
"""),
        seed=12,
    )

    assert all(len(sampler.sample().relation("P", 1)) == 2 for _ in range(30))


@pytest.mark.parametrize(
    ("comparator", "threshold", "accepts"),
    (
        ("=", 1, lambda count: count == 1),
        ("<=", 1, lambda count: count <= 1),
        ("<", 2, lambda count: count < 2),
        (">=", 1, lambda count: count >= 1),
        (">", 0, lambda count: count > 0),
        ("!=", 1, lambda count: count != 1),
    ),
)
def test_all_general_row_and_global_count_comparators(comparator, threshold, accepts):
    sampler = compile_sampler(
        parse_problem(
            rf"""
\forall X: (\exists_{{{comparator}{threshold}}} Y: (R(X,Y) | S(Y,X)))
domain = 3
"""
        ),
        seed=91,
    )

    for _ in range(8):
        sample = sampler.sample()
        for left in sample.domain:
            count = sum(
                (left, right) in sample.relation("R", 2) or (right, left) in sample.relation("S", 2)
                for right in sample.domain
            )
            assert accepts(count)

    global_sampler = compile_sampler(
        parse_problem(
            rf"""
\exists_{{{comparator}{threshold}}} X: (P(X) & Q(X))
domain = 3
"""
        ),
        seed=92,
    )
    for _ in range(8):
        sample = global_sampler.sample()
        count = sum(
            (element,) in sample.relation("P", 1) and (element,) in sample.relation("Q", 1)
            for element in sample.domain
        )
        assert accepts(count)


def test_mod_count_and_binary_cardinality_budget():
    sampler = compile_sampler(
        parse_problem(r"""
\forall X: (\exists_{1mod2} Y: (R(X,Y) & S(Y,X)))
domain = 3
|R| = 4
"""),
        seed=37,
    )

    for _ in range(20):
        sample = sampler.sample()
        assert len(sample.relation("R", 2)) == 4
        for left in sample.domain:
            count = sum(
                (left, right) in sample.relation("R", 2)
                and (right, left) in sample.relation("S", 2)
                for right in sample.domain
            )
            assert count % 2 == 1


def test_weighted_distribution_matches_exact_one_element_probability():
    sampler = compile_sampler(
        parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 1
3 1 P
"""),
        seed=1234,
    )

    draws = 6_000
    positives = sum(bool(sampler.sample().relation("P", 1)) for _ in range(draws))
    assert positives / draws == pytest.approx(0.75, abs=0.025)


def test_nullary_definition_and_linear_order_projection():
    definition = compile_sampler(
        parse_problem(r"""
A <-> (\exists_{1mod2} X: U(X))
domain = 3
2 3 A
"""),
        seed=4,
    )
    for _ in range(20):
        sample = definition.sample()
        assert bool(sample.relation("A", 0)) == (len(sample.relation("U", 1)) % 2 == 1)

    order = compile_sampler(
        parse_problem(r"""
\forall X: (\forall Y: (LEQ(X,Y) | ~LEQ(X,Y)))
domain = 3
"""),
        seed=5,
    )
    assert order.total_weight == 6
    sample = order.sample()
    relation = sample.relation("LEQ", 2)
    for left in sample.domain:
        assert (left, left) in relation
        for right in sample.domain:
            if left != right:
                assert ((left, right) in relation) != ((right, left) in relation)


def test_linear_order_with_unary_evidence_restores_source_mass_and_labels():
    sampler = compile_sampler(
        parse_problem(r"""
\forall X: (\forall Y: ((LEQ(X,Y) | ~LEQ(X,Y)) & (P(X) | ~P(X))))
domain = 3
P(domain0)
"""),
        seed=16,
    )

    assert sampler.total_weight == 24
    for _ in range(20):
        sample = sampler.sample()
        assert any(str(terms[0]) == "domain0" for terms in sample.relation("P", 1))
