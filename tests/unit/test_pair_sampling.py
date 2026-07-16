import pytest
from wfomc import parse_problem

import c2_wms.pair_sampling as pair_sampling
from c2_wms import compile_sampler


def _anonymous_sample(source: str):
    sampler = compile_sampler(parse_problem(source), seed=17)
    anonymous = sampler._sample_anonymous()
    return sampler, anonymous, sampler._pair_samplers[id(anonymous.trace)]


def test_pair_sat_enumeration_limit_is_1024():
    assert pair_sampling._SAT_MODEL_LIMIT == 1024


def test_pair_sampler_reconstructs_projected_orientation():
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")

    assert pair_sampler.cnf is None
    for request in anonymous.pair_requests:
        atoms = pair_sampler.sample(request)
        truth = {(atom.predicate.name, atom.terms) for atom in atoms}
        predicate = anonymous.trace.counting_state.projected_predicates[0]
        assert ((predicate.name, (pair_sampler.a, pair_sampler.b)) in truth) == bool(
            request.projection_mask & 0b10
        )
        assert ((predicate.name, (pair_sampler.b, pair_sampler.a)) in truth) == bool(
            request.projection_mask & 0b01
        )
    assert pair_sampler.cnf is None
    assert not pair_sampler._distributions
    sampler.close()


def test_pair_sampler_handles_general_counting_body_and_free_atoms():
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & S(Y,X)))
domain = 2
""")

    for request in anonymous.pair_requests:
        atoms = set(pair_sampler.sample(request))
        # The reduced projected predicate is definitionally equivalent to the
        # arbitrary body on both pair orientations.
        projected = anonymous.trace.counting_state.projected_predicates[0]
        r_pred = next(
            predicate
            for predicate in anonymous.trace.component.cells[0].preds
            if predicate.name == "R"
        )
        s_pred = next(
            predicate
            for predicate in anonymous.trace.component.cells[0].preds
            if predicate.name == "S"
        )
        for left, right, bit in (
            (pair_sampler.a, pair_sampler.b, 1),
            (pair_sampler.b, pair_sampler.a, 0),
        ):
            marker = projected(left, right) in atoms
            assert marker == (r_pred(left, right) in atoms and s_pred(right, left) in atoms)
            assert marker == bool(request.projection_mask & (1 << bit))
    assert pair_sampler.cnf is not None
    assert pair_sampler._distributions
    assert all(
        isinstance(distribution, pair_sampling._EnumeratedDistribution)
        for distribution in pair_sampler._distributions.values()
    )
    sampler.close()


def test_large_conditioned_distribution_falls_back_to_lazy_sdd(monkeypatch):
    monkeypatch.setattr(pair_sampling, "_SAT_MODEL_LIMIT", 0)
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & S(Y,X)))
domain = 2
""")

    request = anonymous.pair_requests[0]
    atoms = set(pair_sampler.sample(request))
    projected = anonymous.trace.counting_state.projected_predicates[0]
    for left, right, bit in (
        (pair_sampler.a, pair_sampler.b, 1),
        (pair_sampler.b, pair_sampler.a, 0),
    ):
        assert (projected(left, right) in atoms) == bool(request.projection_mask & (1 << bit))
    assert isinstance(
        pair_sampler._distributions[
            (request.left_cell, request.right_cell, request.projection_mask)
        ],
        pair_sampling._SddDistribution,
    )
    sampler.close()


def test_enumerated_pair_distribution_uses_exact_free_atom_weights():
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: R(X,Y)) &
\forall X: (\forall Y: (S(X,Y) | ~S(X,Y)))
domain = 2
3 1 S
""")

    request = anonymous.pair_requests[0]
    s_predicate = next(
        predicate for predicate in anonymous.trace.component.cells[0].preds if predicate.name == "S"
    )
    draws = 4_000
    positives = sum(
        s_predicate(pair_sampler.a, pair_sampler.b) in pair_sampler.sample(request)
        for _ in range(draws)
    )

    assert positives / draws == pytest.approx(0.75, abs=0.025)
    assert all(
        isinstance(distribution, pair_sampling._EnumeratedDistribution)
        for distribution in pair_sampler._distributions.values()
    )
    sampler.close()
