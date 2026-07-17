import pytest
from wfomc import parse_problem
from wfomc.fol import a, b

import c2_wms.pair_sampling as pair_sampling
from c2_wms import compile_sampler


def _anonymous_sample(source: str):
    sampler = compile_sampler(parse_problem(source), seed=17)
    anonymous = sampler._sample_anonymous()
    return sampler, anonymous, sampler._pair_samplers[id(anonymous.trace)]


def _source_truth(pair_sampler, mask, predicate, terms):
    index = pair_sampler._source_binary_indices[(predicate.name, predicate.arity)]
    bit = 2 * index if terms == (b, a) else 2 * index + 1
    return bool(mask & (1 << bit))


def test_pair_sat_enumeration_limit_is_1024():
    assert pair_sampling._SAT_MODEL_LIMIT == 1024


def test_pair_sampler_reconstructs_projected_orientation():
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")

    assert pair_sampler.cnf is None
    for request in anonymous.pair_requests:
        source_mask = pair_sampler.sample_mask(request)
        predicate = anonymous.trace.counting_state.projected_predicates[0]
        assert _source_truth(pair_sampler, source_mask, predicate, (a, b)) == bool(
            request.projection_mask & 0b10
        )
        assert _source_truth(pair_sampler, source_mask, predicate, (b, a)) == bool(
            request.projection_mask & 0b01
        )
    assert pair_sampler.cnf is None
    assert not pair_sampler._distributions
    sampler.close()


def test_direct_structure_projection_does_not_build_conditioned_distribution(monkeypatch):
    sampler = compile_sampler(
        parse_problem(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
"""),
        seed=17,
    )
    pair_sampler = next(iter(sampler._pair_samplers.values()))

    def fail_on_conditioned_distribution(*_args):
        raise AssertionError("direct projection should consume the projection mask")

    monkeypatch.setattr(pair_sampler, "_distribution", fail_on_conditioned_distribution)
    structure = sampler.sample()

    assert len(structure.relation("R", 2)) == 3
    sampler.close()


def test_nondirect_pairs_are_presampled_as_sparse_source_masks(monkeypatch):
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\forall Y: ((E(X,Y) -> E(Y,X)) &
                         (R(X) | B(X)) &
                         (~R(X) | ~B(X)) &
                         (E(X,Y) -> ~(R(X) & R(Y)) & ~(B(X) & B(Y)))))
domain = 4
""")

    assert not pair_sampler.is_direct
    assert anonymous.pair_requests
    assert all(request.source_mask for request in anonymous.pair_requests)

    def fail_on_second_pair_sample(_request):
        raise AssertionError("source masks should be sampled during anonymous traceback")

    monkeypatch.setattr(pair_sampler, "sample_mask", fail_on_second_pair_sample)
    structure = sampler._materialize(anonymous)
    assert len(structure.relation("E", 2)) == 2 * len(anonymous.pair_requests)
    sampler.close()


def test_pair_sampler_handles_general_counting_body_and_free_atoms():
    sampler, anonymous, pair_sampler = _anonymous_sample(r"""
\forall X: (\exists_=1 Y: (R(X,Y) & S(Y,X)))
domain = 2
""")

    for request in anonymous.pair_requests:
        source_mask = pair_sampler.sample_mask(request)
        # The reduced projected predicate is definitionally equivalent to the
        # arbitrary body on both pair orientations.
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
            (a, b, 1),
            (b, a, 0),
        ):
            marker = _source_truth(pair_sampler, source_mask, r_pred, (left, right)) and (
                _source_truth(pair_sampler, source_mask, s_pred, (right, left))
            )
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
    source_mask = pair_sampler.sample_mask(request)
    for left, right, bit in (
        (a, b, 1),
        (b, a, 0),
    ):
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
        marker = _source_truth(pair_sampler, source_mask, r_pred, (left, right)) and (
            _source_truth(pair_sampler, source_mask, s_pred, (right, left))
        )
        assert marker == bool(request.projection_mask & (1 << bit))
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
        _source_truth(
            pair_sampler,
            pair_sampler.sample_mask(request),
            s_predicate,
            (a, b),
        )
        for _ in range(draws)
    )

    assert positives / draws == pytest.approx(0.75, abs=0.025)
    assert all(
        isinstance(distribution, pair_sampling._EnumeratedDistribution)
        for distribution in pair_sampler._distributions.values()
    )
    sampler.close()
