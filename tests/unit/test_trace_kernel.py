from wfomc import AlgoName, compile_problem, parse_problem
from wfomc.algo import AlgoOptions, EvidenceStrategy, ExistentialStrategy
from wfomc.algo.incremental3.solve import _solve_component

import c2_wms.trace.traceback as traceback_module
from c2_wms.discrete_sampling import RandomSource
from c2_wms.trace import TracebackSampler, compile_component_trace


def _compile(source):
    problem = parse_problem(source)
    artifacts = compile_problem(
        problem,
        algo=AlgoName.INCREMENTAL3,
        options=AlgoOptions(
            evidence_strategy=EvidenceStrategy.LIFTED_PROFILES,
            existential_strategy=ExistentialStrategy.COUNTING,
        ),
    )
    return artifacts.algo_input


def test_trace_total_equals_incremental3_component_value():
    algo_input = _compile(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 2
""")
    component = algo_input.components[0]

    trace = compile_component_trace(algo_input, component)
    expected = _solve_component(
        component,
        domain_size=algo_input.domain_size,
        counting_state=algo_input.counting_state,
        unary_masks=algo_input.unary_cardinality_masks,
        has_linear_order=algo_input.has_linear_order,
        arithmetic=algo_input.arithmetic,
    )

    assert trace.total_mass == expected == 4
    assert trace.root_terms


def test_traceback_reconstructs_projected_row_counts():
    algo_input = _compile(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")
    component = algo_input.components[0]
    trace = compile_component_trace(algo_input, component)
    root = trace.root_terms[0]
    sampler = TracebackSampler(trace, RandomSource(5))

    for _ in range(20):
        sampled = sampler.sample(root, ())
        outdegrees = [0, 0, 0]
        predicate = algo_input.counting_state.projected_predicates[0]
        for element, cell_index in enumerate(sampled.cell_indices):
            if component.cells[cell_index].is_positive(predicate):
                outdegrees[element] += 1
        for pair in sampled.pair_requests:
            if pair.projection_mask & 0b10:
                outdegrees[pair.left] += 1
            if pair.projection_mask & 0b01:
                outdegrees[pair.right] += 1

        assert outdegrees == [1, 1, 1]
        assert len(sampled.pair_requests) == 3


def test_warm_traceback_reuses_candidates_without_traversing_trace_layers():
    algo_input = _compile(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")
    component = algo_input.components[0]
    trace = compile_component_trace(algo_input, component)
    root = trace.root_terms[0]
    sampler = TracebackSampler(trace, RandomSource(5))

    expected = sampler.sample(root, ())

    class NoItems(dict):
        def items(self):
            raise AssertionError("warm traceback traversed a cached trace layer")

    for node in trace.domain_nodes.values():
        for target in node.targets:
            target.terminal_weights = NoItems(target.terminal_weights)
            target.g_layers = tuple(NoItems(layer) for layer in target.g_layers)
    for h_trace in trace.h_traces.values():
        h_trace.layers = [NoItems(layer) for layer in h_trace.layers]
    trace.t_update_dict = {
        key: NoItems(transitions) for key, transitions in trace.t_update_dict.items()
    }

    sampler.rng = RandomSource(5)
    assert sampler.sample(root, ()) == expected


def test_h_traceback_aliases_are_shared_across_outer_domain_contexts():
    algo_input = _compile(r"""
\forall X: (~R(X,X)) &
\forall X: (\exists_=1 Y: R(X,Y))
domain = 8
""")
    trace = compile_component_trace(algo_input, algo_input.components[0])
    sampler = TracebackSampler(trace, RandomSource(42))

    for _ in range(50):
        sampler.sample(trace.root_terms[0], ())

    h_keys = [key for key in sampler._aliases if key[1][0] == "h"]
    assert h_keys
    assert all(len(key[1]) == 6 for key in h_keys)
    assert len(h_keys) < 100


def test_traceback_assigns_pair_identifiers_without_sentinels(monkeypatch):
    algo_input = _compile(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")
    trace = compile_component_trace(algo_input, algo_input.components[0])
    original = traceback_module.PairRequest

    def checked_pair_request(left, right, left_cell, right_cell, projection_mask, degree):
        assert left >= 0
        return original(left, right, left_cell, right_cell, projection_mask, degree)

    monkeypatch.setattr(traceback_module, "PairRequest", checked_pair_request)
    sampled = TracebackSampler(trace, RandomSource(5)).sample(trace.root_terms[0], ())

    assert len(sampled.pair_requests) == 3
