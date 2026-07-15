import wfomc
from wfomc import parse_problem

from c2_wms import compile_sampler


def _sampler(source, seed=3):
    return compile_sampler(parse_problem(source), seed=seed)


def test_sampler_matches_scalar_count_and_samples_trace():
    sampler = _sampler(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 3
""")

    assert sampler.total_weight == 27
    sampled = sampler._sample_anonymous()
    assert len(sampled.cell_indices) == 3
    assert len(sampled.pair_requests) == 3


def test_compile_sampler_does_not_run_a_second_wfomc_solve(monkeypatch):
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 2
""")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("compile_sampler must not call wfomc.solve")

    monkeypatch.setattr(wfomc, "solve", fail_if_called)

    assert compile_sampler(problem, seed=4).total_weight == 4


def test_root_mixture_filters_cardinality_marker_degrees():
    sampler = _sampler(r"""
\forall X: (P(X) | ~P(X))
domain = 3
|P| = 1
""")

    assert sampler.total_weight == 3
    for _ in range(20):
        sampled = sampler._sample_anonymous()
        component = sampled.trace.component
        positives = sum(
            component.cells[cell_index].is_positive(
                next(pred for pred in component.cells[cell_index].preds if pred.name == "P")
            )
            for cell_index in sampled.cell_indices
        )
        assert positives == 1
