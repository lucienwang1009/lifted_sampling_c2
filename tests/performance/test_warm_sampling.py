from time import perf_counter

import pytest
from wfomc import parse_problem

from c2_wms import compile_sampler


@pytest.mark.performance
def test_warm_sampling_is_output_quadratic_and_cache_bounded():
    domain_size = 12
    sampler = compile_sampler(
        parse_problem(
            rf"""
\forall X: (\exists_=1 Y: (R(X,Y) & (S(X,Y) | ~S(X,Y))))
domain = {domain_size}
"""
        ),
        seed=42,
    )

    sampler.sample()  # populate lazy aliases and pair condition caches
    started = perf_counter()
    samples = tuple(sampler.sample() for _ in range(8))
    elapsed = perf_counter() - started

    assert all(len(sample.domain) == domain_size for sample in samples)
    assert elapsed < 5.0
    for trace in sampler.traces:
        pair_sampler = sampler._pair_samplers[id(trace)]
        projected = len(trace.counting_state.projected_predicates)
        theoretical_conditions = len(trace.component.cells) ** 2 * 2 ** (2 * projected)
        assert len(pair_sampler._distributions) <= theoretical_conditions
