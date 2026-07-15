import pytest
from wfomc import parse_problem

from c2_wms import compile_sampler
from c2_wms.errors import UnsatisfiableProblemError, UnsupportedSamplingInput
from c2_wms.options import SamplerOptions


def test_sampling_rejects_negative_weights_and_unsatisfiable_inputs():
    negative = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 1
-1 1 P
""")
    with pytest.raises(UnsupportedSamplingInput, match="non-negative"):
        compile_sampler(negative)

    impossible = parse_problem(r"""
\forall X: (P(X) & ~P(X))
domain = 1
""")
    with pytest.raises(UnsatisfiableProblemError):
        compile_sampler(impossible)


def test_options_seed_conflict_and_closed_sampler_contract():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 1
""")
    with pytest.raises(ValueError, match="disagree"):
        compile_sampler(problem, seed=1, options=SamplerOptions(seed=2))

    sampler = compile_sampler(problem, seed=3)
    with sampler:
        assert sampler.sample().domain
    sampler.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        sampler.sample()
