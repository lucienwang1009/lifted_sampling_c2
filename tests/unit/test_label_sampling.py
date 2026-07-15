from wfomc import parse_problem

from c2_wms import compile_sampler
from c2_wms.discrete_sampling import RandomSource
from c2_wms.label_sampling import LabelSampler


def test_label_sampler_preserves_named_unary_evidence():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 3
P(domain0)
    """)
    rng = RandomSource(11)
    sampler = compile_sampler(problem, seed=11)
    label_sampler = LabelSampler(problem, rng)

    for _ in range(30):
        anonymous = sampler._sample_anonymous()
        labels = label_sampler.sample(anonymous)
        element = labels.index(next(value for value in labels if str(value) == "domain0"))
        predicate = next(
            item for item in anonymous.trace.component.cells[0].preds if item.name == "P"
        )
        cell = anonymous.trace.component.cells[anonymous.cell_indices[element]]
        assert cell.is_positive(predicate)


def test_unconstrained_labels_form_domain_permutation():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = {a, b, c}
    """)
    rng = RandomSource(7)
    sampler = compile_sampler(problem, seed=7)
    anonymous = sampler._sample_anonymous()
    labels = LabelSampler(problem, rng).sample(anonymous)

    assert set(labels) == set(problem.domain)
    assert len(labels) == len(problem.domain)
