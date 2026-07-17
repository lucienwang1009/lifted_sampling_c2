from flint import fmpq, fmpq_mpoly_ctx, fmpq_poly

from c2_wms.arithmetic import CoefficientCache, sparse_terms


def test_sparse_terms_handles_scalar_and_univariate_values():
    assert sparse_terms(fmpq(3, 2), 0) == {(): fmpq(3, 2)}
    assert sparse_terms(fmpq_poly([1, 0, 2]), 1) == {
        (0,): fmpq(1),
        (2,): fmpq(2),
    }


def test_product_splits_extract_only_target_degree():
    cache = CoefficientCache(1)
    left = fmpq_poly([1, 2])
    right = fmpq_poly([3, 4])

    assert list(cache.product_splits(left, right, (1,))) == [
        ((0,), (1,), fmpq(4)),
        ((1,), (0,), fmpq(6)),
    ]


def test_sparse_terms_preserves_multivariate_degree_order():
    ctx = fmpq_mpoly_ctx.get(["x", "y"], "lex")
    value = 2 * ctx.gen(0) + 3 * ctx.gen(1) ** 2

    assert sparse_terms(value, 2) == {
        (1, 0): fmpq(2),
        (0, 2): fmpq(3),
    }
