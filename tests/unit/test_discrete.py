from fractions import Fraction

import pytest

from c2_wms.discrete_sampling import ExactAliasTable, RandomSource
from c2_wms.errors import SamplingError


class ScriptedRng:
    def __init__(self, values):
        self.values = iter(values)

    def randrange(self, stop):
        value = next(self.values)
        assert 0 <= value < stop
        return value


def test_exact_alias_uses_integer_cutoffs_for_rationals():
    table = ExactAliasTable(("a", "b"), (Fraction(1, 3), Fraction(2, 3)))

    assert table.total == 3
    assert table.sample(ScriptedRng((0, 0))) == "a"
    assert table.sample(ScriptedRng((0, 1))) == "a"
    assert table.sample(ScriptedRng((0, 2))) == "b"
    assert table.sample(ScriptedRng((1, 2))) == "b"


def test_exact_alias_filters_zero_and_rejects_negative_mass():
    table = ExactAliasTable(("zero", "one"), (0, 1))
    assert table.sample(RandomSource(1)) == "one"

    with pytest.raises(SamplingError, match="non-negative"):
        ExactAliasTable(("bad",), (-1,))


def test_seeded_random_source_is_reproducible():
    table = ExactAliasTable(tuple(range(4)), (1, 2, 3, 4))
    left = RandomSource(42)
    right = RandomSource(42)

    assert [table.sample(left) for _ in range(100)] == [table.sample(right) for _ in range(100)]
