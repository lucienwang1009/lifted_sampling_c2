"""Exact discrete sampling without floating-point normalization."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from fractions import Fraction
from typing import Generic, Protocol, TypeVar

from flint import fmpq, fmpz

from c2_wms.errors import SamplingError

T = TypeVar("T")


class RandRange(Protocol):
    def randrange(self, stop: int) -> int: ...

    def shuffle(self, values: list[object]) -> None: ...


class RandomSource:
    """Seedable random source that also handles arbitrary-size integers."""

    __slots__ = ("_random",)

    def __init__(self, seed: int | None = None):
        self._random = random.Random(seed)

    def randrange(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("stop must be positive")
        return self._random.randrange(stop)

    def shuffle(self, values: list[object]) -> None:
        self._random.shuffle(values)


def _as_fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, fmpz):
        return Fraction(int(value))
    if isinstance(value, fmpq):
        return Fraction(int(value.p), int(value.q))
    raise TypeError(f"Exact sampling requires rational weights, got {type(value).__name__}")


class ExactAliasTable(Generic[T]):
    """Vose alias table with exact integer thresholds.

    Rational input weights are scaled by the least common multiple of their
    denominators. A draw chooses a column uniformly and compares an
    arbitrary-size integer with that column's exact cutoff, so no probability
    is rounded through ``float``.
    """

    __slots__ = ("choices", "cutoffs", "aliases", "total", "_single")

    def __init__(self, choices: Sequence[T], weights: Sequence[object]):
        if len(choices) != len(weights):
            raise ValueError("choices and weights must have the same length")
        if not choices:
            raise ValueError("at least one choice is required")

        positive_choices: list[T] = []
        fractions: list[Fraction] = []
        for choice, raw_weight in zip(choices, weights, strict=True):
            weight = _as_fraction(raw_weight)
            if weight < 0:
                raise SamplingError("sampling weights must be non-negative")
            if weight == 0:
                continue
            positive_choices.append(choice)
            fractions.append(weight)
        if not positive_choices:
            raise SamplingError("cannot sample from a zero-mass distribution")

        self.choices = tuple(positive_choices)
        self._single = self.choices[0] if len(self.choices) == 1 else None
        if self._single is not None:
            self.cutoffs = ()
            self.aliases = ()
            self.total = 1
            return

        denominator = math.lcm(*(weight.denominator for weight in fractions))
        integer_weights = [
            weight.numerator * (denominator // weight.denominator) for weight in fractions
        ]
        total = sum(integer_weights)
        size = len(integer_weights)
        scaled = [weight * size for weight in integer_weights]
        small = [index for index, weight in enumerate(scaled) if weight < total]
        large = [index for index, weight in enumerate(scaled) if weight >= total]
        cutoffs = [total] * size
        aliases = list(range(size))

        while small and large:
            low = small.pop()
            high = large.pop()
            cutoffs[low] = scaled[low]
            aliases[low] = high
            scaled[high] = scaled[high] + scaled[low] - total
            if scaled[high] < total:
                small.append(high)
            else:
                large.append(high)

        self.cutoffs = tuple(cutoffs)
        self.aliases = tuple(aliases)
        self.total = total

    def sample(self, rng: RandRange) -> T:
        if self._single is not None:
            return self._single
        column = rng.randrange(len(self.choices))
        threshold = rng.randrange(self.total)
        index = column if threshold < self.cutoffs[column] else self.aliases[column]
        return self.choices[index]
