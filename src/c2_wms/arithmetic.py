"""Sparse exact coefficient views used during conditioned traceback."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

from flint import fmpq, fmpq_mpoly, fmpq_poly, fmpz

Degree: TypeAlias = tuple[int, ...]


def add_degrees(left: Degree, right: Degree) -> Degree:
    if not left:
        return right
    if not right:
        return left
    if len(left) != len(right):
        raise ValueError("degree vectors must have the same dimension")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def subtract_degrees(total: Degree, part: Degree) -> Degree | None:
    if not total:
        return () if not part else None
    if len(total) != len(part):
        return None
    result = tuple(a - b for a, b in zip(total, part, strict=True))
    return None if any(value < 0 for value in result) else result


def _constant(value: object) -> fmpq:
    if isinstance(value, bool):
        return fmpq(int(value))
    if isinstance(value, int):
        return fmpq(value)
    if isinstance(value, fmpz):
        return fmpq(value)
    if isinstance(value, fmpq):
        return value
    raise TypeError(f"Unsupported exact arithmetic value: {type(value).__name__}")


def sparse_terms(value: object, dimension: int) -> dict[Degree, fmpq]:
    """Return nonzero exact coefficients keyed by full degree vectors."""

    if isinstance(value, (bool, int, fmpz, fmpq)):
        coefficient = _constant(value)
        return {} if coefficient == 0 else {(0,) * dimension: coefficient}
    if isinstance(value, fmpq_poly):
        if dimension != 1:
            raise ValueError("univariate polynomial requires one degree dimension")
        return {
            (degree,): coefficient
            for degree, coefficient in enumerate(value.coeffs())
            if coefficient != 0
        }
    if isinstance(value, fmpq_mpoly):
        if value.context().nvars() != dimension:
            raise ValueError("multivariate polynomial dimension mismatch")
        return {
            tuple(map(int, degree)): coefficient
            for degree, coefficient in value.to_dict().items()
            if coefficient != 0
        }
    raise TypeError(f"Sampling requires exact FLINT arithmetic, got {type(value).__name__}")


@dataclass(slots=True)
class CoefficientCache:
    """Identity-aware cache for immutable FLINT scalar/polynomial values."""

    dimension: int
    _cache: dict[int, tuple[object, dict[Degree, fmpq]]] = field(default_factory=dict)

    def terms(self, value: object) -> dict[Degree, fmpq]:
        key = id(value)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is value:
            return cached[1]
        terms = sparse_terms(value, self.dimension)
        self._cache[key] = (value, terms)
        return terms

    def coefficient(self, value: object, degree: Degree) -> fmpq:
        return self.terms(value).get(degree, fmpq(0))

    def product_splits(
        self,
        left: object,
        right: object,
        target: Degree,
    ) -> Iterable[tuple[Degree, Degree, fmpq]]:
        right_terms = self.terms(right)
        for left_degree, left_coefficient in self.terms(left).items():
            right_degree = subtract_degrees(target, left_degree)
            if right_degree is None:
                continue
            right_coefficient = right_terms.get(right_degree)
            if right_coefficient is None:
                continue
            weight = left_coefficient * right_coefficient
            if weight != 0:
                yield left_degree, right_degree, weight
