"""Public sampled-structure representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class PredicateKey:
    name: str
    arity: int


@dataclass(frozen=True, slots=True)
class SampledStructure:
    domain: tuple[object, ...]
    relations: tuple[tuple[PredicateKey, frozenset[tuple[object, ...]]], ...]

    def relation(self, name: str, arity: int) -> frozenset[tuple[object, ...]]:
        key = PredicateKey(name, arity)
        for predicate, tuples in self.relations:
            if predicate == key:
                return tuples
        return frozenset()

    def true_atoms(self) -> tuple[str, ...]:
        atoms = []
        for predicate, tuples in self.relations:
            for terms in sorted(tuples, key=lambda values: tuple(map(str, values))):
                joined = ", ".join(map(str, terms))
                atoms.append(f"{predicate.name}({joined})")
        return tuple(atoms)

    @classmethod
    def from_mapping(
        cls,
        domain: Iterable[object],
        relations: dict[PredicateKey, set[tuple[object, ...]]],
    ) -> SampledStructure:
        return cls(
            tuple(domain),
            tuple((key, frozenset(values)) for key, values in sorted(relations.items())),
        )
