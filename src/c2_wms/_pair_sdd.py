"""Conditioned SDD backend for local pair distributions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, TypeAlias

from pysdd.sdd import SddManager, Vtree
from wfomc.fol.cnf import TseitinCNF

from c2_wms.arithmetic import CoefficientCache, Degree
from c2_wms.discrete_sampling import ExactAliasTable
from c2_wms.errors import SamplingError, WfomcCompatibilityError

if TYPE_CHECKING:
    from .pair_sampling import PairSampler

WeightPair: TypeAlias = tuple[object, object]


class _SddCircuit:
    def __init__(self, cnf: TseitinCNF, assumptions: Sequence[int] = ()):
        self.vtree = Vtree(
            var_count=cnf.n_vars,
            var_order=list(range(1, cnf.n_vars + 1)),
            vtree_type="balanced",
        )
        self.manager = SddManager.from_vtree(self.vtree)
        root = self.manager.true()
        for literal in assumptions:
            root = root & self.manager.literal(literal)
        for clause in cnf.clauses:
            clause_node = self.manager.false()
            for literal in clause:
                clause_node = clause_node | self.manager.literal(literal)
            root = root & clause_node
        self.root = root
        self.root.ref()
        self._closed = False
        self._scopes: dict[int, frozenset[int]] = {}
        self._scope(self.vtree)

    def close(self) -> None:
        if not self._closed:
            self.root.deref()
            self._closed = True

    def _scope(self, vtree) -> frozenset[int]:
        position = vtree.position()
        cached = self._scopes.get(position)
        if cached is not None:
            return cached
        if vtree.is_leaf():
            result = frozenset((vtree.var(),))
        else:
            result = self._scope(vtree.left()) | self._scope(vtree.right())
        self._scopes[position] = result
        return result


class _SddDistribution:
    """One cell/mask distribution backed by a conditioned SDD."""

    def __init__(
        self,
        sampler: PairSampler,
        weights: Sequence[WeightPair],
        assumptions: Sequence[int],
    ):
        self.sampler = sampler
        self.weights = weights
        self.circuit = _SddCircuit(sampler.cnf, assumptions)
        self.arithmetic = sampler.trace.arithmetic
        self.coefficients = CoefficientCache(len(self.arithmetic.symbolic_variables))
        self._values: dict[tuple[int, int], object] = {}
        self._free_values: dict[tuple[int, ...], object] = {}
        self._suffix_values: dict[tuple[int, ...], tuple[object, ...]] = {}
        self._aliases: dict[object, ExactAliasTable] = {}
        self.total = self._value(self.circuit.root, self.circuit.vtree)

    def close(self) -> None:
        self.circuit.close()

    def _free_value(self, variables: Iterable[int]):
        key = tuple(sorted(variables))
        cached = self._free_values.get(key)
        if cached is not None:
            return cached
        result = self.arithmetic.one()
        for variable in key:
            positive, negative = self.weights[variable]
            result = self.arithmetic.multiply(result, self.arithmetic.add(positive, negative))
        self._free_values[key] = result
        return result

    def _value(self, node, expected_vtree):
        key = (node.id, expected_vtree.position())
        if key in self._values:
            return self._values[key]
        expected_scope = self.circuit._scope(expected_vtree)
        if node.is_false():
            result = self.arithmetic.zero()
        elif node.is_true():
            result = self._free_value(expected_scope)
        else:
            actual_vtree = node.vtree()
            actual_scope = self.circuit._scope(actual_vtree)
            if actual_scope != expected_scope:
                if not actual_scope < expected_scope:
                    raise WfomcCompatibilityError("SDD node is outside its expected vtree")
                result = self.arithmetic.multiply(
                    self._value(node, actual_vtree),
                    self._free_value(expected_scope - actual_scope),
                )
            elif node.is_literal():
                literal = node.literal
                result = self.weights[abs(literal)][0 if literal > 0 else 1]
            else:
                result = self.arithmetic.zero()
                left_vtree = actual_vtree.left()
                right_vtree = actual_vtree.right()
                for prime, sub in node.elements():
                    result = self.arithmetic.add_product(
                        result,
                        self._value(prime, left_vtree),
                        self._value(sub, right_vtree),
                    )
        self._values[key] = result
        return result

    def _product_choice(self, key, triples, degree: Degree):
        cache_key = (key, degree)
        table = self._aliases.get(cache_key)
        if table is None:
            choices = []
            weights = []
            for choice, left, right in triples:
                for left_degree, right_degree, coefficient in self.coefficients.product_splits(
                    left, right, degree
                ):
                    if coefficient < 0:
                        raise SamplingError("pair SDD contains a negative coefficient")
                    if coefficient > 0:
                        choices.append((choice, left_degree, right_degree))
                        weights.append(coefficient)
            table = ExactAliasTable(choices, weights)
            self._aliases[cache_key] = table
        return table.sample(self.sampler.rng)

    def sample(self, degree: Degree) -> int:
        if self.coefficients.coefficient(self.total, degree) <= 0:
            raise SamplingError("pair factor has no mass at the requested degree")
        assignment: dict[int, bool] = {}
        self._sample_node(
            self.circuit.root,
            self.circuit.vtree,
            degree,
            assignment,
        )
        return self.sampler._source_mask_from_assignment(assignment)

    def _sample_node(self, node, expected_vtree, degree, assignment) -> None:
        expected_scope = self.circuit._scope(expected_vtree)
        if node.is_false():
            raise SamplingError("attempted to sample a false SDD node")
        if node.is_true():
            self._sample_free(tuple(sorted(expected_scope)), degree, assignment)
            return

        actual_vtree = node.vtree()
        actual_scope = self.circuit._scope(actual_vtree)
        if actual_scope != expected_scope:
            missing = tuple(sorted(expected_scope - actual_scope))
            (_choice, node_degree, free_degree) = self._product_choice(
                ("scope", node.id, expected_vtree.position()),
                ((None, self._value(node, actual_vtree), self._free_value(missing)),),
                degree,
            )
            self._sample_node(node, actual_vtree, node_degree, assignment)
            self._sample_free(missing, free_degree, assignment)
            return

        if node.is_literal():
            literal = node.literal
            value = self.weights[abs(literal)][0 if literal > 0 else 1]
            if self.coefficients.coefficient(value, degree) <= 0:
                raise SamplingError("literal does not match its degree budget")
            assignment[abs(literal)] = literal > 0
            return

        left_vtree = actual_vtree.left()
        right_vtree = actual_vtree.right()
        triples = tuple(
            (
                (prime, sub),
                self._value(prime, left_vtree),
                self._value(sub, right_vtree),
            )
            for prime, sub in node.elements()
        )
        (prime, sub), left_degree, right_degree = self._product_choice(
            ("decision", node.id, expected_vtree.position()), triples, degree
        )
        self._sample_node(prime, left_vtree, left_degree, assignment)
        self._sample_node(sub, right_vtree, right_degree, assignment)

    def _sample_free(self, variables, degree, assignment) -> None:
        if not variables:
            if degree != (0,) * self.coefficients.dimension:
                raise SamplingError("free-variable traceback left a degree budget")
            return
        suffix = self._suffix_values.get(variables)
        if suffix is None:
            values = [self.arithmetic.one()] * (len(variables) + 1)
            for index in range(len(variables) - 1, -1, -1):
                positive, negative = self.weights[variables[index]]
                values[index] = self.arithmetic.multiply(
                    self.arithmetic.add(positive, negative), values[index + 1]
                )
            suffix = tuple(values)
            self._suffix_values[variables] = suffix
        variable = variables[0]
        positive, negative = self.weights[variable]
        polarity, _literal_degree, suffix_degree = self._product_choice(
            ("free", variables),
            (
                (True, positive, suffix[1]),
                (False, negative, suffix[1]),
            ),
            degree,
        )
        assignment[variable] = polarity
        self._sample_free(variables[1:], suffix_degree, assignment)


__all__ = ["WeightPair", "_SddDistribution"]
