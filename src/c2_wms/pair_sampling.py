"""Lazy exact sampling of a two-element model behind a pair factor."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from time import perf_counter

from pysat.solvers import Solver
from pysdd.sdd import SddManager, Vtree
from wfomc.fol import a, b, context_for, simplify_boolean, substitute, true
from wfomc.fol.cnf import TseitinCNF, encode_tseitin
from wfomc.fol.grounding import ground_on_tuple
from wfomc.weights import compile_weight_mapping

from c2_wms.arithmetic import CoefficientCache, Degree
from c2_wms.discrete_sampling import ExactAliasTable, RandRange
from c2_wms.errors import SamplingError, WfomcCompatibilityError
from c2_wms.trace.kernel import projection_mask
from c2_wms.trace.traceback import PairRequest

WeightPair = tuple[object, object]
_SAT_MODEL_LIMIT = 1024

logger = logging.getLogger(__name__)


def _predicate_weight(weights, predicate, one) -> WeightPair:
    pair = weights.get(predicate)
    if pair is not None:
        return pair
    key = (predicate.name, predicate.arity)
    for candidate, candidate_pair in weights.items():
        if (candidate.name, candidate.arity) == key:
            return candidate_pair
    return one, one


def _substitute_nullary(formula, assignments):
    if not assignments:
        return formula
    context = context_for(formula)
    replacement = {
        predicate(): context.true() if value else context.false()
        for predicate, value in assignments
    }
    return simplify_boolean(substitute(formula, replacement))


def _pair_cnf(trace) -> TseitinCNF:
    reduced = trace.reduced_problem
    if reduced is None:
        raise WfomcCompatibilityError("pair sampling requires the WFOMC reduced problem")
    formula = reduced.normal_form.qf_formula or true()
    formula = _substitute_nullary(formula, trace.component.nullary_assignments)
    grounded = ground_on_tuple(formula, a, b) & ground_on_tuple(formula, b, a)
    if trace.has_linear_order:
        leq = next(
            (
                predicate
                for predicate in trace.component.cells[0].preds
                if predicate.name == "LEQ" and predicate.arity == 2
            ),
            None,
        )
        if leq is None:
            raise WfomcCompatibilityError("linear-order trace has no LEQ predicate")
        grounded = grounded & leq(b, a) & ~leq(a, b)
    cnf = encode_tseitin(grounded)

    # The cell graph gives absent off-diagonal vocabulary atoms their free
    # (w+ + w-) factor. Register them as unconstrained pair variables so the
    # sampler can materialize their truth values instead of only multiplying
    # the mass.
    atoms = list(cnf.atoms)
    atom_to_var = dict(cnf.atom_to_var)
    n_vars = cnf.n_vars
    for predicate in sorted(
        (predicate for predicate in trace.component.cells[0].preds if predicate.arity == 2),
        key=lambda item: (item.name, item.arity),
    ):
        for atom in (predicate(b, a), predicate(a, b)):
            if atom in atom_to_var:
                continue
            n_vars += 1
            atoms.append(atom)
            atom_to_var[atom] = n_vars
    return TseitinCNF(
        atoms=tuple(atoms),
        atom_to_var=atom_to_var,
        clauses=cnf.clauses,
        n_vars=n_vars,
        auxiliary_vars=cnf.auxiliary_vars,
    )


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


class _EnumeratedDistribution:
    """A small conditioned pair distribution enumerated exactly by SAT."""

    def __init__(self, sampler: PairSampler, choices, weights):
        self.sampler = sampler
        self.choices = tuple(choices)
        self.weights = tuple(weights)
        self.coefficients = CoefficientCache(len(sampler.trace.arithmetic.symbolic_variables))
        self._aliases: dict[Degree, ExactAliasTable] = {}
        total = sampler.trace.arithmetic.zero()
        for weight in self.weights:
            total = sampler.trace.arithmetic.add(total, weight)
        self.total = total

    def table(self, degree: Degree) -> ExactAliasTable:
        table = self._aliases.get(degree)
        if table is None:
            choices = []
            weights = []
            for choice, value in zip(self.choices, self.weights, strict=True):
                coefficient = self.coefficients.coefficient(value, degree)
                if coefficient < 0:
                    raise SamplingError("pair distribution contains a negative coefficient")
                if coefficient > 0:
                    choices.append(choice)
                    weights.append(coefficient)
            if not choices:
                raise SamplingError("pair factor has no mass at the requested degree")
            table = ExactAliasTable(choices, weights)
            self._aliases[degree] = table
        return table

    def sample(self, degree: Degree) -> int:
        table = self.table(degree)
        return table.sample(self.sampler.rng)


class PairSampler:
    """Sample source-level pair masks from exact conditioned pair factors."""

    a = a
    b = b

    def __init__(self, trace, rng: RandRange, source_keys, *, validate_masses: bool = True):
        self.trace = trace
        self.rng = rng
        self.validate_masses = validate_masses
        reduced = trace.reduced_problem
        if reduced is None:
            raise WfomcCompatibilityError("missing reduced problem")
        self.predicate_weights = compile_weight_mapping(dict(reduced.weights), trace.arithmetic)
        self._projected_indices = {
            predicate: index
            for index, predicate in enumerate(trace.counting_state.projected_predicates)
        }
        self._binary_predicates = tuple(
            sorted(
                (predicate for predicate in trace.component.cells[0].preds if predicate.arity == 2),
                key=lambda predicate: (predicate.name, predicate.arity),
            )
        )
        self._is_direct = all(
            predicate in self._projected_indices for predicate in self._binary_predicates
        )
        source_binary_keys = tuple(key for key in source_keys if key.arity == 2)
        self._source_binary_indices = {
            (key.name, key.arity): index for index, key in enumerate(source_binary_keys)
        }
        self.source_actions = tuple(
            action
            for key in source_binary_keys
            for action in ((key, True), (key, False))
        )
        self.cnf: TseitinCNF | None = None
        self._source_variable_bits: tuple[tuple[int, int], ...] = ()
        self._distributions: dict[
            tuple[int, int, int], _EnumeratedDistribution | _SddDistribution
        ] = {}
        self._condition_kernels: dict[tuple[int, int, int, Degree], int | ExactAliasTable] = {}
        self._direct_cache: dict[tuple[int, int, int], tuple[int, object]] = {}
        self._coefficients = CoefficientCache(len(trace.arithmetic.symbolic_variables))
        self._expected = self._expected_masses()
        if self._is_direct and self.validate_masses:
            for left, right, mask in self._expected:
                self._direct_pair(left, right, mask)
        self._closed = False
        logger.debug(
            "Initialized pair sampler cells=%d binary_predicates=%d projected_predicates=%d "
            "direct=%s expected_conditions=%d validate_masses=%s",
            len(trace.component.cells),
            len(self._binary_predicates),
            len(self._projected_indices),
            self._is_direct,
            len(self._expected),
            validate_masses,
        )

    @property
    def is_direct(self) -> bool:
        return self._is_direct

    def close(self) -> None:
        if self._closed:
            return
        for distribution in self._distributions.values():
            if isinstance(distribution, _SddDistribution):
                distribution.close()
        self._closed = True
        logger.debug(
            "Closed pair sampler direct_conditions=%d conditioned_distributions=%d",
            len(self._direct_cache),
            len(self._distributions),
        )

    def _ensure_cnf(self) -> TseitinCNF:
        if self.cnf is None:
            self.cnf = _pair_cnf(self.trace)
            source_variable_bits = []
            for atom, variable in self.cnf.atom_to_var.items():
                bit = self._source_bit(atom)
                if bit is not None:
                    source_variable_bits.append((variable, bit))
            self._source_variable_bits = tuple(source_variable_bits)
            logger.debug(
                "Built pair CNF variables=%d atoms=%d clauses=%d auxiliary_variables=%d",
                self.cnf.n_vars,
                len(self.cnf.atoms),
                len(self.cnf.clauses),
                len(self.cnf.auxiliary_vars),
            )
        return self.cnf

    def _source_bit(self, atom) -> int | None:
        index = self._source_binary_indices.get((atom.predicate.name, atom.predicate.arity))
        if index is None:
            return None
        if atom.terms == (b, a):
            return 2 * index
        if atom.terms == (a, b):
            return 2 * index + 1
        return None

    def _source_mask_from_positive(self, positive: set[int]) -> int:
        mask = 0
        for variable, bit in self._source_variable_bits:
            if variable in positive:
                mask |= 1 << bit
        return mask

    def _source_mask_from_assignment(self, assignment: dict[int, bool]) -> int:
        mask = 0
        for variable, bit in self._source_variable_bits:
            if assignment[variable]:
                mask |= 1 << bit
        return mask

    def _direct_source_mask(self, projection: int) -> int:
        source_mask = 0
        while projection:
            bit = projection & -projection
            projected_bit = bit.bit_length() - 1
            predicate = self.trace.counting_state.projected_predicates[projected_bit // 2]
            source_index = self._source_binary_indices.get(
                (predicate.name, predicate.arity)
            )
            if source_index is not None:
                source_mask |= 1 << (2 * source_index + projected_bit % 2)
            projection ^= bit
        return source_mask

    def _expected_masses(self):
        rows = self.trace.component.counting_binary_relation_weights
        if rows is None:
            raise WfomcCompatibilityError("counting pair factors are missing")
        result = {}
        for left, row in enumerate(rows):
            for right, entries in enumerate(row):
                for forward, reverse, weight in entries:
                    mask = projection_mask(forward, reverse, self.trace.counting_state)
                    key = (left, right, mask)
                    result[key] = self.trace.arithmetic.add(
                        result.get(key, self.trace.arithmetic.zero()), weight
                    )
        return result

    def _weights(self, left_index: int, right_index: int, mask: int):
        cnf = self._ensure_cnf()
        one = self.trace.arithmetic.one()
        zero = self.trace.arithmetic.zero()
        left = self.trace.component.cells[left_index]
        right = self.trace.component.cells[right_index]
        result = [(one, one) for _ in range(cnf.n_vars + 1)]
        cell_predicates = frozenset(left.preds)
        for atom, variable in cnf.atom_to_var.items():
            if (
                atom.predicate in cell_predicates
                and atom.terms
                and (all(term == a for term in atom.terms) or all(term == b for term in atom.terms))
            ):
                cell = left if all(term == a for term in atom.terms) else right
                positive = cell.is_positive(atom.predicate)
                result[variable] = (one, zero) if positive else (zero, one)
                continue

            pair = _predicate_weight(self.predicate_weights, atom.predicate, one)
            projection_index = self._projected_indices.get(atom.predicate)
            bit = None
            if projection_index is not None:
                if atom.terms == (b, a):
                    bit = 2 * projection_index
                elif atom.terms == (a, b):
                    bit = 2 * projection_index + 1
            if bit is None:
                result[variable] = pair
            elif mask & (1 << bit):
                result[variable] = (pair[0], zero)
            else:
                result[variable] = (zero, pair[1])
        return tuple(result)

    def _condition_literals(self, left_index: int, right_index: int, mask: int):
        cnf = self._ensure_cnf()
        left = self.trace.component.cells[left_index]
        right = self.trace.component.cells[right_index]
        cell_predicates = frozenset(left.preds)
        literals = []
        for atom, variable in cnf.atom_to_var.items():
            truth = None
            if (
                atom.predicate in cell_predicates
                and atom.terms
                and (all(term == a for term in atom.terms) or all(term == b for term in atom.terms))
            ):
                cell = left if all(term == a for term in atom.terms) else right
                truth = cell.is_positive(atom.predicate)
            else:
                projection_index = self._projected_indices.get(atom.predicate)
                if projection_index is not None:
                    if atom.terms == (b, a):
                        truth = bool(mask & (1 << (2 * projection_index)))
                    elif atom.terms == (a, b):
                        truth = bool(mask & (1 << (2 * projection_index + 1)))
            if truth is not None:
                literals.append(variable if truth else -variable)
        return tuple(literals)

    def _enumerate_distribution(self, weights, assumptions):
        cnf = self._ensure_cnf()
        variables = tuple(cnf.atom_to_var.values())
        clauses = list(cnf.clauses)
        used_variables = {abs(literal) for clause in clauses for literal in clause}
        clauses.extend(
            (variable, -variable) for variable in variables if variable not in used_variables
        )

        choice_weights = {}
        model_count = 0
        with Solver(name="cadical195", bootstrap_with=clauses) as solver:
            while solver.solve(assumptions=assumptions):
                model_count += 1
                if model_count > _SAT_MODEL_LIMIT:
                    return None
                model = solver.get_model()
                positive = {literal for literal in model if literal > 0}

                weight = self.trace.arithmetic.one()
                for variable in variables:
                    value = weights[variable][0 if variable in positive else 1]
                    weight = self.trace.arithmetic.multiply(weight, value)
                if weight != self.trace.arithmetic.zero():
                    source_mask = self._source_mask_from_positive(positive)
                    choice_weights[source_mask] = self.trace.arithmetic.add(
                        choice_weights.get(source_mask, self.trace.arithmetic.zero()),
                        weight,
                    )

                if not variables:
                    break
                solver.add_clause(
                    [-variable if variable in positive else variable for variable in variables]
                )
        return _EnumeratedDistribution(
            self,
            tuple(choice_weights),
            tuple(choice_weights.values()),
        )

    def _direct_pair(self, left: int, right: int, mask: int):
        key = (left, right, mask)
        cached = self._direct_cache.get(key)
        if cached is not None:
            return cached

        one = self.trace.arithmetic.one()
        total = one
        source_mask = 0
        for predicate in self._binary_predicates:
            index = self._projected_indices[predicate]
            source_index = self._source_binary_indices.get((predicate.name, predicate.arity))
            positive, negative = _predicate_weight(self.predicate_weights, predicate, one)
            for left_term, right_term, bit in (
                (b, a, 2 * index),
                (a, b, 2 * index + 1),
            ):
                is_positive = bool(mask & (1 << bit))
                total = self.trace.arithmetic.multiply(total, positive if is_positive else negative)
                if is_positive and source_index is not None:
                    source_bit = 2 * source_index + int((left_term, right_term) == (a, b))
                    source_mask |= 1 << source_bit
        result = (source_mask, total)
        expected = self._expected.get(key, self.trace.arithmetic.zero())
        if self.validate_masses and total != expected:
            raise WfomcCompatibilityError(
                "reconstructed direct pair mass differs from WFOMC pair factor: "
                f"cells=({left}, {right}), mask={mask}, "
                f"observed={total}, expected={expected}"
            )
        self._direct_cache[key] = result
        logger.debug(
            "Cached direct pair condition left_cell=%d right_cell=%d mask=%d source_bits=%d "
            "cache_entries=%d",
            left,
            right,
            mask,
            source_mask.bit_count(),
            len(self._direct_cache),
        )
        return result

    def _distribution(self, left: int, right: int, mask: int):
        key = (left, right, mask)
        distribution = self._distributions.get(key)
        if distribution is None:
            started = perf_counter()
            weights = self._weights(*key)
            assumptions = self._condition_literals(*key)
            distribution = self._enumerate_distribution(weights, assumptions)
            if distribution is None:
                logger.debug(
                    "Pair condition exceeded SAT enumeration limit; switching to SDD "
                    "left_cell=%d right_cell=%d mask=%d limit=%d",
                    left,
                    right,
                    mask,
                    _SAT_MODEL_LIMIT,
                )
                distribution = _SddDistribution(self, weights, assumptions)
            expected = self._expected.get(key, self.trace.arithmetic.zero())
            if self.validate_masses and distribution.total != expected:
                raise WfomcCompatibilityError(
                    "reconstructed pair mass differs from WFOMC pair factor: "
                    f"cells=({left}, {right}), mask={mask}, "
                    f"observed={distribution.total}, expected={expected}"
                )
            self._distributions[key] = distribution
            choices = (
                len(distribution.choices)
                if isinstance(distribution, _EnumeratedDistribution)
                else None
            )
            logger.debug(
                "Cached conditioned pair distribution left_cell=%d right_cell=%d mask=%d "
                "assumptions=%d backend=%s choices=%s cache_entries=%d elapsed_ms=%.3f",
                left,
                right,
                mask,
                len(assumptions),
                "sat" if isinstance(distribution, _EnumeratedDistribution) else "sdd",
                choices,
                len(self._distributions),
                (perf_counter() - started) * 1000,
            )
        return distribution

    def sample_condition(
        self,
        left_cell: int,
        right_cell: int,
        projection_mask: int,
        degree: Degree,
    ) -> int:
        if self._closed:
            raise RuntimeError("pair sampler is closed")
        key = (left_cell, right_cell, projection_mask, degree)
        cached = self._condition_kernels.get(key)
        if cached is not None:
            if isinstance(cached, int):
                return cached
            return cached.sample(self.rng)
        if self._is_direct:
            source_mask = self._direct_source_mask(projection_mask)
            self._condition_kernels[key] = source_mask
            return source_mask
        distribution = self._distribution(
            left_cell,
            right_cell,
            projection_mask,
        )
        if isinstance(distribution, _EnumeratedDistribution):
            table = distribution.table(degree)
            kernel = table.choices[0] if len(table.choices) == 1 else table
            self._condition_kernels[key] = kernel
            if isinstance(kernel, int):
                return kernel
            return kernel.sample(self.rng)
        return distribution.sample(degree)

    def sample_mask(self, request: PairRequest) -> int:
        return self.sample_condition(
            request.left_cell,
            request.right_cell,
            request.projection_mask,
            request.degree,
        )


__all__ = ["PairSampler"]
