"""Lazy exact sampling of a two-element model behind a pair factor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

from pysat.solvers import Solver
from wfomc.fol import a, b, context_for, simplify_boolean, substitute, true
from wfomc.fol.cnf import TseitinCNF, encode_tseitin
from wfomc.fol.grounding import ground_on_tuple
from wfomc.weights import compile_weight_mapping

from c2_wms.arithmetic import CoefficientCache, Degree
from c2_wms.discrete_sampling import ExactAliasTable, RandRange
from c2_wms.errors import SamplingError, WfomcCompatibilityError
from c2_wms.trace.kernel import projection_mask
from c2_wms.trace.traceback import PairRequest

from ._pair_sdd import WeightPair, _SddDistribution

_SAT_MODEL_LIMIT = 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _AtomCondition:
    truth: bool
    uses_predicate_weight: bool


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


class _EnumeratedDistribution:
    """A small conditioned pair distribution enumerated exactly by SAT."""

    def __init__(self, sampler: PairSampler, choices, weights):
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


class PairSampler:
    """Sample source-level pair masks from exact conditioned pair factors."""

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
            action for key in source_binary_keys for action in ((key, True), (key, False))
        )
        self.cnf: TseitinCNF | None = None
        self._source_variable_bits: tuple[tuple[int, int], ...] = ()
        self._distributions: dict[
            tuple[int, int, int], _EnumeratedDistribution | _SddDistribution
        ] = {}
        self._condition_kernels: dict[tuple[int, int, int, Degree], int | ExactAliasTable] = {}
        self._direct_cache: dict[tuple[int, int, int], tuple[int, object]] = {}
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
            source_index = self._source_binary_indices.get((predicate.name, predicate.arity))
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

    def _atom_condition(self, atom, left, right, cell_predicates, mask):
        if atom.predicate in cell_predicates and atom.terms:
            all_left = all(term == a for term in atom.terms)
            all_right = all(term == b for term in atom.terms)
            if all_left or all_right:
                cell = left if all_left else right
                return _AtomCondition(
                    cell.is_positive(atom.predicate),
                    uses_predicate_weight=False,
                )

        projection_index = self._projected_indices.get(atom.predicate)
        if projection_index is None:
            return None
        if atom.terms == (b, a):
            bit = 2 * projection_index
        elif atom.terms == (a, b):
            bit = 2 * projection_index + 1
        else:
            return None
        return _AtomCondition(
            bool(mask & (1 << bit)),
            uses_predicate_weight=True,
        )

    def _weights(self, left_index: int, right_index: int, mask: int):
        cnf = self._ensure_cnf()
        one = self.trace.arithmetic.one()
        zero = self.trace.arithmetic.zero()
        left = self.trace.component.cells[left_index]
        right = self.trace.component.cells[right_index]
        result = [(one, one) for _ in range(cnf.n_vars + 1)]
        cell_predicates = frozenset(left.preds)
        for atom, variable in cnf.atom_to_var.items():
            condition = self._atom_condition(atom, left, right, cell_predicates, mask)
            if condition is not None and not condition.uses_predicate_weight:
                result[variable] = (one, zero) if condition.truth else (zero, one)
                continue

            pair = _predicate_weight(self.predicate_weights, atom.predicate, one)
            if condition is None:
                result[variable] = pair
            elif condition.truth:
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
            condition = self._atom_condition(atom, left, right, cell_predicates, mask)
            if condition is not None:
                literals.append(variable if condition.truth else -variable)
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
