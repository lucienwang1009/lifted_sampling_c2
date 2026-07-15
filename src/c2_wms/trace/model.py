"""Compact value traces for the incremental3 dynamic program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from wfomc.algo.incremental3.counting_kernel import Config, ConfigSpace, State

ArithmeticValue: TypeAlias = object
Outcome: TypeAlias = tuple[State, Config]


@dataclass(frozen=True, slots=True)
class RelationChoice:
    mask: int
    weight: ArithmeticValue


@dataclass(slots=True)
class HTrace:
    layers: list[dict[Outcome, ArithmeticValue]]


@dataclass(slots=True)
class TargetTrace:
    target: State
    other_states: tuple[State, ...]
    g_layers: tuple[dict[Outcome, ArithmeticValue], ...]
    terminal_weights: dict[Config, ArithmeticValue]


@dataclass(slots=True)
class DomainTraceNode:
    config: Config
    targets: tuple[TargetTrace, ...]


@dataclass(slots=True)
class RootTerm:
    cell_config: tuple[int, ...]
    init_config: Config
    base_weight: ArithmeticValue
    domain_weight: ArithmeticValue
    mass: ArithmeticValue


@dataclass(slots=True)
class ComponentTrace:
    component: object
    reduced_problem: object | None
    space: ConfigSpace
    arithmetic: object
    counting_state: object
    has_linear_order: bool
    t_update_dict: dict
    transition_choices: dict
    h_traces: dict[tuple[State, State], HTrace]
    domain_values: dict[Config, ArithmeticValue]
    domain_nodes: dict[Config, DomainTraceNode]
    root_terms: tuple[RootTerm, ...]
    total_mass: ArithmeticValue
