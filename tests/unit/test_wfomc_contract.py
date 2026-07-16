import logging
import tomllib
from fractions import Fraction
from pathlib import Path

from wfomc import AlgoName, compile_problem, parse_problem, solve
from wfomc.algo.incremental3.input import CountingDPInput
from wfomc.errors import GanakError
from wfomc.ganak import GANAK_ARJUN_COMMIT, GANAK_COMMIT

import c2_wms._wfomc_adapter as adapter_module
from c2_wms import compile_sampler
from c2_wms._wfomc_adapter import (
    PINNED_GANAK_ARJUN_REVISION,
    PINNED_GANAK_REVISION,
    PINNED_WFOMC_REVISION,
    compile_incremental3,
)


def test_sampling_adapter_warns_once_when_ganak_is_missing(monkeypatch, caplog):
    def missing_ganak():
        raise GanakError("not installed")

    monkeypatch.setattr(adapter_module, "find_ganak", missing_ganak)
    adapter_module._warn_if_ganak_missing.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="c2_wms"):
            adapter_module._warn_if_ganak_missing()
            adapter_module._warn_if_ganak_missing()
    finally:
        adapter_module._warn_if_ganak_missing.cache_clear()

    warnings = [
        record for record in caplog.records if "Ganak executable not found" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "wfomc-install-ganak" in warnings[0].getMessage()
    assert PINNED_GANAK_REVISION in warnings[0].getMessage()


def test_sampling_adapter_logs_available_ganak_at_debug(monkeypatch, caplog):
    monkeypatch.setattr(adapter_module, "find_ganak", lambda: "/tmp/ganak")
    adapter_module._warn_if_ganak_missing.cache_clear()
    try:
        with caplog.at_level(logging.DEBUG, logger="c2_wms"):
            adapter_module._warn_if_ganak_missing()
    finally:
        adapter_module._warn_if_ganak_missing.cache_clear()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"Found Ganak executable path=/tmp/ganak expected_commit={PINNED_GANAK_REVISION}"
        in messages
    )
    assert not any("Ganak executable not found" in message for message in messages)


def test_uv_source_matches_the_runtime_contract_revision():
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    assert project["tool"]["uv"]["sources"]["wfomc"]["rev"] == (PINNED_WFOMC_REVISION)
    assert project["project"]["scripts"] == {"wfoms": "c2_wms.cli:main"}
    assert PINNED_WFOMC_REVISION in (root / "uv.lock").read_text()
    assert GANAK_COMMIT == PINNED_GANAK_REVISION
    assert GANAK_ARJUN_COMMIT == PINNED_GANAK_ARJUN_REVISION
    assert PINNED_GANAK_REVISION in (root / "README.md").read_text()
    assert PINNED_GANAK_ARJUN_REVISION in (root / "README.md").read_text()


def test_pinned_wfomc_exposes_incremental3_sampling_inputs():
    problem = parse_problem(r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 2
""")

    artifacts = compile_problem(problem, algo=AlgoName.INCREMENTAL3)

    assert isinstance(artifacts.algo_input, CountingDPInput)
    assert artifacts.algo_input.counting_state is not None
    assert artifacts.algo_input.components
    component = artifacts.algo_input.components[0]
    assert component.counting_binary_relation_weights is not None
    assert len(component.counting_initial_states) == len(component.cells)
    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 4


def test_sampling_adapter_always_selects_exact_arithmetic():
    problem = parse_problem(r"""
\forall X: (P(X) | ~P(X))
domain = 1
""")

    artifacts = compile_incremental3(problem)

    assert artifacts.algo_options.weight_options.precision == "exact"


def test_sampling_root_mixture_matches_an_independent_wfomc_result():
    problem = parse_problem(r"""
\forall X: (\forall Y: ((LEQ(X,Y) | ~LEQ(X,Y)) & (P(X) | ~P(X))))
domain = 3
|P| = 1
""")

    expected = solve(problem, algo=AlgoName.INCREMENTAL3).constant_value()

    observed = Fraction(str(compile_sampler(problem, seed=9).total_weight))
    assert observed == expected


def test_facility_location_oracle_is_514080():
    problem = parse_problem(r"""
\forall X: (F(X) | C(X)) &
\forall X: (~(F(X) & C(X))) &
\forall X: (\forall Y: (S(X,Y) -> (C(X) & F(Y)))) &
\forall X: (C(X) -> (\exists_=2 Y: S(X,Y))) &
\forall Y: (F(Y) -> (\exists_=2 X: S(X,Y))) &
(\exists_=5 X: F(X))
domain = 10
""")

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 514_080
