import json
import logging
import sys

import pytest

import c2_wms.sampler as sampler_module
from c2_wms.cli import _model_record, _parser, main
from c2_wms.structure import PredicateKey, SampledStructure


def _write_problem(path):
    path.write_text(
        r"""
\forall X: (P(X) | ~P(X))
domain = 2
""",
        encoding="utf-8",
    )


def test_cli_verbosity_flags_select_info_and_debug():
    assert _parser().parse_args(["--input", "model.wfomcs", "-v"]).verbose == 1
    assert _parser().parse_args(["--input", "model.wfomcs", "-vv"]).verbose == 2


def test_cli_rejects_negative_sample_count(capsys):
    with pytest.raises(SystemExit) as raised:
        _parser().parse_args(["--input", "model.wfomcs", "--samples", "-1"])

    assert raised.value.code == 2
    assert "non-negative" in capsys.readouterr().err


def test_cli_compile_failure_preserves_existing_output(tmp_path, monkeypatch):
    problem = tmp_path / "model.wfomcs"
    output = tmp_path / "samples.jsonl"
    _write_problem(problem)
    output.write_text("existing output\n", encoding="utf-8")

    def fail_compilation(*_args, **_kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(sampler_module, "compile_sampler", fail_compilation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wfoms",
            "--input",
            str(problem),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        main()

    assert output.read_text(encoding="utf-8") == "existing output\n"


def test_cli_writes_complete_models_as_json_lines(tmp_path, monkeypatch, capsys):
    problem = tmp_path / "model.wfomcs"
    output = tmp_path / "samples.jsonl"
    _write_problem(problem)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wfoms",
            "--input",
            str(problem),
            "--samples",
            "2",
            "--seed",
            "7",
            "--validate",
            "--output",
            str(output),
        ],
    )

    main()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert capsys.readouterr().out == ""
    assert len(records) == 2
    for record in records:
        assert len(record["domain"]) == 2
        assert record["relations"][0]["predicate"] == "P"
        assert record["relations"][0]["arity"] == 1
        assert all(len(terms) == 1 for terms in record["relations"][0]["tuples"])


def test_cli_keeps_json_lines_on_stdout_without_output(tmp_path, monkeypatch, capsys):
    problem = tmp_path / "model.wfomcs"
    _write_problem(problem)
    monkeypatch.setattr(
        sys,
        "argv",
        ["wfoms", "--input", str(problem), "--samples", "1", "--seed", "7"],
    )

    main()

    record = json.loads(capsys.readouterr().out)
    assert len(record["domain"]) == 2
    assert record["relations"][0]["predicate"] == "P"


def test_cli_debug_logging_preserves_json_stdout(tmp_path, monkeypatch, capsys, caplog):
    problem = tmp_path / "model.wfomcs"
    _write_problem(problem)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wfoms",
            "--input",
            str(problem),
            "--samples",
            "1",
            "--seed",
            "7",
            "-vv",
        ],
    )

    with caplog.at_level(logging.DEBUG):
        main()

    record = json.loads(capsys.readouterr().out)
    messages = [record.message for record in caplog.records]
    assert len(record["domain"]) == 2
    assert any("wfoms started" in message for message in messages)
    assert any("Compiled sampler" in message for message in messages)
    assert any("Selected root" in message for message in messages)
    assert any("Sampled structure" in message for message in messages)
    assert any("wfoms completed" in message for message in messages)


def test_cli_validation_stops_before_writing_invalid_sample(tmp_path, monkeypatch, capsys):
    problem_path = tmp_path / "required-p.wfomcs"
    output = tmp_path / "samples.jsonl"
    problem_path.write_text(
        r"""
\forall X: P(X)
domain = 1
""",
        encoding="utf-8",
    )

    class FakeSampler:
        def __init__(self, problem):
            (element,) = tuple(problem.domain)
            self.samples = (
                SampledStructure.from_mapping((element,), {PredicateKey("P", 1): {(element,)}}),
                SampledStructure.from_mapping((element,), {PredicateKey("P", 1): set()}),
            )

        def sample_many(self, count):
            yield from self.samples[:count]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        sampler_module,
        "compile_sampler",
        lambda problem, seed=None: FakeSampler(problem),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wfoms",
            "--input",
            str(problem_path),
            "--samples",
            "2",
            "--validate",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "sample 2 failed validation" in captured.err
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_model_record_preserves_nullary_and_empty_relations():
    record = _model_record(
        SampledStructure.from_mapping(
            ("a",),
            {
                PredicateKey("A", 0): {()},
                PredicateKey("P", 1): set(),
            },
        )
    )

    assert record == {
        "domain": ["a"],
        "relations": [
            {"predicate": "A", "arity": 0, "tuples": [[]]},
            {"predicate": "P", "arity": 1, "tuples": []},
        ],
    }
