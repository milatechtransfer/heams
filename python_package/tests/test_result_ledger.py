"""Focused tests for durable result ledger behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from heams.evaluation.schemas import AuditRecord
from heams.result_ledger import (
    ResultLedger,
    ResultLedgerError,
    expected_result_keys,
    result_key,
)
from heams.results import RunResult


def _run(*, scenario_id: str = "scenario-a", sample_index: int = 0) -> RunResult:
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=sample_index,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["q0"],
            failed_keys=[],
            audit_notes="stub audit",
        ),
    )


def test_result_key_uses_scenario_and_sample_only() -> None:
    run = _run()
    changed_components = run.model_copy(
        update={"agent_index": 4, "memory_index": 5, "evaluator_index": 6}
    )

    assert result_key(run) == ("scenario-a", 0)
    assert result_key(changed_components) == result_key(run)
    assert "result_key" not in json.loads(run.model_dump_json())
    assert RunResult.model_validate_json(run.model_dump_json()) == run


def test_expected_result_keys_preserve_scenario_and_sample_axes() -> None:
    assert expected_result_keys(["scenario-a", "scenario-b"], num_samples=2) == {
        ("scenario-a", 0),
        ("scenario-a", 1),
        ("scenario-b", 0),
        ("scenario-b", 1),
    }


def test_append_is_newline_terminated_and_reopens_without_replacing(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "runs.jsonl"
    first = _run()
    second = _run(scenario_id="scenario-b", sample_index=1)

    with ResultLedger(ledger_path) as ledger:
        ledger.append(first)

    with ResultLedger(ledger_path) as ledger:
        assert ledger.completed_keys == {result_key(first)}
        ledger.append(second)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [RunResult.model_validate_json(line) for line in lines] == [first, second]


def test_duplicate_keys_are_rejected_even_when_components_differ(
    tmp_path: Path,
) -> None:
    first = _run()
    duplicate = first.model_copy(
        update={"agent_index": 1, "memory_index": 2, "evaluator_index": 3}
    )
    ledger_path = tmp_path / "runs.jsonl"
    ledger_path.write_text(
        first.model_dump_json() + "\n" + duplicate.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResultLedgerError, match="Duplicate result key"):
        ResultLedger(ledger_path)


@pytest.mark.parametrize(
    "record",
    [
        "{not valid json",
        '{"scenario_id": "missing-required-fields"}',
    ],
)
def test_newline_terminated_malformed_records_are_rejected(
    tmp_path: Path,
    record: str,
) -> None:
    ledger_path = tmp_path / "runs.jsonl"
    ledger_path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ResultLedgerError, match=r"runs\.jsonl:1"):
        ResultLedger(ledger_path)
    assert ledger_path.read_text(encoding="utf-8") == record + "\n"


@pytest.mark.parametrize("record", ["", "  "])
def test_blank_newline_terminated_records_are_rejected(
    tmp_path: Path,
    record: str,
) -> None:
    ledger_path = tmp_path / "runs.jsonl"
    ledger_path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ResultLedgerError, match=r"runs\.jsonl:1"):
        ResultLedger(ledger_path)
    assert ledger_path.read_text(encoding="utf-8") == record + "\n"


def test_unterminated_final_record_is_removed_and_not_loaded(tmp_path: Path) -> None:
    ledger_path = tmp_path / "runs.jsonl"
    first = _run()
    partial = '{"scenario_id": "interrupted"'
    durable_prefix = first.model_dump_json() + "\n"
    ledger_path.write_text(durable_prefix + partial, encoding="utf-8")

    with ResultLedger(ledger_path) as ledger:
        assert ledger.results == [first]
        assert ledger.completed_keys == {result_key(first)}

    assert ledger_path.read_text(encoding="utf-8") == durable_prefix


def test_append_flushes_and_syncs_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_calls: list[int] = []

    def fake_fsync(file_descriptor: int) -> None:
        sync_calls.append(file_descriptor)

    monkeypatch.setattr("heams.result_ledger.os.fsync", fake_fsync)

    with ResultLedger(tmp_path / "runs.jsonl") as ledger:
        ledger.append(_run())

    assert len(sync_calls) == 1
