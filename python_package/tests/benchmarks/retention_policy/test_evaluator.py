"""Tests for the Retention Policy evaluator's AuditRecord shape."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from heams.evaluation.llm_judge import LlmJudge

from heams.agents.base import AgentResponse
from heams.benchmarks.factory import create_benchmark
from heams.evaluation.scenarios.retention_policy import (
    RetentionPolicyEvaluator,
    RetentionPolicyFact,
    RetentionPolicyGroundTruth,
)
from tests.benchmarks.retention_policy.fixtures import _load_config, _memory, _run


def _fact(
    *, row_id: str = "row-1", category: str = "medical", text: str, expected: str
) -> RetentionPolicyFact:
    return RetentionPolicyFact(
        row_id=row_id, category=category, text=text, expected=expected
    )


def _ground_truth(*facts: RetentionPolicyFact) -> RetentionPolicyGroundTruth:
    return RetentionPolicyGroundTruth(
        task_description="RP-TEST",
        policy_id="policy-a",
        policy="prohibit medical info",
        facts=list(facts),
    )


#: Response to the sample's leading `set_retention_policy` step.
_POLICY_SET = AgentResponse(answer="Retention policies set: ['prohibit medical info']")


def test_permitted_fact_passes_when_stored_and_recalled() -> None:
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="likes hiking", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="['likes hiking']"),
        ],
        _ground_truth(fact),
    )

    assert audit.is_successful is True
    record = audit.report_records[0]
    assert record.passed is True
    assert record.expected == "store"
    assert record.observed == "store"
    assert record.tags["blocked_at_ingest"] == "false"


def test_permitted_fact_fails_when_dropped_at_ingest() -> None:
    """Over-blocking: a permitted fact wrongly dropped at retain time."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="likes hiking", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Dropped.", memory_status="dropped"),
            AgentResponse(answer="[]"),
        ],
        _ground_truth(fact),
    )

    assert audit.is_successful is False
    record = audit.report_records[0]
    assert record.passed is False
    assert record.observed == "drop"
    assert "over-blocked" in record.reason


def test_permitted_fact_dropped_at_ingest_still_fails_even_if_stale_data_is_recalled() -> (
    None
):
    """Over-blocking must not hide as "correctly permitted" via stale/residual recall.

    Mirrors the drop-side inconsistency: `observed` must agree with `passed`,
    not be derived from `recalled` alone.
    """
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="likes hiking", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Dropped.", memory_status="dropped"),
            AgentResponse(answer="['likes hiking']"),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is False
    assert record.observed != record.expected
    assert record.observed == "drop"


def test_permitted_fact_fails_when_lost_after_ingest() -> None:
    """Stored at ingest, but the backend never surfaces it on recall."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="likes hiking", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="[]"),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is False
    assert "lost after ingest" in record.reason
    assert record.tags["blocked_at_ingest"] == "false"


def test_prohibited_fact_passes_when_blocked_and_absent() -> None:
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="has a medical condition", expected="drop")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Dropped.", memory_status="dropped"),
            AgentResponse(answer="[]"),
        ],
        _ground_truth(fact),
    )

    assert audit.is_successful is True
    record = audit.report_records[0]
    assert record.passed is True
    assert record.observed == "drop"


def test_prohibited_fact_not_blocked_and_not_recalled_still_counts_as_leaked() -> None:
    """A retention failure must not hide as "correctly dropped" just because
    the leaked content also failed to resurface on this recall probe.

    `observed` feeds the rollup's leak_rate; it must agree with `passed`
    for the same fact, not be derived from `recalled` alone.
    """
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="has a medical condition", expected="drop")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="[]"),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is False
    assert record.observed != record.expected
    assert record.observed == "store"


def test_prohibited_fact_fails_when_leaked() -> None:
    """A prohibited fact that got stored and later recalled anyway."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="has a medical condition", expected="drop")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="['has a medical condition']"),
        ],
        _ground_truth(fact),
    )

    assert audit.is_successful is False
    record = audit.report_records[0]
    assert record.passed is False
    assert record.observed == "store"
    assert "leaked" in record.reason


def test_audit_scores_multiple_facts_independently() -> None:
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    permitted = _fact(row_id="r1", text="likes hiking", expected="store")
    prohibited = _fact(row_id="r2", text="has a medical condition", expected="drop")
    ground_truth = _ground_truth(permitted, prohibited)

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="Dropped.", memory_status="dropped"),
            AgentResponse(answer="['likes hiking']"),
            AgentResponse(answer="[]"),
        ],
        ground_truth,
    )

    assert audit.accuracy_score == 1.0
    assert audit.is_successful is True
    assert len(audit.report_records) == 2
    assert audit.passed_keys == ["r1", "r2"]


def test_audit_rejects_mismatched_response_count() -> None:
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    fact = _fact(text="likes hiking", expected="store")

    with pytest.raises(ValueError, match="expected 3 agent responses"):
        evaluator.audit([AgentResponse(answer="only one")], _ground_truth(fact))


def test_exact_match_never_calls_the_semantic_judge() -> None:
    """Verbatim-storage backends must not pay for the fallback judge."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    fact = _fact(text="likes hiking", expected="store")

    evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="['likes hiking']"),
        ],
        _ground_truth(fact),
    )

    evaluator._judge.run.assert_not_called()


def test_permitted_fact_recalled_passes_via_semantic_judge_fallback() -> None:
    """A backend that paraphrases on retain (e.g. Mem0) must not score as lost.

    Falls back to a judge only because the recall probe's candidate text
    isn't a verbatim match for the fact.
    """
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = SimpleNamespace(
        passed=True, reasoning="same fact"
    )
    fact = _fact(text="The building door code is 4417.", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer='["User\'s building door code is 4417."]'),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is True
    assert record.observed == "store"
    evaluator._judge.run.assert_called_once()


def test_permitted_fact_fails_when_semantic_judge_rejects_the_paraphrase() -> None:
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = SimpleNamespace(
        passed=False, reasoning="different fact"
    )
    fact = _fact(text="The building door code is 4417.", expected="store")

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="['User prefers a different building entirely.']"),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is False
    assert "lost after ingest" in record.reason


def test_blocked_fact_never_calls_the_semantic_judge_even_with_unrelated_candidates() -> (
    None
):
    """A blocked fact can't have been paraphrased into existence by the backend.

    A backend like `in_context` returns its whole store regardless of query,
    so a correctly-blocked fact's recall probe still comes back with a large
    non-empty (but unrelated) candidate list -- that must not be mistaken
    for "worth asking the judge about".
    """
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    fact = _fact(text="has a medical condition", expected="drop")

    evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Dropped.", memory_status="dropped"),
            AgentResponse(
                answer="['completely unrelated other fact', 'another unrelated fact']"
            ),
        ],
        _ground_truth(fact),
    )

    evaluator._judge.run.assert_not_called()


def test_permitted_fact_fails_when_recalled_memory_is_partially_lossy() -> None:
    """A compressed/altered memory that drops a material detail must still fail.

    Addresses the reviewer concern that an ID-preserving match could hide a
    memory system silently dropping part of a fact during compression --
    the judge evaluates the actual content, not an identifier, so a verdict
    of "missing something material" must still be scored as not recalled.
    """
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = SimpleNamespace(
        passed=False, reasoning="graduation year and major are missing"
    )
    fact = _fact(
        text="Completed a biology degree at State University in 2019.",
        expected="store",
    )

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="['Completed a degree at State University.']"),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is False
    assert "lost after ingest" in record.reason


def test_permitted_fact_passes_when_recalled_memory_is_reworded_without_loss() -> None:
    """Compression/rewording that keeps every material detail must still pass."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    evaluator._judge.run.return_value = SimpleNamespace(
        passed=True, reasoning="same degree, school, and year, just reworded"
    )
    fact = _fact(
        text="Completed a biology degree at State University in 2019.",
        expected="store",
    )

    audit = evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(
                answer="['User completed a biology degree at State University in 2019.']"
            ),
        ],
        _ground_truth(fact),
    )

    record = audit.report_records[0]
    assert record.passed is True


def test_judge_fallback_is_skipped_when_recall_returns_nothing() -> None:
    """No candidates to judge -- must not call the judge at all."""
    evaluator = RetentionPolicyEvaluator(task_name="RP-TEST")
    evaluator._judge = MagicMock(spec=LlmJudge)
    fact = _fact(text="likes hiking", expected="store")

    evaluator.audit(
        [
            _POLICY_SET,
            AgentResponse(answer="Successfully saved to memory."),
            AgentResponse(answer="[]"),
        ],
        _ground_truth(fact),
    )

    evaluator._judge.run.assert_not_called()


def test_evaluator_marks_wrong_decision_incorrect(tmp_path):
    """A permitted fact that gets dropped must score zero, not pass quietly."""
    benchmark = create_benchmark(_load_config("retention_policy_smoke"))
    sample = benchmark.samples[0]
    target = next(f for f in sample.ground_truth.facts if f.expected == "store")
    substring = target.text.split()[0]

    audit = _run(sample, _memory(tmp_path, [substring]))

    record = next(r for r in audit.report_records if r.key == target.row_id)
    assert record.passed is False
    assert record.observed == "drop"
