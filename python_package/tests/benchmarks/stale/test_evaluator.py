from __future__ import annotations

import asyncio

import pytest

from heams.benchmarks.stale.judge import (
    StaleLlmJudgeEvaluator,
    StaleQAGroundTruth,
    _DimVerdict,
    _StaleAllInOneVerdict,
    stale_tags_from_key,
)
from heams.benchmarks.stale.samples import (
    sample_to_benchmark_sample,
)
from tests.benchmarks.stale.fixtures import _benchmark_config, _fixture_sample


def _scenario():
    sample = _fixture_sample()
    return sample_to_benchmark_sample(sample, config=_benchmark_config()).to_scenario()


def _ground_truth() -> StaleQAGroundTruth:
    return _benchmark_config().evaluator.to_ground_truth(_scenario())


class _FakeResponse:
    """Minimal stand-in for AgentResponse (evaluator only reads `.answer`)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer


def test_dim_verdict_accepts_pass_alias() -> None:
    assert _DimVerdict.model_validate({"reasoning": "r", "pass": True}).passed is True
    assert _DimVerdict.model_validate({"pass": False}).passed is False


def test_shipped_judge_prompt_constructs_real_judge() -> None:
    """LlmJudge runs ``system_prompt.format(response_schema=...)`` at construction,
    so any unescaped literal ``{``/``}`` in the shipped judge_system_prompt raises
    KeyError (offline, before any API call). Construct the real judge to guard it.
    """
    evaluator = StaleLlmJudgeEvaluator("stale", _benchmark_config().evaluator)
    judge = evaluator._get_judge()  # would raise if the prompt is not format-safe
    assert judge is not None
    assert evaluator._get_judge() is judge


def test_render_context_includes_all_three_dimensions() -> None:
    ground_truth = _ground_truth()
    evaluator = StaleLlmJudgeEvaluator("stale", _benchmark_config().evaluator)
    responses = [_FakeResponse("ANS1"), _FakeResponse("ANS2"), _FakeResponse("ANS3")]

    context = evaluator._render_context(ground_truth.qa, responses)

    assert ground_truth.qa[0].question in context  # dim1 question
    assert ground_truth.qa[2].question in context  # dim3 question
    assert "ANS1" in context and "ANS2" in context and "ANS3" in context
    assert ground_truth.qa[0].m_new in context


def test_configured_prompt_reaches_sync_and_async_judge() -> None:
    """The YAML judge_user_template must not be bypassed by either judge path."""
    ground_truth = _ground_truth()
    responses = [_FakeResponse("A1"), _FakeResponse("A2"), _FakeResponse("A3")]
    seen: list[str] = []
    verdict = {
        "dim1_eval": {"reasoning": "", "pass": True},
        "dim2_eval": {"reasoning": "", "pass": True},
        "dim3_eval": {"reasoning": "", "pass": True},
    }

    class _StubJudge:
        def run(self, *, context, question, prompt_template):
            seen.append(context)
            return verdict

        async def run_async(self, *, context, question, prompt_template):
            seen.append(context)
            return verdict

    evaluator = StaleLlmJudgeEvaluator("stale", _benchmark_config().evaluator)
    evaluator._judge = _StubJudge()  # type: ignore[attr-defined]

    evaluator.audit(responses, ground_truth)
    asyncio.run(evaluator.audit_async(responses, ground_truth))

    assert len(seen) == 2
    for context in seen:  # rendered from the YAML template -> M_new + responses
        assert ground_truth.qa[0].m_new in context
        assert "A1" in context


def test_aggregate_scores_all_in_one_verdict_and_tags_dimensions() -> None:
    ground_truth = _ground_truth()
    evaluator = StaleLlmJudgeEvaluator(
        "stale",
        _benchmark_config().evaluator.model_copy(update={"success_threshold": 1.0}),
    )
    verdict = _StaleAllInOneVerdict.model_validate(
        {
            "dim1_eval": {"reasoning": "aware", "pass": True},
            "dim2_eval": {"reasoning": "trap caught", "pass": True},
            "dim3_eval": {"reasoning": "too vague", "pass": False},
        }
    )

    audit = evaluator._aggregate(verdict, ground_truth)

    assert audit.accuracy_score == pytest.approx(2 / 3)
    assert audit.is_successful is False  # threshold 1.0
    assert len(audit.passed_keys) == 2 and len(audit.failed_keys) == 1
    assert audit.report_records is not None
    assert {r.tags["dimension"] for r in audit.report_records} == {
        "dim1",
        "dim2",
        "dim3",
    }
    failed = next(r for r in audit.report_records if not r.passed)
    assert failed.tags["dimension"] == "dim3"
    assert failed.reason == "too vague"


def test_aggregate_unparsable_verdict_fails_all_dimensions() -> None:
    ground_truth = _ground_truth()
    evaluator = StaleLlmJudgeEvaluator("stale", _benchmark_config().evaluator)

    audit = evaluator._aggregate(None, ground_truth)

    assert audit.accuracy_score == 0.0
    assert len(audit.failed_keys) == 3
    assert "could not be parsed" in audit.audit_notes


def test_stale_tags_from_key_roundtrip() -> None:
    tags = stale_tags_from_key("q0:UID:dim3:T2")
    assert tags == {
        "dimension": "dim3",
        "dimension_label": "implicit_task",
        "conflict_type": "T2",
    }
