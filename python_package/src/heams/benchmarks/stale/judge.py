"""Config-driven STALE all-in-one LLM-as-judge (self-contained).

STALE probes whether a memory-augmented agent notices that a stored memory has
become *stale*: a user's state changed from ``M_old`` to ``M_new`` and the agent
is probed along three dimensions (dim1/dim2/dim3). Matching the STALE repo
(https://github.com/icedreamc/STALE), a **single** judge call grades all three
responses at once and returns JSON ``{dim1_eval, dim2_eval, dim3_eval}`` where
each eval has ``reasoning`` and ``pass``.

Every prompt (the system rubric and the user template) comes from
``StaleLlmJudgeConfig`` — authored in ``configs/benchmarks/*.yaml`` — so there are
no Python prompt fallbacks.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from heams.evaluation.llm_judge import LlmJudge
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.utils import register

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario

#: Human-readable name for each STALE probing dimension.
STALE_DIMENSION_LABELS: dict[str, str] = {
    "dim1": "explicit_probing",
    "dim2": "adversarial_robustness",
    "dim3": "implicit_task",
}

# Wrapper template handed to LlmJudge.run; the real rubric is the system prompt
# plus the fully-rendered context, so this just passes context through.
STALE_LLM_JUDGE_PROMPT_TEMPLATE = "{context}{question}"


# --------------------------------------------------------------------------- #
# Ground truth                                                                 #
# --------------------------------------------------------------------------- #


class StaleQA(BaseModel):
    """A single graded probing question lifted from a STALE scenario."""

    model_config = ConfigDict(extra="forbid")

    uid: str
    dimension: str
    question: str
    m_old: str
    m_new: str
    explanation: str = ""
    conflict_type: str = "unknown"


class StaleQAGroundTruth(BaseGroundTruth):
    """Ordered ground-truth probes (dim1, dim2, dim3) for a STALE scenario."""

    qa: list[StaleQA] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Judge output schema (STALE's all-in-one JSON)                                #
# --------------------------------------------------------------------------- #


class _DimVerdict(BaseModel):
    # STALE emits the key `pass` (a Python keyword), so alias it.
    model_config = ConfigDict(populate_by_name=True)

    reasoning: str = ""
    passed: bool = Field(alias="pass")


class _StaleAllInOneVerdict(BaseModel):
    dim1_eval: _DimVerdict
    dim2_eval: _DimVerdict
    dim3_eval: _DimVerdict

    def by_dimension(self) -> dict[str, _DimVerdict]:
        return {
            "dim1": self.dim1_eval,
            "dim2": self.dim2_eval,
            "dim3": self.dim3_eval,
        }


def stale_qa_key(index: int, qa: StaleQA) -> str:
    """Stable audit key: ``q{idx}:{uid}:{dimension}:{conflict_type}``."""
    return ":".join([f"q{index}", qa.uid, qa.dimension, qa.conflict_type])


def stale_tags_from_key(key: str) -> dict[str, str]:
    """Recover report stratification tags from an audit key produced above."""
    parts = key.split(":")
    dimension = parts[2] if len(parts) > 2 and parts[2] else "unknown"
    conflict_type = parts[3] if len(parts) > 3 and parts[3] else "unknown"
    return {
        "dimension": dimension,
        "dimension_label": STALE_DIMENSION_LABELS.get(dimension, dimension),
        "conflict_type": conflict_type,
    }


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #


@register("stale_llm_judge")
class StaleLlmJudgeEvaluator(BaseEvaluator):
    """All-in-one LLM-as-judge for STALE: one JSON call scores all three dims."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is None:
            existing = LlmJudge(
                model_id=self.config.judge_model_id,
                output_type=_StaleAllInOneVerdict,
                # `timeout` guards against a hung request freezing the whole run.
                model_args={"temperature": 0, "timeout": 120},
                system_prompt=self.config.judge_system_prompt,
            )
            self._judge = existing
        return existing

    def _render_context(self, qa: list[StaleQA], responses: list[AgentResponse]) -> str:
        by_dim = {q.dimension: (q, r) for q, r in zip(qa, responses)}
        missing = [d for d in ("dim1", "dim2", "dim3") if d not in by_dim]
        if missing:
            raise ValueError(
                f"STALE all-in-one judge requires dim1/dim2/dim3; missing {missing}."
            )
        anchor = qa[0]
        return self.config.judge_user_template.format(
            m_old=anchor.m_old,
            m_new=anchor.m_new,
            explanation=anchor.explanation or "(not provided)",
            dim1_question=by_dim["dim1"][0].question,
            dim1_response=by_dim["dim1"][1].answer,
            dim2_question=by_dim["dim2"][0].question,
            dim2_response=by_dim["dim2"][1].answer,
            dim3_question=by_dim["dim3"][0].question,
            dim3_response=by_dim["dim3"][1].answer,
        )

    @staticmethod
    def _coerce(verdict: Any) -> _StaleAllInOneVerdict | None:
        if isinstance(verdict, _StaleAllInOneVerdict):
            return verdict
        if isinstance(verdict, BaseModel):
            verdict = verdict.model_dump()
        try:
            return _StaleAllInOneVerdict.model_validate(verdict)
        except ValidationError:
            return None

    def _aggregate(
        self,
        verdict: _StaleAllInOneVerdict | None,
        ground_truth: StaleQAGroundTruth,
    ) -> AuditRecord:
        by_dim = verdict.by_dimension() if verdict is not None else {}
        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        records: list[EvaluationRecord] = []
        for index, qa in enumerate(ground_truth.qa):
            key = stale_qa_key(index, qa)
            dim_eval = by_dim.get(qa.dimension)
            is_passed = bool(dim_eval and dim_eval.passed)
            reason = (dim_eval.reasoning if dim_eval else "").strip()
            if is_passed:
                passed.append(key)
            else:
                failed.append(key)
                if reason:
                    reasons.append(f"{key}: {reason}")
            records.append(
                EvaluationRecord(
                    key=key,
                    passed=is_passed,
                    score=1.0 if is_passed else 0.0,
                    tags=stale_tags_from_key(key),
                    reason=reason or None,
                )
            )

        total = len(passed) + len(failed)
        accuracy = len(passed) / total if total else 0.0
        threshold = float(getattr(self.config, "success_threshold", 1.0))
        notes = f"All-in-one judged {total} STALE dimensions."
        if verdict is None:
            notes = "Judge output could not be parsed; all dimensions failed."
        elif reasons:
            notes = f"{notes} Failures: " + " | ".join(reasons[:3])

        self._last_audit = AuditRecord(
            is_successful=(total > 0 and accuracy >= threshold),
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=notes,
            report_records=records,
        )
        return self._last_audit

    @staticmethod
    def _check_lengths(
        agent_responses: list[AgentResponse], ground_truth: StaleQAGroundTruth
    ) -> None:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"STALE audit expected {len(ground_truth.qa)} agent responses, "
                f"got {len(agent_responses)}."
            )

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: StaleQAGroundTruth,
    ) -> AuditRecord:
        self._check_lengths(agent_responses, ground_truth)
        context = self._render_context(ground_truth.qa, agent_responses)
        raw = self._get_judge().run(
            context=context,
            question="",
            prompt_template=STALE_LLM_JUDGE_PROMPT_TEMPLATE,
        )
        return self._aggregate(self._coerce(raw), ground_truth)

    async def audit_async(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: StaleQAGroundTruth,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AuditRecord:
        self._check_lengths(agent_responses, ground_truth)
        context = self._render_context(ground_truth.qa, agent_responses)
        judge = self._get_judge()

        async def _run() -> Any:
            return await judge.run_async(
                context=context,
                question="",
                prompt_template=STALE_LLM_JUDGE_PROMPT_TEMPLATE,
            )

        if semaphore is None:
            raw = await _run()
        else:
            async with semaphore:
                raw = await _run()
        return self._aggregate(self._coerce(raw), ground_truth)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class StaleLlmJudgeConfig(BaseModel):
    """Config for the STALE all-in-one LLM-as-judge evaluator.

    Prompt strings are required and authored in configs/benchmarks/*.yaml —
    there are no Python fallback prompt defaults.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["stale_llm_judge"] = "stale_llm_judge"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id forwarded to heams.evaluation.llm_judge.LlmJudge.",
    )
    success_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum per-scenario pass rate (passed dimensions / 3) required "
            "for `is_successful=True` in the AuditRecord."
        ),
    )
    judge_system_prompt: str = Field(
        description="All-in-one judge rubric; passed to LlmJudge as the system prompt.",
    )
    judge_user_template: str = Field(
        description=(
            "User-message template; must include {m_old}, {m_new}, {explanation}, "
            "and {dim1_question}/{dim1_response}/{dim2_question}/{dim2_response}/"
            "{dim3_question}/{dim3_response}."
        ),
    )

    def to_ground_truth(self, scenario: Scenario) -> StaleQAGroundTruth:
        if scenario.ground_truth is not None:
            if not isinstance(scenario.ground_truth, StaleQAGroundTruth):
                raise TypeError(
                    f"Scenario '{scenario.id}' ground_truth must be "
                    f"StaleQAGroundTruth, got {type(scenario.ground_truth)!r}."
                )
            return scenario.ground_truth
        return self._ground_truth_from_markers(scenario)

    def _ground_truth_from_markers(self, scenario: Scenario) -> StaleQAGroundTruth:
        """Compatibility path for scenarios that carry markers but no ground_truth."""
        from heams.scenarios.actions import UserStimulus

        qa: list[StaleQA] = []
        for index, step in enumerate(scenario.execution_sequence):
            if not isinstance(step, UserStimulus):
                continue
            markers = step.step_evaluation_markers or {}
            for required in ("dimension", "m_old", "m_new"):
                if not markers.get(required):
                    raise ValueError(
                        f"Scenario '{scenario.id}' stimulus {index} is missing "
                        f"`step_evaluation_markers.{required}` required by the "
                        "stale_llm_judge evaluator."
                    )
            qa.append(
                StaleQA(
                    uid=str(markers.get("uid") or scenario.id),
                    dimension=str(markers["dimension"]),
                    question=str(markers.get("question") or step.content),
                    m_old=str(markers["m_old"]),
                    m_new=str(markers["m_new"]),
                    explanation=str(markers.get("explanation") or ""),
                    conflict_type=str(markers.get("conflict_type") or "unknown"),
                )
            )
        return StaleQAGroundTruth(task_description=scenario.name, qa=qa)


__all__ = [
    "STALE_DIMENSION_LABELS",
    "STALE_LLM_JUDGE_PROMPT_TEMPLATE",
    "StaleLlmJudgeConfig",
    "StaleLlmJudgeEvaluator",
    "StaleQA",
    "StaleQAGroundTruth",
    "stale_qa_key",
    "stale_tags_from_key",
]
