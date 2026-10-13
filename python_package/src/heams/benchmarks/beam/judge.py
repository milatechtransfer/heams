"""BEAM LLM-as-judge: rubric nuggets plus Kendall τ-b for event_ordering."""

from __future__ import annotations

import statistics
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from heams.evaluation.llm_judge import LlmJudge
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from heams.agents.base import AgentResponse
from heams.benchmarks.beam.metrics import clamp_nugget_score, compute_kendall_tau_b
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.utils import register

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario

BEAM_LLM_JUDGE_PROMPT_TEMPLATE = "{context}{question}"
BEAM_PASS_THRESHOLD = 0.5


class BeamQA(BaseModel):
    """One probing question for a BEAM conversation."""

    model_config = ConfigDict(extra="forbid")

    question: str
    question_type: str
    nuggets: list[str] = Field(default_factory=list)
    token_scale: str = "100K"
    difficulty: str = "unspecified"


class BeamQAGroundTruth(BaseGroundTruth):
    """Ordered probing questions for one BEAM conversation."""

    qa: list[BeamQA] = Field(default_factory=list)
    token_scale: str = "100K"


class BeamLlmJudgeConfig(BaseModel):
    """Config for the BEAM nugget LLM-as-judge evaluator.

    Prompt strings are required and authored in configs/benchmarks/*.yaml.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["beam_llm_judge"] = "beam_llm_judge"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id forwarded to heams.evaluation.llm_judge.LlmJudge.",
    )
    success_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of passing questions for is_successful.",
    )
    pass_threshold: float = Field(
        default=BEAM_PASS_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Harness binarization only (BEAM §2.4). Nine abilities: nugget mean. "
            "event_ordering: recall × (τ-b + 1) / 2. Pass iff score >= this."
        ),
    )
    judge_system_prompt: str
    nugget_judge_template: str = Field(
        description="Rubric template with {question}, {nugget}, and {response}."
    )
    event_extract_template: str = Field(
        description="Template with {response}; must request JSON {events: [...]}."
    )
    event_align_template: str = Field(
        description=(
            "Template with {extracted_event} and {reference_events}; "
            "must request JSON {index: int}."
        )
    )

    def to_ground_truth(self, scenario: Scenario) -> BeamQAGroundTruth:
        if scenario.ground_truth is not None:
            if not isinstance(scenario.ground_truth, BeamQAGroundTruth):
                raise TypeError(
                    f"Scenario '{scenario.id}' ground_truth must be "
                    f"BeamQAGroundTruth, got {type(scenario.ground_truth)!r}."
                )
            return scenario.ground_truth
        raise ValueError(
            f"Scenario '{scenario.id}' is missing ground_truth required "
            "by the beam_llm_judge evaluator."
        )


class _NuggetJudgeOutput(BaseModel):
    score: float = 0.0
    reason: str = ""


class _EventExtractOutput(BaseModel):
    events: list[str] = Field(default_factory=list)


class _EventAlignOutput(BaseModel):
    index: int = -1
    reason: str = ""


def question_type_from_audit_key(key: str) -> str:
    """Derive ability name from audit keys shaped `q{index}:{question_type}`."""
    _, _, suffix = key.partition(":")
    return suffix.strip() or "unspecified"


@register("beam_llm_judge")
class BeamLlmJudgeEvaluator(BaseEvaluator):
    """Per-question nugget judge for BEAM scenarios."""

    def _get_judge(self, output_type: type[BaseModel]) -> LlmJudge:
        cache_name = f"_judge_{output_type.__name__}"
        existing = getattr(self, cache_name, None)
        if existing is not None:
            return existing
        judge = LlmJudge(
            model_id=self.config.judge_model_id,
            output_type=output_type,
            model_args={"temperature": 0, "timeout": 120},
            system_prompt=self.config.judge_system_prompt,
        )
        setattr(self, cache_name, judge)
        return judge

    @staticmethod
    @contextmanager
    def _silence_pydantic_serialization_noise():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*PydanticSerializationUnexpectedValue.*",
                category=UserWarning,
            )
            yield

    def _run_structured(self, prompt: str, output_type: type[BaseModel]) -> BaseModel:
        with self._silence_pydantic_serialization_noise():
            return self._get_judge(output_type).run(
                context=prompt,
                question="",
                prompt_template=BEAM_LLM_JUDGE_PROMPT_TEMPLATE,
            )

    def _judge_nugget(self, qa: BeamQA, response_text: str, nugget: str) -> float:
        prompt = self.config.nugget_judge_template.format(
            question=qa.question,
            nugget=nugget,
            response=response_text,
        )
        verdict = self._run_structured(prompt, _NuggetJudgeOutput)
        payload = (
            verdict.model_dump() if isinstance(verdict, BaseModel) else dict(verdict)
        )
        try:
            parsed = _NuggetJudgeOutput.model_validate(payload)
            return clamp_nugget_score(float(parsed.score))
        except (TypeError, ValueError, ValidationError):
            return 0.0

    def _event_ordering_predicted(self, qa: BeamQA, response_text: str) -> list[int]:
        if len(qa.nuggets) < 2:
            return []
        extract_prompt = self.config.event_extract_template.format(
            response=response_text
        )
        extracted = self._run_structured(extract_prompt, _EventExtractOutput)
        events: list[str] = []
        if isinstance(extracted, _EventExtractOutput):
            events = [str(item) for item in extracted.events]
        elif isinstance(extracted, BaseModel):
            payload = extracted.model_dump()
            events = [str(item) for item in payload.get("events") or []]
        elif isinstance(extracted, list):
            events = [str(item) for item in extracted]
        if not events:
            return []

        predicted: list[int] = []
        reference_events = "\n".join(
            f"{index}. {nugget}" for index, nugget in enumerate(qa.nuggets)
        )
        for event in events:
            align_prompt = self.config.event_align_template.format(
                extracted_event=event,
                reference_events=reference_events,
            )
            aligned = self._run_structured(align_prompt, _EventAlignOutput)
            index = -1
            if isinstance(aligned, _EventAlignOutput):
                index = aligned.index
            elif isinstance(aligned, BaseModel):
                try:
                    index = int(aligned.model_dump().get("index", -1))
                except (TypeError, ValueError):
                    index = -1
            if 0 <= index < len(qa.nuggets):
                predicted.append(index)
        return predicted

    def _score_question(self, response: AgentResponse, qa: BeamQA) -> tuple[float, str]:
        if not qa.nuggets:
            return 0.0, "No rubric nuggets found"
        if qa.question_type == "event_ordering":
            # BEAM §2.4: Kendall τ-b with presence, not nugget-averaged.
            predicted = self._event_ordering_predicted(qa, response.answer)
            if not predicted:
                return 0.0, "recall=0.000 tau_b=0.000"
            n_nuggets = len(qa.nuggets)
            recall = len(set(predicted)) / n_nuggets
            tau_b = compute_kendall_tau_b(predicted, list(range(n_nuggets)))
            score = recall * (tau_b + 1.0) / 2.0
            return score, f"recall={recall:.3f} tau_b={tau_b:.3f}"
        nugget_scores = [
            self._judge_nugget(qa, response.answer, nugget) for nugget in qa.nuggets
        ]
        avg_score = statistics.mean(nugget_scores) if nugget_scores else 0.0
        return avg_score, f"nuggets={avg_score:.3f}"

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: BeamQAGroundTruth,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"BEAM audit expected {len(ground_truth.qa)} agent responses, "
                f"got {len(agent_responses)}."
            )

        passed: list[str] = []
        failed: list[str] = []
        report_records: list[EvaluationRecord] = []
        reasons: list[str] = []
        threshold = float(self.config.pass_threshold)
        for index, (response, qa) in enumerate(zip(agent_responses, ground_truth.qa)):
            key = f"q{index}:{qa.question_type}"
            score, notes = self._score_question(response, qa)
            ok = score >= threshold
            report_records.append(
                EvaluationRecord(
                    key=key,
                    passed=ok,
                    score=score,
                    tags={
                        "question_type": qa.question_type,
                        "token_scale": qa.token_scale or ground_truth.token_scale,
                        "difficulty": qa.difficulty,
                    },
                    expected="\n".join(qa.nuggets),
                    observed=response.answer,
                    reason=notes,
                )
            )
            if ok:
                passed.append(key)
            else:
                failed.append(key)
                reasons.append(f"{key}: {notes}")

        total = len(passed) + len(failed)
        accuracy = len(passed) / total if total else 0.0
        success_threshold = float(self.config.success_threshold)
        audit_notes = f"LLM-judged {total} BEAM probing questions."
        if reasons:
            audit_notes = f"{audit_notes} Failures: " + " | ".join(reasons[:5])
        self._last_audit = AuditRecord(
            is_successful=(total > 0 and accuracy >= success_threshold),
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=audit_notes,
            report_records=report_records,
        )
        return self._last_audit


__all__ = [
    "BEAM_PASS_THRESHOLD",
    "BeamLlmJudgeConfig",
    "BeamLlmJudgeEvaluator",
    "BeamQA",
    "BeamQAGroundTruth",
    "question_type_from_audit_key",
]
