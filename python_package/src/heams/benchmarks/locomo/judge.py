"""LOCOMO LLM-as-judge evaluator.

Grades each LOCOMO question/response pair against
`LocomoQAGroundTruth.qa[*].expected_answer` using the Mem0 community-harness
judge prompt and aggregates per-question verdicts into an `AuditRecord`.
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


class LocomoQA(BaseModel):
    """A single graded question for a LOCOMO sample."""

    model_config = ConfigDict(extra="forbid")

    question: str
    expected_answer: str
    evidence_dia_ids: list[Any] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)
    category: Any | None = None


class LocomoQAGroundTruth(BaseGroundTruth):
    """Ordered ground-truth questions for a LOCOMO scenario."""

    qa: list[LocomoQA] = Field(default_factory=list)


class LocomoLlmJudgeConfig(BaseModel):
    """Config for the LOCOMO LLM-as-judge evaluator.

    Prompt strings are required and authored in configs/benchmarks/*.yaml —
    there are no Python fallback prompt defaults.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["locomo_llm_judge"] = "locomo_llm_judge"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id forwarded to heams.evaluation.llm_judge.LlmJudge.",
    )
    retrieval_match_mode: Literal["deterministic", "llm_judge"] = Field(
        default="deterministic",
        description=(
            "How component retrieval performance should score retrieved "
            "memories when textual evidence is available."
        ),
    )
    retrieval_judge_model_id: str | None = Field(
        default=None,
        description=(
            "Optional model id for retrieval LLM-as-judge scoring. "
            "Falls back to judge_model_id when omitted."
        ),
    )
    success_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum per-scenario pass rate (correct answers / total) required "
            "for `is_successful=True` in the AuditRecord."
        ),
    )
    judge_system_prompt: str
    judge_template: str = Field(
        description=(
            "Rubric template with {evidence_section}, {evidence_rule}, "
            "{evidence_wrong_clause}, and {{question}}/{{answer}}/{{response}}."
        ),
    )
    evidence_chunk: str
    evidence_rule: str
    evidence_wrong_clause: str

    def to_ground_truth(self, scenario: Scenario) -> LocomoQAGroundTruth:
        if scenario.ground_truth is not None:
            if not isinstance(scenario.ground_truth, LocomoQAGroundTruth):
                raise TypeError(
                    f"Scenario '{scenario.id}' ground_truth must be "
                    f"LocomoQAGroundTruth, got {type(scenario.ground_truth)!r}."
                )
            return scenario.ground_truth
        raise ValueError(
            f"Scenario '{scenario.id}' is missing ground_truth required "
            "by the locomo_llm_judge evaluator."
        )


_CATEGORY_ALIASES = {
    "multi-hop": 1,
    "multi_hop": 1,
    "temporal": 2,
    "open-domain": 3,
    "open_domain": 3,
    "single-hop": 4,
    "single_hop": 4,
    "adversarial": 5,
}

LOCOMO_CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}


class _LocomoJudgeOutput(BaseModel):
    reasoning: str = ""
    label: Literal["CORRECT", "WRONG"]


def normalize_locomo_category(category: Any | None) -> int | None:
    if category is None:
        return None
    if isinstance(category, int):
        return category
    text = str(category).strip().lower()
    if text.isdigit():
        return int(text)
    return _CATEGORY_ALIASES.get(text)


def is_scored_locomo_category(category: Any | None) -> bool:
    return normalize_locomo_category(category) != 5


def locomo_category_label(category: Any | None) -> str:
    """Human category label for a LOCOMO QA's raw `category` field."""
    normalized = normalize_locomo_category(category)
    if normalized is not None and normalized in LOCOMO_CATEGORY_LABELS:
        return LOCOMO_CATEGORY_LABELS[normalized]
    if category:
        return str(category)
    return "unspecified"


def locomo_category_label_from_key(key: str) -> str:
    """Derive a human category label from an audit key for report stratification."""
    _, _, suffix = key.partition(":")
    return locomo_category_label(suffix.strip() or None)


def preprocess_locomo_answer(category: Any | None, answer: str) -> str:
    normalized_category = normalize_locomo_category(category)
    if normalized_category == 3 and ";" in answer:
        return answer.split(";", 1)[0].strip()
    return answer


def _build_judge_prompt_template(
    config: LocomoLlmJudgeConfig,
    evidence_context: str | None = None,
) -> str:
    if evidence_context:
        prompt = config.judge_template.format(
            evidence_section=config.evidence_chunk.format(
                evidence_context=evidence_context
            ),
            evidence_rule=config.evidence_rule,
            evidence_wrong_clause=config.evidence_wrong_clause,
        )
        prompt = prompt.replace("\n5. **SEMANTIC OVERLAP", "\n6. **SEMANTIC OVERLAP")
        prompt = prompt.replace("\n6. **SAME REFERENT", "\n7. **SAME REFERENT")
        prompt = prompt.replace(
            "\n7. **FOCUS ON KNOWLEDGE", "\n8. **FOCUS ON KNOWLEDGE"
        )
        return prompt
    return config.judge_template.format(
        evidence_section="",
        evidence_rule="",
        evidence_wrong_clause="",
    )


LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE = "{context}{question}"


def build_locomo_judge_prompt(
    qa: LocomoQA,
    response_text: str,
    *,
    config: LocomoLlmJudgeConfig,
) -> str:
    processed_answer = preprocess_locomo_answer(qa.category, qa.expected_answer)
    evidence_context = (
        "\n".join(text for text in qa.evidence_texts if text.strip()) or None
    )
    prompt_template = _build_judge_prompt_template(
        config, evidence_context=evidence_context
    )
    return prompt_template.format(
        question=qa.question,
        answer=processed_answer,
        response=response_text,
    )


@register("locomo_llm_judge")
class LocomoLlmJudgeEvaluator(BaseEvaluator):
    """Per-question LLM-as-judge evaluator for LOCOMO scenarios."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is not None:
            return existing
        model_id = getattr(self.config, "judge_model_id", "openai:gpt-4o-mini")
        self._judge = LlmJudge(
            model_id=model_id,
            output_type=_LocomoJudgeOutput,
            model_args={"temperature": 0},
            system_prompt=self.config.judge_system_prompt,
        )
        return self._judge

    def _run_judge(self, response: AgentResponse, qa: LocomoQA) -> _LocomoJudgeOutput:
        prompt = build_locomo_judge_prompt(qa, response.answer, config=self.config)
        verdict = self._get_judge().run(
            context=prompt,
            question="",
            prompt_template=LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE,
        )
        return self._coerce_verdict(verdict)

    async def _run_judge_async(
        self,
        response: AgentResponse,
        qa: LocomoQA,
        semaphore: asyncio.Semaphore | None,
    ) -> _LocomoJudgeOutput:
        prompt = build_locomo_judge_prompt(qa, response.answer, config=self.config)
        judge = self._get_judge()
        if semaphore is None:
            verdict = await judge.run_async(
                context=prompt,
                question="",
                prompt_template=LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE,
            )
            return self._coerce_verdict(verdict)
        async with semaphore:
            verdict = await judge.run_async(
                context=prompt,
                question="",
                prompt_template=LOCOMO_LLM_JUDGE_PROMPT_TEMPLATE,
            )
            return self._coerce_verdict(verdict)

    @staticmethod
    def _coerce_verdict(verdict: Any) -> _LocomoJudgeOutput:
        if isinstance(verdict, _LocomoJudgeOutput):
            return verdict
        if isinstance(verdict, BaseModel):
            payload = verdict.model_dump()
            try:
                return _LocomoJudgeOutput.model_validate(payload)
            except ValidationError:
                reasoning = str(payload.get("reasoning") or payload).strip()
                return _LocomoJudgeOutput(label="WRONG", reasoning=reasoning)
        try:
            return _LocomoJudgeOutput.model_validate(verdict)
        except ValidationError:
            return _LocomoJudgeOutput(label="WRONG", reasoning=str(verdict).strip())

    def _aggregate_verdicts(
        self,
        verdicts: list[Any],
        ground_truth: LocomoQAGroundTruth,
        agent_responses: list[AgentResponse],
    ) -> AuditRecord:
        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        records: list[EvaluationRecord] = []
        for index, (verdict, qa, response) in enumerate(
            zip(verdicts, ground_truth.qa, agent_responses)
        ):
            key = self._qa_key(index, qa)
            is_correct = getattr(verdict, "label", "WRONG") == "CORRECT"
            reason = str(getattr(verdict, "reasoning", "") or "").strip()
            if is_correct:
                passed.append(key)
            else:
                failed.append(key)
                if reason:
                    reasons.append(f"{key}: {reason}")
            records.append(
                EvaluationRecord(
                    key=key,
                    passed=is_correct,
                    score=1.0 if is_correct else 0.0,
                    tags={"category": locomo_category_label(qa.category)},
                    expected=preprocess_locomo_answer(qa.category, qa.expected_answer),
                    observed=response.answer,
                    reason=reason or None,
                )
            )

        total = len(passed) + len(failed)
        accuracy = len(passed) / total if total else 0.0
        threshold = float(getattr(self.config, "success_threshold", 1.0))
        notes = f"LLM-judged {total} LOCOMO questions."
        if reasons:
            notes = f"{notes} Failures: " + " | ".join(reasons[:5])

        self._last_audit = AuditRecord(
            is_successful=(total > 0 and accuracy >= threshold),
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=notes,
            report_records=records,
        )
        return self._last_audit

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: LocomoQAGroundTruth,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"LOCOMO audit expected {len(ground_truth.qa)} agent responses, "
                f"got {len(agent_responses)}."
            )

        verdicts = [
            self._run_judge(response, qa)
            for response, qa in zip(agent_responses, ground_truth.qa)
        ]
        return self._aggregate_verdicts(verdicts, ground_truth, agent_responses)

    async def audit_async(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: LocomoQAGroundTruth,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"LOCOMO audit expected {len(ground_truth.qa)} agent responses, "
                f"got {len(agent_responses)}."
            )

        verdicts = await asyncio.gather(
            *[
                self._run_judge_async(response, qa, semaphore)
                for response, qa in zip(agent_responses, ground_truth.qa)
            ]
        )
        return self._aggregate_verdicts(list(verdicts), ground_truth, agent_responses)

    @staticmethod
    def _qa_key(index: int, qa: LocomoQA) -> str:
        normalized_category = normalize_locomo_category(qa.category)
        if normalized_category is not None:
            return f"q{index}:{normalized_category}"
        if qa.category:
            return f"q{index}:{qa.category}"
        return f"q{index}"
