"""LongMemEval LLM-as-judge evaluator."""

from __future__ import annotations

import asyncio
from typing import Any

from heams.evaluation.llm_judge import LlmJudge
from pydantic import BaseModel, ConfigDict, Field

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.utils import register


class LongMemEvalQA(BaseModel):
    """A single graded question lifted from a LongMemEval scenario."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    expected_answer: str
    question_type: Any | None = None
    question_date: Any | None = None
    answer_session_ids: list[Any] = Field(default_factory=list)
    is_abstention: bool = False


class LongMemEvalQAGroundTruth(BaseGroundTruth):
    """Ordered ground-truth questions for a LongMemEval scenario."""

    qa: list[LongMemEvalQA] = Field(default_factory=list)


_FACTUAL_JUDGE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. \n\n"
    "Question: {question}\n\n"
    "Correct Answer: {expected_answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL_JUDGE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct. \n\n"
    "Question: {question}\n\n"
    "Correct Answer: {expected_answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE_JUDGE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {expected_answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE_JUDGE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\n"
    "Rubric: {expected_answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION_JUDGE_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\n"
    "Question: {question}\n\n"
    "Explanation: {expected_answer}\n\n"
    "Model Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer "
    "yes or no only."
)

_FACTUAL_TASK_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "multi-session",
}


def long_mem_eval_tags_from_key(key: str) -> dict[str, str]:
    """Derive report stratification tags from an audit key.

    Keys are produced by ``LongMemEvalLlmJudgeEvaluator._qa_key`` as
    ``q{idx}:{question_id}:{question_type}[:abstention]``. The trailing
    ``abstention`` segment is optional and is only present for abstention
    questions, so it is detected and stripped before reading the question type.
    """
    parts = key.split(":")
    is_abstention = len(parts) > 2 and parts[-1] == "abstention"
    if is_abstention:
        parts = parts[:-1]
    question_type = parts[2] if len(parts) > 2 and parts[2] else "unspecified"
    return {
        "question_type": question_type,
        "abstention": "abstention" if is_abstention else "answerable",
    }


def long_mem_eval_tags_from_qa(qa: LongMemEvalQA) -> dict[str, str]:
    """Report stratification tags read straight off a graded question."""
    return {
        "question_type": str(qa.question_type) if qa.question_type else "unspecified",
        "abstention": "abstention" if qa.is_abstention else "answerable",
    }


def build_long_mem_eval_judge_prompt(qa: LongMemEvalQA, response_text: str) -> str:
    if qa.is_abstention:
        template = _ABSTENTION_JUDGE_TEMPLATE
    elif qa.question_type in _FACTUAL_TASK_TYPES:
        template = _FACTUAL_JUDGE_TEMPLATE
    elif qa.question_type == "temporal-reasoning":
        template = _TEMPORAL_JUDGE_TEMPLATE
    elif qa.question_type == "knowledge-update":
        template = _KNOWLEDGE_UPDATE_JUDGE_TEMPLATE
    elif qa.question_type == "single-session-preference":
        template = _PREFERENCE_JUDGE_TEMPLATE
    else:
        raise NotImplementedError(
            f"Unsupported LongMemEval question type: {qa.question_type!r}"
        )

    return template.format(
        question=qa.question,
        expected_answer=qa.expected_answer,
        response=response_text,
    )


@register("long_mem_eval_llm_judge")
class LongMemEvalLlmJudgeEvaluator(BaseEvaluator):
    """Per-question LLM-as-judge evaluator for LongMemEval scenarios."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is not None:
            return existing
        model_id = getattr(self.config, "judge_model_id", "openai:gpt-4o-mini")
        self._judge = LlmJudge(model_id=model_id)
        return self._judge

    def _run_judge(self, response: AgentResponse, qa: LongMemEvalQA) -> Any:
        prompt = build_long_mem_eval_judge_prompt(qa, response.answer)
        return self._get_judge().run(
            context=prompt,
            question="Is the model response correct?",
        )

    async def _run_judge_async(
        self,
        response: AgentResponse,
        qa: LongMemEvalQA,
        semaphore: asyncio.Semaphore | None,
    ) -> Any:
        prompt = build_long_mem_eval_judge_prompt(qa, response.answer)
        judge = self._get_judge()
        if semaphore is None:
            return await judge.run_async(
                context=prompt,
                question="Is the model response correct?",
            )
        async with semaphore:
            return await judge.run_async(
                context=prompt,
                question="Is the model response correct?",
            )

    def _aggregate_verdicts(
        self,
        verdicts: list[Any],
        ground_truth: LongMemEvalQAGroundTruth,
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
            is_correct = bool(getattr(verdict, "passed", False))
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
                    tags=long_mem_eval_tags_from_qa(qa),
                    expected=qa.expected_answer,
                    observed=response.answer,
                    reason=reason or None,
                )
            )

        total = len(passed) + len(failed)
        accuracy = len(passed) / total if total else 0.0
        threshold = float(getattr(self.config, "success_threshold", 1.0))
        notes = f"LLM-judged {total} LongMemEval questions."
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
        ground_truth: LongMemEvalQAGroundTruth,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"LongMemEval audit expected {len(ground_truth.qa)} "
                f"agent responses, got {len(agent_responses)}."
            )

        verdicts = [
            self._run_judge(response, qa)
            for response, qa in zip(agent_responses, ground_truth.qa)
        ]
        return self._aggregate_verdicts(verdicts, ground_truth, agent_responses)

    async def audit_async(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: LongMemEvalQAGroundTruth,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.qa):
            raise ValueError(
                f"LongMemEval audit expected {len(ground_truth.qa)} "
                f"agent responses, got {len(agent_responses)}."
            )

        verdicts = await asyncio.gather(
            *[
                self._run_judge_async(response, qa, semaphore)
                for response, qa in zip(agent_responses, ground_truth.qa)
            ]
        )
        return self._aggregate_verdicts(list(verdicts), ground_truth, agent_responses)

    @staticmethod
    def _qa_key(index: int, qa: LongMemEvalQA) -> str:
        parts = [f"q{index}", qa.question_id]
        if qa.question_type:
            parts.append(str(qa.question_type))
        if qa.is_abstention:
            parts.append("abstention")
        return ":".join(parts)
