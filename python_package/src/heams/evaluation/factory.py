from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from heams.utils import REGISTRY

from heams.benchmarks.beam.judge import BeamLlmJudgeConfig
from heams.benchmarks.contextual_integrity.judge import CILlmJudgeConfig
from heams.benchmarks.erasability_editability.judge import EELlmJudgeConfig
from heams.benchmarks.locomo.judge import LocomoLlmJudgeConfig
from heams.benchmarks.mpbench.judge import MpbenchLlmJudgeConfig
from heams.benchmarks.stale.judge import StaleLlmJudgeConfig
from heams.benchmarks.transparency.judge import TransparencyLlmJudgeConfig

from .base import BaseEvaluator
from .environments.task_environment_evaluator import TaskEnvironmentGroundTruth
from .scenarios.evomemory_single_turn_task_qa import (
    EvoMemorySingleTurnGroundTruth,
)
from .scenarios.long_mem_eval_llm_judge import (
    LongMemEvalQA,
    LongMemEvalQAGroundTruth,
)
from .scenarios.retention_policy import RetentionPolicyEvaluatorConfig
from .scenarios.task_qa import TaskQA, TaskQAGroundTruth

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario


class TaskQAEvaluatorConfig(BaseModel):
    """Config for generic task QA evaluation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["task_qa"] = "task_qa"
    match_mode: Literal[
        "normalized_exact",
        "contains",
        "multiple_choice_letter",
        "numeric_answer",
        "json_tool_call",
        "evomemory_gpqa",
        "evomemory_mmlu_pro",
        "evomemory_aime",
        "bfcl_ast",
        "llm_judge",
    ] = Field(
        default="normalized_exact",
        description="How to compare answers against expected targets.",
    )
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id used when match_mode='llm_judge'.",
    )
    retrieval_match_mode: Literal["deterministic", "llm_judge"] = Field(
        default="deterministic",
        description="How component retrieval performance should score retrieved memories when textual evidence is available.",
    )
    retrieval_judge_model_id: str | None = Field(
        default=None,
        description="Optional model id for retrieval LLM-as-judge scoring. Falls back to judge_model_id when omitted.",
    )
    success_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum task pass rate required for is_successful=True.",
    )

    def to_ground_truth(self, scenario: Scenario) -> TaskQAGroundTruth:
        from heams.scenarios.actions import UserStimulus

        tasks: list[TaskQA] = []
        for index, step in enumerate(scenario.execution_sequence):
            if not isinstance(step, UserStimulus):
                continue
            markers = step.step_evaluation_markers or {}
            expected = markers.get("expected_answer", markers.get("target"))
            if expected is None or str(expected) == "":
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {index} is missing "
                    "`step_evaluation_markers.expected_answer` or "
                    "`step_evaluation_markers.target` required "
                    "by the task_qa evaluator."
                )
            tasks.append(
                TaskQA(
                    question=step.content,
                    expected_answer=str(expected),
                    task_id=(
                        str(markers["task_id"])
                        if markers.get("task_id") is not None
                        else None
                    ),
                    metadata=dict(markers.get("metadata") or {}),
                    difficulty=(
                        str(markers["difficulty"])
                        if markers.get("difficulty") is not None
                        else None
                    ),
                    domain=(
                        str(markers["domain"])
                        if markers.get("domain") is not None
                        else None
                    ),
                    dataset=(
                        str(markers["dataset"])
                        if markers.get("dataset") is not None
                        else None
                    ),
                )
            )
        return TaskQAGroundTruth(task_description=scenario.name, tasks=tasks)


class EvoMemorySingleTurnTaskQAEvaluatorConfig(TaskQAEvaluatorConfig):
    """Task QA config for interleaved EvoMemory task and feedback turns."""

    type: Literal["evomemory_single_turn_task_qa"] = "evomemory_single_turn_task_qa"

    def to_ground_truth(self, scenario: Scenario) -> EvoMemorySingleTurnGroundTruth:
        from heams.scenarios.actions import UserStimulus

        task_steps: list[UserStimulus] = []
        response_indices: list[int] = []
        expected_turn = "task"
        stimulus_index = -1

        for sequence_index, step in enumerate(scenario.execution_sequence):
            if not isinstance(step, UserStimulus):
                continue
            stimulus_index += 1
            markers = step.step_evaluation_markers or {}
            turn = markers.get("evomemory_turn")
            if turn not in {"task", "feedback"}:
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {sequence_index} has "
                    f"invalid `step_evaluation_markers.evomemory_turn`: {turn!r}."
                )
            if turn != expected_turn:
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {sequence_index} must be "
                    f"an EvoMemory {expected_turn} turn, got {turn!r}."
                )

            if turn == "task":
                task_steps.append(step)
                response_indices.append(stimulus_index)
                expected_turn = "feedback"
            else:
                if "expected_answer" in markers or "target" in markers:
                    raise ValueError(
                        f"Scenario '{scenario.id}' feedback stimulus "
                        f"{sequence_index} must not define a scoring target."
                    )
                expected_turn = "task"

        if not task_steps:
            raise ValueError(
                f"Scenario '{scenario.id}' has no EvoMemory single-turn tasks."
            )
        if expected_turn == "feedback":
            raise ValueError(
                f"Scenario '{scenario.id}' is missing feedback after its final task."
            )

        task_scenario = scenario.model_copy(update={"execution_sequence": task_steps})
        task_ground_truth = super().to_ground_truth(task_scenario)
        return EvoMemorySingleTurnGroundTruth(
            task_description=task_ground_truth.task_description,
            tasks=task_ground_truth.tasks,
            response_indices=response_indices,
        )


class LongMemEvalLlmJudgeConfig(BaseModel):
    """Config for the LongMemEval LLM-as-judge evaluator."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["long_mem_eval_llm_judge"] = "long_mem_eval_llm_judge"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id forwarded to heams.evaluation.llm_judge.LlmJudge.",
    )
    retrieval_match_mode: Literal["deterministic", "llm_judge"] = Field(
        default="deterministic",
        description="How component retrieval performance should score retrieved memories when textual evidence is available.",
    )
    retrieval_judge_model_id: str | None = Field(
        default=None,
        description="Optional model id for retrieval LLM-as-judge scoring. Falls back to judge_model_id when omitted.",
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

    def to_ground_truth(self, scenario: Scenario) -> LongMemEvalQAGroundTruth:
        from heams.scenarios.actions import UserStimulus

        qa: list[LongMemEvalQA] = []
        for step in scenario.execution_sequence:
            if not isinstance(step, UserStimulus):
                continue
            markers = step.step_evaluation_markers or {}
            expected = markers.get("expected_answer")
            if expected is None:
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus is missing "
                    "`step_evaluation_markers.expected_answer` required "
                    "by the long_mem_eval_llm_judge evaluator."
                )
            question_id = markers.get("question_id") or scenario.id
            qa.append(
                LongMemEvalQA(
                    question_id=str(question_id),
                    question=str(markers.get("question") or step.content),
                    expected_answer=str(expected),
                    question_type=markers.get("question_type"),
                    question_date=markers.get("question_date"),
                    answer_session_ids=list(markers.get("answer_session_ids") or []),
                    is_abstention=bool(markers.get("is_abstention", False)),
                )
            )
        return LongMemEvalQAGroundTruth(task_description=scenario.name, qa=qa)


class TaskEnvironmentEvaluatorConfig(BaseModel):
    """Config for generic stateful task-environment evaluation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["task_environment_eval"] = "task_environment_eval"
    expected_task_ids: list[str]
    success_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Benchmark-supplied dimensions (e.g. dataset name) to stamp onto "
            "this run's report_records for sub-metric reporting."
        ),
    )

    def to_ground_truth(self, scenario: Scenario) -> TaskEnvironmentGroundTruth:
        return TaskEnvironmentGroundTruth(
            task_description=scenario.name,
            expected_task_ids=self.expected_task_ids,
            success_threshold=self.success_threshold,
            tags=self.tags,
        )


EvaluatorConfig = Annotated[
    Union[
        LocomoLlmJudgeConfig,
        LongMemEvalLlmJudgeConfig,
        TaskQAEvaluatorConfig,
        EvoMemorySingleTurnTaskQAEvaluatorConfig,
        TransparencyLlmJudgeConfig,
        StaleLlmJudgeConfig,
        EELlmJudgeConfig,
        CILlmJudgeConfig,
        TaskEnvironmentEvaluatorConfig,
        MpbenchLlmJudgeConfig,
        BeamLlmJudgeConfig,
        RetentionPolicyEvaluatorConfig,
    ],
    Field(discriminator="type"),
]


def create_evaluator(config: EvaluatorConfig, *, task_name: str) -> BaseEvaluator:
    class_type = REGISTRY.get(config.type)
    if not class_type:
        raise ValueError(f"Class {config.type} not found in the class registry.")
    return class_type(task_name=task_name, config=config)
