"""Transparency LLM-as-judge evaluator config (typed benchmark path)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.evaluation.scenarios.transparency_llm_judge import (
    TransparencyQA,
    TransparencyQAGroundTruth,
)

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario


class TransparencyLlmJudgeConfig(BaseModel):
    """Config for the transparency audit LLM-as-judge evaluator."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["transparency_llm_judge"] = "transparency_llm_judge"
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
            "Minimum average transparency rubric score across criteria "
            "required for `is_successful=True` in the AuditRecord."
        ),
    )
    item_success_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum per-criterion score required to include a criterion "
            "in passed_keys."
        ),
    )

    def to_ground_truth(self, scenario: Scenario) -> TransparencyQAGroundTruth:
        if scenario.ground_truth is not None:
            if not isinstance(scenario.ground_truth, TransparencyQAGroundTruth):
                raise TypeError(
                    f"Scenario '{scenario.id}' ground_truth must be "
                    f"TransparencyQAGroundTruth, got "
                    f"{type(scenario.ground_truth)!r}."
                )
            return scenario.ground_truth
        return self._ground_truth_from_markers(scenario)

    def _ground_truth_from_markers(
        self, scenario: Scenario
    ) -> TransparencyQAGroundTruth:
        """Compatibility path for legacy scenarios that rely on markers only."""
        from heams.scenarios.actions import UserStimulus

        qa: list[TransparencyQA] = []
        for index, step in enumerate(scenario.execution_sequence):
            if not isinstance(step, UserStimulus):
                continue
            markers = step.step_evaluation_markers or {}
            if "scenario_type" not in markers:
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {index} is missing "
                    "`step_evaluation_markers.scenario_type` required "
                    "by the transparency_llm_judge evaluator."
                )
            scenario_type = markers["scenario_type"]
            if not isinstance(scenario_type, str) or not scenario_type.strip():
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {index} has invalid "
                    "`step_evaluation_markers.scenario_type`: expected a "
                    "non-empty string."
                )
            if "rubric" not in markers:
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {index} is missing "
                    "`step_evaluation_markers.rubric` required "
                    "by the transparency_llm_judge evaluator."
                )
            rubric = markers["rubric"]
            if not isinstance(rubric, str) or not rubric.strip():
                raise ValueError(
                    f"Scenario '{scenario.id}' stimulus {index} has invalid "
                    "`step_evaluation_markers.rubric`: expected a non-empty "
                    "string."
                )
            expected_metadata = list(markers.get("expected_metadata") or [])
            for metadata_index, metadata in enumerate(expected_metadata):
                fact = metadata.get("fact") if isinstance(metadata, dict) else None
                if not isinstance(fact, str) or not fact.strip():
                    raise ValueError(
                        f"Scenario '{scenario.id}' stimulus {index} has invalid "
                        f"`step_evaluation_markers.expected_metadata"
                        f"[{metadata_index}].fact`: expected a non-empty string."
                    )

            question_id = markers.get("question_id") or scenario.id
            retrieval_evidence = markers.get("retrieval_evidence")
            stored_inventory: list[str] = []
            if isinstance(retrieval_evidence, dict):
                raw_contents = retrieval_evidence.get("relevant_contents")
                if isinstance(raw_contents, list):
                    stored_inventory = [
                        str(item) for item in raw_contents if str(item).strip()
                    ]
            qa.append(
                TransparencyQA(
                    question_id=str(question_id),
                    question=str(markers.get("question") or step.content),
                    scenario_type=scenario_type,
                    expected_facts=list(markers.get("expected_facts") or []),
                    expected_metadata=expected_metadata,
                    prohibited_facts=list(markers.get("prohibited_facts") or []),
                    stored_inventory=stored_inventory,
                    rubric=rubric,
                    privacy_domain=(
                        str(markers["privacy_domain"])
                        if markers.get("privacy_domain") is not None
                        else None
                    ),
                    persona=dict(markers.get("persona") or {}),
                    difficulty=dict(markers.get("difficulty") or {}),
                )
            )
        return TransparencyQAGroundTruth(task_description=scenario.name, qa=qa)


__all__ = [
    "TransparencyLlmJudgeConfig",
    "TransparencyQA",
    "TransparencyQAGroundTruth",
]
