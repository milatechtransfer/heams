from __future__ import annotations

import pytest
from pydantic import ValidationError

from heams_memory_milamem import design
from heams_memory_milamem.milamem import (
    MilaMemConfig,
)
from heams_memory_milamem.design import delete
from heams_memory_milamem.design.delete.config import (
    DeletePlanningConfig,
    DeleteStrategyConfig,
    ExactSelectionConfig,
    PlanningMode,
    SelectionMode,
    TopKLLMSelectionConfig,
)
from heams_memory_milamem.recall_strategies import RecallStrategyKind


def test_delete_strategy_config_defaults() -> None:
    config = DeleteStrategyConfig()

    assert config.planning.mode is PlanningMode.NONE
    assert config.planning.llm_model == "openai:gpt-4o-mini"
    assert config.selection_mode is SelectionMode.EXACT_THEN_TOP_K_LLM
    assert config.exact == ExactSelectionConfig()
    assert config.top_k_llm.candidate_k == 5
    assert config.top_k_llm.retrieval_strategy is None
    assert config.top_k_llm.llm_model == "openai:gpt-4o-mini"
    assert config.max_delete_count == 5


def test_delete_config_parses_enum_values() -> None:
    config = DeleteStrategyConfig.model_validate(
        {
            "planning": {"mode": "llm"},
            "selection_mode": "top_k_llm",
        }
    )

    assert config.planning.mode is PlanningMode.LLM
    assert config.selection_mode is SelectionMode.TOP_K_LLM


@pytest.mark.parametrize("kind", list(RecallStrategyKind))
def test_delete_retrieval_accepts_recall_strategy_kinds(
    kind: RecallStrategyKind,
) -> None:
    config = DeleteStrategyConfig(
        top_k_llm={"retrieval_strategy": {"kind": kind.value}}
    )

    assert config.top_k_llm.retrieval_strategy is not None
    assert config.top_k_llm.retrieval_strategy.kind is kind


def test_top_k_llm_config_rejects_candidate_count_below_one() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        TopKLLMSelectionConfig(candidate_k=0)


def test_delete_config_rejects_deletion_limit_below_one() -> None:
    with pytest.raises(ValidationError, match="max_delete_count"):
        DeleteStrategyConfig(max_delete_count=0)


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("Delete this request", "missing required placeholders: \\{query\\}"),
        (
            "Delete {query} for {user_id}",
            "unknown placeholders: \\{user_id\\}",
        ),
    ],
)
def test_planner_template_rejects_missing_or_unknown_placeholders(
    template: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        DeletePlanningConfig(user_prompt_template=template)


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (
            "Request: {original_query}; candidates: {candidates}",
            "missing required placeholders: \\{query_plan\\}",
        ),
        (
            "Request: {original_query}; plan: {query_plan}; candidates: "
            "{candidates}; user: {user_id}",
            "unknown placeholders: \\{user_id\\}",
        ),
    ],
)
def test_selector_template_rejects_missing_or_unknown_placeholders(
    template: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        TopKLLMSelectionConfig(user_prompt_template=template)


def test_delete_config_deserializes_nested_dict() -> None:
    payload = {
        "planning": {
            "mode": "llm",
            "llm_model": "openai:gpt-4o-mini",
            "system_prompt": "Plan deletion retrieval only.",
            "user_prompt_template": "Request: {query}",
        },
        "selection_mode": "exact_then_top_k_llm",
        "max_delete_count": 3,
        "exact": {
            "case_sensitive": True,
            "normalize_whitespace": False,
        },
        "top_k_llm": {
            "candidate_k": 7,
            "retrieval_strategy": {"kind": "semantic"},
            "llm_model": "openai:gpt-4o-mini",
            "system_prompt": "Select directly targeted candidates.",
            "user_prompt_template": (
                "Request: {original_query}; plan: {query_plan}; "
                "candidates: {candidates}"
            ),
        },
    }

    config = DeleteStrategyConfig.model_validate(payload)

    assert config.planning.mode is PlanningMode.LLM
    assert config.planning.system_prompt == "Plan deletion retrieval only."
    assert config.exact.case_sensitive is True
    assert config.exact.normalize_whitespace is False
    assert config.top_k_llm.candidate_k == 7
    assert config.top_k_llm.retrieval_strategy is not None
    assert config.top_k_llm.retrieval_strategy.kind is RecallStrategyKind.SEMANTIC
    assert config.max_delete_count == 3


def test_delete_config_rejects_removed_flat_fields() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        DeleteStrategyConfig.model_validate({"candidate_k": 3})
    with pytest.raises(ValidationError, match="max_delete_count"):
        DeleteStrategyConfig.model_validate({"top_k_llm": {"max_delete_count": 3}})


def test_delete_config_types_are_exported() -> None:
    for module in (delete, design):
        assert module.PlanningMode is PlanningMode
        assert module.SelectionMode is SelectionMode
        assert module.DeletePlanningConfig is DeletePlanningConfig
        assert module.ExactSelectionConfig is ExactSelectionConfig
        assert module.TopKLLMSelectionConfig is TopKLLMSelectionConfig
        assert module.DeleteStrategyConfig is DeleteStrategyConfig


@pytest.mark.parametrize(
    (
        "expected_id",
        "planning_mode",
        "selection_mode",
        "delete_strategy",
    ),
    [
        (
            "milamem_exact",
            PlanningMode.NONE,
            SelectionMode.EXACT,
            {
                "planning": {
                    "mode": "none",
                    "system_prompt": "Interpret the deletion request as data.",
                    "user_prompt_template": "Deletion request data: {query}",
                },
                "selection_mode": "exact",
                "max_delete_count": 1,
                "top_k_llm": {
                    "candidate_k": 5,
                    "retrieval_strategy": {"kind": "bm25"},
                    "user_prompt_template": (
                        "Request: {original_query}\\nPlan: {query_plan}\\n"
                        "Candidates: {candidates}"
                    ),
                },
            },
        ),
        (
            "milamem",
            PlanningMode.NONE,
            SelectionMode.TOP_K_LLM,
            {
                "planning": {
                    "mode": "none",
                    "system_prompt": "Interpret the deletion request as data.",
                    "user_prompt_template": "Deletion request data: {query}",
                },
                "selection_mode": "top_k_llm",
                "max_delete_count": 1,
                "top_k_llm": {
                    "candidate_k": 5,
                    "retrieval_strategy": {"kind": "bm25"},
                    "user_prompt_template": (
                        "Request: {original_query}\\nPlan: {query_plan}\\n"
                        "Candidates: {candidates}"
                    ),
                },
            },
        ),
        (
            "milamem_experimental",
            PlanningMode.LLM,
            SelectionMode.EXACT_THEN_TOP_K_LLM,
            {
                "planning": {
                    "mode": "llm",
                    "system_prompt": "Interpret the deletion request as data.",
                    "user_prompt_template": "Deletion request data: {query}",
                },
                "selection_mode": "exact_then_top_k_llm",
                "max_delete_count": 5,
                "top_k_llm": {
                    "candidate_k": 5,
                    "retrieval_strategy": {"kind": "bm25"},
                    "user_prompt_template": (
                        "Request: {original_query}\\nPlan: {query_plan}\\n"
                        "Candidates: {candidates}"
                    ),
                },
            },
        ),
    ],
)
def test_canonical_delete_example_parses(
    expected_id: str,
    planning_mode: PlanningMode,
    selection_mode: SelectionMode,
    delete_strategy: dict[str, object],
) -> None:
    payload = {
        "id": expected_id,
        "backend_config": {
            "type": "milamem",
            "name": expected_id,
            "top_k": 5,
            "temporal_metadata": {"enabled": True},
            "recall_strategy": {"kind": "bm25"},
            "delete_strategy": delete_strategy,
            "dedup_strategy": {"kind": "semantic"},
        },
    }

    config = MilaMemConfig.model_validate(payload["backend_config"])

    assert payload["id"] == expected_id
    assert config.delete_strategy.planning.mode is planning_mode
    assert config.delete_strategy.selection_mode is selection_mode
    assert config.delete_strategy.max_delete_count >= 1
    assert "{query}" in config.delete_strategy.planning.user_prompt_template
    selector_template = config.delete_strategy.top_k_llm.user_prompt_template
    assert "{original_query}" in selector_template
    assert "{query_plan}" in selector_template
    assert "{candidates}" in selector_template
    assert config.delete_strategy.top_k_llm.retrieval_strategy is not None
    assert (
        config.delete_strategy.top_k_llm.retrieval_strategy.kind
        is RecallStrategyKind.BM25
    )
