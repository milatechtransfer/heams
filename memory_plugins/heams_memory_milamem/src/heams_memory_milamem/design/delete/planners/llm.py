"""Configurable LLM query planner for logical deletion."""

from __future__ import annotations

from typing import Any, cast

from any_llm.types.completion import ChatCompletionMessage, ParsedChatCompletion

from ....recall_strategies import _ensure_thread_event_loop
from ..config import DeletePlanningConfig
from ..models import DeleteQueryPlan
from .base import BaseDeleteQueryPlanner


def request_delete_query_plan(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
) -> DeleteQueryPlan:
    """Request one structured query plan; isolated as the test's LLM seam."""

    import any_llm

    # `plan()` runs from the same run_in_executor worker thread as
    # recall()/retain(); without a persistent loop on that thread, any_llm's
    # `allow_running_loop=True` sync path falls back to `asyncio.run(coro)`,
    # creating and closing a brand-new loop on every call.
    _ensure_thread_event_loop()
    response = cast(
        ParsedChatCompletion[DeleteQueryPlan],
        any_llm.completion(
            model=model,
            messages=messages,
            response_format=DeleteQueryPlan,
            temperature=0.0,
            stream=False,
            allow_running_loop=True,
        ),
    )
    plan = response.choices[0].message.parsed
    if not isinstance(plan, DeleteQueryPlan):
        raise ValueError("LLM response did not contain a parsed DeleteQueryPlan")
    return plan


class LlmDeleteQueryPlanner(BaseDeleteQueryPlanner):
    def __init__(self, config: DeletePlanningConfig):
        self.config = config

    @staticmethod
    def _fail_closed(query: str, reasoning: str) -> DeleteQueryPlan:
        return DeleteQueryPlan(
            original_query=query,
            target_text="",
            search_query="",
            cardinality="one",
            ambiguous=True,
            reasoning=reasoning,
        )

    def plan(self, query: str) -> DeleteQueryPlan:
        if not query.strip():
            return self._fail_closed(query, "The deletion request was empty.")

        messages: list[dict[str, Any] | ChatCompletionMessage] = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": self.config.user_prompt_template.format(query=query),
            },
        ]
        try:
            planned = request_delete_query_plan(self.config.llm_model, messages)
        except Exception:
            return self._fail_closed(query, "Delete query planning failed.")

        if not isinstance(planned, DeleteQueryPlan):
            return self._fail_closed(
                query, "Delete query planning returned malformed output."
            )

        target_text = " ".join(planned.target_text.split())
        search_query = " ".join(planned.search_query.split())
        exclusions = list(
            dict.fromkeys(
                normalized
                for exclusion in planned.exclusions
                if (normalized := " ".join(exclusion.split()))
            )
        )
        if planned.ambiguous:
            return planned.model_copy(
                update={
                    "original_query": query,
                    "target_text": target_text,
                    "search_query": "",
                    "exclusions": exclusions,
                }
            )
        if not target_text or not search_query:
            return self._fail_closed(
                query,
                "Delete query planning produced no usable target or search query.",
            )
        return planned.model_copy(
            update={
                "original_query": query,
                "target_text": target_text,
                "search_query": search_query,
                "exclusions": exclusions,
            }
        )
