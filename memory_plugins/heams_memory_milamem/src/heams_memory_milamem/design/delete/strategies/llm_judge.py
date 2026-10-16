"""LLM-judged logical-delete candidate selection."""

from __future__ import annotations

import json
from typing import Any, cast

from any_llm.types.completion import ChatCompletionMessage, ParsedChatCompletion

from ....recall_strategies import _ensure_thread_event_loop
from ..config import DeleteStrategyConfig
from ..models import DeleteQueryPlan
from .base import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
)


def request_delete_decision(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
) -> DeleteDecision:
    """Request one structured decision; isolated as the test's LLM seam."""

    import any_llm

    # `select()` runs from the same run_in_executor worker thread as
    # recall()/retain(); without a persistent loop on that thread, any_llm's
    # `allow_running_loop=True` sync path falls back to `asyncio.run(coro)`,
    # creating and closing a brand-new loop on every call.
    _ensure_thread_event_loop()
    response = cast(
        ParsedChatCompletion[DeleteDecision],
        any_llm.completion(
            model=model,
            messages=messages,
            response_format=DeleteDecision,
            temperature=0.0,
            stream=False,
            allow_running_loop=True,
        ),
    )
    decision = response.choices[0].message.parsed
    if not isinstance(decision, DeleteDecision):
        raise TypeError("LLM response did not contain a parsed DeleteDecision")
    return decision


class LlmJudgeDeleteStrategy(BaseDeleteStrategy):
    """Uses an LLM to judge which bounded candidates a delete query targets."""

    def __init__(self, config: DeleteStrategyConfig):
        self.config = config

    @staticmethod
    def _fail_closed(reasoning: str) -> DeleteDecision:
        return DeleteDecision(
            selected_ids=[],
            reasoning=reasoning,
            ambiguous=True,
        )

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        if query_plan.ambiguous:
            return self._fail_closed("The delete query plan is ambiguous.")
        if not query_plan.original_query.strip() or not candidates:
            return DeleteDecision(
                selected_ids=[],
                reasoning="No deletion query or candidates were supplied.",
                ambiguous=False,
            )

        exclusions = {
            " ".join(exclusion.split()).casefold()
            for exclusion in query_plan.exclusions
            if exclusion.strip()
        }
        eligible_candidates = [
            candidate
            for candidate in candidates
            if " ".join(candidate.content.split()).casefold() not in exclusions
        ]
        if not eligible_candidates:
            return DeleteDecision(
                selected_ids=[],
                reasoning="No candidates remained after applying query-plan exclusions.",
                ambiguous=False,
            )

        selection_config = self.config.top_k_llm
        # Short aliases: the judge mistypes 32-char hex record IDs.
        alias_to_id = {
            f"c{index}": candidate.memory_id
            for index, candidate in enumerate(eligible_candidates, start=1)
        }
        candidate_data = [
            {"id": alias, "content": candidate.content}
            for alias, candidate in zip(alias_to_id, eligible_candidates)
        ]
        messages: list[dict[str, Any] | ChatCompletionMessage] = [
            {
                "role": "system",
                "content": selection_config.system_prompt,
            },
            {
                "role": "user",
                "content": selection_config.user_prompt_template.format(
                    original_query=query_plan.original_query,
                    query_plan=query_plan.model_dump_json(exclude={"cardinality"}),
                    candidates=json.dumps(candidate_data, ensure_ascii=True),
                ),
            },
        ]

        try:
            decision = request_delete_decision(selection_config.llm_model, messages)
        except Exception:  # noqa: BLE001 - all provider failures must fail closed
            return self._fail_closed("The deletion judge failed; nothing was selected.")

        if not isinstance(decision, DeleteDecision):
            return self._fail_closed(
                "The deletion judge returned malformed output; nothing was selected."
            )
        if decision.ambiguous and decision.selected_ids:
            return self._fail_closed(
                "The deletion judge reported ambiguity; nothing was selected."
            )
        rank = {
            candidate.memory_id: index
            for index, candidate in enumerate(eligible_candidates)
        }
        resolved = [
            alias_to_id.get(selected, selected) for selected in decision.selected_ids
        ]
        selected_ids = sorted(
            {memory_id for memory_id in resolved if memory_id in rank},
            key=rank.__getitem__,
        )
        notes = [decision.reasoning] if decision.reasoning else []
        unknown_count = sum(memory_id not in rank for memory_id in resolved)
        if unknown_count:
            notes.append(f"Dropped {unknown_count} unknown candidate ID(s).")
        if len(selected_ids) > self.config.max_delete_count:
            notes.append(
                f"Trimmed {len(selected_ids)} selections to the deletion limit "
                f"of {self.config.max_delete_count}."
            )
            selected_ids = selected_ids[: self.config.max_delete_count]
        return decision.model_copy(
            update={"selected_ids": selected_ids, "reasoning": " ".join(notes)}
        )
