"""Pure LLM retention: no deterministic rule pass at all -- every entry
goes straight to the judge, unlike hybrid where a rule match changes the
question asked. The shared `PrivacyConfig.rules` is still read (if
present) purely to build the judge's taxonomy via `category_description`
-- the same source hybrid and deterministic modes use -- but no matcher on
those rules is ever evaluated in this mode.

Decoupled from `DeterministicRetentionEvaluator` on purpose: pick this mode
to run the LLM judge on its own, with no dependency on a rule pass.
"""

from __future__ import annotations

from typing import Any

from heams.memory_systems.shared_types import MemoryEntry

from .config import PrivacyConfig
from .models import RetentionAction, RetentionBatchResult, RetentionDecision
from .retention_policy_hybrid import build_judge_system_prompt, judge_and_decide


class LLMRetentionEvaluator:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy_config = privacy_config
        self._judge_system_prompt = build_judge_system_prompt(
            privacy_config.rules,
            privacy_config.retention.policies,
        )

    def evaluate_entry(self, entry: MemoryEntry) -> RetentionDecision:
        return self.evaluate_content(entry.content, entry.metadata)

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> RetentionDecision:
        resolved_metadata = metadata or {}
        return judge_and_decide(
            self.privacy_config.retention.llm_model,
            content,
            resolved_metadata.get("field_name"),
            self._judge_system_prompt,
            decision_source="llm",
        )

    def evaluate_entries(self, entries: list[MemoryEntry]) -> RetentionBatchResult:
        decisions: list[RetentionDecision] = []
        stored_indices: list[int] = []
        dropped_indices: list[int] = []

        for index, entry in enumerate(entries):
            decision = self.evaluate_entry(entry)
            decisions.append(decision)
            if decision.action is RetentionAction.STORE:
                stored_indices.append(index)
            else:
                dropped_indices.append(index)

        return RetentionBatchResult(
            decisions=decisions,
            stored_indices=stored_indices,
            dropped_indices=dropped_indices,
        )

    def apply_stored_metadata(
        self,
        entry: MemoryEntry,
        decision: RetentionDecision,
    ) -> MemoryEntry:
        merged_metadata = dict(entry.metadata or {})
        if decision.action is RetentionAction.STORE:
            merged_metadata.update(decision.metadata)
        return MemoryEntry(
            content=entry.content,
            metadata=merged_metadata,
            score=entry.score,
        )

    def filter_entries(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], RetentionBatchResult]:
        batch_result = self.evaluate_entries(entries)
        stored_entries: list[MemoryEntry] = []

        for entry, decision in zip(entries, batch_result.decisions, strict=False):
            if decision.action is RetentionAction.STORE:
                stored_entries.append(self.apply_stored_metadata(entry, decision))

        return stored_entries, batch_result


__all__ = ["LLMRetentionEvaluator"]
