from __future__ import annotations

from typing import Any

from heams.memory_systems.shared_types import MemoryEntry

from .config import DeterministicRetentionRule, PrivacyConfig, RetentionMode
from .models import RetentionAction, RetentionBatchResult, RetentionDecision


MEMORY_RETENTION_ACTION_KEY = "memory_retention_action"
MEMORY_RETENTION_DECISION_SOURCE_KEY = "memory_retention_decision_source"
MEMORY_RETENTION_POLICY_ID_KEY = "memory_retention_policy_id"


def find_matching_rule(
    rules: list[DeterministicRetentionRule],
    content: str,
    metadata: dict[str, Any],
) -> DeterministicRetentionRule | None:
    """First rule (in order) whose matchers all hit; shared by the
    deterministic evaluator and the hybrid evaluator's rule pass."""

    normalized_content = content.casefold()
    for rule in rules:
        if _rule_matches(rule, normalized_content, metadata):
            return rule
    return None


def _rule_matches(
    rule: DeterministicRetentionRule,
    normalized_content: str,
    metadata: dict[str, Any],
) -> bool:
    if rule.content_substrings and not any(
        substring.casefold() in normalized_content
        for substring in rule.content_substrings
    ):
        return False

    for key, value in rule.metadata_equals.items():
        if metadata.get(key) != value:
            return False

    for key, value in rule.metadata_contains.items():
        candidate = metadata.get(key)
        if candidate is None or value.casefold() not in str(candidate).casefold():
            return False

    return True


def framework_metadata(decision: RetentionDecision) -> dict[str, str]:
    metadata = {
        MEMORY_RETENTION_ACTION_KEY: decision.action.value,
        MEMORY_RETENTION_DECISION_SOURCE_KEY: decision.decision_source
        or "deterministic",
    }
    if decision.policy_id:
        metadata[MEMORY_RETENTION_POLICY_ID_KEY] = decision.policy_id
    return metadata


class DeterministicRetentionEvaluator:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy_config = privacy_config

    def evaluate_entry(self, entry: MemoryEntry) -> RetentionDecision:
        return self.evaluate_content(entry.content, entry.metadata)

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> RetentionDecision:
        skip_reason = self._deterministic_skip_reason()
        if skip_reason is not None:
            return self._skip_retention_decision(skip_reason)

        resolved_metadata = metadata or {}
        matched_rule = find_matching_rule(
            self.privacy_config.rules, content, resolved_metadata
        )
        if matched_rule is not None:
            return self._decision_for_rule(matched_rule)
        return self._store_decision()

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

    def _decision_for_rule(
        self,
        rule: DeterministicRetentionRule,
    ) -> RetentionDecision:
        return RetentionDecision(
            action=RetentionAction.DROP,
            reason="matched_deterministic_rule",
            policy_id=rule.rule_id,
            decision_source="deterministic",
        )

    @property
    def _uses_deterministic_retention(self) -> bool:
        return self._deterministic_skip_reason() is None

    def _deterministic_skip_reason(self) -> str | None:
        if not self.privacy_config.enabled:
            return "privacy_disabled"
        if not self.privacy_config.retention.enabled:
            return "retention_disabled"
        if self.privacy_config.retention.mode is not RetentionMode.DETERMINISTIC:
            return "unsupported_retention_mode"
        if not self.privacy_config.has_rules:
            return "no_retention_rules_given"
        return None

    @staticmethod
    def _skip_retention_decision(reason: str) -> RetentionDecision:
        return RetentionDecision(
            action=RetentionAction.STORE,
            reason=reason,
        )

    def _store_decision(self) -> RetentionDecision:
        decision = RetentionDecision(
            action=RetentionAction.STORE,
            reason="default_store",
            decision_source="deterministic",
        )
        decision.metadata = framework_metadata(decision)
        return decision
