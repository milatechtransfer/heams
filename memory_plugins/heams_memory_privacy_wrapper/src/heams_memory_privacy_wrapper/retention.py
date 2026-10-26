from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from heams.memory_systems.api import MemoryEntry

from .config import PrivacyConfig, RetentionMode
from .models import RetentionBatchResult, RetentionDecision
from .retention_policy_deterministic import DeterministicRetentionEvaluator
from .retention_policy_hybrid import HybridRetentionEvaluator
from .retention_policy_llm import LLMRetentionEvaluator


@runtime_checkable
class RetentionPolicy(Protocol):
    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> RetentionDecision: ...

    def evaluate_entries(self, entries: list[MemoryEntry]) -> RetentionBatchResult: ...

    def filter_entries(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], RetentionBatchResult]: ...


class NoOpRetentionPolicy:
    def __init__(self, *, reason: str):
        self._reason = reason

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> RetentionDecision:
        return RetentionDecision(reason=self._reason)

    def evaluate_entries(self, entries: list[MemoryEntry]) -> RetentionBatchResult:
        decisions = [
            self.evaluate_content(entry.content, entry.metadata) for entry in entries
        ]
        return RetentionBatchResult(
            decisions=decisions,
            stored_indices=list(range(len(entries))),
            dropped_indices=[],
        )

    def filter_entries(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], RetentionBatchResult]:
        return list(entries), self.evaluate_entries(entries)


def build_retention_policy(privacy_config: PrivacyConfig) -> RetentionPolicy:
    if not privacy_config.enabled:
        return NoOpRetentionPolicy(reason="privacy_disabled")
    if not privacy_config.retention.enabled:
        return NoOpRetentionPolicy(reason="retention_disabled")
    if privacy_config.retention.mode is RetentionMode.DETERMINISTIC:
        if not privacy_config.has_rules:
            return NoOpRetentionPolicy(reason="no_retention_rules_given")
        return DeterministicRetentionEvaluator(privacy_config)
    if privacy_config.retention.mode is RetentionMode.HYBRID:
        return HybridRetentionEvaluator(privacy_config)
    if privacy_config.retention.mode is RetentionMode.LLM:
        return LLMRetentionEvaluator(privacy_config)
    if privacy_config.retention.mode is RetentionMode.TRANSFORM:
        # TODO(common_privacy_evaluator): implement a dedicated retention
        # policy for transform mode behind this resolver.
        raise ValueError(
            "Retention mode "
            f"{privacy_config.retention.mode.value!r} is not implemented yet."
        )
    raise ValueError(
        "Unknown retention mode "
        f"{privacy_config.retention.mode.value!r}; no retention policy resolver is registered."
    )
