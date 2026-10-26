from __future__ import annotations
from typing import Any

from heams.memory_systems.api import MemoryEntry

from .config import PrivacyConfig
from .models import RecallDecision, RetentionBatchResult, RetentionDecision
from .recall import NoOpRecallPolicy, build_recall_policy
from .retention import build_retention_policy


class PrivacyPolicyEngine:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy_config = privacy_config
        self.retention_policy = build_retention_policy(privacy_config)
        self.recall_policy = build_recall_policy(privacy_config)

    def evaluate_retain_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> RetentionDecision:
        return self.retention_policy.evaluate_content(content, metadata)

    def filter_priming_entries(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], RetentionBatchResult]:
        return self.retention_policy.filter_entries(entries)

    def evaluate_recall_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        query: str = "",
    ) -> RecallDecision:
        return self.recall_policy.evaluate_content(content, metadata, query=query)

    @property
    def recall_policy_is_active(self) -> bool:
        return not isinstance(self.recall_policy, NoOpRecallPolicy)

    @property
    def recall_window_multiplier(self) -> int:
        return self.privacy_config.recall.window_multiplier
