"""Deterministic recall-time filtering: the same rule taxonomy retention
uses (`DeterministicRetentionRule`, matched via `find_matching_rule`), just
applied to a candidate about to be *exposed* instead of *stored*.
"""

from __future__ import annotations

from typing import Any

from .config import PrivacyConfig
from .models import RecallAction, RecallDecision
from .retention_policy_deterministic import find_matching_rule


class DeterministicRecallEvaluator:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy_config = privacy_config

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        query: str = "",
    ) -> RecallDecision:
        resolved_metadata = metadata or {}
        matched_rule = find_matching_rule(
            self.privacy_config.rules, content, resolved_metadata
        )
        if matched_rule is not None:
            return RecallDecision(
                action=RecallAction.WITHHOLD,
                reason="matched_deterministic_rule",
                policy_id=matched_rule.rule_id,
                decision_source="deterministic",
            )
        return RecallDecision(
            action=RecallAction.EXPOSE,
            reason="default_expose",
            decision_source="deterministic",
        )


__all__ = ["DeterministicRecallEvaluator"]
