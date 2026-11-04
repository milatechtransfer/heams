from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .config import PrivacyConfig, RecallMode
from .models import RecallDecision
from .recall_policy_deterministic import DeterministicRecallEvaluator
from .recall_policy_hybrid import HybridRecallEvaluator
from .recall_policy_llm import LLMRecallEvaluator


@runtime_checkable
class RecallPolicy(Protocol):
    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, object] | None = None,
        *,
        query: str = "",
    ) -> RecallDecision: ...


class NoOpRecallPolicy:
    def __init__(self, *, reason: str):
        self._reason = reason

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        query: str = "",
    ) -> RecallDecision:
        return RecallDecision(reason=self._reason)


def build_recall_policy(privacy_config: PrivacyConfig) -> RecallPolicy:
    if not privacy_config.enabled:
        return NoOpRecallPolicy(reason="privacy_disabled")
    if not privacy_config.recall.enabled:
        return NoOpRecallPolicy(reason="recall_disabled")
    if privacy_config.recall.mode is RecallMode.DETERMINISTIC:
        if not privacy_config.has_rules:
            return NoOpRecallPolicy(reason="no_recall_rules_given")
        return DeterministicRecallEvaluator(privacy_config)
    if privacy_config.recall.mode is RecallMode.LLM:
        return LLMRecallEvaluator(privacy_config)
    if privacy_config.recall.mode is RecallMode.HYBRID:
        return HybridRecallEvaluator(privacy_config)
    raise ValueError(
        "Unknown recall mode "
        f"{privacy_config.recall.mode.value!r}; no recall policy resolver is registered."
    )


__all__ = [
    "NoOpRecallPolicy",
    "RecallPolicy",
    "build_recall_policy",
]
