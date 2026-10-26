"""Pure LLM recall: no deterministic rule pass at all -- every candidate
goes straight to the judge, unlike hybrid where a rule match changes the
question asked.
"""

from __future__ import annotations

from typing import Any

from .config import PrivacyConfig
from .models import RecallDecision
from .recall_policy_hybrid import (
    RecallVerdict,
    build_judge_system_prompt,
    judge_and_decide,
)


class LLMRecallEvaluator:
    def __init__(self, privacy_config: PrivacyConfig):
        self.privacy_config = privacy_config
        self._judge_system_prompt = build_judge_system_prompt(
            privacy_config.rules,
            privacy_config.recall.policies,
        )
        # Same rationale as HybridRecallEvaluator's cache: keyed on
        # (content, field_name, query, matched_category) -- always None for
        # matched_category here since this mode never rule-matches --
        # scoped to this evaluator instance.
        self._llm_verdict_cache: dict[
            tuple[str, str | None, str, str | None], RecallVerdict
        ] = {}

    def evaluate_content(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        query: str = "",
    ) -> RecallDecision:
        resolved_metadata = metadata or {}
        return judge_and_decide(
            self.privacy_config.recall.llm_model,
            content,
            resolved_metadata.get("field_name"),
            query,
            self._judge_system_prompt,
            decision_source="llm",
            cache=self._llm_verdict_cache,
        )


__all__ = ["LLMRecallEvaluator"]
