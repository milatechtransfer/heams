from __future__ import annotations

import json
from contextvars import Token
from typing import Any

from heams.memory_systems.shared_types import MemoryEntry
from heams.memory_systems.base import (
    BaseMemorySystem,
    MemoryEventTarget,
    MemoryOperationResult,
)

from .config import PrivacyConfig, RetentionMode
from .engine import PrivacyPolicyEngine
from .models import RecallAction, RetentionAction

RETENTION_DROPPED_MESSAGE = "Memory dropped by privacy retention policy."


def _parse_rendered_recall_entry(rendered: str) -> tuple[str, dict[str, Any]]:
    """Recover (content, metadata) from a recall result string.

    Backends render via `render_tool_memory_entry`, which emits a JSON
    object only when metadata is non-empty and a bare content string
    otherwise. This degrades gracefully instead of raising: unparsable or
    shapeless input is treated as content with no metadata.
    """
    try:
        parsed = json.loads(rendered)
    except (json.JSONDecodeError, TypeError):
        return rendered, {}
    if isinstance(parsed, dict) and set(parsed.keys()) == {"content", "metadata"}:
        raw_metadata = parsed.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        return str(parsed["content"]), metadata
    return rendered, {}


class PrivacyMemoryWrapper(BaseMemorySystem):
    def __init__(self, inner: BaseMemorySystem, privacy_config: PrivacyConfig):
        self.inner = inner
        self.update_privacy_config(privacy_config)
        # Without this, events this wrapper logs on itself are
        # silently dropped in the real orchestration path, even though the
        # inner backend's own events still reach the sink fine.
        super().__init__(
            inner.config,
            storage_dir=inner.storage_dir,
            event_sink=inner._memory_event_sink,
            run_context=inner._memory_event_run_context,
        )
        self.config = inner.config

    @property
    def privacy_config(self) -> PrivacyConfig:
        return self._privacy_config

    def update_privacy_config(self, privacy_config: PrivacyConfig) -> None:
        self._apply_privacy_config(privacy_config)
        # The configured baseline `set_retention_policy` derives from, so
        # successive calls replace the runtime policy rather than stack on it.
        self._base_privacy_config = privacy_config

    def _apply_privacy_config(self, privacy_config: PrivacyConfig) -> None:
        # Built before assigning so a config the engine rejects leaves the
        # previous policy in force.
        policy_engine = PrivacyPolicyEngine(privacy_config)
        self._privacy_config = privacy_config
        self._policy_engine = policy_engine

    def set_retention_policy(self, policies: list[str]) -> str:
        """Enforce `policies` on top of the configured baseline.

        The statements replace `retention.policies` (read by the llm/hybrid
        judge). In deterministic/hybrid modes each statement's
        `retention.policy_rules` entry is appended to the baseline `rules`;
        those rules are the shared ruleset, so an active recall layer applies
        them too. With a `policy_rules` table, a statement missing from it
        raises rather than being silently unenforced, which would score as a
        leak instead of surfacing the config gap. With no table at all the
        layer compiles no policies and enforces only its baseline rules --
        the ungoverned baseline the retention_policy benchmark measures.
        """
        base = self._base_privacy_config
        retention = base.retention
        rules = list(base.rules)
        if retention.policy_rules and retention.mode in {
            RetentionMode.DETERMINISTIC,
            RetentionMode.HYBRID,
        }:
            missing = [p for p in policies if p not in retention.policy_rules]
            if missing:
                raise ValueError(
                    f"retention.mode is {retention.mode.value!r} but "
                    f"retention.policy_rules has no rules for: {missing}."
                )
            for policy in policies:
                rules.extend(retention.policy_rules[policy])

        self._apply_privacy_config(
            base.model_copy(
                update={
                    "rules": rules,
                    "retention": retention.model_copy(
                        update={"policies": list(policies)}
                    ),
                }
            )
        )
        return f"Retention policies set: {list(policies)}"

    def attach_event_sink(self, event_sink: MemoryEventTarget | None) -> None:
        super().attach_event_sink(event_sink)
        self.inner.attach_event_sink(event_sink)

    def set_run_context(self, run_context: dict[str, Any] | None = None) -> None:
        super().set_run_context(run_context)
        self.inner.set_run_context(run_context)

    def bind_run_context(
        self, run_context: dict[str, Any] | None = None
    ) -> tuple[
        Token[dict[str, Any] | None] | None, Token[dict[str, Any] | None] | None
    ]:
        # Binding only `inner` meant any event this wrapper
        # logs on itself mid-turn missed the active run context,
        # even though `inner`'s own events picked it up fine.
        self_token = super().bind_run_context(run_context)
        inner_token = self.inner.bind_run_context(run_context)
        return (self_token, inner_token)

    def reset_run_context(
        self,
        token: tuple[
            Token[dict[str, Any] | None] | None, Token[dict[str, Any] | None] | None
        ],
    ) -> None:
        self_token, inner_token = token
        super().reset_run_context(self_token)
        self.inner.reset_run_context(inner_token)

    def close(self) -> None:
        self.inner.close()

    def retain(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        decision = self._policy_engine.evaluate_retain_content(content, metadata)
        if decision.action is RetentionAction.DROP:
            return RETENTION_DROPPED_MESSAGE

        retained_metadata = dict(metadata or {})
        retained_metadata.update(decision.metadata)
        return self.inner.retain(content=content, metadata=retained_metadata)

    def recall(
        self,
        query: str = "",
        top_k: int | None = None,
    ) -> list[str]:
        if not self._policy_engine.recall_policy_is_active:
            return self.inner.recall(query=query, top_k=top_k)

        resolved_top_k = top_k if top_k is not None else self.config.top_k
        if resolved_top_k <= 0:
            # Backends treat a falsy top_k as "no limit" (e.g.
            # `effective_top_k = top_k if top_k else len(entries)`), so
            # window = 0 would silently fetch the *entire* pool instead of
            # nothing -- exactly the unbounded cost this design exists to
            # avoid -- and the survivor loop below would return 1 result
            # instead of the 0 that was actually asked for. Short-circuit
            # before ever calling the inner backend.
            return []
        window = resolved_top_k * self._policy_engine.recall_window_multiplier
        candidates = self.inner.recall(query=query, top_k=window)

        survivors: list[str] = []
        for rendered in candidates:
            content, metadata = _parse_rendered_recall_entry(rendered)
            decision = self._policy_engine.evaluate_recall_content(
                content, metadata, query=query
            )
            self._log_memory_event(
                operation_type="recall",
                status="success"
                if decision.action is RecallAction.EXPOSE
                else "skipped",
                source="runtime",
                component="privacy_recall",
                payload={
                    "field_name": metadata.get("field_name"),
                    "content_length": len(content),
                    "query": query,
                },
                result={
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "policy_id": decision.policy_id,
                    "decision_source": decision.decision_source,
                },
            )
            if decision.action is RecallAction.EXPOSE:
                survivors.append(rendered)
                if len(survivors) >= resolved_top_k:
                    break

        return survivors

    def delete(self, query: str) -> str:
        return self.inner.delete(query)

    def consolidate(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.inner.consolidate(context=context, metadata=metadata)

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            self.inner.prime_memory_entries(entries)
            return

        admitted_entries, batch_result = self._policy_engine.filter_priming_entries(
            entries
        )
        for entry, decision in zip(entries, batch_result.decisions, strict=True):
            self._log_memory_event(
                operation_type="retain",
                status="success"
                if decision.action is RetentionAction.STORE
                else "skipped",
                source="runtime",
                component="privacy_retention",
                payload={
                    "field_name": (entry.metadata or {}).get("field_name"),
                    "content_length": len(entry.content),
                },
                result={
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "policy_id": decision.policy_id,
                    "decision_source": decision.decision_source,
                },
            )
        if admitted_entries:
            self.inner.prime_memory_entries(admitted_entries)

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        raise NotImplementedError(
            "PrivacyMemoryWrapper delegates retain through the wrapped backend."
        )

    def _recall_impl(
        self,
        query: str,
        top_k: int,
    ) -> MemoryOperationResult:
        raise NotImplementedError(
            "PrivacyMemoryWrapper delegates recall through the wrapped backend."
        )


def wrap_memory_with_privacy(
    memory: BaseMemorySystem,
    privacy_config: PrivacyConfig | None,
) -> BaseMemorySystem:
    if isinstance(memory, PrivacyMemoryWrapper):
        if privacy_config is None:
            return memory
        if not privacy_config.is_active:
            return memory.inner
        memory.update_privacy_config(privacy_config)
        return memory
    if privacy_config is None or not privacy_config.is_active:
        return memory
    return PrivacyMemoryWrapper(inner=memory, privacy_config=privacy_config)
