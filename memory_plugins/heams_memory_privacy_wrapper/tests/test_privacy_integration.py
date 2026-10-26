from pathlib import Path
from typing import Any

import pytest

from heams.memory_systems import (
    create_memory,
)
from heams_memory_privacy_wrapper import (
    MEMORY_RETENTION_ACTION_KEY,
    MEMORY_RETENTION_DECISION_SOURCE_KEY,
    PrivacyConfig,
    PrivacyMemoryWrapper,
    RecallConfig,
    RecallMode,
    RecallVerdict,
    RetentionMode,
    wrap_memory_with_privacy,
)
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams_memory_example.plugin import ExampleMemoryConfig
from heams_memory_privacy_wrapper.wrapper import _parse_rendered_recall_entry
from heams.memory_systems.shared_types import MemoryEntry, render_tool_memory_entry


class RecordingMemory(BaseMemorySystem):
    def __init__(self) -> None:
        super().__init__(
            BaseMemoryConfig(type="recording", name="recording"),
            storage_dir=Path(),
        )
        self.retain_calls: list[tuple[str, dict[str, Any]]] = []
        self.recall_calls: list[tuple[str, int]] = []
        self.delete_calls: list[str] = []
        self.consolidate_calls: list[tuple[str | None, dict[str, Any]]] = []
        self.primed_batches: list[list[MemoryEntry]] = []
        self.recall_results: list[str] = ["match"]
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        self.retain_calls.append((content, metadata))
        return MemoryOperationResult(value="retained")

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        self.recall_calls.append((query, top_k))
        return MemoryOperationResult(value=self.recall_results[:top_k])

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        self.delete_calls.append(query)
        return MemoryOperationResult(value="deleted")

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        self.consolidate_calls.append((context, metadata or {}))
        return MemoryOperationResult(value="consolidated")

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        self.primed_batches.append(entries)


def test_privacy_wrapper_delegates_public_operations() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(inner=inner, privacy_config=PrivacyConfig())
    entries = [MemoryEntry(content="fact 2", metadata={"source": "test"})]

    assert wrapper.retain("fact 1", {"kind": "user"}) == "retained"
    assert wrapper.recall("fact", top_k=2) == ["match"]
    assert wrapper.delete("fact") == "deleted"
    assert (
        wrapper.consolidate(context="startup", metadata={"phase": "init"})
        == "consolidated"
    )

    wrapper.prime_memory_entries(entries)

    assert inner.retain_calls == [("fact 1", {"kind": "user"})]
    assert inner.recall_calls == [("fact", 2)]
    assert inner.delete_calls == ["fact"]
    assert inner.consolidate_calls == [("startup", {"phase": "init"})]
    assert len(inner.primed_batches) == 1
    assert inner.primed_batches[0][0].content == "fact 2"


def test_privacy_wrapper_delegates_close() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(inner=inner, privacy_config=PrivacyConfig())

    wrapper.close()

    assert inner.close_calls == 1


def test_privacy_wrapper_propagates_event_sink_and_run_context() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(inner=inner, privacy_config=PrivacyConfig())
    ledger = []

    wrapper.attach_event_sink(ledger)
    wrapper.set_run_context({"scenario_id": "privacy-pass-through"})
    wrapper.retain("fact")

    assert len(ledger) == 1
    assert ledger[0].operation_type.value == "retain"
    assert ledger[0].payload["run_context"]["scenario_id"] == "privacy-pass-through"


def test_privacy_wrapper_drops_blocked_retain_and_annotates_stored_metadata() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[
                {
                    "rule_id": "block-secret",
                    "content_substrings": ["secret"],
                }
            ],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )

    blocked = wrapper.retain("secret launch code", {"source": "user"})
    allowed = wrapper.retain("project timeline", {"source": "user"})

    assert blocked == "Memory dropped by privacy retention policy."
    assert allowed == "retained"
    assert inner.retain_calls == [
        (
            "project timeline",
            {
                "source": "user",
                MEMORY_RETENTION_ACTION_KEY: "store",
                MEMORY_RETENTION_DECISION_SOURCE_KEY: "deterministic",
            },
        )
    ]


def test_privacy_wrapper_filters_priming_batch_and_preserves_loader_metadata() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[
                {
                    "rule_id": "block-secret",
                    "content_substrings": ["secret"],
                }
            ],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )
    entries = [
        MemoryEntry(
            content="Allowed fact",
            metadata={
                "source": "loader",
                "timestamp": "2024-01-01T00:00:00Z",
                "ingest_group_id": "g1",
                "ingest_order": 0,
            },
        ),
        MemoryEntry(
            content="secret launch code",
            metadata={
                "source": "loader",
                "timestamp": "2024-01-01T00:00:01Z",
                "ingest_group_id": "g1",
                "ingest_order": 1,
            },
        ),
        MemoryEntry(
            content="Allowed follow-up",
            metadata={
                "source": "loader",
                "timestamp": "2024-01-01T00:00:02Z",
                "ingest_group_id": "g1",
                "ingest_order": 2,
            },
        ),
    ]

    wrapper.prime_memory_entries(entries)

    assert len(inner.primed_batches) == 1
    admitted_entries = inner.primed_batches[0]
    assert [entry.content for entry in admitted_entries] == [
        "Allowed fact",
        "Allowed follow-up",
    ]
    assert admitted_entries[0].metadata["source"] == "loader"
    assert admitted_entries[0].metadata["timestamp"] == "2024-01-01T00:00:00Z"
    assert admitted_entries[0].metadata["ingest_group_id"] == "g1"
    assert admitted_entries[0].metadata["ingest_order"] == 0
    assert admitted_entries[0].metadata[MEMORY_RETENTION_ACTION_KEY] == "store"
    assert (
        admitted_entries[0].metadata[MEMORY_RETENTION_DECISION_SOURCE_KEY]
        == "deterministic"
    )


def test_privacy_wrapper_emits_priming_decision_events() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )
    ledger = []
    wrapper.attach_event_sink(ledger)
    wrapper.set_run_context({"scenario_id": "CI-train-0000"})

    wrapper.prime_memory_entries(
        [
            MemoryEntry(content="Allowed fact", metadata={"field_name": "Name"}),
            MemoryEntry(
                content="secret launch code", metadata={"field_name": "SecretNotes"}
            ),
        ]
    )

    assert len(ledger) == 2
    stored_event, dropped_event = ledger
    assert stored_event.operation_type.value == "retain"
    assert stored_event.status.value == "success"
    assert stored_event.payload["field_name"] == "Name"
    assert stored_event.result["action"] == "store"
    assert stored_event.payload["run_context"]["scenario_id"] == "CI-train-0000"

    assert dropped_event.status.value == "dropped"
    assert dropped_event.payload["field_name"] == "SecretNotes"
    assert dropped_event.result["action"] == "drop"
    assert dropped_event.result["policy_id"] == "block-secret"


def test_privacy_wrapper_emits_events_when_sink_is_only_attached_to_inner_before_wrapping() -> (
    None
):
    """Regression test: `create_memory` attaches the event sink/run context to
    `inner` (via its constructor) *before* wrapping, and never re-attaches
    afterward -- unlike the tests above, which call `wrapper.attach_event_sink`
    directly. The wrapper must still pick up `inner`'s sink at construction
    time, or events it logs on itself (like priming decisions) are silently
    dropped even though `inner`'s own events keep working."""
    inner = RecordingMemory()
    ledger: list[object] = []
    inner.attach_event_sink(ledger)
    inner.set_run_context({"scenario_id": "CI-train-0000"})

    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )

    wrapper.prime_memory_entries(
        [MemoryEntry(content="secret launch code", metadata={"field_name": "Notes"})]
    )

    assert len(ledger) == 1
    assert ledger[0].component == "privacy_retention"
    assert ledger[0].result["action"] == "drop"
    assert ledger[0].payload["run_context"]["scenario_id"] == "CI-train-0000"


def test_privacy_wrapper_bind_run_context_reaches_self_owned_events_too() -> None:
    """Regression test: bind_run_context/reset_run_context used to delegate
    only to `inner`, so an event the wrapper logs on *itself* (e.g. priming
    decisions) mid-turn missed the actively-bound context even though
    `inner`'s own events picked it up fine."""
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )
    ledger = []
    wrapper.attach_event_sink(ledger)

    token = wrapper.bind_run_context({"scenario_id": "bound-mid-turn"})
    try:
        wrapper.prime_memory_entries(
            [MemoryEntry(content="fact", metadata={"field_name": "Name"})]
        )
    finally:
        wrapper.reset_run_context(token)

    assert ledger[0].payload["run_context"]["scenario_id"] == "bound-mid-turn"

    ledger.clear()
    wrapper.prime_memory_entries(
        [MemoryEntry(content="fact2", metadata={"field_name": "Name2"})]
    )

    assert "scenario_id" not in ledger[0].payload.get("run_context", {})


def _recall_wrapper(
    inner: RecordingMemory, rules: object = None, **recall_kwargs: object
) -> PrivacyMemoryWrapper:
    return PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=rules or [],
            recall=RecallConfig(**recall_kwargs),
        ),
    )


def test_privacy_wrapper_recall_is_passthrough_when_recall_policy_inactive() -> None:
    inner = RecordingMemory()
    inner.recall_results = ["a", "b", "c"]
    wrapper = _recall_wrapper(inner)  # default RecallConfig: no rules -> NoOp

    results = wrapper.recall(query="q", top_k=2)

    assert results == ["a", "b"]
    assert inner.recall_calls == [("q", 2)]  # no window widening


def test_privacy_wrapper_recall_overfetches_window_filters_and_backfills() -> None:
    inner = RecordingMemory()
    inner.recall_results = [
        render_tool_memory_entry(
            MemoryEntry(content="secret one", metadata={"field_name": "a"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "b"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="secret two", metadata={"field_name": "c"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed two", metadata={"field_name": "d"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed three", metadata={"field_name": "e"})
        ),
    ]
    wrapper = _recall_wrapper(
        inner,
        mode=RecallMode.DETERMINISTIC,
        window_multiplier=4,
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
    )

    results = wrapper.recall(query="q", top_k=2)

    # window = top_k(2) * multiplier(4) = 8, clamped by available results (5)
    assert inner.recall_calls == [("q", 8)]
    # two "secret" entries filtered; backfills to the next allowed candidates
    assert results == [
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "b"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed two", metadata={"field_name": "d"})
        ),
    ]


def test_privacy_wrapper_recall_stops_evaluating_once_top_k_survivors_found() -> None:
    inner = RecordingMemory()
    inner.recall_results = [
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "a"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed two", metadata={"field_name": "b"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed three", metadata={"field_name": "c"})
        ),
    ]
    wrapper = _recall_wrapper(
        inner,
        mode=RecallMode.DETERMINISTIC,
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
    )
    ledger = []
    wrapper.attach_event_sink(ledger)

    results = wrapper.recall(query="q", top_k=2)

    assert results[:2] == [
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "a"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed two", metadata={"field_name": "b"})
        ),
    ]
    # third candidate never needed evaluating once 2 survivors were found;
    # ledger has the inner backend's own recall event plus 2 privacy_recall
    # decision events, not 3 -- the third candidate is never looked at.
    recall_decision_events = [e for e in ledger if e.component == "privacy_recall"]
    assert len(recall_decision_events) == 2


def test_privacy_wrapper_recall_emits_audit_events_for_each_candidate() -> None:
    inner = RecordingMemory()
    inner.recall_results = [
        render_tool_memory_entry(
            MemoryEntry(content="secret one", metadata={"field_name": "a"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "b"})
        ),
    ]
    wrapper = _recall_wrapper(
        inner,
        mode=RecallMode.DETERMINISTIC,
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
    )
    ledger = []
    wrapper.attach_event_sink(ledger)

    wrapper.recall(query="my task", top_k=2)

    recall_decision_events = [e for e in ledger if e.component == "privacy_recall"]
    assert len(recall_decision_events) == 2
    withheld_event, exposed_event = recall_decision_events
    assert withheld_event.component == "privacy_recall"
    assert withheld_event.status.value == "skipped"
    assert withheld_event.result["action"] == "withhold"
    assert withheld_event.result["policy_id"] == "block-secret"
    assert withheld_event.payload["field_name"] == "a"
    assert withheld_event.payload["query"] == "my task"

    assert exposed_event.status.value == "success"
    assert exposed_event.result["action"] == "expose"
    assert exposed_event.payload["field_name"] == "b"


def test_privacy_wrapper_recall_returns_empty_for_top_k_zero_without_calling_inner() -> (
    None
):
    inner = RecordingMemory()
    inner.recall_results = ["a", "b", "c"]
    wrapper = _recall_wrapper(
        inner,
        mode=RecallMode.DETERMINISTIC,
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
    )

    results = wrapper.recall(query="q", top_k=0)

    assert results == []
    # regression: top_k=0 must not fall through to the backend's
    # falsy-top_k-means-unlimited convention (would fetch the whole pool)
    assert inner.recall_calls == []


def test_privacy_wrapper_recall_returns_fewer_than_top_k_when_window_exhausted() -> (
    None
):
    inner = RecordingMemory()
    inner.recall_results = [
        render_tool_memory_entry(
            MemoryEntry(content="secret one", metadata={"field_name": "a"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="secret two", metadata={"field_name": "b"})
        ),
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "c"})
        ),
    ]
    wrapper = _recall_wrapper(
        inner,
        mode=RecallMode.DETERMINISTIC,
        window_multiplier=1,  # window = top_k, so only 3 candidates are ever fetched
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
    )

    results = wrapper.recall(query="q", top_k=3)

    # only 1 of the 3 fetched candidates survives; no bigger window is
    # requested to backfill further, so fewer than top_k comes back
    assert results == [
        render_tool_memory_entry(
            MemoryEntry(content="allowed one", metadata={"field_name": "c"})
        )
    ]


def test_privacy_wrapper_recall_does_not_share_cache_across_wrapper_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model, content, field_name, query, system_prompt, matched_category=None
    ):
        calls.append(content)
        return RecallVerdict(action="expose", reasoning="")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )

    for _ in range(2):
        inner = RecordingMemory()
        inner.recall_results = ["same content"]
        wrapper = _recall_wrapper(inner, mode=RecallMode.HYBRID)
        wrapper.recall(query="same task", top_k=1)

    # each wrapper/evaluator is a fresh instance -- the second one must not
    # reuse the first's cached verdict
    assert calls == ["same content", "same content"]


class TestParseRenderedRecallEntry:
    def test_bare_content_string_has_no_metadata(self) -> None:
        assert _parse_rendered_recall_entry("plain text") == ("plain text", {})

    def test_wrapped_json_recovers_content_and_metadata(self) -> None:
        rendered = render_tool_memory_entry(
            MemoryEntry(content="fact", metadata={"field_name": "x"})
        )
        assert _parse_rendered_recall_entry(rendered) == ("fact", {"field_name": "x"})

    def test_valid_json_that_is_not_an_object_falls_back_to_raw_content(self) -> None:
        for rendered in ("null", "123", "[1, 2, 3]", '"just a string"'):
            assert _parse_rendered_recall_entry(rendered) == (rendered, {})

    def test_dict_missing_required_keys_falls_back_to_raw_content(self) -> None:
        # has "content" but not exactly {"content", "metadata"} -- treated
        # as ordinary (unparseable-as-wrapped) content, not spoofed metadata
        rendered = '{"content": "x"}'
        assert _parse_rendered_recall_entry(rendered) == (rendered, {})

    def test_extra_keys_prevent_misparsing_as_wrapped_entry(self) -> None:
        rendered = '{"content": "x", "metadata": {}, "extra": 1}'
        assert _parse_rendered_recall_entry(rendered) == (rendered, {})


def test_direct_retention_adherence_differs_only_when_privacy_is_active(
    tmp_path,
) -> None:
    baseline = create_memory(
        ExampleMemoryConfig(top_k=10),
        storage_dir=tmp_path / "baseline",
    )
    privacy_enabled = create_memory(
        ExampleMemoryConfig(top_k=10),
        storage_dir=tmp_path / "privacy",
        wrapper_configs={
            "privacy_config": PrivacyConfig(
                rules=[
                    {
                        "rule_id": "block-secret",
                        "content_substrings": ["secret"],
                    }
                ],
                retention={"mode": RetentionMode.DETERMINISTIC},
            )
        },
    )
    entries = [
        MemoryEntry(content="Allowed fact about tea", metadata={"source": "loader"}),
        MemoryEntry(
            content="Secret launch code is 1234", metadata={"source": "loader"}
        ),
    ]

    baseline.prime_memory_entries(entries)
    privacy_enabled.prime_memory_entries(entries)

    assert baseline.recall("") == [
        '{"content": "Allowed fact about tea", "metadata": {"source": "loader"}}',
        '{"content": "Secret launch code is 1234", "metadata": {"source": "loader"}}',
    ]
    assert privacy_enabled.recall("") == [
        '{"content": "Allowed fact about tea", "metadata": {"memory_retention_action": "store", "memory_retention_decision_source": "deterministic", "source": "loader"}}'
    ]


def test_privacy_wrapped_backend_skips_retention_without_rules(
    tmp_path,
) -> None:
    memory = create_memory(
        ExampleMemoryConfig(top_k=5),
        storage_dir=tmp_path,
        wrapper_configs={"privacy_config": PrivacyConfig()},
    )

    memory.retain("alpha")
    memory.prime_memory_entries(
        [
            MemoryEntry(content="beta"),
            MemoryEntry(content="gamma"),
        ]
    )

    assert memory.recall("") == ["alpha", "beta", "gamma"]


def test_privacy_wrapper_skips_retention_when_rules_are_empty() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )

    wrapper.retain("secret launch code", {"source": "user"})
    wrapper.prime_memory_entries([MemoryEntry(content="secret launch code")])

    assert inner.retain_calls == [("secret launch code", {"source": "user"})]
    assert len(inner.primed_batches) == 1
    assert [entry.content for entry in inner.primed_batches[0]] == [
        "secret launch code"
    ]


def test_wrap_memory_with_privacy_is_no_op_when_privacy_is_disabled(tmp_path) -> None:
    memory = create_memory(
        ExampleMemoryConfig(name="plain"),
        storage_dir=tmp_path,
    )

    wrapped = wrap_memory_with_privacy(memory, PrivacyConfig(enabled=False))

    assert wrapped is memory


def test_wrap_memory_with_privacy_can_disable_existing_wrapper(tmp_path) -> None:
    memory = create_memory(
        ExampleMemoryConfig(top_k=5),
        storage_dir=tmp_path,
        wrapper_configs={"privacy_config": PrivacyConfig()},
    )

    unwrapped = wrap_memory_with_privacy(memory, PrivacyConfig(enabled=False))

    assert not isinstance(unwrapped, PrivacyMemoryWrapper)
    unwrapped.retain("alpha")
    assert unwrapped.recall("") == ["alpha"]


def test_wrap_memory_with_privacy_can_update_existing_wrapper_config() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(inner=inner, privacy_config=PrivacyConfig())
    updated_config = PrivacyConfig(
        retention={"mode": RetentionMode.DETERMINISTIC},
    )

    result = wrap_memory_with_privacy(wrapper, updated_config)

    assert result is wrapper
    assert wrapper.privacy_config == updated_config


def test_wrap_memory_with_privacy_updates_deterministic_policy_in_place() -> None:
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "old", "content_substrings": ["old"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )

    wrap_memory_with_privacy(
        wrapper,
        PrivacyConfig(
            rules=[{"rule_id": "new", "content_substrings": ["fresh"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
        ),
    )
    wrapper.retain("fresh secret")

    assert inner.retain_calls == []


def test_wrap_memory_with_privacy_preserves_previous_policy_when_update_fails() -> None:
    inner = RecordingMemory()
    original_config = PrivacyConfig(
        rules=[{"rule_id": "block-secret", "content_substrings": ["secret"]}],
        retention={"mode": RetentionMode.DETERMINISTIC},
    )
    wrapper = PrivacyMemoryWrapper(inner=inner, privacy_config=original_config)
    unsupported_config = PrivacyConfig(
        rules=[{"rule_id": "transform-secret", "content_substrings": ["secret"]}],
        retention={"mode": RetentionMode.TRANSFORM},
    )

    with pytest.raises(
        ValueError, match="Retention mode 'transform' is not implemented yet"
    ):
        wrap_memory_with_privacy(wrapper, unsupported_config)

    assert wrapper.privacy_config == original_config
    wrapper.retain("secret launch code")
    assert inner.retain_calls == []


PET_POLICY = "Do not retain the user's pet information."
WORK_POLICY = "Do not retain the user's employment information."


def _policy_wrapper(
    inner: BaseMemorySystem, mode: RetentionMode = RetentionMode.DETERMINISTIC
) -> PrivacyMemoryWrapper:
    return PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "baseline-secret", "content_substrings": ["secret"]}],
            retention={
                "mode": mode,
                "policy_rules": {
                    PET_POLICY: [
                        {"rule_id": "drop-pet", "content_substrings": ["dog"]}
                    ],
                    WORK_POLICY: [
                        {"rule_id": "drop-work", "content_substrings": ["employer"]}
                    ],
                },
            },
            recall={"enabled": False},
        ),
    )


def test_base_memory_system_runs_ungoverned_under_a_retention_policy() -> None:
    ledger: list = []
    memory = RecordingMemory()
    memory.attach_event_sink(ledger)

    summary = memory.set_retention_policy([PET_POLICY])
    memory.retain("My dog is Rex.")

    assert "No retention policy enforced" in summary
    assert [content for content, _ in memory.retain_calls] == ["My dog is Rex."]
    [event, _] = ledger
    assert event.operation_type == "set_retention_policy"
    assert event.status == "skipped"
    assert event.result["enforced"] is False


def test_set_retention_policy_enforces_compiled_rules_on_top_of_baseline() -> None:
    inner = RecordingMemory()
    wrapper = _policy_wrapper(inner)

    assert wrapper.set_retention_policy([PET_POLICY]) == (
        f"Retention policies set: {[PET_POLICY]}"
    )
    wrapper.retain("My dog is Rex.")
    wrapper.retain("A secret code.")
    wrapper.retain("My employer is Acme.")

    assert [content for content, _ in inner.retain_calls] == ["My employer is Acme."]
    assert wrapper.privacy_config.retention.policies == [PET_POLICY]


def test_set_retention_policy_replaces_rather_than_stacks() -> None:
    inner = RecordingMemory()
    wrapper = _policy_wrapper(inner)

    wrapper.set_retention_policy([PET_POLICY])
    wrapper.set_retention_policy([WORK_POLICY])
    wrapper.retain("My dog is Rex.")
    wrapper.retain("My employer is Acme.")

    assert [content for content, _ in inner.retain_calls] == ["My dog is Rex."]
    assert [rule.rule_id for rule in wrapper.privacy_config.rules] == [
        "baseline-secret",
        "drop-work",
    ]


def test_set_retention_policy_rejects_statement_without_compiled_rules() -> None:
    inner = RecordingMemory()
    wrapper = _policy_wrapper(inner)
    wrapper.set_retention_policy([PET_POLICY])

    with pytest.raises(ValueError, match="no rules for"):
        wrapper.set_retention_policy(["Do not retain the user's travel plans."])

    assert wrapper.privacy_config.retention.policies == [PET_POLICY]
    wrapper.retain("My dog is Rex.")
    assert inner.retain_calls == []


def test_set_retention_policy_llm_mode_passes_statement_without_rules() -> None:
    wrapper = _policy_wrapper(RecordingMemory(), mode=RetentionMode.LLM)

    wrapper.set_retention_policy(["Do not retain the user's travel plans."])

    assert wrapper.privacy_config.retention.policies == [
        "Do not retain the user's travel plans."
    ]
    assert [rule.rule_id for rule in wrapper.privacy_config.rules] == [
        "baseline-secret"
    ]


def test_update_privacy_config_resets_the_runtime_policy_baseline() -> None:
    inner = RecordingMemory()
    wrapper = _policy_wrapper(inner)
    wrapper.set_retention_policy([PET_POLICY])

    wrapper.update_privacy_config(
        PrivacyConfig(
            retention={
                "mode": RetentionMode.DETERMINISTIC,
                "policy_rules": {
                    WORK_POLICY: [
                        {"rule_id": "drop-work", "content_substrings": ["employer"]}
                    ]
                },
            },
            recall={"enabled": False},
        )
    )
    wrapper.set_retention_policy([WORK_POLICY])
    wrapper.retain("A secret code.")

    assert [content for content, _ in inner.retain_calls] == ["A secret code."]


def test_set_retention_policy_without_policy_rules_enforces_only_baseline() -> None:
    """No table means the layer compiles no policies: the ungoverned baseline."""
    inner = RecordingMemory()
    wrapper = PrivacyMemoryWrapper(
        inner=inner,
        privacy_config=PrivacyConfig(
            rules=[{"rule_id": "baseline-secret", "content_substrings": ["secret"]}],
            retention={"mode": RetentionMode.DETERMINISTIC},
            recall={"enabled": False},
        ),
    )

    wrapper.set_retention_policy([PET_POLICY])
    wrapper.retain("My dog is Rex.")
    wrapper.retain("A secret code.")

    assert [content for content, _ in inner.retain_calls] == ["My dog is Rex."]
    assert wrapper.privacy_config.retention.policies == [PET_POLICY]
