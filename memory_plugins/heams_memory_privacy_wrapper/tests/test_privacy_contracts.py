import pytest
from pydantic import ValidationError

from heams_memory_privacy_wrapper import (
    MEMORY_RETENTION_ACTION_KEY,
    MEMORY_RETENTION_DECISION_SOURCE_KEY,
    MEMORY_RETENTION_POLICY_ID_KEY,
    DeterministicRecallEvaluator,
    DeterministicRetentionEvaluator,
    DeterministicRetentionRule,
    HybridRecallEvaluator,
    HybridRetentionEvaluator,
    LLMRecallEvaluator,
    LLMRetentionEvaluator,
    NoOpRecallPolicy,
    PrivacyConfig,
    PrivacyPolicyEngine,
    RecallAction,
    RecallConfig,
    RecallMode,
    RecallVerdict,
    RetentionAction,
    RetentionBatchResult,
    RetentionConfig,
    RetentionDecision,
    RetentionMode,
    RetentionVerdict,
    build_recall_policy,
    build_retention_policy,
)
from heams_memory_privacy_wrapper.recall_policy_hybrid import (
    build_judge_system_prompt as build_recall_judge_system_prompt,
)
from heams_memory_privacy_wrapper.retention_policy_hybrid import (
    build_judge_system_prompt,
)
from heams.memory_systems.shared_types import MemoryEntry


def test_deterministic_retention_rule_requires_at_least_one_matcher() -> None:
    with pytest.raises(ValidationError):
        DeterministicRetentionRule()


def test_deterministic_retention_evaluator_skips_when_privacy_disabled() -> None:
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(
            enabled=False,
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC),
        )
    )

    decision = evaluator.evaluate_content("Secret launch code", {"source": "user"})
    stored_entry = evaluator.apply_stored_metadata(
        MemoryEntry(content="Secret launch code", metadata={"source": "user"}),
        decision,
    )

    assert decision.action is RetentionAction.STORE
    assert decision.reason == "privacy_disabled"
    assert decision.decision_source is None
    assert decision.metadata == {}
    assert stored_entry.metadata == {"source": "user"}


def test_deterministic_retention_evaluator_skips_when_retention_disabled() -> None:
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(
                enabled=False,
                mode=RetentionMode.DETERMINISTIC,
            ),
        )
    )

    decision = evaluator.evaluate_content("Secret launch code", {"source": "user"})

    assert decision.action is RetentionAction.STORE
    assert decision.reason == "retention_disabled"
    assert decision.decision_source is None
    assert decision.metadata == {}


def test_deterministic_retention_evaluator_skips_when_mode_is_not_deterministic() -> (
    None
):
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("Secret launch code", {"source": "user"})

    assert decision.action is RetentionAction.STORE
    assert decision.reason == "unsupported_retention_mode"
    assert decision.decision_source is None
    assert decision.metadata == {}


def test_deterministic_retention_evaluator_skips_when_no_rules_are_given() -> None:
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC))
    )

    batch = evaluator.evaluate_entries(
        [
            MemoryEntry(content="Allowed fact", metadata={"source": "user"}),
            MemoryEntry(content="Secret launch code", metadata={"source": "user"}),
        ]
    )
    filtered_entries, batch_result = evaluator.filter_entries(
        [
            MemoryEntry(content="Allowed fact", metadata={"source": "user"}),
            MemoryEntry(content="Secret launch code", metadata={"source": "user"}),
        ]
    )

    assert batch.stored_indices == [0, 1]
    assert batch.dropped_indices == []
    assert [decision.reason for decision in batch.decisions] == [
        "no_retention_rules_given",
        "no_retention_rules_given",
    ]
    assert batch_result.dropped_indices == []
    assert [entry.metadata for entry in filtered_entries] == [
        {"source": "user"},
        {"source": "user"},
    ]


def test_deterministic_retention_evaluator_drops_matching_entries_and_tracks_indices() -> (
    None
):
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC),
        )
    )

    batch = evaluator.evaluate_entries(
        [
            MemoryEntry(content="Allowed fact", metadata={"source": "user"}),
            MemoryEntry(content="Secret launch code", metadata={"source": "user"}),
        ]
    )

    assert batch.stored_indices == [0]
    assert batch.dropped_indices == [1]
    assert batch.decisions[1].action is RetentionAction.DROP
    assert batch.decisions[1].policy_id == "block-secret"


def test_deterministic_retention_evaluator_adds_reserved_metadata_only_to_stored_entries() -> (
    None
):
    evaluator = DeterministicRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC),
        )
    )
    entry = MemoryEntry(
        content="Allowed fact",
        metadata={"channel": "safe", "source": "loader"},
    )

    decision = evaluator.evaluate_entry(entry)
    stored_entry = evaluator.apply_stored_metadata(entry, decision)

    assert decision.action is RetentionAction.STORE
    assert stored_entry.metadata["channel"] == "safe"
    assert stored_entry.metadata["source"] == "loader"
    assert stored_entry.metadata[MEMORY_RETENTION_ACTION_KEY] == "store"
    assert (
        stored_entry.metadata[MEMORY_RETENTION_DECISION_SOURCE_KEY] == "deterministic"
    )
    assert MEMORY_RETENTION_POLICY_ID_KEY not in stored_entry.metadata


def test_privacy_config_defaults_to_enabled_with_no_retention_rules() -> None:
    config = PrivacyConfig()

    assert config.enabled is True
    assert config.is_active is True
    assert config.retention.enabled is True
    assert config.retention.mode is RetentionMode.DETERMINISTIC
    assert config.rules == []
    assert config.has_rules is False


def test_build_retention_policy_resolves_deterministic_policy_when_rules_exist() -> (
    None
):
    policy = build_retention_policy(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC),
        )
    )

    assert isinstance(policy, DeterministicRetentionEvaluator)


@pytest.mark.parametrize(
    "mode",
    [RetentionMode.TRANSFORM],
)
def test_build_retention_policy_rejects_unimplemented_modes(
    mode: RetentionMode,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"Retention mode '{mode.value}' is not implemented yet",
    ):
        build_retention_policy(
            PrivacyConfig(
                rules=[
                    DeterministicRetentionRule(
                        rule_id="block-secret",
                        content_substrings=["secret"],
                    )
                ],
                retention=RetentionConfig(mode=mode),
            )
        )


def test_build_retention_policy_resolves_hybrid_policy() -> None:
    policy = build_retention_policy(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    assert isinstance(policy, HybridRetentionEvaluator)


def test_build_retention_policy_resolves_llm_policy() -> None:
    policy = build_retention_policy(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.LLM))
    )

    assert isinstance(policy, LLMRetentionEvaluator)


def test_hybrid_retention_evaluator_drops_when_llm_verifies_rule_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        calls.append(matched_category)
        return RetentionVerdict(action="drop", reasoning="confirmed sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    category_description="launch codes",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("secret launch code")

    assert decision.action is RetentionAction.DROP
    assert decision.policy_id == "block-secret"
    assert decision.decision_source == "hybrid_rule_verified"
    assert decision.reason == "confirmed sensitive"
    assert calls == ["launch codes"]


def test_hybrid_retention_evaluator_stores_when_llm_overturns_rule_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        return RetentionVerdict(
            action="store", reasoning="false-positive keyword match"
        )

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="drop-family_background-family",
                    category_description="family or relationship background",
                    metadata_contains={"field_name": "family"},
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content(
        "Runs a family-friendly restaurant downtown",
        metadata={"field_name": "family_business_notes"},
    )

    assert decision.action is RetentionAction.STORE
    assert decision.policy_id == "drop-family_background-family"
    assert decision.decision_source == "hybrid_rule_overturned"
    assert decision.reason == "false-positive keyword match"
    assert decision.metadata[MEMORY_RETENTION_ACTION_KEY] == "store"


def test_hybrid_retention_evaluator_fails_closed_to_rule_when_verifier_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RetentionVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _raise,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("secret launch code")

    assert decision.action is RetentionAction.DROP
    assert decision.policy_id == "block-secret"
    assert decision.decision_source == "hybrid_rule"
    assert decision.reason == "matched_deterministic_rule_llm_verifier_unavailable"


def test_hybrid_retention_evaluator_ambiguous_verdict_keeps_rule_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous on a rule match has an existing verdict to fall back to (the
    rule's own drop) -- unlike the no-rule-match path, it shouldn't overturn
    it on an unconfident verdict."""

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        return RetentionVerdict(action="ambiguous", reasoning="not sure either way")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("secret launch code")

    assert decision.action is RetentionAction.DROP
    assert decision.policy_id == "block-secret"
    assert decision.decision_source == "hybrid_rule"
    assert decision.reason == "not sure either way"


def test_hybrid_retention_evaluator_ambiguous_verdict_defaults_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No rule matched, so there's no prior verdict to fall back to -- an
    ambiguous judgment defaults to store rather than an unconfident drop."""

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        return RetentionVerdict(action="ambiguous", reasoning="unclear if sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.HYBRID))
    )

    decision = evaluator.evaluate_content("Mentioned a trip next month")

    assert decision.action is RetentionAction.STORE
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "unclear if sensitive"
    assert decision.metadata[MEMORY_RETENTION_ACTION_KEY] == "store"


def test_hybrid_retention_evaluator_falls_back_to_llm_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model: str, content: str, field_name: str | None, system_prompt: str
    ) -> RetentionVerdict:
        calls.append((model, content, field_name))
        return RetentionVerdict(action="drop", reasoning="looks like a medical record")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(
                mode=RetentionMode.HYBRID,
                llm_model="openai:gpt-4o-mini",
            ),
        )
    )

    decision = evaluator.evaluate_content(
        "Currently taking RX2019 for hypertension",
        metadata={"field_name": "MedicalNotes"},
    )

    assert decision.action is RetentionAction.DROP
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "looks like a medical record"
    assert calls == [
        (
            "openai:gpt-4o-mini",
            "Currently taking RX2019 for hypertension",
            "MedicalNotes",
        )
    ]


def test_hybrid_retention_evaluator_builds_judge_prompt_from_rule_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge's taxonomy list must come from `category_description` on the
    configured rules, not a hardcoded prompt -- add a category to config and
    the judge should see it without a code change."""
    captured_prompts = []

    def _fake_judge(
        model: str, content: str, field_name: str | None, system_prompt: str
    ) -> RetentionVerdict:
        captured_prompts.append(system_prompt)
        return RetentionVerdict(action="store", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="drop-medical-general",
                    category_description="medical or health information",
                    metadata_contains={"field_name": "medical"},
                ),
                DeterministicRetentionRule(
                    rule_id="drop-medical-prescription",
                    category_description="medical or health information",
                    metadata_contains={"field_name": "prescript"},
                ),
                DeterministicRetentionRule(
                    rule_id="drop-financial-general",
                    category_description="financial or payment information",
                    metadata_contains={"field_name": "financial"},
                ),
            ],
            retention=RetentionConfig(mode=RetentionMode.HYBRID),
        )
    )

    evaluator.evaluate_content("Prefers window seats")

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "- medical or health information" in prompt
    assert "- financial or payment information" in prompt
    # Same description repeated across sibling rules in a category appears once.
    assert prompt.count("medical or health information") == 1


def test_hybrid_retention_evaluator_builds_judge_prompt_from_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`policies` are free-text, user-declared instructions -- distinct
    from `rules`' category taxonomy -- and must reach the judge prompt too,
    as a separate section, without needing a matching rule at all."""
    captured_prompts = []

    def _fake_judge(
        model: str, content: str, field_name: str | None, system_prompt: str
    ) -> RetentionVerdict:
        captured_prompts.append(system_prompt)
        return RetentionVerdict(action="store", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(
            retention=RetentionConfig(
                mode=RetentionMode.HYBRID,
                policies=[
                    "do not record my medical info",
                    "don't store anything about my ex",
                ],
            )
        )
    )

    evaluator.evaluate_content("Some unrelated content")

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "- do not record my medical info" in prompt
    assert "- don't store anything about my ex" in prompt


def test_build_judge_system_prompt_omits_policies_section_when_empty() -> None:
    prompt = build_judge_system_prompt(rules=[], policies=[])

    assert "also explicitly asked" not in prompt


def test_hybrid_retention_evaluator_stores_when_llm_judge_says_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str, content: str, field_name: str | None, system_prompt: str
    ) -> RetentionVerdict:
        return RetentionVerdict(action="store", reasoning="ordinary preference")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.HYBRID))
    )

    decision = evaluator.evaluate_content("Prefers window seats")

    assert decision.action is RetentionAction.STORE
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "ordinary preference"
    assert decision.metadata[MEMORY_RETENTION_ACTION_KEY] == "store"


def test_hybrid_retention_evaluator_fails_closed_when_llm_judge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RetentionVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _raise,
    )
    evaluator = HybridRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.HYBRID))
    )

    decision = evaluator.evaluate_content("some unmatched content")

    assert decision.action is RetentionAction.DROP
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "llm_judge_unavailable_fail_closed"


def test_llm_retention_evaluator_never_evaluates_rule_matchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoupled from deterministic mode on purpose: even a rule whose
    matcher would obviously hit is never consulted -- everything goes
    straight to the judge."""

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        assert matched_category is None
        return RetentionVerdict(action="store", reasoning="not sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = LLMRetentionEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.LLM),
        )
    )

    decision = evaluator.evaluate_content("secret launch code")

    assert decision.action is RetentionAction.STORE
    assert decision.decision_source == "llm"


def test_llm_retention_evaluator_drops_when_judge_says_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        return RetentionVerdict(action="drop", reasoning="medical record")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = LLMRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.LLM))
    )

    decision = evaluator.evaluate_content(
        "Currently taking RX2019 for hypertension",
        metadata={"field_name": "MedicalNotes"},
    )

    assert decision.action is RetentionAction.DROP
    assert decision.decision_source == "llm"
    assert decision.reason == "medical record"


def test_llm_retention_evaluator_ambiguous_verdict_defaults_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RetentionVerdict:
        return RetentionVerdict(action="ambiguous", reasoning="unclear if sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = LLMRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.LLM))
    )

    decision = evaluator.evaluate_content("Mentioned a trip next month")

    assert decision.action is RetentionAction.STORE
    assert decision.decision_source == "llm"
    assert decision.reason == "unclear if sensitive"
    assert decision.metadata[MEMORY_RETENTION_ACTION_KEY] == "store"


def test_llm_retention_evaluator_fails_closed_when_judge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RetentionVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _raise,
    )
    evaluator = LLMRetentionEvaluator(
        PrivacyConfig(retention=RetentionConfig(mode=RetentionMode.LLM))
    )

    decision = evaluator.evaluate_content("some content")

    assert decision.action is RetentionAction.DROP
    assert decision.decision_source == "llm"
    assert decision.reason == "llm_judge_unavailable_fail_closed"


def test_llm_retention_evaluator_builds_judge_prompt_from_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompts = []

    def _fake_judge(
        model: str, content: str, field_name: str | None, system_prompt: str
    ) -> RetentionVerdict:
        captured_prompts.append(system_prompt)
        return RetentionVerdict(action="store", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.retention_policy_hybrid.judge_retention_content",
        _fake_judge,
    )
    evaluator = LLMRetentionEvaluator(
        PrivacyConfig(
            retention=RetentionConfig(
                mode=RetentionMode.LLM,
                policies=["do not record my medical info"],
            )
        )
    )

    evaluator.evaluate_content("Some unrelated content")

    assert len(captured_prompts) == 1
    assert "- do not record my medical info" in captured_prompts[0]


def test_build_recall_policy_defaults_to_noop_without_rules() -> None:
    policy = build_recall_policy(PrivacyConfig())

    assert isinstance(policy, NoOpRecallPolicy)
    decision = policy.evaluate_content("anything")
    assert decision.action is RecallAction.EXPOSE


def test_build_recall_policy_resolves_deterministic_policy_when_rules_exist() -> None:
    policy = build_recall_policy(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret", content_substrings=["secret"]
                )
            ],
            recall=RecallConfig(mode=RecallMode.DETERMINISTIC),
        )
    )

    assert isinstance(policy, DeterministicRecallEvaluator)


def test_deterministic_recall_evaluator_withholds_on_rule_match() -> None:
    evaluator = DeterministicRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret", content_substrings=["secret"]
                )
            ],
            recall=RecallConfig(mode=RecallMode.DETERMINISTIC),
        )
    )

    withheld = evaluator.evaluate_content("secret launch code")
    exposed = evaluator.evaluate_content("ordinary preference")

    assert withheld.action is RecallAction.WITHHOLD
    assert withheld.policy_id == "block-secret"
    assert exposed.action is RecallAction.EXPOSE


def test_build_recall_policy_resolves_hybrid_policy() -> None:
    policy = build_recall_policy(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    assert isinstance(policy, HybridRecallEvaluator)


def test_hybrid_recall_evaluator_withholds_when_llm_verifies_rule_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        calls.append(matched_category)
        return RecallVerdict(action="withhold", reasoning="confirmed sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    category_description="launch codes",
                    content_substrings=["secret"],
                )
            ],
            recall=RecallConfig(mode=RecallMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("secret launch code", query="book a room")

    assert decision.action is RecallAction.WITHHOLD
    assert decision.policy_id == "block-secret"
    assert decision.decision_source == "hybrid_rule_verified"
    assert decision.reason == "confirmed sensitive"
    assert calls == ["launch codes"]


def test_hybrid_recall_evaluator_exposes_when_llm_overturns_rule_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        return RecallVerdict(action="expose", reasoning="not relevant here")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="withhold-family_background-family",
                    category_description="family or relationship background",
                    metadata_contains={"field_name": "family"},
                )
            ],
            recall=RecallConfig(mode=RecallMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content(
        "Runs a family-friendly restaurant downtown",
        metadata={"field_name": "family_business_notes"},
        query="book a table",
    )

    assert decision.action is RecallAction.EXPOSE
    assert decision.policy_id == "withhold-family_background-family"
    assert decision.decision_source == "hybrid_rule_overturned"
    assert decision.reason == "not relevant here"


def test_hybrid_recall_evaluator_fails_closed_to_rule_when_verifier_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RecallVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _raise,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret", content_substrings=["secret"]
                )
            ],
            recall=RecallConfig(mode=RecallMode.HYBRID),
        )
    )

    decision = evaluator.evaluate_content("secret launch code", query="book a room")

    assert decision.action is RecallAction.WITHHOLD
    assert decision.policy_id == "block-secret"
    assert decision.decision_source == "hybrid_rule"
    assert decision.reason == "matched_deterministic_rule_llm_verifier_unavailable"


def test_hybrid_recall_evaluator_falls_back_to_llm_judge_with_query_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        calls.append((model, content, field_name, query))
        return RecallVerdict(action="withhold", reasoning="reveals a medical detail")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    decision = evaluator.evaluate_content(
        "Currently taking RX2019 for hypertension",
        metadata={"field_name": "MedicalNotes"},
        query="Message the hotel to book a room",
    )

    assert decision.action is RecallAction.WITHHOLD
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "reveals a medical detail"
    assert calls == [
        (
            "openai:gpt-4o-mini",
            "Currently taking RX2019 for hypertension",
            "MedicalNotes",
            "Message the hotel to book a room",
        )
    ]


def test_hybrid_recall_evaluator_exposes_when_llm_judge_says_expose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model, content, field_name, query, system_prompt, matched_category=None
    ) -> RecallVerdict:
        return RecallVerdict(action="expose", reasoning="ordinary preference")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    decision = evaluator.evaluate_content("Prefers window seats", query="book a flight")

    assert decision.action is RecallAction.EXPOSE
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "ordinary preference"


def test_hybrid_recall_evaluator_caches_llm_verdict_by_content_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model, content, field_name, query, system_prompt, matched_category=None
    ) -> RecallVerdict:
        calls.append((content, query))
        return RecallVerdict(action="expose", reasoning="ok")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    evaluator.evaluate_content("same content", query="task A")
    evaluator.evaluate_content("same content", query="task A")  # cache hit
    evaluator.evaluate_content("same content", query="task B")  # different query

    assert calls == [("same content", "task A"), ("same content", "task B")]


def test_hybrid_recall_evaluator_does_not_share_cache_across_matched_and_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: identical content text recurring with different
    metadata must not let a verdict answering one question (open-ended
    'is this sensitive') get reused to answer a different one (narrower
    'confirm this specific rule-matched category') -- the cache key must
    distinguish them, not collide on (content, query) alone."""
    matched_categories = []

    def _fake_judge(
        model, content, field_name, query, system_prompt, matched_category=None
    ) -> RecallVerdict:
        matched_categories.append(matched_category)
        # Answer differently depending on whether a rule matched, so a
        # collision would be visible in the returned decision.
        if matched_category is None:
            return RecallVerdict(action="expose", reasoning="not sensitive")
        return RecallVerdict(action="withhold", reasoning="confirmed sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="withhold-medical-history",
                    category_description="medical or health information",
                    metadata_contains={"field_name": "medical_history"},
                )
            ],
            recall=RecallConfig(mode=RecallMode.HYBRID),
        )
    )

    unmatched = evaluator.evaluate_content(
        "Lisinopril", metadata={"field_name": "notes"}, query="book a room"
    )
    matched = evaluator.evaluate_content(
        "Lisinopril", metadata={"field_name": "medical_history"}, query="book a room"
    )

    assert matched_categories == [None, "medical or health information"]
    assert unmatched.action is RecallAction.EXPOSE
    assert unmatched.decision_source == "hybrid_llm"
    assert matched.action is RecallAction.WITHHOLD
    assert matched.decision_source == "hybrid_rule_verified"
    assert matched.policy_id == "withhold-medical-history"


def test_hybrid_recall_evaluator_does_not_cache_llm_judge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _raise(*args: object, **kwargs: object) -> RecallVerdict:
        calls.append(1)
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _raise,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    evaluator.evaluate_content("same content", query="task A")
    evaluator.evaluate_content("same content", query="task A")

    assert len(calls) == 2  # not cached -- retried both times


def test_hybrid_recall_evaluator_fails_closed_when_llm_judge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RecallVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _raise,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.HYBRID))
    )

    decision = evaluator.evaluate_content("some unmatched content", query="q")

    assert decision.action is RecallAction.WITHHOLD
    assert decision.decision_source == "hybrid_llm"
    assert decision.reason == "llm_judge_unavailable_fail_closed"


def test_hybrid_recall_evaluator_builds_judge_prompt_from_rule_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the retention-side taxonomy test: the recall judge's category
    list must come from `category_description` on the configured recall
    rules, not a hardcoded prompt."""
    captured_prompts = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        captured_prompts.append(system_prompt)
        return RecallVerdict(action="expose", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="withhold-medical-general",
                    category_description="medical or health information",
                    metadata_contains={"field_name": "medical"},
                ),
                DeterministicRetentionRule(
                    rule_id="withhold-medical-prescription",
                    category_description="medical or health information",
                    metadata_contains={"field_name": "prescript"},
                ),
                DeterministicRetentionRule(
                    rule_id="withhold-financial-general",
                    category_description="financial or payment information",
                    metadata_contains={"field_name": "financial"},
                ),
            ],
            recall=RecallConfig(mode=RecallMode.HYBRID),
        )
    )

    evaluator.evaluate_content("Prefers window seats", query="book a flight")

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "- medical or health information" in prompt
    assert "- financial or payment information" in prompt
    # Same description repeated across sibling rules in a category appears once.
    assert prompt.count("medical or health information") == 1


def test_hybrid_recall_evaluator_builds_judge_prompt_from_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`policies` are free-text, user-declared instructions -- distinct
    from `rules`' category taxonomy -- and must reach the judge prompt too,
    as a separate section, without needing a matching rule at all."""
    captured_prompts = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        captured_prompts.append(system_prompt)
        return RecallVerdict(action="expose", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = HybridRecallEvaluator(
        PrivacyConfig(
            recall=RecallConfig(
                mode=RecallMode.HYBRID,
                policies=[
                    "do not expose my travel plans to anyone but family",
                    "never reveal my current location",
                ],
            )
        )
    )

    evaluator.evaluate_content("Some unrelated content", query="book a room")

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "- do not expose my travel plans to anyone but family" in prompt
    assert "- never reveal my current location" in prompt


def test_build_recall_judge_system_prompt_omits_policies_section_when_empty() -> None:
    prompt = build_recall_judge_system_prompt(rules=[], policies=[])

    assert "also explicitly asked" not in prompt


def test_build_recall_policy_resolves_llm_policy() -> None:
    policy = build_recall_policy(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.LLM))
    )

    assert isinstance(policy, LLMRecallEvaluator)


def test_llm_recall_evaluator_never_evaluates_rule_matchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoupled from deterministic mode on purpose: even a rule whose
    matcher would obviously hit is never consulted -- everything goes
    straight to the judge."""

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        assert matched_category is None
        return RecallVerdict(action="expose", reasoning="not sensitive")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = LLMRecallEvaluator(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="withhold-secret",
                    content_substrings=["secret"],
                )
            ],
            recall=RecallConfig(mode=RecallMode.LLM),
        )
    )

    decision = evaluator.evaluate_content("secret launch code", query="book a room")

    assert decision.action is RecallAction.EXPOSE
    assert decision.decision_source == "llm"


def test_llm_recall_evaluator_withholds_when_judge_says_withhold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        return RecallVerdict(action="withhold", reasoning="medical record")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = LLMRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.LLM))
    )

    decision = evaluator.evaluate_content("some content", query="q")

    assert decision.action is RecallAction.WITHHOLD
    assert decision.decision_source == "llm"
    assert decision.reason == "medical record"


def test_llm_recall_evaluator_fails_closed_when_judge_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: object, **kwargs: object) -> RecallVerdict:
        raise RuntimeError("provider outage")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _raise,
    )
    evaluator = LLMRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.LLM))
    )

    decision = evaluator.evaluate_content("some content", query="q")

    assert decision.action is RecallAction.WITHHOLD
    assert decision.decision_source == "llm"
    assert decision.reason == "llm_judge_unavailable_fail_closed"


def test_llm_recall_evaluator_caches_verdict_by_content_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        calls.append((content, query))
        return RecallVerdict(action="expose", reasoning="ok")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = LLMRecallEvaluator(
        PrivacyConfig(recall=RecallConfig(mode=RecallMode.LLM))
    )

    evaluator.evaluate_content("same content", query="task A")
    evaluator.evaluate_content("same content", query="task A")  # cache hit
    evaluator.evaluate_content("same content", query="task B")  # different query

    assert calls == [("same content", "task A"), ("same content", "task B")]


def test_llm_recall_evaluator_builds_judge_prompt_from_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompts = []

    def _fake_judge(
        model: str,
        content: str,
        field_name: str | None,
        query: str,
        system_prompt: str,
        matched_category: str | None = None,
    ) -> RecallVerdict:
        captured_prompts.append(system_prompt)
        return RecallVerdict(action="expose", reasoning="n/a")

    monkeypatch.setattr(
        "heams_memory_privacy_wrapper.recall_policy_hybrid.judge_recall_content",
        _fake_judge,
    )
    evaluator = LLMRecallEvaluator(
        PrivacyConfig(
            recall=RecallConfig(
                mode=RecallMode.LLM,
                policies=["do not expose my medical info"],
            )
        )
    )

    evaluator.evaluate_content("Some unrelated content", query="book a room")

    assert len(captured_prompts) == 1
    assert "- do not expose my medical info" in captured_prompts[0]


def test_privacy_policy_engine_delegates_priming_filter_to_selected_policy() -> None:
    engine = PrivacyPolicyEngine(
        PrivacyConfig(
            rules=[
                DeterministicRetentionRule(
                    rule_id="block-secret",
                    content_substrings=["secret"],
                )
            ],
            retention=RetentionConfig(mode=RetentionMode.DETERMINISTIC),
        )
    )

    admitted_entries, batch = engine.filter_priming_entries(
        [
            MemoryEntry(content="Allowed fact", metadata={"source": "user"}),
            MemoryEntry(content="secret launch code", metadata={"source": "user"}),
        ]
    )

    assert [entry.content for entry in admitted_entries] == ["Allowed fact"]
    assert batch.stored_indices == [0]
    assert batch.dropped_indices == [1]


def test_retention_decision_rejects_unsupported_action() -> None:
    with pytest.raises(ValidationError):
        RetentionDecision.model_validate({"action": "transform"})


def test_retention_batch_result_tracks_store_drop_decisions() -> None:
    result = RetentionBatchResult(
        decisions=[
            RetentionDecision(action=RetentionAction.STORE, reason="default"),
            RetentionDecision(action=RetentionAction.DROP, reason="policy"),
        ],
        stored_indices=[0],
        dropped_indices=[1],
    )

    assert [decision.action for decision in result.decisions] == [
        RetentionAction.STORE,
        RetentionAction.DROP,
    ]
    assert result.stored_indices == [0]
    assert result.dropped_indices == [1]


def test_retention_config_parses_policy_rules_keyed_by_statement() -> None:
    config = PrivacyConfig.model_validate(
        {
            "retention": {
                "policy_rules": {
                    "Do not retain the user's pet information.": [
                        {"rule_id": "drop-pet", "content_substrings": ["dog", "cat"]}
                    ]
                }
            }
        }
    )

    (rule,) = config.retention.policy_rules["Do not retain the user's pet information."]
    assert isinstance(rule, DeterministicRetentionRule)
    assert rule.content_substrings == ["dog", "cat"]
    assert config.rules == []


def test_retention_config_policy_rules_require_a_matcher() -> None:
    with pytest.raises(ValidationError, match="at least one matcher"):
        RetentionConfig.model_validate({"policy_rules": {"policy": [{"rule_id": "x"}]}})
