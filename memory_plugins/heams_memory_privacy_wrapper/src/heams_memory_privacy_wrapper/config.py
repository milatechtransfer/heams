from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heams.memory_systems.api import BaseMemoryWrapperConfig


class RetentionMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HYBRID = "hybrid"
    TRANSFORM = "transform"


class RetentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: RetentionMode = Field(default=RetentionMode.DETERMINISTIC)
    max_batch_tokens: int | None = Field(default=None, ge=1)
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description=(
            "any_llm model id used to judge entries the deterministic rules "
            "don't match; only consulted in hybrid/llm retention modes."
        ),
    )
    policies: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text, user-declared policies (e.g. 'do not record my "
            "medical info'), read directly into the LLM judge's prompt as "
            "instructions -- not matched against content like the shared "
            "`PrivacyConfig.rules`. Only consulted in hybrid/llm retention "
            "modes; deterministic mode has no LLM to act on them."
        ),
    )
    policy_rules: dict[str, list["DeterministicRetentionRule"]] = Field(
        default_factory=dict,
        description=(
            "Rules each free-text policy compiles to, keyed by the exact "
            "policy statement -- how a rule-matching layer enforces a policy "
            "it can't read. Consulted only by "
            "`PrivacyMemoryWrapper.set_retention_policy`, which appends the "
            "active policies' rules to the shared `PrivacyConfig.rules` in "
            "deterministic/hybrid modes; llm mode reads the statement itself."
        ),
    )


class RecallMode(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    HYBRID = "hybrid"


class RecallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: RecallMode = Field(default=RecallMode.DETERMINISTIC)
    window_multiplier: int = Field(
        default=4,
        ge=1,
        description=(
            "recall() over-fetches top_k * window_multiplier candidates from "
            "the inner backend, filters them, then returns the first top_k "
            "survivors. Bounded (not the whole pool) so cost stays flat on "
            "large memory pools."
        ),
    )
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description=(
            "any_llm model id used to judge candidates the deterministic "
            "rules don't match; only consulted in hybrid/llm recall modes."
        ),
    )
    policies: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text, user-declared policies (e.g. 'do not expose my "
            "travel plans to anyone but family'), read directly into the "
            "LLM judge's prompt as instructions -- not matched against "
            "content like the shared `PrivacyConfig.rules`. Only consulted "
            "in hybrid/llm recall modes. Independent of `retention.policies` "
            "-- the two layers can carry different standing instructions."
        ),
    )


class DeterministicRetentionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str | None = None
    category_description: str | None = Field(
        default=None,
        description=(
            "One-line description of this rule's privacy category (e.g. "
            "'government identification (passport, SSN, national ID, "
            "driver's license)'). Set on one representative rule per "
            "category; the hybrid retention/recall LLM judge prompt is "
            "built from these so the deterministic ruleset and the judge's "
            "taxonomy can't drift apart."
        ),
    )
    content_substrings: list[str] = Field(default_factory=list)
    metadata_equals: dict[str, Any] = Field(default_factory=dict)
    metadata_contains: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_matchers(self) -> "DeterministicRetentionRule":
        if (
            not self.content_substrings
            and not self.metadata_equals
            and not self.metadata_contains
        ):
            raise ValueError(
                "Deterministic retention rules require at least one matcher."
            )
        return self


# `RetentionConfig.policy_rules` names this model before it is defined.
RetentionConfig.model_rebuild()


#: Retention modes that enforce a policy via its `retention.policy_rules` entry.
_RULE_COMPILING_MODES = frozenset({RetentionMode.DETERMINISTIC, RetentionMode.HYBRID})


class PrivacyConfig(BaseMemoryWrapperConfig):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    rules: list[DeterministicRetentionRule] = Field(
        default_factory=list,
        description=(
            "The PII ruleset, shared by every layer that reads it (today: "
            "retention and recall). Deliberately kept at this level rather "
            "than nested under `retention` or `recall` -- a single "
            "definition that any privacy layer can reference means the "
            "taxonomy can't drift between layers the way per-layer copies "
            "could."
        ),
    )

    @property
    def has_rules(self) -> bool:
        return bool(self.rules)

    @property
    def is_active(self) -> bool:
        return self.enabled and (self.retention.enabled or self.recall.enabled)

    def check_retention_policies(self, policies: list[str]) -> None:
        """Reject a configuration that would make retention_policy scores ambiguous.

        Inactive, the wrapper isn't applied and the memory system runs
        ungoverned, so there is nothing to check.
        """
        if not self.is_active:
            return
        from .recall import NoOpRecallPolicy, build_recall_policy

        # A recall policy filtering results on top of retention makes a missing
        # recall ambiguous between "retention blocked it at ingest" and "a recall
        # policy withheld it separately". Probe with every compiled policy's rules
        # included: in rule-based modes `set_retention_policy` appends them at
        # runtime, which would silently activate a recall layer that looks
        # rule-less in the YAML. llm mode never appends them.
        retention = self.retention
        compiles_rules = retention.mode in _RULE_COMPILING_MODES
        runtime_rules = (
            [rule for rules in retention.policy_rules.values() for rule in rules]
            if compiles_rules
            else []
        )
        probe = self.model_copy(update={"rules": [*self.rules, *runtime_rules]})
        # `build_recall_policy` alone, not the full `PrivacyPolicyEngine`: the
        # engine also builds a retention policy this check never reads, which can
        # raise for retention modes unrelated to recall (e.g. the unimplemented
        # `transform` mode) and does real judge-prompt-building work for
        # hybrid/llm retention that would be wasted here.
        if not isinstance(build_recall_policy(probe), NoOpRecallPolicy):
            raise ValueError(
                "retention_policy requires privacy_config.recall.enabled: false. "
                "With a recall policy active, a fact that isn't recalled is "
                "ambiguous between 'retention blocked it at ingest' and 'a recall "
                "policy withheld it separately' -- exactly the ambiguity this "
                "benchmark exists to eliminate."
            )
        if compiles_rules and retention.policy_rules:
            missing = sorted(set(policies) - retention.policy_rules.keys())
            if missing:
                raise ValueError(
                    "retention.policy_rules has no rules for these benchmark "
                    f"policies: {missing}."
                )
