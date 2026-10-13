from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class PrivacyConfig(BaseModel):
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
