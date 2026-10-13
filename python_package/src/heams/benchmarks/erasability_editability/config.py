"""Typed workload config for the erasability/editability (EE) benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.info import BenchmarkInfo, ReportDimension
from heams.benchmarks.erasability_editability.judge import EELlmJudgeConfig

EE_SUPPORTED_TASK_TYPES: frozenset[str] = frozenset({"erase", "edit"})


class EESourceConfig(BaseModel):
    """Where the MEME corpus comes from.

    The published dataset is a plain JSON file on HuggingFace
    (`meme-benchmark/MEME` / `meme_filler32k.json`, ~20 MB, a bare list of 100
    episode objects). By default the first run downloads it (cached under
    `~/.cache/huggingface`). Set ``local_path`` to read a local copy instead, or
    ``auto_download: false`` to make a missing dataset a hard error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str = Field(
        default="meme-benchmark/MEME", description="HuggingFace dataset repo id."
    )
    filename: str = Field(
        default="meme_filler32k.json",
        description="Dataset file name inside the HuggingFace repo.",
    )
    revision: str | None = Field(
        default=None, description="Optional HuggingFace revision (branch/tag/commit)."
    )
    auto_download: bool = Field(
        default=True,
        description=(
            "Download from HuggingFace when `local_path` is unset. When false, a "
            "missing dataset is an error."
        ),
    )
    local_path: Path | None = Field(
        default=None,
        description=(
            "Optional local MEME episodes JSON; used instead of downloading when set."
        ),
    )


class EEBenchmarkConfig(BaseModel):
    """Typed workload config for the EE (erasability / editability) benchmark.

    The probe, replay and judge prompts are authored in
    ``configs/benchmarks/*.yaml``; there are no Python prompt fallbacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["ee"] = "ee"

    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="Erasability & Editability",
        dimension=ReportDimension.PRIVACY,
        criterion_tags=(("task_type", "phase"), ("task_type",), ("phase",)),
    )
    source: EESourceConfig = Field(default_factory=EESourceConfig)
    limit: int | None = Field(
        default=None, ge=1, description="Cap on source episodes, applied after loading."
    )
    task_types: tuple[str, ...] = Field(
        default=("erase", "edit"),
        description=(
            "Task types to grade. Only 'erase' (MEME's Del) and 'edit' (MEME's "
            "Cas) are supported; other questions in the episode are dropped."
        ),
    )
    post_change_filler: Literal["prime", "drop"] = Field(
        default="prime",
        description=(
            "What to do with the inert filler sessions that follow the "
            "change+delete event. 'prime' keeps the full haystack in memory; "
            "'drop' discards them."
        ),
    )
    probe_prompt: str = Field(
        description="UserStimulus template for each probe; must include {query}.",
    )
    replay_prompt: str = Field(
        default="{turn}",
        description=(
            "UserStimulus template for each replayed change+delete turn; must "
            "include {turn}. Defaults to the identity template: the agent must "
            "see the deletion request verbatim, since any wrapper would cue it "
            "that the turn is special. Override only for framing ablations."
        ),
    )
    agent_conversation: Literal["stateful", "independent"] = Field(
        default="stateful",
        description=(
            "stateful: later turns see earlier probes and the replayed change "
            "session in the agent's messages. independent: reset agent messages "
            "before each turn, so after-probes can only learn the change from memory."
        ),
    )
    evaluator: EELlmJudgeConfig

    @field_validator("probe_prompt")
    @classmethod
    def _require_query_placeholder(cls, value: str) -> str:
        if "{query}" not in value:
            raise ValueError("probe_prompt must include the {query} placeholder.")
        return value

    @field_validator("replay_prompt")
    @classmethod
    def _require_turn_placeholder(cls, value: str) -> str:
        if "{turn}" not in value:
            raise ValueError("replay_prompt must include the {turn} placeholder.")
        return value

    @field_validator("task_types")
    @classmethod
    def _require_supported_task_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("task_types must not be empty.")
        unsupported = sorted(set(value) - EE_SUPPORTED_TASK_TYPES)
        if unsupported:
            supported = ", ".join(sorted(EE_SUPPORTED_TASK_TYPES))
            raise ValueError(
                f"Unsupported EE task_types {unsupported}; supported: {supported}."
            )
        return value


__all__ = ["EE_SUPPORTED_TASK_TYPES", "EEBenchmarkConfig", "EESourceConfig"]
