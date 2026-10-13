"""Project Transparency source samples into shared BenchmarkSample instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig
from heams.benchmarks.transparency.judge import (
    TransparencyLlmJudgeConfig,
    TransparencyQA,
    TransparencyQAGroundTruth,
)
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

TRANSPARENCY_DEFAULT_FILENAME = "seed.json"
TRANSPARENCY_SCENARIO_TYPES = {
    "direct_inspection",
    "category_filtering",
    "provenance_inspection",
    "metadata_inspection",
    "negative_audit",
    "full_memory_export",
    "conflict_disclosure",
    "opacity_disclosure",
}


class TransparencyPersona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str


class TransparencyDifficulty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["easy", "medium", "hard"]
    memory_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    distractor_count: int = Field(ge=0)
    metadata_required: bool = False


class TransparencyMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransparencyExpectedMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    fact: str


class TransparencySample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    privacy_domain: str
    source: dict[str, Any] = Field(default_factory=dict)
    persona: TransparencyPersona
    difficulty: TransparencyDifficulty
    question: str
    target_memories: list[TransparencyMemory] = Field(default_factory=list)
    background_memories: list[TransparencyMemory] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    expected_metadata: list[TransparencyExpectedMetadata] = Field(default_factory=list)
    prohibited_facts: list[str] = Field(default_factory=list)
    rubric: str

    @model_validator(mode="after")
    def validate_sample_contract(self) -> "TransparencySample":
        if self.type not in TRANSPARENCY_SCENARIO_TYPES:
            supported = ", ".join(sorted(TRANSPARENCY_SCENARIO_TYPES))
            raise ValueError(
                f"Unsupported transparency scenario type {self.type!r}. "
                f"Supported: {supported}."
            )
        if self.difficulty.target_count != len(self.target_memories):
            raise ValueError(
                f"{self.id}: difficulty.target_count does not match target_memories."
            )
        if self.difficulty.distractor_count != len(self.background_memories):
            raise ValueError(
                f"{self.id}: difficulty.distractor_count does not match "
                "background_memories."
            )
        if self.difficulty.memory_count != len(self.target_memories) + len(
            self.background_memories
        ):
            raise ValueError(
                f"{self.id}: difficulty.memory_count does not match total memories."
            )
        return self


def _memory_to_entry(
    memory: TransparencyMemory,
    *,
    sample: TransparencySample,
    memory_role: Literal["target", "background"],
    memory_index: int,
) -> MemoryEntry:
    metadata = {
        **memory.metadata,
        "source_benchmark": "transparency",
        "scenario_id": sample.id,
        "memory_id": memory.id,
        "memory_role": memory_role,
        "memory_index": memory_index,
        "scenario_type": sample.type,
        "privacy_domain": sample.privacy_domain,
        "persona_id": sample.persona.id,
    }
    if sample.source:
        metadata["sample_source"] = dict(sample.source)
    return MemoryEntry(content=memory.content, metadata=metadata)


def sample_to_benchmark_sample(
    sample: TransparencySample,
    *,
    config: TransparencyBenchmarkConfig,
    evaluator: TransparencyLlmJudgeConfig | None = None,
) -> BenchmarkSample:
    """Convert one Transparency source sample into a shared BenchmarkSample."""
    memory_entries: list[MemoryEntry] = []
    target_memory_ids: list[str] = []
    target_contents: list[str] = []

    for index, memory in enumerate(sample.target_memories):
        memory_entries.append(
            _memory_to_entry(
                memory,
                sample=sample,
                memory_role="target",
                memory_index=index,
            )
        )
        target_memory_ids.append(memory.id)
        target_contents.append(memory.content)

    for index, memory in enumerate(sample.background_memories):
        memory_entries.append(
            _memory_to_entry(
                memory,
                sample=sample,
                memory_role="background",
                memory_index=len(sample.target_memories) + index,
            )
        )

    expected_metadata = [metadata.model_dump() for metadata in sample.expected_metadata]
    markers: dict[str, Any] = {
        "question_id": sample.id,
        "question": sample.question,
        "scenario_type": sample.type,
        "privacy_domain": sample.privacy_domain,
        "source": dict(sample.source),
        "persona": sample.persona.model_dump(),
        "difficulty": sample.difficulty.model_dump(),
        "expected_facts": list(sample.expected_facts),
        "expected_metadata": expected_metadata,
        "prohibited_facts": list(sample.prohibited_facts),
        "rubric": sample.rubric,
        "retrieval_evidence": {
            "match_key": "memory_id",
            "relevant_values": target_memory_ids,
            "relevant_contents": target_contents,
        },
    }

    ground_truth = TransparencyQAGroundTruth(
        task_description=f"Transparency {sample.id}",
        qa=[
            TransparencyQA(
                question_id=sample.id,
                question=sample.question,
                scenario_type=sample.type,
                expected_facts=list(sample.expected_facts),
                expected_metadata=expected_metadata,
                prohibited_facts=list(sample.prohibited_facts),
                stored_inventory=list(target_contents),
                rubric=sample.rubric,
                privacy_domain=sample.privacy_domain,
                persona=sample.persona.model_dump(),
                difficulty=sample.difficulty.model_dump(),
            )
        ],
    )

    return BenchmarkSample(
        id=sample.id,
        name=f"Transparency {sample.id}",
        tags=[
            "transparency",
            sample.type,
            f"domain:{sample.privacy_domain}",
            f"persona:{sample.persona.id}",
            f"difficulty:{sample.difficulty.level}",
        ],
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=memory_entries)),
        execution_sequence=[
            UserStimulus(
                content=config.audit_prompt.format(question=sample.question),
                step_evaluation_markers=markers,
            )
        ],
        evaluators=[evaluator or config.evaluator],
        ground_truth=ground_truth,
    )


def load_transparency_samples(
    source: str | Path | list[dict[str, Any]],
) -> list[TransparencySample]:
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source

    if not isinstance(payload, list):
        raise ValueError("Transparency data must be a list of sample objects.")

    return [TransparencySample.model_validate(sample) for sample in payload]


__all__ = [
    "TRANSPARENCY_DEFAULT_FILENAME",
    "TRANSPARENCY_SCENARIO_TYPES",
    "TransparencyDifficulty",
    "TransparencyExpectedMetadata",
    "TransparencyMemory",
    "TransparencyPersona",
    "TransparencySample",
    "load_transparency_samples",
    "sample_to_benchmark_sample",
]
