"""Project STALE source samples into shared BenchmarkSample instances.

STALE records describe a user state that changed from ``M_old`` to ``M_new`` inside
a primed conversation (``haystack_session``), plus three probing queries. Each
source record becomes **one** BenchmarkSample / scenario with three probing
stimuli (dim1/dim2/dim3) so the agent answers all probes in a single run and the
all-in-one judge can score them together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.stale.config import StaleBenchmarkConfig
from heams.benchmarks.stale.judge import (
    StaleLlmJudgeConfig,
    StaleQA,
    StaleQAGroundTruth,
)
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

STALE_DEFAULT_FILENAME = "seed.json"

#: (dimension key, probing-query field) in probe order.
STALE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("dim1", "dim1_query"),
    ("dim2", "dim2_query"),
    ("dim3", "dim3_query"),
)


class StaleProbingQueries(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    dim1_query: str = Field(validation_alias=AliasChoices("dim1_query", "dim1"))
    dim2_query: str = Field(validation_alias=AliasChoices("dim2_query", "dim2"))
    dim3_query: str = Field(validation_alias=AliasChoices("dim3_query", "dim3"))


class StaleSample(BaseModel):
    """One validated STALE source record."""

    # STALE MAIN.json carries auxiliary generation fields we do not need, so we
    # ignore extras rather than forbidding them.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    uid: str = Field(validation_alias=AliasChoices("uid", "id"))
    m_old: str = Field(validation_alias=AliasChoices("M_old", "m_old", "old_info"))
    m_new: str = Field(validation_alias=AliasChoices("M_new", "m_new"))
    explanation: str = ""
    probing_queries: StaleProbingQueries
    haystack_session: list[Any] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "haystack_session", "haystack_sessions", "sessions"
        ),
    )
    # Session-level timestamps (published dataset) aligned to `haystack_session`
    # sessions; empty for the flat demo fixture.
    timestamps: list[str] = Field(default_factory=list)
    # Indices of the sessions carrying the M_old / M_new observations.
    relevant_session_index: list[int] = Field(default_factory=list)
    # The published dataset names the conflict class `type` (T1/T2); the demo
    # fixture uses `conflict_type`.
    conflict_type: str | None = Field(
        default=None, validation_alias=AliasChoices("type", "conflict_type")
    )


def _normalize_sessions(haystack_session: list[Any]) -> list[list[dict[str, Any]]]:
    """Normalize to a list of sessions (each a list of turn dicts).

    The published dataset is a list of sessions (list-of-lists); the demo fixture
    is a flat list of turn dicts, which is treated as a single session.
    """
    sessions: list[list[dict[str, Any]]] = []
    flat: list[dict[str, Any]] = []
    for item in haystack_session:
        if isinstance(item, list):
            sessions.append([turn for turn in item if isinstance(turn, dict)])
        elif isinstance(item, dict):
            flat.append(item)
    if flat:
        sessions.append(flat)
    return sessions


def _build_memory_entries(
    sample: StaleSample, *, conflict_type: str
) -> tuple[list[MemoryEntry], list[str]]:
    """Build session-aware memory entries and the relevant-session contents."""
    relevant = set(sample.relevant_session_index)
    entries: list[MemoryEntry] = []
    relevant_contents: list[str] = []
    turn_index = 0
    for session_index, session in enumerate(
        _normalize_sessions(sample.haystack_session)
    ):
        session_timestamp = (
            sample.timestamps[session_index]
            if session_index < len(sample.timestamps)
            else None
        )
        for turn in session:
            content = turn.get("content")
            if not content:
                continue
            metadata: dict[str, Any] = {
                "source_benchmark": "stale",
                "scenario_id": sample.uid,
                "session_index": session_index,
                "turn_index": turn_index,
                "conflict_type": conflict_type,
            }
            role = turn.get("role")
            if role is not None:
                metadata["role"] = role
            timestamp = (
                session_timestamp
                or turn.get("timestamp")
                or turn.get("time")
                or turn.get("date")
            )
            if timestamp is not None:
                metadata["timestamp"] = timestamp
            if session_index in relevant:
                metadata["relevant_session"] = True
                relevant_contents.append(str(content))
            entries.append(MemoryEntry(content=str(content), metadata=metadata))
            turn_index += 1
    return entries, relevant_contents


def sample_to_benchmark_sample(
    sample: StaleSample,
    *,
    config: StaleBenchmarkConfig,
    evaluator: StaleLlmJudgeConfig | None = None,
) -> BenchmarkSample:
    """Convert one STALE source sample into one scenario with three probes.

    The three probing queries become three ``UserStimulus`` steps in a single
    scenario so the agent answers all of them in one run and the all-in-one judge
    can score them together (see ``stale/judge.py``).
    """
    conflict_type = sample.conflict_type or config.default_conflict_type
    resolved_evaluator = evaluator or config.evaluator

    memory_entries, relevant_contents = _build_memory_entries(
        sample, conflict_type=conflict_type
    )

    retrieval_evidence: dict[str, Any] | None = None
    if sample.relevant_session_index:
        retrieval_evidence = {
            "match_key": "session_index",
            "relevant_values": list(sample.relevant_session_index),
        }
        if relevant_contents:
            retrieval_evidence["relevant_contents"] = relevant_contents

    stimuli: list[UserStimulus] = []
    qa: list[StaleQA] = []
    for dimension, query_field in STALE_DIMENSIONS:
        query = getattr(sample.probing_queries, query_field)
        markers: dict[str, Any] = {
            "uid": sample.uid,
            "dimension": dimension,
            "question": query,
            "m_old": sample.m_old,
            "m_new": sample.m_new,
            "explanation": sample.explanation,
            "conflict_type": conflict_type,
        }
        if sample.relevant_session_index:
            markers["relevant_session_index"] = list(sample.relevant_session_index)
        if retrieval_evidence is not None:
            markers["retrieval_evidence"] = retrieval_evidence
        stimuli.append(
            UserStimulus(
                content=config.probe_prompt.format(query=query),
                step_evaluation_markers=markers,
            )
        )
        qa.append(
            StaleQA(
                uid=sample.uid,
                dimension=dimension,
                question=query,
                m_old=sample.m_old,
                m_new=sample.m_new,
                explanation=sample.explanation,
                conflict_type=conflict_type,
            )
        )

    return BenchmarkSample(
        id=sample.uid,
        name=f"STALE {sample.uid}",
        tags=["stale", f"conflict:{conflict_type}"],
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(entries=list(memory_entries))
        ),
        execution_sequence=stimuli,
        evaluators=[resolved_evaluator],
        ground_truth=StaleQAGroundTruth(task_description=f"STALE {sample.uid}", qa=qa),
        agent_conversation=config.agent_conversation,
    )


def load_stale_samples(
    source: str | Path | list[dict[str, Any]],
) -> list[StaleSample]:
    """Load and validate STALE source samples from a path or in-memory list."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source

    # STALE writes either a bare list or an object with a "data" field.
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("samples"))
    if not isinstance(payload, list):
        raise ValueError("STALE data must be a list of sample objects.")

    return [StaleSample.model_validate(sample) for sample in payload]


__all__ = [
    "STALE_DEFAULT_FILENAME",
    "STALE_DIMENSIONS",
    "StaleProbingQueries",
    "StaleSample",
    "load_stale_samples",
    "sample_to_benchmark_sample",
]
