"""Project MEME episodes into shared BenchmarkSample instances.

A MEME episode plants a dependency chain of facts across many sessions, then
fires exactly one **Change+Delete Event** session that both changes a *root*
entity and explicitly asks the assistant to forget a fact. Probes fired before
and after that event measure deletion compliance and cascade correctness.

Each episode becomes **one** BenchmarkSample whose execution sequence is three
phases::

    [before probes]  ->  [replayed change+delete turns]  ->  [after probes]

Everything except the change+delete session is injected up front through
``MemoryPriming`` (which runs the real ``retain()`` pipeline). The change+delete
session is *replayed* as live ``UserStimulus`` turns instead, so the agent's tool
loop actually sees "please remove that from your memory" and can call the
``delete`` tool - the behaviour this benchmark exists to measure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig
from heams.benchmarks.erasability_editability.judge import (
    EELlmJudgeConfig,
    EEQa,
    EEQaGroundTruth,
    ee_probe_key,
)
from heams.benchmarks.sample import BenchmarkSample
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

CHANGE_DELETE_EVIDENCE_TYPE = "Change+Delete Event"


class EERootChange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    before: str = ""
    after: str = ""


class EEQuestion(BaseModel):
    """One probing question from an episode's before/after question block."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    task_type: str
    question: str
    # The before block names the gold value `expected_answer`; the after block
    # names it `gold_answer`. Same meaning, one field.
    gold_answer: str = Field(
        validation_alias=AliasChoices("gold_answer", "expected_answer")
    )
    entity: list[str] = Field(default_factory=list)
    entity_values: dict[str, Any] = Field(default_factory=dict)
    hop: int | None = None

    @property
    def entity_label(self) -> str:
        """Join key between the before and after phases.

        Every in-scope (erase/edit) question is single-entity in the published data,
        so this is the bare entity name; it is joined with ", " only as a
        defensive fallback.
        """
        if self.entity:
            return ", ".join(self.entity)
        return ", ".join(self.entity_values)

    @property
    def gold_value(self) -> str:
        """The entity's value, which is what the judge compares against.

        The reference implementation passes ``entity_values[entity]`` as GOLD, not
        the ``gold_answer`` string. The two agree for every edit probe, but diverge
        for every erase after-probe, where ``gold_answer`` is the constant
        "No - explicitly removed" while the value is the fact that must NOT be
        revealed. The deletion rubric cannot detect leakage without the value.
        """
        for key in self.entity or list(self.entity_values):
            if key in self.entity_values:
                return str(self.entity_values[key])
        return ""


class EEQuestionBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str = ""
    position_after_session: int | None = None
    questions: list[EEQuestion] = Field(default_factory=list)


class EETurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = ""
    content: str = ""


class EESession(BaseModel):
    """One conversation session. Filler sessions omit `evidence_type`."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = ""
    type: str = ""
    evidence_type: str | None = None
    timestamp: str | None = None
    conversation: list[EETurn] = Field(default_factory=list)


class EEEpisode(BaseModel):
    """One validated MEME episode."""

    # MEME episodes carry generation metadata (gold_facts, dependency_edges_used,
    # token counts) we do not need, so ignore extras rather than forbidding them.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    episode_id: str = Field(validation_alias=AliasChoices("episode_id", "id"))
    domain: str = "unknown"
    root: str = ""
    root_change: EERootChange = Field(default_factory=EERootChange)
    sessions: list[EESession] = Field(default_factory=list)
    evidence_session_indices: list[int] = Field(default_factory=list)
    before_questions: EEQuestionBlock = Field(default_factory=EEQuestionBlock)
    after_questions: EEQuestionBlock = Field(default_factory=EEQuestionBlock)


def _change_session_index(episode: EEEpisode) -> int:
    """Index of the single Change+Delete Event session.

    Raises ``ValueError`` (surfaced as a `SkipReason`) when the episode does not
    carry exactly one, since the whole three-phase projection depends on it.
    """
    matches = [
        index
        for index, session in enumerate(episode.sessions)
        if session.evidence_type == CHANGE_DELETE_EVIDENCE_TYPE
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Episode '{episode.episode_id}' must contain exactly one "
            f"'{CHANGE_DELETE_EVIDENCE_TYPE}' session, found {len(matches)}."
        )
    return matches[0]


def _primed_session_indices(
    episode: EEEpisode, change_index: int, *, post_change_filler: str
) -> list[int]:
    """Every session index that is primed into memory before execution."""
    indices = [index for index in range(len(episode.sessions)) if index != change_index]
    if post_change_filler == "drop":
        indices = [index for index in indices if index < change_index]
    return indices


def _build_memory_entries(
    episode: EEEpisode, session_indices: list[int]
) -> list[MemoryEntry]:
    """Build session-aware memory entries for the primed sessions.

    ``session_index`` keeps the episode's *original* session numbering so it lines
    up with ``evidence_session_indices`` for the retrieval-evidence metrics.
    """
    relevant = set(episode.evidence_session_indices)
    entries: list[MemoryEntry] = []
    turn_index = 0
    for session_index in session_indices:
        session = episode.sessions[session_index]
        for turn in session.conversation:
            if not turn.content:
                continue
            metadata: dict[str, Any] = {
                "source_benchmark": "ee",
                "scenario_id": episode.episode_id,
                "session_index": session_index,
                "turn_index": turn_index,
                "session_type": session.type,
                "domain": episode.domain,
            }
            if turn.role:
                metadata["role"] = turn.role
            if session.evidence_type:
                metadata["evidence_type"] = session.evidence_type
            if session.timestamp:
                metadata["timestamp"] = session.timestamp
            if session_index in relevant:
                metadata["relevant_session"] = True
            entries.append(MemoryEntry(content=turn.content, metadata=metadata))
            turn_index += 1
    return entries


MEME_TASK_TYPES: dict[str, str] = {
    "Del": "erase",
    "Cas": "edit",
}


def _in_scope_questions(
    block: EEQuestionBlock, task_types: tuple[str, ...]
) -> list[tuple[str, EEQuestion]]:
    """Return ``(task_type, question)`` for questions this run grades.

    ``task_type`` is the translated name; the raw MEME tag is not used past
    this point.
    """
    allowed = set(task_types)
    resolved = (
        (MEME_TASK_TYPES.get(question.task_type), question)
        for question in block.questions
    )
    return [
        (task_type, question)
        for task_type, question in resolved
        if task_type is not None and task_type in allowed
    ]


def _probe_step(
    episode: EEEpisode,
    question: EEQuestion,
    *,
    task_type: str,
    phase: str,
    ordinal: int,
    response_index: int,
    config: EEBenchmarkConfig,
) -> tuple[UserStimulus, EEQa]:
    key = ee_probe_key(episode.episode_id, task_type, phase, ordinal)
    markers: dict[str, Any] = {
        "probe_key": key,
        "episode_id": episode.episode_id,
        "phase": phase,
        "task_type": task_type,
        "entity": question.entity_label,
        "question": question.question,
        "gold_answer": question.gold_answer,
        "gold_value": question.gold_value,
        "expected_answer": question.gold_answer,
        "domain": episode.domain,
        "root": episode.root,
        "root_change_before": episode.root_change.before,
        "root_change_after": episode.root_change.after,
    }
    if question.hop is not None:
        markers["hop"] = question.hop
    if episode.evidence_session_indices:
        markers["retrieval_evidence"] = {
            "match_key": "session_index",
            "relevant_values": list(episode.evidence_session_indices),
        }
    stimulus = UserStimulus(
        content=config.probe_prompt.format(query=question.question),
        step_evaluation_markers=markers,
    )
    qa = EEQa(
        probe_key=key,
        response_index=response_index,
        episode_id=episode.episode_id,
        phase=phase,
        task_type=task_type,
        entity=question.entity_label,
        question=question.question,
        gold_answer=question.gold_answer,
        gold_value=question.gold_value,
        hop=question.hop,
        domain=episode.domain,
        root=episode.root,
        root_change_before=episode.root_change.before,
        root_change_after=episode.root_change.after,
    )
    return stimulus, qa


def episode_to_benchmark_sample(
    episode: EEEpisode,
    *,
    config: EEBenchmarkConfig,
    evaluator: EELlmJudgeConfig | None = None,
) -> BenchmarkSample:
    """Convert one MEME episode into one three-phase scenario."""
    resolved_evaluator = evaluator or config.evaluator
    change_index = _change_session_index(episode)
    change_session = episode.sessions[change_index]

    replay_turns = [
        turn.content
        for turn in change_session.conversation
        if turn.role == "user" and turn.content
    ]
    if not replay_turns:
        raise ValueError(
            f"Episode '{episode.episode_id}' change+delete session has no user turns "
            "to replay."
        )

    after_questions = _in_scope_questions(episode.after_questions, config.task_types)
    if not after_questions:
        raise ValueError(
            f"Episode '{episode.episode_id}' has no after-probes for task types "
            f"{list(config.task_types)}."
        )
    # Always asked: these are what trivial-pass filtering joins against.
    before_questions = _in_scope_questions(episode.before_questions, config.task_types)

    stimuli: list[UserStimulus] = []
    qa: list[EEQa] = []
    # Probe ordinals are 1-based and restart for each (task_type, phase), so a
    # key set reads as "two edits and one erase" without counting rows.
    ordinals: dict[tuple[str, str], int] = {}

    def add_probes(questions: list[tuple[str, EEQuestion]], phase: str) -> None:
        for task_type, question in questions:
            slot = (task_type, phase)
            ordinals[slot] = ordinals.get(slot, 0) + 1
            stimulus, probe = _probe_step(
                episode,
                question,
                task_type=task_type,
                phase=phase,
                ordinal=ordinals[slot],
                response_index=len(stimuli),
                config=config,
            )
            stimuli.append(stimulus)
            qa.append(probe)

    add_probes(before_questions, "before")

    # Ungraded ingestion steps: no evaluation markers, so the judge's
    # marker-based ground-truth path skips them.
    for turn in replay_turns:
        stimuli.append(UserStimulus(content=config.replay_prompt.format(turn=turn)))

    add_probes(after_questions, "after")

    memory_entries = _build_memory_entries(
        episode,
        _primed_session_indices(
            episode, change_index, post_change_filler=config.post_change_filler
        ),
    )

    task_tags = sorted({task_type for task_type, _question in after_questions})
    return BenchmarkSample(
        id=episode.episode_id,
        name=f"EE {episode.episode_id}",
        tags=["ee", f"domain:{episode.domain}", *(f"task:{t}" for t in task_tags)],
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=memory_entries)),
        execution_sequence=stimuli,
        evaluators=[resolved_evaluator],
        ground_truth=EEQaGroundTruth(
            task_description=f"EE {episode.episode_id}", qa=qa
        ),
        agent_conversation=config.agent_conversation,
    )


def load_ee_episodes(
    source: str | Path | list[dict[str, Any]],
) -> list[EEEpisode]:
    """Load and validate MEME episodes from a path or in-memory list."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source

    # MEME ships a bare list; tolerate a wrapper object too.
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("episodes"))
    if not isinstance(payload, list):
        raise ValueError("MEME data must be a list of episode objects.")

    return [EEEpisode.model_validate(episode) for episode in payload]


__all__ = [
    "CHANGE_DELETE_EVIDENCE_TYPE",
    "EEEpisode",
    "EEQuestion",
    "EEQuestionBlock",
    "EERootChange",
    "EESession",
    "EETurn",
    "episode_to_benchmark_sample",
    "load_ee_episodes",
]
