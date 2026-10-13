"""Project BEAM conversations into BenchmarkSample instances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from heams.benchmarks.beam.dataset import (
    BEAM_DATASET_NAME,
    extract_probing_questions,
    extract_rubric_nuggets,
    iter_flat_turns,
)
from heams.benchmarks.beam.judge import BeamQA, BeamQAGroundTruth
from heams.benchmarks.sample import BenchmarkSample
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

if TYPE_CHECKING:
    from heams.benchmarks.beam.config import BeamBenchmarkConfig

_INGEST_CHUNK_SIZE = 2


def _time_anchor_to_epoch(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("-", " ")):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            continue
    for fmt in (
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B-%d-%Y",
    ):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            continue
    return None


def _turn_to_memory_entry(
    turn: dict[str, Any],
    *,
    conversation_id: str,
    split: str,
    turn_index: int,
) -> MemoryEntry | None:
    content = str(turn.get("content") or "").strip()
    if not content:
        return None
    role = str(turn.get("role") or "user")
    if role not in {"user", "assistant"}:
        role = "user" if role.lower() in {"human", "user"} else "assistant"
    pair_index = turn_index // _INGEST_CHUNK_SIZE
    time_anchor = turn.get("time_anchor")
    metadata: dict[str, Any] = {
        "source": "beam",
        "source_benchmark": "beam",
        "source_dataset": BEAM_DATASET_NAME,
        "source_split": split,
        "token_scale": split,
        "conversation_id": conversation_id,
        "turn_id": turn.get("id"),
        "turn_index": turn_index,
        "ingest_group_id": f"beam:{conversation_id}:{pair_index}",
        "ingest_order": turn_index % _INGEST_CHUNK_SIZE,
        "ingest_scope_id": conversation_id,
        "ingest_role": role,
        "role": role,
    }
    if time_anchor is not None:
        metadata["timestamp"] = str(time_anchor)
        epoch = _time_anchor_to_epoch(time_anchor)
        if epoch is not None:
            metadata["ingest_timestamp_epoch"] = epoch
    return MemoryEntry(content=content, metadata=metadata)


def record_to_benchmark_sample(
    record: dict[str, Any],
    *,
    config: BeamBenchmarkConfig,
) -> BenchmarkSample:
    conversation_id = str(record.get("conversation_id") or "")
    if not conversation_id:
        raise ValueError("BEAM sample is missing a `conversation_id`.")
    split = config.split
    questions = extract_probing_questions(record)
    if config.limit_stimuli is not None:
        questions = questions[: config.limit_stimuli]
    if not questions:
        raise ValueError(
            f"BEAM conversation {conversation_id!r} has no probing questions."
        )

    memory_entries: list[MemoryEntry] = []
    turns_by_id: dict[str, str] = {}
    for turn_index, turn in enumerate(iter_flat_turns(record.get("chat"))):
        entry = _turn_to_memory_entry(
            turn,
            conversation_id=conversation_id,
            split=split,
            turn_index=turn_index,
        )
        if entry is None:
            continue
        memory_entries.append(entry)
        turn_id = turn.get("id")
        if turn_id is not None and entry.content:
            turns_by_id[str(turn_id)] = entry.content

    stimuli: list[UserStimulus] = []
    gt_items: list[BeamQA] = []
    types_seen: list[str] = []
    for question in questions:
        question_text = str(
            question.get("question") or question.get("question_text") or ""
        ).strip()
        if not question_text:
            continue
        question_type = str(question.get("question_type") or "unknown")
        nuggets = extract_rubric_nuggets(question)
        source_ids = [
            str(value) for value in _flatten_ids(question.get("source_chat_ids"))
        ]
        markers: dict[str, Any] = {
            "question": question_text,
            "question_type": question_type,
            "token_scale": split,
        }
        if source_ids:
            markers["retrieval_evidence"] = {
                "match_key": "turn_id",
                "relevant_values": source_ids,
            }
            relevant = [
                turns_by_id[source_id]
                for source_id in source_ids
                if source_id in turns_by_id
            ]
            if relevant:
                markers["retrieval_evidence"]["relevant_contents"] = relevant
        stimuli.append(
            UserStimulus(
                content=config.answer_prompt.format(question=question_text),
                step_evaluation_markers=markers,
            )
        )
        gt_items.append(
            BeamQA(
                question=question_text,
                question_type=question_type,
                nuggets=nuggets,
                token_scale=split,
                difficulty=str(question.get("difficulty") or "unspecified"),
            )
        )
        if question_type not in types_seen:
            types_seen.append(question_type)

    if not stimuli:
        raise ValueError(
            f"BEAM conversation {conversation_id!r} has no answerable probing questions."
        )

    sample_id = f"BEAM-{split}-{conversation_id}"
    return BenchmarkSample(
        id=sample_id,
        name=f"BEAM {split} {conversation_id}",
        tags=["beam", f"split:{split}", *types_seen],
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=memory_entries)),
        execution_sequence=list(stimuli),
        evaluators=[config.evaluator],
        agent_conversation="independent",
        ground_truth=BeamQAGroundTruth(
            task_description=sample_id,
            qa=gt_items,
            token_scale=split,
        ),
    )


def _flatten_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(_flatten_ids(nested))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, str):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_ids(nested))
        return flattened
    return [str(value)]


__all__ = ["record_to_benchmark_sample"]
