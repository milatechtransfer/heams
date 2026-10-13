"""Project LOCOMO JSON records into shared BenchmarkSample instances."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from heams.benchmarks.locomo.judge import (
    LocomoLlmJudgeConfig,
    LocomoQA,
    LocomoQAGroundTruth,
)
from heams.benchmarks.sample import BenchmarkSample
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

if TYPE_CHECKING:
    from heams.benchmarks.locomo.config import LocomoBenchmarkConfig

_SESSION_KEY_PATTERN = re.compile(r"^session_(\d+)$")


def _session_sort_key(key: str) -> int:
    match = _SESSION_KEY_PATTERN.match(key)
    if match is None:
        raise ValueError(f"Invalid LOCOMO session key: {key}")
    return int(match.group(1))


def _locomo_date_to_epoch(date_str: str | None) -> int | None:
    if not date_str:
        return None
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except (ValueError, TypeError):
            continue
    return None


def _iter_sessions(
    conversation: dict[str, Any],
) -> Iterable[tuple[str, str | None, list[dict[str, Any]]]]:
    session_keys = [key for key in conversation if _SESSION_KEY_PATTERN.match(key)]
    session_keys.sort(key=_session_sort_key)
    for session_key in session_keys:
        turns = conversation.get(session_key) or []
        timestamp = conversation.get(f"{session_key}_date_time")
        yield session_key, timestamp, turns


def _turn_to_memory_entry(
    turn: dict[str, Any],
    session_id: str,
    timestamp: str | None,
    speaker_a: str,
    turn_index: int,
) -> MemoryEntry | None:
    text = turn.get("text")
    if not text:
        return None

    speaker = turn.get("speaker")
    dia_id = turn.get("dia_id")
    metadata: dict[str, Any] = {
        "source": "locomo",
        "source_benchmark": "locomo",
        "session_id": session_id,
        "ingest_group_id": f"locomo:{session_id}",
        "ingest_order": turn_index,
        "ingest_role": "user" if speaker == speaker_a else "assistant",
    }
    if timestamp is not None:
        metadata["timestamp"] = timestamp
        epoch = _locomo_date_to_epoch(timestamp)
        if epoch is not None:
            metadata["ingest_timestamp_epoch"] = epoch
    if speaker is not None:
        metadata["speaker"] = speaker
        metadata["ingest_speaker_name"] = speaker
    if dia_id is not None:
        metadata["dia_id"] = dia_id

    return MemoryEntry(content=text, metadata=metadata)


def build_locomo_answer_prompt(
    *,
    question: str,
    category: Any | None = None,
    answer_prompt: str,
    answer_prompt_cat5: str,
) -> str:
    template = answer_prompt_cat5 if str(category) == "5" else answer_prompt
    return template.format(question=question)


def _qa_to_stimulus_and_gt(
    qa: dict[str, Any],
    dialogue_text_by_dia_id: dict[str, str],
    *,
    answer_prompt: str,
    answer_prompt_cat5: str,
) -> tuple[UserStimulus, LocomoQA] | None:
    question = qa.get("question")
    answer = qa.get("answer")
    if not question or not answer:
        return None

    evidence_dia_ids = [str(value) for value in qa.get("evidence") or []]
    relevant_contents = [
        dialogue_text_by_dia_id[dia_id]
        for dia_id in evidence_dia_ids
        if dia_id in dialogue_text_by_dia_id
    ]
    markers: dict[str, Any] = {
        "expected_answer": answer,
        "question": question,
        "evidence_dia_ids": evidence_dia_ids,
    }
    if evidence_dia_ids:
        markers["retrieval_evidence"] = {
            "match_key": "dia_id",
            "relevant_values": evidence_dia_ids,
        }
        if relevant_contents:
            markers["retrieval_evidence"]["relevant_contents"] = relevant_contents
    category = qa.get("category")
    if category is not None:
        markers["category"] = category

    stimulus = UserStimulus(
        content=build_locomo_answer_prompt(
            question=str(question),
            category=category,
            answer_prompt=answer_prompt,
            answer_prompt_cat5=answer_prompt_cat5,
        ),
        step_evaluation_markers=markers,
    )
    gt_item = LocomoQA(
        question=str(question),
        expected_answer=str(answer),
        evidence_dia_ids=evidence_dia_ids,
        evidence_texts=relevant_contents,
        category=category,
    )
    return stimulus, gt_item


def record_to_benchmark_sample(
    sample: dict[str, Any],
    *,
    config: LocomoBenchmarkConfig,
    evaluator: LocomoLlmJudgeConfig | None = None,
    limit_stimuli: int | None = None,
) -> BenchmarkSample:
    """Convert one LOCOMO JSON sample into a shared BenchmarkSample."""
    sample_id = str(sample.get("sample_id") or sample.get("id") or "")
    if not sample_id:
        raise ValueError("LOCOMO sample is missing a `sample_id`.")

    conversation = sample.get("conversation") or {}
    speaker_a = str(conversation.get("speaker_a") or "")
    if not speaker_a:
        for _, _, turns in _iter_sessions(conversation):
            if turns:
                first_speaker = turns[0].get("speaker")
                if first_speaker:
                    speaker_a = str(first_speaker)
                break

    memory_entries: list[MemoryEntry] = []
    dialogue_text_by_dia_id: dict[str, str] = {}
    for session_id, timestamp, turns in _iter_sessions(conversation):
        for turn_index, turn in enumerate(turns):
            entry = _turn_to_memory_entry(
                turn, session_id, timestamp, speaker_a, turn_index
            )
            if entry is not None:
                entry.metadata.setdefault("ingest_scope_id", sample_id)
                memory_entries.append(entry)
                dia_id = entry.metadata.get("dia_id")
                if isinstance(dia_id, str) and dia_id and entry.content:
                    dialogue_text_by_dia_id[dia_id] = entry.content

    qa_items = sample.get("qa") or []
    stimuli: list[UserStimulus] = []
    gt_items: list[LocomoQA] = []
    categories: list[str] = []
    seen_categories: set[str] = set()
    for qa in qa_items:
        projected = _qa_to_stimulus_and_gt(
            qa,
            dialogue_text_by_dia_id,
            answer_prompt=config.answer_prompt,
            answer_prompt_cat5=config.answer_prompt_cat5,
        )
        if projected is None:
            continue
        stimulus, gt_item = projected
        stimuli.append(stimulus)
        gt_items.append(gt_item)
        category = qa.get("category")
        if category is not None and category not in seen_categories:
            seen_categories.add(category)
            categories.append(str(category))

    effective_limit = (
        limit_stimuli if limit_stimuli is not None else config.limit_stimuli
    )
    if effective_limit is not None:
        stimuli = stimuli[:effective_limit]
        gt_items = gt_items[:effective_limit]

    if not stimuli:
        raise ValueError(
            f"LOCOMO sample {sample_id!r} has no answerable QA stimuli after filtering."
        )

    return BenchmarkSample(
        id=sample_id,
        name=f"LOCOMO {sample_id}",
        tags=["locomo", *categories],
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=memory_entries)),
        execution_sequence=list(stimuli),
        evaluators=[evaluator or config.evaluator],
        ground_truth=LocomoQAGroundTruth(
            task_description=f"LOCOMO {sample_id}",
            qa=gt_items,
        ),
        agent_conversation=config.agent_conversation,
    )
