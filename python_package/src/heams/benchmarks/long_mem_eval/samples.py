"""Project LongMemEval records into shared BenchmarkSample instances."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from heams.benchmarks.long_mem_eval.dataset import split_name_from_path
from heams.benchmarks.sample import BenchmarkSample
from heams.evaluation.scenarios.long_mem_eval_llm_judge import (
    LongMemEvalQA,
    LongMemEvalQAGroundTruth,
)
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus

if TYPE_CHECKING:
    from pathlib import Path

    from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig


_SESSION_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_session_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in _SESSION_DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _date_to_epoch(date_str: str | None) -> int | None:
    parsed = _parse_session_date(date_str)
    return int(parsed.timestamp()) if parsed is not None else None


def _iter_sorted_sessions(
    *,
    session_ids: list[str],
    session_dates: list[str],
    sessions: list[Any],
) -> list[tuple[str, str | None, list[dict[str, Any]]]]:
    paired: list[tuple[int, str, str | None, list[dict[str, Any]]]] = []
    for index, turns in enumerate(sessions):
        session_id = (
            session_ids[index] if index < len(session_ids) else f"session_{index}"
        )
        session_date = session_dates[index] if index < len(session_dates) else None
        paired.append((index, session_id, session_date, list(turns or [])))

    def sort_key(
        item: tuple[int, str, str | None, list[dict[str, Any]]],
    ) -> tuple[int, int, str, int]:
        index, _, session_date, _ = item
        parsed = _date_to_epoch(session_date)
        if parsed is not None:
            return (0, parsed, session_date or "", index)
        return (1, 0, session_date or "", index)

    paired.sort(key=sort_key)
    return [
        (session_id, session_date, turns)
        for _, session_id, session_date, turns in paired
    ]


def _turn_to_memory_entry(
    *,
    turn: dict[str, Any],
    question_id: str,
    session_id: str,
    session_date: str | None,
    session_index: int,
    turn_index: int,
) -> MemoryEntry | None:
    content = turn.get("content")
    if not content:
        return None

    role = turn.get("role")
    pair_index = turn_index // 2
    metadata: dict[str, Any] = {
        "source": "long_mem_eval",
        "source_benchmark": "long_mem_eval",
        "session_id": session_id,
        "session_index": session_index,
        "turn_index": turn_index,
        "ingest_group_id": f"long_mem_eval:{question_id}:{session_id}:{pair_index}",
        "ingest_order": turn_index % 2,
        "ingest_scope_id": question_id,
    }
    if role is not None:
        metadata["role"] = role
        metadata["ingest_role"] = role
    if session_date is not None:
        metadata["timestamp"] = session_date
        parsed_session_date = _parse_session_date(session_date)
        if parsed_session_date is not None:
            metadata["ingest_timestamp_epoch"] = int(parsed_session_date.timestamp())
            metadata["temporal_received_at"] = parsed_session_date.isoformat().replace(
                "+00:00", "Z"
            )
            metadata["temporal_source"] = "long_mem_eval.session_date"
    return MemoryEntry(content=str(content), metadata=metadata)


def _collect_relevant_contents(
    relevant_entries: list[tuple[MemoryEntry, bool]],
) -> list[str]:
    answer_contents: list[str] = []
    session_contents: list[str] = []
    seen_contents: set[str] = set()
    for entry, has_answer in relevant_entries:
        content = entry.content.strip()
        if not content or content in seen_contents:
            continue
        seen_contents.add(content)
        if has_answer:
            answer_contents.append(content)
        else:
            session_contents.append(content)
    return answer_contents or session_contents


def record_to_benchmark_sample(
    record: dict[str, Any],
    *,
    config: LongMemEvalBenchmarkConfig,
    dataset_path: Path,
) -> BenchmarkSample:
    """Convert one LongMemEval JSON record into a shared BenchmarkSample."""
    question_id = str(record.get("question_id") or "")
    if not question_id:
        raise ValueError("LongMemEval sample is missing a `question_id`.")
    question = record.get("question")
    answer = record.get("answer")
    if not question or answer is None:
        raise ValueError(
            f"LongMemEval sample '{question_id}' is missing `question` or `answer`."
        )

    answer_session_id_list = [
        str(value) for value in record.get("answer_session_ids") or []
    ]
    answer_session_ids = set(answer_session_id_list)
    session_ids = [str(value) for value in record.get("haystack_session_ids") or []]
    session_dates = [str(value) for value in record.get("haystack_dates") or []]
    sessions = record.get("haystack_sessions") or []

    memory_entries: list[MemoryEntry] = []
    relevant_entries: list[tuple[MemoryEntry, bool]] = []
    for session_index, (session_id, session_date, turns) in enumerate(
        _iter_sorted_sessions(
            session_ids=session_ids,
            session_dates=session_dates,
            sessions=sessions,
        )
    ):
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            entry = _turn_to_memory_entry(
                turn=turn,
                question_id=question_id,
                session_id=session_id,
                session_date=session_date,
                session_index=session_index,
                turn_index=turn_index,
            )
            if entry is not None:
                memory_entries.append(entry)
                if session_id in answer_session_ids:
                    relevant_entries.append((entry, turn.get("has_answer") is True))

    question_type = record.get("question_type")
    question_date = record.get("question_date")
    is_abstention = question_id.endswith("_abs")
    markers: dict[str, Any] = {
        "expected_answer": answer,
        "question": question,
        "question_id": question_id,
        "answer_session_ids": answer_session_id_list,
        "is_abstention": is_abstention,
    }
    if answer_session_id_list:
        markers["retrieval_evidence"] = {
            "match_key": "session_id",
            "relevant_values": answer_session_id_list,
        }
        relevant_contents = _collect_relevant_contents(relevant_entries)
        if relevant_contents:
            markers["retrieval_evidence"]["relevant_contents"] = relevant_contents
    if question_type is not None:
        markers["question_type"] = question_type
    if question_date is not None:
        markers["question_date"] = question_date

    split_name = split_name_from_path(dataset_path)
    tags = ["long_mem_eval"]
    if split_name is not None:
        tags.append(f"split:{split_name}")
    if question_type is not None:
        tags.append(str(question_type))
    if is_abstention:
        tags.append("abstention")

    task_name = f"LongMemEval {question_id}"
    return BenchmarkSample(
        id=question_id,
        name=task_name,
        tags=tags,
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=memory_entries)),
        execution_sequence=[
            UserStimulus(
                content=config.answer_prompt.format(
                    question=question,
                    question_date=question_date or "unknown",
                ),
                step_evaluation_markers=markers,
            )
        ],
        evaluators=[config.evaluator],
        ground_truth=LongMemEvalQAGroundTruth(
            task_description=task_name,
            qa=[
                LongMemEvalQA(
                    question_id=question_id,
                    question=str(question),
                    expected_answer=str(answer),
                    question_type=question_type,
                    question_date=question_date,
                    answer_session_ids=answer_session_id_list,
                    is_abstention=is_abstention,
                )
            ],
        ),
    )
