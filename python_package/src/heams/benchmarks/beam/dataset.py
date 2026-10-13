"""BEAM HuggingFace download, chat flattening, and probing-question parse."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BEAM_DATASET_NAME = "BEAM"
BEAM_DEFAULT_SPLIT = "100K"
BEAM_SPLITS = ("100K", "500K", "1M", "10M")
BEAM_TRANSPARENCY_SPLITS = ("100K", "500K", "1M")
BEAM_HF_SOURCES: dict[str, tuple[str, str]] = {
    "100K": ("Mohammadta/BEAM", "100K"),
    "500K": ("Mohammadta/BEAM", "500K"),
    "1M": ("Mohammadta/BEAM", "1M"),
    "10M": ("Mohammadta/BEAM-10M", "10M"),
}
BEAM_QUESTION_TYPES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def load_beam_records(
    *, split: str, local_path: Path | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load BEAM conversations for one token-length split.

    ``local_path`` is a JSON list of conversation dicts (unit tests). Otherwise
    the split is downloaded from HuggingFace via ``datasets``.
    ``limit`` stops after that many conversations so 10M smokes do not parse
    unused chats.
    """
    if local_path is not None:
        records = [
            _normalize_record(record) for record in _load_local_records(local_path)
        ]
        return records[:limit] if limit is not None else records
    if split not in BEAM_HF_SOURCES:
        supported = ", ".join(BEAM_SPLITS)
        raise ValueError(f"Unsupported BEAM split {split!r}. Supported: {supported}.")
    repo_id, hf_split = BEAM_HF_SOURCES[split]
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on local setup
        raise ValueError(
            "Loading BEAM requires the datasets package. Run through uv from "
            "python_package, for example: uv run heams run --config "
            "configs/runs/testing/beam_smoke.yaml"
        ) from error
    dataset = load_dataset(repo_id, split=hf_split)
    records: list[dict[str, Any]] = []
    for item in dataset:
        records.append(_normalize_record(dict(item)))
        if limit is not None and len(records) >= limit:
            break
    return records


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("BEAM local_path must be a JSON list of conversation objects.")
    return payload


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["probing_questions"] = parse_probing_questions(
        normalized.get("probing_questions")
    )
    return normalized


def parse_probing_questions(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("BEAM probing_questions must be a mapping.")
    questions: dict[str, list[dict[str, Any]]] = {}
    for key, items in parsed.items():
        if isinstance(items, Mapping):
            questions[str(key)] = [dict(items)]
            continue
        if not isinstance(items, Sequence) or isinstance(items, str):
            continue
        questions[str(key)] = [
            {"question": item} if isinstance(item, str) else dict(item)
            for item in items
        ]
    return questions


def extract_probing_questions(conversation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten probing questions in BEAM ability order (typically 20 per chat)."""
    probing = parse_probing_questions(conversation.get("probing_questions", {}))
    questions: list[dict[str, Any]] = []
    for question_type in BEAM_QUESTION_TYPES:
        for question in probing.get(question_type, []):
            payload = dict(question)
            payload["question_type"] = question_type
            questions.append(payload)
    return questions


def extract_rubric_nuggets(question: Mapping[str, Any]) -> list[str]:
    rubric = question.get("rubric", {})
    if isinstance(rubric, Mapping):
        nuggets = rubric.get("nuggets", [])
        return [
            str(item.get("description", item))
            if isinstance(item, Mapping)
            else str(item)
            for item in nuggets
            if str(
                item.get("description", item) if isinstance(item, Mapping) else item
            ).strip()
        ]
    if isinstance(rubric, str):
        text = rubric.strip()
        return [text] if text else []
    if isinstance(rubric, Sequence):
        return [str(item) for item in rubric if str(item).strip()]
    return []


def _unwrap_batch_dicts(
    batch_dicts: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    for batch in batch_dicts:
        turns = batch.get("turns", [])
        flat_turns: list[dict[str, Any]] = []
        for item in turns:
            if isinstance(item, list):
                flat_turns.extend(
                    dict(turn) for turn in item if isinstance(turn, Mapping)
                )
            elif isinstance(item, Mapping):
                flat_turns.append(dict(item))
        batches.append(flat_turns)
    return batches


def parse_beam_chat(chat_data: Any) -> list[list[dict[str, Any]]]:
    """Parse BEAM chat into batches of turn dicts (100K–1M and 10M layouts)."""
    if not chat_data:
        return []
    if not isinstance(chat_data, list):
        return []

    first = chat_data[0]
    if isinstance(first, Mapping) and "turns" in first:
        return _unwrap_batch_dicts(chat_data)

    if isinstance(first, Mapping):
        sample_val = next(iter(first.values()), None) if first else None
        is_plan_format = (
            isinstance(sample_val, list)
            and sample_val
            and isinstance(sample_val[0], Mapping)
            and "turns" in sample_val[0]
        )
        if is_plan_format:
            batches: list[list[dict[str, Any]]] = []
            for session in chat_data:
                if not isinstance(session, Mapping):
                    continue
                plan_keys = sorted(
                    session.keys(),
                    key=lambda key: (
                        int(str(key).split("-")[-1])
                        if str(key).split("-")[-1].isdigit()
                        else 0
                    ),
                )
                for plan_key in plan_keys:
                    plan_batches = session[plan_key]
                    if not isinstance(plan_batches, list):
                        continue
                    batches.extend(_unwrap_batch_dicts(plan_batches))
            return batches
        if "role" in first or "content" in first:
            return [[dict(turn) for turn in chat_data if isinstance(turn, Mapping)]]
        return []

    if isinstance(first, list):
        return [
            [dict(turn) for turn in batch if isinstance(turn, Mapping)]
            for batch in chat_data
            if isinstance(batch, list)
        ]
    return []


def iter_flat_turns(chat_data: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for batch in parse_beam_chat(chat_data):
        turns.extend(batch)
    return turns


__all__ = [
    "BEAM_DATASET_NAME",
    "BEAM_DEFAULT_SPLIT",
    "BEAM_HF_SOURCES",
    "BEAM_QUESTION_TYPES",
    "BEAM_SPLITS",
    "BEAM_TRANSPARENCY_SPLITS",
    "extract_probing_questions",
    "extract_rubric_nuggets",
    "iter_flat_turns",
    "load_beam_records",
    "parse_beam_chat",
    "parse_probing_questions",
]
