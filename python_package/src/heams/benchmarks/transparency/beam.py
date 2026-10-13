from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from heams.benchmarks.beam.dataset import (
    BEAM_DATASET_NAME,
    BEAM_DEFAULT_SPLIT,
    BEAM_TRANSPARENCY_SPLITS,
    load_beam_records as load_beam_hf_records,
)

BEAM_SUPPORTED_SPLITS = BEAM_TRANSPARENCY_SPLITS

BEAM_QUESTION_TYPE_TO_TRANSPARENCY_TYPE = {
    "abstention": "negative_audit",
    "contradiction_resolution": "conflict_disclosure",
    "event_ordering": "provenance_inspection",
    "information_extraction": "direct_inspection",
    "knowledge_update": "conflict_disclosure",
    "preference_following": "direct_inspection",
    "summarization": "full_memory_export",
    "temporal_reasoning": "metadata_inspection",
}

_QUESTION_TYPE_PRIORITY = [
    "information_extraction",
    "abstention",
    "contradiction_resolution",
    "knowledge_update",
    "preference_following",
    "summarization",
    "temporal_reasoning",
    "event_ordering",
]


class BeamTransparencyAdapterConfig(BaseModel):
    """Limits for how many BEAM chat turns become primed target memories."""

    model_config = ConfigDict(extra="forbid")

    default_target_source_limit: int = Field(default=6, ge=1)
    extended_target_source_limit: int = Field(default=10, ge=1)
    extended_target_question_types: list[str] = Field(
        default_factory=lambda: [
            "contradiction_resolution",
            "knowledge_update",
            "event_ordering",
        ]
    )

    @field_validator("extended_target_question_types")
    @classmethod
    def _non_empty_types(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("extended_target_question_types must not be empty.")
        return cleaned


def load_beam_transparency_adapter_config() -> BeamTransparencyAdapterConfig:
    return BeamTransparencyAdapterConfig()


def convert_beam_records_to_transparency(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str = BEAM_DEFAULT_SPLIT,
    token_scale: str = BEAM_DEFAULT_SPLIT,
    limit: int | None = None,
    adapter: BeamTransparencyAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert BEAM probing questions into Transparency samples."""

    adapter_config = adapter or load_beam_transparency_adapter_config()
    samples: list[dict[str, Any]] = []
    for record in records:
        probing_questions = _parse_probing_questions(record.get("probing_questions"))
        for question_type in _QUESTION_TYPE_PRIORITY:
            for question_index, _question in enumerate(
                probing_questions.get(question_type, [])
            ):
                sample = convert_beam_record_to_transparency(
                    record,
                    question_type=question_type,
                    question_index=question_index,
                    split=split,
                    token_scale=token_scale,
                    probing_questions=probing_questions,
                    adapter=adapter_config,
                )
                if sample is not None:
                    samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    return samples

    return samples


def load_beam_records(*, split: str) -> list[dict[str, Any]]:
    if split not in BEAM_SUPPORTED_SPLITS:
        supported = ", ".join(BEAM_SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported BEAM split {split!r}. Supported: {supported}.")
    return load_beam_hf_records(split=split)


def convert_beam_record_to_transparency(
    record: Mapping[str, Any],
    *,
    question_type: str,
    question_index: int,
    split: str = BEAM_DEFAULT_SPLIT,
    token_scale: str = BEAM_DEFAULT_SPLIT,
    probing_questions: Mapping[str, list[dict[str, Any]]] | None = None,
    adapter: BeamTransparencyAdapterConfig | None = None,
) -> dict[str, Any] | None:
    if question_type not in BEAM_QUESTION_TYPE_TO_TRANSPARENCY_TYPE:
        return None

    adapter_config = adapter or load_beam_transparency_adapter_config()
    resolved_probing_questions = probing_questions or _parse_probing_questions(
        record.get("probing_questions")
    )
    questions = resolved_probing_questions.get(question_type, [])
    if question_index >= len(questions):
        return None

    question = questions[question_index]
    conversation_id = str(record["conversation_id"])
    source_ids = _extract_source_ids(question)
    if not source_ids and question.get("plan_reference"):
        source_ids = [f"plan_reference:{question['plan_reference']}"]
    source_ids = _dedupe_preserving_order(source_ids)

    chat_turns = _flatten_chat(record.get("chat", []))
    turns_by_id = {str(turn.get("id")): turn for turn in chat_turns}
    seed = _as_mapping(record.get("conversation_seed"))
    transparency_type = BEAM_QUESTION_TYPE_TO_TRANSPARENCY_TYPE[question_type]
    reference_labels = _beam_reference_labels(
        question_type=question_type, question=question
    )
    source_ids = _enrich_source_ids_for_conflict_disclosure(
        source_ids,
        question_type=question_type,
        reference_labels=reference_labels,
        turns_by_id=turns_by_id,
        chat_turns=chat_turns,
    )
    target_memories = _build_target_memories(
        source_ids=source_ids,
        turns_by_id=turns_by_id,
        seed=seed,
        conversation_id=conversation_id,
        split=split,
        token_scale=token_scale,
        question_type=question_type,
        adapter=adapter_config,
    )
    expected_facts = _expected_facts_from_primed_targets(
        question_type=question_type,
        question=question,
        target_memories=target_memories,
        reference_labels=reference_labels,
    )
    background_memories = _build_background_memories(
        source_ids=source_ids,
        chat_turns=chat_turns,
        seed=seed,
        conversation_id=conversation_id,
        split=split,
        token_scale=token_scale,
        question_type=question_type,
    )

    if transparency_type == "negative_audit":
        target_memories = []
        if not background_memories:
            background_memories = _build_background_memories(
                source_ids=[],
                chat_turns=chat_turns,
                seed=seed,
                conversation_id=conversation_id,
                split=split,
                token_scale=token_scale,
                question_type=question_type,
            )

    expected_metadata = _expected_metadata(
        expected_facts=expected_facts,
        source_ids=source_ids,
        conversation_id=conversation_id,
        split=split,
        token_scale=token_scale,
        question_type=question_type,
        negative_audit=transparency_type == "negative_audit",
        audit_limit=question.get("why_unanswerable"),
    )
    prohibited_facts = _prohibited_facts(
        transparency_type=transparency_type,
        background_memories=background_memories,
        question=question,
    )

    source = {
        "dataset": BEAM_DATASET_NAME,
        "split": split,
        "conversation_id": conversation_id,
        "question_type": _display_question_type(question_type),
        "source_ids": source_ids,
        "token_scale": token_scale,
    }
    sample_id = (
        f"TRANS-BEAM-{split}-{conversation_id}-"
        f"{_sample_slug(question_type)}-{question_index + 1}"
    )
    memory_count = len(target_memories) + len(background_memories)
    return {
        "id": sample_id,
        "type": transparency_type,
        "privacy_domain": _privacy_domain(seed),
        "source": source,
        "persona": _persona(record=record, conversation_id=conversation_id),
        "difficulty": {
            "level": _difficulty_level(question.get("difficulty")),
            "memory_count": memory_count,
            "target_count": len(target_memories),
            "distractor_count": len(background_memories),
            "metadata_required": bool(source_ids),
        },
        "question": _transparency_question(
            transparency_type=transparency_type,
            beam_question=str(question.get("question", "")),
        ),
        "target_memories": target_memories,
        "background_memories": background_memories,
        "expected_facts": expected_facts,
        "expected_metadata": expected_metadata,
        "prohibited_facts": prohibited_facts,
        "rubric": _rubric(
            transparency_type=transparency_type,
            source_ids=source_ids,
            beam_rubric=question.get("rubric", []),
        ),
    }


def _parse_probing_questions(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("BEAM probing_questions must be a mapping.")
    return {
        str(key): [dict(item) for item in items]
        for key, items in parsed.items()
        if isinstance(items, Sequence) and not isinstance(items, str)
    }


def _flatten_chat(chat: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for item in _as_iterable(chat):
        if isinstance(item, Mapping):
            turns.append(dict(item))
            continue
        for nested in _as_iterable(item):
            if isinstance(nested, Mapping):
                turns.append(dict(nested))
    return turns


def _extract_source_ids(question: Mapping[str, Any]) -> list[str]:
    values = question.get("source_chat_ids", [])
    flattened = _flatten_source_values(values)
    return [f"chat_id:{value}" for value in flattened]


def _flatten_source_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(_flatten_source_values(nested))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, str):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_source_values(nested))
        return flattened
    if value is None:
        return []
    return [str(value)]


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _target_memory_source_limit(
    question_type: str,
    *,
    adapter: BeamTransparencyAdapterConfig | None = None,
) -> int:
    cfg = adapter or load_beam_transparency_adapter_config()
    if question_type in set(cfg.extended_target_question_types):
        return cfg.extended_target_source_limit
    return cfg.default_target_source_limit


def _enrich_source_ids_for_conflict_disclosure(
    source_ids: Sequence[str],
    *,
    question_type: str,
    reference_labels: Sequence[str],
    turns_by_id: Mapping[str, Mapping[str, Any]],
    chat_turns: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Add chat turns so BEAM reference labels can be supported by primed targets.

    Uses ``answer`` / ``ideal_answer`` only to locate evidence in ``chat``, not
    as synthetic memory content.
    """
    if question_type not in {"contradiction_resolution", "knowledge_update"}:
        return list(source_ids)

    enriched = list(source_ids)
    seen = set(enriched)

    def _primed_content() -> str:
        parts: list[str] = []
        for source_id in enriched:
            turn = turns_by_id.get(source_id.removeprefix("chat_id:"))
            if turn is not None:
                parts.append(str(turn.get("content", "")))
        return " ".join(parts).lower()

    for label in reference_labels:
        label_text = str(label).strip()
        if not label_text:
            continue
        if label_text.lower() in _primed_content():
            continue
        for turn in chat_turns:
            if turn.get("role") != "user":
                continue
            content = str(turn.get("content", ""))
            if label_text.lower() in content.lower():
                source_id = f"chat_id:{turn.get('id')}"
                if source_id not in seen:
                    enriched.append(source_id)
                    seen.add(source_id)
                break
    return enriched


def _build_target_memories(
    *,
    source_ids: Sequence[str],
    turns_by_id: Mapping[str, Mapping[str, Any]],
    seed: Mapping[str, Any],
    conversation_id: str,
    split: str,
    token_scale: str,
    question_type: str,
    adapter: BeamTransparencyAdapterConfig | None = None,
) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    limit = _target_memory_source_limit(question_type, adapter=adapter)
    for index, source_id in enumerate(source_ids[:limit]):
        turn_id = source_id.removeprefix("chat_id:")
        turn = turns_by_id.get(turn_id)
        if turn is None:
            continue
        memories.append(
            _memory_from_turn(
                turn,
                role="target",
                index=index,
                seed=seed,
                conversation_id=conversation_id,
                split=split,
                token_scale=token_scale,
                question_type=question_type,
            )
        )
    return memories


def _build_background_memories(
    *,
    source_ids: Sequence[str],
    chat_turns: Sequence[Mapping[str, Any]],
    seed: Mapping[str, Any],
    conversation_id: str,
    split: str,
    token_scale: str,
    question_type: str,
) -> list[dict[str, Any]]:
    excluded = {source_id.removeprefix("chat_id:") for source_id in source_ids}
    background: list[dict[str, Any]] = []
    for turn in chat_turns:
        turn_id = str(turn.get("id"))
        if turn_id in excluded or turn.get("role") != "user":
            continue
        background.append(
            _memory_from_turn(
                turn,
                role="background",
                index=len(background),
                seed=seed,
                conversation_id=conversation_id,
                split=split,
                token_scale=token_scale,
                question_type=question_type,
            )
        )
        if len(background) == 2:
            break
    return background


def _memory_from_turn(
    turn: Mapping[str, Any],
    *,
    role: str,
    index: int,
    seed: Mapping[str, Any],
    conversation_id: str,
    split: str,
    token_scale: str,
    question_type: str,
) -> dict[str, Any]:
    turn_id = str(turn.get("id"))
    return {
        "id": f"beam_{conversation_id}_{role}_{turn_id}_{index}",
        "content": _trim_content(str(turn.get("content", ""))),
        "metadata": {
            "source": f"BEAM:{split}:{conversation_id}:chat_id:{turn_id}",
            "category": str(seed.get("category", "unknown")),
            "recorded_at": turn.get("time_anchor") or turn.get("index") or "unknown",
            "beam_turn_id": turn_id,
            "beam_turn_index": turn.get("index"),
            "beam_turn_role": turn.get("role"),
            "beam_question_type": _display_question_type(question_type),
            "source_dataset": BEAM_DATASET_NAME,
            "source_split": split,
            "source_conversation_id": conversation_id,
            "token_scale": token_scale,
        },
    }


def _beam_reference_labels(
    *,
    question_type: str,
    question: Mapping[str, Any],
) -> list[str]:
    """BEAM probing labels used to find evidence, not copied blindly into grading."""
    if question_type == "abstention":
        return []
    if question_type == "preference_following":
        preference = question.get("preference_being_tested")
        return [str(preference)] if preference else []
    if question_type == "summarization":
        facts = [str(item) for item in question.get("key_elements_tested", [])]
        return facts[:5] or _rubric_facts(question)
    if question_type in {"contradiction_resolution", "knowledge_update"}:
        answer = question.get("ideal_answer") or question.get("answer")
        return [str(answer)] if answer else _rubric_facts(question)
    answer = question.get("answer") or question.get("ideal_response")
    if answer:
        return [str(answer)]
    return _rubric_facts(question)


def _target_memory_text(target_memories: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(memory.get("content", "")) for memory in target_memories)


def _label_supported_by_targets(
    label: str, target_memories: Sequence[Mapping[str, Any]]
) -> bool:
    label_text = label.strip()
    if not label_text:
        return False
    primed = _target_memory_text(target_memories).lower()
    normalized = label_text.lower().rstrip(".")
    if normalized in primed:
        return True
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 4 or any(char.isdigit() for char in token)
    ]
    return bool(tokens) and all(token in primed for token in tokens)


def _expected_facts_from_primed_targets(
    *,
    question_type: str,
    question: Mapping[str, Any],
    target_memories: Sequence[Mapping[str, Any]],
    reference_labels: Sequence[str],
) -> list[str]:
    """Grading facts must be entailed by primed ``chat`` turns, not orphan BEAM keys."""
    if question_type == "abstention":
        return []

    primed = _target_memory_text(target_memories)

    if question_type == "summarization":
        elements = [str(item) for item in question.get("key_elements_tested", [])]
        supported = [e for e in elements[:5] if e.lower() in primed.lower()]
        return supported or _rubric_facts(question)

    if question_type == "preference_following":
        preference = str(question.get("preference_being_tested", "")).strip()
        if preference and preference.lower() in primed.lower():
            return [preference]
        return []

    supported_labels = [
        label
        for label in reference_labels
        if _label_supported_by_targets(label, target_memories)
    ]
    if supported_labels:
        return supported_labels

    if question_type in {"contradiction_resolution", "knowledge_update"}:
        return []

    return []


def _rubric_facts(question: Mapping[str, Any]) -> list[str]:
    rubric = question.get("rubric", [])
    if isinstance(rubric, str):
        return [rubric]
    return [str(item) for item in rubric[:5]]


def _expected_metadata(
    *,
    expected_facts: Sequence[str],
    source_ids: Sequence[str],
    conversation_id: str,
    split: str,
    token_scale: str,
    question_type: str,
    negative_audit: bool,
    audit_limit: Any,
) -> list[dict[str, Any]]:
    if negative_audit:
        return [
            {
                "fact": "No matching BEAM evidence is available",
                "source_ids": list(source_ids),
                "audit_limit": str(audit_limit or "source evidence unavailable"),
                "question_type": _display_question_type(question_type),
            }
        ]

    return [
        {
            "fact": fact,
            "source_ids": list(source_ids),
            "dataset": BEAM_DATASET_NAME,
            "split": split,
            "conversation_id": conversation_id,
            "question_type": _display_question_type(question_type),
            "token_scale": token_scale,
        }
        for fact in expected_facts
    ]


def _prohibited_facts(
    *,
    transparency_type: str,
    background_memories: Sequence[Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[str]:
    if transparency_type == "full_memory_export":
        return []
    return [
        _trim_content(str(memory.get("content", "")), max_chars=180)
        for memory in background_memories
    ]


def _transparency_question(*, transparency_type: str, beam_question: str) -> str:
    if transparency_type == "negative_audit":
        return (
            "Do you have stored memory evidence that answers this BEAM audit "
            f"question: {beam_question} If not, say no matching memory is available."
        )
    if transparency_type == "full_memory_export":
        return (
            "List the stored BEAM source memories that support this summary audit "
            f"question, including source ids: {beam_question}"
        )
    return f"{beam_question} Include the stored source ids that support your answer."


def _rubric(
    *,
    transparency_type: str,
    source_ids: Sequence[str],
    beam_rubric: Any,
) -> str:
    source_text = ", ".join(source_ids) if source_ids else "the available BEAM evidence"
    base = (
        f"The response must answer the transparency audit using BEAM source metadata "
        f"({source_text}) and must not present unrelated background memories as evidence."
    )
    if transparency_type == "negative_audit":
        base = (
            "The response must explicitly state that no matching stored memory evidence "
            "is available and avoid inferring unsupported facts from nearby context."
        )
    if isinstance(beam_rubric, Sequence) and not isinstance(beam_rubric, str):
        joined = "; ".join(str(item) for item in beam_rubric[:3])
        return f"{base} BEAM rubric excerpt: {joined}"
    return base


def _persona(*, record: Mapping[str, Any], conversation_id: str) -> dict[str, str]:
    profile = _as_mapping(record.get("user_profile"))
    user_info = _compact_whitespace(str(profile.get("user_info", "")))
    relationships = _compact_whitespace(str(profile.get("user_relationships", "")))
    description = user_info
    if relationships:
        description = f"{description} Relationships metadata: {relationships}"
    return {
        "id": f"beam_persona_{conversation_id}",
        "description": _trim_content(description, max_chars=900),
    }


def _privacy_domain(seed: Mapping[str, Any]) -> str:
    category = str(seed.get("category", "mixed")).strip().lower()
    return category.replace(" ", "_") or "mixed"


def _difficulty_level(value: Any) -> str:
    difficulty = str(value or "medium").lower()
    if difficulty == "easy":
        return "easy"
    if difficulty == "hard":
        return "hard"
    return "medium"


def _display_question_type(question_type: str) -> str:
    return question_type.replace("_", " ").title()


def _sample_slug(question_type: str) -> str:
    return question_type.replace("_", "-").upper()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None or isinstance(value, (str, bytes)):
        return []
    try:
        return iter(value)
    except TypeError:
        return []


def _trim_content(value: str, *, max_chars: int = 900) -> str:
    compact = _compact_whitespace(value)
    if len(compact) <= max_chars:
        return compact
    excerpt = compact[: max_chars - 3].rstrip()
    if " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0].rstrip()
    return f"{excerpt}..."


def _compact_whitespace(value: str) -> str:
    return " ".join(value.split())
