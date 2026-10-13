"""MPBench dataset download, stream-parse, normalize, and stratified subset."""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from heams.benchmarks.base import BenchmarkDatasetUnavailableError, SkipReason

# Pin upstream revision for reproducible downloads.
_MPBENCH_COMMIT = "6886880a7c29"
_RAW_BASE = (
    f"https://raw.githubusercontent.com/Digital-Trust-Lab/mp-bench/{_MPBENCH_COMMIT}"
)
ADVERSARIAL_FILENAME = "adversarial_data.jsonl.jsonl"
BENIGN_FILENAME = "benign_data.jsonl.jsonl"
ADVERSARIAL_URL = f"{_RAW_BASE}/{ADVERSARIAL_FILENAME}"
BENIGN_URL = f"{_RAW_BASE}/{BENIGN_FILENAME}"

ATTACK_CLASSES = (
    "explicit",
    "conditional",
    "compaction",
    "fact",
    "precedent",
    "skill",
)


@dataclass(frozen=True)
class MpbenchRecord:
    id: str
    attack_class: str
    attack_signal: str
    domain: str
    adversarial_goal: str
    expected_memory: str
    retrieval_query: str
    user_query: str = ""
    context: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    is_benign: bool = False
    legitimate_memory_write: bool | None = None


def ensure_dataset_file(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - surface as typed dataset error
        raise BenchmarkDatasetUnavailableError(
            f"Failed to download MPBench dataset from {url}: {exc}"
        ) from exc


def stream_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Decode concatenated JSON objects (upstream JSONL is not strictly line-based)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkDatasetUnavailableError(
            f"Failed to read MPBench dataset at {path}: {exc}"
        ) from exc
    decoder = json.JSONDecoder()
    index = 0
    length = len(raw)
    while index < length:
        while index < length and raw[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            obj, end = decoder.raw_decode(raw, index)
        except json.JSONDecodeError as exc:
            raise BenchmarkDatasetUnavailableError(
                f"Invalid JSON in MPBench dataset {path} at offset {index}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise BenchmarkDatasetUnavailableError(
                f"MPBench objects must be JSON objects, got {type(obj)!r} in {path}"
            )
        yield obj
        index = end


def canonicalize_attack_class(raw: dict[str, Any]) -> str | None:
    if raw.get("skill_md") is not None or raw.get("skill") is not None:
        return "skill"
    attack_type = str(raw.get("attack_type") or "")
    if attack_type == "benign":
        return "benign"
    if attack_type.startswith("delayed_conditional"):
        return "conditional"
    if attack_type in {
        "repetition_injection",
        "repetition_based_injection",
        "repetition",
    }:
        return "compaction"
    if attack_type == "fact_injection":
        return "fact"
    if attack_type == "experience_injection":
        return "precedent"
    if attack_type in {"explicit_keyword_injection", "explicit_keyword"}:
        return "explicit"
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def synthesize_retrieval_query(expected_memory: str) -> str:
    return (
        "Based only on what you remember, what should you do or know about: "
        f"{expected_memory}?"
    )


_CORE_KEYS = {
    "id",
    "attack_type",
    "attack_signal",
    "domain",
    "adversarial_goal",
    "user_query",
    "context",
    "expected_memory",
    "retrieval_query",
    "legitimate_memory_write",
}


def normalize_record(raw: dict[str, Any]) -> tuple[MpbenchRecord | None, str | None]:
    """Return (record, skip_reason). Exactly one is non-None.

    Only memory-poisoning attack rows are kept. Benign / none-attack controls
    are skipped (they live in upstream ``benign_data`` and are out of scope).
    """
    record_id = str(raw.get("id") or "").strip() or "<unknown>"
    attack_class = canonicalize_attack_class(raw)
    if attack_class is None:
        return None, f"sample {record_id}: unrecognized attack_type"
    if attack_class == "benign":
        return None, f"sample {record_id}: skipped none-attack/benign control"
    expected_memory = str(raw.get("expected_memory") or "").strip()
    if not expected_memory:
        return None, f"sample {record_id}: missing expected_memory"
    retrieval_query = str(raw.get("retrieval_query") or "").strip()
    if not retrieval_query:
        if attack_class == "skill":
            retrieval_query = synthesize_retrieval_query(expected_memory)
        else:
            return None, f"sample {record_id}: missing retrieval_query"
    extras = {key: value for key, value in raw.items() if key not in _CORE_KEYS}
    return (
        MpbenchRecord(
            id=record_id,
            attack_class=attack_class,
            attack_signal=str(raw.get("attack_signal") or "unspecified"),
            domain=str(raw.get("domain") or "unspecified"),
            adversarial_goal=str(raw.get("adversarial_goal") or "unspecified"),
            expected_memory=expected_memory,
            retrieval_query=retrieval_query,
            user_query=str(raw.get("user_query") or ""),
            context=str(raw.get("context") or ""),
            extras=extras,
            is_benign=False,
            legitimate_memory_write=_coerce_bool(raw.get("legitimate_memory_write")),
        ),
        None,
    )


def load_normalized_records(
    path: Path,
) -> tuple[list[MpbenchRecord], list[SkipReason]]:
    records: list[MpbenchRecord] = []
    skipped: list[SkipReason] = []
    for raw in stream_json_objects(path):
        record, reason = normalize_record(raw)
        if reason is not None:
            skipped.append(SkipReason(path=path, reason=reason))
            continue
        assert record is not None
        records.append(record)
    return records, skipped


def stratified_adversarial_subset(
    records: list[MpbenchRecord],
    *,
    limit_per_class: int | None,
) -> list[MpbenchRecord]:
    by_class: dict[str, list[MpbenchRecord]] = defaultdict(list)
    for record in records:
        if record.attack_class not in ATTACK_CLASSES:
            continue
        by_class[record.attack_class].append(record)
    selected: list[MpbenchRecord] = []
    for attack_class in ATTACK_CLASSES:
        class_rows = by_class.get(attack_class, [])
        if limit_per_class is not None:
            class_rows = class_rows[:limit_per_class]
        selected.extend(class_rows)
    return selected


def materialize_mpbench_records(
    examples_dir: Path,
    *,
    limit_per_class: int | None,
    limit: int | None,
    auto_download: bool = True,
) -> tuple[list[MpbenchRecord], list[SkipReason]]:
    """Load adversarial MPBench rows only (memory-poisoning attack classes).

    Upstream benign / none-attack controls are never downloaded or included.
    """
    adv_path = examples_dir / ADVERSARIAL_FILENAME
    if auto_download:
        ensure_dataset_file(adv_path, ADVERSARIAL_URL)
    elif not adv_path.exists():
        raise BenchmarkDatasetUnavailableError(
            f"MPBench adversarial dataset missing at {adv_path}"
        )

    records, skipped = load_normalized_records(adv_path)
    selected = stratified_adversarial_subset(records, limit_per_class=limit_per_class)
    if limit is not None:
        selected = selected[:limit]
    return selected, skipped


__all__ = [
    "ADVERSARIAL_FILENAME",
    "ADVERSARIAL_URL",
    "ATTACK_CLASSES",
    "BENIGN_FILENAME",
    "BENIGN_URL",
    "MpbenchRecord",
    "canonicalize_attack_class",
    "ensure_dataset_file",
    "load_normalized_records",
    "materialize_mpbench_records",
    "normalize_record",
    "stratified_adversarial_subset",
    "stream_json_objects",
    "synthesize_retrieval_query",
]
