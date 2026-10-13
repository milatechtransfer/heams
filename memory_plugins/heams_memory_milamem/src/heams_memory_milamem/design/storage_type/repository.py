import json
import os
import tempfile
from typing import Any

from pydantic import ValidationError

from .canonical_projection import CanonicalMemoryRecord
from .raw_records import RawMemoryRecord
from .store_state import CanonicalMemoryStoreState, RawMemoryStoreState


def load_json_object(path: str) -> Any | None:
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    return raw


def load_raw_store(path: str, *, schema_version: int) -> RawMemoryStoreState:
    raw = load_json_object(path)
    if raw is None:
        return RawMemoryStoreState(schema_version=schema_version)

    if isinstance(raw, list):
        records: list[RawMemoryRecord] = []
        for item in raw:
            try:
                records.append(RawMemoryRecord.model_validate(item))
            except ValidationError:
                continue
        return RawMemoryStoreState(
            schema_version=schema_version,
            records=records,
        )

    if not isinstance(raw, dict):
        return RawMemoryStoreState(schema_version=schema_version)

    records: list[RawMemoryRecord] = []
    for item in raw.get("records", []):
        try:
            records.append(RawMemoryRecord.model_validate(item))
        except ValidationError:
            continue

    parsed_schema_version = raw.get("schema_version", schema_version)
    if not isinstance(parsed_schema_version, int):
        parsed_schema_version = schema_version

    return RawMemoryStoreState(
        schema_version=parsed_schema_version,
        records=records,
    )


def load_canonical_store(
    path: str,
    *,
    schema_version: int,
) -> CanonicalMemoryStoreState:
    raw = load_json_object(path)
    if raw is None:
        return CanonicalMemoryStoreState(schema_version=schema_version)

    if not isinstance(raw, dict):
        return CanonicalMemoryStoreState(schema_version=schema_version)

    records: list[CanonicalMemoryRecord] = []
    for item in raw.get("records", []):
        try:
            records.append(CanonicalMemoryRecord.model_validate(item))
        except ValidationError:
            continue

    parsed_schema_version = raw.get("schema_version", schema_version)
    if not isinstance(parsed_schema_version, int):
        parsed_schema_version = schema_version

    pending_consolidation_ids = raw.get("pending_consolidation_ids", [])
    if not isinstance(pending_consolidation_ids, list):
        pending_consolidation_ids = []

    return CanonicalMemoryStoreState(
        schema_version=parsed_schema_version,
        records=records,
        pending_consolidation_ids=pending_consolidation_ids,
    )


def save_store(
    path: str, state: RawMemoryStoreState | CanonicalMemoryStoreState
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(state.model_dump(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def save_raw_store(path: str, raw_store: RawMemoryStoreState) -> None:
    save_store(path, raw_store)


def save_canonical_store(path: str, canonical_store: CanonicalMemoryStoreState) -> None:
    save_store(path, canonical_store)
