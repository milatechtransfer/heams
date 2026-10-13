"""Pure temporal metadata normalization for memory-system callers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any


UTCClock = Callable[[], datetime]

_ABSOLUTE_TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


@dataclass(frozen=True)
class TemporalMetadataResolution:
    metadata: dict[str, Any]
    temporal_received_at: str | None = None
    temporal_source: str | None = None


def add_temporal_metadata(
    metadata: dict[str, Any],
    enabled: bool,
    utc_clock: UTCClock,
) -> dict[str, Any]:
    """Return metadata with a normalized temporal receipt timestamp when enabled."""
    return resolve_temporal_metadata(metadata, enabled, utc_clock).metadata


def resolve_temporal_metadata(
    metadata: dict[str, Any],
    enabled: bool,
    utc_clock: UTCClock,
) -> TemporalMetadataResolution:
    """Resolve temporal metadata and return its values for callers that need them."""
    result = dict(metadata)
    if not enabled:
        return TemporalMetadataResolution(metadata=result)

    existing_value = result.get("temporal_received_at")
    existing = _parse_absolute_timestamp(existing_value)
    if existing is not None:
        temporal_received_at = _format_utc(existing)
        existing_source = result.get("temporal_source")
        temporal_source = (
            existing_source
            if isinstance(existing_source, str) and existing_source
            else "metadata.temporal_received_at"
        )
        result["temporal_source"] = temporal_source
        result["temporal_received_at"] = temporal_received_at
        return TemporalMetadataResolution(
            metadata=result,
            temporal_received_at=temporal_received_at,
            temporal_source=temporal_source,
        )

    resolved = _parse_epoch_seconds(result.get("ingest_timestamp_epoch"))
    source = "metadata.ingest_timestamp_epoch"
    if resolved is None:
        resolved = _parse_absolute_timestamp(result.get("timestamp"))
        source = "metadata.timestamp"
    if resolved is None:
        resolved = _parse_absolute_timestamp(result.get("recorded_at"))
        source = "metadata.recorded_at"
    if resolved is None:
        resolved = _as_utc(utc_clock())
        source = "ingestion_clock"

    temporal_received_at = _format_utc(resolved)
    result["temporal_received_at"] = temporal_received_at
    result["temporal_source"] = source
    return TemporalMetadataResolution(
        metadata=result,
        temporal_received_at=temporal_received_at,
        temporal_source=source,
    )


def _parse_epoch_seconds(value: Any) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def _parse_absolute_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        return _as_utc(datetime.fromisoformat(candidate))
    except ValueError:
        pass

    for timestamp_format in _ABSOLUTE_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(candidate, timestamp_format).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def parse_temporal_timestamp(value: Any) -> datetime | None:
    """Parse a `temporal_received_at`-formatted string back to a UTC datetime."""
    return _parse_absolute_timestamp(value)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


__all__ = [
    "add_temporal_metadata",
    "resolve_temporal_metadata",
    "parse_temporal_timestamp",
]
