from datetime import datetime, timezone

from heams_memory_milamem.temporal import (
    add_temporal_metadata,
    resolve_temporal_metadata,
)


FROZEN_UTC = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def frozen_utc_clock() -> datetime:
    return FROZEN_UTC


def test_disabled_returns_equivalent_copy_without_temporal_changes():
    metadata = {"timestamp": "2024-01-01T12:00:00Z", "nested": {"key": "value"}}

    result = add_temporal_metadata(metadata, enabled=False, utc_clock=frozen_utc_clock)

    assert result == metadata
    assert result is not metadata


def test_generates_receipt_time_from_frozen_clock():
    result = add_temporal_metadata({"source": "test"}, True, frozen_utc_clock)

    assert result["temporal_received_at"] == "2025-01-02T03:04:05Z"
    assert result["temporal_source"] == "ingestion_clock"


def test_resolution_exposes_generated_temporal_values_explicitly():
    resolution = resolve_temporal_metadata(
        {"timestamp": "2024-03-01 10:00:00"}, True, frozen_utc_clock
    )

    assert resolution.temporal_received_at == "2024-03-01T10:00:00Z"
    assert resolution.temporal_source == "metadata.timestamp"
    assert (
        resolution.metadata["temporal_received_at"] == resolution.temporal_received_at
    )
    assert resolution.metadata["temporal_source"] == resolution.temporal_source


def test_ingest_epoch_has_precedence_over_other_absolute_timestamps():
    result = add_temporal_metadata(
        {
            "ingest_timestamp_epoch": 1709287200,
            "timestamp": "2025-01-01T00:00:00Z",
            "recorded_at": "2025-02-01T00:00:00Z",
        },
        True,
        frozen_utc_clock,
    )

    assert result["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert result["temporal_source"] == "metadata.ingest_timestamp_epoch"


def test_timestamp_is_normalized_to_utc():
    result = add_temporal_metadata(
        {"timestamp": "2024/03/01 10:00"}, True, frozen_utc_clock
    )

    assert result["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert result["temporal_source"] == "metadata.timestamp"


def test_relative_recorded_at_is_preserved_and_receipt_uses_clock():
    result = add_temporal_metadata({"recorded_at": "T-10d"}, True, frozen_utc_clock)

    assert result["recorded_at"] == "T-10d"
    assert result["temporal_received_at"] == "2025-01-02T03:04:05Z"
    assert result["temporal_source"] == "ingestion_clock"


def test_existing_valid_receipt_time_is_preserved_idempotently():
    metadata = {
        "temporal_received_at": "2024-03-01T10:00:00Z",
        "temporal_source": "caller-provided",
    }

    result = add_temporal_metadata(metadata, True, frozen_utc_clock)

    assert result["temporal_received_at"] == metadata["temporal_received_at"]
    assert result["temporal_source"] == "caller-provided"
    assert add_temporal_metadata(result, True, frozen_utc_clock) == result


def test_non_normalized_existing_receipt_time_is_normalized_and_preserved():
    result = add_temporal_metadata(
        {
            "temporal_received_at": "2024-03-01T12:00:00+02:00",
            "timestamp": "2025-01-01T00:00:00Z",
        },
        True,
        frozen_utc_clock,
    )

    assert result["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert result["temporal_source"] == "metadata.temporal_received_at"


def test_invalid_source_values_fall_back_to_frozen_clock():
    result = add_temporal_metadata(
        {
            "ingest_timestamp_epoch": "not-an-epoch",
            "timestamp": "not-a-timestamp",
            "recorded_at": "T-10d",
        },
        True,
        frozen_utc_clock,
    )

    assert result["temporal_received_at"] == "2025-01-02T03:04:05Z"
    assert result["temporal_source"] == "ingestion_clock"


def test_input_metadata_is_not_mutated():
    metadata = {
        "timestamp": "2024-03-01 10:00:00",
        "temporal_source": "caller-provided",
        "value": "original",
    }

    result = add_temporal_metadata(metadata, True, frozen_utc_clock)

    assert metadata == {
        "timestamp": "2024-03-01 10:00:00",
        "temporal_source": "caller-provided",
        "value": "original",
    }
    assert result["timestamp"] == metadata["timestamp"]
    assert result["temporal_source"] == "metadata.timestamp"
    assert result["temporal_received_at"] == "2024-03-01T10:00:00Z"
