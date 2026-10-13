"""Transparency dataset path resolution and BEAM materialization."""

from __future__ import annotations

from heams.benchmarks.base import BenchmarkDatasetUnavailableError
from heams.benchmarks.transparency.beam import (
    BEAM_DEFAULT_SPLIT,
    BeamTransparencyAdapterConfig,
    convert_beam_records_to_transparency,
    load_beam_records,
)
from heams.benchmarks.transparency.config import (
    BeamTransparencySourceConfig,
    JsonTransparencySourceConfig,
    TransparencyBenchmarkConfig,
    TransparencySourceConfig,
)
from heams.benchmarks.transparency.samples import (
    TransparencySample,
    load_transparency_samples,
)


def materialize_source(
    source: TransparencySourceConfig,
) -> list[TransparencySample]:
    """Resolve JSON or BEAM source into validated transparency samples."""
    if isinstance(source, JsonTransparencySourceConfig):
        return load_transparency_samples(source.path)

    assert isinstance(source, BeamTransparencySourceConfig)
    return materialize_beam_or_raise(
        split=source.split,
        materialization_limit=source.materialization_limit,
        adapter=source.adapter,
    )


def materialize_beam_or_raise(
    *,
    split: str = BEAM_DEFAULT_SPLIT,
    materialization_limit: int | None = None,
    adapter: BeamTransparencyAdapterConfig | None = None,
) -> list[TransparencySample]:
    """Download and convert BEAM records, or raise an actionable error."""
    try:
        records = load_beam_records(split=split)
        samples = convert_beam_records_to_transparency(
            records,
            split=split,
            token_scale=split,
            limit=materialization_limit,
            adapter=adapter,
        )
        return load_transparency_samples(samples)
    except Exception as error:
        raise BenchmarkDatasetUnavailableError(
            f"Could not download and convert the BEAM {split} transparency dataset. "
            "Enable download with network access, e.g. `uv run heams run --config "
            "configs/runs/testing/transparency_smoke.yaml`."
        ) from error


def load_samples_for_config(
    config: TransparencyBenchmarkConfig,
) -> list[TransparencySample]:
    samples = materialize_source(config.source)
    if config.limit is not None:
        return samples[: config.limit]
    return samples


__all__ = [
    "load_samples_for_config",
    "materialize_beam_or_raise",
    "materialize_source",
]
