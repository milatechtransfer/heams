"""Iterable BEAM benchmark: materialize conversations, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import (
    BaseBenchmark,
    BenchmarkDatasetUnavailableError,
    SkipReason,
)
from heams.benchmarks.beam.config import BeamBenchmarkConfig
from heams.benchmarks.beam.dataset import load_beam_records
from heams.benchmarks.beam.samples import record_to_benchmark_sample
from heams.benchmarks.sample import BenchmarkSample


class BeamBenchmark(BaseBenchmark):
    """First-class BEAM workload: full-chat priming plus probing-question stimuli."""

    def __init__(self, config: BeamBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        try:
            records = load_beam_records(
                split=self.config.split,
                local_path=self.config.local_path,
                limit=self.config.limit,
            )
        except Exception as error:
            raise BenchmarkDatasetUnavailableError(
                f"Could not load the BEAM {self.config.split} dataset. "
                "Use `local_path` for a JSON fixture or run with network access, "
                "e.g. `uv run heams run --config configs/runs/testing/beam_smoke.yaml`."
            ) from error

        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []
        for record in records:
            conversation_id = str(record.get("conversation_id") or "unknown")
            try:
                samples.append(record_to_benchmark_sample(record, config=self.config))
            except ValueError as error:
                skipped.append(
                    SkipReason(
                        path=self.config.local_path
                        or Path(f"beam-{self.config.split}"),
                        reason=f"{conversation_id}: {error}",
                    )
                )
        if self.config.limit is not None:
            samples = samples[: self.config.limit]
        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["BeamBenchmark"]
