"""Iterable CI benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig
from heams.benchmarks.contextual_integrity.dataset import (
    load_records,
    resolve_dataset_paths,
)
from heams.benchmarks.contextual_integrity.samples import record_to_benchmark_sample
from heams.benchmarks.sample import BenchmarkSample


class CIBenchmark(BaseBenchmark):
    """New-path Contextual Integrity workload: iterable of BenchmarkSample.

    Unlike the legacy `CIBenchmark`, this class applies no
    agent or memory overlay. Instructions, request timeout, and memory `top_k`
    are component settings and are read from configs/agents/*.yaml and
    configs/memory_systems/*.yaml.
    """

    def __init__(self, config: CIBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []

        for dataset_path in resolve_dataset_paths(self.config):
            try:
                records = load_records(dataset_path)
            except (OSError, ValueError) as error:
                skipped.append(SkipReason(path=dataset_path, reason=str(error)))
                continue
            for index, record in enumerate(records):
                try:
                    samples.append(
                        record_to_benchmark_sample(record, config=self.config)
                    )
                except ValueError as error:
                    # One malformed row must not sink the whole dataset.
                    skipped.append(
                        SkipReason(
                            path=dataset_path,
                            reason=f"record {index}: {error}",
                        )
                    )

        if self.config.limit is not None:
            samples = samples[: self.config.limit]

        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["CIBenchmark"]
