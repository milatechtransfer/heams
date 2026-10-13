"""Iterable Transparency benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig
from heams.benchmarks.transparency.dataset import materialize_source
from heams.benchmarks.transparency.samples import sample_to_benchmark_sample


class TransparencyBenchmark(BaseBenchmark):
    """New-path Transparency workload: iterable of BenchmarkSample."""

    def __init__(self, config: TransparencyBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        source_samples = materialize_source(self.config.source)
        if self.config.limit is not None:
            source_samples = source_samples[: self.config.limit]
        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []

        for source_sample in source_samples:
            try:
                samples.append(
                    sample_to_benchmark_sample(source_sample, config=self.config)
                )
            except ValueError as error:
                skipped.append(
                    SkipReason(path=Path(source_sample.id), reason=str(error))
                )

        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["TransparencyBenchmark"]
