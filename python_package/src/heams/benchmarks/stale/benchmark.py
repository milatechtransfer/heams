"""Iterable STALE benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.stale.config import StaleBenchmarkConfig
from heams.benchmarks.stale.dataset import materialize_source
from heams.benchmarks.stale.samples import sample_to_benchmark_sample


class StaleBenchmark(BaseBenchmark):
    """New-path STALE workload: iterable of BenchmarkSample (1 per source sample; 3 probes)"""

    def __init__(self, config: StaleBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        source_samples = materialize_source(self.config)
        if self.config.limit is not None:
            source_samples = source_samples[: self.config.limit]

        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []
        for source_sample in source_samples:
            try:
                samples.append(
                    sample_to_benchmark_sample(source_sample, config=self.config)
                )
            except (ValueError, KeyError) as error:
                skipped.append(
                    SkipReason(path=Path(source_sample.uid), reason=str(error))
                )

        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["StaleBenchmark"]
