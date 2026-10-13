"""Iterable MPBench benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.mpbench.config import MpbenchBenchmarkConfig
from heams.benchmarks.mpbench.dataset import materialize_mpbench_records
from heams.benchmarks.mpbench.samples import record_to_benchmark_sample
from heams.benchmarks.sample import BenchmarkSample


class MpbenchBenchmark(BaseBenchmark):
    """New-path MPBench workload: iterable of BenchmarkSample."""

    def __init__(self, config: MpbenchBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        records, skipped = materialize_mpbench_records(
            self.config.examples_dir,
            limit_per_class=self.config.limit_per_class,
            limit=self.config.limit,
            auto_download=self.config.auto_download,
        )
        self.skipped = skipped
        self._samples = [
            record_to_benchmark_sample(record, config=self.config) for record in records
        ]

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["MpbenchBenchmark"]
