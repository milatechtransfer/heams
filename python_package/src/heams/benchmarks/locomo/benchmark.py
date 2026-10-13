"""Iterable LOCOMO benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.locomo.config import LocomoBenchmarkConfig
from heams.benchmarks.locomo.dataset import (
    resolve_dataset_paths,
    skip_reasons_for_unanswerable,
    split_payload,
)
from heams.benchmarks.locomo.samples import record_to_benchmark_sample
from heams.benchmarks.sample import BenchmarkSample


class LocomoBenchmark(BaseBenchmark):
    """New-path LOCOMO workload: iterable of BenchmarkSample."""

    def __init__(self, config: LocomoBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []

        for dataset_path in resolve_dataset_paths(self.config.examples_dir):
            answerable_payload, unanswerable_ids = split_payload(dataset_path)
            skipped.extend(
                skip_reasons_for_unanswerable(dataset_path, unanswerable_ids)
            )
            if not answerable_payload:
                continue
            for record in answerable_payload:
                samples.append(
                    record_to_benchmark_sample(
                        record,
                        config=self.config,
                    )
                )

        if self.config.limit is not None:
            samples = samples[: self.config.limit]

        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples
