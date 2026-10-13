"""Iterable LongMemEval benchmark for the config-driven CLI path."""

from __future__ import annotations

from collections.abc import Sequence

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig
from heams.benchmarks.long_mem_eval.dataset import load_payload, resolve_dataset_paths
from heams.benchmarks.long_mem_eval.samples import record_to_benchmark_sample
from heams.benchmarks.sample import BenchmarkSample


class LongMemEvalBenchmark(BaseBenchmark):
    """Materialized LongMemEval workload with one sample per source record."""

    def __init__(self, config: LongMemEvalBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []
        for dataset_path in resolve_dataset_paths(self.config.examples_dir):
            for record in load_payload(dataset_path):
                try:
                    samples.append(
                        record_to_benchmark_sample(
                            record,
                            config=self.config,
                            dataset_path=dataset_path,
                        )
                    )
                except ValueError as error:
                    sample_id = str(record.get("question_id") or "<unknown>")
                    skipped.append(
                        SkipReason(
                            path=dataset_path, reason=f"sample {sample_id}: {error}"
                        )
                    )
        if self.config.limit is not None:
            samples = samples[: self.config.limit]
        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples
