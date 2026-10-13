"""Iterable EE benchmark: materialize dataset, yield BenchmarkSample."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig
from heams.benchmarks.erasability_editability.dataset import materialize_source
from heams.benchmarks.erasability_editability.samples import (
    episode_to_benchmark_sample,
)
from heams.benchmarks.sample import BenchmarkSample


class EEBenchmark(BaseBenchmark):
    """New-path EE workload: one BenchmarkSample per MEME episode."""

    def __init__(self, config: EEBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        episodes = materialize_source(self.config)
        if self.config.limit is not None:
            episodes = episodes[: self.config.limit]

        samples: list[BenchmarkSample] = []
        skipped: list[SkipReason] = []
        for episode in episodes:
            try:
                samples.append(episode_to_benchmark_sample(episode, config=self.config))
            except (ValueError, KeyError) as error:
                skipped.append(
                    SkipReason(path=Path(episode.episode_id), reason=str(error))
                )

        self._samples = samples
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["EEBenchmark"]
