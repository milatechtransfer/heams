"""EvoMemory benchmark implementations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import (
    BaseBenchmark,
    BenchmarkDatasetUnavailableError,
    SkipReason,
)
from heams.benchmarks.evomemory.config import (
    EvoMemorySingleTurnBenchmarkConfig,
)
from heams.benchmarks.evomemory.data import (
    MATCH_MODE_BY_FILE,
    SINGLE_TURN_FILES,
    materialize_single_turn_dataset,
)
from heams.benchmarks.evomemory.samples import (
    load_single_turn_streams,
    record_to_benchmark_sample,
)
from heams.benchmarks.sample import BenchmarkSample


class EvoMemorySingleTurnBenchmark(BaseBenchmark):
    """Benchmark policy for ordered Evo-Memory single-turn memory streams.

    Datasets are materialized on demand into the gitignored datasets directory.
    AIME, GPQA, and MMLU-Pro come from Hugging Face; ToolBench is rebuilt from
    Berkeley Function-Calling Leaderboard source files. GPQA requires an
    accepted-terms Hugging Face token (``HF_TOKEN``).
    """

    def __init__(self, config: EvoMemorySingleTurnBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples = self._materialize_samples()

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples

    def _materialize_samples(self) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        for filename in SINGLE_TURN_FILES:
            dataset_path = self.config.examples_dir / filename
            try:
                self._materialize_dataset(dataset_path)
                streams = load_single_turn_streams(dataset_path)
            except Exception as exc:
                raise BenchmarkDatasetUnavailableError(
                    f"Unable to materialize required EvoMemory stream "
                    f"'{dataset_path}': {exc}"
                ) from exc

            for index, stream in enumerate(streams):
                samples.append(
                    record_to_benchmark_sample(
                        stream,
                        default_id=f"{dataset_path.stem}-{index}",
                        match_mode=self._match_mode_for(dataset_path),
                        limit_stimuli=self.config.limit_stimuli,
                    )
                )
        if self.config.limit is not None:
            samples = samples[: self.config.limit]
        return samples

    @staticmethod
    def _materialize_dataset(dataset_path: Path) -> None:
        if dataset_path.exists():
            return
        materialize_single_turn_dataset(dataset_path)

    @staticmethod
    def _match_mode_for(dataset_path: Path) -> str:
        return MATCH_MODE_BY_FILE.get(dataset_path.name, "normalized_exact")
