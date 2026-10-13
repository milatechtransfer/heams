from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from heams.benchmarks.base import BaseBenchmark
from heams.benchmarks.contextual_integrity.benchmark import CIBenchmark
from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig
from heams.benchmarks.erasability_editability.benchmark import EEBenchmark
from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig
from heams.benchmarks.long_mem_eval.benchmark import LongMemEvalBenchmark
from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig
from heams.benchmarks.evomemory.benchmark import (
    EvoMemorySingleTurnBenchmark,
)
from heams.benchmarks.evomemory.config import (
    EvoMemorySingleTurnBenchmarkConfig,
)
from heams.benchmarks.locomo.benchmark import LocomoBenchmark
from heams.benchmarks.locomo.config import LocomoBenchmarkConfig
from heams.benchmarks.mpbench.benchmark import MpbenchBenchmark
from heams.benchmarks.mpbench.config import MpbenchBenchmarkConfig
from heams.benchmarks.retention_policy.benchmark import RetentionPolicyBenchmark
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.beam.benchmark import BeamBenchmark
from heams.benchmarks.beam.config import BeamBenchmarkConfig
from heams.benchmarks.sample import BenchmarkSample
from heams.benchmarks.stale.benchmark import StaleBenchmark
from heams.benchmarks.stale.config import StaleBenchmarkConfig
from heams.benchmarks.transparency.benchmark import TransparencyBenchmark
from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig

BenchmarkConfig = Annotated[
    Union[
        LocomoBenchmarkConfig,
        CIBenchmarkConfig,
        LongMemEvalBenchmarkConfig,
        EvoMemorySingleTurnBenchmarkConfig,
        TransparencyBenchmarkConfig,
        MpbenchBenchmarkConfig,
        StaleBenchmarkConfig,
        RetentionPolicyBenchmarkConfig,
        EEBenchmarkConfig,
        BeamBenchmarkConfig,
    ],
    Field(discriminator="type"),
]


def create_benchmark(config: BenchmarkConfig) -> BaseBenchmark:
    """Dispatch typed benchmark config to a family implementation.

    Create-time side effect may include dataset materialization when missing.
    """
    if config.type == "locomo":
        return LocomoBenchmark(config)
    if config.type == "ci":
        return CIBenchmark(config)
    if config.type == "long_mem_eval":
        return LongMemEvalBenchmark(config)
    if config.type == "evomemory_single_turn":
        return EvoMemorySingleTurnBenchmark(config)
    if config.type == "transparency":
        return TransparencyBenchmark(config)
    if config.type == "mpbench":
        return MpbenchBenchmark(config)
    if config.type == "stale":
        return StaleBenchmark(config)
    if config.type == "retention_policy":
        return RetentionPolicyBenchmark(config)
    if config.type == "ee":
        return EEBenchmark(config)
    if config.type == "beam":
        return BeamBenchmark(config)
    raise ValueError(f"Unsupported benchmark type: {config.type}")


__all__ = [
    "BenchmarkConfig",
    "BenchmarkSample",
    "CIBenchmarkConfig",
    "EEBenchmarkConfig",
    "EvoMemorySingleTurnBenchmarkConfig",
    "LocomoBenchmarkConfig",
    "StaleBenchmarkConfig",
    "LongMemEvalBenchmarkConfig",
    "MpbenchBenchmarkConfig",
    "RetentionPolicyBenchmarkConfig",
    "TransparencyBenchmarkConfig",
    "BeamBenchmarkConfig",
    "create_benchmark",
]
