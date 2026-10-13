"""Shared LongMemEval test fixtures backed by checked-in component YAML."""

from __future__ import annotations

import yaml
from pydantic import TypeAdapter

from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.long_mem_eval.config import LongMemEvalBenchmarkConfig
from tests.paths import PACKAGE_ROOT

_LONG_MEM_EVAL_YAML = (
    PACKAGE_ROOT / "configs" / "benchmarks" / "full_campaign" / "long_mem_eval.yaml"
)
_BENCHMARK_ADAPTER = TypeAdapter(BenchmarkConfig)


def load_long_mem_eval_benchmark_config(
    **overrides: object,
) -> LongMemEvalBenchmarkConfig:
    """Load LongMemEval component YAML and apply test-local overrides."""
    payload = yaml.safe_load(_LONG_MEM_EVAL_YAML.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {_LONG_MEM_EVAL_YAML}")
    payload.update(overrides)
    config = _BENCHMARK_ADAPTER.validate_python(payload)
    if not isinstance(config, LongMemEvalBenchmarkConfig):
        raise TypeError(f"Expected LongMemEvalBenchmarkConfig, got {type(config)!r}")
    return config
