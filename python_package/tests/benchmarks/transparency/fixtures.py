"""Shared helpers for Transparency tests that need typed YAML config."""

from __future__ import annotations

import yaml
from pydantic import TypeAdapter

from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig
from heams.benchmarks.transparency.judge import TransparencyLlmJudgeConfig
from tests.paths import PACKAGE_ROOT, TESTS_DIR

_TRANSPARENCY_YAML = (
    PACKAGE_ROOT / "configs" / "benchmarks" / "testing" / "transparency_smoke.yaml"
)
_TRANSPARENCY_FIXTURE = (
    TESTS_DIR / "fixtures" / "smoke_e2e" / "transparency" / "smoke.json"
)
_BENCHMARK_ADAPTER = TypeAdapter(BenchmarkConfig)


def load_transparency_benchmark_config(
    **overrides: object,
) -> TransparencyBenchmarkConfig:
    """Load Transparency policy with a test-only JSON source."""
    payload = yaml.safe_load(_TRANSPARENCY_YAML.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {_TRANSPARENCY_YAML}")
    payload["source"] = {"type": "json", "path": str(_TRANSPARENCY_FIXTURE)}
    payload.update(overrides)
    config = _BENCHMARK_ADAPTER.validate_python(payload)
    if not isinstance(config, TransparencyBenchmarkConfig):
        raise TypeError(f"Expected TransparencyBenchmarkConfig, got {type(config)!r}")
    return config


def load_transparency_judge_config(
    **overrides: object,
) -> TransparencyLlmJudgeConfig:
    config = load_transparency_benchmark_config()
    if not overrides:
        return config.evaluator
    return config.evaluator.model_copy(update=overrides)
