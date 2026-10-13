"""Shared helpers for Locomo tests that need YAML-authored prompt fields."""

from __future__ import annotations

import yaml
from pydantic import TypeAdapter

from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.locomo.config import LocomoBenchmarkConfig
from heams.benchmarks.locomo.judge import LocomoLlmJudgeConfig
from tests.paths import PACKAGE_ROOT

_LOCOMO_YAML = PACKAGE_ROOT / "configs" / "benchmarks" / "full_campaign" / "locomo.yaml"
_BENCHMARK_ADAPTER = TypeAdapter(BenchmarkConfig)


def load_locomo_benchmark_config(**overrides: object) -> LocomoBenchmarkConfig:
    """Load the full campaign LoCoMo config and apply test overrides."""
    payload = yaml.safe_load(_LOCOMO_YAML.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {_LOCOMO_YAML}")
    payload.update(overrides)
    config = _BENCHMARK_ADAPTER.validate_python(payload)
    if not isinstance(config, LocomoBenchmarkConfig):
        raise TypeError(f"Expected LocomoBenchmarkConfig, got {type(config)!r}")
    return config


def load_locomo_judge_config(**overrides: object) -> LocomoLlmJudgeConfig:
    """Load evaluator prompt fields from the full campaign LoCoMo config."""
    config = load_locomo_benchmark_config()
    if not overrides:
        return config.evaluator
    return config.evaluator.model_copy(update=overrides)
