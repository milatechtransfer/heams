from __future__ import annotations

import yaml

from heams.benchmarks.stale.config import StaleBenchmarkConfig
from heams.benchmarks.stale.samples import (
    StaleSample,
    load_stale_samples,
)
from heams.utils.settings import settings
from tests.paths import TESTS_DIR

CONFIG_DIR = settings.config_dir

FIXTURE = TESTS_DIR / "fixtures" / "smoke_e2e" / "stale" / "smoke.json"


def _benchmark_config(**overrides) -> StaleBenchmarkConfig:
    """Build a config from the shipped YAML, pointed at the local fixture."""
    raw = yaml.safe_load(
        (CONFIG_DIR / "benchmarks" / "full_campaign" / "stale.yaml").read_text()
    )
    raw["source"] = {"local_path": str(FIXTURE)}
    raw.update(overrides)
    return StaleBenchmarkConfig.model_validate(raw)


def _fixture_sample() -> StaleSample:
    return load_stale_samples(FIXTURE)[0]
