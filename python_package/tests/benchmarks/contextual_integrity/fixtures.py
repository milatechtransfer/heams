from __future__ import annotations

import yaml

from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig
from heams.utils.settings import settings
from tests.paths import TESTS_DIR

FIXTURE = TESTS_DIR / "fixtures" / "smoke_e2e" / "contextual_integrity" / "smoke.json"

CONFIG_DIR = settings.config_dir


def _benchmark_config(**overrides) -> CIBenchmarkConfig:
    """Build a config from the shipped YAML, pointed at the local fixture."""
    raw = yaml.safe_load(
        (
            CONFIG_DIR / "benchmarks" / "full_campaign" / "contextual_integrity.yaml"
        ).read_text()
    )
    raw["examples_dir"] = str(FIXTURE)
    raw.update(overrides)
    return CIBenchmarkConfig.model_validate(raw)
