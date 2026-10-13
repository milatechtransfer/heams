from __future__ import annotations

import yaml

from heams.benchmarks.erasability_editability.config import (
    EEBenchmarkConfig,
)
from heams.benchmarks.erasability_editability.samples import (
    EEEpisode,
    episode_to_benchmark_sample,
    load_ee_episodes,
)
from heams.utils.settings import settings
from tests.paths import TESTS_DIR

CONFIG_DIR = settings.config_dir

FIXTURE = (
    TESTS_DIR / "fixtures" / "smoke_e2e" / "erasability_editability" / "smoke.json"
)

FULL_CONFIG = "full_campaign/erasability_editability.yaml"

EXPECTED_PROBE_KEYS = [
    "pl_001:edit:before1",
    "pl_001:edit:before2",
    "pl_001:erase:before1",
    "pl_001:edit:after1",
    "pl_001:edit:after2",
    "pl_001:erase:after1",
]


def _benchmark_config(**overrides) -> EEBenchmarkConfig:
    """Build a config from the shipped YAML, pointed at the local fixture."""
    raw = yaml.safe_load((CONFIG_DIR / "benchmarks" / FULL_CONFIG).read_text())
    raw["source"] = {"local_path": str(FIXTURE)}
    raw.update(overrides)
    return EEBenchmarkConfig.model_validate(raw)


def _fixture_episode() -> EEEpisode:
    return load_ee_episodes(FIXTURE)[0]


def _scenario():
    return episode_to_benchmark_sample(
        _fixture_episode(), config=_benchmark_config()
    ).to_scenario()
