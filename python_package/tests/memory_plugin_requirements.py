from pathlib import Path

import pytest
import yaml

from heams.memory_systems import get_memory_plugin_catalog


def skip_if_memory_plugin_missing(payload: dict) -> None:
    type_name = payload.get("backend_config", payload).get("type")
    if get_memory_plugin_catalog().get(type_name) is None:
        pytest.skip(f"memory plugin {type_name!r} is not installed")


def skip_if_run_config_memory_plugins_missing(memory_config_paths: list[Path]) -> None:
    for path in memory_config_paths:
        skip_if_memory_plugin_missing(yaml.safe_load(path.read_text(encoding="utf-8")))
