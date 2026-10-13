"""Resolve the CI corpus to local JSON, bootstrapping from HuggingFace if asked.

Parsing of the upstream two-JSON-column parquet rows is reused from
``heams.scenarios.loaders.contextual_integrity``; what this module adds is
config-driven resolution: which repo, which split, which parquet, how many
rows to cache, and whether downloading is permitted at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heams.benchmarks.base import BenchmarkDatasetUnavailableError
from heams.scenarios.loaders.contextual_integrity import parse_ci_records

if TYPE_CHECKING:
    from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig


def resolve_dataset_paths(config: CIBenchmarkConfig) -> list[Path]:
    """Return the JSON files backing this run, downloading only if permitted."""

    examples_dir = config.examples_dir

    if examples_dir.suffix == ".json":
        if examples_dir.exists():
            return [examples_dir]
        return [_bootstrap(config, examples_dir)]

    fixture_path = examples_dir / config.fixture_name()
    if fixture_path.exists():
        return [fixture_path]
    return [_bootstrap(config, fixture_path)]


def _bootstrap(config: CIBenchmarkConfig, fixture_path: Path) -> Path:
    if not config.dataset_source.auto_download:
        raise BenchmarkDatasetUnavailableError(
            f"CI dataset not found at {fixture_path} and "
            "`dataset_source.auto_download` is false. Either point "
            "`examples_dir` at an existing JSON file/directory or enable "
            "auto_download."
        )
    records = download_ci_records(config)
    samples = parse_ci_records(
        records,
        split=config.dataset_source.split,
        limit=config.dataset_source.download_limit,
    )
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return fixture_path


def download_ci_records(config: CIBenchmarkConfig) -> list[dict[str, Any]]:
    """Download the configured parquet from HuggingFace and return raw rows."""

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on local setup
        raise BenchmarkDatasetUnavailableError(
            "Loading the Contextual Integrity dataset requires pandas and "
            "huggingface_hub. Run through uv from python_package, for example: "
            "uv run --with pyarrow heams run --config configs/runs/testing/contextual_integrity_smoke.yaml"
        ) from error

    parquet_path = hf_hub_download(
        repo_id=config.dataset_source.hf_repo_id,
        repo_type="dataset",
        filename=config.parquet_path(),
    )
    try:
        frame = pd.read_parquet(parquet_path)
    except ImportError as error:  # pragma: no cover - depends on optional pyarrow
        raise BenchmarkDatasetUnavailableError(
            "Reading the Contextual Integrity parquet requires a parquet engine "
            "such as pyarrow. Use `uv run --with pyarrow ...` or install pyarrow "
            "in the environment."
        ) from error
    return frame.to_dict(orient="records")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read one parsed-CI JSON file, validating only its top-level shape."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"Contextual Integrity data at {path} must be a list of record objects."
        )
    return payload


__all__ = ["download_ci_records", "load_records", "resolve_dataset_paths"]
