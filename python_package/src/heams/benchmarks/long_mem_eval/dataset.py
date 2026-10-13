"""LongMemEval dataset resolution and materialization helpers."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

LONG_MEM_EVAL_DATASET_URLS = {
    "s": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json"
    ),
    "m": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_m_cleaned.json"
    ),
    "oracle": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_oracle.json"
    ),
}
LONG_MEM_EVAL_DEFAULT_SPLIT = "s"
LONG_MEM_EVAL_DEFAULT_FILENAME = "longmemeval_s_cleaned.json"


def split_name_from_path(path: Path) -> str | None:
    """Infer a dataset split from its standard LongMemEval filename."""
    name = path.name.lower()
    if "oracle" in name:
        return "oracle"
    if "_m" in name or "-m" in name:
        return "m"
    if "_s" in name or "-s" in name:
        return "s"
    return None


def ensure_dataset(path: Path) -> None:
    """Download a missing LongMemEval dataset file."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    split = split_name_from_path(path) or LONG_MEM_EVAL_DEFAULT_SPLIT
    with urllib.request.urlopen(LONG_MEM_EVAL_DATASET_URLS[split]) as response:
        path.write_bytes(response.read())


def resolve_dataset_paths(examples_dir: Path) -> list[Path]:
    """Resolve a configured file or directory to LongMemEval JSON files."""
    if examples_dir.is_file():
        return [examples_dir]
    if not examples_dir.exists():
        return [examples_dir / LONG_MEM_EVAL_DEFAULT_FILENAME]
    json_paths = sorted(examples_dir.glob("*.json"))
    if json_paths:
        return json_paths
    return [examples_dir / LONG_MEM_EVAL_DEFAULT_FILENAME]


def load_payload(dataset_path: Path) -> list[dict[str, Any]]:
    """Read one LongMemEval dataset file after materializing it when absent."""
    ensure_dataset(dataset_path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LongMemEval data must be a list of sample objects.")
    if not all(isinstance(sample, dict) for sample in payload):
        raise ValueError("LongMemEval data must contain only sample objects.")
    return payload
