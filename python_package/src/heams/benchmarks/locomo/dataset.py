"""LOCOMO dataset path resolution, download, and answerable-sample filtering."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from heams.benchmarks.base import SkipReason
from heams.benchmarks.locomo.judge import is_scored_locomo_category

LOCOMO_DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
_DEFAULT_DATASET_FILE = "locomo10.json"


def ensure_dataset(path: Path) -> None:
    """Download the upstream LOCOMO file when `path` is missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LOCOMO_DATASET_URL) as response:
        path.write_bytes(response.read())


def resolve_dataset_paths(examples_dir: Path) -> list[Path]:
    """Pick the LOCOMO JSON file(s) to load."""
    if examples_dir.is_file():
        return [examples_dir]
    if not examples_dir.exists():
        return [examples_dir / _DEFAULT_DATASET_FILE]
    json_paths = sorted(examples_dir.glob("*.json"))
    if json_paths:
        return json_paths
    return [examples_dir / _DEFAULT_DATASET_FILE]


def split_payload(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read `dataset_path` and partition samples into answerable vs not."""
    ensure_dataset(dataset_path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LOCOMO data must be a list of sample objects.")

    answerable: list[dict[str, Any]] = []
    unanswerable: list[str] = []
    for sample in payload:
        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        qa_items = sample.get("qa") or []
        scored_qa_items = [
            qa for qa in qa_items if is_scored_locomo_category(qa.get("category"))
        ]
        if any(qa.get("question") and qa.get("answer") for qa in scored_qa_items):
            filtered_sample = dict(sample)
            filtered_sample["qa"] = scored_qa_items
            answerable.append(filtered_sample)
        else:
            unanswerable.append(sample_id or "<unknown>")
    return answerable, unanswerable


def skip_reasons_for_unanswerable(
    dataset_path: Path, unanswerable_ids: list[str]
) -> list[SkipReason]:
    return [
        SkipReason(
            path=dataset_path,
            reason=f"sample {sample_id}: no answerable QA stimuli",
        )
        for sample_id in unanswerable_ids
    ]
