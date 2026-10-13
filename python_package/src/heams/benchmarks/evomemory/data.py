"""Evo-Memory benchmark adapters."""

from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Callable

from heams.utils.settings import settings

_DEFAULT_SINGLE_TURN_DIR = settings.datasets_dir / "evomemory" / "single_turn"
_SINGLE_TURN_FILES: tuple[str, ...] = (
    "aime_2024.json",
    "gpqa_diamond.json",
    "mmlu_pro.json",
    "toolbench.json",
)
_MATCH_MODE_BY_FILE = {
    "aime_2024.json": "evomemory_aime",
    "gpqa_diamond.json": "evomemory_gpqa",
    "mmlu_pro.json": "evomemory_mmlu_pro",
    "toolbench.json": "bfcl_ast",
}


# --- HuggingFace single-turn builders --------------------------------------
# These builders normalize upstream single-turn sources for the existing
# loader/evaluator: AIME/GPQA/MMLU-Pro come from `datasets`, while ToolBench is
# rebuilt from Berkeley Function-Calling Leaderboard raw files paired with their
# `possible_answer` ground truth.
_GPQA_OPTION_KEYS = (
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)
_HF_REVISIONS = {
    "HuggingFaceH4/aime_2024": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
}
_BFCL_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
_BFCL_CATEGORIES = ("simple_python", "multiple")


def _hf_hub_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_hf_dataset(path: str, *config: str, split: str) -> Any:
    # Imported lazily so importing this module never requires `datasets`; only
    # the builders that materialize real data pull in the dependency.
    from datasets import load_dataset

    # Pass the token explicitly: gated datasets (GPQA) are not served on the
    # implicit-token path even when HF_TOKEN is exported.
    token = _hf_hub_token()
    kwargs: dict[str, Any] = {"split": split, "revision": _HF_REVISIONS[path]}
    if token:
        kwargs["token"] = token
    return load_dataset(path, *config, **kwargs)


def build_aime(year: int = 2024) -> list[dict[str, Any]]:
    dataset = _load_hf_dataset(f"HuggingFaceH4/aime_{year}", split="train")
    records: list[dict[str, Any]] = []
    for item in dataset:
        problem = item.get("problem") or item.get("question") or ""
        answer = str(item.get("answer") or item.get("solution") or "")
        records.append({"problem": problem, "answer": answer, "year": year})
    return records


def build_gpqa(subset: str = "diamond") -> list[dict[str, Any]]:
    dataset = _load_hf_dataset("Idavidrein/gpqa", f"gpqa_{subset}", split="train")
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        options = [item.get(key) for key in _GPQA_OPTION_KEYS]
        correct_answer = item.get("Correct Answer", "")
        if not all(isinstance(option, str) and option for option in options):
            raise ValueError(f"GPQA row {idx} does not contain four answer options.")
        if correct_answer not in options:
            raise ValueError(f"GPQA row {idx} does not contain its correct answer.")
        # Per-item seeded RNG reproduces upstream's `random.seed(idx)` shuffle
        # ordering while avoiding mutation of global RNG state.
        random.Random(idx).shuffle(options)
        correct_index = options.index(correct_answer)
        domain = item.get("High-level domain", "science")
        records.append(
            {
                "question": item["Question"],
                "options": options,
                "answer": chr(65 + correct_index),
                "answer_index": correct_index,
                "correct_answer_text": correct_answer,
                "category": domain,
                "domain": domain,
            }
        )
    return records


def build_mmlu_pro() -> list[dict[str, Any]]:
    dataset = _load_hf_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records: list[dict[str, Any]] = []
    for item in dataset:
        records.append(
            {
                "question": item["question"],
                "options": list(item.get("options", [])),
                "answer": item.get("answer", ""),
                "answer_index": item.get("answer_index"),
                "category": item.get("category", "general"),
            }
        )
    return records


def build_bfcl_toolbench(
    categories: tuple[str, ...] = _BFCL_CATEGORIES,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category in categories:
        data = _fetch_bfcl_jsonl(f"BFCL_v4_{category}.json")
        answers = _fetch_bfcl_jsonl(f"possible_answer/BFCL_v4_{category}.json")
        ground_truth_by_id = {row.get("id"): row.get("ground_truth") for row in answers}
        for item in data:
            item_id = item.get("id")
            ground_truth = ground_truth_by_id.get(item_id)
            if ground_truth is None:
                raise ValueError(
                    f"BFCL task {item_id!r} does not have possible-answer ground truth."
                )
            records.append(
                {
                    "task_id": str(item_id),
                    "question": _bfcl_question_text(item.get("question")),
                    "functions": item.get("function", []),
                    # The repo loader carries `answer` into the expected-answer
                    # marker; encoding BFCL ground truth here lets the bfcl_ast
                    # scorer recover the acceptable-value lists.
                    "answer": json.dumps(ground_truth, ensure_ascii=False),
                    "ground_truth": ground_truth,
                    "category": f"bfcl_{category}",
                }
            )
    return records


def _fetch_bfcl_jsonl(relative_path: str) -> list[dict[str, Any]]:
    url = f"{_BFCL_RAW_BASE}/{relative_path}"
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _bfcl_question_text(question: Any) -> str:
    if isinstance(question, list) and question and isinstance(question[0], list):
        parts = [
            str(message.get("content", ""))
            for message in question[0]
            if isinstance(message, dict) and message.get("content")
        ]
        return "\n".join(parts).strip()
    return str(question)


_BUILDERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "aime_2024.json": lambda: build_aime(2024),
    "gpqa_diamond.json": lambda: build_gpqa("diamond"),
    "mmlu_pro.json": build_mmlu_pro,
    "toolbench.json": build_bfcl_toolbench,
}


def build_single_turn_dataset(name: str) -> list[dict[str, Any]]:
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"No HuggingFace builder for Evo-Memory dataset {name!r}.")
    return builder()


def materialize_single_turn_dataset(path: Path) -> None:
    records = build_single_turn_dataset(path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT_SINGLE_TURN_DIR = _DEFAULT_SINGLE_TURN_DIR
MATCH_MODE_BY_FILE = _MATCH_MODE_BY_FILE
SINGLE_TURN_FILES = _SINGLE_TURN_FILES
