"""Parse Contextual Integrity dataset rows into flat, validated records.

Source: ``huseyinatahaninan/ContextualIntegritySyntheticDataset`` on HuggingFace.
Each row has two JSON-string columns:

* ``seed`` -- the context norm: scenario, domain, sender, recipient,
  transmission principle, and the allowed/disallowed information *types*.
* ``dataset_item`` -- the concrete instance: the ``user_task``, the full pool of
  ``information`` the agent holds, and the ``annotation`` marking each value as
  ``allowed`` (fine to share here) or ``disallowed`` (a privacy breach if shared).

``parse_ci_records`` flattens the two JSON columns into plain dicts, and
``ContextualIntegritySample`` validates one such record. The benchmark package
(``heams.benchmarks.contextual_integrity``) downloads the parquet, calls these to
parse it, and projects the records into ``BenchmarkSample`` objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CI_DATASET_NAME = "ContextualIntegritySyntheticDataset"
CI_DEFAULT_SPLIT = "train"

_SEED_REQUIRED_KEYS = (
    "scenario",
    "domain",
    "user_intention",
    "sender",
    "recipient",
    "data_subject",
    "information_type",
    "transmission_principle",
)
_ITEM_REQUIRED_KEYS = ("user_task", "information", "annotation")


def parse_ci_records(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str = CI_DEFAULT_SPLIT,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Parse raw rows (two JSON-string columns) into flat, faithful dicts."""

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        samples.append(parse_ci_record(record, split=split, row_index=index))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_ci_record(
    record: Mapping[str, Any],
    *,
    split: str = CI_DEFAULT_SPLIT,
    row_index: int,
) -> dict[str, Any]:
    """Parse a single CI row, validating the expected source structure."""

    seed = _load_json_field(record, "seed", row_index=row_index)
    item = _load_json_field(record, "dataset_item", row_index=row_index)
    _require_keys(seed, _SEED_REQUIRED_KEYS, field="seed", row_index=row_index)
    _require_keys(item, _ITEM_REQUIRED_KEYS, field="dataset_item", row_index=row_index)
    annotation = item["annotation"]
    _require_keys(
        annotation, ("allowed", "disallowed"), field="annotation", row_index=row_index
    )

    return {
        "id": f"CI-{split}-{row_index:04d}",
        "source": {
            "dataset": CI_DATASET_NAME,
            "split": split,
            "row_index": row_index,
        },
        "scenario": seed["scenario"],
        "domain": seed["domain"],
        "transmission_principle": seed["transmission_principle"],
        "user_intention": seed["user_intention"],
        "sender": seed["sender"],
        "recipient": seed["recipient"],
        "data_subject": seed["data_subject"],
        "information_type": seed["information_type"],
        "user_task": item["user_task"],
        "information": item["information"],
        "annotation": {
            "allowed": annotation["allowed"],
            "disallowed": annotation["disallowed"],
        },
    }


def _load_json_field(
    record: Mapping[str, Any], field: str, *, row_index: int
) -> dict[str, Any]:
    if field not in record:
        raise ValueError(f"CI row {row_index} is missing the {field!r} column.")
    value = record[field]
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(
            f"CI row {row_index} field {field!r} must be a JSON string, got "
            f"{type(value).__name__}."
        )
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"CI row {row_index} field {field!r} must decode to a JSON object."
        )
    return parsed


def _require_keys(
    payload: Mapping[str, Any], keys: Iterable[str], *, field: str, row_index: int
) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(
            f"CI row {row_index} field {field!r} is missing keys: {', '.join(missing)}."
        )


# ---------------------------------------------------------------------------
# Validated parsed record (consumed by the benchmark's sample builder)
# ---------------------------------------------------------------------------


class CIInformationType(BaseModel):
    """The contextual-integrity norm's allowed/disallowed information *types*.

    This is descriptive context from the dataset seed (kept for stratified
    reporting); it is not the per-instance ground truth (see ``CIAnnotation``).
    """

    model_config = ConfigDict(extra="allow")

    allowed: list[str] = Field(default_factory=list)
    disallowed: list[str] = Field(default_factory=list)


class CIAnnotation(BaseModel):
    """Per-instance ground truth: which stored values are appropriate to share.

    ``allowed`` / ``disallowed`` map an ``information`` field name to the
    canonical value token graders should look for in the agent's message.
    This is eval-only and is never exposed to the agent.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: dict[str, Any] = Field(default_factory=dict)
    disallowed: dict[str, Any] = Field(default_factory=dict)


class CISource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    split: str
    row_index: int


class ContextualIntegritySample(BaseModel):
    """A parsed Contextual Integrity record, validated for scenario building."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: CISource
    scenario: str
    domain: str
    transmission_principle: str
    user_intention: str
    sender: str
    recipient: str
    data_subject: str
    information_type: CIInformationType
    user_task: str
    information: dict[str, Any]
    annotation: CIAnnotation
