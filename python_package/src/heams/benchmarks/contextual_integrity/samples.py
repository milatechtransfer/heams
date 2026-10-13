"""Project parsed CI records into shared BenchmarkSample instances.

The full ``information`` pool becomes primed memories carrying only their real
``field_name`` provenance; the allowed/disallowed ground truth rides along in
``step_evaluation_markers``, which the agent never sees.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping

from heams.benchmarks.sample import BenchmarkSample
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import MemoryPriming, ScenarioPriming, UserStimulus
from heams.scenarios.loaders.contextual_integrity import ContextualIntegritySample

if TYPE_CHECKING:
    from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig


def _slug(value: str) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "unknown"
    )


def _information_to_entries(sample: ContextualIntegritySample) -> list[MemoryEntry]:
    """Prime the information pool as memories, with NO disclosure label.

    Metadata is restricted to the genuine field label. Memory metadata reaches
    the agent when memory reaches its prompt, so anything revealing that this is
    a benchmark (``source_benchmark``, ``scenario_id``) is deliberately omitted,
    as is the allowed/disallowed verdict under test.
    """
    return [
        MemoryEntry(content=str(value), metadata={"field_name": field})
        for field, value in sample.information.items()
    ]


def _disclosure_markers(annotation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"field": field, "value": str(value)} for field, value in annotation.items()
    ]


def record_to_benchmark_sample(
    record: Mapping[str, Any],
    *,
    config: CIBenchmarkConfig,
) -> BenchmarkSample:
    """Convert one parsed CI record into a shared BenchmarkSample."""

    sample = ContextualIntegritySample.model_validate(record)

    # Eval-only ground truth and stratifiers. Never rendered to the agent.
    markers: dict[str, Any] = {
        "sample_id": sample.id,
        "user_task": sample.user_task,
        "scenario": sample.scenario,
        "domain": sample.domain,
        "transmission_principle": sample.transmission_principle,
        "sender": sample.sender,
        "recipient": sample.recipient,
        "data_subject": sample.data_subject,
        "user_intention": sample.user_intention,
        "information_type": sample.information_type.model_dump(),
        "information_pool": {
            field: str(value) for field, value in sample.information.items()
        },
        "allowed_disclosures": _disclosure_markers(sample.annotation.allowed),
        "disallowed_disclosures": _disclosure_markers(sample.annotation.disallowed),
        "source": sample.source.model_dump(),
    }

    return BenchmarkSample(
        id=sample.id,
        name=f"Contextual Integrity {sample.id}",
        tags=[
            "ci",
            "contextual_integrity",
            f"scenario:{_slug(sample.scenario)}",
            f"domain:{_slug(sample.domain)}",
            f"principle:{_slug(sample.transmission_principle)}",
        ],
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(entries=_information_to_entries(sample))
        ),
        execution_sequence=[
            UserStimulus(
                content=config.user_task_prompt.format(user_task=sample.user_task),
                step_evaluation_markers=markers,
            )
        ],
        evaluators=[config.evaluator],
        ground_truth=None,
    )


__all__ = ["record_to_benchmark_sample"]
