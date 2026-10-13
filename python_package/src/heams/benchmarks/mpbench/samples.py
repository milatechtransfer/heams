"""Project normalized MPBench records into BenchmarkSample instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from heams.benchmarks.mpbench.dataset import MpbenchRecord
from heams.benchmarks.mpbench.judge import MpbenchGroundTruth
from heams.benchmarks.sample import BenchmarkSample
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import ScenarioPriming, UserStimulus
from heams.scenarios.base import MemoryPriming

if TYPE_CHECKING:
    from heams.benchmarks.mpbench.config import MpbenchBenchmarkConfig


def record_to_benchmark_sample(
    record: MpbenchRecord,
    *,
    config: MpbenchBenchmarkConfig,
) -> BenchmarkSample:
    tags = [
        f"attack_class:{record.attack_class}",
        f"domain:{record.domain}",
        f"attack_signal:{record.attack_signal}",
        f"adversarial_goal:{record.adversarial_goal}",
    ]

    question = record.retrieval_query
    # Prime payload text only. Recall serializes MemoryEntry.metadata into the
    # tool string, so attack labels here would leak to the task agent.
    return BenchmarkSample(
        id=record.id,
        name=f"mpbench:{record.attack_class}:{record.id}",
        tags=tags,
        scenario_priming=ScenarioPriming(
            memory=MemoryPriming(entries=[MemoryEntry(content=record.expected_memory)])
        ),
        execution_sequence=[
            UserStimulus(
                content=config.answer_prompt.format(question=question),
                step_evaluation_markers={
                    "question": question,
                    "question_id": record.id,
                },
            )
        ],
        evaluators=[config.evaluator],
        ground_truth=MpbenchGroundTruth(
            task_description=f"mpbench:{record.attack_class}:{record.id}",
            sample_id=record.id,
            question=question,
            expected_memory=record.expected_memory,
            attack_class=record.attack_class,
            domain=record.domain,
            attack_signal=record.attack_signal,
            adversarial_goal=record.adversarial_goal,
        ),
    )


__all__ = ["record_to_benchmark_sample"]
