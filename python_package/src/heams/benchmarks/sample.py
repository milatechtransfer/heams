"""Shared workload unit emitted by every new-path benchmark."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heams.scenarios.actions import MetaAction
from heams.scenarios.base import AgentConversation, Scenario, ScenarioPriming


class BenchmarkSample(BaseModel):
    """One workload unit; fields align with Scenario so mapping is mechanical."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    tags: list[str] = Field(default_factory=list)
    scenario_priming: ScenarioPriming = Field(default_factory=ScenarioPriming)
    execution_sequence: list[MetaAction] = Field(..., min_length=1)
    # EvaluatorConfig / BaseGroundTruth typed as Any to avoid import cycles with
    # evaluation.factory (which imports per-benchmark judge configs).
    evaluators: list[Any] = Field(default_factory=list)
    ground_truth: Any | None = None
    agent_conversation: AgentConversation = "stateful"

    def to_scenario(self) -> Scenario:
        return Scenario(
            id=self.id,
            name=self.name,
            tags=list(self.tags),
            scenario_priming=self.scenario_priming,
            execution_sequence=list(self.execution_sequence),
            evaluators=list(self.evaluators),
            ground_truth=self.ground_truth,
            agents=[],
            memory_systems=[],
            agent_conversation=self.agent_conversation,
        )
