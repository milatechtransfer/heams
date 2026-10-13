from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from heams.agents.base import AgentResponse
from heams.benchmarks.factory import create_benchmark
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord
from heams.memory_systems import InContextConfig, create_memory
from heams.orchestration.scenario_execution import execute_scenario
from tests.benchmarks.locomo.fixtures import load_locomo_benchmark_config

_Q1 = "Where does Alice live?"
_Q2 = "What did Bob ask about winter?"


class _WrapperMessageAgent:
    """Same message protocol as TinyAgent.run / reset_state (no model API)."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = [{"role": "system", "content": "seed"}]
        self.messages_at_run: list[list[dict[str, str]]] = []

    def reset_state(self) -> None:
        self.messages = [{"role": "system", "content": "seed"}]

    def run(
        self, prompt: str, *, run_context: dict[str, Any] | None = None
    ) -> AgentResponse:
        self.messages.append({"role": "user", "content": prompt})
        self.messages_at_run.append(list(self.messages))
        self.messages.append({"role": "assistant", "content": "ok"})
        return AgentResponse(answer="ok")


class _NoJudge(BaseEvaluator):
    def audit(self, agent_responses, ground_truth) -> AuditRecord:
        self._last_audit = AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=[],
            failed_keys=[],
            audit_notes="no-judge",
        )
        return self._last_audit


def _two_qa_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "locomo_two_qa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "LOC-ISOLATION-1",
                    "conversation": {
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "text": "I moved to Montreal last spring.",
                                "dia_id": "D1:1",
                            },
                            {
                                "speaker": "Bob",
                                "text": "Nice, how was the winter?",
                                "dia_id": "D1:2",
                            },
                        ],
                        "session_1_date_time": "2024-03-01 10:00:00",
                    },
                    "qa": [
                        {
                            "question": _Q1,
                            "answer": "Montreal",
                            "evidence": ["D1:1"],
                            "category": 4,
                        },
                        {
                            "question": _Q2,
                            "answer": "how was the winter?",
                            "evidence": ["D1:2"],
                            "category": 4,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("agent_conversation", ["independent", "stateful"])
def test_execute_scenario_isolates_locomo_qa_when_independent(
    tmp_path: Path,
    agent_conversation: Literal["independent", "stateful"],
) -> None:
    sample = list(
        create_benchmark(
            load_locomo_benchmark_config(
                examples_dir=_two_qa_fixture(tmp_path),
                limit=1,
                limit_stimuli=2,
                agent_conversation=agent_conversation,
            )
        )
    )[0]
    scenario = sample.to_scenario()
    memory = create_memory(InContextConfig(), storage_dir=tmp_path / "memory")
    memory.prime_memory_entries(scenario.scenario_priming.memory.entries)
    agent = _WrapperMessageAgent()

    execute_scenario(
        scenario=scenario,
        agent=agent,
        environment=None,
        evaluator=_NoJudge("locomo-isolation"),
        eval_cfg=scenario.evaluators[0],
        build_ground_truth=lambda sc, cfg: cfg.to_ground_truth(sc),
    )

    second = "\n".join(message["content"] for message in agent.messages_at_run[1])
    recalled = "\n".join(memory.recall(query="", top_k=10))
    assert len(agent.messages_at_run) == 2
    assert _Q1 in scenario.execution_sequence[0].content
    assert _Q2 in second
    assert "Montreal" in recalled
    if agent_conversation == "independent":
        assert _Q1 not in second
    else:
        assert _Q1 in second
