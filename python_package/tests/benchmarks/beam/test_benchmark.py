from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import yaml
from pydantic import TypeAdapter

from heams.agents.base import AgentResponse
from heams.benchmarks.base import BaseBenchmark
from heams.benchmarks.beam.dataset import (
    extract_rubric_nuggets,
    iter_flat_turns,
    parse_beam_chat,
)
from heams.benchmarks.beam.judge import (
    BeamQAGroundTruth,
)
from heams.benchmarks.beam.samples import record_to_benchmark_sample
from heams.benchmarks.factory import (
    BeamBenchmarkConfig,
    BenchmarkConfig,
    create_benchmark,
)
from heams.benchmarks.sample import BenchmarkSample
from heams.evaluation.schemas import AuditRecord
from heams.orchestration.scenario_execution import execute_scenario
from heams.scenarios.actions import UserStimulus
from tests.benchmarks.beam.fixtures import _SMOKE_YAML
from tests.paths import TESTS_DIR

_FIXTURE = TESTS_DIR / "fixtures" / "beam" / "conversation.json"

_BENCHMARK_ADAPTER = TypeAdapter(BenchmarkConfig)


def _load_smoke_config(**overrides: object) -> BeamBenchmarkConfig:
    payload = yaml.safe_load(_SMOKE_YAML.read_text(encoding="utf-8"))
    payload.update(overrides)
    config = _BENCHMARK_ADAPTER.validate_python(payload)
    assert isinstance(config, BeamBenchmarkConfig)
    return config


def test_parse_beam_chat_layouts() -> None:
    nested = [[{"role": "user", "content": "hi", "id": 1}]]
    assert parse_beam_chat(nested)[0][0]["content"] == "hi"
    flat = [{"role": "user", "content": "hi", "id": 1}]
    assert iter_flat_turns(flat)[0]["content"] == "hi"
    plan = [
        {
            "plan-1": [
                {
                    "turns": [
                        [{"role": "user", "content": "from plan", "id": 9}],
                    ]
                }
            ]
        }
    ]
    assert iter_flat_turns(plan)[0]["content"] == "from plan"


def test_extract_rubric_nuggets_shapes() -> None:
    assert extract_rubric_nuggets(
        {"rubric": {"nuggets": [{"description": "A"}, "B"]}}
    ) == ["A", "B"]
    assert extract_rubric_nuggets({"rubric": ["one", "two"]}) == ["one", "two"]
    assert extract_rubric_nuggets({"rubric": "solo"}) == ["solo"]


def test_factory_dispatch_and_full_chat_priming() -> None:
    config = _load_smoke_config(local_path=_FIXTURE, limit=1, limit_stimuli=2)
    benchmark = create_benchmark(config)
    assert isinstance(benchmark, BaseBenchmark)
    assert config.type == "beam"
    assert len(benchmark.samples) == 1
    sample = benchmark.samples[0]
    assert isinstance(sample, BenchmarkSample)
    assert sample.id == "BEAM-100K-fixture-1"
    assert "split:100K" in sample.tags
    entries = sample.scenario_priming.memory.entries
    assert len(entries) == 3
    assert entries[0].metadata["ingest_group_id"] == "beam:fixture-1:0"
    assert "question_type" not in entries[0].metadata
    assert len(sample.execution_sequence) == 2
    assert isinstance(sample.execution_sequence[0], UserStimulus)
    contents = [item.content for item in sample.execution_sequence]
    assert any("Where does Alex live?" in content for content in contents)
    assert any("start a job" in content for content in contents)
    ground_truth = sample.ground_truth
    assert isinstance(ground_truth, BeamQAGroundTruth)
    assert {item.question_type for item in ground_truth.qa} == {
        "information_extraction",
        "event_ordering",
    }
    scenario = sample.to_scenario()
    assert scenario.id == sample.id
    assert len(scenario.execution_sequence) == 2
    assert sample.agent_conversation == "independent"
    assert scenario.agent_conversation == "independent"


def test_limit_stimuli_one_question() -> None:
    config = _load_smoke_config(local_path=_FIXTURE, limit=1, limit_stimuli=1)
    sample = create_benchmark(config).samples[0]
    assert len(sample.execution_sequence) == 1
    assert sample.ground_truth.qa[0].question_type == "event_ordering"


def test_yaml_round_trip_smoke_config() -> None:
    config = _BENCHMARK_ADAPTER.validate_python(
        yaml.safe_load(_SMOKE_YAML.read_text(encoding="utf-8"))
    )
    assert config.type == "beam"
    assert config.split == "100K"
    assert config.evaluator.type == "beam_llm_judge"
    formatted = config.evaluator.nugget_judge_template.format(
        question="Where?", nugget="Montreal", response="Montreal."
    )
    assert "Where?" in formatted
    assert '"score"' in formatted


def test_month_dash_time_anchor_sets_ingest_epoch() -> None:
    config = _load_smoke_config(local_path=_FIXTURE, limit=1, limit_stimuli=1)
    sample = record_to_benchmark_sample(
        {
            "conversation_id": "dates",
            "chat": [
                [
                    {
                        "id": 1,
                        "role": "user",
                        "content": "March turn",
                        "time_anchor": "March-15-2024",
                    },
                    {
                        "id": 2,
                        "role": "user",
                        "content": "April turn",
                        "time_anchor": "April-25-2024",
                    },
                ]
            ],
            "probing_questions": {
                "information_extraction": [
                    {
                        "question": "Where?",
                        "rubric": {"nuggets": ["x"]},
                    }
                ]
            },
        },
        config=config,
    )
    epochs = [
        entry.metadata["ingest_timestamp_epoch"]
        for entry in sample.scenario_priming.memory.entries
    ]
    assert epochs == [
        int(datetime(2024, 3, 15, tzinfo=UTC).timestamp()),
        int(datetime(2024, 4, 25, tzinfo=UTC).timestamp()),
    ]


def test_local_path_loads_fixture_list() -> None:
    config = _load_smoke_config(local_path=_FIXTURE, limit=1, limit_stimuli=1)
    assert len(create_benchmark(config).samples) == 1


def test_beam_probes_reset_agent_between_questions() -> None:
    config = _load_smoke_config(local_path=_FIXTURE, limit=1, limit_stimuli=2)
    scenario = create_benchmark(config).samples[0].to_scenario()
    first_question = scenario.execution_sequence[0].content

    class _Agent:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = [
                {"role": "system", "content": "seed"}
            ]
            self.seen_at_run: list[list[dict[str, str]]] = []

        def reset_state(self) -> None:
            self.messages = [{"role": "system", "content": "seed"}]

        def run(
            self, prompt: str, *, run_context: dict[str, Any] | None = None
        ) -> AgentResponse:
            self.messages.append({"role": "user", "content": prompt})
            self.seen_at_run.append(list(self.messages))
            self.messages.append({"role": "assistant", "content": "ok"})
            return AgentResponse(answer="ok")

    agent = _Agent()
    evaluator = MagicMock()
    evaluator.audit.return_value = AuditRecord(
        is_successful=True,
        accuracy_score=1.0,
        passed_keys=[],
        failed_keys=[],
        audit_notes="no-judge",
    )
    execute_scenario(
        scenario=scenario,
        agent=agent,
        environment=None,
        evaluator=evaluator,
        eval_cfg=scenario.evaluators[0],
        build_ground_truth=lambda sc, cfg: cfg.to_ground_truth(sc),
    )
    second = "\n".join(message["content"] for message in agent.seen_at_run[1])
    assert len(agent.seen_at_run) == 2
    assert first_question not in second
