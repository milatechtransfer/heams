from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from heams.benchmarks.base import BaseBenchmark
from heams.benchmarks.factory import BenchmarkConfig, create_benchmark
from heams.benchmarks.long_mem_eval import LongMemEvalBenchmark
from heams.benchmarks.sample import BenchmarkSample
from heams.evaluation.factory import LongMemEvalLlmJudgeConfig
from heams.memory_systems import InContextConfig, create_memory
from heams.scenarios import Scenario
from heams.utils.settings import settings
from tests.benchmarks.long_mem_eval.fixtures import load_long_mem_eval_benchmark_config


def _fixture_payload() -> list[dict]:
    return [
        {
            "question_id": "LME-FIXTURE-1",
            "question_type": "single-session-user",
            "question": "Where did Alice move?",
            "answer": "Montreal",
            "question_date": "2024-03-02",
            "haystack_session_ids": ["S2", "S1"],
            "haystack_dates": [
                "2024/03/03 (Sun) 09:15",
                "2024/03/01 (Fri) 21:05",
            ],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "I am planning a trip to Toronto.",
                    },
                    {
                        "role": "assistant",
                        "content": "Let me know if you need hotel ideas.",
                    },
                ],
                [
                    {
                        "role": "user",
                        "content": "I moved to Montreal last spring.",
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "That sounds exciting.",
                    },
                ],
            ],
            "answer_session_ids": ["S1"],
        }
    ]


def _write_fixture(tmp_path: Path) -> Path:
    dataset = tmp_path / "longmemeval_s_cleaned.json"
    dataset.write_text(json.dumps(_fixture_payload()), encoding="utf-8")
    return dataset


def test_create_benchmark_yields_benchmark_samples(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)

    benchmark = create_benchmark(
        load_long_mem_eval_benchmark_config(examples_dir=dataset)
    )

    assert isinstance(benchmark, LongMemEvalBenchmark)
    assert benchmark.skipped == []
    samples = list(benchmark)
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, BenchmarkSample)
    assert sample.id == "LME-FIXTURE-1"
    assert "long_mem_eval" in sample.tags
    assert "split:s" in sample.tags
    assert len(sample.scenario_priming.memory.entries) == 4
    assert sample.scenario_priming.memory.entries[0].metadata == {
        "source": "long_mem_eval",
        "source_benchmark": "long_mem_eval",
        "session_id": "S1",
        "session_index": 0,
        "turn_index": 0,
        "ingest_group_id": "long_mem_eval:LME-FIXTURE-1:S1:0",
        "ingest_order": 0,
        "ingest_scope_id": "LME-FIXTURE-1",
        "role": "user",
        "ingest_role": "user",
        "timestamp": "2024/03/01 (Fri) 21:05",
        "ingest_timestamp_epoch": 1709327100,
        "temporal_received_at": "2024-03-01T21:05:00Z",
        "temporal_source": "long_mem_eval.session_date",
    }
    markers = sample.execution_sequence[0].step_evaluation_markers
    stimulus_prompt = sample.execution_sequence[0].content
    assert "Current Date: 2024-03-02" in stimulus_prompt
    assert "Question: Where did Alice move?" in stimulus_prompt
    assert "Use the memory recall tool" in stimulus_prompt
    assert markers["expected_answer"] == "Montreal"
    assert markers["question"] == "Where did Alice move?"
    assert markers["question_id"] == "LME-FIXTURE-1"
    assert markers["question_type"] == "single-session-user"
    assert markers["answer_session_ids"] == ["S1"]
    assert markers["retrieval_evidence"] == {
        "match_key": "session_id",
        "relevant_values": ["S1"],
        "relevant_contents": ["I moved to Montreal last spring."],
    }
    assert len(sample.evaluators) == 1
    assert isinstance(sample.evaluators[0], LongMemEvalLlmJudgeConfig)
    assert sample.ground_truth is not None
    assert sample.ground_truth.qa[0].expected_answer == "Montreal"


def test_long_mem_eval_implements_materialized_benchmark_contract(
    tmp_path: Path,
) -> None:
    dataset = _write_fixture(tmp_path)
    benchmark = create_benchmark(
        load_long_mem_eval_benchmark_config(examples_dir=dataset)
    )

    assert isinstance(benchmark, BaseBenchmark)
    assert list(benchmark) == list(benchmark.samples)
    assert len(benchmark) == len(benchmark.samples) == 1
    sample = benchmark.samples[0]
    assert isinstance(sample.to_scenario(), Scenario)
    assert sample.to_scenario().agents == []
    assert sample.to_scenario().memory_systems == []

    entries = sample.scenario_priming.memory.entries
    assert [entry.metadata["session_id"] for entry in entries] == [
        "S1",
        "S1",
        "S2",
        "S2",
    ]
    assert [entry.metadata["session_index"] for entry in entries] == [0, 0, 1, 1]
    assert [entry.metadata["ingest_group_id"] for entry in entries] == [
        "long_mem_eval:LME-FIXTURE-1:S1:0",
        "long_mem_eval:LME-FIXTURE-1:S1:0",
        "long_mem_eval:LME-FIXTURE-1:S2:0",
        "long_mem_eval:LME-FIXTURE-1:S2:0",
    ]
    assert [entry.metadata["ingest_order"] for entry in entries] == [0, 1, 0, 1]
    assert [entry.metadata["ingest_role"] for entry in entries] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_long_mem_eval_recall_omits_evaluator_only_labels(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)
    sample = create_benchmark(
        load_long_mem_eval_benchmark_config(examples_dir=dataset)
    ).samples[0]
    memory = create_memory(InContextConfig(), storage_dir=tmp_path / "memory")

    memory.prime_memory_entries(sample.scenario_priming.memory.entries)

    recalled = [json.loads(item) for item in memory.recall("ignored")]

    assert len(recalled) == 4
    for item in recalled:
        metadata = item["metadata"]
        assert "answer_session" not in metadata
        assert "has_answer" not in metadata
    markers = sample.execution_sequence[0].step_evaluation_markers
    assert markers["answer_session_ids"] == ["S1"]
    assert markers["retrieval_evidence"]["relevant_contents"] == [
        "I moved to Montreal last spring."
    ]


def test_long_mem_eval_config_validates_through_factory(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)

    config = load_long_mem_eval_benchmark_config(examples_dir=dataset, limit=1)

    assert config.examples_dir == dataset
    assert config.limit == 1
    assert config.answer_prompt
    assert isinstance(config.evaluator, LongMemEvalLlmJudgeConfig)


def test_long_mem_eval_examples_dir_defaults_to_settings_datasets_dir(
    tmp_path: Path,
) -> None:
    default_config = load_long_mem_eval_benchmark_config()

    assert default_config.examples_dir == settings.datasets_dir / "long_mem_eval"
    assert (
        load_long_mem_eval_benchmark_config(examples_dir=tmp_path).examples_dir
        == tmp_path
    )


def test_long_mem_eval_config_requires_prompt_and_evaluator() -> None:
    with pytest.raises(Exception):
        TypeAdapter(BenchmarkConfig).validate_python({"type": "long_mem_eval"})


@pytest.mark.parametrize(
    "answer_prompt",
    [
        "{question} {question_date}",
        "{question_date}: {question} ({question})",
    ],
)
def test_long_mem_eval_config_accepts_required_answer_prompt_fields(
    answer_prompt: str,
) -> None:
    config = load_long_mem_eval_benchmark_config(answer_prompt=answer_prompt)

    assert config.answer_prompt == answer_prompt


@pytest.mark.parametrize(
    "answer_prompt",
    ["{question}", "{question_date}", "{question} {question_date} {foo}"],
)
def test_long_mem_eval_config_rejects_invalid_answer_prompt_fields(
    answer_prompt: str,
) -> None:
    with pytest.raises(ValidationError, match="answer_prompt"):
        load_long_mem_eval_benchmark_config(answer_prompt=answer_prompt)


def test_long_mem_eval_limit_and_invalid_records_are_handled(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)
    payload = _fixture_payload()
    payload.append({"question_id": "invalid"})
    payload.append({**_fixture_payload()[0], "question_id": "LME-FIXTURE-2"})
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    benchmark = create_benchmark(
        load_long_mem_eval_benchmark_config(examples_dir=dataset, limit=1)
    )

    assert [sample.id for sample in benchmark] == ["LME-FIXTURE-1"]
    assert len(benchmark.skipped) == 1
    assert benchmark.skipped[0].path == dataset
    assert "sample invalid" in benchmark.skipped[0].reason
