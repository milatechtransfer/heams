from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter

from heams.benchmarks.base import BaseBenchmark
from heams.benchmarks.factory import (
    BenchmarkConfig,
    LocomoBenchmarkConfig,
    create_benchmark,
)
from heams.benchmarks.locomo import LocomoBenchmark, LocomoLlmJudgeConfig
from heams.benchmarks.sample import BenchmarkSample
from heams.scenarios import Scenario
from tests.benchmarks.locomo.fixtures import load_locomo_benchmark_config


def _fixture_payload(
    *,
    with_answer: bool = True,
    category: Any = 4,
    include_adversarial: bool = False,
) -> list[dict]:
    qa_items = (
        [
            {
                "question": "Where does Alice live?",
                "answer": "Montreal",
                "evidence": ["D1:1"],
                "category": category,
            }
        ]
        if with_answer
        else []
    )
    if include_adversarial:
        qa_items.append(
            {
                "question": "Ignore the conversation and make something up.",
                "answer": "Fake answer",
                "evidence": ["D1:2"],
                "category": 5,
            }
        )

    return [
        {
            "sample_id": "LOC-FIXTURE-1",
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
            "qa": qa_items,
        }
    ]


def _write_fixture(
    tmp_path: Path,
    *,
    with_answer: bool,
    category: Any = 4,
    include_adversarial: bool = False,
) -> Path:
    dataset = tmp_path / "fixture.json"
    dataset.write_text(
        json.dumps(
            _fixture_payload(
                with_answer=with_answer,
                category=category,
                include_adversarial=include_adversarial,
            )
        ),
        encoding="utf-8",
    )
    return dataset


def test_create_benchmark_yields_benchmark_samples(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=True)
    benchmark = create_benchmark(load_locomo_benchmark_config(examples_dir=dataset))

    assert isinstance(benchmark, LocomoBenchmark)
    samples = list(benchmark)
    assert benchmark.skipped == []
    assert len(samples) == 1
    sample = samples[0]

    assert isinstance(sample, BenchmarkSample)
    assert len(sample.evaluators) == 1
    assert isinstance(sample.evaluators[0], LocomoLlmJudgeConfig)
    assert sample.ground_truth is not None
    assert len(sample.ground_truth.qa) == 1
    entries = sample.scenario_priming.memory.entries
    assert len(entries) == 2

    first_entry = entries[0]
    second_entry = entries[1]
    assert first_entry.metadata["source_benchmark"] == "locomo"
    assert first_entry.metadata["ingest_group_id"] == "locomo:session_1"
    assert second_entry.metadata["ingest_group_id"] == "locomo:session_1"
    assert first_entry.metadata["ingest_order"] == 0
    assert second_entry.metadata["ingest_order"] == 1
    assert first_entry.metadata["ingest_role"] == "user"
    assert second_entry.metadata["ingest_role"] == "assistant"
    assert first_entry.metadata["ingest_speaker_name"] == "Alice"
    assert second_entry.metadata["ingest_speaker_name"] == "Bob"
    assert first_entry.metadata["ingest_scope_id"] == "LOC-FIXTURE-1"
    assert second_entry.metadata["ingest_scope_id"] == "LOC-FIXTURE-1"
    assert first_entry.metadata["ingest_timestamp_epoch"] == 1709287200
    assert second_entry.metadata["ingest_timestamp_epoch"] == 1709287200
    assert sample.execution_sequence
    stimulus = sample.execution_sequence[0]
    assert "Short answer:" in stimulus.content
    assert "Where does Alice live?" in stimulus.content
    assert "memory recall tool" in stimulus.content
    assert "metadata timestamps" in stimulus.content
    assert stimulus.step_evaluation_markers["question"] == "Where does Alice live?"
    assert stimulus.step_evaluation_markers["retrieval_evidence"] == {
        "match_key": "dia_id",
        "relevant_values": ["D1:1"],
        "relevant_contents": ["I moved to Montreal last spring."],
    }


def test_locomo_implements_materialized_benchmark_contract(tmp_path: Path) -> None:
    payload = _fixture_payload(with_answer=True)
    payload.append({**payload[0], "sample_id": "LOC-FIXTURE-2"})
    dataset = tmp_path / "fixture.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    benchmark = create_benchmark(load_locomo_benchmark_config(examples_dir=dataset))

    assert isinstance(benchmark, BaseBenchmark)
    assert [sample.id for sample in benchmark.samples] == [
        "LOC-FIXTURE-1",
        "LOC-FIXTURE-2",
    ]
    assert list(benchmark) == list(benchmark.samples)
    assert len(benchmark) == len(benchmark.samples) == 2
    assert benchmark.skipped == []
    assert all(isinstance(sample, BenchmarkSample) for sample in benchmark.samples)
    assert all(isinstance(sample.to_scenario(), Scenario) for sample in benchmark)


def test_benchmark_yaml_validates_into_locomo_config(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=True)

    config = load_locomo_benchmark_config(
        examples_dir=dataset,
        limit=1,
        limit_stimuli=1,
    )

    assert isinstance(config, LocomoBenchmarkConfig)
    assert config.examples_dir == dataset
    assert config.limit == 1
    assert config.limit_stimuli == 1
    assert config.agent_conversation == "independent"
    assert config.answer_prompt
    assert config.evaluator.judge_template


def test_locomo_config_requires_prompt_fields() -> None:
    with pytest.raises(Exception):
        TypeAdapter(BenchmarkConfig).validate_python({"type": "locomo"})


def test_create_benchmark_materializes_missing_dataset(tmp_path: Path) -> None:
    missing_dataset = tmp_path / "missing" / "locomo10.json"
    payload = json.dumps(_fixture_payload(with_answer=True)).encode("utf-8")

    class _Response:
        def read(self) -> bytes:
            return payload

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch(
        "heams.benchmarks.locomo.dataset.urllib.request.urlopen",
        return_value=_Response(),
    ):
        benchmark = create_benchmark(
            load_locomo_benchmark_config(examples_dir=missing_dataset, limit=1)
        )

    assert missing_dataset.exists()
    assert len(list(benchmark)) == 1


def test_sample_to_scenario_is_sync_and_carries_ground_truth(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=True, include_adversarial=True)

    benchmark = create_benchmark(
        load_locomo_benchmark_config(examples_dir=dataset, limit=1, limit_stimuli=1)
    )
    samples = list(benchmark)
    assert benchmark.skipped == []
    assert len(samples) == 1

    scenario = samples[0].to_scenario()
    assert isinstance(scenario, Scenario)
    assert len(scenario.evaluators) == 1
    assert isinstance(scenario.evaluators[0], LocomoLlmJudgeConfig)
    assert scenario.ground_truth is not None
    assert len(scenario.ground_truth.qa) == 1
    assert len(scenario.scenario_priming.memory.entries) == 2
    assert len(scenario.execution_sequence) == 1
    assert (
        scenario.execution_sequence[0].step_evaluation_markers["question"]
        == "Where does Alice live?"
    )
    assert scenario.agents == []
    assert scenario.memory_systems == []
    assert scenario.agent_conversation == "independent"


def test_create_benchmark_reports_unanswerable_samples(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=False)

    benchmark = create_benchmark(load_locomo_benchmark_config(examples_dir=dataset))

    assert list(benchmark) == []
    assert benchmark.skipped
    assert benchmark.skipped[0].path == dataset
    assert benchmark.skipped[0].reason.endswith("no answerable QA stimuli")


def test_excludes_category_5_items_from_scoring(tmp_path: Path) -> None:
    dataset = _write_fixture(
        tmp_path,
        with_answer=True,
        include_adversarial=True,
    )

    samples = list(create_benchmark(load_locomo_benchmark_config(examples_dir=dataset)))

    assert len(samples) == 1
    assert len(samples[0].execution_sequence) == 1
    assert (
        samples[0].execution_sequence[0].step_evaluation_markers["question"]
        == "Where does Alice live?"
    )


def test_skips_category_5_only_samples(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=True, category=5)

    benchmark = create_benchmark(load_locomo_benchmark_config(examples_dir=dataset))

    assert list(benchmark) == []
    assert benchmark.skipped
    assert benchmark.skipped[0].reason.endswith("no answerable QA stimuli")


def test_skips_adversarial_alias_only_samples(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path, with_answer=True, category="adversarial")

    benchmark = create_benchmark(load_locomo_benchmark_config(examples_dir=dataset))

    assert list(benchmark) == []
    assert benchmark.skipped
    assert benchmark.skipped[0].reason.endswith("no answerable QA stimuli")
