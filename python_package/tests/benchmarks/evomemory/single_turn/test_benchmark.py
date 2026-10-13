from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from heams.benchmarks.evomemory import benchmark as benchmark_module
from heams.benchmarks.evomemory import data as evomemory
from heams.benchmarks.evomemory.config import EvoMemorySingleTurnBenchmarkConfig
from heams.benchmarks.evomemory.samples import (
    load_single_turn_streams,
    record_to_benchmark_sample,
)
from heams.benchmarks.factory import create_benchmark
from heams.scenarios import AgentManipulation, UserStimulus


def _tasks_payload() -> dict:
    return {
        "id": "EVOMEM-FIXTURE",
        "name": "Evo-Memory fixture",
        "tags": ["fixture"],
        "tasks": [
            {
                "task_id": "task-1",
                "input_text": "What color is the sky?",
                "target": "blue",
                "metadata": {"source": "unit"},
                "difficulty": "easy",
                "domain": "general",
            }
        ],
    }


def _sample_from_payload(
    payload: dict | list,
    *,
    match_mode: str = "normalized_exact",
):
    streams = load_single_turn_streams(payload)
    assert len(streams) == 1
    return record_to_benchmark_sample(
        streams[0],
        default_id="evomemory-single-turn-0",
        match_mode=match_mode,
    )


def test_record_to_benchmark_sample_accepts_upstream_mmlu_shape() -> None:
    sample = _sample_from_payload(
        [
            {
                "question": "What is the worst-case quicksort complexity?",
                "options": ["O(n)", "O(n log n)", "O(n^2)", "O(log n)"],
                "answer": "C",
                "answer_index": 2,
                "category": "computer_science",
            }
        ]
    )

    stimulus = sample.execution_sequence[0]
    assert "Options:" in stimulus.content
    assert "C. O(n^2)" in stimulus.content
    assert stimulus.step_evaluation_markers["expected_answer"] == "C"
    assert stimulus.step_evaluation_markers["metadata"]["answer_index"] == 2
    assert stimulus.step_evaluation_markers["domain"] == "computer_science"


def test_record_to_benchmark_sample_accepts_upstream_aime_shape() -> None:
    sample = _sample_from_payload(
        [{"problem": "Find the requested integer.", "answer": "038", "year": 2024}]
    )

    stimulus = sample.execution_sequence[0]
    feedback = sample.execution_sequence[1]
    assert stimulus.content == "Find the requested integer."
    assert stimulus.step_evaluation_markers["expected_answer"] == "038"
    assert stimulus.step_evaluation_markers["metadata"] == {"year": 2024}
    assert "immediately preceding task: 038\n" in feedback.content


def test_record_to_benchmark_sample_accepts_upstream_toolbench_shape() -> None:
    expected_answer = (
        '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
    )
    sample = _sample_from_payload(
        [
            {
                "question": "What's the weather in San Francisco?",
                "functions": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {"location": "string"},
                    }
                ],
                "answer": expected_answer,
                "category": "weather",
            }
        ]
    )

    stimulus = sample.execution_sequence[0]
    feedback = sample.execution_sequence[1]
    assert "Available APIs:" in stimulus.content
    assert "get_weather" in stimulus.content
    assert "Return only the JSON object" in stimulus.content
    assert "Do not call the listed APIs" in stimulus.content
    assert stimulus.step_evaluation_markers["domain"] == "weather"
    assert stimulus.step_evaluation_markers["metadata"]["functions"][0]["name"] == (
        "get_weather"
    )
    assert stimulus.step_evaluation_markers["expected_answer"] == expected_answer
    assert expected_answer in feedback.content


def test_record_to_benchmark_sample_keeps_framework_native_shape() -> None:
    sample = _sample_from_payload(_tasks_payload())

    assert sample.id == "EVOMEM-FIXTURE"
    assert sample.tags == [
        "evomemory",
        "single_turn",
        "evomemory-single-turn-0",
        "fixture",
    ]

    stimulus = sample.execution_sequence[0]
    assert stimulus.content == "What color is the sky?"
    assert stimulus.step_evaluation_markers["expected_answer"] == "blue"
    assert stimulus.step_evaluation_markers["task_id"] == "task-1"
    assert stimulus.step_evaluation_markers["metadata"] == {"source": "unit"}
    assert stimulus.step_evaluation_markers["difficulty"] == "easy"
    assert stimulus.step_evaluation_markers["domain"] == "general"
    ground_truth = sample.evaluators[0].to_ground_truth(sample.to_scenario())
    assert ground_truth.task_description == "Evo-Memory fixture"
    assert ground_truth.tasks[0].task_id == "task-1"
    assert ground_truth.tasks[0].dataset == "evomemory-single-turn-0"


def test_record_to_benchmark_sample_interleaves_corrective_feedback() -> None:
    sample = _sample_from_payload(
        {
            "dataset": "fixture",
            "tasks": [
                {"task_id": "task-1", "input_text": "Q1", "target": "A1"},
                {"task_id": "task-2", "input_text": "Q2", "target": "A2"},
            ],
        }
    )

    (
        task_one,
        feedback_one,
        reset_one,
        task_two,
        feedback_two,
        reset_two,
    ) = sample.execution_sequence

    assert isinstance(task_one, UserStimulus)
    assert isinstance(feedback_one, UserStimulus)
    assert isinstance(reset_one, AgentManipulation)
    assert isinstance(task_two, UserStimulus)
    assert isinstance(feedback_two, UserStimulus)
    assert isinstance(reset_two, AgentManipulation)
    assert task_one.content == "Q1"
    assert task_two.content == "Q2"
    assert task_one.step_evaluation_markers["evomemory_turn"] == "task"
    assert task_two.step_evaluation_markers["evomemory_turn"] == "task"
    assert feedback_one.step_evaluation_markers == {
        "source": "evomemory",
        "dataset": "fixture",
        "evomemory_turn": "feedback",
        "feedback_for_task_id": "task-1",
    }
    assert feedback_two.step_evaluation_markers["feedback_for_task_id"] == "task-2"
    assert "expected_answer" not in feedback_one.step_evaluation_markers
    assert "target" not in feedback_one.step_evaluation_markers
    assert "immediately preceding task: A1\n" in feedback_one.content
    assert "immediately preceding task: A2\n" in feedback_two.content
    assert (
        "call the final_answer tool with exactly: Feedback recorded."
        in feedback_one.content
    )
    assert reset_one.command == "reset_state"
    assert reset_two.command == "reset_state"

    ground_truth = sample.evaluators[0].to_ground_truth(sample.to_scenario())
    assert [task.task_id for task in ground_truth.tasks] == ["task-1", "task-2"]
    assert ground_truth.response_indices == [0, 2]


def test_record_to_benchmark_sample_passes_through_retrieval_evidence() -> None:
    sample = _sample_from_payload(
        {
            "id": "EVOMEM-RECALL",
            "tasks": [
                {
                    "task_id": "task-1",
                    "input_text": "Where is the key?",
                    "target": "desk drawer",
                    "retrieval_evidence": {
                        "match_key": "session_id",
                        "relevant_values": ["s1", "s2"],
                        "relevant_contents": ["The key is in the desk drawer."],
                    },
                }
            ],
        }
    )

    stimulus = sample.execution_sequence[0]
    assert stimulus.step_evaluation_markers["retrieval_evidence"] == {
        "match_key": "session_id",
        "relevant_values": ["s1", "s2"],
        "relevant_contents": ["The key is in the desk drawer."],
    }


def test_record_to_benchmark_sample_keeps_content_only_retrieval_evidence() -> None:
    sample = _sample_from_payload(
        {
            "id": "EVOMEM-RECALL-CONTENT",
            "tasks": [
                {
                    "task_id": "task-2",
                    "input_text": "What changed recently?",
                    "target": "new job",
                    "retrieval_evidence": {
                        "relevant_contents": [
                            "The user changed jobs last winter and became a park ranger."
                        ]
                    },
                }
            ],
        }
    )

    stimulus = sample.execution_sequence[0]
    assert stimulus.step_evaluation_markers["retrieval_evidence"] == {
        "relevant_contents": [
            "The user changed jobs last winter and became a park ranger."
        ]
    }


def test_record_to_benchmark_sample_limits_stimuli() -> None:
    streams = load_single_turn_streams(
        {
            "tasks": [
                {"input_text": "Q1", "target": "A1"},
                {"input_text": "Q2", "target": "A2"},
            ]
        }
    )

    sample = record_to_benchmark_sample(
        streams[0],
        default_id="fixture-0",
        match_mode="normalized_exact",
        limit_stimuli=1,
    )

    assert len(sample.execution_sequence) == 3
    assert [
        stimulus.step_evaluation_markers["evomemory_turn"]
        for stimulus in sample.execution_sequence
        if isinstance(stimulus, UserStimulus)
    ] == ["task", "feedback"]
    assert isinstance(sample.execution_sequence[2], AgentManipulation)
    assert sample.execution_sequence[2].command == "reset_state"
    ground_truth = sample.evaluators[0].to_ground_truth(sample.to_scenario())
    assert len(ground_truth.tasks) == 1
    assert ground_truth.tasks[0].expected_answer == "A1"
    assert ground_truth.response_indices == [0]


def test_typed_evomemory_samples_preserve_dataset_owned_evaluators(
    tmp_path: Path,
) -> None:
    payloads = {
        "aime_2024.json": [{"problem": "1 + 1", "answer": "2"}],
        "gpqa_diamond.json": [
            {
                "question": "Q",
                "options": ["a", "b", "c", "d"],
                "answer": "A",
            }
        ],
        "mmlu_pro.json": [
            {
                "question": "Q",
                "options": ["a", "b", "c", "d"],
                "answer": "B",
            }
        ],
        "toolbench.json": [
            {
                "question": "Q",
                "functions": [],
                "answer": "[]",
            }
        ],
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    benchmark = create_benchmark(
        EvoMemorySingleTurnBenchmarkConfig(examples_dir=tmp_path)
    )

    modes = {sample.id: sample.evaluators[0].match_mode for sample in benchmark.samples}
    assert modes == {
        "aime_2024": "evomemory_aime",
        "gpqa_diamond": "evomemory_gpqa",
        "mmlu_pro": "evomemory_mmlu_pro",
        "toolbench": "bfcl_ast",
    }
    assert {sample.evaluators[0].type for sample in benchmark.samples} == {
        "evomemory_single_turn_task_qa"
    }


def test_typed_evomemory_config_rejects_evaluator_override() -> None:
    with pytest.raises(ValidationError, match="evaluator"):
        EvoMemorySingleTurnBenchmarkConfig.model_validate(
            {
                "type": "evomemory_single_turn",
                "evaluator": {"type": "task_qa", "match_mode": "normalized_exact"},
            }
        )


def test_existing_default_dataset_is_not_rematerialized(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "aime_2024.json"
    dataset_path.write_text("cached", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module, "DEFAULT_SINGLE_TURN_DIR", tmp_path, raising=False
    )
    monkeypatch.setattr(
        benchmark_module,
        "materialize_single_turn_dataset",
        lambda path: pytest.fail("cached dataset was rematerialized"),
    )

    benchmark_module.EvoMemorySingleTurnBenchmark._materialize_dataset(dataset_path)

    assert dataset_path.read_text(encoding="utf-8") == "cached"


def test_build_aime_maps_problem_and_stringifies_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        evomemory,
        "_load_hf_dataset",
        lambda *args, **kwargs: [{"problem": "2+2?", "answer": 4}],
    )

    records = evomemory.build_aime(2024)

    assert records == [{"problem": "2+2?", "answer": "4", "year": 2024}]


def test_build_gpqa_targets_correct_shuffled_letter(monkeypatch) -> None:
    row = {
        "Question": "Q?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
        "High-level domain": "Physics",
    }
    monkeypatch.setattr(evomemory, "_load_hf_dataset", lambda *args, **kwargs: [row])

    record = evomemory.build_gpqa("diamond")[0]

    letter_index = ord(record["answer"]) - 65
    assert record["options"][letter_index] == "right"
    assert record["category"] == "Physics"
    assert record["domain"] == "Physics"
    assert sorted(record["options"]) == ["right", "w1", "w2", "w3"]


def test_build_mmlu_pro_passes_letter_answer(monkeypatch) -> None:
    row = {
        "question": "Q",
        "options": ["a", "b", "c"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
    }
    monkeypatch.setattr(evomemory, "_load_hf_dataset", lambda *args, **kwargs: [row])

    record = evomemory.build_mmlu_pro()[0]

    assert record["answer"] == "B"
    assert record["options"] == ["a", "b", "c"]
    assert record["category"] == "math"


def test_build_bfcl_toolbench_joins_ground_truth(monkeypatch) -> None:
    data = [
        {
            "id": "multiple_0",
            "question": [[{"role": "user", "content": "What is the weather?"}]],
            "function": [{"name": "get_weather"}],
        }
    ]
    answers = [{"id": "multiple_0", "ground_truth": [{"get_weather": {"x": [1]}}]}]

    def fake_fetch(relative_path: str):
        return answers if relative_path.startswith("possible_answer") else data

    monkeypatch.setattr(evomemory, "_fetch_bfcl_jsonl", fake_fetch)

    record = evomemory.build_bfcl_toolbench(categories=("multiple",))[0]

    assert record["question"] == "What is the weather?"
    assert record["functions"] == [{"name": "get_weather"}]
    assert json.loads(record["answer"]) == [{"get_weather": {"x": [1]}}]
    assert record["category"] == "bfcl_multiple"
    assert record["task_id"] == "multiple_0"


def test_build_bfcl_toolbench_rejects_items_without_ground_truth(monkeypatch) -> None:
    data = [
        {
            "id": "multiple_0",
            "question": [[{"role": "user", "content": "a"}]],
            "function": [],
        },
        {
            "id": "multiple_1",
            "question": [[{"role": "user", "content": "b"}]],
            "function": [],
        },
    ]
    answers = [{"id": "multiple_0", "ground_truth": [{"f": {}}]}]

    def fake_fetch(relative_path: str):
        return answers if relative_path.startswith("possible_answer") else data

    monkeypatch.setattr(evomemory, "_fetch_bfcl_jsonl", fake_fetch)

    with pytest.raises(ValueError, match="multiple_1"):
        evomemory.build_bfcl_toolbench(categories=("multiple",))


def test_build_gpqa_rejects_missing_answer_option(monkeypatch) -> None:
    row = {
        "Question": "Q?",
        "Correct Answer": "",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
    }
    monkeypatch.setattr(evomemory, "_load_hf_dataset", lambda *args, **kwargs: [row])

    with pytest.raises(ValueError, match="four answer options"):
        evomemory.build_gpqa("diamond")


def test_build_single_turn_dataset_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        evomemory.build_single_turn_dataset("unknown.json")
