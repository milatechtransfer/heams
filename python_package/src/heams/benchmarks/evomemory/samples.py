"""Project EvoMemory single-turn streams into shared BenchmarkSample instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from heams.benchmarks.sample import BenchmarkSample
from heams.evaluation.factory import (
    EvoMemorySingleTurnTaskQAEvaluatorConfig,
)
from heams.scenarios.actions import AgentManipulation, UserStimulus


def load_single_turn_streams(
    source: str | Path | list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    """Load and normalize supported EvoMemory single-turn payload shapes."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        default_id = Path(source).stem
    else:
        payload = source
        default_id = "evomemory-single-turn"

    return _normalize_streams(payload, default_id=default_id)


def record_to_benchmark_sample(
    stream: dict[str, Any],
    *,
    default_id: str,
    match_mode: str,
    limit_stimuli: int | None = None,
) -> BenchmarkSample:
    """Convert one EvoMemory single-turn stream into a BenchmarkSample."""
    tasks = stream.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Evo-Memory stream is missing a `tasks` list.")

    dataset_name = str(stream.get("dataset") or default_id)
    task_stimuli = [
        _task_to_stimulus(task, dataset_name=dataset_name, index=index)
        for index, task in enumerate(tasks)
    ]
    if limit_stimuli is not None:
        task_stimuli = task_stimuli[:limit_stimuli]
    if not task_stimuli:
        raise ValueError("Evo-Memory stream must contain at least one task.")
    stimuli = [
        turn
        for task_stimulus in task_stimuli
        for turn in (
            task_stimulus,
            _feedback_stimulus(task_stimulus),
            AgentManipulation(command="reset_state"),
        )
    ]

    sample_id = str(stream.get("id") or stream.get("stream_id") or default_id)
    name = str(stream.get("name") or f"Evo-Memory {sample_id}")
    tags = [
        "evomemory",
        "single_turn",
        dataset_name,
        *[str(tag) for tag in stream.get("tags", [])],
    ]

    return BenchmarkSample(
        id=sample_id,
        name=name,
        tags=tags,
        execution_sequence=stimuli,
        evaluators=[EvoMemorySingleTurnTaskQAEvaluatorConfig(match_mode=match_mode)],
    )


def _normalize_streams(payload: Any, *, default_id: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [{"id": default_id, "dataset": default_id, "tasks": payload}]
    if not isinstance(payload, dict):
        raise ValueError("Evo-Memory data must be a task list or object.")
    if isinstance(payload.get("streams"), list):
        return list(payload["streams"])
    if isinstance(payload.get("tasks"), list):
        return [payload]
    raise ValueError("Evo-Memory data must contain `tasks` or `streams`.")


def _task_to_stimulus(
    task: dict[str, Any],
    *,
    dataset_name: str,
    index: int,
) -> UserStimulus:
    input_text, target, metadata = _normalize_single_turn_task(
        task,
        dataset_name=dataset_name,
        index=index,
    )
    if not isinstance(input_text, str) or input_text == "":
        raise ValueError("Evo-Memory task is missing non-empty `input_text`.")
    if target is None or str(target) == "":
        raise ValueError("Evo-Memory task is missing non-empty `target`.")

    markers: dict[str, Any] = {
        "expected_answer": str(target),
        "target": str(target),
        "source": "evomemory",
        "dataset": dataset_name,
        "evomemory_turn": "task",
        **metadata,
    }
    for key in ("task_id", "metadata", "difficulty", "domain"):
        value = task.get(key)
        if value is not None:
            markers[key] = value
    markers.setdefault("task_id", f"{dataset_name}_{index}")
    _apply_retrieval_evidence_markers(markers, task)

    return UserStimulus(content=input_text, step_evaluation_markers=markers)


def _feedback_stimulus(task_stimulus: UserStimulus) -> UserStimulus:
    markers = task_stimulus.step_evaluation_markers
    expected_answer = markers["expected_answer"]
    task_id = markers["task_id"]
    return UserStimulus(
        content=(
            "Correct answer for the immediately preceding task: "
            f"{expected_answer}\n"
            "Use this feedback to improve future answers. If useful and "
            "available, use the memory tools to retain this correction. "
            "Do not answer the completed task again. After any optional memory "
            "operation, call the final_answer tool with exactly: Feedback recorded."
        ),
        step_evaluation_markers={
            "source": "evomemory",
            "dataset": markers["dataset"],
            "evomemory_turn": "feedback",
            "feedback_for_task_id": task_id,
        },
    )


def _normalize_single_turn_task(
    task: dict[str, Any],
    *,
    dataset_name: str,
    index: int,
) -> tuple[str | None, Any, dict[str, Any]]:
    if "input_text" in task or "target" in task:
        return task.get("input_text"), task.get("target"), {}

    if "problem" in task:
        return (
            task.get("problem"),
            task.get("answer"),
            {
                "task_id": f"{dataset_name}_{index}",
                "metadata": {"year": task.get("year")},
                "domain": "math",
                "difficulty": "hard",
            },
        )

    question = task.get("question")
    options = task.get("options") or []
    functions = task.get("functions") or []

    if functions:
        return (
            _render_toolbench_prompt(str(question), functions),
            task.get("answer"),
            {
                "task_id": f"{dataset_name}_{index}",
                "metadata": {"functions": functions, "category": task.get("category")},
                "domain": task.get("category", "api"),
            },
        )

    if options:
        return (
            _render_options_prompt(str(question), options),
            task.get("answer"),
            {
                "task_id": f"{dataset_name}_{index}",
                "metadata": {
                    "options": options,
                    "answer_index": task.get("answer_index"),
                    "correct_answer_text": task.get("correct_answer_text"),
                    "category": task.get("category"),
                },
                "domain": task.get("domain") or task.get("category"),
            },
        )

    return question, task.get("answer"), {"task_id": f"{dataset_name}_{index}"}


def _apply_retrieval_evidence_markers(
    markers: dict[str, Any],
    task: dict[str, Any],
) -> None:
    explicit = task.get("retrieval_evidence")
    if isinstance(explicit, dict):
        match_key = explicit.get("match_key")
        relevant_values = explicit.get("relevant_values")
        if isinstance(relevant_values, (list, tuple, set)):
            normalized_values = [
                str(value)
                for value in relevant_values
                if value is not None and str(value)
            ]
        elif relevant_values is None:
            normalized_values = []
        else:
            normalized_values = [str(relevant_values)]

        relevant_contents = explicit.get("relevant_contents")
        if isinstance(relevant_contents, (list, tuple, set)):
            normalized_contents = [
                str(value)
                for value in relevant_contents
                if value is not None and str(value)
            ]
        elif relevant_contents is None:
            normalized_contents = []
        else:
            normalized_contents = [str(relevant_contents)]

        if (isinstance(match_key, str) and normalized_values) or normalized_contents:
            markers["retrieval_evidence"] = {}
            if isinstance(match_key, str) and normalized_values:
                markers["retrieval_evidence"]["match_key"] = match_key
                markers["retrieval_evidence"]["relevant_values"] = normalized_values
            if normalized_contents:
                markers["retrieval_evidence"]["relevant_contents"] = normalized_contents
            return

    evidence_dia_ids = task.get("evidence_dia_ids")
    if isinstance(evidence_dia_ids, list) and evidence_dia_ids:
        markers["evidence_dia_ids"] = [str(value) for value in evidence_dia_ids]

    answer_session_ids = task.get("answer_session_ids")
    if isinstance(answer_session_ids, list) and answer_session_ids:
        markers["answer_session_ids"] = [str(value) for value in answer_session_ids]


def _render_options_prompt(question: str, options: list[Any]) -> str:
    options_text = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return f"{question}\n\nOptions:\n{options_text}"


def _render_toolbench_prompt(question: str, functions: list[Any]) -> str:
    descriptions = []
    for function in functions:
        if not isinstance(function, dict):
            continue
        name = function.get("name", "unknown")
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        descriptions.append(f"- {name}: {description} Parameters: {parameters}")
    if not descriptions:
        return question
    return (
        f"{question}\n\n"
        "Available APIs:\n" + "\n".join(descriptions) + "\n\n"
        "Return only the JSON object for the API call you would make, with "
        "`name` and `arguments` fields. Do not call the listed APIs or include "
        "prose."
    )
