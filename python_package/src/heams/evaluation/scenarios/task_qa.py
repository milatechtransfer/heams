"""Generic task QA evaluator.

This evaluator is intentionally benchmark-neutral: adapters can map external
single-turn task datasets into `UserStimulus` steps with target answers in
`step_evaluation_markers`, then reuse this scorer.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heams.evaluation.llm_judge import LlmJudge

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.utils import register


def _task_qa_record_tags(task: "TaskQA") -> dict[str, str]:
    """Benchmark-neutral stratifier tags persisted per task for reporting."""
    tags: dict[str, str] = {}
    if task.dataset:
        tags["dataset"] = task.dataset
    if task.domain:
        tags["domain"] = task.domain
    if task.difficulty:
        tags["difficulty"] = task.difficulty
    return tags


class TaskQA(BaseModel):
    """A single answerable task in an ordered QA-style scenario."""

    model_config = ConfigDict(extra="forbid")

    question: str
    expected_answer: str
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    difficulty: str | None = None
    domain: str | None = None
    # Source dataset label (e.g. ``mmlu_pro``) carried through so reports can
    # stratify per-dataset from serialized records without a dataset reload.
    dataset: str | None = None


class TaskQAGroundTruth(BaseGroundTruth):
    """Ordered ground truth for generic task QA scenarios."""

    tasks: list[TaskQA] = Field(default_factory=list)


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)
_EXPLICIT_LETTER_PATTERNS = [
    re.compile(
        r"(?:answer|correct answer|correct option|final answer|option|choice)"
        r"\s*(?:is|:)?\s*"
        r"(?:\*\*)?\(?([A-E])\)?(?:\*\*)?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\\boxed\{\s*([A-E])\s*\}", flags=re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:\*\*)?\(?([A-E])\)?[.)]", flags=re.IGNORECASE),
]
_EVO_MEMORY_CHOICE_PATTERNS = [
    re.compile(
        r"(?:answer|correct answer|correct option|final answer|option|choice)"
        r"\s*(?:is|:)?\s*"
        r"(?:\*\*)?\(?([A-J])\)?(?:\*\*)?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-J])\s*[.)]",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-J])\s+is\s+correct\b",
        flags=re.IGNORECASE,
    ),
]
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_AIME_INTEGER_RE = re.compile(r"\b(\d+)\b")


def normalize_answer(value: str) -> str:
    """Normalize answer text using the common QA exact-match convention."""

    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = _ARTICLE_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


_JUDGE_PROMPT = (
    "Question: {question}\n"
    "Expected answer: {expected_answer}\n\n"
    "Does the agent's answer convey the same factual information as the "
    "expected answer for this task? Minor phrasing differences are fine; "
    "contradictions, fabricated facts, or a missing key fact should fail."
)


@register("task_qa")
class TaskQAEvaluator(BaseEvaluator):
    """Evaluate ordered single-turn task answers."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is not None:
            return existing
        model_id = getattr(self.config, "judge_model_id", "openai:gpt-4o-mini")
        self._judge = LlmJudge(model_id=model_id)
        return self._judge

    def _score_response(
        self, response: AgentResponse, task: TaskQA
    ) -> tuple[bool, str]:
        mode = getattr(self.config, "match_mode", "normalized_exact")
        if mode == "llm_judge":
            prompt = _JUDGE_PROMPT.format(
                question=task.question,
                expected_answer=task.expected_answer,
            )
            verdict = self._get_judge().run(context=response.answer, question=prompt)
            passed = bool(getattr(verdict, "passed", False))
            reason = str(getattr(verdict, "reasoning", "") or "").strip()
            return passed, reason

        if mode == "evomemory_gpqa":
            actual_letter = extract_gpqa_choice_letter(response.answer)
            expected_letter = extract_gpqa_choice_letter(task.expected_answer)
            passed = actual_letter == expected_letter and expected_letter != ""
            if passed:
                return True, ""
            return (
                False,
                _mismatch("GPQA choice", actual_letter, expected_letter),
            )

        if mode == "evomemory_mmlu_pro":
            actual_letter = extract_mmlu_pro_choice_letter(response.answer)
            expected_letter = extract_mmlu_pro_choice_letter(task.expected_answer)
            passed = actual_letter == expected_letter and expected_letter != ""
            if passed:
                return True, ""
            return (
                False,
                _mismatch("MMLU-Pro choice", actual_letter, expected_letter),
            )

        if mode == "evomemory_aime":
            actual_number = extract_evomemory_aime_answer(response.answer)
            expected_number = extract_evomemory_aime_answer(task.expected_answer)
            passed = actual_number == expected_number and expected_number != ""
            if passed:
                return True, ""
            return (
                False,
                _mismatch("AIME answer", actual_number, expected_number),
            )

        if mode == "bfcl_ast":
            functions = (
                task.metadata.get("functions")
                if isinstance(task.metadata, dict)
                else None
            )
            passed = score_bfcl_ast(
                response.answer, task.expected_answer, functions or []
            )
            if passed:
                return True, ""
            return False, "BFCL call does not match the expected call"

        if mode == "multiple_choice_letter":
            actual_letter = extract_choice_letter(response.answer)
            expected_letter = extract_choice_letter(task.expected_answer)
            passed = actual_letter == expected_letter and expected_letter != ""
            if passed:
                return True, ""
            return (
                False,
                _mismatch("choice", actual_letter, expected_letter),
            )

        if mode == "numeric_answer":
            actual_number = extract_numeric_answer(response.answer)
            expected_number = extract_numeric_answer(task.expected_answer)
            passed = actual_number == expected_number and expected_number != ""
            if passed:
                return True, ""
            return (
                False,
                _mismatch("numeric answer", actual_number, expected_number),
            )

        if mode == "json_tool_call":
            passed = compare_tool_call_json(response.answer, task.expected_answer)
            if passed:
                return True, ""
            return False, "tool call does not match the expected call"

        actual = normalize_answer(response.answer)
        expected = normalize_answer(task.expected_answer)
        if mode == "contains":
            passed = expected != "" and expected in actual
        else:
            passed = actual == expected
        if passed:
            return True, ""
        relation = "does not contain" if mode == "contains" else "does not equal"
        return False, f"normalized answer {relation} the expected answer"

    def score_response(self, response: AgentResponse, task: TaskQA) -> tuple[bool, str]:
        """Score one task response for stream-time feedback."""

        return self._score_response(response, task)

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: TaskQAGroundTruth,
    ) -> AuditRecord:
        if len(agent_responses) != len(ground_truth.tasks):
            raise ValueError(
                f"task_qa audit expected {len(ground_truth.tasks)} agent responses, "
                f"got {len(agent_responses)}."
            )

        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        report_records: list[EvaluationRecord] = []
        for index, (response, task) in enumerate(
            zip(agent_responses, ground_truth.tasks)
        ):
            key = task.task_id or f"q{index}"
            ok, reason = self._score_response(response, task)
            report_records.append(
                EvaluationRecord(
                    key=key,
                    passed=ok,
                    tags=_task_qa_record_tags(task),
                    expected=task.expected_answer,
                    observed=response.answer,
                    reason=reason.strip() or None,
                )
            )
            if ok:
                passed.append(key)
            else:
                failed.append(key)
                if reason:
                    reasons.append(f"{key}: {reason}")

        total = len(passed) + len(failed)
        accuracy = len(passed) / total if total else 0.0
        threshold = float(getattr(self.config, "success_threshold", 1.0))
        notes = f"Evaluated {total} task QA responses."
        if reasons:
            notes = f"{notes} Failures: " + " | ".join(reasons[:5])

        self._last_audit = AuditRecord(
            is_successful=(total > 0 and accuracy >= threshold),
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=notes,
            report_records=report_records,
        )
        return self._last_audit


def _mismatch(label: str, actual: str, expected: str) -> str:
    """Reason for a failed extraction match; the full texts are in the record."""
    found = repr(actual) if actual else "nothing"
    return f"extracted {label} {found}, expected {expected!r}"


def extract_choice_letter(value: str) -> str:
    stripped = value.strip()
    upper = stripped.upper()
    if len(upper) == 1 and upper in {"A", "B", "C", "D", "E"}:
        return upper

    for pattern in _EXPLICIT_LETTER_PATTERNS:
        matches = pattern.findall(stripped)
        if matches:
            return matches[-1].upper()

    trailing = re.search(r"(?:^|\s)([A-E])[\s.)]*$", stripped, flags=re.IGNORECASE)
    if trailing:
        return trailing.group(1).upper()

    # Upstream GPQA accepts the first A-D anywhere. Keeping a constrained
    # fallback preserves that behavior without letting article "a" pass.
    match = re.search(r"\b([B-E])\b", stripped, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def extract_gpqa_choice_letter(value: str) -> str:
    return _extract_evomemory_choice_letter(value, valid_letters="ABCD")


def extract_mmlu_pro_choice_letter(value: str) -> str:
    return _extract_evomemory_choice_letter(value, valid_letters="ABCDEFGHIJ")


def _extract_evomemory_choice_letter(value: str, *, valid_letters: str) -> str:
    """Extract the final clearly labeled answer choice from an agent response."""

    stripped = value.strip()
    upper = stripped.upper()
    if len(upper) == 1 and upper in valid_letters:
        return upper

    for pattern in _EVO_MEMORY_CHOICE_PATTERNS:
        matches = [match.upper() for match in pattern.findall(stripped)]
        valid_matches = [match for match in matches if match in valid_letters]
        if valid_matches:
            return valid_matches[-1]
    return ""


def extract_numeric_answer(value: str) -> str:
    stripped = value.strip()

    boxed_matches = _BOXED_RE.findall(stripped)
    for boxed in reversed(boxed_matches):
        numbers = _NUMBER_RE.findall(boxed)
        if numbers:
            return _normalize_numeric_answer(numbers[-1])

    final_match = re.findall(
        r"(?:answer|final answer|therefore|thus)\D*(-?\d+(?:\.\d+)?)",
        stripped,
        flags=re.IGNORECASE,
    )
    if final_match:
        return _normalize_numeric_answer(final_match[-1])

    numbers = _NUMBER_RE.findall(stripped)
    if numbers:
        if len(numbers) == 1:
            return _normalize_numeric_answer(numbers[0])
        return _normalize_numeric_answer(numbers[-1])
    return stripped


def extract_evomemory_aime_answer(value: str) -> str:
    stripped = value.strip()

    boxed_match = re.search(r"\\boxed\{(\d+)\}", stripped)
    if boxed_match:
        return _normalize_aime_answer(boxed_match.group(1))

    answer_match = re.search(
        r"(?:answer|result)(?:\s+is)?[:\s]*(\d+)",
        stripped,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return _normalize_aime_answer(answer_match.group(1))

    numbers = _AIME_INTEGER_RE.findall(stripped)
    for number in reversed(numbers):
        normalized = _normalize_aime_answer(number)
        if normalized:
            return normalized

    return ""


def _normalize_aime_answer(value: str) -> str:
    normalized = str(int(value))
    numeric = int(normalized)
    if 0 <= numeric <= 999:
        return normalized
    return ""


def _normalize_numeric_answer(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def compare_tool_call_json(prediction: str, target: str) -> bool:
    target_obj = parse_tool_call_json(target)
    if not isinstance(target_obj, dict):
        return False

    for pred_obj in _iter_parsed_json_objects(prediction):
        if not isinstance(pred_obj, dict):
            continue
        if _tool_call_json_matches(pred_obj, target_obj):
            return True
    return False


def score_bfcl_ast(
    prediction: str,
    ground_truth_json: str,
    functions: list[Any],
) -> bool:
    """Faithful AST-style match against BFCL `possible_answer` ground truth.

    Mirrors the Berkeley Function-Calling Leaderboard checker for the Python
    `simple`/`multiple` subset: the predicted call must name a ground-truth
    function and, for that function, every required parameter must take one of
    the acceptable values (with type coercion), optional parameters may be
    omitted, and no parameter outside the ground truth may be supplied.
    """

    predicted = _parse_predicted_call(prediction)
    if predicted is None:
        return False
    pred_name, pred_args = predicted

    entries = _parse_bfcl_ground_truth(ground_truth_json)
    if not entries:
        return False

    schema_by_name = _bfcl_param_schema(functions)
    for entry in entries:
        for func_name, param_specs in entry.items():
            if pred_name != func_name:
                continue
            schema = schema_by_name.get(func_name, {})
            if _bfcl_args_match(pred_args, param_specs, schema):
                return True
    return False


def _parse_predicted_call(prediction: str) -> tuple[str, dict[str, Any]] | None:
    parsed = parse_tool_call_json(prediction)
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if isinstance(name, str) and name:
            arguments = parsed.get("arguments")
            return name, arguments if isinstance(arguments, dict) else {}
    return None


def _parse_bfcl_ground_truth(ground_truth_json: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(ground_truth_json)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(parsed, list):
        return [entry for entry in parsed if isinstance(entry, dict)]

    return []


def _bfcl_param_schema(functions: list[Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for function in functions or []:
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parameters = function.get("parameters", {})
        properties = (
            parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        )
        schema[name] = {
            "properties": properties if isinstance(properties, dict) else {},
        }
    return schema


def _bfcl_args_match(
    pred_args: dict[str, Any],
    param_specs: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    if any(key not in param_specs for key in pred_args):
        return False

    properties = schema.get("properties", {})
    for param, acceptable in param_specs.items():
        if not isinstance(acceptable, list):
            acceptable = [acceptable]
        optional = "" in acceptable
        if param not in pred_args:
            if optional:
                continue
            return False
        param_type = None
        prop = properties.get(param)
        if isinstance(prop, dict):
            param_type = prop.get("type")
        if not _bfcl_value_acceptable(pred_args[param], acceptable, param_type):
            return False
    return True


def _bfcl_value_acceptable(
    value: Any,
    acceptable: list[Any],
    param_type: str | None,
) -> bool:
    for candidate in acceptable:
        if candidate == "":
            if value == "":
                return True
            continue
        if _bfcl_values_equal(value, candidate, param_type):
            return True
    return False


def _bfcl_values_equal(value: Any, candidate: Any, param_type: str | None) -> bool:
    if value == candidate:
        return True

    if param_type in {"integer", "float", "number"} or isinstance(
        candidate, (int, float)
    ):
        try:
            if float(value) == float(candidate):
                return True
        except (TypeError, ValueError):
            pass

    if isinstance(candidate, str) and isinstance(value, str):
        if value.strip().lower() == candidate.strip().lower():
            return True

    if isinstance(candidate, list) and isinstance(value, list):
        if len(value) == len(candidate) and all(
            _bfcl_values_equal(item, expected, None)
            for item, expected in zip(value, candidate)
        ):
            return True

    return False


def _tool_call_json_matches(
    pred_obj: dict[str, Any], target_obj: dict[str, Any]
) -> bool:
    pred_name = str(pred_obj.get("name", "")).lower()
    target_name = str(target_obj.get("name", "")).lower()
    if pred_name != target_name:
        return False
    return pred_obj.get("arguments", {}) == target_obj.get("arguments", {})


def parse_tool_call_json(value: str) -> Any:
    for parsed in _iter_parsed_json_objects(value):
        return parsed
    return None


def _iter_parsed_json_objects(value: str) -> list[Any]:
    parsed_objects: list[Any] = []
    try:
        parsed_objects.append(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        pass

    for candidate in _iter_json_object_candidates(value):
        try:
            parsed_objects.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', value)
    if name_match:
        parsed_objects.append({"name": name_match.group(1)})
    return parsed_objects


def _iter_json_object_candidates(value: str) -> list[str]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.DOTALL)
    candidates = list(fenced)

    for start, char in enumerate(value):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(value)):
            current = value[end]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(value[start : end + 1])
                    break

    return candidates
