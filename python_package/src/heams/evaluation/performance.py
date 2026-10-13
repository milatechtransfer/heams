"""Post-run aggregation of component-level retrieval performance metrics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from heams.evaluation.llm_judge import LlmJudge

from heams.evaluation.schemas import (
    ComponentMetricsSummary,
    ComponentPerformanceMetricsSummary,
)
from heams.utils.event_loop import ensure_thread_event_loop
from heams.memory_systems.schemas import MemoryOperationEvent, MemoryOperationType
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


_CONTENT_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}

_RETRIEVAL_JUDGE_PROMPT = (
    "Reference evidence:\n{reference_evidence}\n\n"
    "Retrieved memory candidate:\n{candidate_text}\n\n"
    "Does the retrieved memory candidate contain information that is relevant "
    "support for the reference evidence? Accept paraphrases and semantically "
    "equivalent wording. Reject unrelated, contradictory, or unsupported "
    "content."
)


@dataclass(frozen=True)
class RetrievalEvidence:
    match_key: str | None
    relevant_values: tuple[str, ...] = ()
    relevant_contents: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedItemCandidate:
    rank: int
    content: str | None
    metadata: dict[str, Any] | None
    normalized_content: str


def _normalize_retrieval_evidence(
    markers: dict[str, Any] | None,
) -> RetrievalEvidence | None:
    if not isinstance(markers, dict):
        return None

    explicit = markers.get("retrieval_evidence")
    if isinstance(explicit, dict):
        match_key = explicit.get("match_key")
        raw_values = explicit.get("relevant_values")
        normalized_match_key = str(match_key) if isinstance(match_key, str) else None
        if isinstance(raw_values, (list, tuple, set)):
            relevant_values = [
                str(value) for value in raw_values if value is not None and str(value)
            ]
        elif raw_values is None:
            relevant_values = []
        else:
            relevant_values = [str(raw_values)]

        raw_contents = explicit.get("relevant_contents")
        if isinstance(raw_contents, (list, tuple, set)):
            relevant_contents = [
                str(value) for value in raw_contents if value is not None and str(value)
            ]
        elif raw_contents is None:
            relevant_contents = []
        else:
            relevant_contents = [str(raw_contents)]

        if relevant_contents or (normalized_match_key and relevant_values):
            return RetrievalEvidence(
                match_key=normalized_match_key,
                relevant_values=tuple(relevant_values),
                relevant_contents=tuple(relevant_contents),
            )

    evidence_dia_ids = markers.get("evidence_dia_ids") or []
    if isinstance(evidence_dia_ids, list) and evidence_dia_ids:
        return RetrievalEvidence(
            match_key="dia_id",
            relevant_values=tuple(str(value) for value in evidence_dia_ids),
        )

    answer_session_ids = markers.get("answer_session_ids") or []
    if isinstance(answer_session_ids, list) and answer_session_ids:
        return RetrievalEvidence(
            match_key="session_id",
            relevant_values=tuple(str(value) for value in answer_session_ids),
        )

    return None


def _build_retrieval_evidence_by_stimulus(
    scenario: Scenario,
) -> dict[int, RetrievalEvidence]:
    evidence_by_stimulus: dict[int, RetrievalEvidence] = {}
    for stimulus_index, step in enumerate(scenario.execution_sequence):
        if not isinstance(step, UserStimulus):
            continue
        evidence = _normalize_retrieval_evidence(step.step_evaluation_markers)
        if evidence is not None:
            evidence_by_stimulus[stimulus_index] = evidence
    return evidence_by_stimulus


def _get_event_stimulus_index(event: MemoryOperationEvent) -> int | None:
    payload = event.payload if isinstance(event.payload, dict) else {}
    run_context = payload.get("run_context")
    if not isinstance(run_context, dict):
        return None
    stimulus_index = run_context.get("stimulus_index")
    if isinstance(stimulus_index, bool):
        return None
    if isinstance(stimulus_index, int):
        return stimulus_index
    return None


def _parse_returned_item(item: Any) -> tuple[str | None, dict[str, Any] | None]:
    if isinstance(item, dict):
        content = item.get("content")
        metadata = item.get("metadata")
        return (
            str(content) if isinstance(content, str) else None,
            metadata if isinstance(metadata, dict) else None,
        )

    if not isinstance(item, str):
        return None, None

    try:
        parsed = json.loads(item)
    except (TypeError, ValueError, json.JSONDecodeError):
        return item, None

    if not isinstance(parsed, dict):
        return item, None

    content = parsed.get("content")
    metadata = parsed.get("metadata")
    return (
        str(content) if isinstance(content, str) else None,
        metadata if isinstance(metadata, dict) else None,
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _CONTENT_MATCH_STOPWORDS
    }


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(sorted(_content_tokens(text)))


def _claim_fragments(text: str) -> list[str]:
    fragments = [
        fragment.strip()
        for fragment in re.split(r"[.!?;\n]+", text)
        if fragment and fragment.strip()
    ]
    return fragments or [text]


def _content_matches_relevant(
    candidate_content: str | None, relevant_contents: list[str]
) -> bool:
    if not candidate_content:
        return False

    candidate_tokens = _content_tokens(candidate_content)
    if not candidate_tokens:
        return False

    for relevant_content in relevant_contents:
        relevant_tokens = _content_tokens(relevant_content)
        if not relevant_tokens:
            continue
        overlap = candidate_tokens & relevant_tokens
        if len(overlap) >= 3 and (len(overlap) / len(relevant_tokens)) >= 0.5:
            return True

    return False


def _content_supports_claim(
    candidate_content: str | None, relevant_content: str
) -> bool:
    if not candidate_content:
        return False

    candidate_tokens = _content_tokens(candidate_content)
    if not candidate_tokens:
        return False

    for fragment in _claim_fragments(relevant_content):
        fragment_tokens = _content_tokens(fragment)
        if not fragment_tokens:
            continue
        overlap = candidate_tokens & fragment_tokens
        minimum_overlap = min(3, len(fragment_tokens))
        if (
            len(overlap) >= minimum_overlap
            and (len(overlap) / len(fragment_tokens)) >= 0.6
        ):
            return True

    return False


def _get_retrieval_match_mode(eval_cfg: Any | None) -> str:
    mode = getattr(eval_cfg, "retrieval_match_mode", "deterministic")
    if mode in {"deterministic", "llm_judge"}:
        return str(mode)
    return "deterministic"


def _get_retrieval_judge_model_id(eval_cfg: Any | None) -> str:
    explicit_model_id = getattr(eval_cfg, "retrieval_judge_model_id", None)
    if isinstance(explicit_model_id, str) and explicit_model_id:
        return explicit_model_id
    fallback_model_id = getattr(eval_cfg, "judge_model_id", "openai:gpt-4o-mini")
    return str(fallback_model_id)


@lru_cache(maxsize=8)
def _get_retrieval_judge(model_id: str) -> LlmJudge:
    return LlmJudge(model_id=model_id)


def _candidate_text_for_judge(candidate: RetrievedItemCandidate) -> str:
    parts: list[str] = []
    if candidate.content:
        parts.append(f"Content: {candidate.content}")
    if isinstance(candidate.metadata, dict) and candidate.metadata:
        parts.append(
            "Metadata: "
            + json.dumps(candidate.metadata, ensure_ascii=True, sort_keys=True)
        )
    return "\n".join(parts)


def _build_candidate_items(returned_items: list[Any]) -> list[RetrievedItemCandidate]:
    candidates: list[RetrievedItemCandidate] = []
    for rank, item in enumerate(returned_items, start=1):
        content, metadata = _parse_returned_item(item)
        candidates.append(
            RetrievedItemCandidate(
                rank=rank,
                content=content,
                metadata=metadata,
                normalized_content=_normalize_text(content),
            )
        )
    return candidates


def _match_by_normalized_text(
    candidates: list[RetrievedItemCandidate],
    evidence: RetrievalEvidence,
) -> int | None:
    if not evidence.relevant_contents:
        return None

    normalized_references = [
        _normalize_text(reference) for reference in evidence.relevant_contents
    ]
    normalized_references = [
        reference for reference in normalized_references if reference
    ]
    if not normalized_references:
        return None

    for candidate in candidates:
        if not candidate.normalized_content:
            continue
        for reference in normalized_references:
            if (
                candidate.normalized_content == reference
                or candidate.normalized_content in reference
                or reference in candidate.normalized_content
            ):
                return candidate.rank
    return None


def _match_by_claim_support(
    candidates: list[RetrievedItemCandidate],
    evidence: RetrievalEvidence,
) -> int | None:
    if not evidence.relevant_contents:
        return None

    for candidate in candidates:
        for reference in evidence.relevant_contents:
            if _content_supports_claim(candidate.content, reference):
                return candidate.rank
    return None


def _match_by_content_similarity(
    candidates: list[RetrievedItemCandidate],
    evidence: RetrievalEvidence,
) -> int | None:
    if not evidence.relevant_contents:
        return None

    relevant_contents = [
        str(value) for value in evidence.relevant_contents if str(value)
    ]
    if not relevant_contents:
        return None

    for candidate in candidates:
        if _content_matches_relevant(candidate.content, relevant_contents):
            return candidate.rank
    return None


def _match_by_metadata_value(
    candidates: list[RetrievedItemCandidate],
    evidence: RetrievalEvidence,
) -> int | None:
    if not evidence.match_key or not evidence.relevant_values:
        return None

    relevant_values = {str(value) for value in evidence.relevant_values}
    for candidate in candidates:
        metadata = candidate.metadata
        if not isinstance(metadata, dict):
            continue
        raw_candidate = metadata.get(evidence.match_key)
        if isinstance(raw_candidate, (list, tuple, set)):
            candidate_values = {str(value) for value in raw_candidate}
        elif raw_candidate is None:
            candidate_values = set()
        else:
            candidate_values = {str(raw_candidate)}
        if candidate_values & relevant_values:
            return candidate.rank
    return None


def _match_by_llm_judge(
    candidates: list[RetrievedItemCandidate],
    evidence: RetrievalEvidence,
    *,
    eval_cfg: Any | None,
) -> int | None:
    if _get_retrieval_match_mode(eval_cfg) != "llm_judge":
        return None
    if not evidence.relevant_contents:
        return None

    judge = _get_retrieval_judge(_get_retrieval_judge_model_id(eval_cfg))
    reference_evidence = "\n".join(f"- {item}" for item in evidence.relevant_contents)
    for candidate in candidates:
        candidate_text = _candidate_text_for_judge(candidate)
        if not candidate_text:
            continue
        prompt = _RETRIEVAL_JUDGE_PROMPT.format(
            reference_evidence=reference_evidence,
            candidate_text=candidate_text,
        )
        # LlmJudge.run() calls any_llm synchronously; without a persistent
        # loop on this thread it churns a fresh event loop per call. See
        # `heams.utils.event_loop.ensure_thread_event_loop`. This call site runs
        # from the orchestrator directly, outside `execute_scenario`'s own
        # guard, so it needs its own.
        ensure_thread_event_loop()
        verdict = judge.run(context=candidate_text, question=prompt)
        if bool(getattr(verdict, "passed", False)):
            return candidate.rank
    return None


def _match_strategies_for_evidence(
    evidence: RetrievalEvidence,
    *,
    eval_cfg: Any | None,
):
    strategies = []
    if evidence.relevant_contents:
        if _get_retrieval_match_mode(eval_cfg) == "llm_judge":
            strategies.append(
                lambda candidates, current_evidence: _match_by_llm_judge(
                    candidates,
                    current_evidence,
                    eval_cfg=eval_cfg,
                )
            )
        strategies.extend(
            [
                _match_by_normalized_text,
                _match_by_claim_support,
                _match_by_content_similarity,
            ]
        )
    if evidence.match_key and evidence.relevant_values:
        strategies.append(_match_by_metadata_value)
    return strategies


def _match_relevant_rank(
    returned_items: list[Any],
    evidence: RetrievalEvidence,
    *,
    eval_cfg: Any | None = None,
) -> int | None:
    candidates = _build_candidate_items(returned_items)
    if not candidates:
        return None

    for strategy in _match_strategies_for_evidence(evidence, eval_cfg=eval_cfg):
        matched_rank = strategy(candidates, evidence)
        if matched_rank is not None:
            return matched_rank

    return None


def _get_returned_items(event: MemoryOperationEvent) -> list[Any] | None:
    result = event.result if isinstance(event.result, dict) else {}
    returned_items = result.get("returned_items")
    if isinstance(returned_items, list):
        return returned_items
    return None


def compute_component_performance_metrics(
    component_metrics: list[ComponentMetricsSummary],
    memory_ledger: list[MemoryOperationEvent] | None,
    scenario: Scenario,
    eval_cfg: Any | None = None,
) -> list[ComponentPerformanceMetricsSummary]:
    """Compute component-level recall quality metrics from raw ledger events."""

    if not component_metrics:
        return []

    evidence_by_stimulus = _build_retrieval_evidence_by_stimulus(scenario)

    if not memory_ledger:
        return [
            ComponentPerformanceMetricsSummary(
                component=summary.component,
                component_type=summary.component_type,
                recall_eval_count=0,
            )
            for summary in component_metrics
        ]

    events_by_component: dict[str, list[MemoryOperationEvent]] = defaultdict(list)
    for event in memory_ledger:
        component = event.component or "unknown"
        events_by_component[component].append(event)

    performance_metrics: list[ComponentPerformanceMetricsSummary] = []
    for summary in component_metrics:
        recall_eval_count = 0
        recall_hits = 0
        reciprocal_rank_sum = 0.0
        recall_missing_stimulus_index_count = 0
        recall_missing_reference_evidence_count = 0
        recall_missing_result_evidence_count = 0

        for event in events_by_component.get(summary.component, []):
            if event.operation_type != MemoryOperationType.RECALL:
                continue

            stimulus_index = _get_event_stimulus_index(event)
            if stimulus_index is None:
                recall_missing_stimulus_index_count += 1
                continue

            evidence = evidence_by_stimulus.get(stimulus_index)
            if evidence is None:
                recall_missing_reference_evidence_count += 1
                continue

            returned_items = _get_returned_items(event)
            if returned_items is None:
                recall_missing_result_evidence_count += 1
                continue

            recall_eval_count += 1
            matched_rank = _match_relevant_rank(
                returned_items,
                evidence,
                eval_cfg=eval_cfg,
            )
            if matched_rank is not None:
                recall_hits += 1
                reciprocal_rank_sum += 1.0 / matched_rank

        performance_metrics.append(
            ComponentPerformanceMetricsSummary(
                component=summary.component,
                component_type=summary.component_type,
                recall_eval_count=recall_eval_count,
                recall_missing_stimulus_index_count=(
                    recall_missing_stimulus_index_count
                ),
                recall_missing_reference_evidence_count=(
                    recall_missing_reference_evidence_count
                ),
                recall_missing_result_evidence_count=(
                    recall_missing_result_evidence_count
                ),
                recall_hit_at_k=(recall_hits / recall_eval_count)
                if recall_eval_count
                else None,
                recall_mrr=(reciprocal_rank_sum / recall_eval_count)
                if recall_eval_count
                else None,
            )
        )

    return performance_metrics
