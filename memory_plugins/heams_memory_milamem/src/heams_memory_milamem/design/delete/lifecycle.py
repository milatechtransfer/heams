"""Pure planning helpers for the logical-delete lifecycle."""

from collections.abc import Callable
from typing import Any, cast

from heams.memory_systems.api import MemoryEntry

from ..storage_type import CanonicalMemoryRecord
from .config import DeleteStrategyConfig, PlanningMode, SelectionMode
from .models import (
    DeleteOutcome,
    DeletePlan,
    DeleteQueryPlan,
    DeleteReceipt,
    DeleteVerification,
)
from .planners import BaseDeleteQueryPlanner, build_delete_query_planner
from .strategies import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
)

_CANONICAL_MEMORY_ID_METADATA_KEY = "_canonical_memory_id"

DeleteCandidateRanker = Callable[
    [list[MemoryEntry], str, int],
    list[MemoryEntry],
]


def build_active_delete_candidates(
    records: list[CanonicalMemoryRecord],
) -> list[DeleteCandidate]:
    return [
        DeleteCandidate(memory_id=record.id, content=record.content)
        for record in records
        if record.status == "active"
    ]


def build_delete_candidates(
    query: str,
    candidates: list[DeleteCandidate],
    *,
    top_k: int,
    rank: DeleteCandidateRanker,
) -> list[DeleteCandidate]:
    effective_top_k = min(max(top_k, 0), len(candidates))
    if effective_top_k == 0:
        return []

    entries = [
        MemoryEntry(
            content=candidate.content,
            metadata={_CANONICAL_MEMORY_ID_METADATA_KEY: candidate.memory_id},
        )
        for candidate in candidates
    ]
    ranked = rank(entries, query, effective_top_k)[:effective_top_k]

    candidates_by_id = {candidate.memory_id: candidate for candidate in candidates}
    ranked_candidates: list[DeleteCandidate] = []
    for entry in ranked:
        memory_id = entry.metadata.get(_CANONICAL_MEMORY_ID_METADATA_KEY)
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("Ranked delete candidate is missing a canonical memory ID")
        candidate = candidates_by_id.get(memory_id)
        if candidate is None:
            raise ValueError(
                "Ranked delete candidate has an unknown canonical memory ID"
            )
        if entry.score is None:
            raise ValueError("Ranked delete candidate is missing a score")
        ranked_candidates.append(candidate.model_copy(update={"score": entry.score}))
    return ranked_candidates


def build_delete_plan(
    query: str,
    candidates: list[DeleteCandidate],
    decision: DeleteDecision,
    *,
    strategy: str,
    query_plan: DeleteQueryPlan | None = None,
    planning_mode: PlanningMode = PlanningMode.NONE,
) -> DeletePlan:
    normalized_query = " ".join(query.split())
    candidate_ids = [candidate.memory_id for candidate in candidates]
    selected_ids = list(dict.fromkeys(decision.selected_ids))
    candidate_id_set = set(candidate_ids)

    executable = bool(normalized_query and selected_ids)
    if decision.ambiguous or any(
        selected_id not in candidate_id_set for selected_id in selected_ids
    ):
        executable = False
        selected_ids = []
    elif not normalized_query:
        selected_ids = []

    return DeletePlan(
        query=normalized_query,
        strategy=strategy,
        candidate_ids=candidate_ids,
        selected_ids=selected_ids,
        reasoning=decision.reasoning,
        ambiguous=decision.ambiguous,
        executable=executable,
        query_plan=query_plan,
        planning_mode=planning_mode.value,
    )


def execute_delete_plan(
    plan: DeletePlan,
    records: list[CanonicalMemoryRecord],
    *,
    deleted_at: float,
) -> DeleteReceipt:
    requested_ids = list(plan.selected_ids)
    if not plan.executable:
        return DeleteReceipt(
            requested_ids=requested_ids,
            deleted_ids=[],
            skipped_ids=requested_ids,
        )

    record_indices = {record.id: index for index, record in enumerate(records)}
    deleted_ids: list[str] = []
    skipped_ids: list[str] = []
    for memory_id in requested_ids:
        index = record_indices.get(memory_id)
        if index is None or records[index].status != "active":
            skipped_ids.append(memory_id)
            continue
        records[index] = records[index].model_copy(
            update={"status": "deleted", "deleted_at": deleted_at}
        )
        deleted_ids.append(memory_id)

    return DeleteReceipt(
        requested_ids=requested_ids,
        deleted_ids=deleted_ids,
        skipped_ids=skipped_ids,
    )


def verify_logical_delete(
    plan: DeletePlan,
    receipt: DeleteReceipt,
    records: list[CanonicalMemoryRecord],
) -> DeleteVerification:
    if not plan.executable or not receipt.deleted_ids:
        return DeleteVerification(status="not_run")

    active_ids = {record.id for record in records if record.status == "active"}
    remaining_active_ids = [
        memory_id for memory_id in receipt.deleted_ids if memory_id in active_ids
    ]
    return DeleteVerification(
        status="failed" if remaining_active_ids else "verified",
        remaining_active_ids=remaining_active_ids,
    )


class DeleteLifecycle:
    def __init__(
        self,
        config: DeleteStrategyConfig,
        strategy: BaseDeleteStrategy,
        query_planner: BaseDeleteQueryPlanner | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.query_planner = query_planner or build_delete_query_planner(
            config.planning
        )

    def run(
        self,
        query: str,
        records: list[CanonicalMemoryRecord],
        *,
        rank: DeleteCandidateRanker,
        deleted_at: float,
    ) -> DeleteOutcome:
        normalized_query = " ".join(query.split())
        query_plan = self.query_planner.plan(normalized_query)
        all_candidates = build_active_delete_candidates(records)
        retrieved_candidates: list[DeleteCandidate] = []
        retrieval_performed = False

        def retrieve(search_query: str, top_k: int) -> list[DeleteCandidate]:
            nonlocal retrieval_performed, retrieved_candidates
            retrieval_performed = True
            retrieved_candidates = build_delete_candidates(
                search_query,
                all_candidates,
                top_k=top_k,
                rank=rank,
            )
            return retrieved_candidates

        if query_plan.ambiguous or not query_plan.search_query:
            candidates: list[DeleteCandidate] = []
            decision = DeleteDecision(
                selected_ids=[],
                reasoning=query_plan.reasoning,
                ambiguous=query_plan.ambiguous,
            )
        elif self.config.selection_mode is SelectionMode.EXACT_THEN_TOP_K_LLM:
            decision = cast(Any, self.strategy).select(
                query_plan,
                all_candidates,
                retrieve=retrieve,
            )
            candidates = retrieved_candidates if retrieval_performed else all_candidates
        elif self.config.selection_mode is SelectionMode.EXACT:
            candidates = all_candidates
            decision = self.strategy.select(query_plan, candidates)
        elif self.config.selection_mode is SelectionMode.TOP_K_LLM:
            candidates = retrieve(
                query_plan.search_query,
                self.config.top_k_llm.candidate_k,
            )
            if candidates:
                decision = self.strategy.select(query_plan, candidates)
            else:
                decision = DeleteDecision(
                    selected_ids=[],
                    reasoning="No active delete candidates were found.",
                    ambiguous=False,
                )
        else:
            raise NotImplementedError(
                "Delete selection mode is not integrated yet: "
                f"{self.config.selection_mode.value}"
            )

        plan = build_delete_plan(
            normalized_query,
            candidates,
            decision,
            strategy=self.config.selection_mode.value,
            query_plan=query_plan,
            planning_mode=self.config.planning.mode,
        )
        receipt = execute_delete_plan(plan, records, deleted_at=deleted_at)
        verification = verify_logical_delete(plan, receipt, records)
        return DeleteOutcome(
            plan=plan,
            receipt=receipt,
            verification=verification,
        )
