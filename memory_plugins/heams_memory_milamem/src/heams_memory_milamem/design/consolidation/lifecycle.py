"""Orchestration for the canonical consolidation lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from ... import consolidation
from ..canonical_helpers import (
    _active_canonical_records,
    _apply_anchor_verdict,
    _apply_verdict,
)
from ..storage_type.store_state import CanonicalMemoryStoreState


@dataclass(slots=True)
class ConsolidationOutcome:
    changed_count: int
    deduplicated_count: int
    conflict_resolved_count: int
    skipped_cluster_count: int
    processed_pending_count: int
    remaining_pending_count: int


class ConsolidationRuntime(Protocol):
    config: Any
    _embedding_cache: Any

    def _load_canonical_store(self) -> CanonicalMemoryStoreState: ...

    def _entry_cache_key_for(self, embed_model: str) -> Any: ...

    def _save_embedding_cache(self) -> None: ...

    def _save_canonical_store(
        self, canonical_store: CanonicalMemoryStoreState
    ) -> None: ...


def _judge_anchor_or_none(
    config: consolidation.ConsolidationConfig, anchor: Any, candidates: list
) -> Any:
    try:
        return consolidation.judge_anchor(config, anchor, candidates)
    except Exception:
        # Fail open: the anchor stays pending for the next pass.
        return None


class ConsolidationLifecycle:
    def run(self, runtime: ConsolidationRuntime) -> ConsolidationOutcome:
        canonical_store = runtime._load_canonical_store()
        active_records = _active_canonical_records(canonical_store.records)
        active_ids = {record.id for record in active_records}
        pending_order = [
            record_id
            for record_id in canonical_store.pending_consolidation_ids
            if record_id in active_ids
        ]
        had_pending_entries = bool(canonical_store.pending_consolidation_ids)
        selected_pending = pending_order[
            : runtime.config.consolidation.max_pending_per_run
        ]
        deferred_pending = pending_order[
            runtime.config.consolidation.max_pending_per_run :
        ]
        pending_ids = set(selected_pending)
        canonical_store.pending_consolidation_ids = list(deferred_pending)

        if not pending_ids:
            if had_pending_entries:
                runtime._save_canonical_store(canonical_store)
            return ConsolidationOutcome(
                changed_count=0,
                deduplicated_count=0,
                conflict_resolved_count=0,
                skipped_cluster_count=0,
                processed_pending_count=0,
                remaining_pending_count=len(deferred_pending),
            )

        changed_count = 0
        deduplicated_count = 0
        conflict_resolved_count = 0
        skipped_cluster_count = 0
        retry_pending_ids: set[str] = set()
        records = canonical_store.records
        record_index_by_id = {record.id: index for index, record in enumerate(records)}

        embed_model = runtime.config.dedup_strategy.embed_model
        config = runtime.config.consolidation
        clusters: list = []
        anchor_groups: list = []
        if config.anchor_top_k is not None:
            anchor_groups = consolidation.anchor_candidates(
                active_records,
                embed_model=embed_model,
                similarity_threshold=config.similarity_threshold,
                embedding_cache=runtime._embedding_cache,
                entry_key_fn=runtime._entry_cache_key_for(embed_model),
                pending_ids=pending_ids,
                top_k=config.anchor_top_k,
                anchor_roles=config.anchor_roles,
            )
        else:
            clusters = consolidation.cluster_by_similarity(
                active_records,
                embed_model=embed_model,
                similarity_threshold=config.similarity_threshold,
                embedding_cache=runtime._embedding_cache,
                entry_key_fn=runtime._entry_cache_key_for(embed_model),
                pending_ids=pending_ids,
            )
        runtime._save_embedding_cache()

        if anchor_groups:
            with ThreadPoolExecutor(max_workers=config.judge_concurrency) as pool:
                anchor_verdicts = list(
                    pool.map(
                        lambda group: _judge_anchor_or_none(
                            config, group[0], group[1]
                        ),
                        anchor_groups,
                    )
                )
            # Applied in sequence order so older anchors resolve first.
            for (anchor, candidates), verdict in zip(anchor_groups, anchor_verdicts):
                if verdict is None:
                    skipped_cluster_count += 1
                    retry_pending_ids.add(anchor.id)
                    continue
                superseded = _apply_anchor_verdict(
                    anchor,
                    candidates,
                    verdict,
                    store_records=records,
                    record_index_by_id=record_index_by_id,
                )
                if not superseded:
                    skipped_cluster_count += 1
                    continue
                changed_count += len(superseded)
                deduplicated_count += sum(
                    reason == "deduplicated" for reason in superseded.values()
                )
                conflict_resolved_count += sum(
                    reason == "conflict_resolved" for reason in superseded.values()
                )

        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            for chunk in consolidation.chunk_cluster(
                cluster, max_size=runtime.config.consolidation.max_cluster_size
            ):
                if len(chunk) <= 1:
                    retry_pending_ids.update(
                        record.id for record in chunk if record.id in pending_ids
                    )
                    continue

                try:
                    verdict = consolidation.judge_cluster(config, chunk)
                except Exception:
                    skipped_cluster_count += 1
                    retry_pending_ids.update(
                        record.id for record in chunk if record.id in pending_ids
                    )
                    continue

                outcome = _apply_verdict(
                    chunk,
                    verdict,
                    store_records=records,
                    record_index_by_id=record_index_by_id,
                )
                if outcome.reason is None:
                    skipped_cluster_count += 1
                    if verdict.relationship not in ("related", "unrelated"):
                        retry_pending_ids.update(
                            record.id for record in chunk if record.id in pending_ids
                        )
                    continue

                changed_count += len(outcome.superseded_ids)
                if outcome.reason == "deduplicated":
                    deduplicated_count += len(outcome.superseded_ids)
                else:
                    conflict_resolved_count += len(outcome.superseded_ids)

        canonical_store.pending_consolidation_ids.extend(
            record_id
            for record_id in selected_pending
            if record_id in retry_pending_ids
        )
        runtime._save_canonical_store(canonical_store)
        return ConsolidationOutcome(
            changed_count=changed_count,
            deduplicated_count=deduplicated_count,
            conflict_resolved_count=conflict_resolved_count,
            skipped_cluster_count=skipped_cluster_count,
            processed_pending_count=len(selected_pending) - len(retry_pending_ids),
            remaining_pending_count=len(canonical_store.pending_consolidation_ids),
        )
