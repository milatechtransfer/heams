import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Self

from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEntry,
    MemoryEventTarget,
    MemoryOperationResult,
    render_tool_memory_entry,
)
from pydantic import BaseModel, Field, model_validator

from .consolidation import ConsolidationConfig
from .dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
    build_dedup_strategy,
)
from .design.canonical_helpers import (
    _active_canonical_records,
    _is_duplicate,
    _next_sequence,
)
from .design.consolidation.lifecycle import ConsolidationLifecycle
from .design.delete import (
    BaseDeleteQueryPlanner,
    BaseDeleteStrategy,
    DeleteLifecycle,
    DeleteStrategyConfig,
    build_delete_query_planner,
    build_delete_strategy,
)
from .design.recall.lifecycle import RecallLifecycle
from .design.retain.lifecycle import (
    CandidateResult,
    RetainLifecycle,
    RetainOutcome,
)
from .design.storage_type import (
    BaseCanonicalProjector,
    CanonicalDesignConfig,
    CanonicalMemoryRecord,
    RawMemoryRecord,
    build_canonical_projector,
)
from .design.storage_type.repository import (
    load_canonical_store,
    load_raw_store,
    save_canonical_store,
    save_raw_store,
)
from .design.storage_type.store_state import (
    CanonicalMemoryStoreState,
    RawMemoryStoreState,
)
from .recall_strategies import (
    EmbeddingCache,
    RecallStrategyConfig,
    RecallStrategyKind,
    build_recall_strategy,
)
from .temporal import parse_temporal_timestamp

STATE_SCHEMA_VERSION = 1

# Gates whether a retain-time top-candidate match is worth an LLM judge
# call (semantic dedup_strategy only). Kept high so retain-time only judges
# near-duplicates: on EE chat turns with text-embedding-3-small, 0.5 sent
# ~78% of writes to the judge (median top-1 cosine 0.69); 0.85 sends ~5%.
RETAIN_JUDGE_SIMILARITY_FLOOR = 0.85


class TemporalMetadataConfig(BaseModel):
    enabled: bool = False


class MilaMemConfig(BaseMemoryConfig):
    type: Literal["milamem"] = "milamem"
    priming_speaker_prefix: bool = Field(
        default=False,
        description="Prefix primed content with the speaker name when available.",
    )
    temporal_metadata: TemporalMetadataConfig = Field(
        default_factory=TemporalMetadataConfig,
        description="Optional temporal metadata enrichment configuration.",
    )
    canonical_projection_config: CanonicalDesignConfig = Field(
        default_factory=CanonicalDesignConfig,
        description="Configuration used to build the canonical projection strategy.",
    )
    raw_store_path: str = Field(
        default="raw_memory.json",
        description="Path to save the raw-memory store state.",
    )
    canonical_store_path: str = Field(
        default="canonical_memory.json",
        description="Path to save the canonical-memory store state.",
    )
    recall_strategy: RecallStrategyConfig = Field(default_factory=RecallStrategyConfig)
    delete_strategy: DeleteStrategyConfig = Field(default_factory=DeleteStrategyConfig)
    dedup_strategy: DedupStrategyConfig = Field(
        default_factory=lambda: DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
        ),
        description=(
            "Controls retain-time duplicate and conflict reconciliation. "
            "enabled=false disables both exact and semantic reconciliation; "
            "enabled=true enables both over the configured window."
        ),
    )

    @model_validator(mode="after")
    def validate_semantic_dedup_strategy(self) -> Self:
        if (
            self.dedup_strategy.enabled
            and self.dedup_strategy.kind is not DedupStrategyKind.SEMANTIC
        ):
            raise ValueError(
                "dedup_strategy.kind must be 'semantic' when "
                "dedup_strategy.enabled is true"
            )
        return self

    embedding_cache_path: str | None = Field(
        default=None,
        description=(
            "Path to the persisted entry-id -> embedding-vector cache. "
            "Always used, since consolidate()'s clustering embeds "
            "unconditionally. Defaults to a sibling of canonical_store_path."
        ),
    )
    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig,
        description=(
            "Config for consolidate()'s clustering + LLM-judged backstop; "
            "embeds via dedup_strategy.embed_model."
        ),
    )
    recall_redirect_superseded: bool = Field(
        default=False,
        description=(
            "Rank superseded records too, and return each superseded hit's "
            "active successor (via superseded_by) in its place."
        ),
    )

    def with_runtime_storage(self, storage_dir: Path) -> Self:
        return self.model_copy(
            update={
                "raw_store_path": str(storage_dir / "raw.json"),
                "canonical_store_path": str(storage_dir / "canonical.json"),
                "embedding_cache_path": str(storage_dir / "canonical.embeddings.json"),
            }
        )


class MilaMemMemorySystem(BaseMemorySystem):
    """MilaMem: experimental raw+canonical backend. Retain writes both stores; recall
    ranks the canonical store via a pluggable recall strategy (BM25 /
    semantic / hybrid / rerank), same as `BasicCompleteMemorySystem`.
    Retain-time conflict search is windowed + top-k bounded (see
    `dedup_strategy.window`); `consolidate()` is the
    unwindowed, exhaustive backstop for whatever the window misses.
    Delete is a logical (local) delete via a pluggable `delete_strategy`
    (default: LLM-judged selection over recall-ranked candidates)."""

    config: MilaMemConfig
    _canonical_projector: BaseCanonicalProjector
    _consolidation_lifecycle: ConsolidationLifecycle
    _recall_lifecycle: RecallLifecycle
    _retain_lifecycle: RetainLifecycle
    _delete_lifecycle: DeleteLifecycle
    _delete_query_planner: BaseDeleteQueryPlanner
    _delete_strategy: BaseDeleteStrategy

    def __init__(
        self,
        config: MilaMemConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ):
        super().__init__(
            config,
            storage_dir=storage_dir,
            event_sink=event_sink,
            run_context=run_context,
        )
        delete_retrieval_config = (
            self.config.delete_strategy.top_k_llm.retrieval_strategy
        )
        self._canonical_projector = build_canonical_projector(
            self.config.canonical_projection_config
        )
        self._consolidation_lifecycle = ConsolidationLifecycle()
        self._retain_lifecycle = RetainLifecycle(
            similarity_floor=RETAIN_JUDGE_SIMILARITY_FLOOR,
        )
        # Recall, dedup, and consolidate share one cache — content embedded
        # for one is reused by the others when `embed_model` matches.
        self._embedding_cache = EmbeddingCache(self._load_embedding_cache())
        self._embedding_cache_lock = threading.Lock()
        self._recall_strategy = build_recall_strategy(
            self.config.recall_strategy,
            embedding_cache=self._embedding_cache,
            entry_key_fn=self._entry_cache_key_for(
                self.config.recall_strategy.embed_model
            ),
        )
        self._recall_lifecycle = RecallLifecycle()
        self._delete_retrieval_strategy = (
            self._recall_strategy
            if delete_retrieval_config is None
            else build_recall_strategy(
                delete_retrieval_config,
                embedding_cache=self._embedding_cache,
                entry_key_fn=self._entry_cache_key_for(
                    delete_retrieval_config.embed_model
                ),
            )
        )
        self._dedup_strategy = build_dedup_strategy(
            self.config.dedup_strategy,
            embedding_cache=self._embedding_cache,
            entry_key_fn=self._entry_cache_key_for(
                self.config.dedup_strategy.embed_model
            ),
        )
        self._delete_query_planner = build_delete_query_planner(
            self.config.delete_strategy.planning
        )
        self._delete_strategy = build_delete_strategy(self.config.delete_strategy)
        self._delete_lifecycle = DeleteLifecycle(
            self.config.delete_strategy,
            self._delete_strategy,
            query_planner=self._delete_query_planner,
        )

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        for entry in entries:
            content = entry.content
            if self.config.priming_speaker_prefix:
                speaker_name = entry.metadata.get(
                    "ingest_speaker_name"
                ) or entry.metadata.get("speaker")
                if (
                    isinstance(speaker_name, str)
                    and speaker_name
                    and not content.startswith(f"{speaker_name}:")
                ):
                    content = f"{speaker_name}: {content}"
            self.retain(content=content, metadata=entry.metadata)

    def _windowed_active_canonical_records(
        self,
        records: list[CanonicalMemoryRecord],
        *,
        now: datetime,
    ) -> list[CanonicalMemoryRecord]:
        """With `temporal_metadata.enabled=True`, bounds to active records
        within `dedup_strategy.window.days` of `now` (missing timestamps excluded);
        otherwise falls back to the `dedup_strategy.window.turns` most recent active
        records. Both unset (default) = unbounded."""
        active = _active_canonical_records(records)
        if self.config.temporal_metadata.enabled:
            if self.config.dedup_strategy.window.days is None:
                return active
            cutoff = timedelta(days=self.config.dedup_strategy.window.days)
            windowed = []
            for record in active:
                parsed = parse_temporal_timestamp(
                    record.metadata.get("temporal_received_at")
                )
                if parsed is not None and (now - parsed) <= cutoff:
                    windowed.append(record)
            return windowed
        if self.config.dedup_strategy.window.turns is None:
            return active
        return sorted(active, key=lambda r: r.sequence)[
            -self.config.dedup_strategy.window.turns :
        ]

    _CANDIDATE_ID_METADATA_KEY = "_canonical_record_id"

    def _candidate_records(
        self,
        content: str,
        records: list[CanonicalMemoryRecord],
        *,
        now: datetime,
    ) -> CandidateResult:
        """Up to `dedup_strategy.candidate_k` windowed active records ranked
        most likely to duplicate/conflict with `content`, plus the
        top-ranked one and its score as `top_record`/`top_score`.
        `.records` stays in chronological (not rank) order."""
        windowed = self._windowed_active_canonical_records(records, now=now)
        if not windowed:
            return CandidateResult(records=[], top_score=None)
        # MemoryEntry has no id field, so smuggle it through metadata to
        # join back correctly even if two records share identical content.
        entries = [
            MemoryEntry(
                content=r.content,
                metadata={**r.metadata, self._CANDIDATE_ID_METADATA_KEY: r.id},
            )
            for r in windowed
        ]
        top_k = min(self.config.dedup_strategy.candidate_k, len(entries))
        ranked = self._dedup_strategy.candidates(entries, content, top_k)
        if not ranked:
            return CandidateResult(records=[], top_score=None)
        ranked_ids = {e.metadata.get(self._CANDIDATE_ID_METADATA_KEY) for e in ranked}
        matched = [r for r in windowed if r.id in ranked_ids]
        matched_by_id = {r.id: r for r in matched}
        # Both dedup strategies sort descending, so ranked[0] is top-scored.
        top_entry = ranked[0]
        top_record_id = top_entry.metadata.get(self._CANDIDATE_ID_METADATA_KEY)
        top_record = (
            matched_by_id.get(top_record_id) if isinstance(top_record_id, str) else None
        )
        return CandidateResult(
            records=matched, top_record=top_record, top_score=top_entry.score
        )

    def _is_duplicate(
        self,
        content: str,
        active_records: list[CanonicalMemoryRecord],
    ) -> bool:
        return _is_duplicate(content, active_records)

    def _load_embedding_cache(self) -> dict[str, list[float]]:
        path = self.config.embedding_cache_path
        if path is None or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_embedding_cache(self) -> None:
        path = self.config.embedding_cache_path
        if path is None:
            return
        with self._embedding_cache_lock:
            snapshot = self._embedding_cache.snapshot_if_dirty()
            if snapshot is None:
                return
            vectors, revision = snapshot
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(vectors, handle)
            self._embedding_cache.mark_persisted(revision)

    @staticmethod
    def _entry_cache_key_for(embed_model: str) -> Any:
        # Keyed by model so recall/dedup only share cache entries when
        # both use the same embed_model.
        def _key(entry: MemoryEntry) -> str:
            return json.dumps(
                {"embed_model": embed_model, "content": entry.content},
                sort_keys=True,
            )

        return _key

    def _load_raw_store(self) -> RawMemoryStoreState:
        return load_raw_store(
            self.config.raw_store_path,
            schema_version=STATE_SCHEMA_VERSION,
        )

    def _load_canonical_store(self) -> CanonicalMemoryStoreState:
        return load_canonical_store(
            self.config.canonical_store_path,
            schema_version=STATE_SCHEMA_VERSION,
        )

    def _save_raw_store(self, raw_store: RawMemoryStoreState) -> None:
        save_raw_store(self.config.raw_store_path, raw_store)

    def _save_canonical_store(self, canonical_store: CanonicalMemoryStoreState) -> None:
        save_canonical_store(self.config.canonical_store_path, canonical_store)

    def maintain(self) -> None:
        config = self.config.consolidation
        if not config.enabled or config.pending_threshold is None:
            return

        pending_count = len(self._load_canonical_store().pending_consolidation_ids)
        if pending_count >= config.pending_threshold:
            self.consolidate()

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        outcome: RetainOutcome = self._retain_lifecycle.run(
            self,
            content,
            metadata,
        )
        self.maintain()
        result = {
            "message": outcome.message,
            "raw_write_count": outcome.raw_write_count,
            "canonical_write_count": outcome.canonical_write_count,
            "stored_raw_count": outcome.stored_raw_count,
            "stored_canonical_count": outcome.stored_canonical_count,
            **outcome.temporal_result,
        }
        if outcome.canonical_write_count:
            result.update(
                {
                    "superseded_ids": outcome.superseded_ids,
                    "consolidation_reason": outcome.consolidation_reason,
                }
            )
        return MemoryOperationResult(
            value=outcome.message,
            status=outcome.status,
            result=result,
        )

    def _recall_impl(
        self,
        query: str = "",
        top_k: int = 0,
    ) -> MemoryOperationResult:
        # Recall reads the canonical (deduplicated) store, not raw; excludes
        # superseded records unless recall_redirect_superseded is set.
        outcome = self._recall_lifecycle.run(
            self,
            query,
            self._load_canonical_store().records,
            top_k=top_k,
        )
        if not outcome.entries:
            return MemoryOperationResult(
                value=[],
                payload={"history_size": 0},
                result={
                    "returned_items": [],
                    "result_count": 0,
                    "returned_count": 0,
                    "output_bytes": 0,
                    "output_size_chars": 0,
                },
            )

        results = [render_tool_memory_entry(entry) for entry in outcome.entries]

        output_bytes = self._memory_size_proxy(results)
        event_result = {
            "returned_items": results,
            "result_count": len(results),
            "returned_count": len(results),
            "output_bytes": output_bytes,
            "output_size_chars": output_bytes,
        }
        if any(score is not None for score in outcome.scores):
            event_result["scores"] = outcome.scores

        return MemoryOperationResult(
            value=results,
            payload={
                "effective_top_k": outcome.effective_top_k,
                "history_size": outcome.history_size,
            },
            result=event_result,
        )

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        stripped_query = query.strip()
        strategy = self.config.delete_strategy.selection_mode.value
        raw_store = self._load_raw_store()
        raw_store.records.append(
            RawMemoryRecord(
                id=uuid.uuid4().hex,
                content=query,
                metadata={"record_type": "delete_request"},
                sequence=_next_sequence(raw_store.records),
            )
        )
        self._save_raw_store(raw_store)
        if not stripped_query:
            message = "Delete request requires a non-empty query."
            return MemoryOperationResult(
                value=message,
                status="skipped",
                result={
                    "summary": message,
                    "message": message,
                    "supported": True,
                    "strategy": strategy,
                    "planning_mode": self.config.delete_strategy.planning.mode.value,
                    "selection_mode": self.config.delete_strategy.selection_mode.value,
                    "target_text": "",
                    "cardinality": "one",
                    "exclusions": [],
                    "candidate_count": 0,
                    "selected_ids": [],
                    "deleted_count": 0,
                    "ambiguous": False,
                    "reasoning": "The deletion query was empty.",
                    "raw_erasure": False,
                    "verification_status": "not_run",
                },
            )

        canonical_store = self._load_canonical_store()

        def rank_delete_candidates(
            entries: list[MemoryEntry],
            ranked_query: str,
            top_k: int,
        ) -> list[MemoryEntry]:
            ranked = self._delete_retrieval_strategy.rank(entries, ranked_query, top_k)
            delete_retrieval_config = (
                self.config.delete_strategy.top_k_llm.retrieval_strategy
                or self.config.recall_strategy
            )
            if delete_retrieval_config.kind != RecallStrategyKind.BM25:
                self._save_embedding_cache()
            return ranked

        outcome = self._delete_lifecycle.run(
            stripped_query,
            canonical_store.records,
            rank=rank_delete_candidates,
            deleted_at=datetime.now(timezone.utc).timestamp(),
        )
        plan = outcome.plan
        receipt = outcome.receipt
        candidate_count = len(plan.candidate_ids)
        selected_ids = plan.selected_ids
        deleted_count = len(receipt.deleted_ids) if receipt is not None else 0
        query_plan = plan.query_plan

        if outcome.changed:
            self._save_canonical_store(canonical_store)
            message = (
                f"Logically deleted {deleted_count} matching canonical memory record(s)."
                if query_plan is not None and query_plan.cardinality == "all"
                else f"Logically deleted {deleted_count} canonical memory record(s)."
            )
            status = "success"
        else:
            message = (
                "Delete request was ambiguous; no canonical memory was deleted."
                if plan.ambiguous
                else "Delete request did not match any canonical memory."
            )
            status = "skipped"

        return MemoryOperationResult(
            value=message,
            status=status,
            result={
                "summary": message,
                "message": message,
                "supported": True,
                "strategy": strategy,
                "planning_mode": plan.planning_mode,
                "selection_mode": self.config.delete_strategy.selection_mode.value,
                "target_text": query_plan.target_text if query_plan is not None else "",
                "cardinality": (
                    query_plan.cardinality if query_plan is not None else "one"
                ),
                "exclusions": query_plan.exclusions if query_plan is not None else [],
                "candidate_count": candidate_count,
                "selected_ids": selected_ids,
                "deleted_count": deleted_count,
                "ambiguous": plan.ambiguous,
                "reasoning": plan.reasoning,
                "raw_erasure": receipt.raw_erasure if receipt is not None else False,
                "verification_status": outcome.verification.status,
            },
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        del context, metadata
        if not self.config.consolidation.enabled:
            summary = "Canonical consolidation is disabled."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={"summary": summary, "supported": True, "changed_count": 0},
            )
        outcome = self._consolidation_lifecycle.run(self)
        if outcome.changed_count:
            summary = (
                f"Consolidated memories: {outcome.deduplicated_count} deduplicated, "
                f"{outcome.conflict_resolved_count} conflicts resolved."
            )
        else:
            summary = "No active memories required consolidation."

        return MemoryOperationResult(
            value=summary,
            result={
                "summary": summary,
                "supported": True,
                "changed_count": outcome.changed_count,
                "deduplicated_count": outcome.deduplicated_count,
                "conflict_resolved_count": outcome.conflict_resolved_count,
                "skipped_cluster_count": outcome.skipped_cluster_count,
                "processed_pending_count": outcome.processed_pending_count,
                "remaining_pending_count": outcome.remaining_pending_count,
            },
        )
