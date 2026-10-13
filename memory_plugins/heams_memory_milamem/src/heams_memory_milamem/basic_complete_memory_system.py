import json
import os
import threading
import time
import uuid
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
from pydantic import Field

from .dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
    build_dedup_strategy,
)
from .design.storage_type import RawMemoryRecord
from .recall_strategies import (
    EmbeddingCache,
    RecallStrategyConfig,
    RecallStrategyKind,
    build_recall_strategy,
)
from .retrieval import bm25_top_k, tokenize_text


class BasicCompleteMemorySystemConfig(BaseMemoryConfig):
    type: Literal["basic_complete_memory_system"] = "basic_complete_memory_system"
    file_path: str = Field(
        default="memory.json",
        description="Path to save the JSON log.",
    )
    recall_strategy: RecallStrategyConfig = Field(default_factory=RecallStrategyConfig)
    dedup_strategy: DedupStrategyConfig = Field(default_factory=DedupStrategyConfig)
    embedding_cache_path: str | None = Field(
        default=None,
        description=(
            "Path to the persisted entry-id -> embedding-vector cache, used "
            "when `recall_strategy.kind` or `dedup_strategy.kind` is not "
            "`bm25`. Defaults to a sibling of `file_path` so parallel "
            "scenario runs (each with their own unique `file_path`) don't "
            "collide."
        ),
    )

    def with_runtime_storage(self, storage_dir: Path) -> Self:
        needs_embeddings = self.recall_strategy.kind != RecallStrategyKind.BM25 or (
            self.dedup_strategy.enabled
            and self.dedup_strategy.kind != DedupStrategyKind.BM25
        )
        return self.model_copy(
            update={
                "file_path": str(storage_dir / "memory.json"),
                "embedding_cache_path": (
                    str(storage_dir / "memory.embeddings.json")
                    if needs_embeddings
                    else None
                ),
            }
        )


class BasicCompleteMemorySystem(BaseMemorySystem):
    """Append-only raw-record backend with a pluggable recall strategy
    (BM25 / semantic / hybrid / rerank) and retain-time deduplication and
    conflict resolution."""

    config: BasicCompleteMemorySystemConfig

    def __init__(
        self,
        config: BasicCompleteMemorySystemConfig,
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
        needs_embeddings = (
            self.config.recall_strategy.kind != RecallStrategyKind.BM25
            or (
                self.config.dedup_strategy.enabled
                and self.config.dedup_strategy.kind != DedupStrategyKind.BM25
            )
        )
        # bm25-only (recall and dedup) never touches the embedding cache, so
        # the default path never loads/creates an `*.embeddings.json`
        # sibling file. Recall and dedup share one cache — content embedded
        # for one is reused by the other when `embed_model` matches.
        self._embedding_cache = EmbeddingCache(
            self._load_embedding_cache() if needs_embeddings else {}
        )
        # Concurrent stimuli (e.g. locomo's default_concurrency=4) can call
        # `recall`/`retain` on this same instance in parallel; without this
        # lock, overlapping full-file JSON writes to `embedding_cache_path`
        # can interleave and corrupt the cache or silently drop embeddings.
        self._embedding_cache_lock = threading.Lock()
        self._recall_strategy = build_recall_strategy(
            self.config.recall_strategy,
            embedding_cache=self._embedding_cache,
            entry_key_fn=self._entry_cache_key_for(
                self.config.recall_strategy.embed_model
            ),
        )
        self._dedup_strategy = build_dedup_strategy(
            self.config.dedup_strategy,
            embedding_cache=self._embedding_cache,
            entry_key_fn=self._entry_cache_key_for(
                self.config.dedup_strategy.embed_model
            ),
        )

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
        # Content-addressed *and* model-addressed: identical content reuses
        # the same embedding even if retained under a different record id,
        # but a cache entry from a different embed_model never matches, so
        # switching --memory_embed_model/provider over an existing workspace
        # can't silently reload vectors from the wrong model (wrong
        # dimensionality crashes or wrong-but-plausible-looking rankings).
        # Parameterized (rather than reading `self.config.recall_strategy`
        # directly) so recall and dedup — which can each set a different
        # `embed_model` — get correctly distinct keys; they only end up
        # sharing cache entries when both happen to use the same model.
        def _key(entry: MemoryEntry) -> str:
            return json.dumps(
                {"embed_model": embed_model, "content": entry.content},
                sort_keys=True,
            )

        return _key

    def _load_data(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.config.file_path):
            return []

        try:
            with open(self.config.file_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _save_data(self, data: list[dict[str, Any]]) -> None:
        os.makedirs(
            os.path.dirname(os.path.abspath(self.config.file_path)),
            exist_ok=True,
        )
        with open(self.config.file_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _load_records(self) -> list[RawMemoryRecord]:
        records: list[RawMemoryRecord] = []

        for item in self._load_data():
            try:
                records.append(RawMemoryRecord.model_validate(item))
            except Exception:
                continue

        return records

    @staticmethod
    def _active_records(records: list[RawMemoryRecord]) -> list[RawMemoryRecord]:
        return [record for record in records if record.status == "active"]

    @staticmethod
    def _active_entries(records: list[RawMemoryRecord]) -> list[MemoryEntry]:
        return [
            MemoryEntry(content=record.content, metadata=record.metadata)
            for record in records
            if record.status == "active"
        ]

    @classmethod
    def _token_set(cls, value: str) -> set[str]:
        return set(tokenize_text(value))

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(value.casefold().split())

    @classmethod
    def _duplicate_key(cls, record: RawMemoryRecord) -> str:
        return cls._normalized_text(record.content)

    @classmethod
    def _fact_key_value(
        cls, content_or_record: "str | RawMemoryRecord"
    ) -> tuple[str, str] | None:
        content = (
            content_or_record
            if isinstance(content_or_record, str)
            else content_or_record.content
        )
        marker = " is "
        lowered = content.casefold()
        if marker not in lowered:
            return None
        left, right = lowered.split(marker, 1)
        subject = " ".join(left.split())
        value = " ".join(right.split())
        if not subject or not value:
            return None
        return subject, value

    @staticmethod
    def _mark_superseded(
        record: RawMemoryRecord,
        *,
        canonical_id: str,
        reason: str,
    ) -> RawMemoryRecord:
        return record.model_copy(
            update={
                "status": "superseded",
                "superseded_by": canonical_id,
                "consolidation_reason": reason,
            }
        )

    @staticmethod
    def _matched_record_indices(
        records: list[RawMemoryRecord],
        entry: MemoryEntry,
    ) -> list[int]:
        return [
            index
            for index, record in enumerate(records)
            if record.content == entry.content and record.metadata == entry.metadata
        ]

    @staticmethod
    def _next_sequence(records: list[RawMemoryRecord]) -> int:
        if not records:
            return 1
        return max(record.sequence for record in records) + 1

    def _candidate_records(
        self,
        content: str,
        records: list[RawMemoryRecord],
    ) -> list[RawMemoryRecord]:
        """Return up to `dedup_strategy.candidate_k` active records most
        likely to duplicate/conflict with `content`, via `dedup_strategy`
        (independent of `recall_strategy` — this is an internal near-
        duplicate/conflict lookup, not the user-facing recall ranking)."""
        active = self._active_records(records)
        if not active:
            return []
        entries = [MemoryEntry(content=r.content, metadata=r.metadata) for r in active]
        top_k = min(self.config.dedup_strategy.candidate_k, len(entries))
        ranked = self._dedup_strategy.candidates(entries, content, top_k)
        ranked_contents = {e.content for e in ranked}
        return [r for r in active if r.content in ranked_contents]

    def _is_duplicate(
        self,
        content: str,
        active_records: list[RawMemoryRecord],
    ) -> bool:
        """Exact-text duplicate check against every active record, not just
        `dedup_strategy`'s bounded candidates — a normalized-text comparison
        is cheap regardless of store size, so there's no reason to let a
        true duplicate slip through just because it ranked outside the
        candidate_k window."""

        incoming_key = self._normalized_text(content)
        return any(self._duplicate_key(r) == incoming_key for r in active_records)

    def _find_conflict(
        self,
        content: str,
        candidates: list[RawMemoryRecord],
    ) -> RawMemoryRecord | None:

        incoming = self._fact_key_value(content)
        if incoming is None:
            return None
        incoming_subject, incoming_value = incoming
        for record in candidates:
            fact = self._fact_key_value(record)
            if fact is None:
                continue
            subject, value = fact
            if subject == incoming_subject and self._normalized_text(
                value
            ) != self._normalized_text(incoming_value):
                return record
        return None

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        records = self._load_records()

        dedup_enabled = self.config.dedup_strategy.enabled
        candidates = self._candidate_records(content, records) if dedup_enabled else []
        if dedup_enabled and self.config.dedup_strategy.kind != DedupStrategyKind.BM25:
            self._save_embedding_cache()

        if dedup_enabled and self._is_duplicate(content, self._active_records(records)):
            message = "Memory already exists; skipping duplicate."
            return MemoryOperationResult(
                value=message,
                status="skipped",
                result={
                    "message": message,
                    "write_count": 0,
                    "stored_count": len(self._active_records(records)),
                },
            )

        conflict = self._find_conflict(content, candidates) if dedup_enabled else None
        if conflict is not None:
            record_index_by_id = {r.id: i for i, r in enumerate(records)}
            idx = record_index_by_id[conflict.id]
            new_id = uuid.uuid4().hex
            records[idx] = self._mark_superseded(
                records[idx],
                canonical_id=new_id,
                reason="conflict_resolved",
            )
            records.append(
                RawMemoryRecord(
                    id=new_id,
                    content=content,
                    metadata=metadata,
                    sequence=self._next_sequence(records),
                    status="active",
                )
            )
            self._save_data([r.model_dump() for r in records])
            message = "Successfully saved to memory."
            return MemoryOperationResult(
                value=message,
                result={
                    "message": message,
                    "write_count": 1,
                    "stored_count": len(self._active_records(records)),
                },
            )

        before_count = len(records)
        records.append(
            RawMemoryRecord(
                id=uuid.uuid4().hex,
                content=content,
                metadata=metadata,
                sequence=self._next_sequence(records),
                status="active",
            )
        )
        self._save_data([record.model_dump() for record in records])

        message = "Successfully saved to memory."
        return MemoryOperationResult(
            value=message,
            result={
                "message": message,
                "write_count": len(records) - before_count,
                "stored_count": len(self._active_records(records)),
            },
        )

    def _recall_impl(
        self,
        query: str = "",
        top_k: int = 0,
    ) -> MemoryOperationResult:
        entries = self._active_entries(self._load_records())
        if not entries:
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

        scores: list[float | None] = []
        effective_top_k = top_k if top_k else len(entries)

        if not query.strip():
            selected = entries[-top_k:] if top_k else entries
            results = [render_tool_memory_entry(entry) for entry in selected]
            scores = [entry.score for entry in selected]
        else:
            ranked = self._recall_strategy.rank(entries, query, effective_top_k)
            if self.config.recall_strategy.kind != RecallStrategyKind.BM25:
                self._save_embedding_cache()
            results = [render_tool_memory_entry(entry) for entry in ranked]
            scores = [entry.score for entry in ranked]

        output_bytes = self._memory_size_proxy(results)
        event_result = {
            "returned_items": results,
            "result_count": len(results),
            "returned_count": len(results),
            "output_bytes": output_bytes,
            "output_size_chars": output_bytes,
        }
        if any(score is not None for score in scores):
            event_result["scores"] = scores

        return MemoryOperationResult(
            value=results,
            payload={"effective_top_k": effective_top_k, "history_size": len(entries)},
            result=event_result,
        )

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        stripped_query = query.strip()
        if not stripped_query:
            summary = "Delete request was ambiguous: provide a specific query."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={
                    "summary": summary,
                    "message": summary,
                    "supported": True,
                    "deleted_count": 0,
                },
            )

        records = self._load_records()
        active_records = self._active_records(records)
        if not active_records:
            summary = "Delete request did not match any active memory."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={
                    "summary": summary,
                    "message": summary,
                    "supported": True,
                    "deleted_count": 0,
                },
            )

        entries = [
            MemoryEntry(content=record.content, metadata=record.metadata)
            for record in active_records
        ]

        ranked = bm25_top_k(entries, stripped_query, min(2, len(entries)))
        if not ranked:
            summary = "Delete request did not match any active memory."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={
                    "summary": summary,
                    "message": summary,
                    "supported": True,
                    "deleted_count": 0,
                },
            )

        top = ranked[0]
        matched_indices = self._matched_record_indices(active_records, top)
        query_tokens = self._token_set(stripped_query)
        content_tokens = self._token_set(top.content)

        if not query_tokens or not (query_tokens & content_tokens):
            summary = "Delete request did not match any active memory."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={
                    "summary": summary,
                    "message": summary,
                    "supported": True,
                    "deleted_count": 0,
                },
            )

        if len(matched_indices) != 1:
            summary = "Delete request was ambiguous; no memory was deleted."
            return MemoryOperationResult(
                value=summary,
                status="skipped",
                result={
                    "summary": summary,
                    "message": summary,
                    "supported": True,
                    "deleted_count": 0,
                    "matched_content_preview": top.content[:120],
                },
            )

        record = active_records[matched_indices[0]].model_copy(
            update={"status": "deleted", "deleted_at": time.time()}
        )
        active_record_id = active_records[matched_indices[0]].id
        for index, existing in enumerate(records):
            if existing.id == active_record_id:
                records[index] = record
                break
        self._save_data([stored.model_dump() for stored in records])

        summary = "Deleted 1 memory record."
        return MemoryOperationResult(
            value=summary,
            result={
                "summary": summary,
                "message": summary,
                "supported": True,
                "deleted_count": 1,
                "matched_content_preview": top.content[:120],
            },
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        del context, metadata

        records = self._load_records()
        if not records:
            summary = "No active memories required consolidation."
            return MemoryOperationResult(
                value=summary,
                result={
                    "summary": summary,
                    "supported": True,
                    "changed_count": 0,
                    "deduplicated_count": 0,
                    "conflict_resolved_count": 0,
                },
            )

        changed_count = 0
        deduplicated_count = 0
        conflict_resolved_count = 0
        record_index_by_id = {record.id: index for index, record in enumerate(records)}

        active_records = self._active_records(records)
        if not active_records:
            summary = "No active memories required consolidation."
            return MemoryOperationResult(
                value=summary,
                result={
                    "summary": summary,
                    "supported": True,
                    "changed_count": 0,
                    "deduplicated_count": 0,
                    "conflict_resolved_count": 0,
                },
            )

        duplicate_groups: dict[str, list[RawMemoryRecord]] = {}
        for record in active_records:
            duplicate_groups.setdefault(self._duplicate_key(record), []).append(record)

        for group in duplicate_groups.values():
            if len(group) <= 1:
                continue
            canonical = max(group, key=lambda record: record.sequence)
            canonical_id = canonical.id
            for record in group:
                if record.id == canonical_id:
                    continue
                index = record_index_by_id[record.id]
                if records[index].status != "active":
                    continue
                records[index] = self._mark_superseded(
                    records[index],
                    canonical_id=canonical_id,
                    reason="deduplicated",
                )
                changed_count += 1
                deduplicated_count += 1

        active_records = self._active_records(records)
        conflict_groups: dict[str, list[tuple[RawMemoryRecord, str]]] = {}
        for record in active_records:
            fact = self._fact_key_value(record)
            if fact is None:
                continue
            subject, value = fact
            conflict_groups.setdefault(subject, []).append((record, value))

        for group in conflict_groups.values():
            if len(group) <= 1:
                continue

            value_groups: dict[str, list[RawMemoryRecord]] = {}
            for record, value in group:
                value_groups.setdefault(self._normalized_text(value), []).append(record)
            if len(value_groups) <= 1:
                continue

            canonical = max(
                (record for record, _value in group), key=lambda record: record.sequence
            )
            canonical_id = canonical.id
            for record, _value in group:
                if record.id == canonical_id:
                    continue
                index = record_index_by_id[record.id]
                if records[index].status != "active":
                    continue
                records[index] = self._mark_superseded(
                    records[index],
                    canonical_id=canonical_id,
                    reason="conflict_resolved",
                )
                changed_count += 1
                conflict_resolved_count += 1

        if changed_count:
            self._save_data([record.model_dump() for record in records])
            summary = (
                f"Consolidated memories: {deduplicated_count} deduplicated, "
                f"{conflict_resolved_count} conflicts resolved."
            )
        else:
            summary = "No active memories required consolidation."

        return MemoryOperationResult(
            value=summary,
            result={
                "summary": summary,
                "supported": True,
                "changed_count": changed_count,
                "deduplicated_count": deduplicated_count,
                "conflict_resolved_count": conflict_resolved_count,
            },
        )
