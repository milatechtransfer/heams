"""Pluggable candidate-retrieval strategies for retain-time deduplication and
conflict resolution, decoupled from where entries come from.

A dedup strategy answers one question: "given this incoming content, which
already-stored entries are plausible near-duplicates or conflicts?".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from heams.memory_systems.api import MemoryEntry
from pydantic import BaseModel, Field

from .retrieval import bm25_top_k
from . import recall_strategies
from .recall_strategies import EmbeddingCache


class DedupStrategyKind(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"


class DedupWindowConfig(BaseModel):
    days: float | None = Field(
        default=None,
        gt=0,
        description=(
            "When temporal metadata is enabled, reconcile only records received "
            "within the last N days. None means unbounded."
        ),
    )
    turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When temporal metadata is disabled, reconcile only the N most "
            "recent active records. None means unbounded."
        ),
    )


class DedupStrategyConfig(BaseModel):
    enabled: bool = True
    kind: DedupStrategyKind = DedupStrategyKind.BM25
    candidate_k: int = Field(
        default=5,
        ge=1,
        description="Number of candidate entries to fetch per retain() call.",
    )
    embed_model: str = Field(
        default="openai:text-embedding-3-small",
        description="any_llm model id used for embedding (semantic only).",
    )
    window: DedupWindowConfig = Field(default_factory=DedupWindowConfig)


class BaseDedupStrategy(ABC):
    """Fetches candidate entries a new `content` string might duplicate or
    conflict with. Does not judge duplication/conflict itself."""

    @abstractmethod
    def candidates(
        self,
        entries: list[MemoryEntry],
        content: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        """Return up to `top_k` entries most likely to relate to `content`."""


class BM25DedupStrategy(BaseDedupStrategy):
    """Thin wrapper around the existing BM25 helper; behavior-preserving."""

    def candidates(
        self,
        entries: list[MemoryEntry],
        content: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        if not entries:
            return []
        return bm25_top_k(entries, content, min(top_k, len(entries)))


class SemanticDedupStrategy(BaseDedupStrategy):
    """Ranks stored entries by cosine similarity to the incoming content.

    Shares an `EmbeddingCache` with the memory system's recall strategy (when
    both are non-bm25 and use the same `embed_model`), so content embedded
    once for recall isn't re-embedded for dedup, and vice versa.
    """

    def __init__(
        self,
        *,
        embed_model: str,
        embedding_cache: EmbeddingCache,
        entry_key_fn: Any,
    ):
        self.embed_model = embed_model
        self.embedding_cache = embedding_cache
        self._entry_key_fn = entry_key_fn

    def _resolve_entry_vectors(self, entries: list[MemoryEntry]) -> list[np.ndarray]:
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        keys = [self._entry_key_fn(entry) for entry in entries]

        for key, entry in zip(keys, entries):
            if key not in self.embedding_cache:
                missing_keys.append(key)
                missing_texts.append(entry.content)

        if missing_texts:
            vectors = recall_strategies.embed_texts(self.embed_model, missing_texts)
            for key, vector in zip(missing_keys, vectors):
                self.embedding_cache.set(key, vector)

        return [np.array(self.embedding_cache.get(key)) for key in keys]

    def candidates(
        self,
        entries: list[MemoryEntry],
        content: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        if not entries or top_k <= 0:
            return []

        entry_vectors = self._resolve_entry_vectors(entries)
        content_vector = np.array(
            recall_strategies.embed_texts(self.embed_model, [content])[0]
        )
        scores = [
            recall_strategies.cosine_similarity(content_vector, vector)
            for vector in entry_vectors
        ]

        ranked_indices = sorted(
            range(len(entries)), key=lambda idx: scores[idx], reverse=True
        )[:top_k]

        return [
            entries[idx].model_copy(update={"score": scores[idx]})
            for idx in ranked_indices
        ]


def build_dedup_strategy(
    config: DedupStrategyConfig,
    *,
    embedding_cache: EmbeddingCache,
    entry_key_fn: Any,
) -> BaseDedupStrategy:
    if config.kind is DedupStrategyKind.BM25:
        return BM25DedupStrategy()

    if config.kind is DedupStrategyKind.SEMANTIC:
        return SemanticDedupStrategy(
            embed_model=config.embed_model,
            embedding_cache=embedding_cache,
            entry_key_fn=entry_key_fn,
        )

    raise ValueError(f"Unsupported dedup strategy kind: {config.kind}")
