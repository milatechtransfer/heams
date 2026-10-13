"""Pluggable recall strategies, decoupled from where entries come from.

Each strategy ranks a list of already-fetched `MemoryEntry` objects against a
query. None of them know or care whether those entries came from a raw store
or a canonical store.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

import numpy as np
from heams.memory_systems.api import MemoryEntry
from pydantic import BaseModel, Field, model_validator

from .retrieval import bm25_top_k, tokenize_text
from .temporal import parse_temporal_timestamp


class RecallStrategyKind(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    RERANK = "rerank"


class FusionKind(str, Enum):
    MINMAX = "minmax"
    ABSOLUTE = "absolute"


class RecallStrategyConfig(BaseModel):
    kind: RecallStrategyKind = RecallStrategyKind.BM25
    embed_model: str = Field(
        default="openai:text-embedding-3-small",
        description=(
            "any_llm model id, or `st:<hf id>` for an in-process "
            "sentence-transformers model (semantic/hybrid only)."
        ),
    )
    query_prompt: str | None = Field(
        default=None,
        description=(
            "Prepended to the query (never to stored entries) before embedding, "
            "e.g. Qwen3-Embedding's 'Instruct: ...\\nQuery: '."
        ),
    )
    hybrid_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight given to the semantic score in hybrid mode; "
            "0.0 is pure BM25, 1.0 is pure semantic."
        ),
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        description=(
            "sentence-transformers cross-encoder model id used to rerank the "
            "BM25 + semantic candidate pool (rerank only)."
        ),
    )
    rerank_candidate_k: int = Field(
        default=25,
        ge=1,
        description=(
            "Number of candidates each of BM25 and semantic retrieval "
            "contribute to the pool the cross-encoder reranks (rerank only)."
        ),
    )
    fusion: FusionKind = Field(
        default=FusionKind.MINMAX,
        description=(
            "Hybrid score fusion. `minmax` rescales per query (ranking only); "
            "`absolute` uses upper-bound BM25 + clamped cosine, so scores are "
            "comparable across queries and can be filtered (hybrid only)."
        ),
    )
    filter_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Keep entries scoring >= ratio * top score (absolute hybrid only).",
    )
    filter_min_keep: int = Field(
        default=1,
        ge=1,
        description="Floor on entries kept by the relative cut (absolute hybrid only).",
    )
    filter_min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Hard absolute floor; overrides filter_min_keep, so it can return "
            "nothing (absolute hybrid only)."
        ),
    )

    @model_validator(mode="after")
    def _validate_fusion_and_filter(self) -> Self:
        is_absolute_hybrid = (
            self.kind is RecallStrategyKind.HYBRID
            and self.fusion is FusionKind.ABSOLUTE
        )
        if self.fusion is FusionKind.ABSOLUTE and self.kind is not RecallStrategyKind.HYBRID:
            raise ValueError("fusion 'absolute' requires kind 'hybrid'")
        filter_set = (
            self.filter_ratio != 0.0
            or self.filter_min_keep != 1
            or self.filter_min_score != 0.0
        )
        if filter_set and not is_absolute_hybrid:
            raise ValueError(
                "filter_* fields require kind 'hybrid' with fusion 'absolute'"
            )
        return self


class BaseRecallStrategy(ABC):
    """Ranks pre-fetched entries against a query. Does not fetch entries itself."""

    @abstractmethod
    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        """Return up to `top_k` entries, each with `.score` populated."""


class BM25RecallStrategy(BaseRecallStrategy):
    """Thin wrapper around the existing BM25 helper; behavior-preserving."""

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        return bm25_top_k(entries, query, top_k)


class EmbeddingCache:
    """Id-keyed embedding cache, persisted as a flat JSON-able dict.

    Embeddings are computed once per entry at retain time rather than on
    every recall call; callers are responsible for loading/saving the
    underlying dict (kept here as plain `list[float]` values for JSON
    round-tripping).

    Internally thread-safe: the memory-system classes run recall()/retain()
    via `loop.run_in_executor`, i.e. real OS threads, and one `EmbeddingCache`
    instance is shared between the recall and dedup strategies. Without a
    lock here, `to_dict()`'s `dict(self._vectors)` copy can race with a
    concurrent `set()` on a new key and raise `RuntimeError: dictionary
    changed size during iteration`.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self._vectors: dict[str, list[float]] = dict(vectors or {})
        self._lock = threading.Lock()
        self._revision = 0
        self._persisted_revision = 0

    def to_dict(self) -> dict[str, list[float]]:
        with self._lock:
            return dict(self._vectors)

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            return self._vectors.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        with self._lock:
            if key not in self._vectors:
                self._revision += 1
            self._vectors[key] = vector

    def snapshot_if_dirty(self) -> tuple[dict[str, list[float]], int] | None:
        with self._lock:
            if self._revision == self._persisted_revision:
                return None
            return dict(self._vectors), self._revision

    def mark_persisted(self, revision: int) -> None:
        with self._lock:
            if self._revision == revision:
                self._persisted_revision = revision

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._vectors


def _ensure_thread_event_loop() -> None:
    """Give the current thread a persistent event loop if it has none.

    `retain()`/`recall()` run via `loop.run_in_executor` (real OS threads),
    which have no event loop by default in Python 3.10+. Without one,
    any_llm's sync embedding wrapper (`allow_running_loop=True`) can't find
    a "current" loop either, so it falls back to `asyncio.run(coro)`,
    creating and closing a brand-new event loop on *every single* embedding
    call. That churn orphans the internal httpx `AsyncClient`'s cleanup onto
    an already-closed loop, producing "Event loop is closed" / "no current
    event loop" noise (harmless to results, but noisy) once garbage
    collection catches up. Setting one persistent loop per worker thread
    (reused across every embed_texts call on that thread) avoids the churn.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# Provider per-request caps (OpenAI: 2048 inputs, 300k tokens), with headroom.
_MAX_EMBED_BATCH_INPUTS = 2048
_MAX_EMBED_BATCH_TOKENS = 250_000
# OpenAI enforces the per-request cap on len(text)/4, not real tokens.
_CHARS_PER_ESTIMATED_TOKEN = 4
# text-embedding-3-* per-input cap, counted in its own tokenizer.
_MAX_EMBED_INPUT_TOKENS = 8191
_EMBED_TOKENIZER = "cl100k_base"


def _embed_batches(texts: list[str]) -> list[list[str]]:
    import tiktoken

    encoding = tiktoken.get_encoding(_EMBED_TOKENIZER)
    batches: list[list[str]] = []
    current: list[str] = []
    current_cost = 0
    # encode_ordinary: memories can contain literal "<|endoftext|>".
    for text, tokens in zip(texts, encoding.encode_ordinary_batch(texts)):
        if len(tokens) > _MAX_EMBED_INPUT_TOKENS:
            tokens = tokens[:_MAX_EMBED_INPUT_TOKENS]
            text = encoding.decode(tokens)
        cost = max(len(tokens), math.ceil(len(text) / _CHARS_PER_ESTIMATED_TOKEN))
        if current and (
            len(current) >= _MAX_EMBED_BATCH_INPUTS
            or current_cost + cost > _MAX_EMBED_BATCH_TOKENS
        ):
            batches.append(current)
            current, current_cost = [], 0
        current.append(text)
        current_cost += cost
    if current:
        batches.append(current)
    return batches


_SENTENCE_TRANSFORMERS_PREFIX = "st:"
# Matches text-embedding-3-small's input limit so models truncate alike.
_ST_MAX_SEQ_LENGTH = 8192
_ST_ENCODE_BATCH_SIZE = 8
_st_models: dict[str, Any] = {}
# recall/retain run on executor threads; one lock serializes load + GPU encode.
_st_lock = threading.Lock()


def _embed_with_sentence_transformers(name: str, texts: list[str]) -> list[list[float]]:
    import torch
    from sentence_transformers import SentenceTransformer

    with _st_lock:
        model = _st_models.get(name)
        if model is None:
            model = SentenceTransformer(
                name, model_kwargs={"torch_dtype": torch.bfloat16}
            )
            model.max_seq_length = min(
                model.max_seq_length or _ST_MAX_SEQ_LENGTH, _ST_MAX_SEQ_LENGTH
            )
            _st_models[name] = model
        vectors = model.encode(
            texts,
            batch_size=_ST_ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return vectors.tolist()


def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Embed texts via any_llm (split into provider-sized requests) or, for
    `st:` ids, an in-process sentence-transformers model. Isolated here so
    it's the single mockable seam for tests (no real API calls in the test
    suite)."""

    if not texts:
        return []

    if model.startswith(_SENTENCE_TRANSFORMERS_PREFIX):
        return _embed_with_sentence_transformers(
            model.removeprefix(_SENTENCE_TRANSFORMERS_PREFIX), texts
        )

    import any_llm

    # `recall`/`retain` run synchronously but get invoked as agent tools from
    # inside a running event loop (TinyAgent's async tool-call loop); without
    # `allow_running_loop=True`, any_llm's sync API refuses to run there at
    # all rather than risk a deadlock, and raises instead of embedding.
    _ensure_thread_event_loop()
    vectors: list[list[float]] = []
    for batch in _embed_batches(texts):
        response = any_llm.embedding(model=model, inputs=batch, allow_running_loop=True)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticRecallStrategy(BaseRecallStrategy):
    """Ranks entries by cosine similarity between query and content embeddings.

    Entry embeddings are resolved via `entry_key_fn` against `embedding_cache`
    and computed (then cached by the caller) on a miss, so repeated recalls
    don't re-embed unchanged content.
    """

    def __init__(
        self,
        *,
        embed_model: str,
        embedding_cache: EmbeddingCache,
        entry_key_fn: Any,
        query_prompt: str | None = None,
    ):
        self.embed_model = embed_model
        self.embedding_cache = embedding_cache
        self._entry_key_fn = entry_key_fn
        self._query_prompt = query_prompt

    def _resolve_entry_vectors(self, entries: list[MemoryEntry]) -> list[np.ndarray]:
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        keys = [self._entry_key_fn(entry) for entry in entries]

        for key, entry in zip(keys, entries):
            if key not in self.embedding_cache:
                missing_keys.append(key)
                missing_texts.append(entry.content)

        if missing_texts:
            vectors = embed_texts(self.embed_model, missing_texts)
            for key, vector in zip(missing_keys, vectors):
                self.embedding_cache.set(key, vector)

        return [np.array(self.embedding_cache.get(key)) for key in keys]

    def score(self, entries: list[MemoryEntry], query: str) -> list[float]:
        if not entries:
            return []

        entry_vectors = self._resolve_entry_vectors(entries)
        query_text = f"{self._query_prompt}{query}" if self._query_prompt else query
        query_vector = np.array(embed_texts(self.embed_model, [query_text])[0])

        return [cosine_similarity(query_vector, vector) for vector in entry_vectors]

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        if not query or not entries or top_k <= 0:
            return []

        scores = self.score(entries, query)
        ranked_indices = sorted(
            range(len(entries)), key=lambda idx: scores[idx], reverse=True
        )[:top_k]

        return [
            entries[idx].model_copy(update={"score": scores[idx]})
            for idx in ranked_indices
        ]


_cross_encoder_cache: dict[str, Any] = {}


def cross_encoder_scores(model: str, query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs with a sentence-transformers cross-encoder.

    Isolated here (mirroring `embed_texts`) so it's the single mockable seam
    for tests (no real model downloads/inference in the test suite). The
    loaded model is cached by id since construction loads weights from disk."""

    if not texts:
        return []

    from sentence_transformers import CrossEncoder

    cross_encoder = _cross_encoder_cache.get(model)
    if cross_encoder is None:
        cross_encoder = CrossEncoder(model)
        _cross_encoder_cache[model] = cross_encoder

    pairs = [[query, text] for text in texts]
    return [float(score) for score in cross_encoder.predict(pairs)]


class RerankRecallStrategy(BaseRecallStrategy):
    """Two-stage recall: BM25 + semantic candidate pool, cross-encoder reranked to top_k."""

    def __init__(
        self,
        semantic: SemanticRecallStrategy,
        *,
        cross_encoder_model: str,
        candidate_k: int,
    ):
        self._bm25 = BM25RecallStrategy()
        self._semantic = semantic
        self.cross_encoder_model = cross_encoder_model
        self.candidate_k = candidate_k

    @staticmethod
    def _dedup_key(entry: MemoryEntry) -> str:
        # Deliberately distinct from the embedding-cache key used elsewhere
        # in this module (content-only by design, so identical text reuses
        # one embedding regardless of metadata). Here we need to tell
        # *records* apart, not just text, so two active entries with the
        # same content but different metadata don't collapse into one
        # candidate.
        return json.dumps(
            {"content": entry.content, "metadata": entry.metadata},
            sort_keys=True,
            default=str,
        )

    def _candidate_pool(
        self, entries: list[MemoryEntry], query: str
    ) -> list[MemoryEntry]:
        bm25_candidates = self._bm25.rank(entries, query, self.candidate_k)
        semantic_candidates = self._semantic.rank(entries, query, self.candidate_k)

        pool: dict[str, MemoryEntry] = {}
        for entry in (*bm25_candidates, *semantic_candidates):
            pool.setdefault(self._dedup_key(entry), entry)
        return list(pool.values())

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        if not query or not entries or top_k <= 0:
            return []

        candidates = self._candidate_pool(entries, query)
        scores = cross_encoder_scores(
            self.cross_encoder_model, query, [entry.content for entry in candidates]
        )
        ranked_indices = sorted(
            range(len(candidates)), key=lambda idx: scores[idx], reverse=True
        )[:top_k]

        return [
            candidates[idx].model_copy(update={"score": scores[idx]})
            for idx in ranked_indices
        ]


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def upper_bound_bm25(
    raw_scores: list[float],
    query_tokens: list[str],
    idf: dict[str, float],
    k1: float,
) -> list[float]:
    """Scale raw BM25 into [0, 1] by the query's maximum attainable score.

    Each term contributes at most idf * (k1 + 1) (the tf -> inf limit).
    Duplicates and unknown terms are counted exactly as `get_scores` does.
    """
    bound = sum(idf.get(token, 0.0) for token in query_tokens) * (k1 + 1)
    if bound <= 0:
        return [0.0] * len(raw_scores)
    return [min(max(score / bound, 0.0), 1.0) for score in raw_scores]


def filter_count(
    sorted_scores: list[float],
    *,
    ratio: float,
    min_keep: int,
    min_score: float,
    top_k: int,
) -> int:
    """How many of the descending `sorted_scores` to keep."""
    window = sorted_scores[:top_k]
    above_floor = sum(1 for score in window if score >= min_score)
    if above_floor == 0:
        return 0
    cutoff = ratio * window[0]
    relative = sum(1 for score in window[:above_floor] if score >= cutoff)
    return min(max(relative, min_keep), above_floor)


def _recency(entry: MemoryEntry) -> float:
    parsed = parse_temporal_timestamp(entry.metadata.get("temporal_received_at"))
    return parsed.timestamp() if parsed is not None else float("-inf")


class HybridRecallStrategy(BaseRecallStrategy):
    """Weighted combination of normalized BM25 and semantic scores."""

    def __init__(
        self,
        semantic: SemanticRecallStrategy,
        *,
        weight: float,
        fusion: FusionKind = FusionKind.MINMAX,
        filter_ratio: float = 0.0,
        filter_min_keep: int = 1,
        filter_min_score: float = 0.0,
    ):
        self._semantic = semantic
        self._weight = weight
        self._fusion = fusion
        self._filter_ratio = filter_ratio
        self._filter_min_keep = filter_min_keep
        self._filter_min_score = filter_min_score

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        if not query or not entries or top_k <= 0:
            return []

        if self._fusion is FusionKind.ABSOLUTE:
            return self._rank_absolute(entries, query, top_k)

        tokenized_query = tokenize_text(query)
        if tokenized_query:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [tokenize_text(entry.content) for entry in entries]
            bm25_scores = (
                list(BM25Okapi(tokenized_corpus).get_scores(tokenized_query))
                if any(tokenized_corpus)
                else [0.0] * len(entries)
            )
        else:
            bm25_scores = [0.0] * len(entries)

        semantic_scores = self._semantic.score(entries, query)

        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_semantic = _normalize_scores(semantic_scores)

        combined = [
            (1 - self._weight) * bm25_score + self._weight * semantic_score
            for bm25_score, semantic_score in zip(normalized_bm25, normalized_semantic)
        ]

        ranked_indices = sorted(
            range(len(entries)), key=lambda idx: combined[idx], reverse=True
        )[:top_k]

        return [
            entries[idx].model_copy(update={"score": combined[idx]})
            for idx in ranked_indices
        ]

    def _rank_absolute(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        cosines = [
            min(max(score, 0.0), 1.0)
            for score in self._semantic.score(entries, query)
        ]
        tokenized_query = tokenize_text(query)
        tokenized_corpus = [tokenize_text(entry.content) for entry in entries]
        if tokenized_query and any(tokenized_corpus):
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = upper_bound_bm25(
                list(bm25.get_scores(tokenized_query)),
                tokenized_query,
                bm25.idf,
                bm25.k1,
            )
            combined = [
                (1 - self._weight) * bm25_score + self._weight * cosine
                for bm25_score, cosine in zip(bm25_scores, cosines)
            ]
        else:
            combined = cosines

        recency = [_recency(entry) for entry in entries]
        ranked_indices = sorted(
            range(len(entries)),
            key=lambda idx: (combined[idx], recency[idx]),
            reverse=True,
        )[:top_k]
        keep = filter_count(
            [combined[idx] for idx in ranked_indices],
            ratio=self._filter_ratio,
            min_keep=self._filter_min_keep,
            min_score=self._filter_min_score,
            top_k=top_k,
        )

        return [
            entries[idx].model_copy(update={"score": combined[idx]})
            for idx in ranked_indices[:keep]
        ]


def build_recall_strategy(
    config: RecallStrategyConfig,
    *,
    embedding_cache: EmbeddingCache,
    entry_key_fn: Any,
) -> BaseRecallStrategy:
    if config.kind is RecallStrategyKind.BM25:
        return BM25RecallStrategy()

    semantic = SemanticRecallStrategy(
        embed_model=config.embed_model,
        embedding_cache=embedding_cache,
        entry_key_fn=entry_key_fn,
        query_prompt=config.query_prompt,
    )

    if config.kind is RecallStrategyKind.SEMANTIC:
        return semantic

    if config.kind is RecallStrategyKind.HYBRID:
        return HybridRecallStrategy(
            semantic,
            weight=config.hybrid_weight,
            fusion=config.fusion,
            filter_ratio=config.filter_ratio,
            filter_min_keep=config.filter_min_keep,
            filter_min_score=config.filter_min_score,
        )

    if config.kind is RecallStrategyKind.RERANK:
        return RerankRecallStrategy(
            semantic,
            cross_encoder_model=config.cross_encoder_model,
            candidate_k=config.rerank_candidate_k,
        )

    raise ValueError(f"Unsupported recall strategy kind: {config.kind}")
