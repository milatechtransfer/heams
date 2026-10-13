"""LLM + clustering consolidation: the global, unwindowed backstop for
canonical-store duplicates/conflicts that retain-time's windowed local
search missed. Groups active canonical records by embedding similarity,
then asks an LLM to judge each cluster (duplicate / conflict / unrelated)
and pick which existing record should remain canonical."""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
from heams.memory_systems.api import BaseConsolidationConfig
from pydantic import BaseModel, Field

from . import recall_strategies
from .design.storage_type import CanonicalMemoryRecord
from .recall_strategies import EmbeddingCache


class ConsolidationConfig(BaseConsolidationConfig):
    enabled: bool = Field(
        default=True,
        description="Whether canonical consolidation may run.",
    )
    pending_threshold: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Automatically consolidate when unresolved canonical records reach "
            "this count. None disables this trigger."
        ),
    )
    max_pending_per_run: int = Field(
        default=50,
        ge=1,
        description="Maximum pending record IDs processed by one consolidation pass.",
    )
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description="any_llm model id used to judge clustered canonical records.",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold (over the dedup_strategy embedding "
            "space) for clustering active canonical records before LLM "
            "judgment. Records below this threshold to every other record "
            "form their own singleton cluster and are never judged."
        ),
    )
    max_cluster_size: int = Field(
        default=8,
        ge=2,
        description=(
            "Cap on records judged in a single LLM call. Similarity "
            "clusters larger than this are split into consecutive chunks "
            "(by `sequence`) of this size, to bound prompt size/cost."
        ),
    )


class ConsolidationVerdict(BaseModel):
    relationship: Literal["duplicate", "conflict", "unrelated"]
    canonical_id: str | None = Field(
        default=None,
        description=(
            "id of the record that should remain active; required unless "
            "relationship is 'unrelated'."
        ),
    )
    reasoning: str = Field(default="", description="Brief justification.")


def cluster_by_similarity(
    records: list[CanonicalMemoryRecord],
    *,
    embed_model: str,
    similarity_threshold: float,
    embedding_cache: EmbeddingCache,
    entry_key_fn: Any,
    pending_ids: set[str] | None = None,
) -> list[list[CanonicalMemoryRecord]]:
    """Group `records` into connected components via pairwise cosine
    similarity ≥ `similarity_threshold` (single-linkage, via union-find).
    Records with no sufficiently similar peer end up as singletons.

    `pending_ids`, when given, skips pairs where neither id is in the
    set — those are assumed already resolved by an earlier pass."""
    if not records:
        return []

    missing_keys: list[str] = []
    missing_texts: list[str] = []
    keys = [entry_key_fn(record) for record in records]
    for key, record in zip(keys, records):
        if key not in embedding_cache:
            missing_keys.append(key)
            missing_texts.append(record.content)
    if missing_texts:
        vectors = recall_strategies.embed_texts(embed_model, missing_texts)
        for key, vector in zip(missing_keys, vectors):
            embedding_cache.set(key, vector)

    vectors = [np.array(embedding_cache.get(key)) for key in keys]

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    if pending_ids is None:
        pairs = itertools.combinations(range(len(records)), 2)
    else:
        pending_indices = [i for i, r in enumerate(records) if r.id in pending_ids]
        other_indices = [i for i, r in enumerate(records) if r.id not in pending_ids]
        # Old-vs-old pairs are assumed already resolved by an earlier pass,
        # so only pairs touching at least one pending record are considered.
        pairs = itertools.chain(
            itertools.combinations(pending_indices, 2),
            itertools.product(pending_indices, other_indices),
        )

    for i, j in pairs:
        similarity = recall_strategies.cosine_similarity(vectors[i], vectors[j])
        if similarity >= similarity_threshold:
            union(i, j)

    clusters_by_root: dict[int, list[CanonicalMemoryRecord]] = {}
    for index, record in enumerate(records):
        clusters_by_root.setdefault(find(index), []).append(record)

    return list(clusters_by_root.values())


def chunk_cluster(
    cluster: list[CanonicalMemoryRecord],
    *,
    max_size: int,
) -> list[list[CanonicalMemoryRecord]]:
    """Split an oversized cluster (ordered by `sequence`) into consecutive
    chunks of at most `max_size` records, so no single LLM call gets an
    unbounded prompt."""
    ordered = sorted(cluster, key=lambda record: record.sequence)
    return [ordered[i : i + max_size] for i in range(0, len(ordered), max_size)]


def judge_cluster(
    model: str,
    records: list[CanonicalMemoryRecord],
) -> ConsolidationVerdict:
    """The isolated LLM call — single mockable seam for tests, mirroring
    `embed_texts`'s isolation rationale (no real API calls in the test
    suite)."""
    import any_llm

    listing = "\n".join(
        f"- id={record.id!r} sequence={record.sequence} content={record.content!r}"
        for record in records
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You judge whether a cluster of candidate memory records "
                "are duplicates of each other, conflict with each other "
                "(mutually exclusive facts about the same subject), or are "
                "actually unrelated (a similarity-search false positive). "
                "When duplicate or conflict, pick the single record id that "
                "should remain canonical — normally the most recent/most "
                "specific one."
            ),
        },
        {
            "role": "user",
            "content": f"Candidate records:\n{listing}",
        },
    ]

    # `retain`/`recall`/`consolidate` run synchronously but get invoked as
    # agent tools from inside a running event loop (TinyAgent's async
    # tool-call loop); without `allow_running_loop=True`, any_llm's sync
    # API refuses to run there at all and raises instead of completing —
    # same hazard as `embed_texts` in recall_strategies.py.
    recall_strategies._ensure_thread_event_loop()
    response = any_llm.completion(
        model=model,
        messages=messages,
        response_format=ConsolidationVerdict,
        allow_running_loop=True,
    )
    content = response.choices[0].message.content
    return ConsolidationVerdict.model_validate_json(content)


__all__ = [
    "ConsolidationConfig",
    "ConsolidationVerdict",
    "cluster_by_similarity",
    "chunk_cluster",
    "judge_cluster",
]
