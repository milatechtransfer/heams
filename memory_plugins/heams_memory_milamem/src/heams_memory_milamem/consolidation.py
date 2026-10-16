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
    cluster_system_prompt: str = Field(
        default=(
            "You judge whether a cluster of candidate memory records are "
            "duplicates, conflict, are related, or are unrelated. Choose "
            "exactly one:\n"
            "- duplicate: the records convey the same information; "
            "keeping only one loses nothing.\n"
            "- conflict: a later record explicitly says a value stated in "
            "an earlier record is no longer true (it was changed, "
            "replaced, resolved, switched, moved, etc.), or the records "
            "assert mutually exclusive current values for the same "
            "attribute of the same subject.\n"
            "- related: same subject or topic, but all records can be "
            "true at once and each carries information the others lack. "
            'Hypothetical or conditional statements ("if X changes, I '
            'will ..."), plans, questions and advice never conflict with '
            "a currently stated fact.\n"
            "- unrelated: different subjects (a similarity-search false "
            "positive).\n"
            "When duplicate or conflict, pick the single record id that "
            "should remain canonical: for conflict, the record stating "
            "the current value (normally the most recent); for "
            "duplicate, the most complete one."
        ),
        description="System prompt for cluster consolidation and retain-time reconciliation.",
    )
    anchor_system_prompt: str = Field(
        default=(
            "You maintain a user's long-term memory. You get one NEW "
            "memory record and several EARLIER records that were "
            "retrieved because they look similar. Records may be "
            "conversation turns; ignore small talk and generic advice "
            "and compare only facts about the user. For EACH earlier "
            "record, decide its relationship to the NEW record:\n"
            "- duplicate: it states the same specific facts as the NEW "
            "record, so it adds nothing. Same topic with different "
            "details is NOT a duplicate.\n"
            "- conflict: it states a value for a specific attribute of "
            "the user (e.g. city of residence, employer, diet, "
            "medication, device) that the NEW record shows is no longer "
            'current, either explicitly ("I moved", "it was '
            'resolved", "I switched to") or through a new situation '
            "that cannot coexist with it (e.g. 'settling into my new "
            "place in Austin' vs 'I've been based in Seattle'). You must "
            "be able to name the attribute and both values.\n"
            "- related: same topic but both can be true at once. "
            'Hypothetical or conditional statements ("if X changes, I '
            'will ..."), plans, questions and advice never make an '
            "earlier fact outdated.\n"
            "- unrelated: different subject.\n"
            "Return one verdict per earlier record id."
        ),
        description="System prompt for anchor-mode consolidation.",
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
    anchor_top_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When set, replaces clustering: each pending record is judged "
            "against its top-k most similar earlier active records of the "
            "same role (>= similarity_threshold), with one verdict per "
            "candidate. Only candidates named duplicate/conflict are "
            "superseded, in favour of the pending record."
        ),
    )
    anchor_roles: list[str] | None = Field(
        default=None,
        description=(
            "Anchor mode only: pending records act as anchors only if their "
            "metadata role is listed. None means every role."
        ),
    )
    judge_concurrency: int = Field(
        default=8,
        ge=1,
        description="Anchor mode only: parallel judge calls per consolidate().",
    )


class ConsolidationVerdict(BaseModel):
    relationship: Literal["duplicate", "conflict", "related", "unrelated"]
    canonical_id: str | None = Field(
        default=None,
        description=(
            "id of the record that should remain active; required unless "
            "relationship is 'related' or 'unrelated'."
        ),
    )
    reasoning: str = Field(default="", description="Brief justification.")


class AnchorCandidateVerdict(BaseModel):
    id: str
    relationship: Literal["duplicate", "conflict", "related", "unrelated"]


class AnchorVerdict(BaseModel):
    verdicts: list[AnchorCandidateVerdict]


def _record_vectors(
    records: list[CanonicalMemoryRecord],
    *,
    embed_model: str,
    embedding_cache: EmbeddingCache,
    entry_key_fn: Any,
) -> list[np.ndarray]:
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
    return [np.array(embedding_cache.get(key)) for key in keys]


def anchor_candidates(
    records: list[CanonicalMemoryRecord],
    *,
    embed_model: str,
    similarity_threshold: float,
    embedding_cache: EmbeddingCache,
    entry_key_fn: Any,
    pending_ids: set[str],
    top_k: int,
    anchor_roles: list[str] | None,
) -> list[tuple[CanonicalMemoryRecord, list[CanonicalMemoryRecord]]]:
    """Pairs each pending record (in `sequence` order) with its top-k most
    similar earlier records of the same metadata role. Anchors with no
    candidate above `similarity_threshold` are left out."""
    if not records:
        return []
    matrix = np.array(
        _record_vectors(
            records,
            embed_model=embed_model,
            embedding_cache=embedding_cache,
            entry_key_fn=entry_key_fn,
        ),
        dtype=float,
    )
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    groups = []
    for i in sorted(range(len(records)), key=lambda index: records[index].sequence):
        anchor = records[i]
        role = anchor.metadata.get("role")
        if anchor.id not in pending_ids or (
            anchor_roles is not None and role not in anchor_roles
        ):
            continue
        similarities = matrix @ matrix[i]
        scored = [
            (similarities[j], record)
            for j, record in enumerate(records)
            if record.sequence < anchor.sequence
            and record.metadata.get("role") == role
            and similarities[j] >= similarity_threshold
        ]
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            groups.append((anchor, [record for _, record in scored[:top_k]]))
    return groups


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

    vectors = _record_vectors(
        records,
        embed_model=embed_model,
        embedding_cache=embedding_cache,
        entry_key_fn=entry_key_fn,
    )

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
    config: ConsolidationConfig,
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
            "content": config.cluster_system_prompt,
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
        model=config.llm_model,
        messages=messages,
        response_format=ConsolidationVerdict,
        allow_running_loop=True,
    )
    content = response.choices[0].message.content
    return ConsolidationVerdict.model_validate_json(content)


def judge_anchor(
    config: ConsolidationConfig,
    anchor: CanonicalMemoryRecord,
    candidates: list[CanonicalMemoryRecord],
) -> AnchorVerdict:
    """One verdict per earlier candidate, relative to the newer anchor."""
    import any_llm

    listing = "\n".join(
        f"- id={record.id!r} content={record.content!r}" for record in candidates
    )
    messages = [
        {
            "role": "system",
            "content": config.anchor_system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"NEW record:\n{anchor.content!r}\n\nEARLIER records:\n{listing}"
            ),
        },
    ]
    recall_strategies._ensure_thread_event_loop()
    response = any_llm.completion(
        model=config.llm_model,
        messages=messages,
        response_format=AnchorVerdict,
        allow_running_loop=True,
    )
    return AnchorVerdict.model_validate_json(response.choices[0].message.content)


__all__ = [
    "AnchorVerdict",
    "ConsolidationConfig",
    "ConsolidationVerdict",
    "anchor_candidates",
    "cluster_by_similarity",
    "chunk_cluster",
    "judge_anchor",
    "judge_cluster",
]
