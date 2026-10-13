"""Shared BM25 retrieval helper for the bundled memory backends.

Earlier deterministic backends used to ignore the `query` argument of
`recall` and return the most recent entries unconditionally. That made the
benchmark unable to distinguish backends on anything more complex than a
single-fact fixture. This module provides a single `bm25_top_k` helper the
deterministic backends delegate to when a non-empty query is supplied.

BM25 was chosen over semantic / embedding retrieval intentionally: it is
fully deterministic, carries no model download, and is a credible baseline
for the short factoid-style entries these scenarios use.
"""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from heams.memory_systems.api import MemoryEntry

_TOKEN_RE = re.compile(r"[0-9a-zA-Z]+")


def tokenize_text(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics.

    Short entries ("Phone number is 555-0142") do not warrant stemming or
    stop-word filtering; BM25 handles the rest.
    """
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def bm25_top_k(entries: Sequence[MemoryEntry], query: str, k: int) -> list[MemoryEntry]:
    """Return the top `k` entries scored against `query` via BM25.

    Entries are returned as fresh `MemoryEntry` copies with `score` populated
    so callers can inspect ranking quality. Ties keep the original
    (chronological) order of `entries`. An empty query, empty corpus, or
    non-positive `k` short-circuits to an empty list.
    """

    if not query or not entries or k <= 0:
        return []

    tokenized_corpus = [tokenize_text(entry.content) for entry in entries]
    # BM25Okapi raises on an empty corpus; guard explicitly so the caller
    # never has to.
    if not any(tokenized_corpus):
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize_text(query))

    # `sorted` is stable in Python, so enumerating in chronological order and
    # sorting by descending score preserves the original tie-break contract.
    ranked_indices = sorted(
        range(len(entries)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:k]

    return [
        entries[idx].model_copy(update={"score": float(scores[idx])})
        for idx in ranked_indices
    ]
