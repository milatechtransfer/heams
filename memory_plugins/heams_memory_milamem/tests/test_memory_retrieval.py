from heams_memory_milamem.retrieval import bm25_top_k
from heams.memory_systems.api import MemoryEntry

_FACTS = [
    MemoryEntry(content="Full name is Maria Chen", metadata={"ts": 30}),
    MemoryEntry(content="Email is maria.chen@university.edu", metadata={"ts": 29}),
    MemoryEntry(content="Phone number is 555-0142", metadata={"ts": 15}),
    MemoryEntry(content="Employer is State University", metadata={"ts": 10}),
    MemoryEntry(content="Prefers window seating", metadata={"ts": 5}),
    MemoryEntry(content="BMI is 23", metadata={"ts": 4}),
]


def test_bm25_top_k_picks_field_specific_entry() -> None:
    [top] = bm25_top_k(_FACTS, query="phone", k=1)
    assert top.content == "Phone number is 555-0142"
    assert top.score is not None and top.score > 0


def test_bm25_top_k_populates_scores_in_descending_order() -> None:
    results = bm25_top_k(_FACTS, query="university email", k=3)
    scores = [entry.score for entry in results]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_top_k_empty_query_returns_empty_list() -> None:
    assert bm25_top_k(_FACTS, query="", k=1) == []


def test_bm25_top_k_empty_corpus_returns_empty_list() -> None:
    assert bm25_top_k([], query="anything", k=3) == []


def test_bm25_top_k_ties_preserve_chronological_order() -> None:
    # Two entries that contain neither query token get the same BM25 score of
    # 0; the helper must keep them in insertion order.
    corpus = [
        MemoryEntry(content="alpha"),
        MemoryEntry(content="beta"),
        MemoryEntry(content="gamma"),
    ]
    results = bm25_top_k(corpus, query="zzz", k=3)
    assert [entry.content for entry in results] == ["alpha", "beta", "gamma"]
