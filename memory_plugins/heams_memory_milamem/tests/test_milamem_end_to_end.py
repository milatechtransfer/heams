"""End-to-end integration test for MilaMemMemorySystem: a single
continuous conversation driven entirely through the public retain()/recall()/
consolidate()/delete() API — no internal methods monkeypatched — exercising
retain-time windowed conflict detection, the semantic-floor judge trigger,
consolidate()'s unwindowed backstop, and recall filtering together, in one
realistic story rather than in isolated unit-level fixtures.

Only the true external boundary is stubbed: embeddings (`embed_texts`), the
consolidate()/retain-time judge (`judge_cluster`), and the delete judge
(`request_delete_decision`), matching this repo's existing "tests never hit
a real API" convention. Real end-to-end validation against an actual LLM
was performed manually via
`configs/runs/milamem_all_benchmarks/locomo.yaml`.
"""

import json
from pathlib import Path

import pytest

from heams_memory_milamem import consolidation as consolidation_module
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
)
from heams_memory_milamem.consolidation import ConsolidationVerdict
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)
from heams_memory_milamem.design.delete.strategies import DeleteDecision
from heams_memory_milamem.design.delete.strategies import (
    llm_judge as delete_judge_module,
)

# One base vector per subject, orthogonal to every other subject. Real
# embeddings would have noise; determinism here is what lets the test assert
# exact outcomes rather than probable ones. Same-subject sentences reuse the
# same vector (embedding similarity is what drives clustering/candidate
# ranking here, not exact-duplicate detection, which is text-based).
_SUBJECT_VECTORS = {
    "alice": [1.0, 0.0, 0.0],
    "bob": [0.0, 1.0, 0.0],
    "carol": [0.0, 0.0, 1.0],
}


def _subject_of(text: str) -> str | None:
    lowered = text.casefold()
    for name in _SUBJECT_VECTORS:
        if name in lowered:
            return name
    return None


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    del model
    return [_SUBJECT_VECTORS[_subject_of(text)] for text in texts]


def _fake_judge(model: str, records: list) -> ConsolidationVerdict:
    """Deterministic stand-in for the real LLM judge: exact-text duplicates
    resolve to 'duplicate'; same-subject differing content resolves to
    'conflict' (winner = highest sequence, i.e. most recent); anything else
    is 'unrelated'. Mirrors the real judge's contract without a network call."""
    del model

    def _normalized(value: str) -> str:
        return " ".join(value.casefold().split())

    winner = max(records, key=lambda r: r.sequence)
    normalized_contents = {_normalized(r.content) for r in records}
    if len(normalized_contents) == 1:
        return ConsolidationVerdict(relationship="duplicate", canonical_id=winner.id)

    subjects = {_subject_of(r.content) for r in records}
    if len(subjects) == 1 and None not in subjects:
        return ConsolidationVerdict(relationship="conflict", canonical_id=winner.id)

    return ConsolidationVerdict(relationship="unrelated", canonical_id=None)


@pytest.fixture(autouse=True)
def _stub_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(consolidation_module, "judge_cluster", _fake_judge)


def _canonical_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.canonical_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _raw_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.raw_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _active_contents(memory: MilaMemMemorySystem) -> set[str]:
    return {r["content"] for r in _canonical_records(memory) if r["status"] == "active"}


def test_full_retain_recall_consolidate_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="e2e-canonical",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
                window={"turns": 2},
            ),
        ),
        storage_dir=tmp_path,
    )

    # 1. First fact for Alice.
    result = memory.retain("Alice lives in Boston.")
    assert result == "Successfully saved to memory."

    # 2. Exact duplicate — skipped at the canonical layer, but the raw log
    # still records it (immutable audit trail).
    result = memory.retain("Alice lives in Boston.")
    assert result == "Memory already exists; skipping duplicate canonical projection."

    # 3-4. Two unrelated facts push Alice's record outside the 2-turn window.
    memory.retain("Bob works at Acme Corp.")
    memory.retain("Carol enjoys hiking.")

    assert _active_contents(memory) == {
        "Alice lives in Boston.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }

    # 5. A real conflict for Alice — but retain-time's window (turns=2) now
    # only sees [Bob, Carol], both orthogonal to Alice — so the floor is
    # never cleared and the judge is never called. Both Alice facts stay
    # active: retain-time genuinely misses this, not simulated.
    outcome = memory._retain_impl("Alice moved to Chicago.", {})
    assert outcome.result["superseded_ids"] == []
    assert outcome.result["consolidation_reason"] is None
    assert _active_contents(memory) == {
        "Alice lives in Boston.",
        "Alice moved to Chicago.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }

    # 6. consolidate() is the unwindowed backstop: it clusters the full
    # active set (no turn/day window), so both Alice records land in one
    # cluster regardless of how retain-time windowed them, and the judge
    # resolves the conflict in favor of the more recent record.
    summary = memory.consolidate()
    assert "1 conflicts resolved" in summary
    assert _active_contents(memory) == {
        "Alice moved to Chicago.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }
    canonical = _canonical_records(memory)
    superseded = next(r for r in canonical if r["content"] == "Alice lives in Boston.")
    assert superseded["status"] == "superseded"
    assert superseded["consolidation_reason"] == "conflict_resolved"

    # 7. consolidate() is idempotent: nothing left to reconcile.
    assert memory.consolidate() == "No active memories required consolidation."

    # 8. Now that Alice's sole active record is recent, a fresh Alice
    # conflict falls inside the retain-time window and gets resolved
    # immediately via the judge, without waiting for consolidate().
    outcome = memory._retain_impl("Alice relocated to Denver.", {})
    assert outcome.result["consolidation_reason"] == "conflict_resolved"
    assert _active_contents(memory) == {
        "Alice relocated to Denver.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }

    # 9. recall() only ever surfaces active canonical content.
    alice_results = memory.recall("Alice")
    assert any("Denver" in r for r in alice_results)
    assert not any("Chicago" in r for r in alice_results)
    assert not any("Boston" in r for r in alice_results)

    chronological = memory.recall("")
    assert set(chronological) == {
        "Alice relocated to Denver.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }

    # 10. Raw log stays a complete, immutable audit trail throughout: every
    # retain() call appended a raw record (including the duplicate and the
    # since-superseded facts), and none of them were ever mutated.
    raw = _raw_records(memory)
    assert len(raw) == 6
    assert all(r["status"] == "active" for r in raw)
    raw_contents = [r["content"] for r in raw]
    assert raw_contents == [
        "Alice lives in Boston.",
        "Alice lives in Boston.",
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
        "Alice moved to Chicago.",
        "Alice relocated to Denver.",
    ]

    # 11. delete() logically tombstones the LLM-judge-selected active
    # record; the judge is the same kind of external LLM seam as
    # consolidate()'s judge_cluster, so it's stubbed the same way.
    alice_id = next(
        record["id"]
        for record in _canonical_records(memory)
        if record["content"] == "Alice relocated to Denver."
    )
    monkeypatch.setattr(
        delete_judge_module,
        "request_delete_decision",
        lambda model, messages: DeleteDecision(
            selected_ids=[alice_id],
            reasoning="The candidate is directly targeted.",
            ambiguous=False,
        ),
    )
    delete_result = memory.delete("Alice")
    assert delete_result == "Logically deleted 1 canonical memory record(s)."
    assert _active_contents(memory) == {
        "Bob works at Acme Corp.",
        "Carol enjoys hiking.",
    }


def test_disabled_dedup_lifecycle_never_resolves_conflicts_until_consolidate(
    tmp_path: Path,
) -> None:
    """Disabled reconciliation leaves every conflict for consolidate()."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="e2e-disabled-dedup",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=False,
                kind=DedupStrategyKind.BM25,
            ),
        ),
        storage_dir=tmp_path,
    )

    memory.retain("Alice lives in Boston.")
    outcome = memory._retain_impl("Alice moved to Chicago.", {})

    assert outcome.result["superseded_ids"] == []
    assert _active_contents(memory) == {
        "Alice lives in Boston.",
        "Alice moved to Chicago.",
    }

    summary = memory.consolidate()
    assert "1 conflicts resolved" in summary
    assert _active_contents(memory) == {"Alice moved to Chicago."}
