"""MilaMem reports the usage of its model calls on each operation's event."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import any_llm
import pytest
import sentence_transformers
import torch
from heams.memory_systems.api import MemoryOperationEvent, ModelCallKind

from heams_memory_milamem import recall_strategies
from heams_memory_milamem.consolidation import ConsolidationConfig
from heams_memory_milamem.dedup_strategies import DedupStrategyConfig, DedupStrategyKind
from heams_memory_milamem.milamem import MilaMemConfig, MilaMemMemorySystem
from heams_memory_milamem.recall_strategies import (
    RecallStrategyConfig,
    RecallStrategyKind,
)


def _vector(text: str) -> list[float]:
    """Texts sharing a prefix before "|" embed identically."""
    digest = hashlib.md5(text.split("|", 1)[0].encode()).hexdigest()
    return [int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


@pytest.fixture(autouse=True)
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embeddings bill 2 tokens per text; each judge call 30 in and 5 out."""

    def embedding(*, model: str, inputs: list[str], allow_running_loop: bool):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=_vector(text)) for text in inputs],
            usage=SimpleNamespace(prompt_tokens=2 * len(inputs)),
        )

    def completion(**kwargs):
        anchor = "verdicts" in kwargs["messages"][0]["content"].lower()
        content = '{"verdicts": []}' if anchor else '{"relationship": "related"}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5),
        )

    monkeypatch.setattr(any_llm, "embedding", embedding)
    monkeypatch.setattr(any_llm, "completion", completion)


def _memory(tmp_path: Path, ledger: list, **config: object) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        MilaMemConfig(
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            **config,
        ),
        storage_dir=tmp_path,
        event_sink=ledger,
    )


def _totals(event: MemoryOperationEvent) -> dict[ModelCallKind, tuple[int, int, int]]:
    totals: dict[ModelCallKind, tuple[int, int, int]] = {}
    for record in event.usage or []:
        calls, tokens_in, tokens_out = totals.get(record.kind, (0, 0, 0))
        totals[record.kind] = (
            calls + record.call_count,
            tokens_in + record.input_tokens,
            tokens_out + record.output_tokens,
        )
    return totals


def test_retain_dedup_reports_its_embeddings_and_judge_call(tmp_path: Path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        dedup_strategy=DedupStrategyConfig(enabled=True, kind=DedupStrategyKind.SEMANTIC),
    )

    memory.retain("same|old fact")
    memory.retain("same|new fact")

    first, second = ledger
    assert ModelCallKind.LLM not in _totals(first)
    assert _totals(second)[ModelCallKind.LLM] == (1, 30, 5)
    assert _totals(second)[ModelCallKind.EMBEDDING][1] > 0
    (llm,) = [r for r in second.usage if r.kind is ModelCallKind.LLM]
    assert llm.model == ConsolidationConfig().llm_model
    assert not llm.local


def test_consolidation_judges_on_worker_threads_are_reported(tmp_path: Path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        dedup_strategy=DedupStrategyConfig(enabled=False, kind=DedupStrategyKind.SEMANTIC),
        consolidation=ConsolidationConfig(anchor_top_k=1, judge_concurrency=2),
    )
    memory.retain("same|old fact")
    memory.retain("same|new fact")
    memory.retain("other|fact")

    memory.consolidate()

    consolidate = ledger[-1]
    assert consolidate.operation_type == "consolidate"
    calls, _, _ = _totals(consolidate)[ModelCallKind.LLM]
    assert calls >= 1
    # Every judge call is counted once, on the consolidation only.
    assert sum(_totals(event).get(ModelCallKind.LLM, (0,))[0] for event in ledger) == calls


def test_semantic_recall_reports_its_query_embedding(tmp_path: Path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC),
    )
    memory.retain("lives in Montreal")

    memory.recall("where does the user live?")

    recall = ledger[-1]
    assert recall.operation_type == "recall"
    assert ModelCallKind.EMBEDDING in _totals(recall)
    assert ModelCallKind.LLM not in _totals(recall)


def test_bm25_only_operations_report_zero(tmp_path: Path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        dedup_strategy=DedupStrategyConfig(enabled=False, kind=DedupStrategyKind.BM25),
    )

    memory.retain("lives in Montreal")
    memory.recall("Montreal")

    assert [event.usage for event in ledger] == [[], []]


def test_local_sentence_transformer_embeddings_count_tokens_as_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSentenceTransformer:
        max_seq_length = 512

        def __init__(self, name: str, *, model_kwargs=None) -> None:
            self.name = name

        def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
            # One token per word plus two special tokens, padded to the longest.
            lengths = [len(text.split()) + 2 for text in texts]
            width = max(lengths)
            return {
                "attention_mask": torch.tensor(
                    [[1] * n + [0] * (width - n) for n in lengths]
                )
            }

        def encode(self, texts, **kwargs):
            return torch.tensor([_vector(text) for text in texts]).numpy()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(recall_strategies, "_st_models", {})
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        recall_strategy=RecallStrategyConfig(
            kind=RecallStrategyKind.SEMANTIC, embed_model="st:fake/embedder"
        ),
    )
    memory.retain("lives in Montreal")

    memory.recall("where")

    (record,) = ledger[-1].usage
    assert record.local
    assert record.model == "st:fake/embedder"
    # Recall embeds the stored entry (3 words) and the query (1 word), each
    # with 2 special tokens; padding is not counted.
    assert (record.call_count, record.input_tokens) == (2, 8)
    assert ledger[0].usage == []  # retain made no model call


def test_model_usage_is_the_sum_of_every_event(tmp_path: Path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = _memory(
        tmp_path,
        ledger,
        dedup_strategy=DedupStrategyConfig(enabled=True, kind=DedupStrategyKind.SEMANTIC),
        recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC),
    )
    memory.retain("same|old fact")
    memory.retain("same|new fact")
    memory.recall("same")

    logged = sum(r.input_tokens for event in ledger for r in event.usage or [])
    assert logged == sum(r.input_tokens for r in memory.model_usage())
