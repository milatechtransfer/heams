from __future__ import annotations

import any_llm
import numpy as np
import pytest
import sentence_transformers
import torch
from heams.memory_systems.api import MemoryEntry

from heams_memory_milamem import recall_strategies
from heams_memory_milamem.recall_strategies import (
    EmbeddingCache,
    RecallStrategyConfig,
    RecallStrategyKind,
    SemanticRecallStrategy,
    build_recall_strategy,
)


class _FakeSentenceTransformer:
    instances: list[_FakeSentenceTransformer] = []

    def __init__(self, name: str, *, model_kwargs=None):
        self.name = name
        self.model_kwargs = model_kwargs
        self.max_seq_length = 32768
        self.encoded: list[list[str]] = []
        _FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, *, batch_size, normalize_embeddings, convert_to_numpy):
        assert normalize_embeddings is True and convert_to_numpy is True
        assert batch_size == 8
        self.encoded.append(list(texts))
        return np.array([[float(len(text)), 1.0] for text in texts])


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    _FakeSentenceTransformer.instances = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(recall_strategies, "_st_models", {})

    def _no_api(**kwargs):
        raise AssertionError("st: models must not call any_llm")

    monkeypatch.setattr(any_llm, "embedding", _no_api)
    return _FakeSentenceTransformer


def test_st_model_runs_in_process_and_loads_once(fake_st) -> None:
    first = recall_strategies.embed_texts("st:Qwen/Qwen3-Embedding-0.6B", ["ab", "abc"])
    second = recall_strategies.embed_texts("st:Qwen/Qwen3-Embedding-0.6B", ["a"])

    assert first == [[2.0, 1.0], [3.0, 1.0]]
    assert second == [[1.0, 1.0]]
    assert len(fake_st.instances) == 1
    model = fake_st.instances[0]
    assert model.name == "Qwen/Qwen3-Embedding-0.6B"
    assert model.max_seq_length == 8192
    assert model.model_kwargs == {"torch_dtype": torch.bfloat16}


def test_st_empty_input_does_not_load_model(fake_st) -> None:
    assert recall_strategies.embed_texts("st:Qwen/Qwen3-Embedding-0.6B", []) == []
    assert fake_st.instances == []


def test_query_prompt_applies_to_query_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake(model: str, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(recall_strategies, "embed_texts", _fake)
    cache = EmbeddingCache()
    strategy = SemanticRecallStrategy(
        embed_model="st:fake",
        embedding_cache=cache,
        entry_key_fn=lambda entry: entry.content,
        query_prompt="Instruct: find memories\nQuery: ",
    )

    strategy.score([MemoryEntry(content="alice lives in paris")], "where is alice")

    assert calls == [["alice lives in paris"], ["Instruct: find memories\nQuery: where is alice"]]
    assert "alice lives in paris" in cache


def test_build_recall_strategy_passes_query_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake(model: str, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(recall_strategies, "embed_texts", _fake)
    strategy = build_recall_strategy(
        RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC, query_prompt="Q: "),
        embedding_cache=EmbeddingCache(),
        entry_key_fn=lambda entry: entry.content,
    )

    strategy.rank([MemoryEntry(content="note")], "hello", top_k=1)

    assert calls[-1] == ["Q: hello"]


def test_no_query_prompt_leaves_query_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake(model: str, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(recall_strategies, "embed_texts", _fake)
    strategy = SemanticRecallStrategy(
        embed_model="openai:x",
        embedding_cache=EmbeddingCache(),
        entry_key_fn=lambda entry: entry.content,
    )

    strategy.score([MemoryEntry(content="note")], "hello")

    assert calls[-1] == ["hello"]
