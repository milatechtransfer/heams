from __future__ import annotations

import math
from types import SimpleNamespace

import any_llm
import pytest
import tiktoken

from heams_memory_milamem import recall_strategies


@pytest.fixture
def embedding_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake_embedding(*, model: str, inputs: list[str], allow_running_loop: bool):
        assert allow_running_loop is True
        calls.append(list(inputs))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))]) for text in inputs]
        )

    monkeypatch.setattr(any_llm, "embedding", _fake_embedding)
    return calls


def test_empty_input_makes_no_call(embedding_calls: list[list[str]]) -> None:
    assert recall_strategies.embed_texts("openai:x", []) == []
    assert embedding_calls == []


def test_splits_by_input_count_and_preserves_order(
    embedding_calls: list[list[str]],
) -> None:
    texts = [f"t{i}" for i in range(5000)]

    vectors = recall_strategies.embed_texts("openai:x", texts)

    assert [len(call) for call in embedding_calls] == [2048, 2048, 904]
    assert [text for call in embedding_calls for text in call] == texts
    assert vectors == [[float(len(text))] for text in texts]


def _token_count(text: str) -> int:
    encoding = tiktoken.get_encoding(recall_strategies._EMBED_TOKENIZER)
    return len(encoding.encode_ordinary(text))


def test_splits_by_token_budget(embedding_calls: list[list[str]]) -> None:
    text = "QmFzZTY0IGJsb2I=" * 300
    assert _token_count(text) > len(text) / 4
    per_batch = recall_strategies._MAX_EMBED_BATCH_TOKENS // _token_count(text)
    texts = [text] * (2 * per_batch + 3)

    vectors = recall_strategies.embed_texts("openai:x", texts)

    assert [len(call) for call in embedding_calls] == [per_batch, per_batch, 3]
    assert len(vectors) == len(texts)


def test_splits_by_char_estimate(embedding_calls: list[list[str]]) -> None:
    # BEAM 1M failure: few tiktoken tokens, but len/4 over OpenAI's 300k cap.
    text = "memory " * 4000
    assert len(text) / 4 > _token_count(text)
    per_batch = recall_strategies._MAX_EMBED_BATCH_TOKENS // math.ceil(len(text) / 4)
    texts = [text] * (2 * per_batch + 3)

    vectors = recall_strategies.embed_texts("openai:x", texts)

    assert [len(call) for call in embedding_calls] == [per_batch, per_batch, 3]
    assert all(sum(map(len, call)) / 4 <= 300_000 for call in embedding_calls)
    assert len(vectors) == len(texts)


def test_oversized_text_is_truncated_to_token_cap(embedding_calls: list[list[str]]) -> None:
    # Base64-like text tokenizes near 1 char/token, which broke a char-based cap.
    blob = "QmFzZTY0IGJsb2I=" * 5000
    texts = ["short", blob, "short"]

    vectors = recall_strategies.embed_texts("openai:x", texts)

    [[first, sent, last]] = embedding_calls
    assert (first, last) == ("short", "short")
    assert _token_count(sent) == recall_strategies._MAX_EMBED_INPUT_TOKENS
    assert blob.startswith(sent)
    assert len(vectors) == 3


def test_special_token_text_is_embedded_verbatim(embedding_calls: list[list[str]]) -> None:
    recall_strategies.embed_texts("openai:x", ["see <|endoftext|> in the prompt"])

    assert embedding_calls == [["see <|endoftext|> in the prompt"]]
