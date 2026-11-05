"""Unit tests for the Letta (MemGPT) memory system.

Letta's in-process client is replaced by a fake that records every REST call and
returns scripted agent responses, so these tests run without Postgres or an API key.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tiktoken
from heams.memory_systems.api import MemoryEntry
from heams.memory_systems.plugins import discover_memory_plugins
from heams_memory_letta import LettaMemoryConfig, LettaMemorySystem, plugin
from pydantic import ValidationError

# The package re-exports the descriptor as `plugin`, which shadows the submodule name.
letta_plugin = importlib.import_module("heams_memory_letta.plugin")

AGENT_ID = "agent-1"
ARCHIVE_ID = "archive-1"


class FakeResponse:
    def __init__(self, body: Any = None, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self) -> Any:
        return self._body


class FakeLetta:
    """Records requests; answers agent messages from a queue of scripted responses."""

    def __init__(self, version: str = letta_plugin.LETTA_VERSION) -> None:
        self.version = version
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.replies: deque[dict[str, Any]] = deque()
        self.blocks = [
            {"label": "persona", "value": "I am Sam."},
            {"label": "human", "value": ""},
        ]
        # Applied to core memory when the next agent message is handled.
        self.core_edits: deque[str | None] = deque()
        self.fail_messages = False
        self.fail_inserts = False
        self.fail_attach = False
        self.fail_agent_delete = False
        self.inserted: list[list[dict[str, Any]]] = []
        # (tool rules in force, message) per agent message, and the most requests seen
        # inside a message at once: what the instance lock must keep to one.
        self.rules_at_message: list[tuple[list[dict[str, str]], str]] = []
        self.current_rules: list[dict[str, str]] = []
        self.inflight = self.max_inflight = 0
        # Letta's message log (what conversation_search reads) and the agent's context.
        self.log = ["system"]
        self.context = ["system"]

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        path = url.removeprefix("/v1")
        self.calls.append((method, path, kwargs))
        if path == "/health/":
            return FakeResponse({"version": self.version, "status": "ok"})
        if method == "POST" and path == "/agents/":
            self.current_rules = kwargs["json"]["tool_rules"]
            return FakeResponse({"id": AGENT_ID})
        if method == "GET" and path == f"/agents/{AGENT_ID}":
            return FakeResponse({"id": AGENT_ID, "message_ids": list(self.context)})
        if method == "PATCH" and path == f"/agents/{AGENT_ID}":
            self.current_rules = kwargs["json"].get("tool_rules", self.current_rules)
            self.context = kwargs["json"].get("message_ids", self.context)
            return FakeResponse({"id": AGENT_ID})
        if method == "POST" and path == f"/agents/{AGENT_ID}/messages":
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            rules_now = self.current_rules
            time.sleep(0.002)  # widen the window a concurrent rule change would hit
            self.rules_at_message.append(
                (rules_now, kwargs["json"]["messages"][-1]["content"])
            )
            self.inflight -= 1
            for message in [*kwargs["json"]["messages"], "reply"]:
                self.log.append(f"m{len(self.log)}")
                self.context.append(self.log[-1])
            if self.fail_messages:
                return FakeResponse({"detail": "boom"}, status_code=500)
            if self.core_edits:
                edit = self.core_edits.popleft()
                if edit is not None:
                    self.blocks[1]["value"] = edit
            return FakeResponse(
                self.replies.popleft() if self.replies else {"messages": []}
            )
        if method == "GET" and path == f"/agents/{AGENT_ID}/core-memory/blocks":
            return FakeResponse([dict(block) for block in self.blocks])
        if method == "DELETE" and path == f"/agents/{AGENT_ID}":
            if self.fail_agent_delete:
                return FakeResponse({"detail": "delete failed"}, status_code=500)
            return FakeResponse(None)
        if method == "POST" and path == "/archives/":
            return FakeResponse({"id": ARCHIVE_ID})
        if (
            method == "PATCH"
            and path == f"/agents/{AGENT_ID}/archives/attach/{ARCHIVE_ID}"
        ):
            if self.fail_attach:
                return FakeResponse({"detail": "attach failed"}, status_code=500)
            return FakeResponse(None)
        if method == "POST" and path == f"/archives/{ARCHIVE_ID}/passages/batch":
            if self.fail_inserts:
                return FakeResponse({"detail": "embedding failed"}, status_code=500)
            passages = kwargs["json"]["passages"]
            self.inserted.append(passages)
            return FakeResponse(
                [{"id": f"p{i}", **passage} for i, passage in enumerate(passages)]
            )
        if method == "DELETE" and path == f"/archives/{ARCHIVE_ID}":
            return FakeResponse(None)
        raise AssertionError(f"unexpected request {method} {path}")

    # Letta's message manager, as the plugin's _in_letta reaches it.
    async def list_messages(
        self, actor: Any, agent_id: str, after: str | None = None, **_: Any
    ) -> list[SimpleNamespace]:
        ids = self.log[self.log.index(after) + 1 :] if after else self.log[-1:]
        return [SimpleNamespace(id=message_id) for message_id in ids]

    async def delete_messages_by_ids_async(
        self, message_ids: list[str], actor: Any
    ) -> None:
        self.log = [m for m in self.log if m not in message_ids]

    def requests_to(self, method: str, path: str) -> list[dict[str, Any]]:
        return [kwargs for m, p, kwargs in self.calls if m == method and p == path]

    def sent_messages(self) -> list[list[dict[str, str]]]:
        return [
            kwargs["json"]["messages"]
            for kwargs in self.requests_to("POST", f"/agents/{AGENT_ID}/messages")
        ]

    def tool_rules(self) -> list[list[dict[str, str]]]:
        return [
            kwargs["json"]["tool_rules"]
            for kwargs in self.requests_to("PATCH", f"/agents/{AGENT_ID}")
            if "tool_rules" in kwargs["json"]
        ]


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_type": "tool_call_message",
        "tool_call": {
            "name": name,
            "arguments": json.dumps(arguments),
            "tool_call_id": call_id,
        },
    }


def tool_return(call_id: str, value: Any) -> dict[str, Any]:
    return {
        "message_type": "tool_return_message",
        "tool_call_id": call_id,
        "tool_return": value,
    }


def reply(text: str) -> dict[str, Any]:
    return {"message_type": "assistant_message", "content": text}


def rules(*tools: str) -> list[dict[str, str]]:
    return [{"type": "run_first", "tool_name": tool} for tool in tools]


@pytest.fixture()
def fake_letta(monkeypatch: pytest.MonkeyPatch) -> FakeLetta:
    fake = FakeLetta()
    monkeypatch.setattr(letta_plugin, "_letta_client", lambda pg_uri_env: fake)
    monkeypatch.setattr(
        LettaMemorySystem, "_in_letta", lambda self, work: asyncio.run(work(fake, None))
    )
    return fake


@pytest.fixture()
def config() -> LettaMemoryConfig:
    return LettaMemoryConfig(
        model="openai/gpt-4o-mini",
        embedding="openai/text-embedding-3-small",
        top_k=2,
    )


@pytest.fixture()
def events() -> list[Any]:
    return []


@pytest.fixture()
def memory(
    fake_letta: FakeLetta, config: LettaMemoryConfig, events: list[Any], tmp_path
) -> LettaMemorySystem:
    return LettaMemorySystem(
        config=config,
        storage_dir=tmp_path,
        event_sink=events,
        run_context={
            "scenario_id": "locomo/conv 1",
            "combo_key": "a:b",
            "sample_index": 0,
        },
    )


# ---- configuration and plugin descriptor ----------------------------------------------


@pytest.fixture()
def agent_memory(
    fake_letta: FakeLetta, config: LettaMemoryConfig, events: list[Any], tmp_path: Path
) -> LettaMemorySystem:
    return LettaMemorySystem(
        config=config.model_copy(update={"prime_mode": "agent"}),
        storage_dir=tmp_path,
        event_sink=events,
    )


def test_config_requires_model_and_embedding() -> None:
    with pytest.raises(ValidationError) as error:
        LettaMemoryConfig.model_validate({})
    assert {tuple(item["loc"]) for item in error.value.errors()} == {
        ("model",),
        ("embedding",),
    }


def test_config_defaults_follow_the_agreed_profile(config: LettaMemoryConfig) -> None:
    assert config.type == "letta"
    assert config.write_tools == [
        "core_memory_append",
        "core_memory_replace",
        "archival_memory_insert",
    ]
    assert config.search_tool == "archival_memory_search"
    assert config.block_limit == 2000
    assert config.context_window == 16000
    assert config.temperature == 0.7
    assert config.human == ""
    assert config.pg_uri_env == "LETTA_PG_URI"
    assert config.prime_mode == "insert"
    assert config.max_output_tokens == 4096
    assert config.passage_max_tokens == 1024


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LettaMemoryConfig(
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            unknown=1,
        )


def test_plugin_is_discovered_as_letta() -> None:
    assert plugin.type_name == "letta"
    assert plugin.config_model is LettaMemoryConfig
    assert "letta" in discover_memory_plugins().names()


# ---- agent creation and teardown ------------------------------------------------------


def test_creates_one_memgpt_agent_with_the_papers_tools(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    (create,) = fake_letta.requests_to("POST", "/agents/")
    body = create["json"]
    assert body["agent_type"] == "memgpt_agent"
    assert body["include_base_tools"] is False
    assert body["tools"] == letta_plugin.MEMGPT_TOOLS
    assert body["tool_rules"] == rules(
        "core_memory_append", "core_memory_replace", "archival_memory_insert"
    )
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["embedding"] == "openai/text-embedding-3-small"
    assert body["model_settings"] == {
        "provider_type": "openai",
        "temperature": 0.7,
        "max_output_tokens": 4096,
    }
    assert body["context_window_limit"] == 16000
    assert body["compaction_settings"] == {"model": "openai/gpt-4o-mini"}
    assert body["memory_blocks"] == [
        {
            "label": "persona",
            "value": letta_plugin.LETTA_DEFAULT_PERSONA,
            "limit": 2000,
        },
        {"label": "human", "value": "", "limit": 2000},
    ]
    assert body["enable_sleeptime"] is False
    assert body["initial_message_sequence"] == []


def test_reasoning_effort_is_sent_in_model_settings(
    fake_letta: FakeLetta, tmp_path
) -> None:
    config = LettaMemoryConfig(
        model="openai/gpt-5.6-luna",
        embedding="openai/text-embedding-3-small",
        write_tools=["archival_memory_insert"],
        reasoning_effort="none",
        max_output_tokens=512,
    )
    LettaMemorySystem(config=config, storage_dir=tmp_path)
    (create,) = fake_letta.requests_to("POST", "/agents/")
    body = create["json"]
    assert body["model_settings"]["reasoning"] == {"reasoning_effort": "none"}
    assert body["model_settings"]["max_output_tokens"] == 512
    assert body["compaction_settings"] == {"model": "openai/gpt-5.6-luna"}


def test_agent_name_is_sanitised_for_letta(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    name = fake_letta.requests_to("POST", "/agents/")[0]["json"]["name"]
    assert name.startswith("heams-locomo-conv 1-a-b-0-")
    assert "/" not in name and ":" not in name


def test_refuses_a_different_letta_version(
    monkeypatch: pytest.MonkeyPatch, config: LettaMemoryConfig, tmp_path
) -> None:
    fake = FakeLetta(version="0.32.16")
    monkeypatch.setattr(letta_plugin, "_letta_client", lambda pg_uri_env: fake)
    with pytest.raises(RuntimeError, match="letta==0.16.8"):
        LettaMemorySystem(config=config, storage_dir=tmp_path)
    assert fake.requests_to("POST", "/agents/") == []


def test_close_deletes_the_agent_once(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    memory.close()
    memory.close()
    assert len(fake_letta.requests_to("DELETE", f"/agents/{AGENT_ID}")) == 1


def test_a_second_database_in_one_process_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = object()
    monkeypatch.setattr(letta_plugin, "_client", running)
    monkeypatch.setattr(letta_plugin, "_client_uri", "postgresql://localhost/letta")
    monkeypatch.setenv("LETTA_SAME", "postgresql://localhost/letta")
    monkeypatch.setenv("LETTA_OTHER", "postgresql://localhost/other")
    assert letta_plugin._letta_client("LETTA_SAME") is running
    with pytest.raises(RuntimeError, match="LETTA_OTHER must hold the same URI"):
        letta_plugin._letta_client("LETTA_OTHER")


def test_concurrent_operations_each_run_under_their_own_rule(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    operations = [
        (lambda i=i: memory.recall(f"question {i}"))
        if i % 2
        else (lambda i=i: memory.retain(f"fact {i}"))
        for i in range(12)
    ]
    threads = [threading.Thread(target=operation) for operation in operations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert fake_letta.max_inflight == 1
    assert len(fake_letta.rules_at_message) == 12
    for rules_now, content in fake_letta.rules_at_message:
        expected = (
            letta_plugin._search_only("archival_memory_search")
            if content.startswith("question")
            else rules(
                "core_memory_append", "core_memory_replace", "archival_memory_insert"
            )
        )
        assert rules_now == expected, content


def test_an_unset_database_uri_starts_a_private_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Started(Exception):
        pass

    def start() -> str:
        raise Started

    monkeypatch.setattr(letta_plugin, "_client", None)
    monkeypatch.setattr(letta_plugin, "local_database_uri", start)
    monkeypatch.setenv("LETTA_TEST_PG_URI", "")
    with pytest.raises(Started):
        letta_plugin._letta_client("LETTA_TEST_PG_URI")


def test_an_unset_database_uri_reuses_the_running_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = object()
    monkeypatch.setattr(letta_plugin, "_client", running)
    monkeypatch.setattr(letta_plugin, "_client_uri", "postgresql://localhost/private")
    monkeypatch.delenv("LETTA_TEST_PG_URI", raising=False)
    assert letta_plugin._letta_client("LETTA_TEST_PG_URI") is running


# ---- priming ---------------------------------------------------------------------------


def turn(
    content: str, group: str, order: int, role: str, **metadata: Any
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        metadata={
            "ingest_group_id": group,
            "ingest_order": order,
            "ingest_role": role,
            **metadata,
        },
    )


def test_primes_one_request_per_group_in_order_with_roles_and_speakers(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    agent_memory.prime_memory_entries(
        [
            turn("How are you?", "s1", 1, "assistant", ingest_speaker_name="Bob"),
            turn(
                "I moved to Toronto.",
                "s1",
                0,
                "user",
                ingest_speaker_name="Alice",
                timestamp="8 May, 2023",
            ),
            turn("Nice!", "s2", 0, "user", ingest_speaker_name="Alice"),
        ]
    )
    first, second = fake_letta.sent_messages()
    assert first == [
        {"role": "system", "content": "Conversation on 8 May, 2023."},
        {"role": "user", "content": "Alice: I moved to Toronto."},
        {"role": "assistant", "content": "Bob: How are you?"},
        {"role": "system", "content": "The conversation has ended."},
    ]
    assert second == [
        {"role": "user", "content": "Alice: Nice!"},
        {"role": "system", "content": "The conversation has ended."},
    ]


def test_priming_keeps_the_write_rule_set_at_creation(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    agent_memory.prime_memory_entries([turn("hi", "s1", 0, "user")])
    assert (
        fake_letta.tool_rules() == []
    )  # created with the write rule; nothing to change


def test_agent_priming_restores_write_rules_without_creating_an_archive(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    agent_memory.recall("switch to search rules")
    agent_memory.prime_memory_entries([turn("hi", "s1", 0, "user")])
    assert fake_letta.tool_rules()[-1] == rules(*letta_plugin.WRITE_TOOLS)
    assert fake_letta.requests_to("POST", "/archives/") == []


def test_entries_without_a_group_are_primed_one_by_one(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    agent_memory.prime_memory_entries(
        [
            MemoryEntry(
                content="first", metadata={"ingest_timestamp_epoch": 1717000000}
            ),
            MemoryEntry(content="second", metadata={"role": "narrator"}),
            MemoryEntry(content="   ", metadata={}),
        ]
    )
    sent = fake_letta.sent_messages()
    assert len(sent) == 3
    assert sent[0][0] == {"role": "system", "content": "Conversation on 2024-05-29."}
    assert sent[1][0] == {
        "role": "user",
        "content": "second",
    }  # unknown roles enter as user
    assert sent[2] == [
        {"role": "system", "content": "The conversation has ended."}
    ]  # blank turn dropped


def test_priming_emits_one_retain_event_per_group_with_the_agents_writes(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    fake_letta.replies.append(
        {
            "messages": [
                tool_call(
                    "c1",
                    "core_memory_replace",
                    {
                        "label": "human",
                        "old_content": "Montreal",
                        "new_content": "Toronto",
                    },
                ),
                tool_return("c1", "ok"),
                tool_call("c2", "send_message", {"message": "Noted!"}),
                tool_return("c2", "sent"),
            ],
            "stop_reason": {"stop_reason": "end_turn"},
        }
    )
    agent_memory.prime_memory_entries([turn("I moved to Toronto.", "s1", 0, "user")])
    (event,) = events
    assert event.operation_type == "retain"
    assert event.status == "success"
    assert event.payload["is_priming"] is True
    assert event.payload["entry_count"] == 1
    assert event.result["write_count"] == 1
    assert [call["tool"] for call in event.result["memory_calls"]] == [
        "core_memory_replace"
    ]
    assert event.result["stop_reason"] == "end_turn"


def test_priming_failure_emits_an_error_event_and_raises(
    agent_memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    fake_letta.fail_messages = True
    with pytest.raises(RuntimeError, match="HTTP 500.*boom"):
        agent_memory.prime_memory_entries([turn("hi", "s1", 0, "user")])
    (event,) = events
    assert event.status == "error"
    assert event.result["write_count"] == 0


# ---- insert-mode priming ---------------------------------------------------------------


def test_insert_mode_writes_groups_without_messaging_the_agent(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    before = (list(fake_letta.log), list(fake_letta.context))
    memory.prime_memory_entries(
        [
            turn("How are you?", "s1", 1, "assistant", ingest_speaker_name="Bob"),
            turn(
                "I moved to Toronto.",
                "s1",
                0,
                "user",
                ingest_speaker_name="Alice",
                timestamp="8 May, 2023",
                ingest_timestamp_epoch=1683504000,
            ),
            turn("Nice!", "s2", 0, "user"),
            turn("   ", "s3", 0, "user"),
        ]
    )
    assert fake_letta.sent_messages() == []
    assert (fake_letta.log, fake_letta.context) == before
    (create,) = fake_letta.requests_to("POST", "/archives/")
    assert create["json"] == {
        "name": f"{AGENT_ID}-archival",
        "embedding": "openai/text-embedding-3-small",
    }
    attach = f"/agents/{AGENT_ID}/archives/attach/{ARCHIVE_ID}"
    assert len(fake_letta.requests_to("PATCH", attach)) == 1
    assert fake_letta.inserted == [
        [
            {
                "text": "Conversation on 8 May, 2023.\nAlice: I moved to Toronto.\n"
                "Bob: How are you?",
                "created_at": "2023-05-08T00:00:00+00:00",
            }
        ],
        [{"text": "user: Nice!"}],
    ]
    assert [event.result["write_count"] for event in events] == [1, 1, 0]
    assert all(event.payload["is_priming"] for event in events)
    assert all(event.operation_type == "retain" for event in events)


def test_insert_mode_splits_long_groups_and_preserves_content(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    content = "remember this detail number 42 please. " * 3000
    memory.prime_memory_entries(
        [turn(content, "s1", 0, "user", timestamp="8 May, 2023")]
    )
    (passages,) = fake_letta.inserted
    encoding = tiktoken.get_encoding("cl100k_base")
    header = "Conversation on 8 May, 2023.\n"
    assert len(passages) > 1
    assert all(p["text"].startswith(header) for p in passages)
    assert all(len(encoding.encode_ordinary(p["text"])) <= 1024 for p in passages)
    assert (
        "".join(p["text"].removeprefix(header) for p in passages) == f"user: {content}"
    )


@pytest.mark.parametrize(
    ("header", "body"),
    [
        ("Conversation on today.", "\u4f60\u597d\U0001f642 calf\u00e9 " * 30),
        ("", "<|endoftext|>" * 30),
    ],
)
def test_passage_splitting_preserves_unicode_and_token_caps(
    header: str, body: str
) -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    prefix = f"{header}\n" if header else ""
    passages = letta_plugin._split_passages(header, body, 16)
    assert "".join(p.removeprefix(prefix) for p in passages) == body
    assert all(len(encoding.encode_ordinary(p)) <= 16 for p in passages)


@pytest.mark.parametrize(
    ("header", "body", "limit"),
    [("Conversation on today.", "hello", 1), ("", "\U0001f642", 1)],
)
def test_passage_splitting_rejects_impossible_budgets(
    header: str, body: str, limit: int
) -> None:
    with pytest.raises(ValueError, match="passage_max_tokens"):
        letta_plugin._split_passages(header, body, limit)


@pytest.mark.parametrize("failure", ["fail_inserts", "fail_attach"])
def test_insert_failure_emits_an_error_and_keeps_archive_for_cleanup(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any], failure: str
) -> None:
    setattr(fake_letta, failure, True)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        memory.prime_memory_entries([turn("hi", "s1", 0, "user")])
    (event,) = events
    assert event.status == "error"
    assert event.result["write_count"] == 0
    memory.close()
    assert len(fake_letta.requests_to("DELETE", f"/archives/{ARCHIVE_ID}")) == 1


@pytest.mark.parametrize("fail_agent_delete", [False, True])
def test_close_deletes_archive_once_even_if_agent_cleanup_fails(
    memory: LettaMemorySystem, fake_letta: FakeLetta, fail_agent_delete: bool, caplog
) -> None:
    memory.prime_memory_entries([turn("hi", "s1", 0, "user")])
    memory.prime_memory_entries([turn("again", "s2", 0, "user")])
    fake_letta.fail_agent_delete = fail_agent_delete
    memory.close()
    memory.close()
    assert len(fake_letta.requests_to("POST", "/archives/")) == 1
    assert len(fake_letta.requests_to("DELETE", f"/archives/{ARCHIVE_ID}")) == 1
    assert len(fake_letta.requests_to("DELETE", f"/agents/{AGENT_ID}")) == 1
    if fail_agent_delete:
        assert "Could not delete Letta /agents/" in caplog.text


# ---- core memory limits ----------------------------------------------------------------


@pytest.fixture()
def core_tools(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    from letta.schemas.block import Block
    from letta.schemas.memory import Memory
    from letta.services.tool_executor.core_tool_executor import LettaCoreToolExecutor

    for name in ("core_memory_append", "core_memory_replace"):
        monkeypatch.setattr(
            LettaCoreToolExecutor, name, getattr(LettaCoreToolExecutor, name)
        )
    letta_plugin._enforce_core_memory_limits()

    async def update_memory_if_changed_async(**_: Any) -> None:
        return None

    executor = object.__new__(LettaCoreToolExecutor)
    executor.agent_manager = SimpleNamespace(
        update_memory_if_changed_async=update_memory_if_changed_async
    )
    state = SimpleNamespace(
        id=AGENT_ID,
        memory=Memory(
            blocks=[Block(label="human", value="Alice lives in Paris", limit=30)]
        ),
    )
    return executor, state


def test_core_memory_append_refuses_overflow_without_mutation(
    core_tools: tuple[Any, Any],
) -> None:
    executor, state = core_tools
    with pytest.raises(
        ValueError, match=r"Exceeds 30 character limit \(requested 33\)"
    ):
        asyncio.run(executor.core_memory_append(state, None, "human", "x" * 12))
    assert state.memory.get_block("human").value == "Alice lives in Paris"
    asyncio.run(executor.core_memory_append(state, None, "human", "x" * 9))
    assert len(state.memory.get_block("human").value) == 30


def test_core_memory_replace_refuses_empty_old_content_and_overflow(
    core_tools: tuple[Any, Any],
) -> None:
    executor, state = core_tools
    with pytest.raises(ValueError, match="old_content must not be empty"):
        asyncio.run(executor.core_memory_replace(state, None, "human", "", "Rome"))
    with pytest.raises(ValueError, match="Exceeds 30 character limit"):
        asyncio.run(
            executor.core_memory_replace(
                state, None, "human", "Paris", "Paris, Rome and Oslo"
            )
        )
    assert state.memory.get_block("human").value == "Alice lives in Paris"
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(executor.core_memory_replace(state, None, "human", "Oslo", "Rome"))
    asyncio.run(executor.core_memory_replace(state, None, "human", "Paris", "Rome"))
    assert state.memory.get_block("human").value == "Alice lives in Rome"


def test_core_memory_guards_preserve_read_only_blocks(
    core_tools: tuple[Any, Any],
) -> None:
    executor, state = core_tools
    state.memory.get_block("human").read_only = True
    with pytest.raises(ValueError):
        asyncio.run(executor.core_memory_append(state, None, "human", "Tea"))
    with pytest.raises(ValueError):
        asyncio.run(executor.core_memory_replace(state, None, "human", "Paris", "Rome"))
    assert state.memory.get_block("human").value == "Alice lives in Paris"


# ---- retain ----------------------------------------------------------------------------


def test_retain_sends_the_content_as_a_user_message_under_the_write_rule(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    memory.recall("switch the rules first")
    fake_letta.replies.append(
        {
            "messages": [
                tool_call(
                    "c1", "archival_memory_insert", {"content": "User likes tea."}
                ),
                tool_return("c1", "ok"),
            ]
        }
    )
    message = memory.retain("I like tea.")
    assert fake_letta.sent_messages()[-1] == [
        {"role": "user", "content": "I like tea."}
    ]
    assert fake_letta.tool_rules()[-1] == rules(
        "core_memory_append", "core_memory_replace", "archival_memory_insert"
    )
    assert message == "Letta made 1 memory write(s)."
    assert events[-1].result["write_count"] == 1


# ---- recall ----------------------------------------------------------------------------


def test_recall_returns_core_then_top_k_hits_from_the_agents_searches(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    fake_letta.blocks[1]["value"] = "Lives in Toronto."
    archival = [
        {
            "id": "p1",
            "timestamp": "t1",
            "content": "User moved to Toronto.",
            "tags": ["home"],
        },
        {"id": "p2", "timestamp": "t2", "content": "User lived in Montreal."},
        {"id": "p3", "timestamp": "t3", "content": "User likes tea."},
    ]
    fake_letta.replies.append(
        {
            "messages": [
                tool_call(
                    "c1",
                    "archival_memory_search",
                    {"query": "where does the user live"},
                ),
                tool_return(
                    "c1", repr(archival)
                ),  # archival returns a bare Python-repr list
                tool_call("c2", "conversation_search", {"query": "Toronto"}),
                tool_return(
                    "c2",
                    json.dumps(
                        {
                            "status": "OK",
                            "message": {
                                "message": "x",
                                "results": [
                                    {
                                        "role": "user",
                                        "content": "I moved to Toronto.",
                                        "timestamp": "t0",
                                    }
                                ],
                            },
                        }
                    ),
                ),
                reply("You live in Toronto."),
            ]
        }
    )
    results = memory.recall("Where do I live?")

    assert fake_letta.tool_rules() == [
        letta_plugin._search_only("archival_memory_search")
    ]
    assert fake_letta.sent_messages()[-1] == [
        {"role": "user", "content": "Where do I live?"}
    ]
    assert [json.loads(item)["metadata"]["source"] for item in results] == [
        "core",
        "archival",
        "archival",
    ]
    assert (
        json.loads(results[0])["content"]
        == "persona: I am Sam.\nhuman: Lives in Toronto."
    )
    assert json.loads(results[1]) == {
        "content": "User moved to Toronto.",
        "metadata": {"source": "archival", "tags": ["home"], "timestamp": "t1"},
    }

    event = events[-1]
    assert event.operation_type == "recall"
    assert event.result["hit_count"] == 3  # archival only, before the top_k=2 cut
    assert event.result["returned_count"] == 3  # core + top_k
    assert event.result["letta_reply"] == "You live in Toronto."
    assert [(s["tool"], s["result_count"]) for s in event.result["searches"]] == [
        ("archival_memory_search", 3),
        ("conversation_search", 1),
    ]


def test_recall_keeps_duplicate_passages_once(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    passage = [{"id": "p1", "content": "User moved to Toronto."}]
    fake_letta.replies.append(
        {
            "messages": [
                tool_call("c1", "archival_memory_search", {"query": "a"}),
                tool_return("c1", repr(passage)),
                tool_call("c2", "archival_memory_search", {"query": "b"}),
                tool_return("c2", repr(passage)),
            ]
        }
    )
    results = memory.recall("Where do I live?", top_k=5)
    assert len(results) == 2  # core + one passage


def test_recall_may_only_search_then_answer() -> None:
    children = ["archival_memory_search", "conversation_search", "send_message"]
    assert letta_plugin._search_only("archival_memory_search") == [
        {"type": "run_first", "tool_name": "archival_memory_search"},
        {
            "type": "constrain_child_tools",
            "tool_name": "archival_memory_search",
            "children": children,
        },
        {
            "type": "constrain_child_tools",
            "tool_name": "conversation_search",
            "children": children,
        },
        {"type": "exit_loop", "tool_name": "send_message"},
    ]


def test_recall_and_delete_leave_no_messages_behind(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    memory.prime_memory_entries([turn("I moved to Toronto.", "s1", 0, "user")])
    primed_log, primed_context = list(fake_letta.log), list(fake_letta.context)
    fake_letta.core_edits.extend([None, None, "forgotten"])

    memory.recall("Where do I live?")
    memory.recall("Where did I move?")
    memory.delete("where I live")

    assert fake_letta.log == primed_log
    assert fake_letta.context == primed_context
    assert fake_letta.blocks[1]["value"] == "forgotten"  # delete's edit stays


def test_empty_recall_query_returns_core_memory_without_messaging_the_agent(
    memory: LettaMemorySystem, fake_letta: FakeLetta
) -> None:
    results = memory.recall("  ")
    assert fake_letta.sent_messages() == []
    assert [json.loads(item)["metadata"]["source"] for item in results] == ["core"]


# ---- delete ----------------------------------------------------------------------------


def test_delete_is_a_plain_request_and_succeeds_when_core_memory_changes(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    fake_letta.blocks[1]["value"] = "Lives in Toronto."
    fake_letta.core_edits.append("")
    fake_letta.replies.append(
        {
            "messages": [
                tool_call("c1", "core_memory_replace", {"label": "human"}),
                tool_return("c1", "ok"),
                reply("Done."),
            ]
        }
    )
    memory.delete("where I live")
    assert fake_letta.tool_rules()[-1] == []  # no rule: the agent decides
    assert fake_letta.sent_messages()[-1] == [
        {"role": "user", "content": "Please forget this: where I live"}
    ]
    event = events[-1]
    assert event.operation_type == "delete"
    assert event.status == "success"
    assert event.result["deleted_count"] == 1
    assert event.result["core_changed"] is True
    assert event.result["letta_reply"] == "Done."


def test_delete_is_skipped_when_core_memory_is_unchanged(
    memory: LettaMemorySystem, fake_letta: FakeLetta, events: list[Any]
) -> None:
    memory.delete("my archived trip")
    event = events[-1]
    assert event.status == "skipped"
    assert event.result["deleted_count"] == 0
    assert event.result["core_changed"] is False


def test_consolidate_is_not_supported(
    memory: LettaMemorySystem, events: list[Any]
) -> None:
    memory.consolidate()
    assert events[-1].status == "skipped"


# ---- parsing Letta's responses ---------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_return_value", "expected"),
    [
        ("[{'id': 'p1', 'content': 'a'}]", [{"id": "p1", "content": "a"}]),
        (json.dumps([{"content": "a"}]), [{"content": "a"}]),
        (
            json.dumps({"status": "OK", "message": {"results": [{"content": "a"}]}}),
            [{"content": "a"}],
        ),
        (
            json.dumps({"status": "OK", "message": "[{'content': 'a'}]"}),
            [{"content": "a"}],
        ),
        (json.dumps({"status": "Failed", "message": ""}), []),
        ("No results found.", []),
        (None, []),
    ],
)
def test_search_results_accept_every_return_shape(
    tool_return_value: Any, expected: list[dict[str, Any]]
) -> None:
    assert letta_plugin._search_results(tool_return_value) == expected


def test_tool_events_pair_calls_with_returns_including_tool_calls_lists() -> None:
    response = {
        "messages": [
            {
                "message_type": "tool_call_message",
                "tool_calls": [{"name": "a", "arguments": "{}", "tool_call_id": "1"}],
            },
            tool_return("1", "r1"),
            tool_return("unknown", "r2"),
        ]
    }
    assert letta_plugin._tool_events(response) == [
        ("a", "{}", "r1"),
        (None, None, "r2"),
    ]


def test_repair_closes_a_tool_call_cut_off_on_a_newline_run() -> None:
    # Ends on the backslash, which is where Letta's closer leaves the quote escaped.
    raw = '{"thinking":"stored it","message":"Hello\\n\\n' + "  \\n" * 30 + "\\"
    repaired = json.loads(letta_plugin._repair_tool_arguments(raw))
    assert repaired["thinking"] == "stored it"
    assert repaired["message"].count("\n") == 2
    assert repaired["message"].startswith("Hello")
    assert len(repaired["message"]) < 40


def test_repair_leaves_valid_tool_arguments() -> None:
    raw = '{"message":"ok"}'
    assert letta_plugin._repair_tool_arguments(raw) == raw


def test_repair_replaces_lettas_truncated_json_closer() -> None:
    from letta.llm_api.llm_client_base import LLMClientBase

    original = LLMClientBase._fix_truncated_json_response
    raw = '{"message":"Hello\\n\\n' + "\\n" * 30 + "\\"
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(arguments=raw))
                    ]
                )
            )
        ]
    )
    try:
        letta_plugin._repair_truncated_tool_arguments()
        from letta.llm_api.openai_client import OpenAIClient

        assert LLMClientBase._fix_truncated_json_response is not original
        assert (
            OpenAIClient._fix_truncated_json_response
            is LLMClientBase._fix_truncated_json_response
        )
        OpenAIClient._fix_truncated_json_response(None, response)
    finally:
        LLMClientBase._fix_truncated_json_response = original
    repaired = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    assert repaired["message"].startswith("Hello")
    assert repaired["message"].count("\n") == 2


def test_repair_encodes_a_tab_and_drops_a_carriage_return() -> None:
    tabbed = json.loads(
        letta_plugin._repair_tool_arguments('{"message":"Hello\n\tcontinued')
    )
    assert tabbed["message"] == "Hello\n\tcontinued"
    returned = json.loads(
        letta_plugin._repair_tool_arguments('{"message":"Hello\n\rcontinued')
    )
    assert returned["message"] == "Hello\ncontinued"
    runaway = '{"message":"Hello\n\n' + "\n\t \r" * 10 + "tail"
    collapsed = json.loads(letta_plugin._repair_tool_arguments(runaway))
    assert collapsed["message"] == "Hello\n\ntail"


def test_reply_joins_text_and_content_parts() -> None:
    response = {
        "messages": [
            reply("Hi"),
            {"message_type": "assistant_message", "content": [{"text": "there"}]},
        ]
    }
    assert letta_plugin._reply(response) == "Hi\nthere"
    assert letta_plugin._reply({"messages": []}) is None
