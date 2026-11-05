"""Memory systems report model usage through `model_usage()`."""

from pathlib import Path
from threading import Barrier, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEntry,
    MemoryOperationEvent,
    MemoryOperationResult,
    ModelCallKind,
    ModelUsage,
    UsageCounter,
)
from heams.memory_systems.in_context import InContextConfig, InContextMemorySystem
from heams.memory_systems.no_memory import NoMemoryConfig, NoMemoryMemorySystem

LLM = "openai:gpt-4o-mini"
EMBED = "openai:text-embedding-3-small"


def _completion(prompt: int, completion: int) -> Any:
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _embedding(tokens: int) -> Any:
    return {"model": "text-embedding-3-small", "usage": {"prompt_tokens": tokens}}


class CountingMemory(BaseMemorySystem):
    """Keeps its totals with a `UsageCounter`, as a plugin would."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            BaseMemoryConfig(type="counting"), storage_dir=Path(), **kwargs
        )
        self.usage = UsageCounter()

    def model_usage(self) -> list[ModelUsage]:
        return self.usage.totals()

    def _retain_impl(self, content: str, metadata: dict) -> MemoryOperationResult:
        self.usage.add_completion(_completion(100, 20), LLM)
        self.usage.add_completion(_completion(50, 10), LLM)
        self.usage.add_embedding(_embedding(7), EMBED)
        if metadata.get("fail"):
            raise RuntimeError("backend down")
        return MemoryOperationResult(value="ok")

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        self.usage.add_embedding(_embedding(3), EMBED)
        return MemoryOperationResult(value=[query])


class SilentMemory(BaseMemorySystem):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(BaseMemoryConfig(type="silent"), storage_dir=Path(), **kwargs)

    def _retain_impl(self, content: str, metadata: dict) -> MemoryOperationResult:
        return MemoryOperationResult(value="ok")

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        return MemoryOperationResult(value=[])


def _usage(event: MemoryOperationEvent) -> dict:
    return {
        (record.kind, record.model): (
            record.call_count,
            record.input_tokens,
            record.output_tokens,
        )
        for record in event.usage or []
    }


def test_each_event_carries_the_usage_since_the_previous_one() -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = CountingMemory(event_sink=ledger)

    memory.retain("fact")
    memory.recall("query")
    memory.recall("query")

    retain, first_recall, second_recall = ledger
    assert _usage(retain) == {
        (ModelCallKind.LLM, LLM): (2, 150, 30),
        (ModelCallKind.EMBEDDING, EMBED): (1, 7, 0),
    }
    assert _usage(first_recall) == {(ModelCallKind.EMBEDDING, EMBED): (1, 3, 0)}
    assert _usage(second_recall) == _usage(first_recall)


def test_an_operation_without_calls_reports_zero() -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = CountingMemory(event_sink=ledger)

    memory.recall("query")
    memory.delete("fact")  # unsupported: makes no calls

    assert ledger[-1].usage == []


def test_a_failed_operation_keeps_the_usage_it_spent() -> None:
    ledger: list[MemoryOperationEvent] = []
    CountingMemory(event_sink=ledger).retain("fact", {"fail": True})

    (event,) = ledger
    assert event.status == "error"
    assert _usage(event)[(ModelCallKind.LLM, LLM)] == (2, 150, 30)


def test_calls_between_events_land_on_the_next_event() -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = CountingMemory(event_sink=ledger)

    memory.usage.add_embedding(_embedding(5), EMBED)  # e.g. warm-up at startup
    memory.recall("query")

    assert _usage(ledger[0]) == {(ModelCallKind.EMBEDDING, EMBED): (2, 8, 0)}


def test_events_a_plugin_logs_itself_carry_usage_too() -> None:
    class PrimingMemory(CountingMemory):
        def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
            self.usage.add_embedding(_embedding(11), EMBED)
            self._log_memory_event(
                operation_type="retain", status="success", source="runtime"
            )

    ledger: list[MemoryOperationEvent] = []
    PrimingMemory(event_sink=ledger).prime_memory_entries([])

    assert _usage(ledger[0]) == {(ModelCallKind.EMBEDDING, EMBED): (1, 11, 0)}


def test_concurrent_operations_never_count_a_call_twice() -> None:
    barrier = Barrier(2)

    class ConcurrentMemory(CountingMemory):
        def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
            self.usage.add_embedding(_embedding(3), EMBED)
            barrier.wait()  # both calls are made before either event is logged
            return MemoryOperationResult(value=[])

    ledger: list[MemoryOperationEvent] = []
    memory = ConcurrentMemory(event_sink=ledger)
    workers = [Thread(target=memory.recall, args=("query",)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    tokens = sum(record.input_tokens for event in ledger for record in event.usage)
    assert tokens == 6


def test_a_system_that_does_not_report_usage_is_not_reported_not_zero() -> None:
    ledger: list[MemoryOperationEvent] = []
    SilentMemory(event_sink=ledger).retain("fact")

    (event,) = ledger
    assert event.usage is None
    assert "usage" not in event.model_dump()


def test_a_failing_model_usage_is_not_reported() -> None:
    class BrokenMemory(SilentMemory):
        def model_usage(self) -> list[ModelUsage]:
            raise RuntimeError("counter unavailable")

    ledger: list[MemoryOperationEvent] = []
    BrokenMemory(event_sink=ledger).retain("fact")

    assert ledger[0].usage is None
    assert ledger[0].status == "success"


@pytest.mark.parametrize(
    "memory_factory",
    [
        lambda ledger: InContextMemorySystem(
            InContextConfig(), storage_dir=Path(), event_sink=ledger
        ),
        lambda ledger: NoMemoryMemorySystem(
            NoMemoryConfig(), storage_dir=Path(), event_sink=ledger
        ),
    ],
)
def test_built_in_systems_report_zero(memory_factory: Any) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = memory_factory(ledger)

    memory.retain("fact")
    memory.recall("fact")

    assert [event.usage for event in ledger] == [[], []]


def test_counter_reads_openai_shaped_responses_and_tracks_local_and_cost() -> None:
    counter = UsageCounter()
    counter.add_completion(_completion(10, 2))  # model from the response
    counter.add_completion(SimpleNamespace(), "bare")  # no usage: still a call
    counter.add("embedding", "st:qwen", input_tokens=40, local=True)
    counter.add("llm", "gateway-model", input_tokens=5, cost=0.25)
    counter.add("llm", "gateway-model", input_tokens=5, cost=0.25)

    totals = {(record.model, record.local): record for record in counter.totals()}
    assert totals[("gpt-4o-mini-2024-07-18", False)].input_tokens == 10
    assert totals[("bare", False)].call_count == 1
    assert totals[("st:qwen", True)].kind == ModelCallKind.EMBEDDING
    assert totals[("gateway-model", False)].cost == pytest.approx(0.5)


def test_usage_round_trips_through_the_ledger_json() -> None:
    ledger: list[MemoryOperationEvent] = []
    CountingMemory(event_sink=ledger).retain("fact")

    restored = MemoryOperationEvent.model_validate_json(ledger[0].model_dump_json())

    assert restored.usage == ledger[0].usage
