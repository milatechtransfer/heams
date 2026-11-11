"""Model usage that memory systems report to the framework.

The contract is one method, `BaseMemorySystem.model_usage()`: it returns the
running totals of every LLM and embedding call the system has made since it was
created, or `None` when the system does not report usage. How a system counts
is its own business. The framework reads the totals each time it logs an event
for the system and records the difference on that event.

`UsageCounter` is an optional helper for keeping those totals.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from heams.memory_systems.schemas import ModelCallKind, ModelUsage

_UsageKey = tuple[ModelCallKind, str | None, bool]


def _key(record: ModelUsage) -> _UsageKey:
    return (record.kind, record.model, record.local)


class UsageCounter:
    """Thread-safe running totals of model calls, keyed by (kind, model, local).

    Typical use in a memory system::

        self._usage = UsageCounter()
        ...
        response = any_llm.completion(model=model, ...)
        self._usage.add_completion(response, model)
        ...
        def model_usage(self):
            return self._usage.totals()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[_UsageKey, ModelUsage] = {}

    def add(
        self,
        kind: ModelCallKind | str,
        model: str | None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_count: int = 1,
        local: bool = False,
        cost: float | None = None,
    ) -> None:
        key = (ModelCallKind(kind), model, local)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = ModelUsage(kind=key[0], model=model, local=local, call_count=0)
                self._records[key] = record
            record.call_count += max(int(call_count), 0)
            record.input_tokens += max(int(input_tokens), 0)
            record.output_tokens += max(int(output_tokens), 0)
            if cost is not None:
                record.cost = (record.cost or 0.0) + max(float(cost), 0.0)

    def add_completion(
        self, response: Any, model: str | None = None, *, local: bool = False
    ) -> None:
        """Count one chat completion from an OpenAI-shaped response.

        Pass `model` as the id the system called, so it is priced as
        configured. A response without usage still counts as a call.
        """

        usage = _get(response, "usage")
        self.add(
            ModelCallKind.LLM,
            model or _model_of(response),
            input_tokens=_tokens(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_tokens(usage, "completion_tokens", "output_tokens"),
            local=local,
        )

    def add_embedding(
        self, response: Any, model: str | None = None, *, local: bool = False
    ) -> None:
        """Count one embedding request from an OpenAI-shaped response."""

        usage = _get(response, "usage")
        self.add(
            ModelCallKind.EMBEDDING,
            model or _model_of(response),
            input_tokens=_tokens(
                usage, "prompt_tokens", "total_tokens", "input_tokens"
            ),
            local=local,
        )

    def totals(self) -> list[ModelUsage]:
        with self._lock:
            return [record.model_copy() for record in self._records.values()]


def usage_difference(
    before: Iterable[ModelUsage], after: Iterable[ModelUsage]
) -> list[ModelUsage]:
    """Usage added between two running totals; records that did not grow are dropped."""

    previous = {_key(record): record for record in before}
    difference: list[ModelUsage] = []
    for record in after:
        old = previous.get(_key(record))
        if old is None:
            grown = record.model_copy()
        else:
            grown = record.model_copy(
                update={
                    "call_count": max(record.call_count - old.call_count, 0),
                    "input_tokens": max(record.input_tokens - old.input_tokens, 0),
                    "output_tokens": max(record.output_tokens - old.output_tokens, 0),
                    "cost": (
                        max(record.cost - (old.cost or 0.0), 0.0)
                        if record.cost is not None
                        else None
                    ),
                }
            )
        if grown.call_count or grown.input_tokens or grown.output_tokens or grown.cost:
            difference.append(grown)
    return difference


def _get(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _tokens(usage: Any, *names: str) -> int:
    for name in names:
        value = _get(usage, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def _model_of(response: Any) -> str | None:
    model = _get(response, "model")
    return model if isinstance(model, str) and model else None


__all__ = ["UsageCounter", "usage_difference"]
