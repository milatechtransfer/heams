"""MilaMem's model calls, counted for usage reporting.

Every LLM and embedding call MilaMem makes goes through this module, so usage
is counted in one place and the call sites stay unaware of it. To report usage
the same way in another memory system:

1. Route its model calls through wrappers like `completion` and `embedding`.
2. Decorate its operation methods with `counted`, so calls made while an
   operation runs are added to that system's `UsageCounter`.
3. Return `self._usage.totals()` from `model_usage()`.

Work the system hands to its own threads must run in `contextvars.copy_context()`
so it still reaches the counter (see `design/consolidation/lifecycle.py`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from heams.memory_systems.api import UsageCounter

_active_counter: ContextVar[UsageCounter | None] = ContextVar(
    "milamem_usage_counter", default=None
)

_Method = TypeVar("_Method", bound=Callable[..., Any])


@contextmanager
def counting(counter: UsageCounter) -> Iterator[None]:
    """Count model calls made in this context on `counter`."""

    token = _active_counter.set(counter)
    try:
        yield
    finally:
        _active_counter.reset(token)


def counted(method: _Method) -> _Method:
    """Count the model calls of a method on its system's `_usage` counter."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with counting(self._usage):
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def completion(**kwargs: Any) -> Any:
    """`any_llm.completion`, counted."""

    import any_llm

    response = any_llm.completion(**kwargs)
    counter = _active_counter.get()
    if counter is not None:
        counter.add_completion(response, kwargs.get("model"))
    return response


def embedding(**kwargs: Any) -> Any:
    """`any_llm.embedding`, counted."""

    import any_llm

    response = any_llm.embedding(**kwargs)
    counter = _active_counter.get()
    if counter is not None:
        counter.add_embedding(response, kwargs.get("model"))
    return response


def is_counting() -> bool:
    """Whether calls made here are counted, to skip work done only for counting."""

    return _active_counter.get() is not None


def count_local_embedding(model: str, input_tokens: int) -> None:
    """Count an in-process embedding batch; local models are priced at $0."""

    counter = _active_counter.get()
    if counter is not None:
        counter.add("embedding", model, input_tokens=input_tokens, local=True)


__all__ = [
    "completion",
    "count_local_embedding",
    "counted",
    "counting",
    "embedding",
    "is_counting",
]
