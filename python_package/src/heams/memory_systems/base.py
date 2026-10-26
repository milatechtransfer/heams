from __future__ import annotations

import json
import time
from collections.abc import Callable, MutableSequence
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from heams.memory_systems.shared_types import MemoryEntry
    from heams.memory_systems.schemas import (
        MemoryOperationEvent,
        MemoryOperationSource,
        MemoryOperationStatus,
        MemoryOperationType,
    )


class MemoryMode(str, Enum):
    INJECT = "inject"
    TOOL = "tool"


class BaseConsolidationConfig(BaseModel):
    """Consolidation settings shared by all memory systems.

    Plugins may subclass this to add backend-specific consolidation knobs.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "When true, the benchmark calls consolidate() after priming and "
            "after each session."
        ),
    )


class BaseMemoryConfig(BaseModel):
    """
    Pure data validation config shared by all memory systems.
    Subclasses will inherit this and add their own specific fields (like file paths or API keys).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: str = Field(
        ...,
        description="The registry identifier for this memory system (e.g., 'milamem', 'mem0').",
    )
    name: str = Field(
        default="agent_memory", description="Display name for telemetry and logging."
    )
    top_k: int = Field(
        default=5, description="Default number of items to retrieve during a recall."
    )
    consolidation: BaseConsolidationConfig = Field(
        default_factory=BaseConsolidationConfig,
        description="Controls benchmark-triggered consolidate() calls.",
    )

    def with_runtime_storage(self, storage_dir: Path) -> Self:
        return self

    def check_retention_policies(self, policies: list[str]) -> None:
        """Fail before a run if this system can't enforce `policies` as configured.

        Called once per retention_policy experiment, before any scenario runs.
        A governed system raises `ValueError` for a configuration that would
        make the benchmark's scores ambiguous; the default does nothing, so a
        system without governance always passes and runs ungoverned.
        """


MemoryEventSink = Callable[["MemoryOperationEvent"], None]
MemoryEventLedger = MutableSequence["MemoryOperationEvent"]
MemoryEventTarget = MemoryEventSink | MemoryEventLedger


@dataclass(slots=True)
class MemoryOperationResult:
    value: Any
    status: "MemoryOperationStatus | str" = "success"
    result: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ==========================================
# 2. THE RUNTIME INTERFACE
# ==========================================


class BaseMemorySystem(ABC):
    """
    Abstract base class defining the runtime contract for all memory systems.
    Notice the public methods use simple primitive types (`str`, `list[str]`).
    This makes them incredibly easy to convert into LLM tools.
    """

    def __init__(
        self,
        config: BaseMemoryConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ):
        """Initializes the memory system with a pre-validated Pydantic config."""
        self.storage_dir = storage_dir
        self.config = config.with_runtime_storage(storage_dir)
        self._memory_event_sink = event_sink
        self._memory_event_run_context = dict(run_context or {})
        self._active_memory_event_run_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"memory_event_run_context_{type(self).__name__}_{id(self)}",
                default=None,
            )
        )

    def close(self) -> None:
        """Release resources owned by this memory system."""

    def attach_event_sink(self, event_sink: MemoryEventTarget | None) -> None:
        """Attach or clear the optional sink used for memory observability events."""

        self._memory_event_sink = event_sink

    def set_run_context(self, run_context: dict[str, Any] | None = None) -> None:
        """Attach lightweight run-scoped context that will be copied into emitted events."""

        self._memory_event_run_context = dict(run_context or {})

    def bind_run_context(
        self, run_context: dict[str, Any] | None = None
    ) -> Token[dict[str, Any] | None] | None:
        """Bind transient stimulus-scoped context for the current task only."""

        if not run_context:
            return None
        return self._active_memory_event_run_context.set(dict(run_context))

    def reset_run_context(self, token: Token[dict[str, Any] | None] | None) -> None:
        if token is None:
            return
        self._active_memory_event_run_context.reset(token)

    def _current_run_context(self) -> dict[str, Any]:
        combined_run_context = dict(self._memory_event_run_context)
        active_run_context = self._active_memory_event_run_context.get()
        if active_run_context:
            combined_run_context.update(active_run_context)
        return combined_run_context

    def _log_memory_event(
        self,
        event: MemoryOperationEvent | None = None,
        *,
        operation_type: MemoryOperationType | None = None,
        status: MemoryOperationStatus | None = None,
        source: MemoryOperationSource | None = None,
        component: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> MemoryOperationEvent:
        """Build and emit a memory event when an optional sink is attached."""

        from heams.memory_systems.schemas import MemoryOperationEvent

        if event is None:
            if operation_type is None or status is None or source is None:
                raise ValueError(
                    "operation_type, status, and source are required when event is not provided."
                )
            event_payload = dict(payload or {})
            event = MemoryOperationEvent(
                operation_type=operation_type,
                status=status,
                source=source,
                component=component
                or getattr(self.config, "type", None)
                or self.config.name,
                duration_ms=duration_ms,
                payload=event_payload,
                result=dict(result or {}),
                error=error,
            )

        combined_run_context = self._current_run_context()
        if run_context:
            combined_run_context.update(run_context)

        if combined_run_context and "run_context" not in event.payload:
            payload_with_context = dict(event.payload)
            payload_with_context["run_context"] = combined_run_context
            event = event.model_copy(update={"payload": payload_with_context})

        if self._memory_event_sink is None:
            return event

        if callable(self._memory_event_sink):
            self._memory_event_sink(event)
        else:
            self._memory_event_sink.append(event)
        return event

    @staticmethod
    def _memory_size_proxy(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=True, sort_keys=True))
        except (TypeError, ValueError):
            return len(str(value))

    def _emit_memory_operation(
        self,
        *,
        start_time: float,
        operation_type: MemoryOperationType | str,
        status: MemoryOperationStatus | str,
        payload: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        source: MemoryOperationSource | str = "runtime",
        component: str | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> MemoryOperationEvent:
        from heams.memory_systems.schemas import (
            MemoryOperationSource,
            MemoryOperationStatus,
            MemoryOperationType,
        )

        resolved_operation_type = (
            operation_type
            if isinstance(operation_type, MemoryOperationType)
            else MemoryOperationType(operation_type)
        )
        resolved_status = (
            status
            if isinstance(status, MemoryOperationStatus)
            else MemoryOperationStatus(status)
        )
        resolved_source = (
            source
            if isinstance(source, MemoryOperationSource)
            else MemoryOperationSource(source)
        )

        return self._log_memory_event(
            operation_type=resolved_operation_type,
            status=resolved_status,
            source=resolved_source,
            component=component,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            payload=payload,
            result=result,
            error=error,
            run_context=run_context,
        )

    def _run_logged_operation(
        self,
        *,
        operation_type: MemoryOperationType | str,
        payload: dict[str, Any],
        operation: Callable[[], MemoryOperationResult],
        on_error: Callable[[Exception], MemoryOperationResult],
    ) -> Any:
        start_time = time.perf_counter()
        try:
            operation_result = operation()
        except Exception as exc:
            operation_result = on_error(exc)
            self._emit_memory_operation(
                start_time=start_time,
                operation_type=operation_type,
                status="error",
                payload={**payload, **operation_result.payload},
                result=operation_result.result,
                error=operation_result.error or str(exc),
            )
            return operation_result.value

        self._emit_memory_operation(
            start_time=start_time,
            operation_type=operation_type,
            status=operation_result.status,
            payload={**payload, **operation_result.payload},
            result=operation_result.result,
            error=operation_result.error,
        )
        return operation_result.value

    def retain(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        TOOL: Saves a new fact, observation, or experience to long-term memory.

        Args:
            content: The raw string of the memory to save.
            metadata: Optional dictionary of tags (e.g., {"role": "user", "timestamp": "..."}).

        Returns:
            A string confirming the save status, which the LLM can read.
        """
        input_metadata = metadata or {}
        payload = {
            "content": content,
            "metadata": input_metadata,
            "payload_size_chars": self._memory_size_proxy(
                {"content": content, "metadata": input_metadata}
            ),
        }

        return self._run_logged_operation(
            operation_type="retain",
            payload=payload,
            operation=lambda: self._retain_impl(content, input_metadata),
            on_error=lambda exc: MemoryOperationResult(
                value=f"Failed to retain memory: {exc}",
                result={
                    "message": f"Failed to retain memory: {exc}",
                    "write_count": 0,
                },
                error=str(exc),
            ),
        )

    def recall(
        self,
        query: str = "",
        top_k: int | None = None,
    ) -> list[str]:
        """
        TOOL: Searches long-term memory for information related to the query.

        Args:
            query: The question or keyword to search for.
            top_k: Optional override for how many results to return.

        Returns:
            A list of string memories that match the query.
        """
        limit = top_k if top_k is not None else self.config.top_k
        query_size_bytes = self._memory_size_proxy({"query": query, "top_k": top_k})
        payload = {
            "query": query,
            "requested_top_k": top_k,
            "effective_top_k": limit,
            "payload_size_chars": query_size_bytes,
            "payload_size_bytes": query_size_bytes,
        }

        return self._run_logged_operation(
            operation_type="recall",
            payload=payload,
            operation=lambda: self._recall_impl(query, limit),
            on_error=lambda exc: MemoryOperationResult(
                value=[f"Failed to recall memories: {exc}"],
                result={
                    "returned_items": [f"Failed to recall memories: {exc}"],
                    "result_count": 1,
                    "returned_count": 1,
                    "output_bytes": self._memory_size_proxy(
                        [f"Failed to recall memories: {exc}"]
                    ),
                    "output_size_chars": self._memory_size_proxy(
                        [f"Failed to recall memories: {exc}"]
                    ),
                },
                error=str(exc),
            ),
        )

    def delete(self, query: str) -> str:
        """
        TOOL: Deletes memories matching the provided query.

        Args:
            query: The search text describing which memory or memories to remove.

        Returns:
            A string describing the delete outcome.
        """
        payload = {
            "query": query,
            "payload_size_chars": self._memory_size_proxy({"query": query}),
        }
        return self._run_logged_operation(
            operation_type="delete",
            payload=payload,
            operation=lambda: self._delete_impl(query),
            on_error=lambda exc: MemoryOperationResult(
                value=f"Failed to delete memory: {exc}",
                result={
                    "message": f"Failed to delete memory: {exc}",
                    "deleted_count": 0,
                },
                error=str(exc),
            ),
        )

    def consolidate(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Perform maintenance or consolidation work for the memory backend.

        Args:
            context: Optional recent context to use during consolidation.
            metadata: Optional dictionary of tags or maintenance hints.

        Returns:
            A string describing the consolidation outcome.
        """
        input_metadata = metadata or {}
        payload = {
            "context": context,
            "metadata": input_metadata,
            "payload_size_chars": self._memory_size_proxy(
                {"context": context, "metadata": input_metadata}
            ),
        }
        return self._run_logged_operation(
            operation_type="consolidate",
            payload=payload,
            operation=lambda: self._consolidate_impl(context, input_metadata),
            on_error=lambda exc: MemoryOperationResult(
                value=f"Failed to consolidate memory: {exc}",
                result={
                    "summary": f"Failed to consolidate memory: {exc}",
                    "changed_count": 0,
                },
                error=str(exc),
            ),
        )

    def set_retention_policy(self, policies: list[str]) -> str:
        """Replace the user-declared retention policies this system enforces.

        Orchestrator-only: dispatched as a `MemoryManipulation` so a benchmark
        can vary the active policy per scenario, never exposed as an agent
        tool. Not abstract -- a backend with no governance layer has nothing
        to configure, so the default says so loudly instead of every backend
        having to stub it.

        Args:
            policies: Free-text policy statements, e.g. "Do not retain the
                user's health information."

        Returns:
            A string confirming the policies now in force.

        The default enforces nothing: it logs a `skipped`
        `set_retention_policy` event, so results show the system ran
        ungoverned, and the system keeps storing everything.
        """
        summary = (
            f"No retention policy enforced: {type(self).__name__} has no governance."
        )
        self._log_memory_event(
            operation_type="set_retention_policy",
            status="skipped",
            source="runtime",
            payload={"policies": list(policies)},
            result={"summary": summary, "enforced": False},
        )
        return summary

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        """Prime memory using the public retain API when a backend has no batch path."""

        for entry in entries:
            self.retain(content=entry.content, metadata=entry.metadata)

    @abstractmethod
    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        raise NotImplementedError

    @abstractmethod
    def _recall_impl(
        self,
        query: str,
        top_k: int,
    ) -> MemoryOperationResult:
        raise NotImplementedError

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        summary = "Delete is not supported by this memory system."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={"summary": summary, "supported": False, "deleted_count": 0},
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        summary = "This memory system has no consolidation step."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={"summary": summary, "supported": False, "changed_count": 0},
        )
