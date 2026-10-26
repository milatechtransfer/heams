"""Public plugin surface for externally installed memory systems.

This module is the supported import boundary for plugin packages. The
implementations remain in the internal memory-system modules so built-in
systems can migrate independently.
"""

from .base import (
    BaseConsolidationConfig,
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEventLedger,
    MemoryEventSink,
    MemoryEventTarget,
    MemoryOperationResult,
)
from .plugins import (
    MEMORY_PLUGIN_API_VERSION,
    BaseMemoryWrapperConfig,
    MemoryPlugin,
    MemoryPluginFactory,
    MemoryRuntimeContext,
    MemoryWrapFunction,
    MemoryWrapperPlugin,
)
from .schemas import (
    MemoryOperationEvent,
    MemoryOperationSource,
    MemoryOperationStatus,
    MemoryOperationType,
)
from .shared_types import MemoryEntry, render_tool_memory_entry

__all__ = [
    "BaseConsolidationConfig",
    "BaseMemoryConfig",
    "BaseMemorySystem",
    "BaseMemoryWrapperConfig",
    "MEMORY_PLUGIN_API_VERSION",
    "MemoryEntry",
    "MemoryEventLedger",
    "MemoryEventSink",
    "MemoryEventTarget",
    "MemoryOperationEvent",
    "MemoryOperationResult",
    "MemoryOperationSource",
    "MemoryOperationStatus",
    "MemoryOperationType",
    "MemoryPlugin",
    "MemoryPluginFactory",
    "MemoryRuntimeContext",
    "MemoryWrapFunction",
    "MemoryWrapperPlugin",
    "render_tool_memory_entry",
]
