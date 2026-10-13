from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryPlugin,
    MemoryPluginFactory,
    MemoryRuntimeContext,
)

from .basic_complete_memory_system import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
)
from .milamem import MilaMemConfig, MilaMemMemorySystem


def _factory(backend: type[BaseMemorySystem]) -> MemoryPluginFactory:
    def create(
        config: BaseMemoryConfig, runtime: MemoryRuntimeContext
    ) -> BaseMemorySystem:
        return backend(
            config=config,
            storage_dir=runtime.storage_dir,
            event_sink=runtime.event_sink,
            run_context=runtime.run_context,
        )

    return create


milamem_plugin = MemoryPlugin(
    type_name="milamem",
    display_name="MilaMem",
    config_model=MilaMemConfig,
    create=_factory(MilaMemMemorySystem),
)

basic_complete_plugin = MemoryPlugin(
    type_name="basic_complete_memory_system",
    display_name="Basic complete memory",
    config_model=BasicCompleteMemorySystemConfig,
    create=_factory(BasicCompleteMemorySystem),
)
