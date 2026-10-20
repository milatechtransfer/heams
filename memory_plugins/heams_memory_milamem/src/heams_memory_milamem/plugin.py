from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryPlugin,
    MemoryPluginFactory,
    MemoryRuntimeContext,
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
