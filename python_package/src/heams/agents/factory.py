from typing import Annotated, Union, Callable, Any
from pydantic import Field

from .config import AgentParameters
from .tiny_agent import TinyAgent as _RegisteredTinyAgent  # noqa: F401
from heams.memory_systems.base import BaseMemorySystem
from heams.utils import REGISTRY

AgentConfig = Annotated[
    Union[AgentParameters,],
    Field(discriminator="type"),
]


def create_agent(
    config: AgentConfig,
    tools: list[Callable[..., Any]] | None = None,
    memory_system: BaseMemorySystem | None = None,
):
    class_type = REGISTRY.get(config.type, None)
    if not class_type:
        raise ValueError(f"Class {config.type} not found in the class registry.")
    return class_type(config=config, tools=tools, memory_system=memory_system)
