from typing import Annotated, Union

from pydantic import Field

from heams.utils import REGISTRY

from .alfworld import AlfWorldEnvironmentConfig
from .babyai import BabyAIEnvironmentConfig
from .pddl import PddlEnvironmentConfig
from .scienceworld import ScienceWorldEnvironmentConfig

EnvironmentConfig = Annotated[
    Union[
        AlfWorldEnvironmentConfig,
        BabyAIEnvironmentConfig,
        PddlEnvironmentConfig,
        ScienceWorldEnvironmentConfig,
    ],
    Field(discriminator="type"),
]


def create_environment(config: EnvironmentConfig):
    class_type = REGISTRY.get(config.type)
    if not class_type:
        raise ValueError(f"Class {config.type} not found in the class registry.")
    return class_type(config)
