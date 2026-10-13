from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from heams.environments import (
    AlfWorldEnvironment,
    AlfWorldEnvironmentConfig,
    EnvironmentConfig,
    create_environment,
)


def test_create_environment_returns_alfworld_environment() -> None:
    environment = create_environment(
        AlfWorldEnvironmentConfig(
            task_id="t",
            tasks=[{"task_id": "t0", "goal": "g", "kind": "heat"}],
        )
    )
    assert isinstance(environment, AlfWorldEnvironment)


class UnknownEnvironmentConfig(BaseModel):
    type: Literal["unknown_environment"] = "unknown_environment"


def test_create_environment_raises_for_unknown_registry_key() -> None:
    config = UnknownEnvironmentConfig()
    with pytest.raises(ValueError, match="not found in the class registry"):
        create_environment(cast(Any, config))


def test_environment_config_discriminator_rejects_unknown_type() -> None:
    adapter = TypeAdapter(EnvironmentConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown_environment", "fields": ["name"]})
