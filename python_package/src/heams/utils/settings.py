from pathlib import Path
from typing import ClassVar

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define default paths to environment files
CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parent.parent.parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
ENV_FILES = [
    REPO_ROOT / ".env",
    PACKAGE_ROOT / ".env",
]


class GlobalSettings(BaseSettings):
    """A global settings singleton."""

    project_dir: Path = PACKAGE_ROOT
    config_dir: Path = PACKAGE_ROOT / "configs"
    datasets_dir: Path = PACKAGE_ROOT / "datasets"

    #: Root directory for timestamped benchmark runs (relative to cwd unless absolute).
    default_results_dir: Path = Field(default=Path("results"))

    #: Default directory for persisted memory artefacts when ``--memory_dir`` is omitted.
    default_memory_dir: Path = Field(default=Path("memories"))

    #: Provider secret used when agent configs do not provide an explicit key.
    openai_api_key: SecretStr | None = Field(default=None)

    # Load environment variables from .env files
    model_config: ClassVar = SettingsConfigDict(
        env_file=ENV_FILES, env_ignore_empty=True, extra="ignore"
    )


# Instantiate a global singleton
settings = GlobalSettings()
