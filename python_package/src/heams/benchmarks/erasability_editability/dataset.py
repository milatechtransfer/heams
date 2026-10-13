"""EE dataset resolution: local JSON override, else HuggingFace download."""

from __future__ import annotations

from typing import TYPE_CHECKING

from heams.benchmarks.base import BenchmarkDatasetUnavailableError
from heams.benchmarks.erasability_editability.samples import (
    EEEpisode,
    load_ee_episodes,
)

if TYPE_CHECKING:
    from heams.benchmarks.erasability_editability.config import EEBenchmarkConfig


def materialize_source(config: EEBenchmarkConfig) -> list[EEEpisode]:
    """Resolve the configured dataset into validated MEME episodes.

    A local ``source.local_path`` wins when set; otherwise the dataset is
    downloaded from HuggingFace (when ``source.auto_download`` is enabled).
    """
    source = config.source
    if source.local_path is not None:
        if not source.local_path.is_file():
            raise BenchmarkDatasetUnavailableError(
                f"EE source.local_path not found: {source.local_path}. Point it at "
                "a MEME episodes JSON file (e.g. meme_filler32k.json), or unset it "
                "to download from HuggingFace."
            )
        return load_ee_episodes(source.local_path)

    if not source.auto_download:
        raise BenchmarkDatasetUnavailableError(
            f"EE dataset would be downloaded from {source.repo_id!r} but "
            "`source.auto_download` is false. Set `source.local_path` or enable "
            "auto_download."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on local setup
        raise BenchmarkDatasetUnavailableError(
            "Loading the MEME dataset from HuggingFace requires huggingface_hub. "
            "Run through uv from python_package, e.g. `uv run heams run --config "
            "configs/runs/erasability_editability.yaml`."
        ) from error
    try:
        dataset_path = hf_hub_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            filename=source.filename,
            revision=source.revision,
        )
    except Exception as error:
        raise BenchmarkDatasetUnavailableError(
            f"Could not download {source.filename} from the HuggingFace dataset "
            f"repo {source.repo_id!r}. Check network access and the repo_id / "
            "filename / revision."
        ) from error
    return load_ee_episodes(dataset_path)


__all__ = ["materialize_source"]
