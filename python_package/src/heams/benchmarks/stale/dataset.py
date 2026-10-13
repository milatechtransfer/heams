"""STALE dataset resolution: local JSON override, else HuggingFace download."""

from __future__ import annotations

from typing import TYPE_CHECKING

from heams.benchmarks.base import BenchmarkDatasetUnavailableError
from heams.benchmarks.stale.samples import StaleSample, load_stale_samples

if TYPE_CHECKING:
    from heams.benchmarks.stale.config import StaleBenchmarkConfig


def materialize_source(config: StaleBenchmarkConfig) -> list[StaleSample]:
    """Resolve the configured dataset into validated STALE samples.

    A local ``source.local_path`` wins when set; otherwise the dataset is
    downloaded from HuggingFace (when ``source.auto_download`` is enabled).
    """
    source = config.source
    if source.local_path is not None:
        if not source.local_path.is_file():
            raise BenchmarkDatasetUnavailableError(
                f"STALE source.local_path not found: {source.local_path}. Point it "
                "at a STALE *_MAIN.json file, or unset it to download from HuggingFace."
            )
        return load_stale_samples(source.local_path)

    if not source.auto_download:
        raise BenchmarkDatasetUnavailableError(
            f"STALE dataset would be downloaded from {source.repo_id!r} but "
            "`source.auto_download` is false. Set `source.local_path` or enable "
            "auto_download."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on local setup
        raise BenchmarkDatasetUnavailableError(
            "Loading the STALE dataset from HuggingFace requires huggingface_hub. "
            "Run through uv from python_package, e.g. `uv run heams run --config "
            "configs/runs/stale.yaml --output-dir results/stale`."
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
    return load_stale_samples(dataset_path)


__all__ = ["materialize_source"]
