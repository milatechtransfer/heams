"""Live Mem0 integration coverage for LOCOMO temporal extraction."""

from __future__ import annotations

import json
import gc
import os
import platform
import shutil
import subprocess

import pytest

from heams.memory_systems.api import MemoryEntry
from heams_memory_mem0 import (
    EmbedderConfig,
    LLMConfig,
    Mem0MemoryConfig,
    Mem0MemorySystem,
    VectorStoreConfig,
)


def _open_file_descriptor_count() -> int:
    pid = os.getpid()
    if platform.system() == "Linux":
        return len(os.listdir(f"/proc/{pid}/fd"))
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["lsof", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("lsof is required for descriptor measurement") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "lsof failed")
        return max(0, len(result.stdout.splitlines()) - 1)
    raise RuntimeError(f"Unsupported platform: {platform.system()}")


def _chroma_cache_state() -> tuple[set[str], dict[str, int]]:
    from chromadb.api.shared_system_client import SharedSystemClient

    return (
        set(SharedSystemClient._identifier_to_system),
        dict(SharedSystemClient._identifier_to_refcount),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for the Mem0 integration test",
)
def test_locomo_relative_dates_are_resolved_by_real_mem0(tmp_path) -> None:
    memory = Mem0MemorySystem(
        config=Mem0MemoryConfig(
            user_id="locomo-temporal-integration",
            temporal_prompt_enabled=True,
            llm=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=2000,
            ),
            embedder=EmbedderConfig(
                provider="openai",
                model="text-embedding-3-small",
                embedding_dims=512,
            ),
            vector_store=VectorStoreConfig(provider="chroma"),
        ),
        storage_dir=tmp_path,
    )
    source_content = (
        "I went to a LGBTQ support group yesterday. "
        "I returned to the group last week. "
        "I started volunteering there last year. "
        "I joined another group last month."
    )

    memory.prime_memory_entries(
        [
            MemoryEntry(
                content=source_content,
                metadata={
                    "source": "locomo",
                    "source_benchmark": "locomo",
                    "session_id": "session_1",
                    "timestamp": "08 May 2023",
                    "ingest_timestamp_epoch": 1683504000,
                    "ingest_group_id": "locomo:session_1",
                    "ingest_order": 0,
                    "ingest_role": "user",
                    "ingest_speaker_name": "Caroline",
                },
            )
        ]
    )

    expected_dates_by_query = {
        "When did Caroline attend the LGBTQ support group?": (
            "attended an LGBTQ support group",
            "May 7, 2023",
            "7 May 2023",
            "2023-05-07",
        ),
        "When did Caroline return to the LGBTQ support group?": (
            "returned to the group",
            "May 1, 2023",
            "May 7, 2023",
            "2023-05-01",
            "2023-05-07",
            "April 30, 2023",
            "30 April 2023",
            "2023-04-30",
        ),
        "When did Caroline start volunteering at the LGBTQ support group?": (
            "started volunteering",
            "2022",
        ),
        "When did Caroline join another group?": (
            "joined another group",
            "April 2023",
            "2023-04",
        ),
    }

    for query, expected_values in expected_dates_by_query.items():
        event_phrase, *expected_dates = expected_values
        recalled = memory.recall(query, top_k=10)
        recalled_records = [json.loads(result) for result in recalled]

        assert any(
            event_phrase in record["content"]
            and any(date in record["content"] for date in expected_dates)
            for record in recalled_records
        ), (
            f"Expected {event_phrase!r} with one of {expected_dates!r} for "
            f"{query!r}; recalled: {recalled_records}"
        )
        assert all(
            "2026" not in record["content"] for record in recalled_records
        ), f"recalled: {recalled_records}"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for the Mem0 resource cleanup test",
)
def test_mem0_cleanup_does_not_accumulate_resources(tmp_path) -> None:
    """Exercise repeated real Mem0 lifecycles before running a campaign."""
    try:
        baseline_fds = _open_file_descriptor_count()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    baseline_systems, baseline_refcounts = _chroma_cache_state()
    fd_counts = [baseline_fds]
    cycle_systems: set[str] = set()
    config = Mem0MemoryConfig(
        user_id="resource-cleanup-integration",
        llm=LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=200,
        ),
        embedder=EmbedderConfig(
            provider="openai",
            model="text-embedding-3-small",
            embedding_dims=512,
        ),
        vector_store=VectorStoreConfig(provider="chroma"),
    )

    for cycle in range(10):
        storage_dir = tmp_path / f"cycle-{cycle}"
        storage_dir.mkdir()
        memory = Mem0MemorySystem(config=config, storage_dir=storage_dir)
        systems_after_create, _ = _chroma_cache_state()
        cycle_systems.update(systems_after_create - baseline_systems)
        try:
            memory.retain(
                f"Cleanup cycle {cycle}: Alice lives in Montreal.",
                metadata={"source": "resource-cleanup-integration"},
            )
            recalled = memory.recall("Where does Alice live?", top_k=3)
            assert any("Montreal" in result for result in recalled)
        finally:
            try:
                memory.close()
            finally:
                gc.collect()
                fd_counts.append(_open_file_descriptor_count())
                shutil.rmtree(storage_dir)
                assert not storage_dir.exists()

        systems_after_close, refcounts_after_close = _chroma_cache_state()
        assert not cycle_systems & systems_after_close
        assert not any(
            identifier in refcounts_after_close for identifier in cycle_systems
        )

    final_systems, final_refcounts = _chroma_cache_state()
    final_fds = _open_file_descriptor_count()
    print(
        "Mem0 cleanup resource summary: "
        f"fd baseline/peak/final={baseline_fds}/{max(fd_counts)}/{final_fds}; "
        f"fd after cycles={fd_counts}; "
        f"Chroma baseline/final={len(baseline_systems)}/{len(final_systems)}; "
        f"refcounts baseline/final={len(baseline_refcounts)}/{len(final_refcounts)}"
    )

    assert not cycle_systems & final_systems
    assert not any(identifier in final_refcounts for identifier in cycle_systems)
    assert final_fds <= fd_counts[1] + 2
    assert max(fd_counts[-3:]) - min(fd_counts[-3:]) <= 2
