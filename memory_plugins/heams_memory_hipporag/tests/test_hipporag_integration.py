"""Live HippoRAG coverage through the bundled OpenAI preset."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from heams.memory_systems.api import MemoryEntry
from heams.memory_systems.factory import create_memory
from heams.orchestration.run_orchestration import _load_memory_system_config

OPENAI_PRESET = (
    Path(__file__).parents[3]
    / "python_package/configs/memory_systems/campaigns/hipporag_openai.yaml"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for the HippoRAG integration test",
)
def test_openai_preset_primes_recalls_and_deletes(tmp_path) -> None:
    memory = create_memory(
        _load_memory_system_config(OPENAI_PRESET).backend_config,
        storage_dir=tmp_path,
    )
    moved = "user: I moved from Montreal to Toronto last spring."
    try:
        memory.prime_memory_entries(
            [
                MemoryEntry(
                    content="I moved from Montreal to Toronto last spring.",
                    metadata={"role": "user", "session_index": 0},
                ),
                MemoryEntry(
                    content="My sister Priya is a nurse in Vancouver.",
                    metadata={"role": "user", "session_index": 1},
                ),
            ]
        )

        top = json.loads(memory.recall("Where does the user live now?", top_k=1)[0])
        assert top == {"content": moved, "metadata": {"role": "user", "session_index": 0}}

        memory.delete(moved)
        remaining = [
            json.loads(item)["content"]
            for item in memory.recall("Where does the user live now?", top_k=2)
        ]
        assert remaining == ["user: My sister Priya is a nurse in Vancouver."]
    finally:
        memory.close()
