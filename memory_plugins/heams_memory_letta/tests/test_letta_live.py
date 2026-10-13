"""Live check of the Letta preset against Postgres and the model provider.

Runs the campaign preset through one small LoCoMo-shaped session and one recall, so a
broken setup fails in about a minute instead of partway through a LoCoMo ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from heams.memory_systems.api import MemoryEntry

from heams_memory_letta import LettaMemoryConfig, LettaMemorySystem

PRESET = (
    Path(__file__).resolve().parents[3]
    / "python_package/configs/memory_systems/campaigns/letta_openai.yaml"
)
CONFIG = LettaMemoryConfig.model_validate(
    yaml.safe_load(PRESET.read_text())["backend_config"]
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(CONFIG.pg_uri_env) or not os.environ.get("OPENAI_API_KEY"),
        reason=f"needs {CONFIG.pg_uri_env} and OPENAI_API_KEY",
    ),
]

SESSION = [
    ("user", "Caroline", "I adopted a golden retriever puppy. I named her Biscuit!"),
    ("assistant", "Melanie", "That's wonderful! How is Biscuit settling in?"),
    ("user", "Caroline", "She loves the dog park near my apartment in Portland."),
]


def test_preset_primes_a_session_and_recalls_it(tmp_path: Path) -> None:
    events: list[Any] = []
    memory = LettaMemorySystem(
        config=CONFIG,
        storage_dir=tmp_path,
        event_sink=events,
        run_context={"scenario_id": "live-check"},
    )
    try:
        memory.prime_memory_entries(
            [
                MemoryEntry(
                    content=content,
                    metadata={
                        "ingest_group_id": "session_1",
                        "ingest_order": order,
                        "ingest_role": role,
                        "ingest_speaker_name": speaker,
                        "timestamp": "1:56 pm on 8 May, 2023",
                    },
                )
                for order, (role, speaker, content) in enumerate(SESSION)
            ]
        )
        (prime,) = events

        def messages() -> tuple[list[str], list[str]]:
            """The agent's context and Letta's whole message log for it."""
            log = memory._in_letta(
                lambda manager, actor: manager.list_messages(
                    actor=actor, agent_id=memory._agent_id, limit=None
                )
            )
            context = memory._api("GET", f"/agents/{memory._agent_id}")["message_ids"]
            return context, [message.id for message in log]

        assert prime.status == "success", prime.error
        assert prime.result["stop_reason"] == "end_turn"
        assert prime.result["write_count"] >= 1

        before = messages()
        results = memory.recall("What is the name of Caroline's dog?")
        recall = events[-1]
        assert recall.operation_type == "recall"
        assert recall.status == "success", recall.error
        assert "biscuit" in "\n".join(results).lower(), recall.result
        assert messages() == before  # the probe left nothing behind
    finally:
        memory.close()
