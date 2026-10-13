import re
from pathlib import Path

import pytest
import yaml

from heams.agents.config import (
    AgentParameters,
    MemoryStrategy,
)
from heams.agents.tiny_agent import (
    TinyAgent,
)
from heams.benchmarks.factory import create_benchmark
from heams.memory_systems import InContextConfig, InContextMemorySystem
from heams.memory_systems.shared_types import MemoryEntry
from heams.scenarios import Scenario, UserStimulus
from tests.benchmarks.locomo.fixtures import load_locomo_benchmark_config

QWEN_25_7B_MODEL_ID = "ollama:qwen2.5:7b"

# Cache LOCOMO samples beside the python_package so reruns skip the download.
LOCOMO_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "locomo" / "locomo10.json"
)


def _first_grounded_locomo_qa(
    scenario: Scenario,
) -> tuple[UserStimulus, list[MemoryEntry], str]:
    for step in scenario.execution_sequence:
        if not isinstance(step, UserStimulus):
            continue
        markers = step.step_evaluation_markers
        expected = str(markers.get("expected_answer") or "")
        evidence_ids = set(markers.get("evidence_dia_ids") or [])
        if not expected or not evidence_ids:
            continue
        evidence_entries = [
            entry
            for entry in scenario.scenario_priming.memory.entries
            if entry.metadata.get("dia_id") in evidence_ids
        ]
        combined_text = " ".join(entry.content.lower() for entry in evidence_entries)
        if expected.lower() in combined_text:
            return step, evidence_entries, expected
    raise AssertionError(
        "No LOCOMO QA with its expected answer appearing in the evidence turns."
    )


# End-to-end LOCOMO test: loads the upstream dataset via create_benchmark,
# primes an InContextMemorySystem with the dialogue turns flagged as evidence,
# and verifies the wrapper's deterministic memory injection can answer the
# first grounded QA. The dataset auto-downloads on first access.
@pytest.mark.integration
@pytest.mark.requires_ollama
def test_locomo_scenario_answer_recalled_via_agent(tmp_path) -> None:
    samples = list(
        create_benchmark(load_locomo_benchmark_config(examples_dir=LOCOMO_DATASET_PATH))
    )
    assert samples, "LOCOMO dataset produced no samples."

    scenario = samples[0].to_scenario()
    agent_yaml = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "agents"
        / "testing"
        / "gpt4o_mini_tool_use.yaml"
    )
    agent_payload = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))

    # Pick the first QA whose expected answer appears verbatim in its
    # evidence turns. This keeps the assertion anchored on single-hop
    # retrieval rather than multi-hop reasoning that a small local model
    # may not handle reliably.
    stimulus, evidence_entries, expected_answer = _first_grounded_locomo_qa(scenario)

    memory = InContextMemorySystem(
        config=InContextConfig(
            name="locomo_integration_memory",
            top_k=len(evidence_entries),
        ),
        storage_dir=tmp_path,
    )
    for entry in evidence_entries:
        speaker = entry.metadata.get("speaker")
        content = f"{speaker}: {entry.content}" if speaker else entry.content
        memory.retain(content, metadata=entry.metadata)

    agent = TinyAgent(
        config=AgentParameters(
            memory_strategy=MemoryStrategy.DETERMINISTIC,
            model_id=QWEN_25_7B_MODEL_ID,
            instructions=agent_payload["instructions"],
            model_args={"temperature": 0.0},
        ),
        tools=[memory.recall, memory.delete, memory.consolidate],
        memory_system=memory,
    )

    response = agent.run(stimulus.content)
    assert response.answer.strip(), "Agent returned an empty response."

    normalized_response = re.sub(r"\s+", " ", response.answer.lower()).strip()
    normalized_expected = re.sub(r"\s+", " ", expected_answer.lower()).strip()
    assert normalized_expected in normalized_response, (
        f"Expected '{expected_answer}' in response for question "
        f"{stimulus.content!r}: got {response.answer!r}"
    )
