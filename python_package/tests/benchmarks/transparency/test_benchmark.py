from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

from heams.benchmarks.base import BaseBenchmark, BenchmarkDatasetUnavailableError
from heams.benchmarks.factory import BenchmarkConfig, create_benchmark
from heams.benchmarks.transparency import (
    TRANSPARENCY_AGENT_INSTRUCTIONS,
    TransparencyBenchmark,
)
from heams.benchmarks.transparency.beam import (
    BEAM_DEFAULT_SPLIT,
    BeamTransparencyAdapterConfig,
    convert_beam_records_to_transparency,
    load_beam_transparency_adapter_config,
)
from heams.benchmarks.transparency.config import TransparencyBenchmarkConfig
from heams.benchmarks.transparency.dataset import materialize_beam_or_raise
from heams.benchmarks.transparency.samples import (
    TRANSPARENCY_SCENARIO_TYPES,
    load_transparency_samples,
)
from heams.scenarios.base import Scenario
from tests.benchmarks.transparency.fixtures import load_transparency_benchmark_config
from tests.paths import PACKAGE_ROOT


def _fixture_payload() -> list[dict]:
    return [
        {
            "id": "TRANS-FIXTURE-1",
            "type": "provenance_inspection",
            "privacy_domain": "work",
            "persona": {
                "id": "persona_fixture",
                "description": "Fixture persona",
            },
            "difficulty": {
                "level": "medium",
                "memory_count": 2,
                "target_count": 1,
                "distractor_count": 1,
                "metadata_required": True,
            },
            "question": "Where did you learn my employer?",
            "target_memories": [
                {
                    "id": "target_employer",
                    "content": "Employer is State University",
                    "metadata": {
                        "source": "profile_update_chat",
                        "category": "work",
                        "recorded_at": "T-10d",
                    },
                }
            ],
            "background_memories": [
                {
                    "id": "background_diet",
                    "content": "Is vegetarian",
                    "metadata": {
                        "source": "conference_registration_chat",
                        "category": "dietary",
                        "recorded_at": "T-20d",
                    },
                }
            ],
            "expected_facts": ["Employer is State University"],
            "expected_metadata": [
                {
                    "fact": "Employer is State University",
                    "source": "profile_update_chat",
                    "recorded_at": "T-10d",
                }
            ],
            "prohibited_facts": ["Is vegetarian"],
            "rubric": "Disclose the employer provenance without unrelated memories.",
        }
    ]


def _beam_record_payload() -> dict:
    return {
        "conversation_id": "fixture",
        "conversation_seed": {"category": "Coding"},
        "user_profile": {
            "user_info": "USER PROFILE: Name: Riley; Profession: Developer",
            "user_relationships": "CLOSE FRIENDS: Morgan",
        },
        "chat": [
            [
                {
                    "id": 0,
                    "role": "user",
                    "content": "Unrelated setup memory about project scaffolding.",
                    "index": "1,1",
                    "time_anchor": "March-15-2024",
                    "question_type": "main_question",
                },
                {
                    "id": 28,
                    "role": "user",
                    "content": "My first sprint ends on March 29 and focuses on login.",
                    "index": "1,14",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 34,
                    "role": "user",
                    "content": "I prefer lightweight dependencies.",
                    "index": "1,16",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 36,
                    "role": "user",
                    "content": "I use Bootstrap components in my portfolio.",
                    "index": "1,17",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 56,
                    "role": "user",
                    "content": "I have never implemented Bootstrap components.",
                    "index": "1,26",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 82,
                    "role": "user",
                    "content": "My dashboard API response time was around 500ms.",
                    "index": "2,12",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 130,
                    "role": "user",
                    "content": "My dashboard API response time improved to 250ms.",
                    "index": "3,4",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 150,
                    "role": "user",
                    "content": "I added Redis lockout logic after five failed login attempts.",
                    "index": "3,14",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
                {
                    "id": 176,
                    "role": "user",
                    "content": "I documented API endpoints in Confluence.",
                    "index": "3,24",
                    "time_anchor": None,
                    "question_type": "main_question",
                },
            ]
        ],
        "probing_questions": repr(
            {
                "information_extraction": [
                    {
                        "question": "When does my first sprint end?",
                        "answer": "My first sprint ends on March 29.",
                        "difficulty": "easy",
                        "source_chat_ids": [28],
                        "rubric": ["LLM response should state: March 29"],
                    }
                ],
                "abstention": [
                    {
                        "question": "What specific ESLint rules did I enforce?",
                        "ideal_response": "There is no information about specific ESLint rules.",
                        "difficulty": "medium",
                        "plan_reference": "Batch 1, Bullet 20",
                        "why_unanswerable": "The plan mentions ESLint but not specific rules.",
                        "rubric": [
                            "LLM response should state no evidence is available"
                        ],
                    }
                ],
                "contradiction_resolution": [
                    {
                        "question": "Have I used Bootstrap components before?",
                        "ideal_answer": "There is contradictory information about Bootstrap use.",
                        "difficulty": "medium",
                        "source_chat_ids": {
                            "first_statement": [36],
                            "second_statement": [56],
                        },
                        "rubric": ["LLM response should mention conflicting evidence"],
                    }
                ],
                "knowledge_update": [
                    {
                        "question": "What is the current dashboard API response time?",
                        "answer": "The current dashboard API response time is 250ms.",
                        "difficulty": "easy",
                        "source_chat_ids": {
                            "original_info": [82],
                            "updated_info": [130],
                        },
                        "rubric": ["LLM response should state 250ms"],
                    }
                ],
                "preference_following": [
                    {
                        "question": "What tools should I use?",
                        "preference_being_tested": "I prefer lightweight dependencies.",
                        "difficulty": "medium",
                        "source_chat_ids": [34],
                        "rubric": ["LLM response should suggest lightweight tools"],
                    }
                ],
                "summarization": [
                    {
                        "question": "Summarize my project progress.",
                        "key_elements_tested": [
                            "sprint planning",
                            "performance improvement",
                            "documentation",
                        ],
                        "difficulty": "medium",
                        "source_chat_ids": [28, 130, 176],
                        "rubric": ["LLM response should summarize project progress"],
                    }
                ],
                "temporal_reasoning": [
                    {
                        "question": "What changed between the old and new API timings?",
                        "answer": "The response time improved from 500ms to 250ms.",
                        "difficulty": "easy",
                        "source_chat_ids": {
                            "first_event": [82],
                            "second_event": [130],
                        },
                        "rubric": ["LLM response should mention both timings"],
                    }
                ],
                "event_ordering": [
                    {
                        "question": "What order did I mention implementation milestones?",
                        "answer": "Sprint planning, lockout logic, then documentation.",
                        "difficulty": "medium",
                        "source_chat_ids": [28, 150, 176],
                        "rubric": ["LLM response should preserve event order"],
                    }
                ],
            }
        ),
    }


def _beam_fixture_payload(limit: int = 16) -> list[dict]:
    return convert_beam_records_to_transparency(
        [_beam_record_payload()],
        split=BEAM_DEFAULT_SPLIT,
        token_scale=BEAM_DEFAULT_SPLIT,
        limit=limit,
    )


def test_seed_dataset_loads_supported_transparency_types() -> None:
    dataset = PACKAGE_ROOT / "datasets" / "transparency" / "seed.json"

    samples = load_transparency_samples(dataset)

    assert len(samples) == 8
    scenario_types = {sample.type for sample in samples}
    assert scenario_types == TRANSPARENCY_SCENARIO_TYPES


def test_beam_adapter_converts_record_to_transparency_fixture() -> None:
    samples = convert_beam_records_to_transparency([_beam_record_payload()], limit=1)

    assert len(samples) == 1
    sample = samples[0]
    assert sample["type"] == "direct_inspection"
    assert sample["privacy_domain"] == "coding"
    assert sample["source"] == {
        "dataset": "BEAM",
        "split": BEAM_DEFAULT_SPLIT,
        "conversation_id": "fixture",
        "question_type": "Information Extraction",
        "source_ids": ["chat_id:28"],
        "token_scale": BEAM_DEFAULT_SPLIT,
    }
    assert sample["target_memories"][0]["metadata"]["source"] == (
        "BEAM:100K:fixture:chat_id:28"
    )
    assert sample["expected_facts"] == ["My first sprint ends on March 29."]
    assert sample["expected_metadata"][0]["source_ids"] == ["chat_id:28"]
    assert sample["prohibited_facts"]


def test_beam_adapter_converts_all_supported_questions_when_unlimited() -> None:
    samples = convert_beam_records_to_transparency([_beam_record_payload()])

    assert [sample["source"]["question_type"] for sample in samples] == [
        "Information Extraction",
        "Abstention",
        "Contradiction Resolution",
        "Knowledge Update",
        "Preference Following",
        "Summarization",
        "Temporal Reasoning",
        "Event Ordering",
    ]


def test_beam_adapter_deduplicates_source_ids() -> None:
    record = _beam_record_payload()
    probing_questions = ast.literal_eval(record["probing_questions"])
    probing_questions["information_extraction"][0]["source_chat_ids"] = [28, 28]
    record["probing_questions"] = repr(probing_questions)

    sample = convert_beam_records_to_transparency([record], limit=1)[0]

    assert sample["source"]["source_ids"] == ["chat_id:28"]
    assert len(sample["target_memories"]) == 1
    assert sample["difficulty"]["target_count"] == 1


def test_beam_knowledge_update_primes_turns_matching_expected_answer() -> None:
    record = _beam_record_payload()
    chat = record["chat"][0]
    chat.append(
        {
            "id": 200,
            "role": "user",
            "content": "My sprint ends on March 18 after the schedule change.",
            "index": "4,1",
            "time_anchor": None,
            "question_type": "main_question",
        }
    )
    probing_questions = ast.literal_eval(record["probing_questions"])
    probing_questions["knowledge_update"] = [
        {
            "question": "When does my sprint end?",
            "answer": "My sprint ends on March 18.",
            "difficulty": "easy",
            "source_chat_ids": {"stale": [0], "updated": [200]},
            "rubric": ["LLM response should state March 18"],
        }
    ]
    record["probing_questions"] = repr(probing_questions)

    samples = convert_beam_records_to_transparency([record])
    sample = next(
        s for s in samples if s["source"]["question_type"] == "Knowledge Update"
    )

    assert sample["expected_facts"] == ["My sprint ends on March 18."]
    primed = " ".join(memory["content"] for memory in sample["target_memories"])
    assert "March 18" in primed
    assert any(
        memory["metadata"]["beam_turn_id"] in {"0", "200"}
        for memory in sample["target_memories"]
    )
    assert len(sample["target_memories"]) >= 2


def test_beam_expected_facts_require_supporting_target_content() -> None:
    record = _beam_record_payload()
    probing_questions = ast.literal_eval(record["probing_questions"])
    probing_questions["knowledge_update"] = [
        {
            "question": "When does my sprint end?",
            "answer": "My sprint ends on March 18.",
            "difficulty": "easy",
            "source_chat_ids": [0],
            "rubric": ["LLM response should state March 18"],
        }
    ]
    record["probing_questions"] = repr(probing_questions)

    samples = convert_beam_records_to_transparency([record])
    sample = next(
        s for s in samples if s["source"]["question_type"] == "Knowledge Update"
    )

    assert sample["expected_facts"] == []
    primed = " ".join(m["content"] for m in sample["target_memories"])
    assert "March 18" not in primed


def test_beam_adapter_config_loads_target_memory_source_limits() -> None:
    cfg = load_beam_transparency_adapter_config()
    assert cfg.default_target_source_limit == 6
    assert cfg.extended_target_source_limit == 10
    assert "knowledge_update" in cfg.extended_target_question_types

    with pytest.raises(ValueError, match="extended_target_question_types"):
        BeamTransparencyAdapterConfig(extended_target_question_types=[])


def test_beam_pilot_fixture_loads_with_source_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "beam_100k_pilot.json"
    payload = _beam_fixture_payload()
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    assert len(payload) == 8
    assert all(sample["source"]["dataset"] == "BEAM" for sample in payload)
    assert all(sample["source"]["split"] == "100K" for sample in payload)
    assert all(sample["source"]["token_scale"] == "100K" for sample in payload)
    assert all(sample["source"]["source_ids"] for sample in payload)
    assert {sample["source"]["question_type"] for sample in payload} >= {
        "Abstention",
        "Contradiction Resolution",
        "Information Extraction",
        "Knowledge Update",
        "Preference Following",
        "Summarization",
        "Temporal Reasoning",
        "Event Ordering",
    }

    for sample in payload:
        if sample["type"] == "negative_audit":
            assert sample["expected_facts"] == []
            continue
        primed = " ".join(m["content"] for m in sample["target_memories"]).lower()
        for fact in sample["expected_facts"]:
            normalized = fact.lower().rstrip(".")
            assert normalized in primed or any(
                token in primed
                for token in normalized.split()
                if len(token) >= 4 or any(c.isdigit() for c in token)
            )
        if sample["type"] != "full_memory_export":
            assert sample["prohibited_facts"]

    samples = load_transparency_samples(dataset)

    assert len(samples) == 8
    assert samples[0].source is not None
    assert samples[0].source["dataset"] == "BEAM"
    assert samples[0].source["split"] == "100K"


def test_benchmark_yaml_validates_into_transparency_config() -> None:
    payload = yaml.safe_load(
        Path("configs/benchmarks/testing/transparency_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = TypeAdapter(BenchmarkConfig).validate_python(payload)
    assert isinstance(config, TransparencyBenchmarkConfig)
    assert config.source.type == "beam"
    assert config.source.materialization_limit == 2
    assert config.limit == 2
    assert "{question}" in config.audit_prompt


def test_transparency_config_requires_question_placeholder() -> None:
    with pytest.raises(ValueError, match="audit_prompt"):
        load_transparency_benchmark_config(audit_prompt="no placeholder here")


def test_transparency_config_validates_beam_source() -> None:
    payload = yaml.safe_load(
        Path("configs/benchmarks/full_campaign/transparency.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = TypeAdapter(BenchmarkConfig).validate_python(payload)
    assert isinstance(config, TransparencyBenchmarkConfig)
    assert config.source.type == "beam"
    assert config.source.adapter.default_target_source_limit == 6
    assert "event_ordering" in config.source.adapter.extended_target_question_types


def test_factory_creates_transparency_benchmark() -> None:
    config = load_transparency_benchmark_config(limit=1)
    benchmark = create_benchmark(config)
    assert isinstance(benchmark, TransparencyBenchmark)
    assert isinstance(benchmark, BaseBenchmark)
    assert len(benchmark) == 1
    assert list(benchmark)[0].id


def test_transparency_implements_materialized_benchmark_contract() -> None:
    config = load_transparency_benchmark_config(limit=2)
    benchmark = create_benchmark(config)
    assert len(benchmark.samples) == len(benchmark)
    assert [sample.id for sample in benchmark] == [s.id for s in benchmark.samples]
    assert benchmark.skipped == []


def test_sample_to_scenario_carries_ground_truth_and_evaluator() -> None:
    config = load_transparency_benchmark_config(limit=1)
    sample = create_benchmark(config).samples[0]
    scenario = sample.to_scenario()
    assert isinstance(scenario, Scenario)
    assert scenario.ground_truth is not None
    assert scenario.evaluators[0].type == "transparency_llm_judge"
    assert sample.ground_truth == scenario.ground_truth
    assert "{question}" not in scenario.execution_sequence[0].content


def test_transparency_benchmark_applies_limit(tmp_path: Path) -> None:
    dataset = tmp_path / "many.json"
    dataset.write_text(json.dumps(_fixture_payload() * 3), encoding="utf-8")
    config = load_transparency_benchmark_config(
        source={"type": "json", "path": str(dataset)},
        limit=1,
    )
    assert len(create_benchmark(config)) == 1


def test_transparency_beam_materialization_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_load(*, split: str) -> list[dict]:
        seen["download_split"] = split
        return [{}]

    def fake_convert(
        records: list[dict],
        *,
        split: str,
        token_scale: str,
        limit: int | None = None,
        adapter: BeamTransparencyAdapterConfig | None = None,
    ) -> list[dict]:
        seen.update(
            {
                "record_count": len(records),
                "convert_split": split,
                "token_scale": token_scale,
                "limit": limit,
                "adapter_default": None
                if adapter is None
                else adapter.default_target_source_limit,
            }
        )
        return _beam_fixture_payload(limit=1)

    monkeypatch.setattr(
        "heams.benchmarks.transparency.dataset.load_beam_records",
        fake_load,
    )
    monkeypatch.setattr(
        "heams.benchmarks.transparency.dataset.convert_beam_records_to_transparency",
        fake_convert,
    )
    config = load_transparency_benchmark_config(
        source={
            "type": "beam",
            "split": "500K",
            "materialization_limit": None,
            "adapter": {
                "default_target_source_limit": 6,
                "extended_target_source_limit": 10,
                "extended_target_question_types": [
                    "contradiction_resolution",
                    "knowledge_update",
                    "event_ordering",
                ],
            },
        },
        limit=1,
    )
    benchmark = create_benchmark(config)
    assert len(benchmark) == 1
    assert seen["download_split"] == "500K"
    assert seen["convert_split"] == "500K"
    assert seen["token_scale"] == "500K"
    assert seen["record_count"] == 1
    assert seen["limit"] is None
    assert seen["adapter_default"] == 6


def test_transparency_beam_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> list[dict]:
        raise ValueError("simulated hub failure")

    monkeypatch.setattr(
        "heams.benchmarks.transparency.dataset.load_beam_records",
        boom,
    )
    with pytest.raises(BenchmarkDatasetUnavailableError) as excinfo:
        materialize_beam_or_raise(split="100K")
    assert "uv run heams run" in str(excinfo.value)


def test_agent_yaml_owns_transparency_policy() -> None:
    agent = yaml.safe_load(
        Path(
            "configs/agents/testing/transparency_gpt4o_mini_recall_only_tool_use.yaml"
        ).read_text(encoding="utf-8")
    )
    memory = yaml.safe_load(
        Path(
            "configs/memory_systems/testing/basic_complete_memory_system_transparency.yaml"
        ).read_text(encoding="utf-8")
    )
    assert agent["memory_strategy"] == "recall_only_tool_use"
    assert agent["instructions"].startswith("You are evaluating transparent access")
    assert TRANSPARENCY_AGENT_INSTRUCTIONS in agent["instructions"] or agent[
        "instructions"
    ].startswith("You are evaluating")
    assert memory["backend_config"]["top_k"] == 50
