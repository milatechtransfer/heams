from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from heams.benchmarks.base import BaseBenchmark, BenchmarkDatasetUnavailableError
from heams.benchmarks.factory import BenchmarkConfig, create_benchmark
from heams.benchmarks.stale import StaleBenchmark
from heams.benchmarks.stale.config import StaleBenchmarkConfig, StaleSourceConfig
from heams.benchmarks.stale.judge import (
    StaleLlmJudgeConfig,
    StaleQAGroundTruth,
)
from heams.benchmarks.stale.samples import (
    load_stale_samples,
    sample_to_benchmark_sample,
)
from heams.scenarios.base import Scenario
from tests.benchmarks.stale.fixtures import (
    CONFIG_DIR,
    FIXTURE,
    _benchmark_config,
    _fixture_sample,
)


def _flat_payload(**changes) -> list[dict]:
    """A tiny hand-authored record in the older flat shape (backward-compat)."""
    record = {
        "uid": "STALE-FLAT-1",
        "conflict_type": "T1",
        "M_old": "The user is vegetarian.",
        "M_new": "The user eats meat again.",
        "explanation": "M_new supersedes M_old.",
        "haystack_session": [
            {"role": "user", "content": "I'm vegetarian."},
            {"role": "user", "content": "Update: I eat meat again."},
        ],
        "probing_queries": {
            "dim1_query": "Q1",
            "dim2_query": "Q2",
            "dim3_query": "Q3",
        },
    }
    record.update(changes)
    return [record]


def test_shipped_yaml_configs_validate() -> None:
    for config_path in (
        CONFIG_DIR / "benchmarks" / "full_campaign" / "stale.yaml",
        CONFIG_DIR / "benchmarks" / "testing" / "stale_smoke.yaml",
    ):
        config = TypeAdapter(BenchmarkConfig).validate_python(
            yaml.safe_load(config_path.read_text())
        )
        assert config.type == "stale"
        assert config.evaluator.type == "stale_llm_judge"
    smoke = TypeAdapter(BenchmarkConfig).validate_python(
        yaml.safe_load(
            (CONFIG_DIR / "benchmarks" / "testing" / "stale_smoke.yaml").read_text()
        )
    )
    assert smoke.limit == 1, "smoke config must stay cheap"


def test_prompts_have_no_python_defaults() -> None:
    """Prompts are authored in YAML; omitting them must fail loudly."""
    with pytest.raises(ValidationError):
        StaleBenchmarkConfig.model_validate({"type": "stale"})

    evaluator = _benchmark_config().evaluator
    for required in ("judge_system_prompt", "judge_user_template"):
        payload = evaluator.model_dump()
        payload.pop(required)
        with pytest.raises(ValidationError):
            StaleLlmJudgeConfig.model_validate(payload)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _benchmark_config(nonexistent_setting=1)


def test_probe_prompt_requires_placeholder() -> None:
    with pytest.raises(ValidationError, match="probe_prompt"):
        _benchmark_config(probe_prompt="no placeholder here")


def test_prompt_templates_expose_expected_placeholders() -> None:
    config = _benchmark_config()
    assert "{query}" in config.probe_prompt
    for placeholder in (
        "{m_old}",
        "{m_new}",
        "{explanation}",
        "{dim1_question}",
        "{dim1_response}",
        "{dim2_question}",
        "{dim2_response}",
        "{dim3_question}",
        "{dim3_response}",
    ):
        assert placeholder in config.evaluator.judge_user_template
    assert "dim1_eval" in config.evaluator.judge_system_prompt


def test_source_defaults_to_huggingface() -> None:
    raw = yaml.safe_load(
        (CONFIG_DIR / "benchmarks" / "full_campaign" / "stale.yaml").read_text()
    )
    config = StaleBenchmarkConfig.model_validate(raw)  # shipped source (HF)
    assert isinstance(config.source, StaleSourceConfig)
    assert config.source.repo_id == "STALEproj/STALE"
    assert config.source.filename == "T1_T2_400_FULL.json"
    assert config.source.local_path is None
    assert config.source.auto_download is True


def test_fixture_is_published_schema() -> None:
    sample = _fixture_sample()
    assert sample.conflict_type == "T1"  # parsed from `type`
    assert len(sample.timestamps) == 50  # session-level timestamps
    assert sample.relevant_session_index == [13, 33]
    assert isinstance(sample.haystack_session[0], list)


def test_sample_becomes_one_scenario_with_three_stimuli() -> None:
    sample = _fixture_sample()
    bs = sample_to_benchmark_sample(sample, config=_benchmark_config())

    assert bs.id == sample.uid
    assert bs.tags == ["stale", "conflict:T1"]
    assert [s.step_evaluation_markers["dimension"] for s in bs.execution_sequence] == [
        "dim1",
        "dim2",
        "dim3",
    ]
    assert isinstance(bs.ground_truth, StaleQAGroundTruth)
    assert [q.dimension for q in bs.ground_truth.qa] == ["dim1", "dim2", "dim3"]
    assert bs.evaluators[0].type == "stale_llm_judge"

    # The shipped probe_prompt is substituted with each dimension query.
    assert "{query}" not in bs.execution_sequence[0].content
    assert sample.probing_queries.dim1_query in bs.execution_sequence[0].content


@pytest.mark.parametrize("conversation", ["stateful", "independent"])
def test_agent_conversation_reaches_the_scenario(conversation: str) -> None:
    config = _benchmark_config(agent_conversation=conversation)

    scenario = sample_to_benchmark_sample(
        _fixture_sample(), config=config
    ).to_scenario()

    assert scenario.agent_conversation == conversation


def test_agent_conversation_defaults_to_stateful() -> None:
    assert _benchmark_config().agent_conversation == "stateful"


def test_session_aware_priming_and_retrieval_evidence() -> None:
    bs = sample_to_benchmark_sample(_fixture_sample(), config=_benchmark_config())
    entries = bs.scenario_priming.memory.entries

    assert len(entries) > 0
    assert entries[0].metadata["session_index"] == 0
    assert entries[0].metadata["timestamp"] == "2020-04-24 18:15"  # session-level
    assert entries[0].metadata["conflict_type"] == "T1"

    relevant_sessions = {
        e.metadata["session_index"]
        for e in entries
        if e.metadata.get("relevant_session")
    }
    assert relevant_sessions == {13, 33}

    markers = bs.execution_sequence[0].step_evaluation_markers
    assert markers["relevant_session_index"] == [13, 33]
    assert markers["retrieval_evidence"]["match_key"] == "session_index"
    assert markers["retrieval_evidence"]["relevant_values"] == [13, 33]


def test_flat_haystack_and_conflict_type_key_load() -> None:
    sample = load_stale_samples(_flat_payload())[0]
    assert sample.conflict_type == "T1"  # from the `conflict_type` key

    bs = sample_to_benchmark_sample(sample, config=_benchmark_config())
    assert [e.content for e in bs.scenario_priming.memory.entries] == [
        "I'm vegetarian.",
        "Update: I eat meat again.",
    ]


def test_default_conflict_type_when_missing() -> None:
    payload = _flat_payload()
    del payload[0]["conflict_type"]
    bs = sample_to_benchmark_sample(
        load_stale_samples(payload)[0], config=_benchmark_config()
    )
    assert bs.tags == ["stale", "conflict:unknown"]


def test_factory_materializes_fixture() -> None:
    benchmark = create_benchmark(_benchmark_config())

    assert isinstance(benchmark, StaleBenchmark)
    assert isinstance(benchmark, BaseBenchmark)
    assert len(benchmark) == 1  # one record -> one scenario (three stimuli)
    assert benchmark.skipped == []
    assert len(benchmark.samples[0].execution_sequence) == 3


def test_limit_applies_to_source_records(tmp_path: Path) -> None:
    records = json.loads(FIXTURE.read_text())
    two = tmp_path / "two.json"
    two.write_text(json.dumps(records + [{**records[0], "uid": "dup"}]))

    assert (
        len(create_benchmark(_benchmark_config(source={"local_path": str(two)}))) == 2
    )
    limited = create_benchmark(
        _benchmark_config(source={"local_path": str(two)}, limit=1)
    )
    assert len(limited) == 1


def test_to_scenario_carries_ground_truth_and_evaluator() -> None:
    scenario = create_benchmark(_benchmark_config()).samples[0].to_scenario()
    assert isinstance(scenario, Scenario)
    assert scenario.ground_truth is not None
    assert len(scenario.ground_truth.qa) == 3
    assert scenario.evaluators[0].type == "stale_llm_judge"
    assert "{query}" not in scenario.execution_sequence[0].content


def test_missing_local_source_raises_dataset_unavailable(tmp_path: Path) -> None:
    config = _benchmark_config(source={"local_path": str(tmp_path / "missing.json")})
    with pytest.raises(
        BenchmarkDatasetUnavailableError, match="source.local_path not found"
    ):
        create_benchmark(config)


def test_auto_download_false_without_local_path_raises() -> None:
    config = _benchmark_config(source={"auto_download": False})
    with pytest.raises(BenchmarkDatasetUnavailableError, match="auto_download"):
        create_benchmark(config)


def test_hf_source_materializes_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    seen: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(FIXTURE)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    config = _benchmark_config(
        source={"repo_id": "STALEproj/STALE", "filename": "T1_T2_400_FULL.json"}
    )
    benchmark = create_benchmark(config)

    assert len(benchmark) == 1
    assert seen["repo_id"] == "STALEproj/STALE"
    assert seen["repo_type"] == "dataset"
    assert seen["filename"] == "T1_T2_400_FULL.json"
