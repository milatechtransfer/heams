from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from heams.benchmarks.base import BaseBenchmark, BenchmarkDatasetUnavailableError
from heams.benchmarks.erasability_editability import EEBenchmark
from heams.benchmarks.erasability_editability.config import (
    EEBenchmarkConfig,
    EESourceConfig,
)
from heams.benchmarks.erasability_editability.judge import (
    EELlmJudgeConfig,
    EEQaGroundTruth,
)
from heams.benchmarks.erasability_editability.samples import (
    CHANGE_DELETE_EVIDENCE_TYPE,
    episode_to_benchmark_sample,
    load_ee_episodes,
)
from heams.benchmarks.factory import BenchmarkConfig, create_benchmark
from tests.benchmarks.erasability_editability.fixtures import (
    CONFIG_DIR,
    EXPECTED_PROBE_KEYS,
    FIXTURE,
    FULL_CONFIG,
    _benchmark_config,
    _fixture_episode,
    _scenario,
)

SMOKE_CONFIG = "testing/erasability_editability_smoke.yaml"

SMOKE_RUN = "testing/erasability_editability_smoke.yaml"

TESTING_AGENT = "testing/erasability_editability_gpt4o_mini_tool_use.yaml"


def _fixture_payload(mutate=None) -> list[dict]:
    payload = json.loads(FIXTURE.read_text())
    if mutate is not None:
        mutate(payload[0])
    return payload


def _write(tmp_path: Path, payload: list[dict]) -> Path:
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_shipped_yaml_configs_validate() -> None:
    for name in (FULL_CONFIG, SMOKE_CONFIG):
        config = TypeAdapter(BenchmarkConfig).validate_python(
            yaml.safe_load((CONFIG_DIR / "benchmarks" / name).read_text())
        )
        assert config.type == "ee"
        assert config.evaluator.type == "ee_llm_judge"
    smoke = TypeAdapter(BenchmarkConfig).validate_python(
        yaml.safe_load((CONFIG_DIR / "benchmarks" / SMOKE_CONFIG).read_text())
    )
    # The exact value is a knob; what matters is that the smoke config stays
    # bounded, so it can never turn into an accidental full-corpus run.
    assert smoke.limit is not None and smoke.limit <= 5, "smoke config must stay cheap"


def test_smoke_config_prompts_match_full_config() -> None:
    """The loader has no include mechanism, so the prompts are duplicated.

    Assert they stay byte-identical rather than letting the two files drift.
    """
    full, smoke = (
        yaml.safe_load((CONFIG_DIR / "benchmarks" / name).read_text())
        for name in (FULL_CONFIG, SMOKE_CONFIG)
    )
    assert full["probe_prompt"] == smoke["probe_prompt"]
    assert full["evaluator"] == smoke["evaluator"]


def test_shipped_yaml_stays_minimal() -> None:
    """Keep the config surface in line with the other benchmarks.

    The house shape is `type / [limit] / source / <one>_prompt / evaluator`.
    `task_types` is the one EE-specific addition, because MEME packs six task
    kinds into a single record and we grade two. Anything else belongs in a
    Python default, not in every shipped config.
    """
    for name, extra in ((FULL_CONFIG, set()), (SMOKE_CONFIG, {"limit"})):
        keys = set(yaml.safe_load((CONFIG_DIR / "benchmarks" / name).read_text()))
        assert (
            keys
            == {
                "type",
                "source",
                "task_types",
                "probe_prompt",
                "evaluator",
            }
            | extra
        ), name


def test_shipped_run_configs_validate() -> None:
    from heams.orchestration.run_config import RunConfig

    path = (CONFIG_DIR / "runs" / SMOKE_RUN).resolve()
    assert RunConfig.load(path).run_name == "erasability_editability_smoke"

    # The campaign runs live one per memory-system campaign folder.
    for campaign in (
        "no_memory",
        "in_context",
        "mem0",
        "milamem",
    ):
        path = (
            CONFIG_DIR
            / "runs"
            / f"{campaign}_all_benchmarks"
            / "erasability_editability.yaml"
        ).resolve()
        assert RunConfig.load(path).run_name == (
            f"{campaign}_all_benchmarks_erasability_editability"
        )


def test_memory_arms_can_reach_the_delete_tool() -> None:
    """EE is meaningless if the agent cannot call `delete` during the replay."""
    from unittest.mock import MagicMock

    from heams.agents.config import MemoryStrategy
    from heams.tools.builders import memory_tool_specs

    agent_yaml = yaml.safe_load((CONFIG_DIR / "agents" / TESTING_AGENT).read_text())
    strategy = MemoryStrategy(agent_yaml["memory_strategy"])
    # `allows_memory_writes` is what gates recall_only in build_tools().
    assert strategy.uses_memory_tools
    assert strategy.allows_memory_writes, (
        "a recall_only strategy hides the delete tool; EE requires the full set"
    )

    specs = memory_tool_specs(
        MagicMock(), recall_only=not strategy.allows_memory_writes
    )
    assert "delete" in {spec.name for spec in specs}


def test_prompts_have_no_python_defaults() -> None:
    """Prompts are authored in YAML; omitting them must fail loudly."""
    with pytest.raises(ValidationError):
        EEBenchmarkConfig.model_validate({"type": "ee"})

    evaluator = _benchmark_config().evaluator
    for required in ("before_prompt", "deletion_prompt", "cascade_prompt"):
        payload = evaluator.model_dump()
        payload.pop(required)
        with pytest.raises(ValidationError):
            EELlmJudgeConfig.model_validate(payload)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _benchmark_config(nonexistent_setting=1)


def test_probe_prompt_requires_placeholder() -> None:
    with pytest.raises(ValidationError, match="probe_prompt"):
        _benchmark_config(probe_prompt="no placeholder here")


def test_replay_prompt_defaults_to_identity_and_still_validates() -> None:
    """Not in the shipped YAML, so the default has to be the correct one.

    Identity matters: the agent must see the deletion request verbatim, since
    any wrapper text would cue it that the turn is special.
    """
    assert _benchmark_config().replay_prompt == "{turn}"

    # Overriding it explicitly is still validated.
    with pytest.raises(ValidationError, match="replay_prompt"):
        _benchmark_config(replay_prompt="no placeholder here")


def test_task_types_reject_out_of_scope_values() -> None:
    with pytest.raises(ValidationError, match="Unsupported EE task_types"):
        _benchmark_config(task_types=["erase", "Abs"])
    with pytest.raises(ValidationError):
        _benchmark_config(task_types=[])


def test_prompt_templates_expose_expected_placeholders() -> None:
    config = _benchmark_config()
    assert "{query}" in config.probe_prompt
    assert "{turn}" in config.replay_prompt

    # Every rubric takes the reference implementation's three fields.
    for name in ("before_prompt", "deletion_prompt", "cascade_prompt"):
        rubric = getattr(config.evaluator, name)
        for placeholder in ("{question}", "{gold}", "{agent_answer}"):
            assert placeholder in rubric, f"{name} missing {placeholder}"
        assert '"correct"' in rubric, f"{name} must request the correct/reason JSON"

    # Rules we specifically ported from the reference and previously lacked.
    assert "BOTH old and new" in config.evaluator.cascade_prompt
    assert "future tense" in config.evaluator.cascade_prompt
    assert "even while saying it was deleted" in config.evaluator.deletion_prompt


def test_source_defaults_to_huggingface() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "benchmarks" / FULL_CONFIG).read_text())
    config = EEBenchmarkConfig.model_validate(raw)  # shipped source (HF)
    assert isinstance(config.source, EESourceConfig)
    assert config.source.repo_id == "meme-benchmark/MEME"
    assert config.source.filename == "meme_filler32k.json"
    assert config.source.local_path is None
    assert config.source.auto_download is True


def test_fixture_is_published_schema() -> None:
    episode = _fixture_episode()
    assert episode.episode_id == "pl_001"
    assert episode.domain == "personal_life"
    assert episode.root == "health_condition"
    assert episode.root_change.before == "lactose intolerance"
    assert episode.root_change.after == "high blood pressure"
    change_sessions = [
        s for s in episode.sessions if s.evidence_type == CHANGE_DELETE_EVIDENCE_TYPE
    ]
    assert len(change_sessions) == 1
    # Filler sessions carry no evidence_type and must still parse.
    assert any(s.type == "filler" and s.evidence_type is None for s in episode.sessions)


def test_gold_answer_alias_resolves_on_both_question_blocks() -> None:
    """The before block names it `expected_answer`, the after block `gold_answer`."""
    raw = json.loads(FIXTURE.read_text())[0]
    assert "expected_answer" in raw["before_questions"]["questions"][0]
    assert "gold_answer" in raw["after_questions"]["questions"][0]

    episode = _fixture_episode()
    assert all(q.gold_answer for q in episode.before_questions.questions)
    assert all(q.gold_answer for q in episode.after_questions.questions)


def test_episode_becomes_one_three_phase_scenario() -> None:
    episode = _fixture_episode()
    sample = episode_to_benchmark_sample(episode, config=_benchmark_config())

    assert sample.id == "pl_001"
    assert sample.tags == ["ee", "domain:personal_life", "task:edit", "task:erase"]

    phases = [
        (step.step_evaluation_markers or {}).get("phase", "replay")
        for step in sample.execution_sequence
    ]
    # 3 before-probes -> 4 replayed change+delete turns -> 3 after-probes.
    assert phases == ["before"] * 3 + ["replay"] * 4 + ["after"] * 3

    keys = [
        (step.step_evaluation_markers or {}).get("probe_key")
        for step in sample.execution_sequence
    ]
    assert [k for k in keys if k] == EXPECTED_PROBE_KEYS


@pytest.mark.parametrize("conversation", ["stateful", "independent"])
def test_agent_conversation_reaches_the_scenario(conversation: str) -> None:
    config = _benchmark_config(agent_conversation=conversation)

    scenario = episode_to_benchmark_sample(
        _fixture_episode(), config=config
    ).to_scenario()

    assert scenario.agent_conversation == conversation


def test_agent_conversation_defaults_to_stateful() -> None:
    assert _benchmark_config().agent_conversation == "stateful"


def test_replay_steps_carry_the_deletion_request_and_no_markers() -> None:
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    replay = [
        step
        for step in sample.execution_sequence
        if not (step.step_evaluation_markers or {}).get("probe_key")
    ]
    assert len(replay) == 4
    assert all(not step.step_evaluation_markers for step in replay)
    # The whole point of replaying: the agent must see the deletion request live.
    assert any("remove that from your memory" in step.content for step in replay)
    # ...and the root change that the cascade probes depend on.
    assert any("high blood pressure" in step.content for step in replay)


def test_change_session_is_not_primed_into_memory() -> None:
    episode = _fixture_episode()
    sample = episode_to_benchmark_sample(episode, config=_benchmark_config())
    change_index = next(
        index
        for index, session in enumerate(episode.sessions)
        if session.evidence_type == CHANGE_DELETE_EVIDENCE_TYPE
    )

    primed = {
        entry.metadata["session_index"]
        for entry in sample.scenario_priming.memory.entries
    }
    assert change_index not in primed
    assert primed == set(range(len(episode.sessions))) - {change_index}

    # Priming the delete turn would defeat the benchmark entirely.
    assert not any(
        "remove that from your memory" in entry.content
        for entry in sample.scenario_priming.memory.entries
    )


def test_priming_metadata_and_retrieval_evidence() -> None:
    episode = _fixture_episode()
    sample = episode_to_benchmark_sample(episode, config=_benchmark_config())

    entry = sample.scenario_priming.memory.entries[0]
    assert entry.metadata["source_benchmark"] == "ee"
    assert entry.metadata["scenario_id"] == "pl_001"
    assert entry.metadata["domain"] == "personal_life"
    assert entry.metadata["role"] in {"user", "assistant"}
    assert "timestamp" in entry.metadata

    evidence = set(episode.evidence_session_indices)
    flagged = {
        e.metadata["session_index"]
        for e in sample.scenario_priming.memory.entries
        if e.metadata.get("relevant_session")
    }
    assert flagged <= evidence

    markers = sample.execution_sequence[0].step_evaluation_markers
    assert markers["retrieval_evidence"] == {
        "match_key": "session_index",
        "relevant_values": episode.evidence_session_indices,
    }


def test_out_of_scope_task_types_are_filtered() -> None:
    """The fixture episode also carries Abs / Tr / Agg / ER questions."""
    episode = _fixture_episode()
    raw_types = {q.task_type for q in episode.after_questions.questions}
    assert {"Abs", "Tr", "Agg", "ER"} <= raw_types, "fixture must exercise filtering"

    sample = episode_to_benchmark_sample(episode, config=_benchmark_config())
    graded = {qa.task_type for qa in sample.ground_truth.qa}
    assert graded == {"erase", "edit"}


def test_before_probes_are_unconditional() -> None:
    """The before phase is load-bearing, so there is no knob to switch it off.

    Trivial-pass filtering joins each after-probe to the before-probe for the
    same entity. With no before-probes every after-probe would fall into the
    `trivial` bucket and the score would be a silent zero, so the config must
    not offer a way to skip them.
    """
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    assert {qa.phase for qa in sample.ground_truth.qa} == {"before", "after"}

    # `extra="forbid"` means the removed knob is now a hard config error rather
    # than a silently ignored field.
    with pytest.raises(ValidationError):
        _benchmark_config(ask_before_questions=False)


def test_post_change_filler_drop_excludes_trailing_sessions() -> None:
    episode = _fixture_episode()
    change_index = next(
        index
        for index, session in enumerate(episode.sessions)
        if session.evidence_type == CHANGE_DELETE_EVIDENCE_TYPE
    )
    assert change_index < len(episode.sessions) - 1, "fixture needs trailing filler"

    sample = episode_to_benchmark_sample(
        episode, config=_benchmark_config(post_change_filler="drop")
    )
    primed = {
        entry.metadata["session_index"]
        for entry in sample.scenario_priming.memory.entries
    }
    assert max(primed) < change_index


def test_response_index_points_at_the_right_stimulus() -> None:
    """Replay turns sit between the phases, so probes are not 1:1 with responses."""
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    for qa in sample.ground_truth.qa:
        stimulus = sample.execution_sequence[qa.response_index]
        assert stimulus.step_evaluation_markers["probe_key"] == qa.probe_key


def test_before_and_after_probes_pair_up_by_entity() -> None:
    """Trivial-pass filtering joins the two phases on entity, as MEME does."""
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    by_phase = {"before": [], "after": []}
    for qa in sample.ground_truth.qa:
        by_phase[qa.phase].append(qa)

    assert len(by_phase["before"]) == len(by_phase["after"])
    before_by_entity = {qa.entity: qa for qa in by_phase["before"]}
    assert len(before_by_entity) == len(by_phase["before"]), "entities must be unique"

    for after in by_phase["after"]:
        before = before_by_entity[after.entity]  # the join the evaluator performs
        assert before.task_type == after.task_type
        assert before.question == after.question
        # The entity is the bare key, not a joined list — the join depends on it.
        assert "," not in after.entity


def test_deletion_gold_answers_flip_across_the_change() -> None:
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    dels = [qa for qa in sample.ground_truth.qa if qa.task_type == "erase"]
    before, after = dels
    assert before.gold_answer == "James"
    assert "removed" in after.gold_answer


def test_gold_value_carries_the_entity_value_not_the_gold_answer_string() -> None:
    """The deletion rubric needs the value that must NOT be revealed.

    The reference passes ``entity_values[entity]`` as GOLD. For every edit probe
    that equals ``gold_answer``, but for an erase after-probe ``gold_answer`` is the
    constant "No - explicitly removed" while the value is the deleted fact.
    """
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())

    deletion_after = next(
        qa
        for qa in sample.ground_truth.qa
        if qa.task_type == "erase" and qa.phase == "after"
    )
    assert deletion_after.gold_value == "James"
    assert deletion_after.gold_value != deletion_after.gold_answer

    for qa in sample.ground_truth.qa:
        assert qa.gold_value, f"{qa.probe_key} has no gold_value"
        if qa.task_type == "edit":
            assert qa.gold_value == qa.gold_answer


def test_cascade_probes_carry_hops() -> None:
    sample = episode_to_benchmark_sample(_fixture_episode(), config=_benchmark_config())
    hops = {qa.hop for qa in sample.ground_truth.qa if qa.task_type == "edit"}
    assert hops == {1, 2}
    assert all(
        qa.hop is None for qa in sample.ground_truth.qa if qa.task_type == "erase"
    )


def test_load_accepts_wrapped_payload() -> None:
    raw = json.loads(FIXTURE.read_text())
    assert load_ee_episodes({"data": raw})[0].episode_id == "pl_001"
    assert load_ee_episodes({"episodes": raw})[0].episode_id == "pl_001"
    with pytest.raises(ValueError, match="must be a list"):
        load_ee_episodes({"nope": 1})


def test_episode_without_change_session_is_skipped(tmp_path: Path) -> None:
    def drop_change(episode: dict) -> None:
        episode["sessions"] = [
            s
            for s in episode["sessions"]
            if s.get("evidence_type") != CHANGE_DELETE_EVIDENCE_TYPE
        ]

    path = _write(tmp_path, _fixture_payload(drop_change))
    benchmark = create_benchmark(_benchmark_config(source={"local_path": str(path)}))

    assert list(benchmark.samples) == []
    assert len(benchmark.skipped) == 1
    assert "exactly one" in benchmark.skipped[0].reason


def test_episode_without_in_scope_probes_is_skipped(tmp_path: Path) -> None:
    def only_out_of_scope(episode: dict) -> None:
        for block in ("before_questions", "after_questions"):
            episode[block]["questions"] = [
                q for q in episode[block]["questions"] if q["task_type"] == "Abs"
            ]

    path = _write(tmp_path, _fixture_payload(only_out_of_scope))
    benchmark = create_benchmark(_benchmark_config(source={"local_path": str(path)}))

    assert list(benchmark.samples) == []
    assert "no after-probes" in benchmark.skipped[0].reason


def test_episode_with_empty_change_session_is_skipped(tmp_path: Path) -> None:
    def blank_change(episode: dict) -> None:
        for session in episode["sessions"]:
            if session.get("evidence_type") == CHANGE_DELETE_EVIDENCE_TYPE:
                session["conversation"] = [
                    {"role": "assistant", "content": "Hi! What's new?"}
                ]

    path = _write(tmp_path, _fixture_payload(blank_change))
    benchmark = create_benchmark(_benchmark_config(source={"local_path": str(path)}))

    assert list(benchmark.samples) == []
    assert "no user turns" in benchmark.skipped[0].reason


def test_factory_materializes_fixture() -> None:
    benchmark = create_benchmark(_benchmark_config())
    assert isinstance(benchmark, EEBenchmark)
    assert isinstance(benchmark, BaseBenchmark)
    assert len(benchmark.samples) == 1
    assert benchmark.skipped == []


def test_limit_applies_to_source_episodes(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text())
    second = json.loads(json.dumps(payload[0]))
    second["episode_id"] = "pl_002"
    path = _write(tmp_path, [payload[0], second])

    unlimited = create_benchmark(_benchmark_config(source={"local_path": str(path)}))
    assert len(unlimited.samples) == 2

    limited = create_benchmark(
        _benchmark_config(source={"local_path": str(path)}, limit=1)
    )
    assert [sample.id for sample in limited.samples] == ["pl_001"]


def test_to_scenario_carries_ground_truth_and_evaluator() -> None:
    scenario = _scenario()
    assert isinstance(scenario.ground_truth, EEQaGroundTruth)
    assert len(scenario.ground_truth.qa) == 6
    assert scenario.evaluators[0].type == "ee_llm_judge"


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
        source={"repo_id": "meme-benchmark/MEME", "filename": "meme_filler32k.json"}
    )
    benchmark = create_benchmark(config)

    assert len(benchmark.samples) == 1
    assert seen["repo_id"] == "meme-benchmark/MEME"
    assert seen["repo_type"] == "dataset"
    assert seen["filename"] == "meme_filler32k.json"
