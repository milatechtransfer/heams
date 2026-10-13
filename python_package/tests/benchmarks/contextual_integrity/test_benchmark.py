from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from heams.benchmarks.base import BenchmarkDatasetUnavailableError
from heams.benchmarks.contextual_integrity.benchmark import CIBenchmark
from heams.benchmarks.contextual_integrity.config import CIBenchmarkConfig
from heams.benchmarks.contextual_integrity.judge import CILlmJudgeConfig
from heams.benchmarks.factory import create_benchmark
from heams.utils.settings import settings
from tests.memory_plugin_requirements import skip_if_run_config_memory_plugins_missing
from tests.benchmarks.contextual_integrity.fixtures import (
    CONFIG_DIR,
    FIXTURE,
    _benchmark_config,
)


def test_shipped_yaml_configs_validate():
    for config_path in (
        CONFIG_DIR / "benchmarks" / "full_campaign" / "contextual_integrity.yaml",
        CONFIG_DIR / "benchmarks" / "testing" / "contextual_integrity_smoke.yaml",
    ):
        config = CIBenchmarkConfig.model_validate(
            yaml.safe_load(config_path.read_text())
        )
        assert config.type == "ci"
        assert config.evaluator.type == "ci_llm_judge"
    smoke = CIBenchmarkConfig.model_validate(
        yaml.safe_load(
            (
                CONFIG_DIR
                / "benchmarks"
                / "testing"
                / "contextual_integrity_smoke.yaml"
            ).read_text()
        )
    )
    assert smoke.limit == 1, "smoke config must stay cheap"
    assert smoke.dataset_source.download_limit == 2, (
        "smoke config must limit dataset bootstrap"
    )


def test_prompts_have_no_python_defaults():
    """Prompts are authored in YAML; omitting them must fail loudly."""
    with pytest.raises(ValidationError):
        CIBenchmarkConfig.model_validate({"type": "ci"})

    base = _benchmark_config()
    for required in (
        "leakage_system_prompt",
        "coverage_system_prompt",
        "helpfulness_system_prompt",
        "leakage_prompt_template",
        "coverage_prompt_template",
        "helpfulness_prompt_template",
    ):
        payload = base.evaluator.model_dump()
        payload.pop(required)
        with pytest.raises(ValidationError):
            CILlmJudgeConfig.model_validate(payload)


def test_examples_dir_defaults_under_settings_datasets_dir(tmp_path):
    """DATASETS_DIR relocates the corpus, and an explicit path still wins."""
    raw = yaml.safe_load(
        (
            CONFIG_DIR / "benchmarks" / "full_campaign" / "contextual_integrity.yaml"
        ).read_text()
    )
    assert "examples_dir" not in raw
    default = CIBenchmarkConfig.model_validate(raw)
    assert default.examples_dir == settings.datasets_dir / "contextual_integrity"
    assert _benchmark_config(examples_dir=str(tmp_path)).examples_dir == tmp_path


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _benchmark_config(nonexistent_setting=1)


def test_user_task_prompt_requires_placeholder():
    """A stimulus template without {user_task} would silently drop the task."""
    with pytest.raises(ValidationError, match="user_task"):
        _benchmark_config(user_task_prompt="do the task")


def test_config_holds_no_agent_or_memory_policy():
    """Agent/memory settings belong in their own component files."""
    fields = set(CIBenchmarkConfig.model_fields)
    assert not fields & {"instructions", "model_args", "top_k", "memory_strategy"}


def test_prompt_templates_expose_expected_placeholders():
    evaluator = _benchmark_config().evaluator
    assert "{secret}" in evaluator.leakage_prompt_template
    assert "{action}" in evaluator.leakage_prompt_template
    assert "{info}" in evaluator.coverage_prompt_template
    assert "{user_instruction}" in evaluator.helpfulness_prompt_template
    assert "{trajectory}" in evaluator.helpfulness_prompt_template


def test_factory_dispatches_ci_type():
    benchmark = create_benchmark(_benchmark_config())
    assert isinstance(benchmark, CIBenchmark)


def test_run_config_resolves_ci_experiments():
    from heams.orchestration.run_config import resolve_run_config
    from heams.orchestration.run_orchestration import build_experiment_execution_configs

    resolved = resolve_run_config(
        CONFIG_DIR / "runs" / "testing" / "contextual_integrity_smoke.yaml"
    )
    skip_if_run_config_memory_plugins_missing(
        [
            experiment.memory_system_config_file.path
            for experiment in resolved.experiments
        ]
    )
    execution_configs = build_experiment_execution_configs(resolved)

    assert execution_configs, "ci_smoke run config must yield experiments"
    for execution_config in execution_configs:
        assert execution_config.benchmark_config.type == "ci"
        # The agent timeout the legacy benchmark used to force onto the config
        # is now an agent component setting, resolved from the agent YAML.
        assert execution_config.agent_config.model_args["timeout"] == 120
        # Memory top_k now comes from the shared memory component config, not a
        # benchmark override (the shared configs use 5).
        assert execution_config.memory_system_config.backend_config.top_k == 5


def test_materializes_samples_from_fixture():
    benchmark = create_benchmark(_benchmark_config())
    assert len(benchmark) == 2
    assert benchmark.skipped == []

    sample = benchmark.samples[0]
    assert sample.id == "CI-train-0000"
    assert "ci" in sample.tags
    assert len(sample.execution_sequence) == 1
    assert sample.evaluators == [benchmark.config.evaluator]


def test_primed_memory_never_reveals_the_disclosure_label():
    """Memory metadata reaches the agent; it must not leak the answer key."""
    benchmark = create_benchmark(_benchmark_config())
    for sample in benchmark.samples:
        entries = sample.scenario_priming.memory.entries
        assert entries, "information pool must be primed as memories"
        for entry in entries:
            assert set(entry.metadata) == {"field_name"}


def test_ground_truth_markers_are_eval_only():
    benchmark = create_benchmark(_benchmark_config())
    sample = benchmark.samples[0]
    markers = sample.execution_sequence[0].step_evaluation_markers

    assert markers["allowed_disclosures"]
    assert markers["disallowed_disclosures"]
    # The stimulus the agent sees is the bare user task.
    assert "allowed" not in sample.execution_sequence[0].content
    assert "disallowed" not in sample.execution_sequence[0].content


def test_ground_truth_round_trips_through_the_evaluator_config():
    config = _benchmark_config()
    scenario = create_benchmark(config).samples[0].to_scenario()
    ground_truth = config.evaluator.to_ground_truth(scenario)

    assert len(ground_truth.tasks) == 1
    task = ground_truth.tasks[0]
    assert task.secrets, "fixture sample must carry disallowed items"
    assert task.allowed, "fixture sample must carry allowed items"
    assert len(task.items) == len(task.secrets) + len(task.allowed)


def test_helpfulness_trajectory_uses_full_pool_not_canonical_values():
    """Helpfulness sees what the agent saw; leak/coverage keep canonical values.

    The information pool holds full text (e.g. the whole MedicalNotes line);
    the annotation holds the canonical token (e.g. "RX2019"). The helpfulness
    judge's trajectory must reflect the full pool, while the leakage/coverage
    items keep the trimmed annotation value.
    """
    from heams.benchmarks.contextual_integrity.judge import _format_trajectory

    config = _benchmark_config()
    scenario = create_benchmark(config).samples[0].to_scenario()
    task = config.evaluator.to_ground_truth(scenario).tasks[0]

    full = "Currently taking RX2019 for hypertension"
    canonical = "RX2019"

    # The pool carries the full text the agent was primed with.
    assert task.information_pool["MedicalNotes"] == full
    # Leakage/coverage still route on the canonical annotation value.
    secret = next(item for item in task.secrets if item.field == "MedicalNotes")
    assert secret.value == canonical

    # The helpfulness trajectory shows the full pool line, not just the token.
    trajectory = _format_trajectory(task)
    assert f"- MedicalNotes: {full}" in trajectory
    assert f"- MedicalNotes: {canonical}\n" not in trajectory
    # Never labels items as allowed/disallowed.
    assert "allowed" not in trajectory
    assert "disallowed" not in trajectory


def test_trajectory_falls_back_to_items_without_information_pool():
    """Older artifacts have no information_pool; trajectory falls back to items."""
    from heams.benchmarks.contextual_integrity.judge import (
        CIDisclosureItem,
        ContextualIntegrityTask,
        _format_trajectory,
    )

    task = ContextualIntegrityTask(
        sample_id="x",
        user_task="t",
        items=[CIDisclosureItem(field="F", value="V", policy="allowed")],
    )
    assert _format_trajectory(task) == "- F: V"


def test_limit_truncates_samples():
    assert len(create_benchmark(_benchmark_config(limit=1))) == 1


def test_malformed_record_is_skipped_not_fatal(tmp_path):
    payload = json.loads(FIXTURE.read_text())
    dataset = tmp_path / "mixed.json"
    dataset.write_text(json.dumps([payload[0], {"id": "broken"}]))

    benchmark = create_benchmark(_benchmark_config(examples_dir=str(dataset)))
    assert len(benchmark) == 1
    assert len(benchmark.skipped) == 1
    assert "record 1" in benchmark.skipped[0].reason


def test_missing_dataset_without_auto_download_raises(tmp_path):
    with pytest.raises(BenchmarkDatasetUnavailableError, match="auto_download"):
        create_benchmark(
            _benchmark_config(
                examples_dir=str(tmp_path / "absent"),
                dataset_source={"auto_download": False},
            )
        )


def test_full_run_does_not_reuse_smoke_cache(tmp_path):
    """Smoke and full differ only by download_limit; their caches must not mix.

    A smoke run caches N=2 rows; a later full run (N=16) must not silently reuse
    the smoke file. Different download_limit -> different fixture file name, and
    the resolver looks for that specific file rather than any *.json in the dir.
    """
    from heams.benchmarks.contextual_integrity.dataset import resolve_dataset_paths

    smoke = _benchmark_config(
        examples_dir=str(tmp_path), dataset_source={"download_limit": 2}
    )
    full = _benchmark_config(
        examples_dir=str(tmp_path),
        dataset_source={"download_limit": 16, "auto_download": False},
    )
    assert smoke.fixture_name() != full.fixture_name()

    # Simulate the smoke run having cached its (small) fixture in the dir.
    (tmp_path / smoke.fixture_name()).write_text("[]", encoding="utf-8")
    assert resolve_dataset_paths(smoke) == [tmp_path / smoke.fixture_name()]

    # The full run must NOT pick up the smoke file; with auto_download off it
    # fails loudly instead of silently using 2 rows.
    with pytest.raises(BenchmarkDatasetUnavailableError):
        resolve_dataset_paths(full)
