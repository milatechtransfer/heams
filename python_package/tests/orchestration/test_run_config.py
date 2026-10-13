from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from heams.memory_systems import MemorySystemVariant
from heams.orchestration import RunConfig, resolve_run_config
from heams.utils.settings import settings
from tests.memory_plugin_requirements import skip_if_memory_plugin_missing


def test_run_config_load_reads_yaml_from_runs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    loaded = RunConfig.load(run_path)

    assert loaded.run_name == "example_campaign"
    assert len(loaded.experiments) == 1
    assert loaded.experiments[0].name == "gpt4o_mini_nomemory_locomo"
    assert (
        loaded.experiments[0].agent_config_file.file_name == "gpt4o_mini_tool_use.yaml"
    )


def test_resolve_run_config_adds_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    monkeypatch.setattr(settings, "project_dir", run_path.parents[2])

    resolved = resolve_run_config(run_path)

    assert resolved.source_config_path == run_path.resolve(strict=False)
    assert resolved.run_name == "example_campaign"
    assert resolved.results_dir == (run_path.parents[2] / "results").resolve()
    assert resolved.executed_at.tzinfo == timezone.utc
    assert resolved.run_id
    assert isinstance(resolved.experiments, tuple)
    assert resolved.experiments[0].name == "gpt4o_mini_nomemory_locomo"


def test_resolve_run_config_uses_explicit_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    output_dir = tmp_path / "campaign-results" / "example_campaign"

    resolved = resolve_run_config(run_path, output_dir=output_dir)

    assert resolved.results_dir == output_dir.resolve()


def test_run_config_validation_rejects_missing_component_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    with pytest.raises(ValidationError, match="Referenced config file does not exist"):
        RunConfig.model_validate(
            {
                "run_name": "missing-file",
                "experiments": [
                    {
                        "name": "missing-agent",
                        "agent_config_file": "missing.yaml",
                        "memory_system_config_file": "no_memory.yaml",
                        "benchmark_config_file": "locomo.yaml",
                    }
                ],
            }
        )


def test_run_config_rejects_results_directory_in_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])

    with pytest.raises(ValidationError, match="results_dir"):
        RunConfig.model_validate(
            {
                "run_name": "legacy-output-directory",
                "results_dir": "results/legacy-output-directory",
                "experiments": [
                    {
                        "name": "example",
                        "agent_config_file": "gpt4o_mini_tool_use.yaml",
                        "memory_system_config_file": "no_memory.yaml",
                        "benchmark_config_file": "locomo.yaml",
                    }
                ],
            }
        )


def test_run_config_validation_allows_nested_component_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    config_root = run_path.parents[1]
    for component_dir, file_name in (
        ("agents", "gpt4o_mini_tool_use.yaml"),
        ("memory_systems", "no_memory.yaml"),
        ("benchmarks", "locomo.yaml"),
    ):
        nested_dir = config_root / component_dir / "campaigns"
        nested_dir.mkdir()
        (nested_dir / file_name).write_text(
            (config_root / component_dir / file_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    monkeypatch.setattr(settings, "config_dir", config_root)

    config = RunConfig.model_validate(
        {
            "run_name": "nested-benchmark",
            "experiments": [
                {
                    "name": "nested-benchmark",
                    "agent_config_file": "campaigns/gpt4o_mini_tool_use.yaml",
                    "memory_system_config_file": "campaigns/no_memory.yaml",
                    "benchmark_config_file": "campaigns/locomo.yaml",
                }
            ],
        }
    )

    experiment = config.experiments[0]
    assert experiment.agent_config_file.file_name == (
        "campaigns/gpt4o_mini_tool_use.yaml"
    )
    assert experiment.memory_system_config_file.file_name == (
        "campaigns/no_memory.yaml"
    )
    assert experiment.benchmark_config_file.file_name == "campaigns/locomo.yaml"


@pytest.mark.parametrize(
    ("field_name", "file_name"),
    [
        ("agent_config_file", "../gpt4o_mini_tool_use.yaml"),
        ("memory_system_config_file", "../no_memory.yaml"),
        ("benchmark_config_file", "../locomo.yaml"),
        ("agent_config_file", "smoke_campaign/../gpt4o_mini_tool_use.yaml"),
        ("memory_system_config_file", "smoke_campaign/../no_memory.yaml"),
        ("benchmark_config_file", "smoke_campaign/../locomo.yaml"),
        ("agent_config_file", "/tmp/gpt4o_mini_tool_use.yaml"),
        ("memory_system_config_file", "/tmp/no_memory.yaml"),
        ("benchmark_config_file", "/tmp/locomo.yaml"),
    ],
)
def test_run_config_validation_rejects_unsafe_component_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    file_name: str,
) -> None:
    run_path = _write_fixture_tree(tmp_path)
    monkeypatch.setattr(settings, "config_dir", run_path.parents[1])
    experiment = {
        "name": "unsafe-component-path",
        "agent_config_file": "gpt4o_mini_tool_use.yaml",
        "memory_system_config_file": "no_memory.yaml",
        "benchmark_config_file": "locomo.yaml",
    }
    experiment[field_name] = file_name

    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "run_name": "unsafe-component-path",
                "experiments": [experiment],
            }
        )


@pytest.mark.parametrize(
    ("file_name", "run_name", "memory_file_name"),
    [
        ("no_memory.yaml", "smoke_no_memory", "no_memory.yaml"),
        ("in_context.yaml", "smoke_in_context", "in_context.yaml"),
        ("milamem.yaml", "smoke_milamem", "milamem.yaml"),
        ("mem0.yaml", "smoke_mem0", "mem0.yaml"),
    ],
)
def test_committed_smoke_run_config_loads(
    file_name: str,
    run_name: str,
    memory_file_name: str,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "runs"
        / "smoke_campaign"
        / file_name
    )

    resolved = resolve_run_config(config_path)

    assert resolved.run_name == run_name
    assert len(resolved.experiments) == 9
    assert {
        experiment.memory_system_config_file.file_name
        for experiment in resolved.experiments
    } == {f"campaigns/{memory_file_name}"}
    assert {
        experiment.agent_config_file.file_name for experiment in resolved.experiments
    } == {
        "campaigns/locomo.yaml",
        "campaigns/evomemory.yaml",
        "campaigns/long_mem_eval.yaml",
        "campaigns/transparency.yaml",
        "campaigns/contextual_integrity.yaml",
        "campaigns/stale.yaml",
        "campaigns/mpbench.yaml",
        "campaigns/erasability_editability.yaml",
        "campaigns/beam.yaml",
    }
    assert {
        experiment.benchmark_config_file.file_name
        for experiment in resolved.experiments
    } == {
        "smoke_campaign/locomo.yaml",
        "smoke_campaign/evomemory_single_turn.yaml",
        "smoke_campaign/long_mem_eval.yaml",
        "smoke_campaign/transparency.yaml",
        "smoke_campaign/contextual_integrity.yaml",
        "smoke_campaign/stale.yaml",
        "smoke_campaign/mpbench.yaml",
        "smoke_campaign/erasability_editability.yaml",
        "smoke_campaign/beam_100k.yaml",
    }
    for memory_config_path in {
        experiment.memory_system_config_file.path for experiment in resolved.experiments
    }:
        payload = yaml.safe_load(memory_config_path.read_text(encoding="utf-8"))
        skip_if_memory_plugin_missing(payload)
        MemorySystemVariant.model_validate(payload)


def test_smoke_test_benchmark_run_config_covers_all_smoke_benchmarks() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "runs"
        / "smoke_test_benchmark.yaml"
    )

    resolved = resolve_run_config(config_path)

    assert resolved.run_name == "smoke_test_benchmark"
    assert len(resolved.experiments) == 9
    assert len({experiment.name for experiment in resolved.experiments}) == 9
    assert {
        experiment.memory_system_config_file.file_name
        for experiment in resolved.experiments
    } == {"campaigns/in_context.yaml"}
    assert {
        experiment.benchmark_config_file.file_name
        for experiment in resolved.experiments
    } == {
        "smoke_campaign/locomo.yaml",
        "smoke_campaign/evomemory_single_turn.yaml",
        "smoke_campaign/long_mem_eval.yaml",
        "smoke_campaign/transparency.yaml",
        "smoke_campaign/contextual_integrity.yaml",
        "smoke_campaign/stale.yaml",
        "smoke_campaign/mpbench.yaml",
        "smoke_campaign/erasability_editability.yaml",
        "smoke_campaign/beam_100k.yaml",
    }


def test_smoke_test_memory_system_run_config_covers_campaign_memory_systems() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "runs"
        / "smoke_test_memory_system.yaml"
    )

    resolved = resolve_run_config(config_path)

    assert resolved.run_name == "smoke_test_memory_system"
    assert len(resolved.experiments) == 4
    assert len({experiment.name for experiment in resolved.experiments}) == 4
    assert {
        experiment.agent_config_file.file_name for experiment in resolved.experiments
    } == {"campaigns/locomo.yaml"}
    assert {
        experiment.memory_system_config_file.file_name
        for experiment in resolved.experiments
    } == {
        "campaigns/no_memory.yaml",
        "campaigns/in_context.yaml",
        "campaigns/milamem.yaml",
        "campaigns/mem0.yaml",
    }
    assert {
        experiment.benchmark_config_file.file_name
        for experiment in resolved.experiments
    } == {"smoke_campaign/locomo.yaml"}


def test_committed_run_configs_load() -> None:
    runs_dir = Path(__file__).resolve().parents[2] / "configs" / "runs"

    for config_path in sorted(runs_dir.rglob("*.yaml")):
        resolve_run_config(config_path)


def test_campaign_memory_defaults_are_consistent() -> None:
    memory_systems_dir = (
        Path(__file__).resolve().parents[2] / "configs" / "memory_systems"
    )
    campaign_mem0 = yaml.safe_load(
        (memory_systems_dir / "campaigns" / "mem0.yaml").read_text(encoding="utf-8")
    )
    testing_no_memory = yaml.safe_load(
        (memory_systems_dir / "testing" / "no_memory.yaml").read_text(encoding="utf-8")
    )
    campaign_no_memory = yaml.safe_load(
        (memory_systems_dir / "campaigns" / "no_memory.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert campaign_mem0["backend_config"]["temporal_prompt_enabled"] is True
    assert "top_k" not in testing_no_memory["backend_config"]
    assert "top_k" not in campaign_no_memory["backend_config"]


def test_committed_smoke_benchmark_caps_run_one_scenario() -> None:
    benchmark_dir = Path(__file__).resolve().parents[2] / "configs" / "benchmarks"
    smoke_dir = benchmark_dir / "smoke_campaign"

    expected_caps = {
        "locomo.yaml": {"limit": 1, "limit_stimuli": 1},
        "evomemory_single_turn.yaml": {"limit": 1, "limit_stimuli": 1},
        "long_mem_eval.yaml": {"limit": 1},
        "transparency.yaml": {"limit": 1},
        "contextual_integrity.yaml": {"limit": 1},
        "stale.yaml": {"limit": 1},
        "mpbench.yaml": {"limit": 1, "limit_per_class": 1},
        "beam_100k.yaml": {"limit": 1},
    }

    for file_name, expected in expected_caps.items():
        payload = yaml.safe_load((smoke_dir / file_name).read_text(encoding="utf-8"))
        assert {key: payload[key] for key in expected} == expected

    contextual_integrity = yaml.safe_load(
        (smoke_dir / "contextual_integrity.yaml").read_text(encoding="utf-8")
    )
    assert contextual_integrity["dataset_source"]["download_limit"] == 1

    transparency = yaml.safe_load(
        (smoke_dir / "transparency.yaml").read_text(encoding="utf-8")
    )
    assert transparency["source"]["materialization_limit"] == 1


def test_committed_full_campaign_benchmark_caps_are_unbounded() -> None:
    benchmark_dir = Path(__file__).resolve().parents[2] / "configs" / "benchmarks"
    full_campaign_dir = benchmark_dir / "full_campaign"

    contextual_integrity = yaml.safe_load(
        (full_campaign_dir / "contextual_integrity.yaml").read_text(encoding="utf-8")
    )
    assert contextual_integrity.get("limit") is None
    assert contextual_integrity["dataset_source"]["download_limit"] is None

    transparency = yaml.safe_load(
        (full_campaign_dir / "transparency.yaml").read_text(encoding="utf-8")
    )
    assert transparency["limit"] is None
    assert transparency["source"]["materialization_limit"] is None


def _write_fixture_tree(tmp_path: Path) -> Path:
    package_root = tmp_path / "python_package"
    config_root = package_root / "configs"
    (config_root / "agents").mkdir(parents=True)
    (config_root / "memory_systems").mkdir(parents=True)
    (config_root / "benchmarks").mkdir(parents=True)
    (config_root / "runs").mkdir(parents=True)

    (config_root / "agents" / "gpt4o_mini_tool_use.yaml").write_text(
        "\n".join(
            [
                "type: agent",
                "memory_strategy: tool_use",
                "model_id: openai:gpt-4o-mini",
                'instructions: ""',
                "model_args:",
                "  temperature: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (config_root / "memory_systems" / "no_memory.yaml").write_text(
        "\n".join(
            [
                "id: no_memory",
                "backend_config:",
                "  type: no_memory",
                "  name: no_memory",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    package_locomo_yaml = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "full_campaign"
        / "locomo.yaml"
    )
    (config_root / "benchmarks" / "locomo.yaml").write_text(
        package_locomo_yaml.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    run_path = config_root / "runs" / "example_campaign.yaml"
    run_path.write_text(
        "\n".join(
            [
                "run_name: example_campaign",
                "experiments:",
                "  - name: gpt4o_mini_nomemory_locomo",
                "    agent_config_file: gpt4o_mini_tool_use.yaml",
                "    memory_system_config_file: no_memory.yaml",
                "    benchmark_config_file: locomo.yaml",
            ]
        ),
        encoding="utf-8",
    )
    return run_path
