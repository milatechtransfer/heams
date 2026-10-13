import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from pydantic import TypeAdapter

from heams.memory_systems import MemorySystemConfig, MemorySystemVariant, create_memory
from heams.orchestration import resolve_run_config
from heams.orchestration.run_orchestration import build_experiment_execution_configs
from heams.orchestration.scenario_runtime import (
    setup_scenario_runtime,
    teardown_scenario_runtime,
)
from heams.scenarios import Scenario, UserStimulus
from heams_memory_milamem import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
    MilaMemConfig,
    MilaMemMemorySystem,
)

CORE_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "python_package" / "configs"


def test_basic_complete_config_dispatches_to_backend(tmp_path) -> None:
    config = BasicCompleteMemorySystemConfig(name="t", top_k=3)
    original_config = config.model_dump()

    memory = create_memory(config, storage_dir=tmp_path)

    assert isinstance(memory, BasicCompleteMemorySystem)
    assert memory.storage_dir == tmp_path
    assert config.model_dump() == original_config
    assert memory.config.name == "t"
    assert memory.config.top_k == 3
    assert memory.config.file_path == str(tmp_path / "memory.json")


def test_basic_complete_retains_and_recalls_through_factory(tmp_path) -> None:
    memory = create_memory(
        BasicCompleteMemorySystemConfig(top_k=2),
        storage_dir=tmp_path,
    )

    memory.retain("a")
    memory.retain("b")
    memory.retain("c")

    assert memory.recall("") == ["b", "c"]
    assert (tmp_path / "memory.json").exists()


def test_memory_system_config_union_resolves_plugin_types() -> None:
    adapter = TypeAdapter(MemorySystemConfig)

    milamem = adapter.validate_python({"type": "milamem", "top_k": 4})
    basic_complete = adapter.validate_python(
        {"type": "basic_complete_memory_system", "top_k": 4}
    )

    assert isinstance(milamem, MilaMemConfig)
    assert isinstance(basic_complete, BasicCompleteMemorySystemConfig)
    assert milamem.top_k == basic_complete.top_k == 4


def test_milamem_config_dispatches_to_backend(tmp_path) -> None:
    storage_dir = tmp_path / "canonical-runtime"
    config = MilaMemConfig(name="canonical", top_k=3)

    memory = create_memory(config, storage_dir=storage_dir)

    assert isinstance(memory, MilaMemMemorySystem)
    assert memory.config.name == "canonical"
    assert memory.config.top_k == 3
    assert memory.config.raw_store_path == str(storage_dir / "raw.json")
    assert memory.config.canonical_store_path == str(storage_dir / "canonical.json")


def test_milamem_factory_uses_runtime_paths_over_configured_store_paths(
    tmp_path,
) -> None:
    config = MilaMemConfig(
        raw_store_path=str(tmp_path / "configured-raw.json"),
        canonical_store_path=str(tmp_path / "configured-canonical.json"),
        embedding_cache_path=str(tmp_path / "configured-embeddings.json"),
    )
    storage_dir = tmp_path / "runtime-memory"

    memory = create_memory(
        config,
        storage_dir=storage_dir,
    )

    assert isinstance(memory, MilaMemMemorySystem)
    assert memory.config.raw_store_path == str(storage_dir / "raw.json")
    assert memory.config.canonical_store_path == str(storage_dir / "canonical.json")
    assert memory.config.embedding_cache_path == str(
        storage_dir / "canonical.embeddings.json"
    )


def test_milamem_campaign_yaml_validates_with_runtime_paths_unset() -> None:
    config_path = CORE_CONFIGS_DIR / "memory_systems" / "campaigns" / "milamem.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    variant = MemorySystemVariant.model_validate(payload)

    assert variant.id == "milamem"
    assert isinstance(variant.backend_config, MilaMemConfig)
    assert variant.backend_config.temporal_metadata.enabled is True
    assert variant.backend_config.dedup_strategy.window.days == 10
    assert variant.backend_config.recall_strategy.kind.value == "bm25"
    assert variant.backend_config.dedup_strategy.kind.value == "semantic"
    assert {
        "raw_store_path",
        "canonical_store_path",
        "embedding_cache_path",
    }.isdisjoint(variant.backend_config.model_fields_set)


def test_milamem_locomo_campaign_resolves_directory_owned_storage(
    tmp_path: Path,
) -> None:
    run_path = CORE_CONFIGS_DIR / "runs" / "milamem_all_benchmarks" / "locomo.yaml"

    [experiment_config] = build_experiment_execution_configs(
        resolve_run_config(run_path)
    )

    assert experiment_config.experiment_name == "gpt56_luna_milamem_locomo"
    backend_config = experiment_config.memory_system_config.backend_config
    assert isinstance(backend_config, MilaMemConfig)
    authored_config = backend_config.model_dump(mode="json")

    scenario = Scenario(
        id="milamem-locomo-smoke",
        name="milamem-locomo-smoke",
        execution_sequence=[UserStimulus(content="question")],
    )
    _, memory, storage_dir, _, _, _ = setup_scenario_runtime(
        scenario,
        workspace_dir=tmp_path,
        agent_cfg=experiment_config.agent_config,
        mem_cfg=experiment_config.memory_system_config,
        eval_cfg=MagicMock(),
        combo_key="a0-m0-e0",
        sample_index=0,
        create_agent_fn=lambda *args, **kwargs: MagicMock(),
        create_evaluator_fn=lambda *args, **kwargs: MagicMock(),
    )

    assert memory.config.raw_store_path == str(storage_dir / "raw.json")
    assert memory.config.canonical_store_path == str(storage_dir / "canonical.json")
    assert backend_config.model_dump(mode="json") == authored_config
    serialized_experiment = experiment_config.model_dump(mode="json")
    serialized_backend = serialized_experiment["memory_system_config"]["backend_config"]
    assert serialized_backend["raw_store_path"] == "raw_memory.json"
    assert serialized_backend["canonical_store_path"] == "canonical_memory.json"
    assert str(storage_dir) not in json.dumps(serialized_experiment)

    teardown_scenario_runtime(None, storage_dir)
    assert not storage_dir.exists()
