from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.evaluation.factory import create_evaluator
from heams.evaluation.schemas import AuditRecord
from heams.memory_systems.factory import create_memory
from heams.memory_systems.in_context import InContextConfig
from heams.memory_systems.privacy import PrivacyConfig
from heams.orchestration.scenario_execution import execute_scenario
from tests.paths import PACKAGE_ROOT

CONFIGS = PACKAGE_ROOT / "configs" / "benchmarks"


class _AgentStub:
    """Holds a memory system; never called, as these scenarios have no stimulus."""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    def run(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("Retention Policy scenarios must not invoke the agent.")


def _load_config(name: str) -> RetentionPolicyBenchmarkConfig:
    doc = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    return RetentionPolicyBenchmarkConfig.model_validate(doc)


def _memory(tmp_path: Path, privacy_config: PrivacyConfig | None):
    return create_memory(
        InContextConfig(),
        storage_dir=tmp_path,
        privacy_config=privacy_config,
    )


def _run(sample, memory) -> AuditRecord:
    scenario = sample.to_scenario()
    eval_cfg = scenario.evaluators[0]
    evaluator = create_evaluator(eval_cfg, task_name=scenario.name)
    audit, _, _ = execute_scenario(
        scenario=scenario,
        agent=_AgentStub(memory),
        environment=None,
        evaluator=evaluator,
        eval_cfg=eval_cfg,
        build_ground_truth=lambda s, c: c.to_ground_truth(s),
    )
    return audit
