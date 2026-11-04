from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.evaluation.factory import create_evaluator
from heams.evaluation.schemas import AuditRecord
from heams.memory_systems.factory import create_memory
from heams.memory_systems.in_context import InContextConfig, InContextMemorySystem
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


class KeywordGovernedMemory(InContextMemorySystem):
    """An in-context memory with native governance, through the contract only.

    Once a policy is set, a retain whose content contains one of
    `drop_substrings` is declined and logged as `dropped`; anything else is
    stored. It stands in for any governed memory system, so these tests don't
    depend on a particular governance implementation.
    """

    def __init__(self, *, storage_dir: Path, drop_substrings: list[str]) -> None:
        super().__init__(InContextConfig(), storage_dir=storage_dir)
        self._drop_substrings = [s.lower() for s in drop_substrings]
        self._enforcing = False

    def set_retention_policy(self, policies: list[str]) -> str:
        self._enforcing = True
        self._log_memory_event(
            operation_type="set_retention_policy",
            status="success",
            source="runtime",
            payload={"policies": list(policies)},
            result={"enforced": True},
        )
        return f"Enforcing {list(policies)}"

    def retain(self, content: str, metadata: dict | None = None) -> str:
        if self._enforcing and any(s in content.lower() for s in self._drop_substrings):
            self._log_memory_event(
                operation_type="retain",
                status="dropped",
                source="runtime",
                payload={"content_length": len(content)},
            )
            return "Not stored: the policy forbids it."
        return super().retain(content, metadata)


def _memory(tmp_path: Path, drop_substrings: list[str] | None):
    """A governed memory dropping `drop_substrings`, or, for None, an ungoverned one."""
    if drop_substrings is None:
        return create_memory(InContextConfig(), storage_dir=tmp_path)
    return KeywordGovernedMemory(storage_dir=tmp_path, drop_substrings=drop_substrings)


def _run(sample, memory) -> AuditRecord:
    scenario = sample.to_scenario()
    eval_cfg = scenario.evaluators[0]
    evaluator = create_evaluator(eval_cfg, task_name=scenario.name)
    # The evaluator reads each retain's status from the memory events, as in a run.
    memory_ledger: list = []
    memory.attach_event_sink(memory_ledger)
    audit, _, _ = execute_scenario(
        scenario=scenario,
        agent=_AgentStub(memory),
        environment=None,
        evaluator=evaluator,
        eval_cfg=eval_cfg,
        build_ground_truth=lambda s, c: c.to_ground_truth(s),
        memory_ledger=memory_ledger,
    )
    return audit
