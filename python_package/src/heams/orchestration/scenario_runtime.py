import shutil
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Literal

from heams.agents.factory import AgentConfig, create_agent
from heams.environments.factory import create_environment
from heams.evaluation.factory import EvaluatorConfig, create_evaluator
from heams.execution_traces import ExecutionTraceCollector
from heams.memory_systems.base import BaseMemorySystem
from heams.memory_systems.factory import (
    MemorySystemConfig,
    MemorySystemVariant,
    create_memory,
)
from heams.memory_systems.schemas import MemoryOperationEvent
from heams.scenarios.base import Scenario
from heams.tools import (
    BenchmarkToolSpec,
    compile_tool,
    final_answer_tool_spec,
    memory_tool_specs,
)


ToolLike = Callable[..., Any] | BenchmarkToolSpec


def compile_tools(tools: list[ToolLike]) -> list[Callable[..., Any]]:
    compiled: list[Callable[..., Any]] = []
    for tool in tools:
        compiled.append(
            compile_tool(tool) if isinstance(tool, BenchmarkToolSpec) else tool
        )
    return compiled


def build_tools(
    environment: Any,
    memory: BaseMemorySystem,
    *,
    agent_cfg: AgentConfig,
    include_final_answer: bool = False,
) -> list[Callable[..., Any]]:
    env_tools: list[ToolLike] = []
    if environment is not None:
        get_tool_specs = getattr(environment, "get_tool_specs", None)
        if callable(get_tool_specs):
            env_tools = list(get_tool_specs())
        if not env_tools:
            env_tools = list(environment.get_available_tools())

    memory_tools: list[BenchmarkToolSpec] = []
    strategy = getattr(agent_cfg, "memory_strategy", None)
    if strategy is not None and strategy.uses_memory_tools:
        memory_tools = memory_tool_specs(
            memory,
            recall_only=not strategy.allows_memory_writes,
        )

    tools: list[ToolLike] = [*env_tools, *memory_tools]

    if include_final_answer:
        tools.append(final_answer_tool_spec())

    return compile_tools(tools)


def needs_explicit_final_answer(agent_cfg: AgentConfig) -> bool:
    model_id = getattr(agent_cfg, "model_id", "")
    return isinstance(model_id, str) and model_id.lower().startswith("openai:")


def resolve_memory_system_entry(
    mem_cfg: MemorySystemConfig | MemorySystemVariant,
) -> tuple[MemorySystemConfig, dict[str, Any]]:
    if isinstance(mem_cfg, MemorySystemVariant):
        return mem_cfg.backend_config, dict(mem_cfg.wrapper_configs)
    return mem_cfg, {}


def consolidate_if_enabled(memory: BaseMemorySystem) -> None:
    if memory.config.consolidation.enabled is True:
        memory.consolidate()


def setup_scenario_runtime(
    scenario: Scenario,
    *,
    workspace_dir: Path,
    agent_cfg: AgentConfig,
    mem_cfg: MemorySystemConfig | MemorySystemVariant,
    eval_cfg: EvaluatorConfig,
    combo_key: str,
    sample_index: int,
    create_environment_fn: Callable[..., Any] = create_environment,
    create_memory_fn: Callable[..., BaseMemorySystem] = create_memory,
    create_agent_fn: Callable[..., Any] = create_agent,
    create_evaluator_fn: Callable[..., Any] = create_evaluator,
    trace_collector: ExecutionTraceCollector | None = None,
) -> tuple[Any, BaseMemorySystem, Path, Any, Any, list[MemoryOperationEvent]]:
    backend_cfg, wrapper_configs = resolve_memory_system_entry(mem_cfg)

    env_priming_config = scenario.scenario_priming.environment.config
    environment = None
    storage_dir = workspace_dir / f"{scenario.id}-{combo_key}-s{sample_index}-memory"
    memory_ledger: list[MemoryOperationEvent] = []
    memory = None
    agent = None
    try:
        if env_priming_config is not None:
            started_ns = time.time_ns()
            try:
                environment = create_environment_fn(env_priming_config)
                environment.reset()
            except BaseException as error:
                if trace_collector is not None:
                    trace_collector.add_event(
                        phase="priming",
                        type="environment_priming",
                        name="reset",
                        content="Environment creation and reset",
                        arguments={
                            "config": env_priming_config.model_dump(mode="json")
                        },
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="error",
                        error=str(error),
                    )
                raise
            if trace_collector is not None:
                trace_collector.add_event(
                    phase="priming",
                    type="environment_priming",
                    name="reset",
                    content="Environment creation and reset",
                    arguments={"config": env_priming_config.model_dump(mode="json")},
                    start_time_ns=started_ns,
                    end_time_ns=time.time_ns(),
                    status="ok",
                )

        storage_dir.mkdir(parents=True, exist_ok=True)

        def _record_memory_event(event: MemoryOperationEvent) -> None:
            memory_ledger.append(event)
            if trace_collector is not None:
                trace_collector.add_memory_event(event)

        memory = create_memory_fn(
            backend_cfg,
            storage_dir=storage_dir,
            event_sink=_record_memory_event,
            run_context={
                "scenario_id": scenario.id,
                "combo_key": combo_key,
                "sample_index": sample_index,
            },
            wrapper_configs=wrapper_configs,
        )
        memory_entries = scenario.scenario_priming.memory.entries
        started_ns = time.time_ns()

        def _record_memory_priming(
            *, status: Literal["ok", "error"], error: str | None = None
        ) -> None:
            if trace_collector is None:
                return
            # Retain payloads already land on per-entry memory_operation events.
            trace_collector.add_event(
                phase="priming",
                type="memory_priming",
                name="prime_memory_entries",
                content=f"{len(memory_entries)} entries",
                arguments={"entry_count": len(memory_entries)},
                start_time_ns=started_ns,
                end_time_ns=time.time_ns(),
                status=status,
                error=error,
            )

        try:
            context = (
                trace_collector.bind_context(phase="priming")
                if trace_collector is not None
                else nullcontext()
            )
            with context:
                memory.prime_memory_entries(memory_entries)
                consolidate_if_enabled(memory)
        except BaseException as error:
            _record_memory_priming(status="error", error=str(error))
            raise
        _record_memory_priming(status="ok")

        tools = build_tools(
            environment,
            memory,
            agent_cfg=agent_cfg,
            include_final_answer=needs_explicit_final_answer(agent_cfg),
        )
        agent = create_agent_fn(agent_cfg, tools=tools, memory_system=memory)
        evaluator = create_evaluator_fn(eval_cfg, task_name=scenario.id)
        return environment, memory, storage_dir, agent, evaluator, memory_ledger
    except BaseException:
        with suppress(BaseException):
            teardown_scenario_runtime(environment, storage_dir, agent, memory)
        raise


def teardown_scenario_runtime(
    environment: Any,
    storage_dir: Path | None,
    agent: Any | None = None,
    memory: BaseMemorySystem | None = None,
) -> None:
    try:
        if agent is not None:
            close = getattr(agent, "close", None)
            if callable(close):
                close()
    finally:
        try:
            if memory is not None:
                close = getattr(memory, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                if environment is not None:
                    environment.close()
            finally:
                if storage_dir is not None:
                    shutil.rmtree(storage_dir, ignore_errors=True)
