import inspect
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

from heams.agents.base import AgentResponse
from heams.evaluation.factory import EvaluatorConfig
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth
from heams.evaluation.trace_utils import iter_tool_calls
from heams.execution_traces import ExecutionTraceCollector
from heams.memory_systems.schemas import MemoryOperationEvent
from heams.results import AgentTurn, ToolResult
from heams.scenarios.actions import (
    AgentManipulation,
    EnvironmentManipulation,
    MemoryManipulation,
    UserStimulus,
)
from heams.scenarios.base import Scenario
from heams.utils.event_loop import ensure_thread_event_loop
from heams.tools import should_trace_tool

_REPORT_TOOL_NAMES = {"finish_task", "step_task"}

#: Allowlist for `MemoryManipulation`, mirroring `BaseEnvironment.dispatch_command`.
#: `retain`/`recall` let a scenario drive the normal `BaseMemorySystem`
#: interface directly (e.g. the retention_policy benchmark's end-to-end
#: retain-then-recall check), so it works against any backend, wrapped or not.
#: `set_retention_policy` lets a scenario choose which user policy the
#: governance layer enforces, so one config can cover every policy.
_DISPATCHABLE_MEMORY_COMMANDS = frozenset({"retain", "recall", "set_retention_policy"})


def _operation_status(
    memory_ledger: Sequence[MemoryOperationEvent] | None,
    start: int | None,
    command: str,
) -> str | None:
    """Status of the last `command` event the memory system logged since `start`."""
    if memory_ledger is None or start is None:
        return None
    for event in reversed(memory_ledger[start:]):
        if event.operation_type == command:
            return str(getattr(event.status, "value", event.status))
    return None


def failed_audit(*, failed_key: str, audit_notes: str) -> AuditRecord:
    return AuditRecord(
        is_successful=False,
        accuracy_score=0.0,
        passed_keys=[],
        failed_keys=[failed_key],
        audit_notes=audit_notes,
        score_available=False,
        score_unavailable_reason=failed_key,
    )


def _should_report_tool(tool_name: str) -> bool:
    return should_trace_tool(tool_name) or tool_name in _REPORT_TOOL_NAMES


def _agent_turn(stimulus: UserStimulus, response: AgentResponse) -> AgentTurn:
    target_answer = (stimulus.step_evaluation_markers or {}).get("expected_answer")
    return AgentTurn(
        question=stimulus.content,
        answer=response.answer,
        target_answer=str(target_answer) if target_answer is not None else None,
        tool_results=[
            ToolResult(
                tool_name=call.tool_name,
                arguments=call.arguments,
                output=call.output,
            )
            for call in iter_tool_calls([response])
            if _should_report_tool(call.tool_name)
        ],
        context_truncated=response.context_truncated,
    )


def _build_memory_event_context(
    stimulus_index: int,
    stimulus: UserStimulus,
) -> dict[str, Any]:
    markers = stimulus.step_evaluation_markers or {}
    context: dict[str, Any] = {"stimulus_index": stimulus_index}
    for key in ("question", "question_id", "task_id"):
        value = markers.get(key)
        if value is not None and str(value):
            context[key] = str(value)
    return context


def _accepts_run_context(callable_obj: Callable[..., Any]) -> bool:
    signature_target = callable_obj
    side_effect = getattr(callable_obj, "side_effect", None)
    if callable(side_effect):
        signature_target = side_effect

    try:
        signature = inspect.signature(signature_target)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "run_context" in signature.parameters


def _invoke_agent_method(
    method: Callable[..., AgentResponse],
    prompt: str,
    run_context: dict[str, Any],
) -> AgentResponse:
    if _accepts_run_context(method):
        return method(prompt, run_context=run_context)
    return method(prompt)


def execute_scenario(
    *,
    scenario: Scenario,
    agent: Any,
    environment: Any,
    evaluator: Any,
    eval_cfg: EvaluatorConfig,
    build_ground_truth: Callable[[Scenario, EvaluatorConfig], BaseGroundTruth],
    on_stimulus_complete: Callable[[int, bool], None] | None = None,
    trace_collector: ExecutionTraceCollector | None = None,
    memory_ledger: Sequence[MemoryOperationEvent] | None = None,
) -> tuple[AuditRecord, list[AgentTurn], list[AgentResponse]]:
    agent_responses: list[AgentResponse] = []
    transcript: list[AgentTurn] = []
    for step_index, meta_action in enumerate(scenario.execution_sequence):
        if isinstance(meta_action, UserStimulus):
            if scenario.agent_conversation == "independent":
                started_ns = time.time_ns()
                try:
                    agent.reset_state()
                except BaseException as error:
                    if trace_collector is not None:
                        trace_collector.add_event(
                            phase="execution",
                            type="agent_manipulation",
                            name="reset_state",
                            content="Implicit reset for independent conversation",
                            start_time_ns=started_ns,
                            end_time_ns=time.time_ns(),
                            status="error",
                            error=str(error),
                            step_index=step_index,
                            stimulus_index=step_index,
                        )
                    raise
                if trace_collector is not None:
                    trace_collector.add_event(
                        phase="execution",
                        type="agent_manipulation",
                        name="reset_state",
                        content="Implicit reset for independent conversation",
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="ok",
                        step_index=step_index,
                        stimulus_index=step_index,
                    )
            started_ns = time.time_ns()
            try:
                context = (
                    trace_collector.bind_context(
                        phase="execution",
                        step_index=step_index,
                        stimulus_index=step_index,
                    )
                    if trace_collector is not None
                    else nullcontext()
                )
                with context:
                    response = _invoke_agent_method(
                        agent.run,
                        meta_action.content,
                        _build_memory_event_context(step_index, meta_action),
                    )
            except BaseException as error:
                if trace_collector is not None:
                    trace_collector.add_agent_trace(
                        getattr(error, "trace", None),
                        step_index=step_index,
                        stimulus_index=step_index,
                    )
                    trace_collector.add_event(
                        phase="execution",
                        type="user_stimulus",
                        name="user_stimulus",
                        content=meta_action.content,
                        arguments={
                            "step_evaluation_markers": (
                                meta_action.step_evaluation_markers
                            )
                        },
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="error",
                        error=str(error),
                        step_index=step_index,
                        stimulus_index=step_index,
                    )
                raise
            if trace_collector is not None:
                trace_collector.add_agent_trace(
                    response.raw_trace,
                    step_index=step_index,
                    stimulus_index=step_index,
                )
                trace_collector.add_event(
                    phase="execution",
                    type="user_stimulus",
                    name="user_stimulus",
                    content=meta_action.content,
                    arguments={
                        "step_evaluation_markers": meta_action.step_evaluation_markers
                    },
                    result={"answer": response.answer},
                    start_time_ns=started_ns,
                    end_time_ns=time.time_ns(),
                    status="ok",
                    step_index=step_index,
                    stimulus_index=step_index,
                )
            agent_responses.append(response)
            transcript.append(_agent_turn(meta_action, response))
            if on_stimulus_complete is not None:
                on_stimulus_complete(step_index, True)
        elif isinstance(meta_action, EnvironmentManipulation):
            if environment is None:
                raise ValueError(
                    f"Scenario '{scenario.id}' issues "
                    f"EnvironmentManipulation '{meta_action.command}' but "
                    "has no `scenario_priming.environment.config`."
                )
            started_ns = time.time_ns()
            try:
                outcome = environment.dispatch_command(
                    meta_action.command, dict(meta_action.parameters)
                )
            except BaseException as error:
                if trace_collector is not None:
                    trace_collector.add_event(
                        phase="execution",
                        type="environment_manipulation",
                        name=meta_action.command,
                        content=meta_action.description,
                        arguments=meta_action.parameters,
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="error",
                        error=str(error),
                        step_index=step_index,
                    )
                raise
            if trace_collector is not None:
                trace_collector.add_event(
                    phase="execution",
                    type="environment_manipulation",
                    name=meta_action.command,
                    content=meta_action.description,
                    arguments=meta_action.parameters,
                    result=outcome,
                    start_time_ns=started_ns,
                    end_time_ns=time.time_ns(),
                    status="ok",
                    step_index=step_index,
                )
        elif isinstance(meta_action, MemoryManipulation):
            memory_system = getattr(agent, "memory_system", None)
            if memory_system is None:
                raise ValueError(
                    f"Scenario '{scenario.id}' issues MemoryManipulation "
                    f"'{meta_action.command}' but the agent has no memory system."
                )
            if meta_action.command not in _DISPATCHABLE_MEMORY_COMMANDS:
                raise ValueError(
                    f"Scenario '{scenario.id}' issues MemoryManipulation "
                    f"'{meta_action.command}', which is not dispatchable. "
                    f"Allowed: {sorted(_DISPATCHABLE_MEMORY_COMMANDS)}."
                )
            method = getattr(memory_system, meta_action.command, None)
            if not callable(method):
                raise ValueError(
                    f"Scenario '{scenario.id}' issues MemoryManipulation "
                    f"'{meta_action.command}', which this memory system does not "
                    "implement."
                )
            started_ns = time.time_ns()
            ledger_start = len(memory_ledger) if memory_ledger is not None else None
            bind_run_context = getattr(memory_system, "bind_run_context", None)
            token = (
                bind_run_context({"step_index": step_index})
                if callable(bind_run_context)
                else None
            )
            try:
                context = (
                    trace_collector.bind_context(
                        phase="execution", step_index=step_index
                    )
                    if trace_collector is not None
                    else nullcontext()
                )
                with context:
                    outcome = method(**dict(meta_action.parameters))
            except BaseException as error:
                if trace_collector is not None:
                    trace_collector.add_event(
                        phase="execution",
                        type="memory_manipulation",
                        name=meta_action.command,
                        content=meta_action.description,
                        arguments=meta_action.parameters,
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="error",
                        error=str(error),
                        step_index=step_index,
                    )
                raise
            finally:
                reset_run_context = getattr(memory_system, "reset_run_context", None)
                if callable(reset_run_context):
                    reset_run_context(token)
            if trace_collector is not None:
                trace_collector.add_event(
                    phase="execution",
                    type="memory_manipulation",
                    name=meta_action.command,
                    content=meta_action.description,
                    arguments=meta_action.parameters,
                    result=outcome,
                    start_time_ns=started_ns,
                    end_time_ns=time.time_ns(),
                    status="ok",
                    step_index=step_index,
                )
            agent_responses.append(
                AgentResponse(
                    answer=str(outcome),
                    memory_status=_operation_status(
                        memory_ledger, ledger_start, meta_action.command
                    ),
                )
            )
        elif isinstance(meta_action, AgentManipulation):
            if meta_action.command != "reset_state":
                raise ValueError(
                    f"Scenario '{scenario.id}' issues unsupported "
                    f"AgentManipulation command '{meta_action.command}'."
                )
            started_ns = time.time_ns()
            try:
                agent.reset_state()
            except BaseException as error:
                if trace_collector is not None:
                    trace_collector.add_event(
                        phase="execution",
                        type="agent_manipulation",
                        name=meta_action.command,
                        content=meta_action.description,
                        arguments=meta_action.parameters,
                        start_time_ns=started_ns,
                        end_time_ns=time.time_ns(),
                        status="error",
                        error=str(error),
                        step_index=step_index,
                    )
                raise
            if trace_collector is not None:
                trace_collector.add_event(
                    phase="execution",
                    type="agent_manipulation",
                    name=meta_action.command,
                    content=meta_action.description,
                    arguments=meta_action.parameters,
                    start_time_ns=started_ns,
                    end_time_ns=time.time_ns(),
                    status="ok",
                    step_index=step_index,
                )
        else:  # pragma: no cover - discriminated union exhausts above
            raise TypeError(f"Unknown meta-action type: {type(meta_action)!r}")

    ground_truth = build_ground_truth(scenario, eval_cfg)
    try:
        # Every LLM-as-judge evaluator (heams.evaluation.llm_judge.LlmJudge)
        # calls any_llm synchronously with `allow_running_loop`; without a
        # persistent loop on whatever thread runs this, any_llm falls back to
        # `asyncio.run(coro)` per call, churning a fresh event loop each time
        # and orphaning httpx client cleanup onto an already-closed one.
        ensure_thread_event_loop()
        audit = evaluator.audit(agent_responses, ground_truth)
    except Exception as audit_error:
        audit = failed_audit(
            failed_key=f"audit_error: {type(audit_error).__name__}",
            audit_notes=f"Evaluator raised: {audit_error}",
        )
    return audit, transcript, agent_responses
