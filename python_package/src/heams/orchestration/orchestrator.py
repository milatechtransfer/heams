import time
from collections.abc import Callable
from contextlib import nullcontext
from itertools import product
from pathlib import Path

from heams.agents.factory import create_agent
from heams.environments.factory import create_environment
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.factory import (
    EvaluatorConfig,
    create_evaluator,
)
from heams.evaluation.efficiency import (
    compute_system_metrics,
    enhance_component_metrics,
)
from heams.evaluation.performance import compute_component_performance_metrics
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth
from heams.execution_traces import ExecutionTraceCollector, ExecutionTraceRecord
from heams.memory_systems.base import BaseMemorySystem
from heams.orchestration.scenario_execution import (
    execute_scenario,
)
from heams.memory_systems.factory import (
    MemorySystemConfig,
    MemorySystemVariant,
    create_memory,
)
from heams.results import (
    AgentTurn,
    RunResult,
    StimulusEvent,
    backfill_missing_stimulus_events,
    make_stimulus_event,
    progress_indices,
    summarize_memory_ledger,
)
from heams.orchestration.scenario_runtime import (
    setup_scenario_runtime,
    teardown_scenario_runtime,
)
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario
from heams.scenarios.config import ScenarioExecutionConfig

from .scenario_runtime import consolidate_if_enabled


class Orchestrator:
    """Executes benchmark scenarios end-to-end per the spec lifecycle
    (setup -> meta-action loop -> teardown)."""

    def __init__(self, scenario_config: ScenarioExecutionConfig) -> None:
        self.config = scenario_config

        self.active_agent = None
        self.active_environment = None
        self.active_memory: BaseMemorySystem | None = None
        self.active_memory_ledger = None
        self.active_evaluator: BaseEvaluator | None = None
        self._active_memory_path: Path | None = None

    def total_runs(self) -> int:
        """Number of `RunResult`s `run()` will produce.

        Exposed so callers (e.g. the CLI progress bar) can size themselves
        before `run()` starts emitting completions.
        """
        return (
            sum(
                self._scenario_run_multiplier(scenario)
                for scenario in self.config.scenarios
            )
            * self.config.num_samples
        )

    def total_stimulus_calls(self) -> int:
        """Total `UserStimulus` invocations across every run.

        Each stimulus is one agent `run` / `run_async` call and therefore
        the natural unit for a progress bar: a LOCOMO run with 2 memory
        backends × ~30 QAs shows as 60 ticks here instead of 2 at the
        run level. Non-stimulus meta-actions (env dispatch, memory/agent
        manipulations) are not counted because they don't invoke the
        agent.

        A scenario whose `execution_sequence` contains zero
        `UserStimulus` steps still contributes one tick per run so the
        bar's totals stay aligned with `run()`'s iteration count (the
        audit alone is work worth surfacing).
        """
        total = 0
        for scenario in self.config.scenarios:
            stimuli = sum(
                1
                for step in scenario.execution_sequence
                if isinstance(step, UserStimulus)
            )
            total += (
                max(stimuli, 1)
                * self._scenario_run_multiplier(scenario)
                * self.config.num_samples
            )
        return total

    @staticmethod
    def _combo_multiplier(scenario: Scenario) -> int:
        """Legacy scenario-local component fan-out factor."""
        return (
            max(len(scenario.agents), 1)
            * max(len(scenario.memory_systems), 1)
            * max(len(scenario.evaluators), 1)
        )

    def _scenario_run_multiplier(self, scenario: Scenario) -> int:
        if (
            self.config.agent_config is not None
            and self.config.memory_system_config is not None
        ):
            return max(len(scenario.evaluators), 1)
        return self._combo_multiplier(scenario)

    def run(
        self,
        *,
        on_run_complete: Callable[[RunResult], None] | None = None,
        on_stimulus_complete: Callable[[StimulusEvent], None] | None = None,
        on_execution_trace_complete: Callable[[ExecutionTraceRecord], None]
        | None = None,
        execution_trace_run_id: str | None = None,
    ) -> dict[str, list[RunResult]]:
        """Execute every scenario.

        If `on_run_complete` is supplied it is invoked exactly once per
        finished run, with the `RunResult` that was just appended. This
        lets the CLI drive a progress bar without coupling the
        orchestrator to any specific presentation layer.

        `on_stimulus_complete` is the finer-grained counterpart: it
        fires once per `UserStimulus` after the agent returns so callers can
        update a progress bar at call resolution, not run resolution.
        """
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, list[RunResult]] = {}
        for scenario in self.config.scenarios:
            results[scenario.id] = self._run_scenario(
                scenario,
                on_run_complete=on_run_complete,
                on_stimulus_complete=on_stimulus_complete,
                on_execution_trace_complete=on_execution_trace_complete,
                execution_trace_run_id=execution_trace_run_id,
            )
        return results

    def _run_scenario(
        self,
        scenario: Scenario,
        *,
        on_run_complete: Callable[[RunResult], None] | None = None,
        on_stimulus_complete: Callable[[StimulusEvent], None] | None = None,
        on_execution_trace_complete: Callable[[ExecutionTraceRecord], None]
        | None = None,
        execution_trace_run_id: str | None = None,
    ) -> list[RunResult]:
        use_resolved_components = (
            self.config.agent_config is not None
            and self.config.memory_system_config is not None
        )
        if not use_resolved_components and not scenario.agents:
            raise ValueError(
                f"Scenario '{scenario.id}' must declare at least one agent."
            )
        if not use_resolved_components and not scenario.memory_systems:
            raise ValueError(
                f"Scenario '{scenario.id}' must declare at least one memory system."
            )
        if not scenario.evaluators:
            raise ValueError(
                f"Scenario '{scenario.id}' must declare at least one evaluator."
            )

        scenario_results: list[RunResult] = []
        agent_configs = (
            [(0, self.config.agent_config)]
            if use_resolved_components
            else list(enumerate(scenario.agents))
        )
        memory_configs = (
            [(0, self.config.memory_system_config)]
            if use_resolved_components
            else list(enumerate(scenario.memory_systems))
        )
        combos = product(
            agent_configs,
            memory_configs,
            enumerate(scenario.evaluators),
        )
        expected_progress_indices = progress_indices(scenario)
        for (agent_idx, agent_cfg), (mem_idx, mem_cfg), (eval_idx, eval_cfg) in combos:
            for sample_index in range(self.config.num_samples):
                if (
                    self.config.pending_keys is not None
                    and (
                        scenario.id,
                        sample_index,
                    )
                    not in self.config.pending_keys
                ):
                    continue
                transcript: list[AgentTurn] = []
                agent_responses = []
                run_memory_ledger = []
                run_failed = False
                run_duration_ms = 0.0
                consolidation_error: Exception | None = None
                teardown_error: BaseException | None = None
                run_error_for_trace: str | None = None
                run_start = time.perf_counter()
                trace_collector = (
                    ExecutionTraceCollector(
                        run_id=execution_trace_run_id,
                        scenario_id=scenario.id,
                        agent_index=agent_idx,
                        memory_index=mem_idx,
                        evaluator_index=eval_idx,
                        sample_index=sample_index,
                    )
                    if on_execution_trace_complete is not None
                    else None
                )
                emitted_stimulus_indices: set[int] = set()
                emit_stimulus: Callable[[int, bool], None] | None = None
                if on_stimulus_complete is not None:
                    _cb = on_stimulus_complete

                    def _emit_stimulus(stimulus_index: int, ok: bool) -> None:
                        emitted_stimulus_indices.add(stimulus_index)
                        _cb(
                            make_stimulus_event(
                                scenario_id=scenario.id,
                                agent_index=agent_idx,
                                memory_index=mem_idx,
                                evaluator_index=eval_idx,
                                sample_index=sample_index,
                                stimulus_index=stimulus_index,
                                ok=ok,
                            )
                        )

                    emit_stimulus = _emit_stimulus

                try:
                    self._setup_scenario_runtime(
                        scenario,
                        agent_cfg=agent_cfg,
                        mem_cfg=mem_cfg,
                        eval_cfg=eval_cfg,
                        combo_key=f"a{agent_idx}-m{mem_idx}-e{eval_idx}",
                        sample_index=sample_index,
                        trace_collector=trace_collector,
                    )
                    (
                        audit,
                        transcript,
                        agent_responses,
                    ) = execute_scenario(
                        scenario=scenario,
                        agent=self.active_agent,
                        environment=self.active_environment,
                        evaluator=self.active_evaluator,
                        eval_cfg=eval_cfg,
                        build_ground_truth=self._build_ground_truth,
                        on_stimulus_complete=emit_stimulus,
                        trace_collector=trace_collector,
                    )
                except (ValueError, NotImplementedError):
                    # Programmer errors (malformed scenario, unimplemented
                    # dispatch) must fail loud so a broken scenario JSON
                    # isn't silently downgraded to a 0.0 accuracy run.
                    raise
                except Exception as run_error:
                    # Last-chance safety net for external runtime failures
                    # (OpenAI 400 context_length_exceeded, transport
                    # timeouts, provider 5xx, etc.): downgrade this run
                    # to a failed `RunResult` instead of tearing down the
                    # whole benchmark. Without this guard a single
                    # mid-scenario LLM error would propagate past
                    # `on_run_complete`, losing every checkpoint written
                    # so far.
                    run_failed = True
                    run_error_for_trace = f"{type(run_error).__name__}: {run_error}"
                    audit = AuditRecord(
                        is_successful=False,
                        accuracy_score=0.0,
                        passed_keys=[],
                        failed_keys=[f"run_error: {type(run_error).__name__}"],
                        audit_notes=f"Scenario raised: {run_error}",
                        score_available=False,
                        score_unavailable_reason=(
                            f"run_error: {type(run_error).__name__}"
                        ),
                    )
                finally:
                    try:
                        if self.active_memory is not None:
                            context = (
                                trace_collector.bind_context(phase="finalization")
                                if trace_collector is not None
                                else nullcontext()
                            )
                            with context:
                                consolidate_if_enabled(self.active_memory)
                    except Exception as error:
                        consolidation_error = error
                        run_failed = True
                        run_error_for_trace = f"{type(error).__name__}: {error}"
                    finally:
                        run_duration_ms = (time.perf_counter() - run_start) * 1000
                        run_memory_ledger = list(self.active_memory_ledger or [])
                        try:
                            self._teardown_scenario_runtime()
                        except BaseException as error:
                            teardown_error = error
                            run_failed = True
                            run_error_for_trace = f"{type(error).__name__}: {error}"

                if consolidation_error is not None:
                    consolidation_error_key = (
                        f"consolidation_error: {type(consolidation_error).__name__}"
                    )
                    consolidation_note = (
                        f"Post-session consolidation raised: {consolidation_error}"
                    )
                    audit = audit.model_copy(
                        update={
                            "is_successful": False,
                            "failed_keys": [
                                *audit.failed_keys,
                                consolidation_error_key,
                            ],
                            "audit_notes": (
                                f"{audit.audit_notes}; {consolidation_note}"
                            ),
                        }
                    )

                if teardown_error is not None:
                    teardown_error_key = (
                        f"teardown_error: {type(teardown_error).__name__}"
                    )
                    teardown_note = f"Scenario teardown raised: {teardown_error}"
                    if audit.is_successful:
                        audit = AuditRecord(
                            is_successful=False,
                            accuracy_score=0.0,
                            passed_keys=[],
                            failed_keys=[teardown_error_key],
                            audit_notes=teardown_note,
                            score_available=False,
                            score_unavailable_reason=teardown_error_key,
                        )
                    else:
                        audit = audit.model_copy(
                            update={
                                "failed_keys": [
                                    *audit.failed_keys,
                                    teardown_error_key,
                                ],
                                "audit_notes": (
                                    f"{audit.audit_notes}; {teardown_note}"
                                ),
                            }
                        )

                backfill_missing_stimulus_events(
                    expected_indices=expected_progress_indices,
                    emitted_indices=emitted_stimulus_indices,
                    emit_stimulus=emit_stimulus,
                    ok=not run_failed,
                )

                run_stimulus_count = len(emitted_stimulus_indices) or len(transcript)

                # Compute component metrics with efficiency enhancements
                component_metrics = None
                component_performance_metrics = None
                if run_memory_ledger:
                    base_component_metrics = summarize_memory_ledger(run_memory_ledger)
                    enhanced_component_metrics = enhance_component_metrics(
                        base_component_metrics,
                        run_memory_ledger,
                        stimulus_count=run_stimulus_count,
                    )
                    component_performance_metrics = (
                        compute_component_performance_metrics(
                            base_component_metrics,
                            run_memory_ledger,
                            scenario,
                            eval_cfg=eval_cfg,
                        )
                    )
                    component_metrics = enhanced_component_metrics

                # Compute system-level efficiency metrics
                system_metrics = compute_system_metrics(
                    transcript=transcript,
                    audit_data={
                        "is_successful": audit.is_successful,
                        "accuracy_score": audit.accuracy_score,
                        "score_available": audit.score_available,
                    },
                    memory_ledger=run_memory_ledger or None,
                    agent_responses=agent_responses,
                    run_duration_ms=run_duration_ms,
                    stimulus_count=run_stimulus_count,
                )

                run_result = RunResult(
                    scenario_id=scenario.id,
                    agent_index=agent_idx,
                    memory_index=mem_idx,
                    evaluator_index=eval_idx,
                    sample_index=sample_index,
                    trace_id=(
                        trace_collector.trace_id
                        if trace_collector is not None
                        else None
                    ),
                    audit=audit,
                    transcript=transcript,
                    system_metrics=system_metrics,
                    memory_ledger=run_memory_ledger or None,
                    component_metrics=component_metrics,
                    component_performance_metrics=component_performance_metrics,
                )
                scenario_results.append(run_result)
                if (
                    on_execution_trace_complete is not None
                    and trace_collector is not None
                ):
                    on_execution_trace_complete(
                        trace_collector.finalize(
                            outcome="failed" if run_failed else "completed",
                            error=run_error_for_trace,
                        )
                    )
                if on_run_complete is not None:
                    on_run_complete(run_result)

        return scenario_results

    def _setup_scenario_runtime(
        self,
        scenario: Scenario,
        *,
        agent_cfg,
        mem_cfg: MemorySystemConfig | MemorySystemVariant,
        eval_cfg: EvaluatorConfig,
        combo_key: str,
        sample_index: int,
        trace_collector: ExecutionTraceCollector | None = None,
    ) -> None:
        (
            self.active_environment,
            self.active_memory,
            self._active_memory_path,
            self.active_agent,
            self.active_evaluator,
            self.active_memory_ledger,
        ) = setup_scenario_runtime(
            scenario,
            workspace_dir=self.config.workspace_dir,
            agent_cfg=agent_cfg,
            mem_cfg=mem_cfg,
            eval_cfg=eval_cfg,
            combo_key=combo_key,
            sample_index=sample_index,
            create_environment_fn=create_environment,
            create_memory_fn=create_memory,
            create_agent_fn=create_agent,
            create_evaluator_fn=create_evaluator,
            trace_collector=trace_collector,
        )

    def _teardown_scenario_runtime(self) -> None:
        try:
            teardown_scenario_runtime(
                self.active_environment,
                self._active_memory_path,
                self.active_agent,
                self.active_memory,
            )
        finally:
            self.active_agent = None
            self.active_environment = None
            self.active_memory = None
            self.active_memory_ledger = None
            self.active_evaluator = None
            self._active_memory_path = None

    @staticmethod
    def _build_ground_truth(
        scenario: Scenario, eval_cfg: EvaluatorConfig
    ) -> BaseGroundTruth:
        return eval_cfg.to_ground_truth(scenario)
