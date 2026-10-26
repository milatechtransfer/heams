from pydantic import BaseModel, Field


class ComponentMetricsSummary(BaseModel):
    """Pre-aggregated metrics derived from component ledger events."""

    component: str = Field(
        ..., description="Stable identifier for the observed component."
    )
    component_type: str = Field(
        default="memory_system",
        description="Component category for downstream grouping and reporting.",
    )
    total_operations: int = Field(default=0, ge=0)
    successful_operations: int = Field(default=0, ge=0)
    failed_operations: int = Field(default=0, ge=0)
    skipped_operations: int = Field(default=0, ge=0)
    operation_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by MemoryOperationType values.",
    )
    status_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by MemoryOperationStatus values.",
    )
    source_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts keyed by MemoryOperationSource values.",
    )
    total_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Aggregate wall-clock duration across observed operations.",
    )
    retain_success_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description=(
            "Successful retain operations divided by retain operations a policy "
            "did not drop."
        ),
    )
    delete_success_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description="Successful delete operations divided by total delete operations.",
    )
    consolidate_success_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description="Successful consolidate operations divided by total consolidate operations.",
    )
    retain_operations_per_stimulus: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Retain operations divided by executed stimuli in the run.",
    )
    delete_operations_per_stimulus: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Delete operations divided by executed stimuli in the run.",
    )
    consolidate_operations_per_stimulus: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Consolidate operations divided by executed stimuli in the run.",
    )
    retain_bytes_per_successful_retain: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Total retained content bytes divided by successful retain operations when byte evidence exists for each successful retain.",
    )
    # Efficiency metrics derived from operation durations and results
    avg_recall_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average duration (ms) of recall operations for this component.",
    )
    p95_recall_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="95th percentile duration (ms) of recall operations for this component.",
    )
    avg_retain_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average duration (ms) of retain operations for this component.",
    )
    avg_delete_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average duration (ms) of delete operations for this component.",
    )
    avg_consolidate_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average duration (ms) of consolidate operations for this component.",
    )
    avg_recall_results_count: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average number of items returned per recall operation.",
    )
    avg_recall_output_bytes: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average serialized output size proxy (bytes) per recall operation.",
    )
    total_recall_output_bytes: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Total cumulative serialized output size proxy (bytes) across all recall operations.",
    )
    avg_retain_content_bytes: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Average serialized content size proxy (bytes) retained per retain operation.",
    )
    total_retained_content_bytes: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Total cumulative serialized content size proxy (bytes) across all retain operations.",
    )
    recall_errors: int = Field(
        default=0, ge=0, description="Count of failed recall operations."
    )
    retain_errors: int = Field(
        default=0, ge=0, description="Count of failed retain operations."
    )
    delete_errors: int = Field(
        default=0, ge=0, description="Count of failed delete operations."
    )
    consolidate_errors: int = Field(
        default=0, ge=0, description="Count of failed consolidate operations."
    )


class ComponentPerformanceMetricsSummary(BaseModel):
    """Component-level retrieval quality metrics derived from recall operations."""

    component: str = Field(
        ..., description="Stable identifier for the observed component."
    )
    component_type: str = Field(
        default="memory_system",
        description="Component category for downstream grouping and reporting.",
    )
    recall_eval_count: int = Field(
        default=0,
        ge=0,
        description="Number of recall operations eligible for benchmark-backed retrieval scoring.",
    )
    recall_missing_stimulus_index_count: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
        description="Count of recall operations that could not be linked back to a stimulus via run context.",
    )
    recall_missing_reference_evidence_count: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
        description="Count of recall operations whose linked stimulus had no benchmark reference evidence for retrieval scoring.",
    )
    recall_missing_result_evidence_count: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
        description="Count of recall operations whose runtime result lacked returned-item evidence for retrieval scoring.",
    )
    recall_hit_at_k: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description="Mean Hit@k across eligible recall operations. `None` means no evaluable retrieval evidence or no eligible recalls; `0.0` means evaluable recalls with zero hits.",
    )
    recall_mrr: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description="Mean reciprocal rank across eligible recall operations. `None` means no evaluable retrieval evidence or no eligible recalls; `0.0` means evaluable recalls with zero retrieval success.",
    )


class SystemMetricsSummary(BaseModel):
    """System-level efficiency metrics aggregated from a single run."""

    run_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="End-to-end wall-clock runtime for the run in milliseconds.",
    )
    stimulus_count: int = Field(
        default=0, ge=0, description="Total number of stimuli presented to the agent."
    )
    agent_turn_count: int = Field(
        default=0, ge=0, description="Total number of agent turns in the run."
    )
    memory_operation_count: int = Field(
        default=0, ge=0, description="Total memory operations across all components."
    )
    memory_operation_error_count: int = Field(
        default=0,
        ge=0,
        description="Total failed memory operations across all components.",
    )
    memory_operation_skipped_count: int = Field(
        default=0,
        ge=0,
        description="Total skipped memory operations across all components.",
    )
    recall_call_count: int = Field(
        default=0, ge=0, description="Total recall operations across all components."
    )
    retain_call_count: int = Field(
        default=0, ge=0, description="Total retain operations across all components."
    )
    delete_call_count: int = Field(
        default=0, ge=0, description="Total delete operations across all components."
    )
    consolidate_call_count: int = Field(
        default=0,
        ge=0,
        description="Total consolidate operations across all components.",
    )
    llm_call_count: int = Field(
        default=0,
        ge=0,
        description="Total LLM invocations recorded in the agent trace for the run.",
    )
    llm_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input tokens recorded across all LLM calls in the run.",
    )
    llm_output_tokens: int = Field(
        default=0,
        ge=0,
        description="Total output tokens recorded across all LLM calls in the run.",
    )
    llm_total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input plus output tokens recorded across all LLM calls in the run.",
    )
    llm_input_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total input-side model cost recorded across all LLM calls in the run.",
    )
    llm_output_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total output-side model cost recorded across all LLM calls in the run.",
    )
    llm_total_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total model cost recorded across all LLM calls in the run.",
    )
    agent_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of agent-loop durations from the trace, in milliseconds.",
    )
    llm_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of call_llm span durations, in milliseconds.",
    )
    tool_execution_count: int = Field(
        default=0,
        ge=0,
        description="Total execute_tool spans in the agent trace.",
    )
    tool_execution_error_count: int = Field(
        default=0,
        ge=0,
        description="execute_tool spans whose status is error.",
    )
    tool_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Sum of execute_tool span durations, in milliseconds.",
    )
    memory_calls_per_stimulus: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Ratio of memory operations to stimuli presented.",
    )
    task_accuracy_per_memory_call: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Task accuracy score divided by total memory operations.",
    )
    task_success_per_recall_call: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Task success indicator (0.0 or 1.0) divided by recall operations.",
    )


class BaseGroundTruth(BaseModel):
    """Base class for task-specific ground truth payloads."""

    task_description: str


class EvaluationRecord(BaseModel):
    """Generic per-criterion payload consumed by benchmark reports."""

    key: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional stable identifier for the evaluated criterion.",
    )
    passed: bool = Field(description="Whether this criterion passed.")
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        exclude_if=lambda value: value is None,
        description="Optional evaluator score for this criterion.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Benchmark-neutral dimensions used by report builders.",
    )
    expected: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional human-readable expected evidence.",
    )
    observed: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional human-readable produced or observed evidence.",
    )
    reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional evaluator rationale for this criterion.",
    )


class AuditRecord(BaseModel):
    is_successful: bool
    accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final accuracy score computed as passed checks over total checks.",
    )
    passed_keys: list[str] = Field(
        default_factory=list,
        description="Keys, selectors, or targets that passed validation.",
    )
    failed_keys: list[str] = Field(
        default_factory=list,
        description="Keys, selectors, or targets that failed validation.",
    )
    audit_notes: str = Field(
        default="Audit complete.",
        description="Human-readable summary of evaluator decisions and context.",
    )
    score_available: bool = Field(
        default=True,
        description="Whether the run produced a valid evaluator score rather than a runtime or evaluator failure.",
    )
    score_unavailable_reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional machine-readable reason explaining why no valid score was produced.",
    )
    report_records: list[EvaluationRecord] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional generic per-criterion payload for benchmark reports.",
    )
