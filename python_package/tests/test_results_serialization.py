import json

from heams.evaluation.schemas import (
    AuditRecord,
    ComponentMetricsSummary,
    ComponentPerformanceMetricsSummary,
    EvaluationRecord,
)
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationSource,
    MemoryOperationStatus,
    MemoryOperationType,
)
from heams.results import RunResult


def _audit() -> AuditRecord:
    return AuditRecord(
        is_successful=True,
        accuracy_score=1.0,
        passed_keys=["ok"],
        failed_keys=[],
        audit_notes="serialization test",
    )


def test_memory_observability_models_serialize_cleanly() -> None:
    event = MemoryOperationEvent(
        operation_type=MemoryOperationType.RECALL,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.TINY_AGENT,
        component="example.in_memory",
        duration_ms=12.5,
        payload={"query": "favorite color", "top_k": 3},
        result={"items_returned": 2},
    )
    summary = ComponentMetricsSummary(
        component="example.in_memory",
        total_operations=2,
        successful_operations=2,
        failed_operations=0,
        operation_counts={"recall": 1, "retain": 1},
        status_counts={"success": 2},
        source_counts={"tool_call": 1, "tiny_agent": 1},
        total_duration_ms=12.5,
        retain_success_rate=1.0,
        retain_operations_per_stimulus=0.5,
        retain_bytes_per_successful_retain=42.0,
    )

    payload = json.loads(
        json.dumps(
            {
                "event": json.loads(event.model_dump_json()),
                "summary": json.loads(summary.model_dump_json()),
            }
        )
    )

    assert payload["event"]["operation_type"] == "recall"
    assert payload["event"]["status"] == "success"
    assert payload["event"]["source"] == "tiny_agent"
    assert payload["event"]["payload"] == {"query": "favorite color", "top_k": 3}
    assert payload["summary"]["component"] == "example.in_memory"
    assert payload["summary"]["operation_counts"] == {"recall": 1, "retain": 1}
    assert payload["summary"]["retain_success_rate"] == 1.0
    assert payload["summary"]["retain_operations_per_stimulus"] == 0.5
    assert payload["summary"]["retain_bytes_per_successful_retain"] == 42.0


def test_new_memory_operation_types_serialize_cleanly() -> None:
    delete_event = MemoryOperationEvent(
        operation_type=MemoryOperationType.DELETE,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.TOOL_CALL,
        component="example.in_memory",
        duration_ms=4.0,
        payload={"memory_id": "fact-1"},
        result={"deleted_count": 1},
    )
    consolidate_event = MemoryOperationEvent(
        operation_type=MemoryOperationType.CONSOLIDATE,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.RUNTIME,
        component="example.in_memory",
        duration_ms=9.5,
        payload={"batch_size": 3},
        result={"merged_count": 2},
    )

    payload = json.loads(
        json.dumps(
            {
                "delete": json.loads(delete_event.model_dump_json()),
                "consolidate": json.loads(consolidate_event.model_dump_json()),
            }
        )
    )

    assert payload["delete"]["operation_type"] == "delete"
    assert payload["delete"]["source"] == "tool_call"
    assert payload["consolidate"]["operation_type"] == "consolidate"
    assert payload["consolidate"]["source"] == "runtime"
    assert (
        MemoryOperationEvent.model_validate(payload["delete"]).operation_type
        == MemoryOperationType.DELETE
    )
    assert (
        MemoryOperationEvent.model_validate(payload["consolidate"]).operation_type
        == MemoryOperationType.CONSOLIDATE
    )


def test_run_result_model_dump_json_works_without_observability_fields() -> None:
    run = RunResult(
        scenario_id="scenario-a",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=_audit(),
    )

    payload = json.loads(run.model_dump_json())

    assert payload["scenario_id"] == "scenario-a"
    assert payload["audit"]["accuracy_score"] == 1.0
    assert payload["audit"]["score_available"] is True
    assert "report_records" not in payload["audit"]
    assert "component_metrics" not in payload
    assert "component_performance_metrics" not in payload
    assert "memory_ledger" not in payload


def test_audit_report_records_serialize_when_present() -> None:
    audit = AuditRecord(
        is_successful=False,
        accuracy_score=0.5,
        passed_keys=["criterion-a"],
        failed_keys=["criterion-b"],
        audit_notes="report payload test",
        report_records=[
            EvaluationRecord(
                key="criterion-a",
                passed=True,
                score=1.0,
                tags={"dimension": "completeness", "scenario_type": "direct"},
                expected="Include the requested fact.",
                observed="Included the requested fact.",
            ),
            EvaluationRecord(
                key="criterion-b",
                passed=False,
                score=0.0,
                tags={"dimension": "data_minimization", "scenario_type": "direct"},
                reason="Disclosed an unrelated memory.",
            ),
        ],
    )
    payload = json.loads(audit.model_dump_json())
    round_tripped = AuditRecord.model_validate_json(audit.model_dump_json())

    assert payload["report_records"][0]["passed"] is True
    assert payload["report_records"][0]["tags"]["dimension"] == "completeness"
    assert payload["report_records"][1]["reason"] == "Disclosed an unrelated memory."
    assert "reason" not in payload["report_records"][0]
    assert round_tripped.report_records is not None
    assert round_tripped.report_records[1].tags["dimension"] == "data_minimization"


def test_run_result_model_dump_json_works_with_observability_fields() -> None:
    summary = ComponentMetricsSummary(
        component="example.in_memory",
        total_operations=3,
        successful_operations=2,
        failed_operations=1,
        operation_counts={"recall": 2, "retain": 1},
        status_counts={"success": 2, "error": 1},
        source_counts={"tool_call": 2, "tiny_agent": 1},
        total_duration_ms=8.75,
        retain_success_rate=1.0,
        retain_operations_per_stimulus=1.0,
        retain_bytes_per_successful_retain=42.0,
    )
    event = MemoryOperationEvent(
        operation_type=MemoryOperationType.RETAIN,
        status=MemoryOperationStatus.ERROR,
        source=MemoryOperationSource.TOOL_CALL,
        component="example.in_memory",
        payload={"content_length": 42},
        error="backend unavailable",
    )
    run = RunResult(
        scenario_id="scenario-b",
        agent_index=0,
        memory_index=1,
        evaluator_index=0,
        sample_index=0,
        audit=_audit(),
        component_metrics=[summary],
        component_performance_metrics=[
            ComponentPerformanceMetricsSummary(
                component="example.in_memory",
                recall_eval_count=2,
                recall_missing_stimulus_index_count=1,
                recall_hit_at_k=0.5,
                recall_mrr=0.25,
            )
        ],
        memory_ledger=[event],
    )

    payload = json.loads(run.model_dump_json())
    round_tripped = RunResult.model_validate_json(run.model_dump_json())

    assert payload["component_metrics"][0]["component"] == "example.in_memory"
    assert payload["component_performance_metrics"][0]["recall_eval_count"] == 2
    assert (
        payload["component_performance_metrics"][0][
            "recall_missing_stimulus_index_count"
        ]
        == 1
    )
    assert payload["memory_ledger"][0]["operation_type"] == "retain"
    assert payload["memory_ledger"][0]["status"] == "error"
    assert payload["memory_ledger"][0]["source"] == "tool_call"
    assert payload["audit"]["score_available"] is True
    assert round_tripped.component_metrics is not None
    assert round_tripped.component_performance_metrics is not None
    assert round_tripped.memory_ledger is not None
    assert (
        round_tripped.component_performance_metrics[
            0
        ].recall_missing_stimulus_index_count
        == 1
    )


def test_run_result_model_dump_json_works_with_extended_operation_fields() -> None:
    summary = ComponentMetricsSummary(
        component="example.in_memory",
        total_operations=5,
        successful_operations=4,
        failed_operations=1,
        operation_counts={
            "delete": 2,
            "consolidate": 2,
            "retain": 1,
        },
        status_counts={"success": 4, "error": 1},
        source_counts={"tool_call": 3, "runtime": 2},
        total_duration_ms=18.5,
        delete_success_rate=0.5,
        delete_operations_per_stimulus=1.0,
        avg_delete_duration_ms=3.5,
        delete_errors=1,
        consolidate_success_rate=1.0,
        consolidate_operations_per_stimulus=1.0,
        avg_consolidate_duration_ms=6.0,
        consolidate_errors=0,
    )
    delete_event = MemoryOperationEvent(
        operation_type=MemoryOperationType.DELETE,
        status=MemoryOperationStatus.ERROR,
        source=MemoryOperationSource.TOOL_CALL,
        component="example.in_memory",
        payload={"memory_id": "fact-1"},
        error="missing record",
    )
    consolidate_event = MemoryOperationEvent(
        operation_type=MemoryOperationType.CONSOLIDATE,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.RUNTIME,
        component="example.in_memory",
        duration_ms=6.0,
        payload={"batch_size": 3},
        result={"merged_count": 2},
    )
    run = RunResult(
        scenario_id="scenario-d",
        agent_index=0,
        memory_index=1,
        evaluator_index=0,
        sample_index=0,
        audit=_audit(),
        component_metrics=[summary],
        memory_ledger=[delete_event, consolidate_event],
    )

    payload = json.loads(run.model_dump_json())
    round_tripped = RunResult.model_validate_json(run.model_dump_json())

    assert payload["component_metrics"][0]["operation_counts"]["delete"] == 2
    assert payload["component_metrics"][0]["operation_counts"]["consolidate"] == 2
    assert payload["component_metrics"][0]["consolidate_success_rate"] == 1.0
    assert payload["component_metrics"][0]["delete_success_rate"] == 0.5
    assert payload["component_metrics"][0]["avg_delete_duration_ms"] == 3.5
    assert payload["component_metrics"][0]["avg_consolidate_duration_ms"] == 6.0
    assert payload["component_metrics"][0]["delete_errors"] == 1
    assert payload["component_metrics"][0]["consolidate_errors"] == 0
    assert payload["memory_ledger"][0]["operation_type"] == "delete"
    assert payload["memory_ledger"][1]["operation_type"] == "consolidate"
    assert round_tripped.component_metrics is not None
    assert round_tripped.memory_ledger is not None
    assert round_tripped.component_metrics[0].delete_success_rate == 0.5
    assert round_tripped.component_metrics[0].consolidate_success_rate == 1.0
    assert round_tripped.memory_ledger[0].operation_type == MemoryOperationType.DELETE
    assert (
        round_tripped.memory_ledger[1].operation_type == MemoryOperationType.CONSOLIDATE
    )


def test_audit_failure_metadata_serializes_cleanly() -> None:
    run = RunResult(
        scenario_id="scenario-c",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=0.0,
            passed_keys=[],
            failed_keys=["audit_error: RuntimeError"],
            audit_notes="Evaluator raised: boom",
            score_available=False,
            score_unavailable_reason="audit_error: RuntimeError",
        ),
    )

    payload = json.loads(run.model_dump_json())

    assert payload["audit"]["score_available"] is False
    assert payload["audit"]["score_unavailable_reason"] == "audit_error: RuntimeError"
