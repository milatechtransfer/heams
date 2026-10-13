from __future__ import annotations

import json

import pytest

from heams_memory_milamem import design
from heams_memory_milamem.design import delete
from heams_memory_milamem.design.delete.models import (
    DeleteOutcome,
    DeletePlan,
    DeleteReceipt,
    DeleteVerification,
)


@pytest.fixture
def plan() -> DeletePlan:
    return DeletePlan(
        query="Forget jasmine tea.",
        strategy="llm_judge",
        candidate_ids=["memory-1", "memory-2"],
        selected_ids=["memory-1"],
        reasoning="The first candidate is directly targeted.",
        ambiguous=False,
        executable=True,
    )


def test_delete_lifecycle_defaults_are_independent(plan: DeletePlan) -> None:
    first_receipt = DeleteReceipt(
        requested_ids=["memory-1"],
        deleted_ids=[],
        skipped_ids=["memory-1"],
    )
    first_verification = DeleteVerification(status="not_run")
    second_verification = DeleteVerification(status="not_run")
    outcome = DeleteOutcome(plan=plan, verification=first_verification)

    first_verification.remaining_active_ids.append("memory-1")

    assert first_receipt.raw_erasure is False
    assert second_verification.remaining_active_ids == []
    assert outcome.receipt is None
    assert outcome.changed is False


def test_delete_outcome_serializes_and_round_trips(plan: DeletePlan) -> None:
    outcome = DeleteOutcome(
        plan=plan,
        receipt=DeleteReceipt(
            requested_ids=["memory-1"],
            deleted_ids=["memory-1"],
            skipped_ids=[],
        ),
        verification=DeleteVerification(status="verified"),
    )

    payload = json.loads(outcome.model_dump_json())
    restored = DeleteOutcome.model_validate(payload)

    assert payload["plan"]["strategy"] == "llm_judge"
    assert payload["receipt"]["raw_erasure"] is False
    assert payload["verification"] == {
        "status": "verified",
        "remaining_active_ids": [],
    }
    assert "changed" not in payload
    assert restored == outcome
    assert restored.changed is True


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        DeleteReceipt(requested_ids=[], deleted_ids=[], skipped_ids=[]),
        DeleteReceipt(
            requested_ids=["memory-1"],
            deleted_ids=[],
            skipped_ids=["memory-1"],
        ),
    ],
)
def test_delete_outcome_changed_is_false_without_deleted_ids(
    plan: DeletePlan,
    receipt: DeleteReceipt | None,
) -> None:
    outcome = DeleteOutcome(
        plan=plan,
        receipt=receipt,
        verification=DeleteVerification(status="not_run"),
    )

    assert outcome.changed is False


def test_delete_outcome_changed_is_true_with_deleted_ids(plan: DeletePlan) -> None:
    outcome = DeleteOutcome(
        plan=plan,
        receipt=DeleteReceipt(
            requested_ids=["memory-1"],
            deleted_ids=["memory-1"],
            skipped_ids=[],
        ),
        verification=DeleteVerification(status="verified"),
    )

    assert outcome.changed is True


@pytest.mark.parametrize("status", ["failed", "not_run"])
def test_delete_verification_preserves_non_verified_states(status: str) -> None:
    verification = DeleteVerification(
        status=status,
        remaining_active_ids=["memory-1"],
    )

    assert verification.model_dump() == {
        "status": status,
        "remaining_active_ids": ["memory-1"],
    }


def test_delete_lifecycle_models_are_exported() -> None:
    assert delete.DeletePlan is DeletePlan
    assert delete.DeleteReceipt is DeleteReceipt
    assert delete.DeleteVerification is DeleteVerification
    assert delete.DeleteOutcome is DeleteOutcome
    assert design.DeletePlan is DeletePlan
    assert design.DeleteReceipt is DeleteReceipt
    assert design.DeleteVerification is DeleteVerification
    assert design.DeleteOutcome is DeleteOutcome
