from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from heams.agents.base import AgentResponse

from .schemas import AuditRecord, BaseGroundTruth


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Evaluators score agent performance by inspecting the agent's tool-call
    trace. All environment state transitions are captured in the trace as
    tool arguments/outputs, so the trace is the single source of truth -
    evaluators do not need direct access to the environment.
    """

    def __init__(self, task_name: str, config: BaseModel | None = None):
        self.task_name = task_name
        # `config` lets evaluators read per-task knobs (e.g. judge model id)
        # without coupling the factory signature to each evaluator family.
        self.config = config
        self._last_audit: AuditRecord | None = None

    @abstractmethod
    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: BaseGroundTruth,
    ) -> AuditRecord:
        """Parse the agent trace(s) and score them against the ground truth.

        Implementations must populate `self._last_audit` and return it.
        """

    def get_metrics(self) -> dict[str, Any]:
        """Standardized telemetry export."""
        if not self._last_audit:
            raise ValueError(
                f"Cannot get metrics. You must run `audit()` on '{self.task_name}' first."
            )

        return {
            "eval_task": self.task_name,
            "is_successful": self._last_audit.is_successful,
            "accuracy": self._last_audit.accuracy_score,
            "error_rate": round(1.0 - self._last_audit.accuracy_score, 2),
            "total_fields_checked": len(self._last_audit.passed_keys)
            + len(self._last_audit.failed_keys),
            "failed_fields": self._last_audit.failed_keys,
            "notes": self._last_audit.audit_notes,
        }
