"""Factory for configured delete query planners."""

from ..config import DeletePlanningConfig, PlanningMode
from .base import BaseDeleteQueryPlanner
from .llm import LlmDeleteQueryPlanner
from .passthrough import PassthroughDeleteQueryPlanner


def build_delete_query_planner(
    config: DeletePlanningConfig,
) -> BaseDeleteQueryPlanner:
    if config.mode is PlanningMode.NONE:
        return PassthroughDeleteQueryPlanner()
    if config.mode is PlanningMode.LLM:
        return LlmDeleteQueryPlanner(config)
    raise ValueError(f"Unsupported delete planning mode: {config.mode}")
