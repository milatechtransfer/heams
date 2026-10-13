"""Factory for configured logical-delete strategies."""

from ..config import DeleteStrategyConfig, SelectionMode
from .base import (
    BaseDeleteStrategy,
    DeleteStrategyFactory,
)
from .composed import ExactThenTopKLLMDeleteStrategy
from .exact import ExactDeleteStrategy
from .llm_judge import LlmJudgeDeleteStrategy


def build_delete_strategy(
    config: DeleteStrategyConfig,
    *,
    llm_judge_factory: DeleteStrategyFactory | None = None,
) -> BaseDeleteStrategy:
    if config.selection_mode is SelectionMode.EXACT:
        return ExactDeleteStrategy(
            config.exact,
            max_delete_count=config.max_delete_count,
        )

    llm_strategy = (
        llm_judge_factory(config)
        if llm_judge_factory is not None
        else LlmJudgeDeleteStrategy(config)
    )
    if not isinstance(llm_strategy, BaseDeleteStrategy):
        raise TypeError("Delete strategy factory must return BaseDeleteStrategy")

    if config.selection_mode is SelectionMode.TOP_K_LLM:
        return llm_strategy
    if config.selection_mode is SelectionMode.EXACT_THEN_TOP_K_LLM:
        return ExactThenTopKLLMDeleteStrategy(
            ExactDeleteStrategy(
                config.exact,
                max_delete_count=config.max_delete_count,
            ),
            llm_strategy,
            candidate_k=config.top_k_llm.candidate_k,
        )

    raise NotImplementedError(
        f"Delete selection mode is not integrated yet: {config.selection_mode.value}"
    )
