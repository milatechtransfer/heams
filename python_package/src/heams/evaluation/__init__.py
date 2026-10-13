"""Evaluation package.

Keep this module light: ``factory`` imports every benchmark judge, and those
judges import ``heams.evaluation.*`` submodules. Loading ``factory`` eagerly here
would cycle whenever a judge module is imported before ``heams.evaluation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heams.evaluation.base import BaseEvaluator

if TYPE_CHECKING:
    from heams.evaluation.factory import EvaluatorConfig, create_evaluator


__all__ = [
    "BaseEvaluator",
    "EvaluatorConfig",
    "create_evaluator",
]


def __getattr__(name: str) -> Any:
    if name in ("EvaluatorConfig", "create_evaluator"):
        from heams.evaluation import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
