"""Deterministic query planner that performs only surface normalization."""

from __future__ import annotations

import re

from ..models import DeleteQueryPlan
from .base import BaseDeleteQueryPlanner

_COMMAND_PREFIXES = (
    "delete everything about",
    "forget everything about",
    "remove everything about",
    "delete all memories about",
    "forget all memories about",
    "remove all memories about",
    "delete anything about",
    "forget anything about",
    "remove anything about",
    "delete that",
    "forget that",
    "remove that",
    "delete",
    "forget",
    "remove",
)
_ALL_CARDINALITY_PATTERN = re.compile(
    r"\b(?:everything about|all memories about|anything about)\b",
    flags=re.IGNORECASE,
)


class PassthroughDeleteQueryPlanner(BaseDeleteQueryPlanner):
    """Normalizes a request without adding semantic interpretation."""

    def plan(self, query: str) -> DeleteQueryPlan:
        original_query = " ".join(query.split())
        target_text = original_query
        lowered = original_query.casefold()

        for prefix in _COMMAND_PREFIXES:
            if lowered == prefix:
                target_text = ""
                break
            if lowered.startswith(f"{prefix} "):
                target_text = original_query[len(prefix) :].strip()
                break

        cardinality = (
            "all" if _ALL_CARDINALITY_PATTERN.search(original_query) else "one"
        )
        ambiguous = not bool(target_text)
        return DeleteQueryPlan(
            original_query=original_query,
            target_text=target_text,
            search_query=target_text,
            cardinality=cardinality,
            ambiguous=ambiguous,
            reasoning=(
                "No meaningful deletion target remained after normalization."
                if ambiguous
                else "Applied deterministic whitespace and command normalization."
            ),
        )
