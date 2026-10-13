"""Storage-independent query-planning contract for logical deletion."""

from abc import ABC, abstractmethod

from ..models import DeleteQueryPlan


class BaseDeleteQueryPlanner(ABC):
    """Interprets a deletion query without accessing memory storage."""

    @abstractmethod
    def plan(self, query: str) -> DeleteQueryPlan:
        """Return a structured interpretation of the deletion request."""
