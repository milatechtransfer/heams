"""Helpers for mining `AgentResponse.raw_trace` objects for tool-call data.

Evaluators rely on the agent trace emitted by the underlying agent framework
(TinyAgent's `AgentTrace`) to reconstruct what happened during a run. Every
tool invocation produces a span with standardised GenAI attributes; these
helpers normalise the extraction of tool name / arguments / output so
evaluators don't have to duplicate that parsing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from heams.agents.base import AgentResponse
from heams.execution_traces import normalized_agent_events


@dataclass(frozen=True)
class ToolCall:
    """Normalised view of a single `execute_tool` span."""

    tool_name: str
    arguments: dict[str, Any]
    raw_output: str
    output: Any


def iter_tool_calls(agent_responses: Iterable[AgentResponse]) -> Iterator[ToolCall]:
    """Yield `ToolCall`s for every `execute_tool` span across the responses
    in call order."""
    for response in agent_responses:
        for event in normalized_agent_events(response.raw_trace):
            if event.type != "tool_call":
                continue

            raw_output = event.content or ""
            yield ToolCall(
                tool_name=event.name,
                arguments=event.arguments,
                raw_output=raw_output,
                output=event.result,
            )
