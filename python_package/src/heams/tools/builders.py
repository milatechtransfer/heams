from __future__ import annotations

from typing import Any

from heams.memory_systems.base import BaseMemorySystem
from heams.tools.spec import (
    BenchmarkToolSpec,
    ToolCategory,
    ToolTracePolicy,
    object_schema,
)


def memory_tool_specs(
    memory: BaseMemorySystem,
    *,
    recall_only: bool = False,
) -> list[BenchmarkToolSpec]:
    recall_spec = BenchmarkToolSpec(
        name="recall",
        description=(
            "Search long-term memory for information related to a query. "
            "Use this before answering when prior user, session, or task "
            "memory may matter."
        ),
        input_schema=object_schema(
            {
                "query": {
                    "type": "string",
                    "description": "The question or keyword to search for.",
                }
            },
            required=["query"],
        ),
        handler=memory.recall,
        category=ToolCategory.MEMORY,
        trace_policy=ToolTracePolicy.SUMMARY,
    )
    if recall_only:
        return [recall_spec]

    return [
        BenchmarkToolSpec(
            name="retain",
            description=(
                "Save a stable fact, user preference, or reusable successful "
                "experience to long-term memory, including lessons from failed "
                "attempts when their resolution is known. Do not save guesses, "
                "unresolved failed attempts, or transient reasoning."
            ),
            input_schema=object_schema(
                {
                    "content": {
                        "type": "string",
                        "description": "The memory content to save.",
                    }
                },
                required=["content"],
            ),
            handler=memory.retain,
            category=ToolCategory.MEMORY,
            trace_policy=ToolTracePolicy.SUMMARY,
        ),
        recall_spec,
        BenchmarkToolSpec(
            name="delete",
            description=(
                "Delete memories matching a specific query from long-term "
                "memory. Use only when the user or benchmark task explicitly "
                "asks to remove memory; do not delete memories just because "
                "they seem irrelevant to the current turn."
            ),
            input_schema=object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": "The query describing which memory or memories to delete.",
                    }
                },
                required=["query"],
            ),
            handler=memory.delete,
            category=ToolCategory.MEMORY,
            trace_policy=ToolTracePolicy.SUMMARY,
        ),
    ]


def evo_task_tool_specs(environment: Any) -> list[BenchmarkToolSpec]:
    # Example AlfWorld LLM-visible exchange after these specs are compiled:
    # assistant tool_call:
    #   get_task_state({})
    # tool result:
    #   {"observation": "You are in the kitchen...", "valid_actions": ["pick up apple", "use fridge"]}
    # assistant tool_call:
    #   step_task({"action": "pick up apple"})
    # tool result:
    #   {"message": "You picked up the apple.", "success": false, "done": false}
    # assistant tool_call:
    #   step_task({"action": "put apple in fridge"})
    # tool result:
    #   {"result": {"task_id": "alfworld_0", "success": true}, "done": true}
    return [
        BenchmarkToolSpec(
            name="get_task_state",
            description="Return the active task state, observation, progress, and valid actions.",
            input_schema=object_schema({}),
            handler=environment.get_task_state,
            category=ToolCategory.ENVIRONMENT,
            trace_policy=ToolTracePolicy.SUMMARY,
        ),
        BenchmarkToolSpec(
            name="get_valid_actions",
            description=(
                "Return valid actions for the current task. Use these actions as "
                "grounding before calling step_task."
            ),
            input_schema=object_schema({}),
            handler=environment.get_valid_actions,
            category=ToolCategory.ENVIRONMENT,
            trace_policy=ToolTracePolicy.SUMMARY,
        ),
        BenchmarkToolSpec(
            name="step_task",
            description=(
                "Execute one concrete environment action for the current task. "
                "Use an action from get_valid_actions, replacing placeholders "
                "such as object or receptacle with concrete names from task state."
            ),
            input_schema=object_schema(
                {
                    "action": {
                        "type": "string",
                        "description": "Concrete environment action to execute.",
                    }
                }
            ),
            handler=environment.step_task,
            category=ToolCategory.ENVIRONMENT,
            trace_policy=ToolTracePolicy.FULL,
        ),
        BenchmarkToolSpec(
            name="finish_task",
            description="Finish the current task and advance the stream when solved.",
            input_schema=object_schema({}),
            handler=environment.finish_task,
            category=ToolCategory.ENVIRONMENT,
            trace_policy=ToolTracePolicy.FULL,
        ),
    ]


def final_answer_tool_spec() -> BenchmarkToolSpec:
    return BenchmarkToolSpec(
        name="final_answer",
        description="Return the final answer to the user and end the agent loop.",
        input_schema=object_schema(
            {
                "answer": {
                    "type": "string",
                    "description": "The final answer to return.",
                }
            }
        ),
        handler=lambda answer: answer,
        category=ToolCategory.CONTROL,
        trace_policy=ToolTracePolicy.NONE,
    )
