from __future__ import annotations

import inspect

from heams.environments.alfworld import AlfWorldEnvironment, AlfWorldEnvironmentConfig
from heams.tools import (
    BenchmarkToolSpec,
    ToolCategory,
    ToolTracePolicy,
    compile_tool,
    get_tool_spec,
    memory_tool_specs,
)


class StubMemory:
    def retain(self, content: str) -> str:
        return content

    def recall(self, query: str) -> list[str]:
        return [query]

    def delete(self, query: str) -> str:
        return query

    def consolidate(self, context: str | None) -> str:
        return context or ""


def test_compile_tool_attaches_schema_signature_and_metadata() -> None:
    def handler(name: str) -> dict[str, str]:
        return {"hello": name}

    spec = BenchmarkToolSpec(
        name="greet",
        description="Greet a named person.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=handler,
        category=ToolCategory.DIAGNOSTIC,
        trace_policy=ToolTracePolicy.FULL,
    )

    tool = compile_tool(spec)

    assert tool.__name__ == "greet"
    assert tool.__doc__ == "Greet a named person."
    assert tool.__input_schema__ == spec.input_schema
    assert inspect.signature(tool).parameters["name"].annotation is str
    assert tool(name="Ada") == {"hello": "Ada"}
    assert get_tool_spec("greet") == spec


def test_compile_tool_returns_error_envelope_for_handler_exception() -> None:
    def handler(name: str) -> dict[str, str]:
        raise ValueError(f"bad {name}")

    tool = compile_tool(
        BenchmarkToolSpec(
            name="explode",
            description="Raise a useful tool error.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=handler,
            category=ToolCategory.DIAGNOSTIC,
        )
    )

    assert tool(name="input") == {
        "status": "error",
        "message": "Error calling tool: bad input",
        "error_code": "ValueError",
    }


def test_memory_tool_specs_describe_memory_use_guidance_without_schema_changes() -> (
    None
):
    memory = StubMemory()

    specs = {spec.name: spec for spec in memory_tool_specs(memory)}

    assert set(specs) == {"retain", "recall", "delete"}
    assert "stable fact" in specs["retain"].description
    assert "lessons from failed attempts" in specs["retain"].description
    assert "resolution is known" in specs["retain"].description
    assert "unresolved failed attempts" in specs["retain"].description
    assert "prior user, session, or task memory" in specs["recall"].description
    assert "explicitly asks to remove memory" in specs["delete"].description
    assert "do not delete memories just because" in specs["delete"].description
    assert specs["retain"].input_schema == {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to save.",
            }
        },
        "required": ["content"],
        "additionalProperties": False,
    }
    assert specs["recall"].input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or keyword to search for.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert specs["delete"].input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query describing which memory or memories to delete.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def test_memory_tool_specs_recall_only_exposes_recall():
    class RecallOnlyMemory:
        def recall(self, query: str, top_k=None) -> list[str]:
            return []

    specs = memory_tool_specs(RecallOnlyMemory(), recall_only=True)

    assert [spec.name for spec in specs] == ["recall"]


def test_evomemory_task_tool_specs_preserve_evaluator_outputs() -> None:
    env = AlfWorldEnvironment(
        AlfWorldEnvironmentConfig(
            task_id="TASK",
            tasks=[
                {
                    "task_id": "TASK-0",
                    "goal": "Cool the apple in the fridge.",
                    "kind": "cool",
                }
            ],
        )
    )
    tools = {tool.__name__: tool for tool in env.get_available_tools()}

    assert set(tools) == {
        "get_task_state",
        "get_valid_actions",
        "step_task",
        "finish_task",
    }
    assert get_tool_spec("step_task").trace_policy is ToolTracePolicy.FULL

    pickup = tools["step_task"](action="pick up apple")
    assert pickup["success"] is False

    output = tools["step_task"](action="use fridge")
    assert output["result"]["task_id"] == "TASK-0"
    assert output["result"]["success"] is True


def test_compiled_tool_is_compatible_with_tinyagent_callable_validation() -> None:
    from tinyagent.tools.wrappers import verify_callable

    def handler(value: str) -> str:
        return value

    tool = compile_tool(
        BenchmarkToolSpec(
            name="echo_value",
            description="Echo a string value.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=handler,
            category=ToolCategory.DIAGNOSTIC,
        )
    )

    assert verify_callable(tool) is None
