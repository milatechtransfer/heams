from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from enum import Enum
from functools import wraps
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCategory(str, Enum):
    ENVIRONMENT = "environment"
    MEMORY = "memory"
    CONTROL = "control"
    DIAGNOSTIC = "diagnostic"


class ToolTracePolicy(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    message: str | None = None
    data: Any = None
    done: bool | None = None
    error_code: str | None = None


class BenchmarkToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any]
    category: ToolCategory
    trace_policy: ToolTracePolicy = ToolTracePolicy.NONE
    metadata: dict[str, Any] = Field(default_factory=dict)


_TOOL_SPEC_ATTR = "__benchmark_tool_spec__"
_TOOL_INPUT_SCHEMA_ATTR = "__input_schema__"
_SPEC_REGISTRY: dict[str, BenchmarkToolSpec] = {}


def compile_tool(spec: BenchmarkToolSpec) -> Callable[..., Any]:
    """Compile a benchmark tool spec into a TinyAgent compatible callable."""

    @wraps(spec.handler)
    def tool(*args: Any, **kwargs: Any) -> Any:
        try:
            call_kwargs = _coerce_tool_kwargs(spec, args, kwargs)
            result = spec.handler(**call_kwargs)
        except Exception as exc:
            return ToolResultEnvelope(
                status="error",
                message=f"Error calling tool: {exc}",
                error_code=type(exc).__name__,
            ).model_dump(exclude_none=True)
        if isinstance(result, ToolResultEnvelope):
            return result.model_dump(exclude_none=True)
        return result

    tool.__name__ = spec.name
    tool.__doc__ = spec.description
    tool.__annotations__ = _annotations_from_schema(spec.input_schema)
    tool.__signature__ = _signature_from_schema(spec.input_schema)  # type: ignore[attr-defined]
    setattr(tool, _TOOL_INPUT_SCHEMA_ATTR, spec.input_schema)
    setattr(tool, _TOOL_SPEC_ATTR, spec)
    _SPEC_REGISTRY[spec.name] = spec
    return tool


def _coerce_tool_kwargs(
    spec: BenchmarkToolSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not args:
        return kwargs

    properties = spec.input_schema.get("properties")
    ordered_names = (
        [str(name) for name in properties] if isinstance(properties, Mapping) else []
    )

    if len(args) > len(ordered_names):
        raise TypeError(
            f"{spec.name}() accepts at most {len(ordered_names)} positional argument(s)"
        )

    call_kwargs = dict(kwargs)
    for index, value in enumerate(args):
        param_name = ordered_names[index]
        if param_name in call_kwargs:
            raise TypeError(
                f"{spec.name}() got multiple values for argument '{param_name}'"
            )
        call_kwargs[param_name] = value

    return call_kwargs


def get_tool_spec(tool: Callable[..., Any] | str) -> BenchmarkToolSpec | None:
    if callable(tool):
        spec = getattr(tool, _TOOL_SPEC_ATTR, None)
        if isinstance(spec, BenchmarkToolSpec):
            return spec
        tool_name = getattr(tool, "__name__", "")
    else:
        tool_name = tool
    return _SPEC_REGISTRY.get(tool_name)


def should_trace_tool(tool_name: str) -> bool:
    spec = get_tool_spec(tool_name)
    if spec is None:
        return False
    return spec.trace_policy is not ToolTracePolicy.NONE


def tool_names_for_trace_policy(
    *policies: ToolTracePolicy,
) -> set[str]:
    allowed = set(policies)
    return {
        name for name, spec in _SPEC_REGISTRY.items() if spec.trace_policy in allowed
    }


def object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: dict(value) for key, value in properties.items()},
        "required": required if required is not None else list(properties),
        "additionalProperties": additional_properties,
    }


def _signature_from_schema(input_schema: Mapping[str, Any]) -> inspect.Signature:
    properties = input_schema.get("properties")
    required = set(input_schema.get("required") or [])
    if not isinstance(properties, Mapping):
        properties = {}

    parameters = []
    annotations = _annotations_from_schema(input_schema)
    for name in properties:
        default = inspect.Parameter.empty if name in required else None
        parameters.append(
            inspect.Parameter(
                name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotations.get(name, Any),
            )
        )
    return inspect.Signature(parameters, return_annotation=Any)


def _annotations_from_schema(input_schema: Mapping[str, Any]) -> dict[str, Any]:
    annotations: dict[str, Any] = {"return": Any}
    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        return annotations
    for name, schema in properties.items():
        annotations[str(name)] = _annotation_from_property(schema)
    return annotations


def _annotation_from_property(schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return str
    property_type = schema.get("type")
    if isinstance(property_type, list):
        non_null = [item for item in property_type if item != "null"]
        if non_null == ["integer"]:
            return int | None
        if non_null == ["number"]:
            return float | None
        if non_null == ["boolean"]:
            return bool | None
        return str | None
    if property_type == "integer":
        return int
    if property_type == "number":
        return float
    if property_type == "boolean":
        return bool
    if property_type == "object":
        return dict[str, Any]
    if property_type == "array":
        return list[Any]
    return str
