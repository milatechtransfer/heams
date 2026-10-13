import asyncio
import inspect
import json
import re
import warnings
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, List

from tinyagent import AgentCancel, AgentConfig
from tinyagent import TinyAgent as TinyAgentRuntime
from tinyagent.callbacks import Callback, Context, get_default_callbacks

from .base import AgentResponse, BaseAgent
from .config import AgentParameters, MemoryStrategy
from heams.memory_systems.base import BaseMemorySystem
from heams.tools import BenchmarkToolSpec, ToolCategory, compile_tool, get_tool_spec
from heams.utils.settings import settings

from heams.utils import register


class ToolExecutionLimitReached(AgentCancel):
    """Raised when the configured tool execution cap is exceeded."""


class _OwnedEventLoop:
    """Run a synchronous TinyAgent lifecycle on one explicitly owned loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._closed = False

    def run(self, operation: Callable[[], Any]) -> Any:
        if self._closed:
            raise RuntimeError("The agent event loop is closed")

        previous_loop = self._current_loop()
        asyncio.set_event_loop(self.loop)
        try:
            result = operation()
            if inspect.isawaitable(result):
                return self.loop.run_until_complete(result)
            return result
        finally:
            asyncio.set_event_loop(previous_loop)

    def close(self, cleanup: Callable[[], Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True

        errors: list[BaseException] = []
        previous_loop = self._current_loop()
        asyncio.set_event_loop(self.loop)
        try:
            if cleanup is not None:
                try:
                    result = cleanup()
                    if inspect.isawaitable(result):
                        self.loop.run_until_complete(result)
                except BaseException as error:
                    errors.append(error)

            try:
                pending_tasks = [
                    task for task in asyncio.all_tasks(self.loop) if not task.done()
                ]
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    self.loop.run_until_complete(
                        asyncio.gather(*pending_tasks, return_exceptions=True)
                    )
            except BaseException as error:
                errors.append(error)

            try:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            except BaseException as error:
                errors.append(error)

            try:
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
            except BaseException as error:
                errors.append(error)
        finally:
            try:
                asyncio.set_event_loop(previous_loop)
                self.loop.close()
            except BaseException as error:
                errors.append(error)

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("Errors while closing event loop", errors)

    @staticmethod
    def _current_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass

        policy_local = getattr(asyncio.get_event_loop_policy(), "_local", None)
        if policy_local is None or not getattr(policy_local, "_set_called", False):
            return None

        loop = getattr(policy_local, "_loop", None)
        if loop is None or loop.is_closed():
            return None
        return loop


_ACTIVE_RUN_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "tiny_agent_run_context",
    default=None,
)
_ACTIVE_RUN_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "tiny_agent_run_metadata",
    default=None,
)


class LimitToolExecutions(Callback):
    def __init__(self, max_tool_executions: int) -> None:
        self.max_tool_executions = max_tool_executions

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        context.shared["n_tool_executions"] = (
            context.shared.get("n_tool_executions", 0) + 1
        )
        if context.shared["n_tool_executions"] > self.max_tool_executions:
            raise ToolExecutionLimitReached(
                f"Exceeded {self.max_tool_executions} tool executions"
            )
        return context


class ContextBudgetGuard(Callback):
    """Trim recoverable message history before each model request."""

    def __init__(self, max_input_tokens: int, approx_chars_per_token: int) -> None:
        self.max_input_tokens = max_input_tokens
        self.approx_chars_per_token = approx_chars_per_token

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return context

        estimated_tokens = self._estimate_request_tokens(kwargs)
        if estimated_tokens <= self.max_input_tokens:
            return context

        context.shared["context_truncated"] = True
        run_metadata = _ACTIVE_RUN_METADATA.get()
        if run_metadata is not None:
            run_metadata["context_truncated"] = True

        original_messages = list(messages)
        protected_prefix, protected_suffix = self._protected_messages(original_messages)
        removable_messages = original_messages[
            len(protected_prefix) : -len(protected_suffix) if protected_suffix else None
        ]

        while removable_messages:
            remove_count = self._first_turn_length(removable_messages)
            if remove_count <= 0:
                break
            del removable_messages[:remove_count]
            candidate_messages = [
                *protected_prefix,
                *removable_messages,
                *protected_suffix,
            ]
            if (
                self._estimate_request_tokens(
                    {**kwargs, "messages": candidate_messages}
                )
                <= self.max_input_tokens
            ):
                messages[:] = candidate_messages
                return context

        messages[:] = [*protected_prefix, *protected_suffix]
        return context

    def _estimate_request_tokens(self, request: dict[str, Any]) -> int:
        serialized = json.dumps(
            {
                "messages": request.get("messages", []),
                "tools": request.get("tools", []),
            },
            default=str,
            separators=(",", ":"),
        )
        return max(
            1,
            (len(serialized) + self.approx_chars_per_token - 1)
            // self.approx_chars_per_token,
        )

    @staticmethod
    def _protected_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prefix_length = 0
        while (
            prefix_length < len(messages)
            and messages[prefix_length].get("role") == "system"
        ):
            prefix_length += 1

        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, prefix_length - 1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        return (
            messages[:prefix_length],
            messages[last_user_index:] if last_user_index is not None else [],
        )

    @staticmethod
    def _first_turn_length(messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0
        for index, message in enumerate(messages[1:], start=1):
            if message.get("role") == "user":
                return index
        return len(messages)


@register("agent")
class TinyAgent(BaseAgent):
    """
    Benchmark agent hosted by mozilla-ai/tinyagent.
    Allows for stateful conversation management by maintaining a local history of messages.
    """

    def __init__(
        self,
        config: AgentParameters,
        tools: List[Callable[..., Any]] | None = None,
        memory_system: BaseMemorySystem | None = None,
    ):
        if memory_system is None and config.memory_strategy != MemoryStrategy.NO_MEMORY:
            raise ValueError(
                "memory_system is required unless memory_strategy is NO_MEMORY"
            )
        self._raise_if_running_loop()

        self.config = config
        self.tools = tools or []
        self.memory_system = memory_system
        self.messages: List[Dict[str, str]] = self._seed_messages()

        prepared_tools = [self._contextualize_tool(tool) for tool in self.tools]
        agent_config_kwargs: Dict[str, Any] = dict(
            model_id=self.config.model_id,
            instructions=self.config.instructions,
            model_args=self.config.model_args,
        )
        if prepared_tools:
            agent_config_kwargs["tools"] = prepared_tools
        agent_config_kwargs["callbacks"] = self._build_callbacks()

        self._loop_owner = _OwnedEventLoop()
        try:
            self.agent = self._loop_owner.run(
                lambda: TinyAgentRuntime.create_async(
                    self._build_agent_config(agent_config_kwargs),
                )
            )
        except BaseException:
            self._loop_owner.close()
            raise

    def _build_agent_config(self, agent_config_kwargs: Dict[str, Any]) -> AgentConfig:
        resolved_kwargs = self._inject_provider_secrets(agent_config_kwargs)
        tools = resolved_kwargs.get("tools")
        if tools:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="^SSE is deprecated in the MCP specification",
                    category=DeprecationWarning,
                )
                return AgentConfig(**resolved_kwargs)
        return AgentConfig(**resolved_kwargs)

    def _inject_provider_secrets(
        self, agent_config_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill provider secrets from GlobalSettings at the last possible boundary."""

        resolved_kwargs = dict(agent_config_kwargs)
        if resolved_kwargs.get("api_key"):
            return resolved_kwargs

        provider_name = self._provider_name_from_model_id(self.config.model_id)
        if provider_name == "openai":
            configured_key = settings.openai_api_key
            if configured_key is not None:
                resolved_kwargs["api_key"] = configured_key.get_secret_value()
        return resolved_kwargs

    @staticmethod
    def _provider_name_from_model_id(model_id: str) -> str | None:
        if ":" not in model_id:
            return None
        provider_name, _, _model_name = model_id.partition(":")
        return provider_name.casefold() or None

    def _seed_messages(self) -> List[Dict[str, str]]:
        """Return the initial message list, prefixed with the system prompt.

        TinyAgent only injects `config.instructions`
        into the message list when the caller passes a string prompt; when
        it receives a list (which we do, to keep per-turn history across
        scenarios), it treats the list as authoritative and skips the
        system-message step entirely. Seeding the list ourselves is the
        only way to keep the system prompt in front of the model.
        """
        if not self.config.instructions:
            return []
        return [{"role": "system", "content": self.config.instructions}]

    def _bind_memory_run_context(
        self,
        run_context: dict[str, Any] | None = None,
    ) -> Any:
        if self.memory_system is None:
            return None
        resolved_context = (
            dict(run_context) if run_context is not None else _ACTIVE_RUN_CONTEXT.get()
        )
        return self.memory_system.bind_run_context(resolved_context)

    def _reset_memory_run_context(self, token: Any) -> None:
        if self.memory_system is None:
            return
        self.memory_system.reset_run_context(token)

    def _contextualize_tool(self, tool: Callable[..., Any]) -> Callable[..., Any]:
        if (
            self.memory_system is not None
            and getattr(tool, "__self__", None) is self.memory_system
            and getattr(tool, "__name__", "")
            in {"retain", "recall", "delete", "consolidate"}
        ):

            @wraps(tool)
            def wrapped_tool(*args: Any, **kwargs: Any) -> Any:
                token = self._bind_memory_run_context()
                try:
                    return tool(*args, **kwargs)
                finally:
                    self._reset_memory_run_context(token)

            wrapped_tool.__self__ = self.memory_system
            return wrapped_tool

        spec = get_tool_spec(tool)
        if spec is not None and spec.category is ToolCategory.MEMORY:

            @wraps(spec.handler)
            def handler(*args: Any, **kwargs: Any) -> Any:
                token = self._bind_memory_run_context()
                try:
                    return spec.handler(*args, **kwargs)
                finally:
                    self._reset_memory_run_context(token)

            contextual_spec = BenchmarkToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema=dict(spec.input_schema),
                handler=handler,
                category=spec.category,
                trace_policy=spec.trace_policy,
                metadata=dict(spec.metadata),
            )
            return compile_tool(contextual_spec)

        return tool

    def _build_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        if self.config.max_tool_executions is not None:
            callbacks.append(LimitToolExecutions(self.config.max_tool_executions))
        if self.config.max_input_tokens is not None:
            callbacks.append(
                ContextBudgetGuard(
                    self.config.max_input_tokens,
                    self.config.approx_chars_per_token,
                )
            )
        if self.config.enable_console_trace:
            callbacks.extend(get_default_callbacks())
        return callbacks

    @staticmethod
    def _format_recalled_memory(entries: List[str]) -> str:
        if not entries:
            return "No relevant memory found."
        return "\n\n".join(
            f"[Memory {index}]\n{entry}" for index, entry in enumerate(entries, start=1)
        )

    @staticmethod
    def _deterministic_memory_prefix() -> str:
        return (
            "Relevant memory context:\n"
            "Use these memories only if relevant. Ignore irrelevant or "
            "contradictory memories.\n\n"
        )

    @staticmethod
    def _deterministic_memory_suffix(prompt: str) -> str:
        return f"\n\nUser query: {prompt}"

    @staticmethod
    def _is_explicit_delete_request(prompt: str) -> bool:
        normalized = " ".join(prompt.split())
        sentence_boundary = r"(?:^|[.!?]\s+)"
        polite_prefix = r"(?:(?:please|can you|could you|would you)\s+)?"
        patterns = (
            r"^/(?:forget|delete-memory|erase-memory|clear-memory|purge-memory)\s+\S",
            rf"{sentence_boundary}{polite_prefix}"
            r"forget\s+(?!it(?:\b|,)|about\s+it\b|to\b)[^.!?\n]+",
            rf"{sentence_boundary}{polite_prefix}"
            r"(?:delete|remove|erase|clear|purge|wipe)\s+"
            r"[^,;.!?\n]+?\s+from\s+your\s+memory\b",
            rf"{sentence_boundary}{polite_prefix}"
            r"i\s+(?:want|need|would like)\s+you\s+to\s+forget\b",
        )
        return any(
            re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns
        )

    @staticmethod
    def _deterministic_delete_prompt(prompt: str, receipt: str) -> str:
        return (
            "Memory deletion result:\n"
            f"{receipt}\n\n"
            "Respond briefly to the user's request. Do not repeat deleted "
            "information.\n\n"
            f"User query: {prompt}"
        )

    def _delete_deterministic_request(
        self,
        prompt: str,
        *,
        run_context: dict[str, Any] | None = None,
    ) -> str | None:
        if (
            not self.config.memory_strategy.uses_deterministic_delete
            or self.memory_system is None
            or not self._is_explicit_delete_request(prompt)
        ):
            return None
        token = self._bind_memory_run_context(run_context)
        try:
            return self.memory_system.delete(prompt)
        finally:
            self._reset_memory_run_context(token)

    def _build_turn_prompt(
        self,
        prompt: str,
        *,
        run_context: dict[str, Any] | None = None,
    ) -> str:
        if (
            not self.config.memory_strategy.uses_deterministic_recall
            or self.memory_system is None
        ):
            return prompt
        token = self._bind_memory_run_context(run_context)
        try:
            recalled_entries = self.memory_system.recall(
                query=prompt,
                top_k=self.memory_system.config.top_k,
            )
        finally:
            self._reset_memory_run_context(token)
        recall_block = self._format_recalled_memory(recalled_entries)
        return (
            f"{self._deterministic_memory_prefix()}"
            f"{recall_block}"
            f"{self._deterministic_memory_suffix(prompt)}"
        )

    @staticmethod
    def _should_retain_deterministic_turn(prompt: str, answer: str) -> bool:
        return (
            bool(prompt.strip())
            and bool(answer.strip())
            and not answer.lstrip().startswith("[agent_error:")
        )

    def _retain_deterministic_turn(
        self,
        prompt: str,
        answer: str,
        *,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        if (
            self.config.memory_strategy
            in {MemoryStrategy.DETERMINISTIC, MemoryStrategy.DETERMINISTIC_WITH_DELETE}
            and self.memory_system is not None
        ):
            if not self._should_retain_deterministic_turn(prompt, answer):
                return
            token = self._bind_memory_run_context(run_context)
            try:
                self.memory_system.retain(content=f"Q: {prompt}\nA: {answer}")
            finally:
                self._reset_memory_run_context(token)

    def run(
        self, prompt: str, *, run_context: dict[str, Any] | None = None
    ) -> AgentResponse:
        self._raise_if_running_loop()
        delete_receipt = self._delete_deterministic_request(
            prompt,
            run_context=run_context,
        )
        turn_prompt = (
            self._deterministic_delete_prompt(prompt, delete_receipt)
            if delete_receipt is not None
            else self._build_turn_prompt(prompt, run_context=run_context)
        )
        self.messages.append({"role": "user", "content": turn_prompt})
        active_run_context = dict(run_context) if run_context is not None else None
        run_metadata: dict[str, Any] = {"context_truncated": False}
        token = _ACTIVE_RUN_CONTEXT.set(active_run_context)
        metadata_token = _ACTIVE_RUN_METADATA.set(run_metadata)
        try:
            trace = self._loop_owner.run(lambda: self.agent.run(list(self.messages)))
        finally:
            _ACTIVE_RUN_CONTEXT.reset(token)
            _ACTIVE_RUN_METADATA.reset(metadata_token)
        answer = getattr(trace, "output", None) or getattr(trace, "final_output", "")
        self.messages.append({"role": "assistant", "content": answer})
        if delete_receipt is None:
            self._retain_deterministic_turn(prompt, answer, run_context=run_context)
        return AgentResponse(
            answer=answer,
            raw_trace=trace,
            context_truncated=run_metadata["context_truncated"],
        )

    def reset_state(self) -> None:
        self.messages = self._seed_messages()

    @staticmethod
    def _raise_if_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "TinyAgent is synchronous and cannot be used from an async context"
        )

    async def _cleanup_agent_resources(self) -> None:
        errors: list[BaseException] = []
        try:
            cleanup = getattr(self.agent, "cleanup_async", None)
            if callable(cleanup):
                maybe_awaitable = cleanup()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
        except BaseException as error:
            errors.append(error)

        try:
            provider = getattr(self.agent, "llm", None)
            client = getattr(provider, "client", None)
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if callable(close):
                maybe_awaitable = close()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
        except BaseException as error:
            errors.append(error)

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("Errors while closing agent resources", errors)

    def close(self) -> None:
        self._raise_if_running_loop()
        self._loop_owner.close(self._cleanup_agent_resources)
