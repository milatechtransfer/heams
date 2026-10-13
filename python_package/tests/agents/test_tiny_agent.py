import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from heams.agents.config import (
    AgentParameters,
    MemoryStrategy,
    default_benchmark_agent_config,
)
from heams.agents.tiny_agent import (
    TinyAgent,
    ContextBudgetGuard,
    LimitToolExecutions,
    ToolExecutionLimitReached,
    _OwnedEventLoop,
)
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams.memory_systems.shared_types import MemoryEntry


class DummyMemorySystem(BaseMemorySystem):
    def __init__(
        self,
        name: str = "dummy",
        top_k: int = 5,
        entries: list[MemoryEntry] | None = None,
    ):
        super().__init__(
            BaseMemoryConfig(type="dummy_memory", name=name, top_k=top_k),
            storage_dir=Path(),
        )
        self.entries: list[MemoryEntry] = list(entries or [])
        self.recall_calls: list[tuple[str, int | None]] = []
        self.delete_calls: list[str] = []
        self.recall_contexts: list[dict[str, Any] | None] = []
        self.retain_contexts: list[dict[str, Any] | None] = []

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        self.retain_contexts.append(self._current_run_context() or None)
        self.entries.append(MemoryEntry(content=content, metadata=metadata))
        return MemoryOperationResult(value="Successfully saved to memory.")

    def _recall_impl(
        self,
        query: str,
        top_k: int,
    ) -> MemoryOperationResult:
        self.recall_calls.append((query, top_k))
        self.recall_contexts.append(self._current_run_context() or None)
        return MemoryOperationResult(
            value=[entry.content for entry in self.entries[:top_k]]
        )

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        self.delete_calls.append(query)
        return MemoryOperationResult(value="Deleted matching memory.")


def _make_config(
    memory_strategy: MemoryStrategy,
    *,
    max_tool_executions: int | None = 10,
) -> AgentParameters:
    return AgentParameters(
        memory_strategy=memory_strategy,
        model_id="test-model",
        instructions="Test instructions",
        model_args={"temperature": 0.0},
        max_tool_executions=max_tool_executions,
    )


@pytest.mark.parametrize(
    "memory_strategy",
    [
        MemoryStrategy.TOOL_USE,
        MemoryStrategy.DETERMINISTIC,
        MemoryStrategy.DETERMINISTIC_WITH_DELETE,
        MemoryStrategy.RECALL_ONLY_TOOL_USE,
        MemoryStrategy.RECALL_ONLY_DETERMINISTIC,
    ],
)
def test_requires_memory_system_for_non_no_memory_strategies(memory_strategy):
    with pytest.raises(
        ValueError,
        match="memory_system is required unless memory_strategy is NO_MEMORY",
    ):
        TinyAgent(config=_make_config(memory_strategy), tools=[], memory_system=None)


def test_no_memory_strategy_runs_without_memory_system():
    mock_trace = MagicMock(output="hello")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )

        response = wrapper.run("Hi there")

    sent_messages = mock_agent.run.call_args.args[0]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[0]["content"] == "Test instructions"
    assert sent_messages[1]["role"] == "user"
    assert sent_messages[1]["content"] == "Hi there"
    assert response.answer == "hello"
    assert wrapper.messages[-1]["role"] == "assistant"


def test_close_awaits_provider_cleanup_before_owned_loop_closes():
    lifecycle_events: list[tuple[str, asyncio.AbstractEventLoop]] = []

    class FakeClient:
        async def close(self) -> None:
            lifecycle_events.append(("provider_close", asyncio.get_running_loop()))

    class FakeAgent:
        def __init__(self) -> None:
            self.llm = SimpleNamespace(client=FakeClient())

        def run(self, messages):
            lifecycle_events.append(("run", asyncio.get_event_loop()))
            return SimpleNamespace(output="hello")

        async def cleanup_async(self) -> None:
            lifecycle_events.append(("agent_cleanup", asyncio.get_running_loop()))

    fake_agent = FakeAgent()

    def create_agent(*args, **kwargs):
        lifecycle_events.append(("create", asyncio.get_event_loop()))
        return fake_agent

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async",
        side_effect=create_agent,
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )
        wrapper.run("Hi there")
        owned_loop = wrapper._loop_owner.loop
        wrapper.close()
        wrapper.close()

    assert [event[0] for event in lifecycle_events] == [
        "create",
        "run",
        "agent_cleanup",
        "provider_close",
    ]
    assert len({id(event[1]) for event in lifecycle_events}) == 1
    assert owned_loop.is_closed()


def test_close_shuts_down_owned_loop_when_cleanup_fails():
    provider_close = MagicMock()

    class FakeAgent:
        llm = SimpleNamespace(client=SimpleNamespace(close=provider_close))

        def run(self, messages):
            return SimpleNamespace(output="hello")

        async def cleanup_async(self) -> None:
            raise RuntimeError("agent cleanup failed")

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async",
        return_value=FakeAgent(),
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )
        owned_loop = wrapper._loop_owner.loop

        with pytest.raises(RuntimeError, match="agent cleanup failed"):
            wrapper.close()

    provider_close.assert_called_once_with()
    assert owned_loop.is_closed()


def test_close_reports_both_agent_and_provider_cleanup_failures():
    class FakeClient:
        async def close(self) -> None:
            raise RuntimeError("provider close failed")

    class FakeAgent:
        llm = SimpleNamespace(client=FakeClient())

        async def cleanup_async(self) -> None:
            raise RuntimeError("agent cleanup failed")

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async",
        return_value=FakeAgent(),
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )

        with pytest.raises(BaseExceptionGroup) as raised:
            wrapper.close()

    assert {str(error) for error in raised.value.exceptions} == {
        "agent cleanup failed",
        "provider close failed",
    }


def test_owned_loop_does_not_install_default_loop_when_none_was_set():
    policy = asyncio.get_event_loop_policy()
    previous_loop = getattr(getattr(policy, "_local", None), "_loop", None)
    asyncio.set_event_loop(None)
    owner = _OwnedEventLoop()

    try:
        owner.run(lambda: None)
        owner.close()
        assert getattr(getattr(policy, "_local", None), "_loop", None) is None
    finally:
        if not owner.loop.is_closed():
            owner.close()
        asyncio.set_event_loop(previous_loop)


def test_wrapper_rejects_construction_from_async_context():
    async def construct_wrapper() -> None:
        with pytest.raises(
            RuntimeError,
            match="synchronous and cannot be used from an async context",
        ):
            TinyAgent(
                config=_make_config(MemoryStrategy.NO_MEMORY),
                tools=[],
                memory_system=None,
            )

    asyncio.run(construct_wrapper())


def test_wrapper_rejects_close_from_async_context():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )

    async def close_wrapper() -> None:
        with pytest.raises(
            RuntimeError,
            match="synchronous and cannot be used from an async context",
        ):
            wrapper.close()

    asyncio.run(close_wrapper())
    wrapper.close()


def test_wrapper_rejects_run_from_async_context():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )

    async def run_wrapper() -> None:
        with pytest.raises(
            RuntimeError,
            match="synchronous and cannot be used from an async context",
        ):
            wrapper.run("Hi there")

    asyncio.run(run_wrapper())
    wrapper.close()


def test_tool_use_strategy_does_not_perform_deterministic_recall():
    memory = DummyMemorySystem(
        name="dummy",
        top_k=3,
        entries=[MemoryEntry(content="prior memory")],
    )
    mock_trace = MagicMock(output="tool result")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    def add_numbers(a: int, b: int) -> int:
        return a + b

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.TOOL_USE),
            tools=[add_numbers],
            memory_system=memory,
        )
        wrapper.run("Use tools")

    sent_messages = mock_agent.run.call_args.args[0]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"
    assert memory.recall_calls == []


def test_tool_use_strategy_does_not_register_unsupplied_memory_tools():
    memory = DummyMemorySystem(name="dummy", top_k=3)
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(
            config=_make_config(MemoryStrategy.TOOL_USE),
            tools=[],
            memory_system=memory,
        )

    agent_config = mock_create.call_args.args[0]
    assert getattr(agent_config, "tools", []) == []


def test_recall_only_tool_use_does_not_perform_deterministic_recall():
    memory = DummyMemorySystem(
        name="dummy",
        top_k=3,
        entries=[MemoryEntry(content="prior memory")],
    )
    mock_trace = MagicMock(output="tool result")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.RECALL_ONLY_TOOL_USE),
            tools=[],
            memory_system=memory,
        )
        wrapper.run("Use tools")

    sent_messages = mock_agent.run.call_args.args[0]
    assert sent_messages[1]["content"] == "Use tools"
    assert memory.recall_calls == []
    assert memory.entries == [MemoryEntry(content="prior memory")]


def test_recall_only_deterministic_recalls_without_retaining_turn():
    memory = DummyMemorySystem(
        name="dummy",
        top_k=2,
        entries=[MemoryEntry(content="prior memory")],
    )
    mock_trace = MagicMock(output="final answer")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.RECALL_ONLY_DETERMINISTIC),
            tools=[],
            memory_system=memory,
        )
        wrapper.run("What did we discuss?")

    assert memory.recall_calls == [("What did we discuss?", 2)]
    assert len(memory.entries) == 1
    sent_messages = mock_agent.run.call_args.args[0]
    assert "[Memory 1]\nprior memory" in sent_messages[1]["content"]
    assert "User query: What did we discuss?" in sent_messages[1]["content"]


def test_wrapper_defaults_tool_execution_limit_to_ten():
    config = AgentParameters(
        memory_strategy=MemoryStrategy.NO_MEMORY,
        model_id="test-model",
        instructions="Test instructions",
    )

    assert config.max_tool_executions == 10


def test_wrapper_registers_tool_execution_limit_callback_by_default():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )

    agent_config = mock_create.call_args.args[0]
    assert isinstance(agent_config.callbacks[0], LimitToolExecutions)
    assert agent_config.callbacks[0].max_tool_executions == 10


def test_wrapper_skips_tool_execution_limit_callback_when_disabled():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(
            config=_make_config(
                MemoryStrategy.NO_MEMORY,
                max_tool_executions=None,
            ),
            tools=[],
            memory_system=None,
        )

    agent_config = mock_create.call_args.args[0]
    assert all(
        not isinstance(callback, LimitToolExecutions)
        for callback in agent_config.callbacks
    )


def test_tool_execution_limit_callback_raises_when_limit_is_exceeded():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(
            config=_make_config(
                MemoryStrategy.NO_MEMORY,
                max_tool_executions=2,
            ),
            tools=[],
            memory_system=None,
        )

    agent_config = mock_create.call_args.args[0]
    callback = agent_config.callbacks[0]
    context = SimpleNamespace(shared={})

    callback.before_tool_execution(context)
    callback.before_tool_execution(context)

    with pytest.raises(ToolExecutionLimitReached, match="Exceeded 2 tool executions"):
        callback.before_tool_execution(context)


def test_context_budget_guard_trims_oldest_complete_turn():
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    context = SimpleNamespace(shared={})
    callback = ContextBudgetGuard(max_input_tokens=1, approx_chars_per_token=4)
    reduced_messages = [messages[0], messages[-1]]
    callback.max_input_tokens = callback._estimate_request_tokens(
        {"messages": reduced_messages, "tools": []}
    )

    callback.before_llm_call(context, messages=messages, tools=[])

    assert messages == reduced_messages


def test_context_budget_guard_drops_history_when_protected_messages_exceed_budget():
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    context = SimpleNamespace(shared={})
    callback = ContextBudgetGuard(max_input_tokens=1, approx_chars_per_token=4)

    callback.before_llm_call(context, messages=messages, tools=[])

    assert messages == [messages[0], messages[-1]]


def test_context_budget_guard_records_truncation_in_callback_context():
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    context = SimpleNamespace(shared={})
    callback = ContextBudgetGuard(max_input_tokens=1, approx_chars_per_token=4)

    callback.before_llm_call(context, messages=messages, tools=[])

    assert context.shared["context_truncated"] is True


def test_context_budget_guard_does_not_mutate_wrapper_history():
    mock_agent = MagicMock()
    callback = ContextBudgetGuard(max_input_tokens=1, approx_chars_per_token=4)

    def run_agent(messages):
        callback.before_llm_call(
            SimpleNamespace(shared={}), messages=messages, tools=[]
        )
        return MagicMock(
            output=(
                "first answer"
                if len(mock_agent.run.call_args_list) == 1
                else "second answer"
            )
        )

    mock_agent.run.side_effect = run_agent
    config = _make_config(MemoryStrategy.NO_MEMORY)
    config.max_input_tokens = callback._estimate_request_tokens(
        {
            "messages": [
                {"role": "system", "content": "Test instructions"},
                {"role": "user", "content": "current question"},
            ],
            "tools": [],
        }
    )
    callback.max_input_tokens = config.max_input_tokens

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(config=config, tools=[], memory_system=None)
        first_response = wrapper.run("old question")
        second_response = wrapper.run("current question")

    assert first_response.context_truncated is False
    assert second_response.context_truncated is True
    assert mock_agent.run.call_args_list[1].args[0] == [
        wrapper.messages[0],
        wrapper.messages[3],
    ]
    assert wrapper.messages == [
        {"role": "system", "content": "Test instructions"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_wrapper_registers_context_budget_guard_when_configured():
    mock_agent = MagicMock()
    config = _make_config(MemoryStrategy.NO_MEMORY)
    config.max_input_tokens = 100

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(config=config, tools=[], memory_system=None)

    callbacks = mock_create.call_args.args[0].callbacks
    guards = [
        callback for callback in callbacks if isinstance(callback, ContextBudgetGuard)
    ]
    assert len(guards) == 1
    assert guards[0].max_input_tokens == 100


def test_deterministic_strategy_augments_user_message_with_memory_context():
    memory = DummyMemorySystem(
        name="dummy",
        top_k=2,
        entries=[
            MemoryEntry(content="User prefers concise responses."),
            MemoryEntry(content="Last session discussed taxes."),
        ],
    )
    mock_trace = MagicMock(output="final answer")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC),
            tools=[],
            memory_system=memory,
        )
        wrapper.run("What did we discuss?")

    assert memory.recall_calls == [("What did we discuss?", 2)]
    sent_messages = mock_agent.run.call_args.args[0]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"
    assert "Relevant memory context:" in sent_messages[1]["content"]
    assert "Use these memories only if relevant." in sent_messages[1]["content"]
    assert "[Memory 1]\nUser prefers concise responses." in sent_messages[1]["content"]
    assert "[Memory 2]\nLast session discussed taxes." in sent_messages[1]["content"]
    assert "User prefers concise responses." in sent_messages[1]["content"]
    assert "User query: What did we discuss?" in sent_messages[1]["content"]


def test_deterministic_strategy_retain_conversation_turn():
    memory = DummyMemorySystem(
        name="dummy",
        top_k=2,
        entries=[],
    )
    mock_trace = MagicMock(output="final answer")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC),
            tools=[],
            memory_system=memory,
        )
        wrapper.run("What did we discuss?")

    assert len(memory.entries) == 1
    assert memory.entries[0].content == "Q: What did we discuss?\nA: final answer"


def test_deterministic_delete_routes_request_without_recall_or_turn_retention():
    memory = DummyMemorySystem(entries=[MemoryEntry(content="Partner is James.")])
    mock_agent = MagicMock()
    mock_agent.run.return_value = MagicMock(output="I removed that memory.")

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC_WITH_DELETE),
            tools=[],
            memory_system=memory,
        )
        wrapper.run("My partner is James. Please remove that from your memory.")

    assert memory.delete_calls == [
        "My partner is James. Please remove that from your memory."
    ]
    assert memory.recall_calls == []
    assert [entry.content for entry in memory.entries] == ["Partner is James."]
    sent_prompt = mock_agent.run.call_args.args[0][-1]["content"]
    assert "Memory deletion result:\nDeleted matching memory." in sent_prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "Clear my former address from your memory.",
        "Please purge my former address from your memory.",
        "Wipe my former address from your memory.",
        "/erase-memory my former address",
        "/clear-memory my former address",
        "/purge-memory my former address",
    ],
)
def test_deterministic_delete_routes_explicit_delete_aliases(prompt: str) -> None:
    assert TinyAgent._is_explicit_delete_request(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Don't forget to remind me about the appointment.",
        "I don't want you to forget my appointment.",
        "Forget it, let's do something else.",
        "Remove that part, I was going from memory.",
    ],
)
def test_deterministic_delete_does_not_match_non_delete_request(
    prompt: str,
):
    memory = DummyMemorySystem(entries=[])
    mock_agent = MagicMock()
    mock_agent.run.return_value = MagicMock(output="I will remind you.")

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC_WITH_DELETE),
            tools=[],
            memory_system=memory,
        )
        wrapper.run(prompt)

    assert memory.delete_calls == []
    assert memory.recall_calls == [(prompt, 5)]
    assert memory.entries[-1].content == f"Q: {prompt}\nA: I will remind you."


@pytest.mark.parametrize(
    "prompt,answer",
    [
        ("", "final answer"),
        ("   ", "final answer"),
        ("What did we discuss?", ""),
        ("What did we discuss?", "   "),
        ("What did we discuss?", "[agent_error: RuntimeError: boom]"),
        ("What did we discuss?", "   [agent_error: RuntimeError: boom]"),
    ],
)
def test_deterministic_strategy_skips_invalid_retention(prompt: str, answer: str):
    memory = DummyMemorySystem(name="dummy", top_k=2, entries=[])
    mock_trace = SimpleNamespace(output=answer, final_output="")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC),
            tools=[],
            memory_system=memory,
        )
        wrapper.run(prompt)

    assert memory.entries == []


def test_deterministic_strategy_forwards_explicit_run_context() -> None:
    memory = DummyMemorySystem(
        name="dummy",
        top_k=2,
        entries=[MemoryEntry(content="User prefers concise responses.")],
    )
    mock_trace = MagicMock(output="final answer")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.DETERMINISTIC),
            tools=[],
            memory_system=memory,
        )
        wrapper.run(
            "What did we discuss?",
            run_context={"stimulus_index": 2, "question_id": "q2"},
        )

    assert memory.recall_contexts == [{"stimulus_index": 2, "question_id": "q2"}]
    assert memory.retain_contexts == [{"stimulus_index": 2, "question_id": "q2"}]


def test_run_uses_final_output_when_output_is_missing():
    mock_trace = MagicMock(output=None, final_output="fallback answer")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )
        response = wrapper.run("Ping")

    assert response.answer == "fallback answer"
    assert response.raw_trace is mock_trace


def test_reset_state_clears_messages():
    mock_trace = MagicMock(output="hello")
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_trace

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY),
            tools=[],
            memory_system=None,
        )
        wrapper.run("one")
        wrapper.run("two")

    assert len(wrapper.messages) >= 4
    wrapper.reset_state()
    assert wrapper.messages == [{"role": "system", "content": "Test instructions"}]


def test_wrapper_without_instructions_starts_with_empty_messages():
    config = AgentParameters(
        memory_strategy=MemoryStrategy.NO_MEMORY,
        model_id="test-model",
        instructions="",
    )
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ):
        wrapper = TinyAgent(config=config, tools=[], memory_system=None)

    assert wrapper.messages == []


def test_wrapper_disables_console_trace_callback_by_default():
    mock_agent = MagicMock()

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(
            config=_make_config(MemoryStrategy.NO_MEMORY, max_tool_executions=None),
            tools=[],
            memory_system=None,
        )

    agent_config = mock_create.call_args.args[0]
    assert agent_config.callbacks == []


def test_wrapper_keeps_default_callbacks_when_console_trace_enabled():
    mock_agent = MagicMock()

    config = AgentParameters(
        memory_strategy=MemoryStrategy.NO_MEMORY,
        model_id="test-model",
        instructions="",
        max_tool_executions=None,
        enable_console_trace=True,
    )

    with patch(
        "heams.agents.tiny_agent.TinyAgentRuntime.create_async", return_value=mock_agent
    ) as mock_create:
        TinyAgent(config=config, tools=[], memory_system=None)

    agent_config = mock_create.call_args.args[0]
    assert len(agent_config.callbacks) >= 1


def test_default_benchmark_agent_config() -> None:
    cfg = default_benchmark_agent_config("openai:gpt-4o-mini")
    assert cfg.model_id == "openai:gpt-4o-mini"
    assert cfg.type == "agent"
    assert cfg.instructions == ""
    assert cfg.model_args == {"temperature": 0.0}
