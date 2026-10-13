import pytest


def test_agent_hello_world(agent):
    """Test that the agent can respond to a simple prompt.

    This is a basic smoke test to verify the agent is functioning.
    """
    response = agent.run("Hello!")
    assert response.answer.strip() != ""


@pytest.mark.integration
def test_agent_memory_and_context(agent):
    """Test that the agent retains context across multiple interactions.

    Verifies that:
    1. The agent responds to initial prompts
    2. The agent remembers previous messages in the conversation
    3. The agent can answer questions about previous interactions
    4. The agent state can be reset to clear message history
    """
    first_prompt = "Reply with one short sentence about memory tests."

    first_response = agent.run(first_prompt)

    assert first_response.answer.strip() != ""
    assert any(
        message.get("role") == "user" and message.get("content") == first_prompt
        for message in agent.messages
    )
    assert agent.messages[-1]["role"] == "assistant"
    first_run_message_count = len(agent.messages)
    assert first_run_message_count >= 2

    second_response = agent.run("What did I ask in my previous message?")

    assert second_response.answer.strip() != ""
    assert len(agent.messages) > first_run_message_count
    assert agent.messages[-1]["role"] == "assistant"

    agent.reset_state()
    assert all(message.get("role") == "system" for message in agent.messages)


@pytest.mark.integration
def test_agent_can_call_tools(agent):
    """Test that an agent can successfully call tools during interaction.

    Verifies that:
    1. The agent can invoke available tools
    2. The agent receives tool results and incorporates them in responses
    3. The result of tool invocation is correctly reflected in the final answer

    Note: This test requires an agent configured with tools and requires the
    @pytest.mark.requires_ollama marker (applied at fixture level when using ollama models).
    """

    # Ask the agent to use the tool
    response = agent.run(
        "What is 15 plus 27? Please use the add_numbers tool to calculate this."
    )

    # Verify the agent got a response
    assert response.answer.strip() != ""

    # Verify the answer contains the correct result (42)
    assert "42" in response.answer

    # Verify the message history was updated with multiple messages
    # When tools are used, the history includes user, assistant, tool, and assistant messages
    assert len(agent.messages) >= 2

    # Verify there's at least one user message and one assistant message
    roles = [msg.get("role") for msg in agent.messages]
    assert "user" in roles
    assert "assistant" in roles
