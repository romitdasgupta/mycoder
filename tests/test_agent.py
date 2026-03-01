# tests/test_agent.py
from unittest.mock import MagicMock
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.providers.base import LLMResponse, ToolCall


def test_agent_simple_text_response():
    provider = MagicMock()
    provider.send.return_value = LLMResponse(text="Hello!", tool_calls=[], done=True)

    reg = create_default_registry()
    agent = Agent(provider=provider, registry=reg)
    messages = [{"role": "user", "content": "Hi"}]
    text, updated = agent.step(messages)

    assert text == "Hello!"
    assert len(updated) > len(messages)


def test_agent_tool_then_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("file content here")

    provider = MagicMock()
    provider.send.side_effect = [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="id1", name="read_file", arguments={"path": str(f)})],
            done=False,
        ),
        LLMResponse(text="I read the file.", tool_calls=[], done=True),
    ]

    reg = create_default_registry()
    agent = Agent(provider=provider, registry=reg)
    messages = [{"role": "user", "content": "Read test.txt"}]
    text, updated = agent.step(messages)

    assert text == "I read the file."
    assert provider.send.call_count == 2
