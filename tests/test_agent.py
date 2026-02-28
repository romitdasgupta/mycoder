from unittest.mock import MagicMock, patch
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry


def _make_text_response(text):
    """Create a mock response with stop_reason='end_turn' and one text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _make_tool_response(tool_name, tool_input, tool_id="id1"):
    """Create a mock response with stop_reason='tool_use'."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


@patch("mycoder.agent.anthropic.Anthropic")
def test_agent_simple_text_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_text_response("Hello!")

    reg = create_default_registry()
    agent = Agent(api_key="test", model="test-model", registry=reg)
    messages = [{"role": "user", "content": "Hi"}]
    text, updated = agent.step(messages)

    assert text == "Hello!"
    assert len(updated) > len(messages)


@patch("mycoder.agent.anthropic.Anthropic")
def test_agent_tool_then_text(mock_cls, tmp_path):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    f = tmp_path / "test.txt"
    f.write_text("file content here")

    mock_client.messages.create.side_effect = [
        _make_tool_response("read_file", {"path": str(f)}),
        _make_text_response("I read the file."),
    ]

    reg = create_default_registry()
    agent = Agent(api_key="test", model="test-model", registry=reg)
    messages = [{"role": "user", "content": "Read test.txt"}]
    text, updated = agent.step(messages)

    assert text == "I read the file."
    assert mock_client.messages.create.call_count == 2
