from unittest.mock import MagicMock, patch
from mycoder.providers.anthropic import AnthropicProvider
from mycoder.providers.base import LLMProvider


def _text_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _tool_response(name, args, tool_id="t1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def test_anthropic_implements_protocol():
    assert isinstance(AnthropicProvider, type)
    with patch("mycoder.providers.anthropic.anthropic.Anthropic"):
        provider = AnthropicProvider(api_key="test", model="test")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.anthropic.anthropic.Anthropic")
def test_send_text_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _text_response("Hello!")

    provider = AnthropicProvider(api_key="test", model="test")
    result = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        system="You are helpful.",
        tools=[],
    )
    assert result.text == "Hello!"
    assert result.done is True
    assert result.tool_calls == []


@patch("mycoder.providers.anthropic.anthropic.Anthropic")
def test_send_tool_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _tool_response(
        "read_file", {"path": "x.py"}
    )

    provider = AnthropicProvider(api_key="test", model="test")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="You are helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "x.py"}
