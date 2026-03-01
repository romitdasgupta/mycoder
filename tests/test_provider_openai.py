# tests/test_provider_openai.py
from unittest.mock import MagicMock, patch
import json
from mycoder.providers.openai import OpenAIProvider
from mycoder.providers.base import LLMProvider


def _text_response(text):
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = text
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_response(name, args, tool_id="call_1"):
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    tc.type = "function"
    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message.content = None
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_openai_implements_protocol():
    with patch("mycoder.providers.openai.openai.OpenAI"):
        provider = OpenAIProvider(api_key="test", model="test")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.openai.openai.OpenAI")
def test_send_text(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _text_response("Hi!")

    provider = OpenAIProvider(api_key="test", model="gpt-4o")
    result = provider.send(
        messages=[{"role": "user", "content": "Hello"}],
        system="Be helpful.",
        tools=[],
    )
    assert result.text == "Hi!"
    assert result.done is True


@patch("mycoder.providers.openai.openai.OpenAI")
def test_send_tool_call(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _tool_response(
        "read_file", {"path": "x.py"}
    )

    provider = OpenAIProvider(api_key="test", model="gpt-4o")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="Be helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"


@patch("mycoder.providers.openai.openai.OpenAI")
def test_openai_compatible_base_url(mock_cls):
    """OpenAI-compatible APIs (Groq, Together, etc.) use base_url."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    provider = OpenAIProvider(api_key="test", model="llama-3", base_url="https://api.groq.com/openai/v1")
    mock_cls.assert_called_with(api_key="test", base_url="https://api.groq.com/openai/v1")
