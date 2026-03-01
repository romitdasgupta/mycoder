from unittest.mock import MagicMock, patch
from mycoder.providers.google import GoogleProvider
from mycoder.providers.base import LLMProvider


def test_google_implements_protocol():
    with patch("mycoder.providers.google.genai.Client"):
        provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.google.genai.Client")
def test_send_text(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    # Simulate Gemini response with text only
    mock_part = MagicMock()
    mock_part.text = "Hello!"
    mock_part.function_call = None
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
    result = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        system="Be helpful.",
        tools=[],
    )
    assert result.text == "Hello!"
    assert result.done is True


@patch("mycoder.providers.google.genai.Client")
def test_send_tool_call(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    mock_fc = MagicMock()
    mock_fc.name = "read_file"
    mock_fc.args = {"path": "x.py"}
    mock_fc.id = "fc1"
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.function_call = mock_fc
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="Be helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
