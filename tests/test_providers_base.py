from mycoder.providers.base import ToolCall, LLMResponse, ToolResult, StreamEvent


def test_llm_response_with_text():
    r = LLMResponse(text="hello", tool_calls=[], done=True)
    assert r.text == "hello"
    assert r.done is True
    assert r.tool_calls == []


def test_llm_response_with_tool_calls():
    tc = ToolCall(id="1", name="read_file", arguments={"path": "f.py"})
    r = LLMResponse(text=None, tool_calls=[tc], done=False)
    assert r.done is False
    assert r.tool_calls[0].name == "read_file"


def test_tool_result():
    tr = ToolResult(tool_call_id="1", content="file contents")
    assert tr.tool_call_id == "1"


def test_stream_event():
    e = StreamEvent(type="text_delta", text="hi")
    assert e.type == "text_delta"
    assert e.tool_call is None
