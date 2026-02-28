import pytest
from mycoder.tools.registry import ToolRegistry


def test_register_and_get_schemas():
    reg = ToolRegistry()
    reg.register(
        name="echo",
        description="Echo input back",
        handler=lambda data: data["text"],
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    schemas = reg.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"
    assert schemas[0]["description"] == "Echo input back"
    assert schemas[0]["input_schema"]["properties"]["text"]["type"] == "string"


def test_execute_registered_tool():
    reg = ToolRegistry()
    reg.register(
        name="echo",
        description="Echo input back",
        handler=lambda data: data["text"],
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    result = reg.execute("echo", {"text": "hello"})
    assert result == "hello"


def test_execute_unknown_tool():
    reg = ToolRegistry()
    result = reg.execute("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_tool_error_returns_string():
    def broken(data):
        raise ValueError("boom")

    reg = ToolRegistry()
    reg.register("broken", "breaks", broken, {"type": "object", "properties": {}, "required": []})
    result = reg.execute("broken", {})
    assert "Error" in result
    assert "boom" in result
