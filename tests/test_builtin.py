from mycoder.tools.builtin import create_default_registry


def test_default_registry_has_all_tools():
    reg = create_default_registry()
    schemas = reg.get_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        "read_file", "write_file", "edit_file",
        "glob", "grep", "list_directory", "run_command",
    }


def test_default_registry_executes_tool(tmp_path):
    reg = create_default_registry()
    f = tmp_path / "test.txt"
    f.write_text("hi")
    result = reg.execute("read_file", {"path": str(f)})
    assert "hi" in result
