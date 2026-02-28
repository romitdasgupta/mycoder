import os
import pytest
from mycoder.tools.files import read_file, write_file, edit_file, glob_files, grep_files, list_directory


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = read_file({"path": str(f)})
    assert result == "hello world"


def test_read_file_with_lines(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("line1\nline2\nline3\nline4\nline5\n")
    result = read_file({"path": str(f), "offset": 2, "limit": 2})
    assert "line2" in result
    assert "line3" in result
    assert "line4" not in result


def test_read_file_not_found():
    result = read_file({"path": "/nonexistent/file.txt"})
    assert "Error" in result


def test_write_file(tmp_path):
    f = tmp_path / "out.txt"
    result = write_file({"path": str(f), "content": "new content"})
    assert "11 bytes" in result
    assert f.read_text() == "new content"


def test_write_file_creates_dirs(tmp_path):
    f = tmp_path / "sub" / "dir" / "out.txt"
    result = write_file({"path": str(f), "content": "nested"})
    assert f.read_text() == "nested"


def test_edit_file(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world\nfoo bar\n")
    result = edit_file({"path": str(f), "old_string": "foo bar", "new_string": "baz qux"})
    assert "Applied" in result
    assert f.read_text() == "hello world\nbaz qux\n"


def test_edit_file_not_found_string(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world\n")
    result = edit_file({"path": str(f), "old_string": "not here", "new_string": "x"})
    assert "not found" in result.lower()


def test_glob_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = glob_files({"pattern": "*.py", "path": str(tmp_path)})
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


def test_grep_files(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    pass\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    result = grep_files({"pattern": "def", "path": str(tmp_path)})
    assert "a.py" in result
    assert "hello" in result


def test_list_directory(tmp_path):
    (tmp_path / "file.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    result = list_directory({"path": str(tmp_path)})
    assert "file.txt" in result
    assert "subdir" in result
