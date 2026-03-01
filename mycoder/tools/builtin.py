"""Built-in tool registration — wires all tools into a ToolRegistry."""

from __future__ import annotations

from mycoder.tools.registry import ToolRegistry
from mycoder.tools.files import read_file, write_file, edit_file, glob_files, grep_files, list_directory
from mycoder.tools.shell import run_command


def create_default_registry(*, safe_mode: bool = False, cwd: str | None = None) -> ToolRegistry:
    reg = ToolRegistry()

    _read = read_file
    _write = write_file
    _edit = edit_file
    _glob = glob_files
    _grep = grep_files
    _ls = list_directory
    _shell = run_command

    if safe_mode:
        from mycoder.tools.files import sandboxed
        from mycoder.tools.shell import make_safe_runner

        resolved_cwd = cwd or "."
        _read = sandboxed(read_file, resolved_cwd)
        _write = sandboxed(write_file, resolved_cwd)
        _edit = sandboxed(edit_file, resolved_cwd)
        _glob = sandboxed(glob_files, resolved_cwd)
        _grep = sandboxed(grep_files, resolved_cwd)
        _ls = sandboxed(list_directory, resolved_cwd)
        _shell = make_safe_runner()

    reg.register("read_file", "Read contents of a file", _read, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "offset": {"type": "integer", "description": "Start line (1-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    })

    reg.register("write_file", "Create or overwrite a file", _write, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    })

    reg.register("edit_file", "Replace a unique string in a file", _edit, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "old_string": {"type": "string", "description": "Exact string to find (must be unique)"},
            "new_string": {"type": "string", "description": "Replacement string"},
        },
        "required": ["path", "old_string", "new_string"],
    })

    reg.register("glob", "Find files matching a glob pattern", _glob, {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/*.ts')"},
            "path": {"type": "string", "description": "Base directory to search from"},
        },
        "required": ["pattern"],
    })

    reg.register("grep", "Search file contents with regex", _grep, {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Base directory to search in"},
            "glob": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
        },
        "required": ["pattern"],
    })

    reg.register("list_directory", "List files and directories at a path", _ls, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
        },
        "required": [],
    })

    reg.register("run_command", "Execute a shell command", _shell, {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
        },
        "required": ["command"],
    })

    return reg
