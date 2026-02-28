"""Built-in tool registration — wires all tools into a ToolRegistry."""

from mycoder.tools.registry import ToolRegistry
from mycoder.tools.files import read_file, write_file, edit_file, glob_files, grep_files, list_directory
from mycoder.tools.shell import run_command


def create_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register("read_file", "Read contents of a file", read_file, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "offset": {"type": "integer", "description": "Start line (1-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    })

    reg.register("write_file", "Create or overwrite a file", write_file, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    })

    reg.register("edit_file", "Replace a unique string in a file", edit_file, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "old_string": {"type": "string", "description": "Exact string to find (must be unique)"},
            "new_string": {"type": "string", "description": "Replacement string"},
        },
        "required": ["path", "old_string", "new_string"],
    })

    reg.register("glob", "Find files matching a glob pattern", glob_files, {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/*.ts')"},
            "path": {"type": "string", "description": "Base directory to search from"},
        },
        "required": ["pattern"],
    })

    reg.register("grep", "Search file contents with regex", grep_files, {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Base directory to search in"},
            "glob": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
        },
        "required": ["pattern"],
    })

    reg.register("list_directory", "List files and directories at a path", list_directory, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
        },
        "required": [],
    })

    reg.register("run_command", "Execute a shell command", run_command, {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
        },
        "required": ["command"],
    })

    return reg
