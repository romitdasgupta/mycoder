"""File tools — read, write, edit, glob, grep, list directory."""

import os
import re
from pathlib import Path
from typing import Callable


def read_file(data: dict) -> str:
    """Read file contents, optionally with line offset and limit."""
    try:
        path = data["path"]
        with open(path) as f:
            lines = f.readlines()

        has_offset = "offset" in data
        has_limit = "limit" in data

        # When neither offset nor limit is given, return raw content.
        if not has_offset and not has_limit:
            return "".join(lines)

        offset = data.get("offset", 1)  # 1-indexed
        limit = data.get("limit")

        start = max(0, offset - 1)
        end = start + limit if limit else len(lines)
        selected = lines[start:end]

        numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(selected)]
        return "".join(numbered)
    except Exception as e:
        return f"Error: {e}"


def write_file(data: dict) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        path = Path(data["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = data["content"]
        path.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def edit_file(data: dict) -> str:
    """Replace old_string with new_string in a file."""
    try:
        path = Path(data["path"])
        content = path.read_text()
        old = data["old_string"]
        new = data["new_string"]

        if old not in content:
            return f"Error: old_string not found in {path}"

        count = content.count(old)
        if count > 1:
            return f"Error: old_string found {count} times — must be unique. Add more context."

        updated = content.replace(old, new, 1)
        path.write_text(updated)
        return f"Applied edit to {path}"
    except Exception as e:
        return f"Error: {e}"


def glob_files(data: dict) -> str:
    """Find files matching a glob pattern."""
    try:
        pattern = data["pattern"]
        base = Path(data.get("path", "."))
        matches = sorted(base.rglob(pattern))
        if not matches:
            return "No files matched."
        return "\n".join(str(m) for m in matches[:100])
    except Exception as e:
        return f"Error: {e}"


def grep_files(data: dict) -> str:
    """Search file contents for a regex pattern."""
    try:
        pattern = re.compile(data["pattern"])
        base = Path(data.get("path", "."))
        glob = data.get("glob", "*")
        results = []

        for fpath in sorted(base.rglob(glob)):
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        results.append(f"{fpath}:{i}: {line}")
            except (PermissionError, OSError):
                continue

        if not results:
            return "No matches found."
        return "\n".join(results[:100])
    except Exception as e:
        return f"Error: {e}"


def list_directory(data: dict) -> str:
    """List files and directories at a path."""
    try:
        base = Path(data.get("path", "."))
        entries = sorted(base.iterdir())
        lines = []
        for e in entries:
            prefix = "d " if e.is_dir() else "f "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


def sandboxed(fn: Callable[[dict], str], cwd: str) -> Callable[[dict], str]:
    """Wrap a file tool so all paths must resolve under *cwd*."""
    cwd_resolved = Path(cwd).resolve()

    def wrapper(data: dict) -> str:
        if "path" in data:
            target = Path(data["path"])
            if not target.is_absolute():
                target = cwd_resolved / target
            resolved = target.resolve()
            if not resolved.is_relative_to(cwd_resolved):
                return f"Error: access denied — path resolves outside the project directory."
        return fn(data)

    return wrapper
