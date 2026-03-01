"""Shell tool — execute commands with timeout and output capture."""

import re
import subprocess
from typing import Callable


_DANGEROUS_PATTERNS = [
    re.compile(p)
    for p in [
        r"\brm\b.*\s-[a-zA-Z]*r",
        r"\bgit\s+push\b.*--force",
        r"\bgit\s+push\b.*\s-f\b",
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\s",
        r">\s*/dev/",
        r"\bchmod\s+777\b",
        r"\|\s*(ba)?sh\b",
        r"\bshutdown\b",
        r"\breboot\b",
    ]
]


def run_command(data: dict) -> str:
    """Execute a shell command and return stdout + stderr."""
    command = data["command"]
    timeout = data.get("timeout", 30)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def make_safe_runner() -> Callable[[dict], str]:
    """Return a shell runner that prompts the user for confirmation."""
    from rich.console import Console

    console = Console()

    def handler(data: dict) -> str:
        command = data["command"]
        is_dangerous = any(p.search(command) for p in _DANGEROUS_PATTERNS)

        if is_dangerous:
            console.print(
                f"\n[bold red]Warning — potentially dangerous command:[/bold red]"
                f"\n  [yellow]$ {command}[/yellow]"
            )
        else:
            console.print(f"\n[bold cyan]Shell command:[/bold cyan]  {command}")

        try:
            answer = input("  Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "Command denied by user."

        if answer not in ("y", "yes"):
            return "Command denied by user."

        return run_command(data)

    return handler
