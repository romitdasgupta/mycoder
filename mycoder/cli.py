"""CLI REPL — interactive terminal interface with streaming output."""

import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mycoder.config import load_config
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.memory.store import SessionStore


console = Console()


def display_tool_call(name: str, input_data: dict, result: str) -> None:
    """Display a tool call in a formatted panel."""
    # Summarize input
    summary_parts = []
    for k, v in input_data.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        summary_parts.append(f"{k}={s}")
    summary = ", ".join(summary_parts)

    # Truncate result for display
    display_result = result if len(result) <= 200 else result[:197] + "..."

    console.print(Panel(
        f"[dim]{summary}[/dim]\n\n{display_result}",
        title=f"[bold cyan]{name}[/bold cyan]",
        border_style="dim",
        expand=False,
    ))


def handle_command(cmd: str, session: dict, store: SessionStore, agent: Agent) -> bool:
    """Handle slash commands. Returns True if should continue REPL, False to quit."""
    if cmd == "/quit":
        return False
    elif cmd == "/clear":
        session["messages"] = []
        console.print("[dim]Conversation cleared.[/dim]")
    elif cmd == "/history":
        for msg in session["messages"]:
            role = msg["role"]
            content = msg["content"] if isinstance(msg["content"], str) else "(tool data)"
            console.print(f"[bold]{role}:[/bold] {content[:100]}")
    elif cmd.startswith("/model"):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 2:
            agent.model = parts[1]
            session["model"] = parts[1]
            console.print(f"[dim]Model set to {parts[1]}[/dim]")
        else:
            console.print(f"[dim]Current model: {agent.model}[/dim]")
    elif cmd == "/resume":
        loaded = store.load_latest()
        if loaded:
            session.update(loaded)
            console.print(f"[dim]Resumed session {loaded['id']} ({len(loaded['messages'])} messages)[/dim]")
        else:
            console.print("[dim]No previous session found.[/dim]")
    else:
        console.print(f"[dim]Unknown command: {cmd}[/dim]")
    return True


def main():
    cfg = load_config()

    registry = create_default_registry()
    agent = Agent(api_key=cfg.api_key, model=cfg.model, registry=registry)
    store = SessionStore()
    session = store.new_session(model=cfg.model, cwd=os.getcwd())

    history_path = os.path.expanduser("~/.mycoder/input_history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    prompt_session = PromptSession(history=FileHistory(history_path))

    console.print("[bold]mycoder[/bold] — your coding assistant")
    console.print("[dim]Type /quit to exit, /clear to reset, /resume to load last session[/dim]\n")

    while True:
        try:
            user_input = prompt_session.prompt("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, session, store, agent):
                break
            continue

        # Add user message
        session["messages"].append({"role": "user", "content": user_input})

        # Run agent
        console.print()
        try:
            text, updated_messages = agent.step(
                session["messages"],
                on_tool_call=display_tool_call,
            )
            session["messages"] = updated_messages
            console.print(f"\n[bold]mycoder:[/bold] {text}\n")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")

        # Auto-save
        store.save(session)

    console.print("\n[dim]Goodbye.[/dim]")
    store.save(session)


if __name__ == "__main__":
    main()
