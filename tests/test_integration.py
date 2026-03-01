"""Integration test — verify all pieces wire together."""

from unittest.mock import MagicMock
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.memory.store import SessionStore
from mycoder.providers.base import LLMResponse, ToolCall


def test_full_round_trip(tmp_path):
    """Agent reads a file via tool, session is saved and loadable."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello from integration test!")

    registry = create_default_registry()
    store = SessionStore(base_dir=tmp_path / "sessions")
    session = store.new_session(model="test", cwd=str(tmp_path))

    provider = MagicMock()
    provider.send.side_effect = [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tool1", name="read_file", arguments={"path": str(test_file)})],
            done=False,
        ),
        LLMResponse(text="The file says hello!", tool_calls=[], done=True),
    ]

    agent = Agent(provider=provider, registry=registry)
    session["messages"].append({"role": "user", "content": "Read hello.txt"})

    text, updated = agent.step(session["messages"])
    session["messages"] = updated

    assert text == "The file says hello!"
    assert len(session["messages"]) == 4  # user, assistant(tool), user(result), assistant(text)

    store.save(session)
    loaded = store.load(session["id"])
    assert loaded is not None
    assert len(loaded["messages"]) == 4
