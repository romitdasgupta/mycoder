"""Integration test — verify all pieces wire together."""

from unittest.mock import MagicMock, patch
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.memory.store import SessionStore


def test_full_round_trip(tmp_path):
    """Agent reads a file via tool, session is saved and loadable."""
    # Setup
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello from integration test!")

    registry = create_default_registry()
    store = SessionStore(base_dir=tmp_path / "sessions")
    session = store.new_session(model="test", cwd=str(tmp_path))

    # Mock the API — first call triggers read_file, second returns text
    with patch("mycoder.agent.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Tool call response
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.input = {"path": str(test_file)}
        tool_block.id = "tool1"
        tool_resp = MagicMock()
        tool_resp.stop_reason = "tool_use"
        tool_resp.content = [tool_block]

        # Text response
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The file says hello!"
        text_resp = MagicMock()
        text_resp.stop_reason = "end_turn"
        text_resp.content = [text_block]

        mock_client.messages.create.side_effect = [tool_resp, text_resp]

        agent = Agent(api_key="test", model="test", registry=registry)
        session["messages"].append({"role": "user", "content": "Read hello.txt"})

        text, updated = agent.step(session["messages"])
        session["messages"] = updated

    # Verify
    assert text == "The file says hello!"
    assert len(session["messages"]) == 4  # user, assistant(tool), user(result), assistant(text)

    # Save and reload
    store.save(session)
    loaded = store.load(session["id"])
    assert loaded is not None
    assert len(loaded["messages"]) == 4
