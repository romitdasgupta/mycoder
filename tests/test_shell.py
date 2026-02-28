from mycoder.tools.shell import run_command


def test_run_command_stdout():
    result = run_command({"command": "echo hello"})
    assert "hello" in result


def test_run_command_stderr():
    result = run_command({"command": "echo err >&2"})
    assert "err" in result


def test_run_command_timeout():
    result = run_command({"command": "sleep 60", "timeout": 1})
    assert "timed out" in result.lower()


def test_run_command_failure():
    result = run_command({"command": "exit 1"})
    # Should still return, not raise
    assert isinstance(result, str)
