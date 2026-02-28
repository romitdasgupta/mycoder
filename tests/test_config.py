import os
import pytest
from mycoder.config import load_config


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-123")
    cfg = load_config()
    assert cfg.api_key == "sk-test-key-123"
    assert cfg.model == "claude-sonnet-4-5-20250929"


def test_load_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        load_config()


def test_load_config_custom_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MYCODER_MODEL", "claude-haiku-4-5-20251001")
    cfg = load_config()
    assert cfg.model == "claude-haiku-4-5-20251001"
