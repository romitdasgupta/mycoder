import pytest
from mycoder.config import load_config


def test_load_config_defaults(monkeypatch):
    """Default provider is anthropic with its default model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-123")
    cfg = load_config()
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-test-key-123"
    assert cfg.model == "claude-sonnet-4-5-20250929"


def test_load_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MYCODER_PROVIDER", raising=False)
    with pytest.raises(SystemExit):
        load_config()


def test_load_config_openai(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    cfg = load_config()
    assert cfg.provider == "openai"
    assert cfg.api_key == "sk-openai-test"
    assert cfg.model == "gpt-4o"


def test_load_config_custom_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MYCODER_MODEL", "claude-haiku-4-5-20251001")
    cfg = load_config()
    assert cfg.model == "claude-haiku-4-5-20251001"


def test_load_config_google(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
    cfg = load_config()
    assert cfg.provider == "google"
    assert cfg.api_key == "AIza-test"
    assert cfg.model == "gemini-2.0-flash"


def test_load_config_unknown_provider(monkeypatch):
    monkeypatch.setenv("MYCODER_PROVIDER", "nonexistent")
    with pytest.raises(SystemExit):
        load_config()
