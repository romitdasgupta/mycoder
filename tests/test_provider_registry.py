# tests/test_provider_registry.py
import pytest
from unittest.mock import patch
from mycoder.providers import create_provider, PROVIDER_MAP


def test_provider_map_has_all_providers():
    assert "anthropic" in PROVIDER_MAP
    assert "openai" in PROVIDER_MAP
    assert "google" in PROVIDER_MAP


@patch("mycoder.providers.anthropic.anthropic.Anthropic")
def test_create_anthropic_provider(mock_cls):
    provider = create_provider("anthropic", api_key="test", model="claude-sonnet-4-5-20250929")
    from mycoder.providers.anthropic import AnthropicProvider
    assert isinstance(provider, AnthropicProvider)


def test_create_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nonexistent", api_key="test", model="test")
