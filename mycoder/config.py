"""Configuration loading — provider, API key, and model selection."""

import os
import sys
from dataclasses import dataclass


PROVIDER_DEFAULTS = {
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-5-20250929"},
    "openai":    {"env_key": "OPENAI_API_KEY",    "default_model": "gpt-4o"},
    "google":    {"env_key": "GOOGLE_API_KEY",     "default_model": "gemini-2.0-flash"},
}


@dataclass
class Config:
    provider: str
    api_key: str
    model: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    provider = os.environ.get("MYCODER_PROVIDER", "anthropic")

    if provider not in PROVIDER_DEFAULTS:
        print(f"Error: Unknown provider {provider!r}. Available: {', '.join(sorted(PROVIDER_DEFAULTS))}")
        sys.exit(1)

    info = PROVIDER_DEFAULTS[provider]
    api_key = os.environ.get(info["env_key"])
    if not api_key:
        print(f"Error: {info['env_key']} environment variable not set.")
        sys.exit(1)

    model = os.environ.get("MYCODER_MODEL", info["default_model"])

    return Config(provider=provider, api_key=api_key, model=model)
