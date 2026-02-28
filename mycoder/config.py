"""Configuration loading — API key from env, model selection."""

import os
import sys
from dataclasses import dataclass


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


@dataclass
class Config:
    api_key: str
    model: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    model = os.environ.get("MYCODER_MODEL", DEFAULT_MODEL)

    return Config(api_key=api_key, model=model)
