# mycoder/providers/__init__.py
"""LLM provider abstraction layer."""

from __future__ import annotations

from mycoder.providers.base import LLMProvider

# Lazy imports to avoid requiring all SDKs to be installed
PROVIDER_MAP: dict[str, str] = {
    "anthropic": "mycoder.providers.anthropic:AnthropicProvider",
    "openai": "mycoder.providers.openai:OpenAIProvider",
    "google": "mycoder.providers.google:GoogleProvider",
}


def create_provider(name: str, *, api_key: str, model: str, **kwargs) -> LLMProvider:
    """Create a provider instance by name.

    Raises ValueError if the provider is unknown.
    Raises ImportError if the provider's SDK is not installed.
    """
    if name not in PROVIDER_MAP:
        available = ", ".join(sorted(PROVIDER_MAP))
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")

    module_path, class_name = PROVIDER_MAP[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(api_key=api_key, model=model, **kwargs)
