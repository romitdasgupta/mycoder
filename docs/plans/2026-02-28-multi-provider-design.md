# Multi-Provider LLM Abstraction Design

## Goal

Abstract the LLM API layer so that plugging in a new provider (OpenAI, Google, Groq, etc.) requires only one new file implementing a simple protocol. The agent loop becomes provider-agnostic.

## Constraints

- All providers must support tool/function calling (no graceful degradation to chat-only)
- Provider selected via `MYCODER_PROVIDER` env var or `--provider` CLI flag
- Model selected via `MYCODER_MODEL` env var or `--model` CLI flag
- Each provider uses its native SDK

## Architecture

```
Agent loop  →  LLMProvider protocol  →  AnthropicProvider (anthropic SDK)
                                     →  OpenAIProvider    (openai SDK)
                                     →  GoogleProvider    (google-genai SDK)
```

## Common Types (`providers/base.py`)

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    done: bool  # True = end_turn, False = needs tool results

@dataclass
class ToolResult:
    tool_call_id: str
    content: str

@dataclass
class StreamEvent:
    type: Literal["text_delta", "tool_call_start", "tool_call_delta", "done"]
    text: str | None = None
    tool_call: ToolCall | None = None
```

## Provider Protocol

```python
class LLMProvider(Protocol):
    def send(self, messages, system, tools) -> LLMResponse: ...
    def stream(self, messages, system, tools) -> Iterator[StreamEvent]: ...
```

Each provider translates:
1. Tool schemas from common format → native SDK format
2. Message history (including tool results) from common format → native format
3. Native response → LLMResponse / StreamEvent

## File Structure

```
mycoder/
  providers/
    __init__.py       # Provider registry + factory function
    base.py           # Protocol, dataclasses, common types
    anthropic.py      # AnthropicProvider
    openai.py         # OpenAIProvider (also covers OpenAI-compatible APIs)
    google.py         # GoogleProvider
  agent.py            # Refactored: uses LLMProvider, not anthropic directly
  config.py           # Extended: provider selection, per-provider env vars
  cli.py              # Updated: --provider flag
```

## Config Changes

Provider registry with per-provider defaults:
```python
PROVIDERS = {
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-5-20250929"},
    "openai":    {"env_key": "OPENAI_API_KEY",    "default_model": "gpt-4o"},
    "google":    {"env_key": "GOOGLE_API_KEY",     "default_model": "gemini-2.0-flash"},
}
```

## Agent Loop Changes

Before: `self.client = anthropic.Anthropic(...)` and Anthropic-specific response handling.

After: `self.provider: LLMProvider` and provider-agnostic response handling via `LLMResponse`.

The agent loop calls `provider.send()` or `provider.stream()`, processes `LLMResponse.tool_calls`, sends back `ToolResult` objects, and repeats until `response.done`.

## Adding a New Provider

1. Create `mycoder/providers/newprovider.py`
2. Implement `send()` and `stream()` methods
3. Add entry to provider registry in `__init__.py`
4. Add the SDK to `pyproject.toml` dependencies (as optional)

## Dependencies

- `anthropic` (existing)
- `openai` (new, optional)
- `google-genai` (new, optional)

Optional deps via `pyproject.toml` extras:
```toml
[project.optional-dependencies]
openai = ["openai>=1.0.0"]
google = ["google-genai>=1.0.0"]
all = ["openai>=1.0.0", "google-genai>=1.0.0"]
```
