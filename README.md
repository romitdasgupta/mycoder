# mycoder

A terminal CLI coding assistant with multi-provider LLM support.

mycoder gives you an interactive REPL with tool-calling capabilities — file reading/writing, shell commands, code search — powered by your choice of LLM provider.

## Supported Providers

| Provider | Models | Env Var |
|----------|--------|---------|
| Anthropic (default) | Claude Sonnet, Haiku, Opus | `ANTHROPIC_API_KEY` |
| OpenAI | GPT-4o, o3, etc. | `OPENAI_API_KEY` |
| Google | Gemini 2.0 Flash, Pro, etc. | `GOOGLE_API_KEY` |
| OpenAI-compatible | Groq, Together, Ollama, OpenRouter | `OPENAI_API_KEY` |

## Installation

```bash
# Default (Anthropic only)
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Google Gemini support
pip install -e ".[google]"

# All providers
pip install -e ".[all]"
```

## Usage

```bash
# Claude (default)
export ANTHROPIC_API_KEY=sk-...
mycoder

# OpenAI
export MYCODER_PROVIDER=openai
export OPENAI_API_KEY=sk-...
mycoder

# Google Gemini
export MYCODER_PROVIDER=google
export GOOGLE_API_KEY=AIza...
mycoder

# Override the default model for any provider
export MYCODER_MODEL=claude-haiku-4-5-20251001
mycoder
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit the REPL |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/model <name>` | Switch model (or show current) |
| `/resume` | Load the most recent session |

## Architecture

```
Agent loop  →  LLMProvider protocol  →  AnthropicProvider
                                     →  OpenAIProvider
                                     →  GoogleProvider
```

The agent loop is provider-agnostic. Each provider adapter translates between the common `LLMResponse`/`StreamEvent` types and its native SDK.

### Adding a New Provider

1. Create `mycoder/providers/yourprovider.py` implementing `send()` and `stream()`
2. Add an entry to `PROVIDER_MAP` in `mycoder/providers/__init__.py`
3. Add the default model to `PROVIDER_DEFAULTS` in `mycoder/config.py`
4. Add the SDK to `[project.optional-dependencies]` in `pyproject.toml`

### Built-in Tools

- `read_file` — Read file contents with optional line range
- `write_file` — Create or overwrite files
- `edit_file` — Replace a unique string in a file
- `glob` — Find files matching a pattern
- `grep` — Search file contents with regex
- `list_directory` — List files and directories
- `run_command` — Execute shell commands

## Project Structure

```
mycoder/
  providers/
    base.py          # LLMProvider protocol, ToolCall, LLMResponse, StreamEvent
    anthropic.py     # Anthropic (Claude) adapter
    openai.py        # OpenAI adapter (+ OpenAI-compatible APIs)
    google.py        # Google Gemini adapter
    __init__.py      # Provider registry and factory
  tools/
    registry.py      # Tool registration and dispatch
    files.py         # File operation tools
    shell.py         # Shell command tool
    builtin.py       # Default tool wiring
  memory/
    store.py         # Session persistence
  agent.py           # Provider-agnostic agent loop
  config.py          # Multi-provider configuration
  cli.py             # Interactive REPL
```
