# Multi-Provider LLM Abstraction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Abstract the LLM API layer behind a `LLMProvider` protocol so adding a new provider requires one file.

**Architecture:** Each provider implements `send()` and `stream()` using its native SDK. The agent loop speaks a common `LLMResponse`/`StreamEvent` format and never imports a provider SDK directly. A factory function maps provider name → class.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `openai` SDK, `google-genai` SDK, `typing.Protocol`

---

### Task 1: Common types (`providers/base.py`)

**Files:**
- Create: `mycoder/providers/__init__.py`
- Create: `mycoder/providers/base.py`
- Test: `tests/test_providers_base.py`

**Step 1: Write the failing test**

```python
# tests/test_providers_base.py
from mycoder.providers.base import ToolCall, LLMResponse, ToolResult, StreamEvent


def test_llm_response_with_text():
    r = LLMResponse(text="hello", tool_calls=[], done=True)
    assert r.text == "hello"
    assert r.done is True
    assert r.tool_calls == []


def test_llm_response_with_tool_calls():
    tc = ToolCall(id="1", name="read_file", arguments={"path": "f.py"})
    r = LLMResponse(text=None, tool_calls=[tc], done=False)
    assert r.done is False
    assert r.tool_calls[0].name == "read_file"


def test_tool_result():
    tr = ToolResult(tool_call_id="1", content="file contents")
    assert tr.tool_call_id == "1"


def test_stream_event():
    e = StreamEvent(type="text_delta", text="hi")
    assert e.type == "text_delta"
    assert e.tool_call is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_providers_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mycoder.providers'`

**Step 3: Write minimal implementation**

```python
# mycoder/providers/__init__.py
"""LLM provider abstraction layer."""
```

```python
# mycoder/providers/base.py
"""Common types and protocol for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Result of executing a tool, sent back to the model."""
    tool_call_id: str
    content: str


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = True


@dataclass
class StreamEvent:
    """A single event from a streaming LLM response."""
    type: Literal["text_delta", "tool_call_start", "tool_call_delta", "done"]
    text: str | None = None
    tool_call: ToolCall | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that every provider adapter must implement."""

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> Iterator[StreamEvent]: ...
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_providers_base.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add mycoder/providers/__init__.py mycoder/providers/base.py tests/test_providers_base.py
git commit -m "feat: add common LLM provider types and protocol"
```

---

### Task 2: Anthropic provider adapter

**Files:**
- Create: `mycoder/providers/anthropic.py`
- Test: `tests/test_provider_anthropic.py`

**Step 1: Write the failing test**

```python
# tests/test_provider_anthropic.py
from unittest.mock import MagicMock, patch
from mycoder.providers.anthropic import AnthropicProvider
from mycoder.providers.base import LLMProvider


def _text_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _tool_response(name, args, tool_id="t1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = args
    block.id = tool_id
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def test_anthropic_implements_protocol():
    assert isinstance(AnthropicProvider, type)
    # runtime_checkable — verify an instance satisfies the protocol
    with patch("mycoder.providers.anthropic.anthropic.Anthropic"):
        provider = AnthropicProvider(api_key="test", model="test")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.anthropic.anthropic.Anthropic")
def test_send_text_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _text_response("Hello!")

    provider = AnthropicProvider(api_key="test", model="test")
    result = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        system="You are helpful.",
        tools=[],
    )
    assert result.text == "Hello!"
    assert result.done is True
    assert result.tool_calls == []


@patch("mycoder.providers.anthropic.anthropic.Anthropic")
def test_send_tool_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _tool_response(
        "read_file", {"path": "x.py"}
    )

    provider = AnthropicProvider(api_key="test", model="test")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="You are helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "x.py"}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_anthropic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mycoder.providers.anthropic'`

**Step 3: Write minimal implementation**

```python
# mycoder/providers/anthropic.py
"""Anthropic (Claude) provider adapter."""

from __future__ import annotations

from typing import Iterator

import anthropic

from mycoder.providers.base import LLMResponse, StreamEvent, ToolCall


class AnthropicProvider:
    """LLMProvider implementation using the Anthropic SDK."""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        response = self.client.messages.create(**kwargs)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> Iterator[StreamEvent]:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamEvent(type="text_delta", text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        yield StreamEvent(type="tool_call_delta", text=event.delta.partial_json)
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        yield StreamEvent(
                            type="tool_call_start",
                            tool_call=ToolCall(
                                id=event.content_block.id,
                                name=event.content_block.name,
                                arguments={},
                            ),
                        )
            # Yield the final parsed response as a done event
            final = stream.get_final_message()
            parsed = self._parse_response(final)
            yield StreamEvent(
                type="done",
                text=parsed.text,
                tool_call=parsed.tool_calls[0] if parsed.tool_calls else None,
            )

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            done=response.stop_reason == "end_turn",
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_anthropic.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add mycoder/providers/anthropic.py tests/test_provider_anthropic.py
git commit -m "feat: add Anthropic provider adapter"
```

---

### Task 3: OpenAI provider adapter

**Files:**
- Create: `mycoder/providers/openai.py`
- Test: `tests/test_provider_openai.py`

**Step 1: Write the failing test**

```python
# tests/test_provider_openai.py
from unittest.mock import MagicMock, patch
import json
from mycoder.providers.openai import OpenAIProvider
from mycoder.providers.base import LLMProvider


def _text_response(text):
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = text
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_response(name, args, tool_id="call_1"):
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    tc.type = "function"
    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message.content = None
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_openai_implements_protocol():
    with patch("mycoder.providers.openai.openai.OpenAI"):
        provider = OpenAIProvider(api_key="test", model="test")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.openai.openai.OpenAI")
def test_send_text(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _text_response("Hi!")

    provider = OpenAIProvider(api_key="test", model="gpt-4o")
    result = provider.send(
        messages=[{"role": "user", "content": "Hello"}],
        system="Be helpful.",
        tools=[],
    )
    assert result.text == "Hi!"
    assert result.done is True


@patch("mycoder.providers.openai.openai.OpenAI")
def test_send_tool_call(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _tool_response(
        "read_file", {"path": "x.py"}
    )

    provider = OpenAIProvider(api_key="test", model="gpt-4o")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="Be helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"


@patch("mycoder.providers.openai.openai.OpenAI")
def test_openai_compatible_base_url(mock_cls):
    """OpenAI-compatible APIs (Groq, Together, etc.) use base_url."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    provider = OpenAIProvider(api_key="test", model="llama-3", base_url="https://api.groq.com/openai/v1")
    mock_cls.assert_called_with(api_key="test", base_url="https://api.groq.com/openai/v1")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_openai.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

The key translation work here: Anthropic uses `input_schema` for tool parameters and content blocks for responses. OpenAI uses `parameters` in function definitions and a different tool_calls structure.

```python
# mycoder/providers/openai.py
"""OpenAI provider adapter — also works with OpenAI-compatible APIs."""

from __future__ import annotations

import json
from typing import Iterator

import openai

from mycoder.providers.base import LLMResponse, StreamEvent, ToolCall


class OpenAIProvider:
    """LLMProvider implementation using the OpenAI SDK.

    Also works with OpenAI-compatible APIs (Groq, Together, Ollama, etc.)
    by setting base_url.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.model = model

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        oai_messages = self._build_messages(messages, system)
        kwargs: dict = dict(model=self.model, messages=oai_messages)
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        response = self.client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> Iterator[StreamEvent]:
        oai_messages = self._build_messages(messages, system)
        kwargs: dict = dict(model=self.model, messages=oai_messages, stream=True)
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        tool_calls_accum: dict[int, dict] = {}
        for chunk in self.client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
                yield StreamEvent(type="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name or "",
                            "arguments": "",
                        }
                        yield StreamEvent(
                            type="tool_call_start",
                            tool_call=ToolCall(
                                id=tool_calls_accum[idx]["id"],
                                name=tool_calls_accum[idx]["name"],
                                arguments={},
                            ),
                        )
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_accum[idx]["arguments"] += tc_delta.function.arguments
                        yield StreamEvent(type="tool_call_delta", text=tc_delta.function.arguments)

            if chunk.choices[0].finish_reason:
                final_tool_calls = []
                for acc in tool_calls_accum.values():
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                    final_tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=args))
                done = chunk.choices[0].finish_reason == "stop"
                yield StreamEvent(
                    type="done",
                    tool_call=final_tool_calls[0] if final_tool_calls else None,
                )

    @staticmethod
    def _build_messages(messages: list[dict], system: str) -> list[dict]:
        """Convert internal message format to OpenAI format.

        Internal format uses Anthropic-style tool results (role=user, content=[{type: tool_result, ...}]).
        OpenAI expects role=tool messages.
        """
        oai: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                # Tool results — convert to OpenAI's role=tool format
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        oai.append({
                            "role": "tool",
                            "tool_call_id": item["tool_use_id"],
                            "content": item["content"],
                        })
            elif msg["role"] == "assistant" and isinstance(msg["content"], list):
                # Assistant content blocks — convert tool_use blocks to OpenAI tool_calls
                text_parts = []
                tool_calls = []
                for block in msg["content"]:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            })
                oai_msg: dict = {"role": "assistant", "content": "".join(text_parts) or None}
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                oai.append(oai_msg)
            else:
                oai.append(msg)
        return oai

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool schemas to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        choice = response.choices[0]
        text = choice.message.content
        tool_calls: list[ToolCall] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        done = choice.finish_reason == "stop"
        return LLMResponse(text=text, tool_calls=tool_calls, done=done)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_openai.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add mycoder/providers/openai.py tests/test_provider_openai.py
git commit -m "feat: add OpenAI provider adapter with OpenAI-compatible API support"
```

---

### Task 4: Google Gemini provider adapter

**Files:**
- Create: `mycoder/providers/google.py`
- Test: `tests/test_provider_google.py`

**Step 1: Write the failing test**

```python
# tests/test_provider_google.py
from unittest.mock import MagicMock, patch
from mycoder.providers.google import GoogleProvider
from mycoder.providers.base import LLMProvider


def test_google_implements_protocol():
    with patch("mycoder.providers.google.genai.Client"):
        provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
        assert isinstance(provider, LLMProvider)


@patch("mycoder.providers.google.genai.Client")
def test_send_text(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    # Simulate Gemini response with text only
    mock_part = MagicMock()
    mock_part.text = "Hello!"
    mock_part.function_call = None
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
    result = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        system="Be helpful.",
        tools=[],
    )
    assert result.text == "Hello!"
    assert result.done is True


@patch("mycoder.providers.google.genai.Client")
def test_send_tool_call(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    mock_fc = MagicMock()
    mock_fc.name = "read_file"
    mock_fc.args = {"path": "x.py"}
    mock_fc.id = "fc1"
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.function_call = mock_fc
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    provider = GoogleProvider(api_key="test", model="gemini-2.0-flash")
    result = provider.send(
        messages=[{"role": "user", "content": "Read x.py"}],
        system="Be helpful.",
        tools=[{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    )
    assert result.done is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_google.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Gemini uses `google.genai` (the new unified SDK). Tool schemas use `function_declarations` and responses use `function_call` parts.

```python
# mycoder/providers/google.py
"""Google Gemini provider adapter."""

from __future__ import annotations

import uuid
from typing import Iterator

from google import genai
from google.genai import types

from mycoder.providers.base import LLMResponse, StreamEvent, ToolCall


class GoogleProvider:
    """LLMProvider implementation using the Google GenAI SDK."""

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def send(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        contents = self._build_contents(messages)
        config = types.GenerateContentConfig(system_instruction=system)
        if tools:
            config.tools = [self._convert_tools(tools)]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> Iterator[StreamEvent]:
        contents = self._build_contents(messages)
        config = types.GenerateContentConfig(system_instruction=system)
        if tools:
            config.tools = [self._convert_tools(tools)]

        final_text_parts: list[str] = []
        final_tool_calls: list[ToolCall] = []

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    final_text_parts.append(part.text)
                    yield StreamEvent(type="text_delta", text=part.text)
                elif part.function_call:
                    tc = ToolCall(
                        id=getattr(part.function_call, "id", None) or uuid.uuid4().hex[:8],
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                    )
                    final_tool_calls.append(tc)
                    yield StreamEvent(type="tool_call_start", tool_call=tc)

        yield StreamEvent(
            type="done",
            text="".join(final_text_parts) if final_text_parts else None,
            tool_call=final_tool_calls[0] if final_tool_calls else None,
        )

    @staticmethod
    def _build_contents(messages: list[dict]) -> list[types.Content]:
        """Convert internal message format to Gemini contents."""
        contents: list[types.Content] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"

            if isinstance(msg["content"], str):
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            elif isinstance(msg["content"], list):
                parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        # Tool result → FunctionResponse
                        parts.append(types.Part.from_function_response(
                            name="tool_result",
                            response={"result": item["content"]},
                        ))
                    elif hasattr(item, "type"):
                        if item.type == "text":
                            parts.append(types.Part.from_text(text=item.text))
                        elif item.type == "tool_use":
                            parts.append(types.Part(function_call=types.FunctionCall(
                                name=item.name, args=item.input,
                            )))
                if parts:
                    contents.append(types.Content(role=role, parts=parts))
        return contents

    @staticmethod
    def _convert_tools(tools: list[dict]) -> types.Tool:
        """Convert Anthropic-style tool schemas to Gemini function declarations."""
        declarations = []
        for t in tools:
            declarations.append(types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            ))
        return types.Tool(function_declarations=declarations)

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                tool_calls.append(ToolCall(
                    id=getattr(part.function_call, "id", None) or uuid.uuid4().hex[:8],
                    name=part.function_call.name,
                    arguments=dict(part.function_call.args) if part.function_call.args else {},
                ))

        has_tool_calls = len(tool_calls) > 0
        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            done=not has_tool_calls,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_google.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add mycoder/providers/google.py tests/test_provider_google.py
git commit -m "feat: add Google Gemini provider adapter"
```

---

### Task 5: Provider registry and factory

**Files:**
- Modify: `mycoder/providers/__init__.py`
- Test: `tests/test_provider_registry.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_provider'`

**Step 3: Write minimal implementation**

```python
# mycoder/providers/__init__.py
"""LLM provider abstraction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mycoder.providers.base import LLMProvider

if TYPE_CHECKING:
    pass

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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_provider_registry.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add mycoder/providers/__init__.py tests/test_provider_registry.py
git commit -m "feat: add provider registry with lazy-import factory"
```

---

### Task 6: Refactor agent.py to use LLMProvider

**Files:**
- Modify: `mycoder/agent.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_integration.py`

**Step 1: Update agent tests to use the new interface**

The agent should accept an `LLMProvider` instead of `api_key` + `model`. Update tests first.

```python
# tests/test_agent.py — full replacement
from unittest.mock import MagicMock
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.providers.base import LLMResponse, ToolCall


def test_agent_simple_text_response():
    provider = MagicMock()
    provider.send.return_value = LLMResponse(text="Hello!", tool_calls=[], done=True)

    reg = create_default_registry()
    agent = Agent(provider=provider, registry=reg)
    messages = [{"role": "user", "content": "Hi"}]
    text, updated = agent.step(messages)

    assert text == "Hello!"
    assert len(updated) > len(messages)


def test_agent_tool_then_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("file content here")

    provider = MagicMock()
    provider.send.side_effect = [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="id1", name="read_file", arguments={"path": str(f)})],
            done=False,
        ),
        LLMResponse(text="I read the file.", tool_calls=[], done=True),
    ]

    reg = create_default_registry()
    agent = Agent(provider=provider, registry=reg)
    messages = [{"role": "user", "content": "Read test.txt"}]
    text, updated = agent.step(messages)

    assert text == "I read the file."
    assert provider.send.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_agent.py -v`
Expected: FAIL — `Agent.__init__() got unexpected keyword argument 'provider'`

**Step 3: Refactor agent.py**

```python
# mycoder/agent.py — full replacement
"""Agent loop — iterate on tool calls until the provider produces a final response."""

from mycoder.providers.base import LLMProvider, LLMResponse, ToolCall
from mycoder.tools.registry import ToolRegistry

SYSTEM_PROMPT = "You are a helpful coding assistant with access to the filesystem and shell. Use your tools to read, write, and edit files, search the codebase, and run commands."

MAX_ITERATIONS = 30


class Agent:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry):
        self.provider = provider
        self.registry = registry

    def step(self, messages: list[dict], on_tool_call=None, on_text=None) -> tuple[str, list[dict]]:
        """Run the agent loop for one user turn.

        Args:
            messages: Full conversation history.
            on_tool_call: Optional callback(name, input_data, result) for display.
            on_text: Optional callback(text_chunk) for streaming display.

        Returns:
            (final_text, updated_messages) — the final text response and the
            full updated message history including assistant + tool results.
        """
        messages = list(messages)  # don't mutate caller's list
        schemas = self.registry.get_schemas()

        for _ in range(MAX_ITERATIONS):
            response: LLMResponse = self.provider.send(messages, SYSTEM_PROMPT, schemas)

            if response.done:
                # Append a simple assistant message for the final text
                messages.append({"role": "assistant", "content": response.text or ""})
                if on_text and response.text:
                    on_text(response.text)
                return response.text or "", messages

            # Tool calls — execute each one and build results
            # Store tool call info in assistant message for history
            messages.append({
                "role": "assistant",
                "content": f"[tool calls: {', '.join(tc.name for tc in response.tool_calls)}]",
                "_tool_calls": response.tool_calls,
            })

            tool_results = []
            for tc in response.tool_calls:
                result = self.registry.execute(tc.name, tc.arguments)
                if on_tool_call:
                    on_tool_call(tc.name, tc.arguments, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        return "(max iterations reached)", messages
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_agent.py -v`
Expected: 2 passed

**Step 5: Update integration test**

```python
# tests/test_integration.py — full replacement
"""Integration test — verify all pieces wire together."""

from unittest.mock import MagicMock
from mycoder.agent import Agent
from mycoder.tools.builtin import create_default_registry
from mycoder.memory.store import SessionStore
from mycoder.providers.base import LLMResponse, ToolCall


def test_full_round_trip(tmp_path):
    """Agent reads a file via tool, session is saved and loadable."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello from integration test!")

    registry = create_default_registry()
    store = SessionStore(base_dir=tmp_path / "sessions")
    session = store.new_session(model="test", cwd=str(tmp_path))

    provider = MagicMock()
    provider.send.side_effect = [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="tool1", name="read_file", arguments={"path": str(test_file)})],
            done=False,
        ),
        LLMResponse(text="The file says hello!", tool_calls=[], done=True),
    ]

    agent = Agent(provider=provider, registry=registry)
    session["messages"].append({"role": "user", "content": "Read hello.txt"})

    text, updated = agent.step(session["messages"])
    session["messages"] = updated

    assert text == "The file says hello!"
    assert len(session["messages"]) == 4  # user, assistant(tool), user(result), assistant(text)

    store.save(session)
    loaded = store.load(session["id"])
    assert loaded is not None
    assert len(loaded["messages"]) == 4
```

**Step 6: Run all tests**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/ -v`
Expected: All tests pass (some config tests may need updating — see Task 7)

**Step 7: Commit**

```bash
git add mycoder/agent.py tests/test_agent.py tests/test_integration.py
git commit -m "refactor: decouple agent loop from Anthropic SDK, use LLMProvider protocol"
```

---

### Task 7: Update config.py for multi-provider support

**Files:**
- Modify: `mycoder/config.py`
- Modify: `tests/test_config.py`

**Step 1: Update config tests**

```python
# tests/test_config.py — full replacement
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_config.py -v`
Expected: FAIL — `Config has no attribute 'provider'`

**Step 3: Update config.py**

```python
# mycoder/config.py — full replacement
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/test_config.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add mycoder/config.py tests/test_config.py
git commit -m "feat: extend config for multi-provider selection"
```

---

### Task 8: Update cli.py to wire everything together

**Files:**
- Modify: `mycoder/cli.py`

**Step 1: No new test needed — this wires existing pieces. Read current cli.py.**

The CLI needs to:
1. Use `cfg.provider` to create the right provider via `create_provider()`
2. Pass `provider` to `Agent()` instead of `api_key` + `model`
3. The `/model` command should update the provider's model attribute

**Step 2: Update cli.py**

Replace lines 13-14 imports and update `main()` and `handle_command()`:

In `cli.py`, change:
- Import: add `from mycoder.providers import create_provider`
- Remove: `from mycoder.agent import Agent` stays, but constructor call changes
- `main()`: create provider, pass to Agent
- `handle_command()`: the `/model` command sets `provider.model`

```python
# mycoder/cli.py — full replacement
"""CLI REPL — interactive terminal interface with streaming output."""

import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mycoder.config import load_config
from mycoder.agent import Agent
from mycoder.providers import create_provider
from mycoder.tools.builtin import create_default_registry
from mycoder.memory.store import SessionStore


console = Console()


def display_tool_call(name: str, input_data: dict, result: str) -> None:
    """Display a tool call in a formatted panel."""
    summary_parts = []
    for k, v in input_data.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        summary_parts.append(f"{k}={s}")
    summary = ", ".join(summary_parts)

    display_result = result if len(result) <= 200 else result[:197] + "..."

    console.print(Panel(
        f"[dim]{summary}[/dim]\n\n{display_result}",
        title=f"[bold cyan]{name}[/bold cyan]",
        border_style="dim",
        expand=False,
    ))


def handle_command(cmd: str, session: dict, store: SessionStore, agent: Agent) -> bool:
    """Handle slash commands. Returns True if should continue REPL, False to quit."""
    if cmd == "/quit":
        return False
    elif cmd == "/clear":
        session["messages"] = []
        console.print("[dim]Conversation cleared.[/dim]")
    elif cmd == "/history":
        for msg in session["messages"]:
            role = msg["role"]
            content = msg["content"] if isinstance(msg["content"], str) else "(tool data)"
            console.print(f"[bold]{role}:[/bold] {content[:100]}")
    elif cmd.startswith("/model"):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 2:
            agent.provider.model = parts[1]
            session["model"] = parts[1]
            console.print(f"[dim]Model set to {parts[1]}[/dim]")
        else:
            console.print(f"[dim]Current model: {agent.provider.model}[/dim]")
    elif cmd == "/resume":
        loaded = store.load_latest()
        if loaded:
            session.update(loaded)
            console.print(f"[dim]Resumed session {loaded['id']} ({len(loaded['messages'])} messages)[/dim]")
        else:
            console.print("[dim]No previous session found.[/dim]")
    else:
        console.print(f"[dim]Unknown command: {cmd}[/dim]")
    return True


def main():
    cfg = load_config()

    provider = create_provider(cfg.provider, api_key=cfg.api_key, model=cfg.model)
    registry = create_default_registry()
    agent = Agent(provider=provider, registry=registry)
    store = SessionStore()
    session = store.new_session(model=cfg.model, cwd=os.getcwd())

    history_path = os.path.expanduser("~/.mycoder/input_history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    prompt_session = PromptSession(history=FileHistory(history_path))

    console.print(f"[bold]mycoder[/bold] — your coding assistant [dim]({cfg.provider}:{cfg.model})[/dim]")
    console.print("[dim]Type /quit to exit, /clear to reset, /resume to load last session[/dim]\n")

    while True:
        try:
            user_input = prompt_session.prompt("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, session, store, agent):
                break
            continue

        session["messages"].append({"role": "user", "content": user_input})

        console.print()
        try:
            text, updated_messages = agent.step(
                session["messages"],
                on_tool_call=display_tool_call,
            )
            session["messages"] = updated_messages
            console.print(f"\n[bold]mycoder:[/bold] {text}\n")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")

        store.save(session)

    console.print("\n[dim]Goodbye.[/dim]")
    store.save(session)


if __name__ == "__main__":
    main()
```

**Step 3: Run all tests**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add mycoder/cli.py
git commit -m "feat: wire CLI to use provider abstraction"
```

---

### Task 9: Update pyproject.toml with optional dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add optional dependency groups so users can install only the SDKs they need:

```toml
[project]
# ... existing fields ...
dependencies = [
    "anthropic>=0.40.0",
    "prompt-toolkit>=3.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0.0"]
google = ["google-genai>=1.0.0"]
all = ["openai>=1.0.0", "google-genai>=1.0.0"]
```

**Step 2: Install and verify**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && pip install -e ".[all]"`
Expected: installs openai and google-genai SDKs

**Step 3: Run all tests one final time**

Run: `cd /Users/romitdasgupta/code/agentic/mycoder && python -m pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add optional deps for OpenAI and Google providers"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Common types + protocol | `providers/base.py` |
| 2 | Anthropic adapter | `providers/anthropic.py` |
| 3 | OpenAI adapter | `providers/openai.py` |
| 4 | Google adapter | `providers/google.py` |
| 5 | Provider registry + factory | `providers/__init__.py` |
| 6 | Refactor agent loop | `agent.py` |
| 7 | Multi-provider config | `config.py` |
| 8 | Wire CLI | `cli.py` |
| 9 | Optional deps | `pyproject.toml` |

Tasks 1-5 can be built independently. Task 6 depends on Task 1. Tasks 7-8 depend on 5+6. Task 9 is standalone.
