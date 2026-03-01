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
