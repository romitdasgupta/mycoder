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
