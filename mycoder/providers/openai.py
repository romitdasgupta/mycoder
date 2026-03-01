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
                yield StreamEvent(
                    type="done",
                    tool_call=final_tool_calls[0] if final_tool_calls else None,
                )

    @staticmethod
    def _build_messages(messages: list[dict], system: str) -> list[dict]:
        """Convert internal message format to OpenAI format."""
        oai: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        oai.append({
                            "role": "tool",
                            "tool_call_id": item["tool_use_id"],
                            "content": item["content"],
                        })
            elif msg["role"] == "assistant" and isinstance(msg["content"], list):
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
