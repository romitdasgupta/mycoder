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
