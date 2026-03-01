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
                messages.append({"role": "assistant", "content": response.text or ""})
                if on_text and response.text:
                    on_text(response.text)
                return response.text or "", messages

            # Tool calls — execute each one and build results
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
