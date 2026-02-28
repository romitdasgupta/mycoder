"""Agent loop — iterate on tool calls until Claude produces a final response."""

import anthropic
from mycoder.tools.registry import ToolRegistry

SYSTEM_PROMPT = "You are a helpful coding assistant with access to the filesystem and shell. Use your tools to read, write, and edit files, search the codebase, and run commands."

MAX_ITERATIONS = 30


class Agent:
    def __init__(self, api_key: str, model: str, registry: ToolRegistry):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=schemas,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Extract final text
                messages.append({"role": "assistant", "content": response.content})
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                        if on_text:
                            on_text(block.text)
                return final_text, messages

            elif response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self.registry.execute(block.name, block.input)
                        if on_tool_call:
                            on_tool_call(block.name, block.input, result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return "(max iterations reached)", messages
