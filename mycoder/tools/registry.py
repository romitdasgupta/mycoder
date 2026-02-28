"""Tool registry — register tools, export schemas, dispatch calls."""

from typing import Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[[dict], str],
        input_schema: dict,
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "input_schema": input_schema,
        }

    def get_schemas(self) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, input_data: dict) -> str:
        if name not in self._tools:
            return f"Unknown tool: {name}"
        try:
            result = self._tools[name]["handler"](input_data)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"
