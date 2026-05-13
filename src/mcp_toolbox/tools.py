"""
LLM-callable meta-tools that expose get_tool_definition and execute_tool
as JSON-schema tool definitions bound to a ToolRepository instance.

Works with any LLM SDK that accepts tool definitions as dicts
(Anthropic, OpenAI, Gemini, etc.).

Example (Anthropic):
    repo = ToolRepository([...])
    await repo.connect()

    meta_tools = make_tools(repo, meta_provider=my_auth_fn)
    # Pass meta_tools.definitions to the LLM as tools.
    # When the LLM calls a tool, dispatch via meta_tools.dispatch(name, args).
"""

from typing import Any, Callable, Optional

from .repository import ToolRepository


def make_tools(
    repo: ToolRepository,
    meta_provider: Optional[Callable[[str], dict | None]] = None,
) -> "MetaToolSet":
    return MetaToolSet(repo, meta_provider=meta_provider)


class MetaToolSet:
    """Holds get_tool_definition and execute_tool bound to a ToolRepository.

    The optional ``meta_provider`` callback is invoked with the tool_name before
    each execute_tool call. Return a dict to forward as MCP ``meta=`` (e.g. auth
    context), or ``None`` to skip.
    """

    def __init__(
        self,
        repo: ToolRepository,
        meta_provider: Optional[Callable[[str], dict | None]] = None,
    ) -> None:
        self._repo = repo
        self._meta_provider = meta_provider
        self.definitions = [
            _GET_TOOL_DEFINITION_SCHEMA,
            _EXECUTE_TOOL_SCHEMA,
        ]

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "get_tool_definition":
            return await self._get_tool_definition(**arguments)
        if tool_name == "execute_tool":
            return await self._execute_tool(**arguments)
        raise ValueError(f"Unknown meta-tool: '{tool_name}'")

    async def _get_tool_definition(self, tool_names: list[str]) -> list[dict[str, Any]]:
        defns = await self._repo.get_tool_definition(tool_names)
        return [
            {
                "name": d.name,
                "server": d.server,
                "description": d.description,
                "input_schema": d.input_schema,
            }
            for d in defns
        ]

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        meta = self._meta_provider(tool_name) if self._meta_provider else None
        return await self._repo.execute_tool(tool_name, tool_args, meta=meta)


_GET_TOOL_DEFINITION_SCHEMA = {
    "name": "get_tool_definition",
    "description": (
        "Get the full definition (description + input schema) for one or more tools by name. "
        "Use this before calling execute_tool to understand required arguments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more exact tool names to look up.",
                "minItems": 1,
            }
        },
        "required": ["tool_names"],
    },
}

_EXECUTE_TOOL_SCHEMA = {
    "name": "execute_tool",
    "description": (
        "Execute a named tool from the MCP tool registry with the given arguments. "
        "Use get_tool_definition first to discover the required argument schema."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "The exact name of the tool to execute.",
            },
            "tool_args": {
                "type": "object",
                "description": "Key-value arguments matching the tool's input schema.",
            },
        },
        "required": ["tool_name", "tool_args"],
    },
}
