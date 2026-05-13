"""MetaToolRepository: wraps any ToolRepository and exposes it to LLM agents
as exactly two tools — get_tool_definition and execute_tool.

Works with any LLM SDK (Anthropic, OpenAI, Gemini, etc.).

Example:
    platform_repo = MCPServerToolRepository(MCPServerConfig(name="platform", url="..."))
    composite     = CompositeToolRepository({"platform": platform_repo})
    meta          = MetaToolRepository(composite)
    await meta.connect()

    # tools the LLM sees
    tools = await meta.list_tools()
    # → [ToolInfo(get_tool_definition), ToolInfo(execute_tool)]

    # dispatch an LLM tool call
    result = await meta.execute_tool(
        "execute_tool",
        {"tool_name": "platform_createPage", "tool_args": {"name": "Home"}},
    )
"""

from typing import Any

from .models import ToolInfo
from .repository import ToolNotFoundError, ToolRepository

_GET_TOOL_DEFINITION_INFO = ToolInfo(
    name="get_tool_definition",
    server="meta",
    description=(
        "Get the full definition (description + input schema) of one or more tools by name. "
        "Use this before execute_tool to discover required arguments."
    ),
    input_schema={
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
)

_EXECUTE_TOOL_INFO = ToolInfo(
    name="execute_tool",
    server="meta",
    description=(
        "Execute a named tool from the MCP tool registry. "
        "Use get_tool_definition first to discover the required argument schema."
    ),
    input_schema={
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
        "required": ["tool_name"],
    },
)


class MetaToolRepository(ToolRepository):
    """Wraps any ToolRepository and exposes it to LLM agents as two meta-tools."""

    def __init__(self, tool_repo: ToolRepository) -> None:
        self._tool_repo = tool_repo

    async def connect(self) -> None:
        await self._tool_repo.connect()

    async def list_tools(self) -> list[ToolInfo]:
        return [_GET_TOOL_DEFINITION_INFO, _EXECUTE_TOOL_INFO]

    async def execute_tool(self, tool_name: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        if tool_name == "get_tool_definition":
            return await self._get_tool_definition(args)
        if tool_name == "execute_tool":
            return await self._execute_tool(args, meta)
        raise ToolNotFoundError(tool_name)

    async def close(self) -> None:
        await self._tool_repo.close()

    async def _get_tool_definition(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        requested_tool_names = args.get("tool_names", [])
        tool_registry = {tool.name: tool for tool in await self._tool_repo.list_tools()}
        missing_tools = [name for name in requested_tool_names if name not in tool_registry]
        if missing_tools:
            raise ToolNotFoundError(missing_tools)
        return [
            {
                "name": tool_registry[name].name,
                "server": tool_registry[name].server,
                "description": tool_registry[name].description,
                "input_schema": tool_registry[name].input_schema,
            }
            for name in requested_tool_names
        ]

    async def _execute_tool(self, args: dict[str, Any], meta: dict[str, Any] | None) -> Any:
        tool_name = args.get("tool_name")
        tool_args = args.get("tool_args") or {}
        if not tool_name:
            raise ValueError("execute_tool requires 'tool_name'")
        return await self._tool_repo.execute_tool(tool_name, tool_args, meta=meta)
