"""Optional LangChain integration for MetaToolRepository.

Requires the [langchain] extra:
    pip install mcp-toolbox[langchain]

Usage:
    from mcp_toolbox import MetaToolRepository
    from mcp_toolbox.langchain_tools import convert_to_langchain_tools

    meta_repo = MetaToolRepository(composite_repo)
    tools = convert_to_langchain_tools(meta_repo, meta_provider=my_provider)
    # → [execute_tool (BaseTool), get_tool_definition (BaseTool)]
"""

from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool, ToolException
from mcp.types import CallToolResult
from pydantic import BaseModel, Field, PrivateAttr, create_model

from .tools import MetaToolRepository


def _mcp_result_to_str(result: Any) -> str:
    if not isinstance(result, CallToolResult):
        return str(result)
    parts = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(f"[image/{getattr(item, 'mimeType', 'unknown')}]")
    text = "\n".join(parts)
    if result.isError:
        raise ToolException(text or "MCP tool returned an error")
    return text


class _ExecuteToolShim(BaseTool):
    name: str = "execute_tool"
    description: str = (
        "Execute a named tool from the MCP tool registry. "
        "Use get_tool_definition first to discover the required argument schema."
    )
    args_schema: type[BaseModel]
    handle_tool_error: bool = True

    _meta_repo: Any = PrivateAttr()
    _meta_provider: Any = PrivateAttr(default=None)

    def __init__(self, meta_repo: MetaToolRepository, meta_provider: Callable | None = None) -> None:
        schema = create_model(
            "ExecuteToolSchema",
            tool_name=(str, Field(description="The exact name of the tool to execute.")),
            tool_args=(Optional[dict], Field(default=None, description="Key-value arguments matching the tool's input schema.")),
        )
        super().__init__(args_schema=schema)
        self._meta_repo = meta_repo
        self._meta_provider = meta_provider

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Only async execution is supported")

    async def _arun(self, tool_name: str, tool_args: dict | None = None) -> Any:
        meta = self._meta_provider(tool_name) if self._meta_provider else None
        result = await self._meta_repo.execute_tool(
            "execute_tool", {"tool_name": tool_name, "tool_args": tool_args or {}}, meta=meta
        )
        return _mcp_result_to_str(result)


class _GetToolDefinitionShim(BaseTool):
    name: str = "get_tool_definition"
    description: str = (
        "Get the full definition (description + input schema) of one or more tools by name. "
        "Use this before execute_tool to discover required arguments."
    )
    args_schema: type[BaseModel]
    handle_tool_error: bool = True

    _meta_repo: Any = PrivateAttr()

    def __init__(self, meta_repo: MetaToolRepository) -> None:
        schema = create_model(
            "GetToolDefinitionSchema",
            tool_names=(list, Field(description="One or more exact tool names to look up.")),
        )
        super().__init__(args_schema=schema)
        self._meta_repo = meta_repo

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Only async execution is supported")

    async def _arun(self, tool_names: list) -> Any:
        return await self._meta_repo.execute_tool("get_tool_definition", {"tool_names": tool_names})


def convert_to_langchain_tools(
    meta_repo: MetaToolRepository,
    meta_provider: Callable[[str], dict | None] | None = None,
) -> list[BaseTool]:
    """Convert a MetaToolRepository into LangChain BaseTool objects.

    Returns [execute_tool, get_tool_definition] as BaseTool instances
    that can be bound to a LangChain agent or graph node.

    Args:
        meta_repo: The MetaToolRepository wrapping your MCP servers.
        meta_provider: Optional callback (tool_name) -> dict | None that
            returns auth/meta context to forward to the MCP server for
            trusted tool calls.
    """
    return [
        _ExecuteToolShim(meta_repo, meta_provider),
        _GetToolDefinitionShim(meta_repo),
    ]
