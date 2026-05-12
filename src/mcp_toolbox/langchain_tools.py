"""LangChain BaseTool shims: execute_tool and get_tool_definition.

These tools let LLM agents discover and call any tool registered in a
ToolRepository by name, without needing individual BaseTool shims for
every MCP tool.

Usage:
    tools = make_langchain_tools(repo, meta_provider=my_meta_fn)
    # register `tools` as internal tools for your LangChain/LangGraph agents

The optional `meta_provider` callback is invoked with the tool_name before
each execute_tool call. It should return a meta dict (e.g.
``{"metaParams": {...}}``) or None. Use it to forward auth context to
trusted MCP servers without coupling this library to any specific framework.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool, ToolException
from mcp.types import CallToolResult
from pydantic import BaseModel, Field, PrivateAttr

from .repository import ToolNotFoundError, ToolRepository

logger = logging.getLogger(__name__)


class _ExecuteToolArgs(BaseModel):
    tool_name: str = Field(description="Name of the tool to invoke.")
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments matching the tool's input schema.",
    )


class _GetToolDefinitionArgs(BaseModel):
    tool_names: list[str] = Field(
        description="One or more exact tool names to look up.",
        min_length=1,
    )


class ExecuteToolShim(BaseTool):
    """LangChain BaseTool that executes any named tool via ToolRepository."""

    name: str = "execute_tool"
    description: str = (
        "Execute a tool by its tool name. "
        "Use get_tool_definition to discover the required args before calling this."
    )
    args_schema: type[BaseModel] = _ExecuteToolArgs
    handle_tool_error: bool = True

    _repo: ToolRepository = PrivateAttr()
    _meta_provider: Optional[Callable[[str], dict | None]] = PrivateAttr(default=None)

    def __init__(
        self,
        repo: ToolRepository,
        meta_provider: Optional[Callable[[str], dict | None]] = None,
    ) -> None:
        super().__init__()
        self._repo = repo
        self._meta_provider = meta_provider

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Only async execution is supported")

    async def _arun(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if not tool_name or not tool_name.strip():
            raise ToolException("tool_name must be a non-empty string")
        tool_name = tool_name.strip()
        args = tool_args or {}
        meta = self._meta_provider(tool_name) if self._meta_provider else None
        try:
            result = await self._repo.execute_tool(tool_name, args, meta=meta)
            return _result_to_str(result)
        except ToolNotFoundError:
            raise ToolException(f"Tool '{tool_name}' not found.")
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise ToolException(
                f"Tool '{tool_name}' timed out — the MCP server is not responding."
            ) from exc
        except Exception as exc:
            logger.error("execute_tool '%s' failed: %s", tool_name, exc, exc_info=True)
            raise ToolException(str(exc) or f"Tool '{tool_name}' failed unexpectedly.") from exc


class GetToolDefinitionShim(BaseTool):
    """LangChain BaseTool that returns schema/definition of named tools."""

    name: str = "get_tool_definition"
    description: str = (
        "Fetch the definition of one or more tools by their tool names. "
        "Returns the server, description, and full input schema so you know "
        "what args to pass to execute_tool."
    )
    args_schema: type[BaseModel] = _GetToolDefinitionArgs
    handle_tool_error: bool = True

    _repo: ToolRepository = PrivateAttr()

    def __init__(self, repo: ToolRepository) -> None:
        super().__init__()
        self._repo = repo

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Only async execution is supported")

    async def _arun(self, tool_names: list[str], **kwargs: Any) -> str:
        if not tool_names:
            raise ToolException("tool_names must be a non-empty list")
        tool_names = [n.strip() for n in tool_names if n and n.strip()]
        if not tool_names:
            raise ToolException("tool_names: all entries were empty strings")
        try:
            definitions = await self._repo.get_tool_definition(tool_names)
        except ToolNotFoundError as exc:
            raise ToolException(str(exc))
        results = [
            {
                "tool_name": d.name,
                "server": d.server,
                "description": d.description,
                "input_schema": d.input_schema,
            }
            for d in definitions
        ]
        return json.dumps({"tools": results}, indent=2)


def make_langchain_tools(
    repo: ToolRepository,
    meta_provider: Optional[Callable[[str], dict | None]] = None,
) -> list[BaseTool]:
    """Return LangChain BaseTool instances for execute_tool and get_tool_definition.

    Args:
        repo: Connected ToolRepository instance.
        meta_provider: Optional callable ``(tool_name: str) -> dict | None``.
            Called before each execute_tool invocation to supply the ``meta=``
            parameter forwarded to the MCP server. Return ``None`` to skip.

    Returns:
        ``[ExecuteToolShim, GetToolDefinitionShim]`` ready to register as
        internal tools for any LangChain/LangGraph agent.
    """
    return [
        ExecuteToolShim(repo=repo, meta_provider=meta_provider),
        GetToolDefinitionShim(repo=repo),
    ]


def _result_to_str(result: Any) -> str:
    """Convert an MCP CallToolResult to a plain string for LangChain."""
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
