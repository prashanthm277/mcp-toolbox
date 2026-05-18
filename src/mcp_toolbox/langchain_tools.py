"""Optional LangChain integration for MetaToolRepository.

Requires the [langchain] extra:
    pip install mcp-toolbox[langchain]

Usage:
    from mcp_toolbox import MetaToolRepository
    from mcp_toolbox.langchain_tools import convert_to_langchain_tools

    meta_repo = MetaToolRepository(composite_repo)
    tools = await meta_repo.list_tools()
    lc_tools = convert_to_langchain_tools(tools, meta_repo.execute_tool)
"""

from typing import Any, Callable

try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field, create_model
except ImportError as e:
    raise ImportError(
        "LangChain integration requires the [langchain] extra. "
        "Install it with: pip install mcp-toolbox[langchain]"
    ) from e

from .models import ToolInfo


def _schema_to_pydantic(model_name: str, schema: dict) -> type[BaseModel]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, prop in props.items():
        desc = prop.get("description", "")
        if field_name in required:
            fields[field_name] = (Any, Field(description=desc))
        else:
            fields[field_name] = (Any, Field(default=None, description=desc))
    return create_model(model_name, **fields)


def _wrap_tool_info(
    tool_info: ToolInfo,
    executor: Callable,
    meta_provider: Callable[[str, dict], dict | None] | None,
) -> BaseTool:
    _name = tool_info.name
    _schema = _schema_to_pydantic(f"{tool_info.name}_schema", tool_info.input_schema)

    class _ToolShim(BaseTool):
        name: str = _name
        description: str = tool_info.description
        args_schema: type[BaseModel] = _schema
        handle_tool_error: bool = True

        def _run(self, **kwargs: Any) -> Any:
            raise NotImplementedError("Only async execution is supported")

        async def _arun(self, **kwargs: Any) -> Any:
            meta = meta_provider(_name, kwargs) if meta_provider else None
            return await executor(_name, kwargs, meta=meta)

    return _ToolShim()


def convert_to_langchain_tools(
    tools: list[ToolInfo],
    executor: Callable,
    meta_provider: Callable[[str, dict], dict | None] | None = None,
) -> list[BaseTool]:
    """Convert ToolInfo objects into LangChain BaseTool instances.

    Args:
        tools: Tool definitions to convert (e.g. from MetaToolRepository.list_tools()).
        executor: Async callable — (tool_name, args, *, meta=None) -> Any.
                  Typically, MetaToolRepository.execute_tool.
        meta_provider: Optional callback (tool_name, kwargs) -> dict | None that
            returns auth/meta context to forward on each call.
    """
    return [_wrap_tool_info(t, executor, meta_provider) for t in tools]
