"""Unit tests for MetaToolRepository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_toolbox import ToolInfo, ToolNotFoundError
from mcp_toolbox.tools import MetaToolRepository


def _make_inner_repo(tools: list[ToolInfo] | None = None, execute_result: object = None):
    repo = MagicMock()
    repo.list_tools = AsyncMock(return_value=tools or [])
    repo.execute_tool = AsyncMock(return_value=execute_result)
    repo.connect = AsyncMock()
    repo.close = AsyncMock()
    return repo


class TestMetaToolRepository:
    async def test_list_tools_returns_two_meta_tools(self):
        meta = MetaToolRepository(_make_inner_repo())
        tools = await meta.list_tools()
        assert {tool.name for tool in tools} == {"get_tool_definition", "execute_tool"}

    async def test_meta_tools_have_valid_input_schema(self):
        meta = MetaToolRepository(_make_inner_repo())
        for tool in await meta.list_tools():
            assert tool.input_schema["type"] == "object"
            assert "required" in tool.input_schema

    async def test_get_tool_definition_returns_matching_tool(self):
        available_tools = [
            ToolInfo(name="createPage", server="platform", description="Create a page",
                     input_schema={"type": "object"}),
        ]
        meta = MetaToolRepository(_make_inner_repo(tools=available_tools))

        result = await meta.execute_tool("get_tool_definition", {"tool_names": ["createPage"]})

        assert len(result) == 1
        assert result[0]["name"] == "createPage"
        assert result[0]["server"] == "platform"
        assert result[0]["description"] == "Create a page"

    async def test_get_tool_definition_raises_for_unknown_tool(self):
        meta = MetaToolRepository(_make_inner_repo(tools=[]))
        with pytest.raises(ToolNotFoundError):
            await meta.execute_tool("get_tool_definition", {"tool_names": ["nonexistentTool"]})

    async def test_execute_tool_dispatches_to_inner_repo(self):
        inner_repo = _make_inner_repo(execute_result={"status": "ok"})
        meta = MetaToolRepository(inner_repo)

        result = await meta.execute_tool(
            "execute_tool", {"tool_name": "createPage", "tool_args": {"name": "Home"}}
        )

        inner_repo.execute_tool.assert_awaited_once_with("createPage", {"name": "Home"}, meta=None)
        assert result == {"status": "ok"}

    async def test_execute_tool_forwards_request_context(self):
        inner_repo = _make_inner_repo(execute_result="done")
        meta = MetaToolRepository(inner_repo)
        request_context = {"auth_cookie": "abc", "projectId": "p1"}

        await meta.execute_tool(
            "execute_tool", {"tool_name": "createPage", "tool_args": {}}, meta=request_context
        )

        inner_repo.execute_tool.assert_awaited_once_with("createPage", {}, meta=request_context)

    async def test_execute_tool_raises_for_unknown_meta_tool(self):
        meta = MetaToolRepository(_make_inner_repo())
        with pytest.raises(ToolNotFoundError):
            await meta.execute_tool("unknownMetaTool", {})

    async def test_connect_delegates_to_inner_repo(self):
        inner_repo = _make_inner_repo()
        meta = MetaToolRepository(inner_repo)
        await meta.connect()
        inner_repo.connect.assert_awaited_once()

    async def test_close_delegates_to_inner_repo(self):
        inner_repo = _make_inner_repo()
        meta = MetaToolRepository(inner_repo)
        await meta.close()
        inner_repo.close.assert_awaited_once()
