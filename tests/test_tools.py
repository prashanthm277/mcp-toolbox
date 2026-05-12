"""Unit tests for MetaToolSet (get_tool_definition, execute_tool)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_toolbox import ToolDefinition, make_tools


def _make_repo(tool_defns: list[ToolDefinition] | None = None, execute_result: object = None):
    repo = MagicMock()
    if tool_defns is not None:
        repo.get_tool_definition = AsyncMock(return_value=tool_defns)
    else:
        repo.get_tool_definition = AsyncMock(side_effect=KeyError("not found"))
    repo.execute_tool = AsyncMock(return_value=execute_result)
    return repo


class TestMetaToolSet:
    def test_definitions_contains_two_tools(self):
        meta = make_tools(MagicMock())
        names = {d["name"] for d in meta.definitions}
        assert names == {"get_tool_definition", "execute_tool"}

    def test_definitions_have_required_input_schema(self):
        meta = make_tools(MagicMock())
        for defn in meta.definitions:
            assert "input_schema" in defn
            assert defn["input_schema"]["type"] == "object"
            assert "required" in defn["input_schema"]

    async def test_dispatch_get_tool_definition(self):
        defn = ToolDefinition(
            name="ui__click",
            server="ui",
            description="Click an element",
            input_schema={"type": "object"},
        )
        repo = MagicMock()
        repo.get_tool_definition = AsyncMock(return_value=[defn])
        meta = make_tools(repo)

        result = await meta.dispatch("get_tool_definition", {"tool_names": ["ui__click"]})

        assert len(result) == 1
        assert result[0]["name"] == "ui__click"
        assert result[0]["server"] == "ui"
        assert result[0]["description"] == "Click an element"

    async def test_dispatch_execute_tool(self):
        repo = _make_repo(execute_result={"status": "ok"})
        meta = make_tools(repo)

        result = await meta.dispatch(
            "execute_tool", {"tool_name": "ui__click", "arguments": {"selector": "#btn"}}
        )

        repo.execute_tool.assert_awaited_once_with("ui__click", {"selector": "#btn"})
        assert result == {"status": "ok"}

    async def test_dispatch_raises_for_unknown_meta_tool(self):
        meta = make_tools(MagicMock())
        with pytest.raises(ValueError, match="Unknown meta-tool"):
            await meta.dispatch("nonexistent", {})
