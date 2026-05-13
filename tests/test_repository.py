"""Unit tests for MCPServerToolRepository, CompositeToolRepository, and MetaToolRepository."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_toolbox import (
    CompositeToolRepository,
    MCPServerConfig,
    MCPServerToolRepository,
    ServerStatus,
    ToolInfo,
    ToolNotFoundError,
)
from mcp_toolbox.models import ServerInfo
from mcp_toolbox.tools import MetaToolRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_mcp_tool(name: str, description: str = "", schema: dict | None = None):
    mock_tool = MagicMock()
    mock_tool.name = name
    mock_tool.description = description
    mock_tool.inputSchema = MagicMock()
    mock_tool.inputSchema.model_dump.return_value = schema or {}
    return mock_tool


def _make_mock_session(mcp_tools: list) -> AsyncMock:
    session = AsyncMock()
    list_response = MagicMock()
    list_response.tools = mcp_tools
    session.list_tools.return_value = list_response
    return session


def _make_connected_server_repo(
    config: MCPServerConfig,
    tool_infos: list[ToolInfo],
    session: Any | None = None,
) -> MCPServerToolRepository:
    """Build an MCPServerToolRepository with pre-populated state (no network calls)."""
    repo = MCPServerToolRepository(config)
    for tool in tool_infos:
        repo._tools[tool.name] = tool
    if session is not None:
        repo._session = session
        repo._server_info = ServerInfo(
            name=config.name,
            status=ServerStatus.CONNECTED,
            tool_count=len(tool_infos),
        )
    else:
        repo._server_info = ServerInfo(
            name=config.name,
            status=ServerStatus.FAILED,
            tool_count=0,
            error="Connection refused",
        )
    return repo


@pytest.fixture
def platform_config():
    return MCPServerConfig(name="platform", url="http://localhost:5010", transport="streamable_http")


@pytest.fixture
def two_server_configs():
    return [
        MCPServerConfig(name="platform", url="http://localhost:5010", transport="streamable_http"),
        MCPServerConfig(name="ui", url="http://localhost:5020", transport="streamable_http"),
    ]


# ---------------------------------------------------------------------------
# MCPServerToolRepository.connect()
# ---------------------------------------------------------------------------

class TestConnect:
    async def test_connect_populates_tools_on_success(self, platform_config):
        mock_session = _make_mock_session([
            _make_mock_mcp_tool("createPage", "Create a page"),
            _make_mock_mcp_tool("deletePage", "Delete a page"),
        ])

        async def fake_server_loop(self_repo, ready_event):
            from mcp_toolbox.repository import _schema_to_dict
            tools_response = await mock_session.list_tools()
            for mcp_tool in tools_response.tools:
                self_repo._tools[mcp_tool.name] = ToolInfo(
                    name=mcp_tool.name,
                    server=self_repo._config.name,
                    description=mcp_tool.description or "",
                    input_schema=_schema_to_dict(mcp_tool.inputSchema),
                )
            self_repo._session = mock_session
            self_repo._server_info = ServerInfo(
                name=self_repo._config.name, status=ServerStatus.CONNECTED,
                tool_count=len(self_repo._tools)
            )
            ready_event.set()

        with patch.object(MCPServerToolRepository, "_run_server_loop", fake_server_loop):
            repo = MCPServerToolRepository(platform_config)
            await repo.connect()

        tools = await repo.list_tools()
        assert len(tools) == 2
        assert {tool.name for tool in tools} == {"createPage", "deletePage"}

    async def test_connect_marks_server_failed_on_error(self, platform_config):
        async def failing_server_loop(self_repo, ready_event):
            self_repo._server_info = ServerInfo(
                name=self_repo._config.name, status=ServerStatus.FAILED,
                tool_count=0, error="Connection refused"
            )
            ready_event.set()

        with patch.object(MCPServerToolRepository, "_run_server_loop", failing_server_loop):
            repo = MCPServerToolRepository(platform_config)
            await repo.connect()

        assert repo.server_info.status == ServerStatus.FAILED
        assert "Connection refused" in repo.server_info.error


# ---------------------------------------------------------------------------
# MCPServerToolRepository.list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    async def test_returns_all_tools_for_server(self, platform_config):
        tool_infos = [
            ToolInfo(name="createPage", server="platform", description="", input_schema={}),
            ToolInfo(name="deletePage", server="platform", description="", input_schema={}),
        ]
        repo = _make_connected_server_repo(platform_config, tool_infos, AsyncMock())
        tools = await repo.list_tools()
        assert len(tools) == 2
        assert {tool.name for tool in tools} == {"createPage", "deletePage"}

    async def test_returns_tools_without_server_prefix(self, platform_config):
        tool_infos = [ToolInfo(name="createPage", server="platform", description="", input_schema={})]
        repo = _make_connected_server_repo(platform_config, tool_infos, AsyncMock())
        tools = await repo.list_tools()
        assert tools[0].name == "createPage"

    async def test_returns_empty_list_when_no_tools(self, platform_config):
        repo = _make_connected_server_repo(platform_config, [], AsyncMock())
        assert await repo.list_tools() == []


# ---------------------------------------------------------------------------
# MCPServerToolRepository.execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    async def test_calls_mcp_session_with_tool_name_and_args(self, platform_config):
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = {"output": "ok"}

        tool_info = ToolInfo(name="createPage", server="platform", description="", input_schema={})
        repo = _make_connected_server_repo(platform_config, [tool_info], mock_session)

        result = await repo.execute_tool("createPage", {"name": "Home"})

        mock_session.call_tool.assert_awaited_once_with("createPage", {"name": "Home"}, meta=None)
        assert result == {"output": "ok"}

    async def test_forwards_request_context_to_mcp_session(self, platform_config):
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = {}

        tool_info = ToolInfo(name="isAuthenticationEnabled", server="platform", description="", input_schema={})
        repo = _make_connected_server_repo(platform_config, [tool_info], mock_session)
        request_context = {"auth_cookie": "abc", "projectId": "p1"}

        await repo.execute_tool("isAuthenticationEnabled", {}, meta=request_context)

        mock_session.call_tool.assert_awaited_once_with("isAuthenticationEnabled", {}, meta=request_context)

    async def test_raises_tool_not_found_for_unknown_tool(self, platform_config):
        repo = _make_connected_server_repo(platform_config, [], AsyncMock())
        with pytest.raises(ToolNotFoundError):
            await repo.execute_tool("nonexistentTool", {})

    async def test_raises_connection_error_when_session_is_missing(self, platform_config):
        tool_info = ToolInfo(name="createPage", server="platform", description="", input_schema={})
        repo = _make_connected_server_repo(platform_config, [tool_info], session=None)
        with pytest.raises(ConnectionError, match="No active session"):
            await repo.execute_tool("createPage", {})


# ---------------------------------------------------------------------------
# CompositeToolRepository
# ---------------------------------------------------------------------------

class TestCompositeToolRepository:
    async def test_list_tools_prefixes_names_with_server_key(self, two_server_configs):
        platform_repo = _make_connected_server_repo(
            two_server_configs[0],
            [ToolInfo(name="createPage", server="platform", description="", input_schema={})],
            AsyncMock(),
        )
        ui_repo = _make_connected_server_repo(
            two_server_configs[1],
            [ToolInfo(name="clickButton", server="ui", description="", input_schema={})],
            AsyncMock(),
        )
        composite = CompositeToolRepository({"platform": platform_repo, "ui": ui_repo})
        tools = await composite.list_tools()

        assert len(tools) == 2
        assert {tool.name for tool in tools} == {"platform_createPage", "ui_clickButton"}

    async def test_execute_tool_strips_prefix_and_routes_to_correct_repo(self, two_server_configs):
        platform_session = AsyncMock()
        platform_session.call_tool.return_value = "created"

        platform_repo = _make_connected_server_repo(
            two_server_configs[0],
            [ToolInfo(name="createPage", server="platform", description="", input_schema={})],
            platform_session,
        )
        ui_repo = _make_connected_server_repo(
            two_server_configs[1],
            [ToolInfo(name="clickButton", server="ui", description="", input_schema={})],
            AsyncMock(),
        )
        composite = CompositeToolRepository({"platform": platform_repo, "ui": ui_repo})

        await composite.execute_tool("platform_createPage", {"name": "Home"})

        platform_session.call_tool.assert_awaited_once_with("createPage", {"name": "Home"}, meta=None)

    async def test_execute_tool_raises_for_unknown_prefixed_tool(self, two_server_configs):
        platform_repo = _make_connected_server_repo(two_server_configs[0], [], AsyncMock())
        composite = CompositeToolRepository({"platform": platform_repo})

        with pytest.raises(ToolNotFoundError):
            await composite.execute_tool("ui_unknownTool", {})

    async def test_list_tools_aggregates_all_tools(self, two_server_configs):
        platform_repo = _make_connected_server_repo(
            two_server_configs[0],
            [
                ToolInfo(name="createPage", server="platform", description="", input_schema={}),
                ToolInfo(name="deletePage", server="platform", description="", input_schema={}),
            ],
            AsyncMock(),
        )
        ui_repo = _make_connected_server_repo(
            two_server_configs[1],
            [ToolInfo(name="clickButton", server="ui", description="", input_schema={})],
            AsyncMock(),
        )
        composite = CompositeToolRepository({"platform": platform_repo, "ui": ui_repo})
        tools = await composite.list_tools()
        assert len(tools) == 3


# ---------------------------------------------------------------------------
# MetaToolRepository
# ---------------------------------------------------------------------------

class TestMetaToolRepository:
    async def test_list_tools_exposes_only_two_meta_tools(self, platform_config):
        inner_repo = _make_connected_server_repo(
            platform_config,
            [ToolInfo(name="createPage", server="platform", description="", input_schema={})],
            AsyncMock(),
        )
        meta = MetaToolRepository(inner_repo)
        tools = await meta.list_tools()
        assert len(tools) == 2
        assert {tool.name for tool in tools} == {"get_tool_definition", "execute_tool"}

    async def test_get_tool_definition_returns_tool_schema(self, platform_config):
        inner_repo = _make_connected_server_repo(
            platform_config,
            [ToolInfo(name="createPage", server="platform", description="Creates a page",
                      input_schema={"type": "object"})],
            AsyncMock(),
        )
        meta = MetaToolRepository(inner_repo)
        result = await meta.execute_tool("get_tool_definition", {"tool_names": ["createPage"]})
        assert len(result) == 1
        assert result[0]["name"] == "createPage"
        assert result[0]["description"] == "Creates a page"

    async def test_get_tool_definition_raises_for_unknown_tool(self, platform_config):
        inner_repo = _make_connected_server_repo(platform_config, [], AsyncMock())
        meta = MetaToolRepository(inner_repo)
        with pytest.raises(ToolNotFoundError):
            await meta.execute_tool("get_tool_definition", {"tool_names": ["nonexistentTool"]})

    async def test_execute_tool_dispatches_to_inner_repo(self, platform_config):
        platform_session = AsyncMock()
        platform_session.call_tool.return_value = "page created"

        inner_repo = _make_connected_server_repo(
            platform_config,
            [ToolInfo(name="createPage", server="platform", description="", input_schema={})],
            platform_session,
        )
        meta = MetaToolRepository(inner_repo)
        await meta.execute_tool("execute_tool", {"tool_name": "createPage", "tool_args": {"name": "Home"}})

        platform_session.call_tool.assert_awaited_once_with("createPage", {"name": "Home"}, meta=None)

    async def test_execute_tool_raises_for_unknown_meta_tool(self, platform_config):
        inner_repo = _make_connected_server_repo(platform_config, [], AsyncMock())
        meta = MetaToolRepository(inner_repo)
        with pytest.raises(ToolNotFoundError):
            await meta.execute_tool("unknownMetaTool", {})


# ---------------------------------------------------------------------------
# MCPServerConfig headers
# ---------------------------------------------------------------------------

class TestMCPServerConfigHeaders:
    def test_headers_stored_on_config(self):
        config = MCPServerConfig(
            name="platform",
            url="http://localhost:5010/mcp",
            transport="streamable_http",
            headers={"Authorization": "Bearer token123"},
        )
        assert config.headers == {"Authorization": "Bearer token123"}

    def test_default_headers_is_empty_dict(self):
        config = MCPServerConfig(name="platform", url="http://localhost:5010/mcp", transport="streamable_http")
        assert config.headers == {}

    async def test_headers_are_forwarded_to_transport_on_connect(self):
        captured: dict = {}

        async def fake_server_loop(self_repo, ready_event):
            captured["headers"] = self_repo._config.headers
            self_repo._server_info = ServerInfo(
                name=self_repo._config.name, status=ServerStatus.CONNECTED, tool_count=0
            )
            ready_event.set()

        config = MCPServerConfig(
            name="platform",
            url="http://localhost:5010/mcp",
            transport="streamable_http",
            headers={"Authorization": "Bearer token123"},
        )
        with patch.object(MCPServerToolRepository, "_run_server_loop", fake_server_loop):
            repo = MCPServerToolRepository(config)
            await repo.connect()

        assert captured["headers"] == {"Authorization": "Bearer token123"}
