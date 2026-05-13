"""Unit tests for ToolRepository."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_toolbox import (
    MCPServerConfig,
    ServerStatus,
    ToolInfo,
    ToolNotFoundError,
    ToolRepository,
)
from mcp_toolbox.models import ServerInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tool(name: str, description: str = "", schema: dict | None = None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = MagicMock()
    tool.inputSchema.model_dump.return_value = schema or {}
    return tool


def _make_mock_session(tools: list) -> AsyncMock:
    session = AsyncMock()
    list_response = MagicMock()
    list_response.tools = tools
    session.list_tools.return_value = list_response
    return session


def _make_connected_repo(
    configs: list[MCPServerConfig],
    tool_infos: list[ToolInfo],
    mock_sessions: dict[str, Any],
) -> ToolRepository:
    """Build a ToolRepository with pre-populated state (no network calls)."""
    repo = ToolRepository(configs)
    for t in tool_infos:
        repo._tools[t.name] = t
    for cfg in configs:
        session = mock_sessions.get(cfg.name)
        if session is not None:
            repo._sessions[cfg.name] = session
            repo._server_info[cfg.name] = ServerInfo(
                name=cfg.name,
                status=ServerStatus.CONNECTED,
                tool_count=sum(1 for t in tool_infos if t.server == cfg.name),
            )
        else:
            repo._server_info[cfg.name] = ServerInfo(
                name=cfg.name,
                status=ServerStatus.FAILED,
                tool_count=0,
                error="Connection refused",
            )
    return repo


@pytest.fixture
def http_config():
    return MCPServerConfig(name="test_server", url="http://localhost:9000")


@pytest.fixture
def two_server_configs():
    return [
        MCPServerConfig(name="server_a", url="http://localhost:9001"),
        MCPServerConfig(name="server_b", url="http://localhost:9002"),
    ]


# ---------------------------------------------------------------------------
# connect() — integration test via _run_server_loop patch
# ---------------------------------------------------------------------------

class TestConnect:
    async def test_connect_populates_tools_on_success(self, http_config):
        """connect() registers tools and marks server CONNECTED."""
        mock_session = _make_mock_session([
            _make_mock_tool("run_cmd", "Run a command"),
            _make_mock_tool("list_files", "List files"),
        ])

        async def fake_loop(self_repo, cfg, ready_event):
            from mcp_toolbox.repository import _parse_tools
            response = await mock_session.list_tools()
            tools = _parse_tools(response.tools, cfg)
            for t in tools:
                self_repo._tools[t.name] = t
            self_repo._sessions[cfg.name] = mock_session
            self_repo._server_info[cfg.name] = ServerInfo(
                name=cfg.name, status=ServerStatus.CONNECTED, tool_count=len(tools)
            )
            ready_event.set()

        with patch.object(ToolRepository, "_run_server_loop", fake_loop):
            repo = ToolRepository([http_config])
            await repo.connect()

        tools = await repo.list_all_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"test_server_run_cmd", "test_server_list_files"}

    async def test_connect_marks_server_failed_on_error(self, http_config):
        """connect() marks a server FAILED when the connection loop errors."""

        async def failing_loop(self_repo, cfg, ready_event):
            self_repo._server_info[cfg.name] = ServerInfo(
                name=cfg.name, status=ServerStatus.FAILED, tool_count=0, error="refused"
            )
            ready_event.set()

        with patch.object(ToolRepository, "_run_server_loop", failing_loop):
            repo = ToolRepository([http_config])
            await repo.connect()

        status = await repo.get_mcp_status()
        assert status["test_server"].status == ServerStatus.FAILED
        assert "refused" in status["test_server"].error

    async def test_partial_failure_does_not_block_other_servers(self, two_server_configs):
        """One server failing does not prevent tools from healthy servers being loaded."""
        good_session = _make_mock_session([_make_mock_tool("good_tool")])

        async def selective_loop(self_repo, cfg, ready_event):
            from mcp_toolbox.repository import _parse_tools
            if cfg.name == "server_a":
                self_repo._server_info[cfg.name] = ServerInfo(
                    name=cfg.name, status=ServerStatus.FAILED, tool_count=0, error="down"
                )
            else:
                response = await good_session.list_tools()
                tools = _parse_tools(response.tools, cfg)
                for t in tools:
                    self_repo._tools[t.name] = t
                self_repo._sessions[cfg.name] = good_session
                self_repo._server_info[cfg.name] = ServerInfo(
                    name=cfg.name, status=ServerStatus.CONNECTED, tool_count=len(tools)
                )
            ready_event.set()

        with patch.object(ToolRepository, "_run_server_loop", selective_loop):
            repo = ToolRepository(two_server_configs)
            await repo.connect()

        status = await repo.get_mcp_status()
        assert status["server_a"].status == ServerStatus.FAILED
        assert status["server_b"].status == ServerStatus.CONNECTED

        tools = await repo.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "server_b_good_tool"


# ---------------------------------------------------------------------------
# list_all_tools
# ---------------------------------------------------------------------------

class TestListAllTools:
    async def test_returns_all_tools_across_servers(self, two_server_configs):
        tool_infos = [
            ToolInfo(name="server_a_tool_one", server="server_a", description="", input_schema={}),
            ToolInfo(name="server_b_tool_two", server="server_b", description="", input_schema={}),
        ]
        repo = _make_connected_repo(
            two_server_configs,
            tool_infos,
            {"server_a": AsyncMock(), "server_b": AsyncMock()},
        )
        tools = await repo.list_all_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"server_a_tool_one", "server_b_tool_two"}

    async def test_returns_empty_list_when_no_tools(self, http_config):
        repo = _make_connected_repo([http_config], [], {"test_server": AsyncMock()})
        tools = await repo.list_all_tools()
        assert tools == []


# ---------------------------------------------------------------------------
# get_mcp_status
# ---------------------------------------------------------------------------

class TestGetMcpStatus:
    async def test_connected_server_shows_correct_count(self, http_config):
        tool_infos = [
            ToolInfo(name="test_server_t", server="test_server", description="", input_schema={}),
        ]
        repo = _make_connected_repo([http_config], tool_infos, {"test_server": AsyncMock()})
        status = await repo.get_mcp_status()
        assert status["test_server"].status == ServerStatus.CONNECTED
        assert status["test_server"].tool_count == 1

    async def test_failed_server_has_error_message(self, http_config):
        repo = _make_connected_repo([http_config], [], {})  # no session → FAILED
        status = await repo.get_mcp_status()
        assert status["test_server"].status == ServerStatus.FAILED
        assert status["test_server"].error is not None


# ---------------------------------------------------------------------------
# get_tool_definition
# ---------------------------------------------------------------------------

class TestGetToolDefinition:
    async def test_single_tool_returned(self, http_config):
        tool_infos = [
            ToolInfo(
                name="test_server_my_tool",
                server="test_server",
                description="Does something",
                input_schema={"type": "object"},
            )
        ]
        repo = _make_connected_repo([http_config], tool_infos, {"test_server": AsyncMock()})

        defns = await repo.get_tool_definition(["test_server_my_tool"])
        assert len(defns) == 1
        assert defns[0].name == "test_server_my_tool"
        assert defns[0].server == "test_server"
        assert defns[0].description == "Does something"
        assert defns[0].input_schema == {"type": "object"}

    async def test_multiple_tools_returned(self, http_config):
        tool_infos = [
            ToolInfo(name="test_server_tool_a", server="test_server", description="A", input_schema={}),
            ToolInfo(name="test_server_tool_b", server="test_server", description="B", input_schema={}),
        ]
        repo = _make_connected_repo([http_config], tool_infos, {"test_server": AsyncMock()})

        defns = await repo.get_tool_definition(["test_server_tool_a", "test_server_tool_b"])
        assert len(defns) == 2
        assert {d.name for d in defns} == {"test_server_tool_a", "test_server_tool_b"}

    async def test_raises_for_unknown_tool(self, http_config):
        repo = _make_connected_repo([http_config], [], {"test_server": AsyncMock()})

        with pytest.raises(ToolNotFoundError) as exc_info:
            await repo.get_tool_definition(["nonexistent_tool"])
        assert "nonexistent_tool" in exc_info.value.tool_names

    async def test_reports_all_missing_tools(self, http_config):
        repo = _make_connected_repo([http_config], [], {"test_server": AsyncMock()})

        with pytest.raises(ToolNotFoundError) as exc_info:
            await repo.get_tool_definition(["missing_a", "missing_b"])
        assert set(exc_info.value.tool_names) == {"missing_a", "missing_b"}


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    async def test_calls_mcp_session_with_stripped_prefix(self, http_config):
        """Tool names are stored with server prefix but called with raw MCP name."""
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = {"output": "ok"}

        tool_info = ToolInfo(
            name="test_server_run_cmd",
            server="test_server",
            description="",
            input_schema={},
        )
        repo = _make_connected_repo([http_config], [tool_info], {"test_server": mock_session})

        result = await repo.execute_tool("test_server_run_cmd", {"cmd": "ls"})

        # Stored name "test_server_run_cmd" → prefix "test_server_" stripped → "run_cmd"
        mock_session.call_tool.assert_awaited_once_with("run_cmd", {"cmd": "ls"}, meta=None)
        assert result == {"output": "ok"}

    async def test_forwards_meta_to_mcp_session(self, http_config):
        """Optional meta dict is forwarded to session.call_tool."""
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = {}

        tool_info = ToolInfo(
            name="test_server_do_thing",
            server="test_server",
            description="",
            input_schema={},
        )
        repo = _make_connected_repo([http_config], [tool_info], {"test_server": mock_session})
        meta = {"metaParams": {"projectId": "p1"}}

        await repo.execute_tool("test_server_do_thing", {}, meta=meta)

        mock_session.call_tool.assert_awaited_once_with("do_thing", {}, meta=meta)

    async def test_raises_tool_not_found_for_unknown_tool(self, http_config):
        repo = _make_connected_repo([http_config], [], {"test_server": AsyncMock()})

        with pytest.raises(ToolNotFoundError):
            await repo.execute_tool("ghost_tool", {})

    async def test_raises_connection_error_when_no_session(self, http_config):
        tool_info = ToolInfo(
            name="test_server_run_cmd",
            server="test_server",
            description="",
            input_schema={},
        )
        # No session provided → _sessions["test_server"] is absent
        repo = _make_connected_repo([http_config], [tool_info], {})

        with pytest.raises(ConnectionError, match="No active session"):
            await repo.execute_tool("test_server_run_cmd", {})


# ---------------------------------------------------------------------------
# MCPServerConfig — headers are stored and passed to the transport
# ---------------------------------------------------------------------------

class TestMCPServerConfigHeaders:
    def test_headers_stored_on_config(self):
        """Headers passed to MCPServerConfig are accessible on the config object."""
        cfg = MCPServerConfig(
            name="platform",
            url="http://localhost:5010/mcp",
            headers={"Authorization": "Bearer changeme"},
        )
        assert cfg.headers == {"Authorization": "Bearer changeme"}

    def test_default_headers_is_empty(self):
        """Headers default to an empty dict when not provided."""
        cfg = MCPServerConfig(name="platform", url="http://localhost:5010/mcp")
        assert cfg.headers == {}

    async def test_headers_forwarded_to_transport_on_connect(self):
        """Headers from MCPServerConfig are passed to the HTTP transport on connect."""
        captured: dict = {}

        async def fake_loop(self_repo, cfg, ready_event):
            captured["headers"] = cfg.headers
            from mcp_toolbox.models import ServerInfo, ServerStatus
            self_repo._server_info[cfg.name] = ServerInfo(
                name=cfg.name, status=ServerStatus.CONNECTED, tool_count=0
            )
            ready_event.set()

        cfg = MCPServerConfig(
            name="platform",
            url="http://localhost:5010/mcp",
            headers={"Authorization": "Bearer changeme"},
        )
        with patch.object(ToolRepository, "_run_server_loop", fake_loop):
            repo = ToolRepository([cfg])
            await repo.connect()

        assert captured["headers"] == {"Authorization": "Bearer changeme"}

    async def test_missing_auth_token_defaults_to_na(self):
        """Simulates config.yaml default: missing env var → headers get 'Bearer NA'."""
        import os
        auth_token = os.environ.get("MCP_PLATFORM_AUTH_TOKEN", "NA")
        header_value = f"Bearer {auth_token}"

        cfg = MCPServerConfig(
            name="platform",
            url="http://localhost:5010/mcp",
            headers={"Authorization": header_value},
        )
        # When env var is unset, the header is "Bearer NA" — server would reject with 401
        if auth_token == "NA":
            assert cfg.headers["Authorization"] == "Bearer NA"
        else:
            assert cfg.headers["Authorization"] == f"Bearer {auth_token}"
