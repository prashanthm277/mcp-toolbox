import asyncio
import logging
from typing import Any

import anyio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from .models import MCPServerConfig, ServerInfo, ServerStatus, ToolDefinition, ToolInfo

logger = logging.getLogger(__name__)


class ToolNotFoundError(KeyError):
    def __init__(self, tool_names: list[str] | str) -> None:
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        self.tool_names = tool_names
        names_str = ", ".join(f"'{n}'" for n in tool_names)
        super().__init__(f"Tools not found in repository: {names_str}")


class ToolRepository:
    """
    Aggregates tools from multiple MCP servers into a unified registry.

    Maintains a persistent MCP session per server so that tool call responses
    (which many servers deliver via the SSE GET stream rather than the POST body)
    are reliably received.

    Usage:
        repo = ToolRepository([
            MCPServerConfig(name="ui", url="http://localhost:5020"),
            MCPServerConfig(name="platform", url="http://localhost:5010"),
        ])
        await repo.connect()

        tools = await repo.list_all_tools()
        status = await repo.get_mcp_status()
        result = await repo.execute_tool("platform_createWebPage", {"name": "Home"})
    """

    def __init__(self, mcp_servers: list[MCPServerConfig]) -> None:
        self._configs: dict[str, MCPServerConfig] = {s.name: s for s in mcp_servers}
        self._tools: dict[str, ToolInfo] = {}
        self._server_info: dict[str, ServerInfo] = {}
        # Persistent sessions kept alive by background tasks
        self._sessions: dict[str, ClientSession] = {}
        self._bg_tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        """Connect to all configured MCP servers and populate the tool registry."""
        ready_events: dict[str, asyncio.Event] = {
            name: asyncio.Event() for name in self._configs
        }
        loop = asyncio.get_running_loop()
        for cfg in self._configs.values():
            task = loop.create_task(
                self._run_server_loop(cfg, ready_events[cfg.name]),
                name=f"mcp-{cfg.name}",
            )
            self._bg_tasks.append(task)

        # Wait for every server to be ready (or fail) with a per-server timeout.
        async with anyio.create_task_group() as tg:
            for cfg in self._configs.values():
                tg.start_soon(_wait_ready, cfg.name, ready_events[cfg.name], 30.0)

    async def list_all_tools(self) -> list[ToolInfo]:
        return list(self._tools.values())

    async def get_mcp_status(self) -> dict[str, ServerInfo]:
        return dict(self._server_info)

    async def get_tool_definition(self, tool_names: list[str]) -> list[ToolDefinition]:
        missing = [n for n in tool_names if n not in self._tools]
        if missing:
            raise ToolNotFoundError(missing)
        return [
            ToolDefinition(
                name=self._tools[n].name,
                server=self._tools[n].server,
                description=self._tools[n].description,
                input_schema=self._tools[n].input_schema,
            )
            for n in tool_names
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> Any:
        info = self._tools.get(tool_name)
        if not info:
            raise ToolNotFoundError(tool_name)
        cfg = self._configs[info.server]
        return await self._call_tool(cfg, tool_name, arguments, meta=meta)

    async def close(self) -> None:
        """Cancel all background connection tasks."""
        for task in self._bg_tasks:
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
        self._sessions.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_server_loop(self, cfg: MCPServerConfig, ready_event: asyncio.Event) -> None:
        """Keep a persistent MCP session alive; reconnect on failure."""
        while True:
            try:
                await self._run_server(cfg, ready_event)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._sessions.pop(cfg.name, None)
                if not ready_event.is_set():
                    self._server_info[cfg.name] = ServerInfo(
                        name=cfg.name,
                        status=ServerStatus.FAILED,
                        tool_count=0,
                        error=str(exc),
                    )
                    ready_event.set()
                    logger.warning(f"Failed to connect to '{cfg.name}': {exc}")
                    return  # Don't retry for initial connection failures
                logger.warning(f"Lost connection to '{cfg.name}': {exc}, reconnecting in 5s")
                await asyncio.sleep(5)

    async def _run_server(self, cfg: MCPServerConfig, ready_event: asyncio.Event) -> None:
        """Open a single persistent MCP session, list tools, then idle until cancelled."""
        headers = cfg.headers or {}
        if cfg.transport == "streamable_http":
            async with streamablehttp_client(cfg.url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, cfg, ready_event)
                    await asyncio.sleep(float("inf"))  # keep context alive

        elif cfg.transport == "sse":
            async with sse_client(cfg.url, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, cfg, ready_event)
                    await asyncio.sleep(float("inf"))

        elif cfg.transport == "stdio":
            params = StdioServerParameters(
                command=cfg.command, args=cfg.args, env=cfg.env or None
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, cfg, ready_event)
                    await asyncio.sleep(float("inf"))

        else:
            raise ValueError(f"Unsupported transport: '{cfg.transport}'")

    async def _init_session(
        self, session: ClientSession, cfg: MCPServerConfig, ready_event: asyncio.Event
    ) -> None:
        """Initialize the session, load tools, and mark the server as ready."""
        await session.initialize()
        response = await session.list_tools()
        tools = _parse_tools(response.tools, cfg)
        for tool in tools:
            self._tools[tool.name] = tool
        self._sessions[cfg.name] = session
        self._server_info[cfg.name] = ServerInfo(
            name=cfg.name,
            status=ServerStatus.CONNECTED,
            tool_count=len(tools),
        )
        logger.info(f"Connected to '{cfg.name}': {len(tools)} tools loaded")
        if not ready_event.is_set():
            ready_event.set()

    async def _call_tool(
        self,
        cfg: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> Any:
        session = self._sessions.get(cfg.name)
        if session is None:
            raise ConnectionError(f"No active session for server '{cfg.name}'")
        mcp_name = _strip_prefix(cfg, tool_name)
        try:
            with anyio.fail_after(cfg.tool_call_timeout):
                return await session.call_tool(mcp_name, arguments, meta=meta)
        except TimeoutError:
            logger.error(
                "Tool call '%s' on server '%s' timed out after %.0fs",
                mcp_name, cfg.name, cfg.tool_call_timeout,
            )
            # Evict the session — it may be in a broken state after the cancelled call.
            # Subsequent calls will get ConnectionError immediately while _run_server_loop reconnects.
            self._sessions.pop(cfg.name, None)
            raise asyncio.TimeoutError(
                f"Tool call '{mcp_name}' on server '{cfg.name}' timed out after {cfg.tool_call_timeout:.0f}s"
            )


async def _wait_ready(name: str, event: asyncio.Event, timeout: float) -> None:
    """Wait for a server's ready event with a timeout."""
    with anyio.move_on_after(timeout):
        await event.wait()


def _parse_tools(raw_tools: list, cfg: MCPServerConfig) -> list[ToolInfo]:
    prefix = f"{cfg.name}_" if cfg.tool_name_prefix else ""
    return [
        ToolInfo(
            name=f"{prefix}{t.name}",
            server=cfg.name,
            description=t.description or "",
            input_schema=_to_dict(t.inputSchema),
        )
        for t in raw_tools
    ]


def _strip_prefix(cfg: MCPServerConfig, tool_name: str) -> str:
    """Return the raw MCP tool name by removing the server prefix added by _parse_tools."""
    if cfg.tool_name_prefix:
        prefix = f"{cfg.name}_"
        if tool_name.startswith(prefix):
            return tool_name[len(prefix):]
    return tool_name


def _to_dict(schema: Any) -> dict:
    if not schema:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    return {}
