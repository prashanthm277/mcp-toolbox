import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .models import MCPServerConfig, ServerInfo, ServerStatus, ToolInfo

logger = logging.getLogger(__name__)


class ToolNotFoundError(KeyError):
    def __init__(self, tool_names: list[str] | str) -> None:
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        self.tool_names = tool_names
        super().__init__(f"Tools not found: {', '.join(repr(n) for n in tool_names)}")


class ToolRepository(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def list_tools(self) -> list[ToolInfo]: ...

    @abstractmethod
    async def execute_tool(self, tool_name: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> Any: ...

    @abstractmethod
    async def close(self) -> None: ...


class MCPServerToolRepository(ToolRepository):
    """Manages a persistent MCP session for a single server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._tools: dict[str, ToolInfo] = {}
        self._session: ClientSession | None = None
        self._server_info: ServerInfo | None = None
        self._server_task: asyncio.Task | None = None
        self._is_closing: bool = False

    @property
    def server_info(self) -> ServerInfo | None:
        return self._server_info

    async def connect(self) -> None:
        logger.info("Connecting to MCP server '%s'", self._config.name)
        ready_event = asyncio.Event()
        self._server_task = asyncio.get_running_loop().create_task(
            self._run_server_loop(ready_event),
            name=f"mcp-{self._config.name}",
        )
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_wait_for_ready, self._config.name, ready_event, 30.0)
        if self._server_info is None:
            self._server_info = ServerInfo(
                name=self._config.name,
                status=ServerStatus.FAILED,
                tool_count=0,
                error="Connection timed out after 30s",
            )

    async def list_tools(self) -> list[ToolInfo]:
        return list(self._tools.values())

    async def execute_tool(self, tool_name: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)
        if self._session is None:
            raise ConnectionError(f"No active session for server '{self._config.name}'")
        try:
            with anyio.fail_after(self._config.tool_call_timeout):
                return await self._session.call_tool(tool_name, args, meta=meta)
        except TimeoutError:
            logger.error("Tool '%s' timed out after %.0fs", tool_name, self._config.tool_call_timeout)
            self._session = None
            if not self._is_closing:
                self._schedule_reconnect()
            raise asyncio.TimeoutError(
                f"Tool '{tool_name}' timed out after {self._config.tool_call_timeout:.0f}s"
            )

    async def close(self) -> None:
        self._is_closing = True
        if self._server_task:
            self._server_task.cancel()
            await asyncio.gather(self._server_task, return_exceptions=True)
        self._server_task = None
        self._session = None

    async def _run_server_loop(self, ready_event: asyncio.Event) -> None:
        while True:
            try:
                await self._run_server(ready_event)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._session = None
                if not ready_event.is_set():
                    self._server_info = ServerInfo(
                        name=self._config.name, status=ServerStatus.FAILED, tool_count=0, error=str(exc)
                    )
                    ready_event.set()
                    logger.warning("Failed to connect to '%s': %s", self._config.name, exc)
                    return
                logger.warning("Lost connection to '%s': %s — reconnecting in 5s", self._config.name, exc)
                await asyncio.sleep(5)

    async def _run_server(self, ready_event: asyncio.Event) -> None:
        headers = self._config.headers or {}
        if self._config.transport == "streamable_http":
            async with streamablehttp_client(self._config.url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, ready_event)
                    try:
                        await asyncio.sleep(float("inf"))
                    finally:
                        if self._session is session:
                            self._session = None
        elif self._config.transport == "sse":
            async with sse_client(self._config.url, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, ready_event)
                    try:
                        await asyncio.sleep(float("inf"))
                    finally:
                        if self._session is session:
                            self._session = None
        elif self._config.transport == "stdio":
            server_params = StdioServerParameters(
                command=self._config.command, args=self._config.args, env=self._config.env or None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await self._init_session(session, ready_event)
                    try:
                        await asyncio.sleep(float("inf"))
                    finally:
                        if self._session is session:
                            self._session = None
        else:
            raise ValueError(f"Unsupported transport: '{self._config.transport}'")

    async def _init_session(self, session: ClientSession, ready_event: asyncio.Event) -> None:
        await session.initialize()
        tools_response = await session.list_tools()
        tools = [
            ToolInfo(
                name=tool.name,
                server=self._config.name,
                description=tool.description or "",
                input_schema=_schema_to_dict(tool.inputSchema),
            )
            for tool in tools_response.tools
        ]
        for tool in tools:
            self._tools[tool.name] = tool
        self._session = session
        self._server_info = ServerInfo(
            name=self._config.name, status=ServerStatus.CONNECTED, tool_count=len(tools)
        )
        logger.info("Connected to '%s': %d tools loaded", self._config.name, len(tools))
        if not ready_event.is_set():
            ready_event.set()

    def _schedule_reconnect(self) -> None:
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
        reconnect_event = asyncio.Event()
        reconnect_event.set()
        self._server_task = asyncio.get_running_loop().create_task(
            self._run_server_loop(reconnect_event),
            name=f"mcp-{self._config.name}",
        )
        logger.info("Scheduled reconnect for server '%s'", self._config.name)


class CompositeToolRepository(ToolRepository):
    """Aggregates multiple ToolRepository instances, prefixing tool names with the map key."""

    def __init__(self, repos: dict[str, ToolRepository]) -> None:
        self._repos = repos

    async def connect(self) -> None:
        async with anyio.create_task_group() as task_group:
            for repo in self._repos.values():
                task_group.start_soon(repo.connect)

    async def list_tools(self) -> list[ToolInfo]:
        all_tools = []
        for server_prefix, repo in self._repos.items():
            for tool in await repo.list_tools():
                all_tools.append(ToolInfo(
                    name=f"{server_prefix}_{tool.name}",
                    server=tool.server,
                    description=tool.description,
                    input_schema=tool.input_schema,
                ))
        return all_tools

    async def execute_tool(self, tool_name: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
        for server_prefix, repo in self._repos.items():
            if tool_name.startswith(f"{server_prefix}_"):
                raw_tool_name = tool_name[len(server_prefix) + 1:]
                return await repo.execute_tool(raw_tool_name, args, meta=meta)
        raise ToolNotFoundError(tool_name)

    async def close(self) -> None:
        async with anyio.create_task_group() as task_group:
            for repo in self._repos.values():
                task_group.start_soon(repo.close)

    async def get_status(self) -> dict[str, ServerInfo]:
        return {
            server_prefix: repo.server_info
            for server_prefix, repo in self._repos.items()
            if isinstance(repo, MCPServerToolRepository) and repo.server_info
        }


async def _wait_for_ready(server_name: str, ready_event: asyncio.Event, timeout_seconds: float) -> None:
    with anyio.move_on_after(timeout_seconds):
        await ready_event.wait()


def _schema_to_dict(schema: Any) -> dict:
    if not schema:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    return {}
