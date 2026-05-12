from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServerStatus(Enum):
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class MCPServerConfig:
    name: str
    url: str | None = None            # for streamable_http / sse transport
    transport: str = "streamable_http"
    command: str | None = None        # for stdio transport
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)  # HTTP headers for http/sse transports
    trusted: bool = False             # if True, metaParams from LangGraph config are forwarded
    tool_name_prefix: bool = True     # if True, stored tool names are prefixed with "{server_name}_"
    tool_call_timeout: float = 60.0   # seconds before a call_tool request is aborted

    def __post_init__(self) -> None:
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(f"Server '{self.name}': 'url' is required for transport '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"Server '{self.name}': 'command' is required for transport 'stdio'")


@dataclass
class ToolInfo:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolDefinition:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ServerInfo:
    name: str
    status: ServerStatus
    tool_count: int
    error: str | None = None
