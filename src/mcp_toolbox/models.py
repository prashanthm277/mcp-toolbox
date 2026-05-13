from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServerStatus(Enum):
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class MCPServerConfig:
    name: str
    url: str | None = None
    transport: str = "sse"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tool_call_timeout: float = 60.0

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
class ServerInfo:
    name: str
    status: ServerStatus
    tool_count: int
    error: str | None = None
