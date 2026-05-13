# mcp-toolbox

Connect LLM agents to one or more [MCP](https://modelcontextprotocol.io/) servers through a clean layered API. The agent always sees exactly two tools — `get_tool_definition` and `execute_tool` — regardless of how many servers or tools are underneath.

## Install

```bash
pip install mcp-toolbox
```

## Quick start

```python
import asyncio
from mcp_toolbox import (
    MCPServerConfig,
    MCPServerToolRepository,
    CompositeToolRepository,
    MetaToolRepository,
)

async def main():
    platform_repo = MCPServerToolRepository(
        MCPServerConfig(name="platform", url="http://localhost:5010/mcp", transport="streamable_http")
    )
    composite = CompositeToolRepository({"platform": platform_repo})
    meta      = MetaToolRepository(composite)

    await meta.connect()

    # Two tools the LLM sees
    tools = await meta.list_tools()
    # → [ToolInfo(get_tool_definition), ToolInfo(execute_tool)]

    # LLM discovers a tool's schema
    definitions = await meta.execute_tool(
        "get_tool_definition",
        {"tool_names": ["platform_createPage"]},
    )

    # LLM calls the tool
    result = await meta.execute_tool(
        "execute_tool",
        {"tool_name": "platform_createPage", "tool_args": {"name": "Home"}},
    )

    await meta.close()

asyncio.run(main())
```

## Architecture

```
MetaToolRepository          ← what the LLM sees (get_tool_definition + execute_tool)
    └── CompositeToolRepository({"platform": ..., "ui": ...})
            ├── MCPServerToolRepository  →  platform MCP server
            └── MCPServerToolRepository  →  ui MCP server
```

| Layer | Responsibility |
|---|---|
| `MCPServerToolRepository` | Persistent session to one MCP server; tools stored without prefix |
| `CompositeToolRepository` | Aggregates repos; adds `{key}_` prefix to tool names |
| `MetaToolRepository` | Wraps any repo; exposes only `get_tool_definition` + `execute_tool` to the LLM |

## API

### `MCPServerToolRepository`

```python
MCPServerToolRepository(config: MCPServerConfig)
```

| Method | Returns | Description |
|---|---|---|
| `connect()` | `None` | Open a persistent MCP session and load tools |
| `list_tools()` | `list[ToolInfo]` | All tools from this server (no prefix) |
| `execute_tool(tool_name, args, meta=None)` | `Any` | Call a tool on this server |
| `close()` | `None` | Cancel the background session task |
| `server_info` | `ServerInfo \| None` | Connection status and tool count |

### `CompositeToolRepository`

```python
CompositeToolRepository(repos: dict[str, ToolRepository])
```

| Method | Returns | Description |
|---|---|---|
| `connect()` | `None` | Connect all sub-repositories in parallel |
| `list_tools()` | `list[ToolInfo]` | All tools with `{key}_` prefix (e.g. `platform_createPage`) |
| `execute_tool(tool_name, args, meta=None)` | `Any` | Strip prefix, route to the correct sub-repo |
| `close()` | `None` | Close all sub-repositories |
| `get_status()` | `dict[str, ServerInfo]` | Per-server connection status |

### `MetaToolRepository`

```python
MetaToolRepository(repo: ToolRepository)
```

Wraps any `ToolRepository` and exposes exactly two tools to the LLM:

| Tool | Args | Description |
|---|---|---|
| `get_tool_definition` | `tool_names: list[str]` | Returns schema + description for each named tool |
| `execute_tool` | `tool_name: str`, `tool_args: dict` | Executes the named tool via the inner repo |

### `MCPServerConfig`

```python
MCPServerConfig(
    name="platform",
    url="http://localhost:5010/mcp",   # required for streamable_http / sse
    transport="sse",                   # "sse" (default) | "streamable_http" | "stdio"
    command="my_server",               # required for stdio
    args=["--flag"],                   # stdio only
    env={"KEY": "value"},              # stdio only
    headers={"Authorization": "..."},  # HTTP transports
    tool_call_timeout=60.0,            # seconds before a call times out and session reconnects
)
```

### `ToolInfo`

```python
@dataclass
class ToolInfo:
    name: str
    server: str
    description: str
    input_schema: dict
```

Returned by `list_tools()` on any repository. Pass `.name`, `.description`, `.input_schema` directly to your LLM SDK as a tool definition.

## Forwarding auth context

Pass a `meta` dict to `execute_tool` — it is forwarded as-is to the MCP server:

```python
request_context = {
    "auth_cookie": "...",
    "projectId": "...",
}

await meta.execute_tool(
    "execute_tool",
    {"tool_name": "platform_createPage", "tool_args": {"name": "Home"}},
    meta=request_context,
)
```

## Transports

| Transport | When to use |
|---|---|
| `sse` (default) | Servers that push responses over Server-Sent Events |
| `streamable_http` | Modern HTTP streaming servers |
| `stdio` | Local servers launched as a subprocess |

## Reconnect behaviour

If a tool call exceeds `tool_call_timeout`, the broken session is evicted and a background reconnect task starts automatically. Calls made during reconnection raise `ConnectionError`.

## Development

```bash
uv sync --extra dev
uv run pytest
```
