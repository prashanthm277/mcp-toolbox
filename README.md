# mcp-toolbox

Aggregate tools from multiple MCP servers into a unified registry, with LLM-callable meta-tools (`get_tool_definition`, `execute_tool`) that let an AI agent discover and invoke any tool dynamically.

## Install

```bash
pip install mcp-toolbox
```

## Quick Start

```python
import asyncio
from mcp_toolbox import MCPServerConfig, ToolRepository, make_tools

async def main():
    repo = ToolRepository([
        MCPServerConfig(name="ui", url="http://localhost:5020"),
        MCPServerConfig(name="platform", url="http://localhost:5010"),
    ])

    await repo.connect()

    tools = await repo.list_all_tools()          # all tools from both servers
    status = await repo.get_mcp_status()         # per-server connection status

    # Expose as LLM-callable meta-tools
    meta = make_tools(repo)
    result = await meta.dispatch("get_tool_definition", {"tool_names": ["ui_click_element"]})
    result = await meta.dispatch("execute_tool", {"tool_name": "ui_click_element", "arguments": {"selector": "#btn"}})

asyncio.run(main())
```

## API

### `ToolRepository(mcp_servers)`

| Method | Returns | Description |
|---|---|---|
| `connect()` | `None` | Connect to all servers (parallel, partial failure safe) |
| `list_all_tools()` | `list[ToolInfo]` | All tools across all connected servers |
| `get_mcp_status()` | `dict[str, ServerInfo]` | Per-server connection status and tool count |
| `get_tool_definition(tool_names)` | `list[ToolDefinition]` | Full schema + description for one or more tools |
| `execute_tool(tool_name, arguments)` | `Any` | Call a tool and return its result |
| `close()` | `None` | Cancel background server tasks and release connections |

### `make_tools(repo) → MetaToolSet`

Returns a `MetaToolSet` with:
- `.definitions` — list of JSON-schema tool dicts to pass to the LLM
- `.dispatch(tool_name, arguments)` — route an LLM tool call to the correct handler

### `MCPServerConfig`

```python
MCPServerConfig(
    name="my_server",
    url="http://localhost:5000",     # required for streamable_http / sse
    transport="streamable_http",     # "streamable_http" | "sse" | "stdio"
    command="my_server_bin",         # required for stdio
    args=["--flag"],
    env={"API_KEY": "..."},
)
```

## Transports

| Transport | Config required |
|---|---|
| `streamable_http` (default) | `url` |
| `sse` | `url` |
| `stdio` | `command`, optionally `args` / `env` |

## Development

```bash
uv sync --extra dev
uv run pytest
```
