# mcp-toolbox

Aggregate tools from multiple MCP servers into a unified registry, with LLM-callable meta-tools (`get_tool_definition`, `execute_tool`) that let an AI agent discover and invoke any tool dynamically.

## Install

```bash
pip install mcp-toolbox
# With LangChain support:
pip install "mcp-toolbox[langchain]"
```

## Quick Start

```python
import asyncio
from mcp_toolbox import MCPServerConfig, ToolRepository, make_tools

async def main():
    repo = ToolRepository([
        MCPServerConfig(name="ui", url="http://localhost:5020"),
        MCPServerConfig(
            name="platform",
            url="http://localhost:5010",
            headers={"Authorization": "Bearer <token>"},
            trusted=True,
            tool_call_timeout=30.0,
        ),
    ])

    await repo.connect()

    tools = await repo.list_all_tools()    # all tools from both servers
    status = await repo.get_mcp_status()   # per-server connection status

    # Expose as LLM-callable meta-tools (Anthropic / OpenAI / Gemini style)
    meta = make_tools(repo)
    await meta.dispatch("get_tool_definition", {"tool_names": ["ui_click_element"]})
    await meta.dispatch("execute_tool", {"tool_name": "ui_click_element", "tool_args": {"selector": "#btn"}})

asyncio.run(main())
```

## API

### `ToolRepository(mcp_servers)`

| Method | Returns | Description |
|---|---|---|
| `connect()` | `None` | Connect to all servers in parallel; partial failures are safe |
| `list_all_tools()` | `list[ToolInfo]` | All tools across all connected servers |
| `get_mcp_status()` | `dict[str, ServerInfo]` | Per-server connection status and tool count |
| `get_tool_definition(tool_names)` | `list[ToolDefinition]` | Full schema + description for one or more tools |
| `execute_tool(tool_name, arguments, meta=None)` | `Any` | Call a tool; optional `meta` dict is forwarded to the MCP server |
| `close()` | `None` | Cancel background server tasks and release connections |

Tool names are stored as `{server_name}_{raw_tool_name}` (e.g. `platform_createWebPage`). The prefix is stripped automatically when calling the MCP server.

### `make_tools(repo, meta_provider=None) → MetaToolSet`

Framework-agnostic meta-tools for any LLM SDK (Anthropic, OpenAI, Gemini, …).

```python
def my_meta_provider(tool_name: str) -> dict | None:
    # Return auth context for trusted tools, or None to skip
    return {"auth_cookie": "...", "projectId": "..."}

meta = make_tools(repo, meta_provider=my_meta_provider)
```

`MetaToolSet` exposes:
- `.definitions` — list of JSON-schema dicts to pass to the LLM as tools
- `.dispatch(tool_name, arguments)` — route an LLM tool call to the correct handler

### `make_langchain_tools(repo, meta_provider=None) → list[BaseTool]`

LangChain/LangGraph integration. Returns `[ExecuteToolShim, GetToolDefinitionShim]` ready to register as agent tools.

```python
from mcp_toolbox.langchain_tools import make_langchain_tools

tools = make_langchain_tools(repo, meta_provider=my_meta_provider)
# Register `tools` as internal tools for your LangChain/LangGraph agent
```

### `MCPServerConfig`

```python
MCPServerConfig(
    name="my_server",
    url="http://localhost:5000",        # required for streamable_http / sse
    transport="streamable_http",        # "streamable_http" | "sse" | "stdio"
    command="my_server_bin",            # required for stdio
    args=["--flag"],                    # stdio only
    env={"API_KEY": "..."},             # stdio only
    headers={"Authorization": "..."},  # HTTP headers for http/sse transports
    trusted=False,                      # if True, meta_provider is called on execute_tool
    tool_call_timeout=60.0,             # seconds before a tool call is aborted and session reconnects
)
```

### Auth / `meta` forwarding

`trusted=True` signals that this server requires per-request auth context (e.g. session cookies). Supply a `meta_provider` callback to `make_tools()` or `make_langchain_tools()` — it is called with the tool name and should return a dict (forwarded as MCP `meta=`) or `None`.

```python
def meta_provider(tool_name: str) -> dict | None:
    if is_trusted_tool(tool_name):
        return {"auth_cookie": session.cookie, "projectId": session.project_id}
    return None
```

## Transports

| Transport | Config required |
|---|---|
| `streamable_http` (default) | `url` |
| `sse` | `url` |
| `stdio` | `command`, optionally `args` / `env` |

## Reconnect behaviour

If a tool call times out (`tool_call_timeout` seconds), the broken session is evicted and a background reconnect task starts automatically. Calls made while reconnecting receive a `ConnectionError` (surfaced as a `ToolException` in the LangChain layer) until the session is re-established.

## Development

```bash
uv sync --extra dev
uv run pytest
```
