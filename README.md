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
    compositeRepo = CompositeToolRepository({"platform": platform_repo})
    metaRepo  = MetaToolRepository(compositeRepo)

    await metaRepo.connect()

    # Two tools the LLM sees
    tools = await metaRepo.list_tools()
    # → [ToolInfo(get_tool_definition), ToolInfo(execute_tool)]

    # LLM discovers a tool's schema
    definitions = await metaRepo.execute_tool(
        "get_tool_definition",
        {"tool_names": ["platform_createPage"]},
    )

    # LLM calls the tool
    result = await metaRepo.execute_tool(
        "execute_tool",
        {"tool_name": "platform_createPage", "tool_args": {"name": "Home"}},
    )

    await metaRepo.close()

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
}

await metaRepo.execute_tool(
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

## Optional: LangChain integration

`mcp_toolbox.langchain_tools` converts any list of `ToolInfo` objects into
LangChain `BaseTool` instances so you can pass them directly to a LangChain
agent or `AgentExecutor`.

### Install

```bash
pip install mcp-toolbox[langchain]
```

### How many tools does the LangChain agent see?

This depends on which repository's `list_tools()` you pass to `convert_to_langchain_tools`:

| Source | Tools exposed to the agent | When to use |
|---|---|---|
| `meta.list_tools()` | **2** — `get_tool_definition` + `execute_tool` | Agent discovers and calls tools dynamically; works well with large or changing tool sets |
| `composite.list_tools()` | **All underlying MCP tools** (one per tool, prefixed) | Agent binds directly to each tool; simpler but the full schema list is in every prompt |

**Using `MetaToolRepository` (2 tools)** — the LLM first calls `get_tool_definition` to learn what a tool expects, then calls `execute_tool` to run it. This keeps the agent's tool list small regardless of how many MCP tools exist underneath.

**Using `CompositeToolRepository` (all tools)** — every MCP tool becomes its own LangChain tool. Easier to reason about, but schema bloat grows with the number of tools.

### Usage

```python
import asyncio
from mcp_toolbox import (
    CompositeToolRepository,
    MCPServerConfig,
    MCPServerToolRepository,
    MetaToolRepository,
)
from mcp_toolbox.langchain_tools import convert_to_langchain_tools

async def main():
    repo = MCPServerToolRepository(
        MCPServerConfig(name="platform", url="http://localhost:5010/mcp", transport="streamable_http")
    )
    composite = CompositeToolRepository({"platform": repo})
    meta = MetaToolRepository(composite)
    await meta.connect()

    # 2 tools: get_tool_definition + execute_tool
    lc_tools = convert_to_langchain_tools(await meta.list_tools(), meta.execute_tool)

    # — or — all underlying MCP tools as individual LangChain tools
    lc_tools_all = convert_to_langchain_tools(await composite.list_tools(), composite.execute_tool)

asyncio.run(main())
```

`convert_to_langchain_tools` accepts three arguments:

| Argument | Type | Description |
|---|---|---|
| `tools` | `list[ToolInfo]` | Tool definitions — typically from `list_tools()` |
| `executor` | `async callable` | Called as `executor(tool_name, args, meta=None)` — typically `MetaToolRepository.execute_tool` |
| `meta_provider` | `callable \| None` | Optional `(tool_name, kwargs) -> dict \| None` — return auth/context to forward with each call |

### Forwarding auth context per call

```python
def my_meta_provider(tool_name: str, kwargs: dict) -> dict | None:
    return {"auth_cookie": get_current_user_cookie()}

lc_tools = convert_to_langchain_tools(tools, meta.execute_tool, meta_provider=my_meta_provider)
```

See [`examples/langchain_usage.py`](examples/langchain_usage.py) for a complete runnable example.

## Development

```bash
uv sync --extra dev
uv run pytest
```
