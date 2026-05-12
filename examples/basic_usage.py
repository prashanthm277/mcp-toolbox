"""
Basic usage example — connect to two MCP servers, list all tools,
then expose them as LLM-callable meta-tools.
"""

import asyncio

from mcp_toolbox import MCPServerConfig, ServerStatus, ToolRepository, make_tools


async def main() -> None:
    repo = ToolRepository([
        MCPServerConfig(name="ui", url="http://localhost:5020"),
        MCPServerConfig(name="platform", url="http://localhost:5010"),
    ])

    await repo.connect()

    # Check which servers came up
    status = await repo.get_mcp_status()
    for name, info in status.items():
        icon = "✓" if info.status == ServerStatus.CONNECTED else "✗"
        print(f"  {icon} {name}: {info.tool_count} tools")

    # List every tool across all servers
    tools = await repo.list_all_tools()
    print(f"\nTotal tools available: {len(tools)}")
    for tool in tools:
        print(f"  [{tool.server}] {tool.name} — {tool.description[:60]}")

    # Get a specific tool's full definition
    if tools:
        first = tools[0]
        defns = await repo.get_tool_definition([first.name])
        defn = defns[0]
        print(f"\nDefinition for '{defn.name}':")
        print(f"  schema: {defn.input_schema}")

    # Build LLM-callable meta-tools (Anthropic format)
    meta = make_tools(repo)
    print(f"\nMeta-tool definitions (pass to LLM):")
    for d in meta.definitions:
        print(f"  - {d['name']}: {d['description'][:60]}")

    # Simulate LLM dispatching a tool call
    if tools:
        result = await meta.dispatch(
            "get_tool_definition", {"tool_names": [tools[0].name]}
        )
        print(f"\nLLM called get_tool_definition → {result}")

    await repo.close()


if __name__ == "__main__":
    asyncio.run(main())
