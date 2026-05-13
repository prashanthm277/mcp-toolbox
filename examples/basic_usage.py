"""
Basic usage — connect to two MCP servers and expose them to an LLM agent
via MetaToolRepository.
"""

import asyncio

from mcp_toolbox import (
    CompositeToolRepository,
    MCPServerConfig,
    MCPServerToolRepository,
    MetaToolRepository,
    ServerStatus,
)


async def main() -> None:
    ui_repo = MCPServerToolRepository(
        MCPServerConfig(name="ui", url="http://localhost:5020", transport="streamable_http")
    )
    platform_repo = MCPServerToolRepository(
        MCPServerConfig(name="platform", url="http://localhost:5010", transport="streamable_http")
    )

    composite = CompositeToolRepository({"ui": ui_repo, "platform": platform_repo})
    await composite.connect()

    # Check which servers came up
    status = await composite.get_status()
    for name, info in status.items():
        icon = "✓" if info.status == ServerStatus.CONNECTED else "✗"
        print(f"  {icon} {name}: {info.tool_count} tools")

    # List every tool across all servers (with prefix)
    tools = await composite.list_tools()
    print(f"\nTotal tools available: {len(tools)}")
    for tool in tools:
        print(f"  [{tool.server}] {tool.name} — {tool.description[:60]}")

    # Wrap in MetaToolRepository — this is what you expose to the LLM
    metaRepo = MetaToolRepository(composite)

    llm_tools = await metaRepo.list_tools()
    print(f"\nTools exposed to LLM:")
    for t in llm_tools:
        print(f"  - {t.name}: {t.description[:60]}")

    # Simulate: LLM calls get_tool_definition
    if tools:
        result = await metaRepo.execute_tool(
            "get_tool_definition", {"tool_names": [tools[0].name]}
        )
        print(f"\nLLM called get_tool_definition → {result}")

    await composite.close()


if __name__ == "__main__":
    asyncio.run(main())
