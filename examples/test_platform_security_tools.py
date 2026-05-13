"""
Integration test — get_tool_definition and execute_tool for platform security tools:
  - platform_isAuthenticationEnabled
  - platform_getConfiguredSecurityProviders
"""

import asyncio
import json

from mcp_toolbox import (
    CompositeToolRepository,
    MCPServerConfig,
    MCPServerToolRepository,
    MetaToolRepository,
    ServerStatus,
)

PLATFORM_URL = "http://localhost:5010/mcp"
PLATFORM_HEADERS = {"Authorization": "Bearer changeme"}

REQUEST_CONTEXT = {
    "auth_cookie": "sdfdsdggfwsgdfggrfed",
    "projectId": "WMPRJ2c92808b9d6b702e019d7100ac930000",
    "projectType": "APPLICATION",
    "platformType": "WEB_PRISM",
    "prismProject": False,
}

SECURITY_TOOLS = [
    "platform_isAuthenticationEnabled",
    "platform_getConfiguredSecurityProviders",
]


async def main() -> None:
    platform_repo = MCPServerToolRepository(
        MCPServerConfig(name="platform", url=PLATFORM_URL, headers=PLATFORM_HEADERS, transport="streamable_http")
    )
    composite = CompositeToolRepository({"platform": platform_repo})

    print(f"Connecting to platform MCP server at {PLATFORM_URL} ...")
    await composite.connect()

    server_status = await composite.get_status()
    platform_info = server_status.get("platform")
    if platform_info is None or platform_info.status != ServerStatus.CONNECTED:
        print(f"  ERROR: platform server not connected — {platform_info}")
        await composite.close()
        return
    print(f"  Connected. {platform_info.tool_count} tools available.\n")

    meta = MetaToolRepository(composite)

    # ── get_tool_definition ──────────────────────────────────────────────────
    print("=" * 60)
    print("get_tool_definition")
    print("=" * 60)

    tool_definitions = await meta.execute_tool("get_tool_definition", {"tool_names": SECURITY_TOOLS})
    for definition in tool_definitions:
        print(f"\nTool  : {definition['name']}")
        print(f"Server: {definition['server']}")
        print(f"Desc  : {definition['description']}")
        print(f"Schema: {json.dumps(definition['input_schema'], indent=2)}")

    # ── execute_tool ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("execute_tool")
    print("=" * 60)

    for tool_name in SECURITY_TOOLS:
        print(f"\n>> Calling {tool_name} ...")
        try:
            result = await meta.execute_tool(
                "execute_tool",
                {"tool_name": tool_name, "tool_args": {}},
                meta=REQUEST_CONTEXT,
            )
            print(f"   Result: {result}")
        except Exception as exc:
            print(f"   ERROR: {exc}")

    await composite.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
