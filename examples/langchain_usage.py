"""
LangChain integration — connect to an MCP server and expose its tools as
LangChain BaseTool instances.

Requires the [langchain] extra:
    pip install mcp-toolbox[langchain]
"""

import asyncio

from mcp_toolbox import (
    CompositeToolRepository,
    MCPServerConfig,
    MCPServerToolRepository,
    MetaToolRepository,
)
from mcp_toolbox.langchain_tools import convert_to_langchain_tools


async def main() -> None:
    platform_repo = MCPServerToolRepository(
        MCPServerConfig(name="platform", url="http://localhost:5010/mcp", transport="streamable_http")
    )
    composite = CompositeToolRepository({"platform": platform_repo})
    meta = MetaToolRepository(composite)
    await meta.connect()

    tools = await meta.list_tools()

    # Basic conversion — no auth forwarding
    lc_tools = convert_to_langchain_tools(tools, meta.execute_tool)

    print(f"LangChain tools ready: {[t.name for t in lc_tools]}")

    # With per-call auth context forwarding
    def meta_provider(tool_name: str, kwargs: dict) -> dict | None:
        # Return whatever auth / context the MCP server expects
        return {"auth_cookie": "my-session-token"}

    lc_tools_with_auth = convert_to_langchain_tools(
        tools, meta.execute_tool, meta_provider=meta_provider
    )

    # Pass lc_tools or lc_tools_with_auth to your LangChain agent, e.g.:
    #
    #   from langchain.agents import AgentExecutor, create_tool_calling_agent
    #   from langchain_openai import ChatOpenAI
    #
    #   llm = ChatOpenAI(model="gpt-4o")
    #   agent = create_tool_calling_agent(llm, lc_tools_with_auth, prompt)
    #   executor = AgentExecutor(agent=agent, tools=lc_tools_with_auth)
    #   result = await executor.ainvoke({"input": "Create a page called Home"})

    await meta.close()


if __name__ == "__main__":
    asyncio.run(main())
