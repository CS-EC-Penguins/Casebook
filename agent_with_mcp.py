"""Discover and exercise Casebook's web-search tool through MCP.

This is a protocol smoke test, not the main agent. ``agent.py`` remains the
non-LangGraph ReAct implementation.
"""

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_web_search(query: str) -> None:
    server_script = Path(__file__).with_name("web_search_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env={"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Discovered tools:")
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description}")

            result = await session.call_tool("web_search", {"query": query})
            print("\nResult:")
            for block in result.content:
                if hasattr(block, "text"):
                    print(block.text)


if __name__ == "__main__":
    search_query = " ".join(sys.argv[1:]).strip()
    if not search_query:
        search_query = "latest EU AI Act implementation developments"
    asyncio.run(call_web_search(search_query))

