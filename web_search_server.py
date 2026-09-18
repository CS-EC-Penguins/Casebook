"""MCP server exposing Casebook's optional Tavily web-search tool."""

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient


mcp = FastMCP("Casebook Web Search")


@mcp.tool()
def web_search(query: str) -> str:
    """Search for current/out-of-corpus information and return cited excerpts."""
    response = TavilyClient().search(
        query=query,
        max_results=3,
        search_depth="basic",
    )
    results = response.get("results", [])
    if not results:
        return "No web results found."

    return "\n\n".join(
        f"[Source: {item.get('url', 'unknown')}]\n{item.get('content', '')}"
        for item in results
    )


if __name__ == "__main__":
    mcp.run()

