"""
A ReAct agent built directly on the google-genai SDK -- no LangGraph, no agent
executor library.

Copy your completed pipeline.py from Week 9 Day 2 into this directory before
running this file. Do not modify pipeline.py.

Usage:
    python agent.py
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
from pipeline import load_vector_store, retrieve

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="europe-west4",
)


#Define retrieve_declaration, a types.FunctionDeclaration describing the
# retrieve() function above (name="retrieve" -- must match exactly, since
# execute_tool() below dispatches on this string -- a clear description, and
# a "query" string parameter), then wrap it:
#   retrieval_tool = types.Tool(function_declarations=[retrieve_declaration])
retrieve_declaration = types.FunctionDeclaration(
    name = "retrieve",
    description = ("Search the internal document corpus fpr passages relevant to a query"
                   "Use this for questions about the documnets in the knowledgebase"
                   "Returns a list of text passages with source document names"
                   ),
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run against the document corpus"
            }
        },
        "required": ["query"]
    }
)

retrieval_tool = types.Tool(function_declarations=[retrieve_declaration])



async def execute_tool(tool_name: str, tool_args: dict, vector_store, session) -> dict:
    if tool_name == "retrieve":
        passages = retrieve(tool_args["query"], vector_store)
        return {"passages": passages, "count": len(passages)}
    if tool_name == "web_search":
        result = await session.call_tool("web_search", tool_args)
        return {"text": result.content[0].text}
    raise ValueError(f"Unknown tool: {tool_name}")



async def run_agent(query: str, vector_store, max_iterations: int = 5) -> dict:
    """Run the ReAct loop: call the model, execute any requested tool, repeat.

    Opens an MCP session to web_search_server.py, discovers the web_search tool
    declaration from the server manifest, then runs the ReAct loop.

    Returns {"answer": str, "tool_calls": list[dict], "iterations": int},
    where each tool_calls entry is {"tool": <name>, "args": <args>}. Keep
    those exact keys -- Day 4's trajectory tests read result["tool_calls"]
    and expect "tool"/"args".
    """
    server_params = StdioServerParameters(
        command="python",
        args=["web_search_server.py"],
        env={"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            ws_tool = next(t for t in mcp_tools.tools if t.name == "web_search")
            web_search_declaration = types.FunctionDeclaration(
                name=ws_tool.name,
                description=ws_tool.description,
                parameters=ws_tool.inputSchema,
            )
            both_tools = types.Tool(function_declarations=[retrieve_declaration, web_search_declaration])

            contents = [query]
            tool_calls = []
            for iter in range(max_iterations):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(tools=[both_tools]))
                if response.function_calls is None:
                    return {
                        "answer": response.text,
                        "tool_calls": tool_calls,
                        "iterations": iter
                    }
                contents.append(response.candidates[0].content)
                response_parts = []
                for tool_call in response.function_calls:
                    tool_res = await execute_tool(tool_call.name, tool_call.args, vector_store, session)
                    tool_calls.append({"tool": tool_call.name, "args": tool_call.args})
                    response_parts.append(types.Part.from_function_response(name=tool_call.name, response=tool_res))
                contents.append(types.Content(role="user", parts=response_parts))
            print(contents)
            return {
                "answer": "Could not produce a final answer within the iteration limit.",
                "tool_calls": tool_calls,
                "iterations": max_iterations
            }

        
async def main():
    vs = load_vector_store()

    # Step 1: a simple single-step query. Expect one call to `retrieve`,
    # then a final answer grounded in the retrieved passages.


    # Step 2: a multi-part query. Does the model break it into sub-queries
    # and call `retrieve` more than once?
    # result = await run_agent(
    #     "How do both bats and honeybees use biological mechanisms to navigate "
    #     "and find food at a distance? What are the key differences in how each "
    #     "species processes environmental information?",
    #     vs,
    # )
    # print(result["answer"])
    # print(f"\nTool calls: {result['tool_calls']}")

    result = await run_agent(
        "What frequency ranges do bats rely on when hunting with echolocation?",
        vs
    )
    print(result["tool_calls"])

    result = await run_agent(
        "What coral reef restoration projects have been announced in the past few months?",
        vs
    )
    print(result["tool_calls"])

    result = await run_agent(
        "What causes coral bleaching, and how have conservation groups responded to it recently?",
        vs
    )
    print(result["tool_calls"])


if __name__ == "__main__":
    asyncio.run(main())



# if __name__ == "__main__":
#     vs = load_vector_store()
#     result = execute_tool("retrieve", {"query": "How do bats navigate in the dark?"}, vs)
#     print(f"Got {result['count']} passages")
#     for p in result["passages"]:
#         print(f"  {p['source']}: {p['content'][:100]}")
