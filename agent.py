"""A direct Google Gen AI ReAct agent for the Casebook knowledge base.

The agent can retrieve from the controlled Casebook corpus or search the web
for explicitly current/out-of-corpus information. It deliberately does not use
LangGraph; the model/tool loop is visible in :func:`run_agent`.

Usage:
    python agent.py "What are the NIST AI RMF Core functions?"
"""

import asyncio
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from langfuse import observe
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pipeline import load_vector_store, retrieve


MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a research assistant for the Casebook AI governance knowledge base.
Use the retrieve tool for questions about the EU AI Act, NIST AI RMF, the
International AI Safety Report 2026, and the Meridian AI governance report.
Use web_search only for current developments or facts outside that corpus.
For a question combining corpus and current information, call both tools and
keep their evidence clearly distinguished. Before answering a factual question,
call the appropriate tool. Cite every factual claim using the source labels
returned by the tools. If the tools do not contain the answer, say so. Do not
present general model knowledge as sourced evidence."""


retrieve_declaration = types.FunctionDeclaration(
    name="retrieve",
    description=(
        "Search the controlled Casebook document corpus. It contains the EU AI "
        "Act, NIST AI Risk Management Framework, International AI Safety Report "
        "2026, and Meridian AI governance material. Use this before answering "
        "questions about those documents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused semantic search query for the corpus",
            }
        },
        "required": ["query"],
    },
)

retrieval_tool = types.Tool(function_declarations=[retrieve_declaration])


def create_client():
    """Create the Vertex AI client only when the agent is actually run."""
    return genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="europe-west4",
    )


def declarations_from_mcp_tools(mcp_tools: list) -> list:
    """Translate discovered MCP tool manifests into Gemini declarations."""
    return [
        types.FunctionDeclaration(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
        )
        for mcp_tool in mcp_tools
    ]


async def execute_tool(
    tool_name: str,
    tool_args: dict,
    vector_store,
    mcp_session,
    mcp_tool_names: set[str],
) -> dict:
    """Dispatch a local retrieval call or a remotely exposed MCP tool."""
    query = tool_args["query"]

    if tool_name == "retrieve":
        passages = retrieve(query, vector_store)
        normalised = [
            {
                "source": Path(passage["source"]).name,
                "content": passage["content"],
                "score": passage["score"],
            }
            for passage in passages
        ]
        return {"passages": normalised, "count": len(normalised)}

    if tool_name in mcp_tool_names:
        result = await mcp_session.call_tool(tool_name, tool_args)
        text_blocks = [
            block.text
            for block in result.content
            if hasattr(block, "text")
        ]
        return {
            "content": "\n\n".join(text_blocks),
            "is_error": bool(getattr(result, "isError", False)),
        }

    raise ValueError(f"Unknown tool: {tool_name}")


def contexts_from_tool_result(tool_result: dict) -> list[str]:
    """Extract source-labelled text for evaluation and inspection."""
    if "content" in tool_result:
        return [tool_result["content"]] if tool_result["content"] else []

    evidence = tool_result.get("passages", tool_result.get("results", []))
    return [
        f"[Source: {item.get('source', 'unknown')}]\n{item.get('content', '')}"
        for item in evidence
    ]


async def _run_agent(
    query: str,
    vector_store,
    mcp_session,
    mcp_tools: list,
    max_iterations: int = 5,
    model_client=None,
) -> dict:
    """Run the model/tool loop using local retrieval and discovered MCP tools.

    ``model_client`` is injectable so the loop can be tested without making a
    paid model call. Returned contexts contain every passage or web excerpt the
    agent saw, in tool-call order.
    """
    client = model_client or create_client()
    mcp_declarations = declarations_from_mcp_tools(mcp_tools)
    agent_tools = [
        retrieval_tool,
        types.Tool(function_declarations=mcp_declarations),
    ]
    mcp_tool_names = {mcp_tool.name for mcp_tool in mcp_tools}
    contents = [query]
    tool_calls = []
    contexts = []

    for iteration in range(1, max_iterations + 1):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=agent_tools,
                temperature=0,
            ),
        )

        function_calls = response.function_calls or []
        if not function_calls:
            return {
                "question": query,
                "answer": response.text or "",
                "tool_calls": tool_calls,
                "contexts": contexts,
                "iterations": iteration,
            }

        contents.append(response.candidates[0].content)
        function_responses = []

        for tool_call in function_calls:
            args = dict(tool_call.args or {})
            result = await execute_tool(
                tool_call.name,
                args,
                vector_store,
                mcp_session,
                mcp_tool_names,
            )
            tool_calls.append({"tool": tool_call.name, "args": args})
            contexts.extend(contexts_from_tool_result(result))
            function_responses.append(
                types.Part.from_function_response(
                    name=tool_call.name,
                    response=result,
                )
            )

        contents.append(types.Content(role="user", parts=function_responses))

    return {
        "question": query,
        "answer": "Could not produce a final answer within the iteration limit.",
        "tool_calls": tool_calls,
        "contexts": contexts,
        "iterations": max_iterations,
    }


async def run_agent_async(
    query: str,
    vector_store,
    max_iterations: int = 5,
    model_client=None,
) -> dict:
    """Open one MCP session and keep it alive for the complete agent run."""
    server_script = Path(__file__).with_name("web_search_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()
            return await _run_agent(
                query,
                vector_store,
                session,
                discovered.tools,
                max_iterations=max_iterations,
                model_client=model_client,
            )


@observe()
def run_agent(
    query: str,
    vector_store,
    max_iterations: int = 5,
    model_client=None,
) -> dict:
    """Synchronous entry point retained for the CLI and evaluation harness."""
    return asyncio.run(
        run_agent_async(
            query,
            vector_store,
            max_iterations=max_iterations,
            model_client=model_client,
        )
    )


@observe()
def ask_agent(question: str, vector_store=None) -> dict:
    """Run one Casebook question through the MCP-enabled agent.

    A vector store may be supplied so batch evaluation can reuse one
    connection. Interactive callers can omit it and let this function connect.
    """
    store = vector_store if vector_store is not None else load_vector_store()
    return run_agent(question, store)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "What are the NIST AI RMF Core functions?"

    result = ask_agent(question)
    print(result["answer"])
    print("\nTool calls:")
    for call in result["tool_calls"]:
        print(f"  {call['tool']}: {call['args']}")
