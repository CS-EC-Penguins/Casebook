"""LangGraph agent for the Casebook knowledge base.

The graph alternates between a Gemini model and its tools until the model
returns a final answer. Corpus retrieval is local; web search is discovered
from ``web_search_server.py`` over MCP.

Usage:
    python graph_agent.py "What are the NIST AI RMF Core functions?"
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langfuse import observe
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pipeline import load_vector_store, retrieve as pipeline_retrieve


MODEL_NAME = "gemini-2.5-flash"
DEFAULT_MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a research assistant for the Casebook AI governance knowledge base.
Use the retrieve tool for questions about the EU AI Act, NIST AI RMF, the
International AI Safety Report 2026, and the Meridian AI governance report.
Use web_search only for current developments or facts outside that corpus.
For a question combining corpus and current information, call both tools and
keep their evidence clearly distinguished. Before answering a factual question,
call the appropriate tool. Cite every factual claim using the source labels
returned by the tools. If the tools do not contain the answer, say so. Do not
present general model knowledge as sourced evidence."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model_calls: int


def create_retrieve_tool(vector_store):
    """Create a LangChain retrieval tool bound to one vector store."""

    @tool("retrieve")
    def retrieve_tool(query: str) -> str:
        """Search the controlled Casebook AI-governance document corpus.

        The corpus contains the EU AI Act, NIST AI Risk Management Framework,
        International AI Safety Report 2026, and Meridian AI governance
        material. Use this before answering questions about those documents.
        """
        passages = pipeline_retrieve(query, vector_store)
        if not passages:
            return "No relevant passages found."

        return "\n\n".join(
            f"[Source: {Path(passage['source']).name}]\n{passage['content']}"
            for passage in passages
        )

    return retrieve_tool


@observe()
def call_model(state: AgentState, model) -> dict:
    """Invoke the tool-bound model and append its response to graph state."""
    messages = state["messages"]
    if not any(isinstance(message, SystemMessage) for message in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = model.invoke(messages)
    return {
        "messages": [response],
        "model_calls": state.get("model_calls", 0) + 1,
    }


def should_continue(state: AgentState, max_iterations: int) -> str:
    """Route tool requests to ToolNode unless the model-call limit was hit."""
    last_message = state["messages"][-1]
    if (
        getattr(last_message, "tool_calls", None)
        and state.get("model_calls", 0) < max_iterations
    ):
        return "call_tools"
    return END


def build_graph(model, tool_node, max_iterations: int = DEFAULT_MAX_ITERATIONS):
    """Compile the model -> tools -> model Casebook state graph."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    builder = StateGraph(AgentState)
    builder.add_node("call_model", lambda state: call_model(state, model))
    builder.add_node("call_tools", tool_node)
    builder.set_entry_point("call_model")
    builder.add_conditional_edges(
        "call_model",
        lambda state: should_continue(state, max_iterations),
        {"call_tools": "call_tools", END: END},
    )
    builder.add_edge("call_tools", "call_model")
    return builder.compile()


def extract_contexts_from_messages(messages: list) -> list[str]:
    """Return every retrieval or web-search result seen by the model."""
    return [
        message.content
        if isinstance(message.content, str)
        else str(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    ]


def summarise_run(question: str, messages: list, model_calls: int) -> dict:
    """Convert LangGraph messages into the existing Casebook result contract."""
    tool_calls = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tool_calls.append(
                {"tool": tool_call["name"], "args": dict(tool_call["args"])}
            )

    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        answer = "Could not produce a final answer within the iteration limit."
    else:
        answer = last_message.content
        if not isinstance(answer, str):
            answer = str(answer)

    return {
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
        "contexts": extract_contexts_from_messages(messages),
        "messages": messages,
        "iterations": model_calls,
    }


async def run_agent_async(
    question: str,
    vector_store,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model=None,
) -> dict:
    """Discover MCP tools and run the graph while their session is open."""
    server_script = Path(__file__).with_name("web_search_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            tools = [create_retrieve_tool(vector_store), *mcp_tools]
            base_model = model or ChatVertexAI(
                model_name=MODEL_NAME,
                temperature=0,
            )
            bound_model = base_model.bind_tools(tools)
            graph = build_graph(
                bound_model,
                ToolNode(tools),
                max_iterations=max_iterations,
            )
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "model_calls": 0,
            }
            final_state = await graph.ainvoke(
                initial_state,
                config={"recursion_limit": max_iterations * 2 + 1},
            )

    return summarise_run(
        question,
        final_state["messages"],
        final_state["model_calls"],
    )


@observe()
def ask_agent(
    question: str,
    vector_store=None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model=None,
) -> dict:
    """Run one question through the MCP-enabled LangGraph agent."""
    store = vector_store if vector_store is not None else load_vector_store()
    return asyncio.run(
        run_agent_async(
            question,
            store,
            max_iterations=max_iterations,
            model=model,
        )
    )


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "What are the NIST AI RMF Core functions?"

    result = ask_agent(query)
    print(result["answer"])
    print("\nTool calls:")
    for call in result["tool_calls"]:
        print(f"  {call['tool']}: {call['args']}")

    print("\nGraph iterations:", result["iterations"])
