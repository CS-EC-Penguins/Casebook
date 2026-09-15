"""
A ReAct agent built directly on the google-genai SDK -- no LangGraph, no agent
executor library.

Copy your completed pipeline.py from Week 9 Day 2 into this directory before
running this file. Do not modify pipeline.py.

Usage:
    python agent.py
"""

import os
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


def execute_tool(tool_name: str, tool_args: dict, vector_store) -> dict:
    """Run the tool the model requested and return its result.
"""
    if tool_name == "retrieve":
        result = retrieve(tool_args["query"], vector_store)
        return {
            "passages": result,
            "count": len(result)
        }
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def run_agent(query: str, vector_store, max_iterations: int = 5) -> dict:
    """Run the ReAct loop: call the model, execute any requested tool, repeat.

    Call client.models.generate_content(model="gemini-2.5-flash",
    contents=contents, config=types.GenerateContentConfig(tools=[retrieval_tool])).
    The model's requested calls are in response.function_calls. Stop when a
    response contains no function call, or after max_iterations.

    Feed tool results back by appending the model's own content
    (response.candidates[0].content) followed by a types.Content(role="user",
    parts=[types.Part.from_function_response(...)]) to the contents list.

    Returns {"answer": str, "tool_calls": list[dict], "iterations": int},
    where each tool_calls entry is {"tool": <name>, "args": <args>}. Keep
    those exact keys -- Day 4's trajectory tests read result["tool_calls"]
    and expect "tool"/"args".
    """
    contents = [query]
    tool_calls = []
    for iter in range(max_iterations):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents, 
            config=types.GenerateContentConfig(
                tools=[retrieval_tool]))
        if iter == max_iterations:
            return {
                "answer": "Could not produce a final answer within the iteration limit.",
                "tool_calls": tool_calls,
                "iterations": iter
            }
        elif response.function_calls is None:
            return {
                "answer": response.text,
                "tool_calls": tool_calls,
                "iterations": iter
            }
        else:
            contents.append(response.candidates[0].content)
            for tool_call in response.function_calls:
                tool_res = execute_tool(tool_call.name, tool_call.args, vector_store)
                tool_calls.append({"tool": tool_call.name, "args": tool_call.args})
                contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=tool_call.name, response=tool_res)]))
    print(contents)
    return {
        "answer": response.text,
        "tool_calls": tool_calls,
        "iterations": iter
    }

        
if __name__ == "__main__":
    vs = load_vector_store()

    # Step 1: a simple single-step query. Expect one call to `retrieve`,
    # then a final answer grounded in the retrieved passages.
    # result = run_agent("How do bats locate prey using sound?", vs)
    # print(result["answer"])
    # print(f"\nTool calls made: {result['tool_calls']}")
    # print(f"Iterations: {result['iterations']}")

    # # Step 2: a multi-part query. Does the model break it into sub-queries
    # # and call `retrieve` more than once?
    # result = run_agent(
    #     "How do both bats and honeybees use biological mechanisms to navigate "
    #     "and find food at a distance? What are the key differences in how each "
    #     "species processes environmental information?",
    #     vs,
    # )
    # print(result["answer"])
    # print(f"\nTool calls: {result['tool_calls']}")

# if __name__ == "__main__":
#     vs = load_vector_store()
#     result = execute_tool("retrieve", {"query": "How do bats navigate in the dark?"}, vs)
#     print(f"Got {result['count']} passages")
#     for p in result["passages"]:
#         print(f"  {p['source']}: {p['content'][:100]}")
