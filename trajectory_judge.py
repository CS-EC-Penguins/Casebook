"""
LLM judge that scores whether the agent chose appropriate tools for each query.

Usage:
    python trajectory_judge.py
"""
import json
from langchain_google_vertexai import ChatVertexAI
from graph_agent import ask_agent

TRAJECTORY_JUDGE_PROMPT = """You are evaluating whether an AI agent made appropriate tool choices.

Available tools:
- retrieve: searches an internal document corpus covering AI governance guidelines and legislation from the UK, US and EU.
- web_search: searches the web for current or general information

Query: {query}

Tool calls made:
{tool_calls}

Score the appropriateness of the tool selection from 0.0 to 1.0.
1.0 = exactly the right tools were called with precise, relevant queries
0.5 = acceptable but suboptimal (unnecessary extra call, slightly imprecise query)
0.0 = clearly wrong (web search used for a corpus question, or no tools called for a question that needs retrieval)

Return a JSON object with exactly two keys: "score" (a float) and "reasoning" (a string).
"""


def judge_trajectory(query: str, tool_calls: list[dict]) -> dict:
    """Score one query's tool-call trajectory using an LLM judge.

    Fill TRAJECTORY_JUDGE_PROMPT and invoke a
    ChatVertexAI(model_name="gemini-2.5-flash", temperature=0,
    response_mime_type="application/json") with it. The response_mime_type
    is required: without it Gemini often wraps the JSON in a ```json fence,
    json.loads fails, and every query quietly falls back to the 0.5 default.
    Parse the response as JSON and return it; on a parse failure return
    {"score": 0.5, "reasoning": "Could not parse judge response."}.
    """
    judge = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, response_mime_type="application/json")
    formatted_calls = "\n".join(
        f"  {i + 1}. {tc['tool']}({json.dumps(tc['args'])})"
        for i, tc in enumerate(tool_calls)
    ) or "  (no tools called)"
    prompt = TRAJECTORY_JUDGE_PROMPT.format(query=query, tool_calls=formatted_calls)
    response = judge.invoke(prompt)
    try:
        return json.loads(response.content)
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {"score": 0.5, "reasoning": "Could not parse judge response."}


def run_trajectory_evaluation(dataset_path: str) -> list[dict]:
    """Run every question in the trajectory dataset through the agent and judge it.

    For each item, call ask_agent() then judge_trajectory(). Collect a dict
    per item with keys "question", "tool_calls", "trajectory_score",
    "reasoning", "answer". Print progress as you go.
    """
    with open(dataset_path, "r") as jsonfile:
        tests = json.load(jsonfile)

    outputs = []
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {test['question'][:70]}")
        agent_result = ask_agent(test["question"])
        judgement = judge_trajectory(test["question"], agent_result["tool_calls"])
        print(f"  tools={[tc['tool'] for tc in agent_result['tool_calls']]}  score={judgement['score']}")
        output = {
            "question": test["question"],
            "answer": agent_result["answer"],
            "tool_calls": agent_result["tool_calls"],
            "trajectory_score": judgement["score"],
            "reasoning": judgement["reasoning"],
        }
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    results = run_trajectory_evaluation("trajectory_dataset.json")

    scores = [r["trajectory_score"] for r in results]
    avg = sum(scores) / len(scores)
    print(f"\nAverage trajectory score: {avg:.2f}")
    print("Lowest scoring queries:")
    for r in sorted(results, key=lambda x: x["trajectory_score"])[:3]:
        print(f"  Score {r['trajectory_score']:.2f}: {r['question'][:70]}")
        print(f"  Tools: {[tc['tool'] for tc in r['tool_calls']]}")
        print(f"  Reason: {r['reasoning'][:150]}")
        print()

    with open("trajectory_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to trajectory_results.json")
