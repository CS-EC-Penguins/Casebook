"""
Pytest assertions on the agent's tool-routing behaviour.

Requires your completed graph_agent.py (Day 3) in the same directory.

Usage:
    pytest test_trajectories.py -v
"""
from graph_agent import ask_agent


def test_corpus_question_calls_retrieve():
    """A question answerable from the sample corpus should call `retrieve`."""
    # Property: whenever the question is about a topic covered by the internal
    # corpus (coral reef ecology), the agent must use `retrieve` rather than
    # going to the web. This should always hold because the system prompt
    # instructs the model to prefer the internal tool for corpus topics, and
    # "coral bleaching" is unambiguously in the knowledge base.
    result = ask_agent("What are the obligations for providers of high-risk AI systems under the EU AI Act?")
    assert "retrieve" in [tc["tool"] for tc in result["tool_calls"]]


def test_current_events_calls_web_search():
    """A question about very recent events should call `web_search`."""
    # Property: whenever the question asks for information that is inherently
    # time-bound and cannot appear in a static corpus ("this week's news"),
    # the agent must route to `web_search`. This should always hold because no
    # static document can contain future or very recent events, so the model
    # has no choice but to reach for the live search tool.
    result = ask_agent("What were the main news stories in science this week?")
    assert "web_search" in [tc["tool"] for tc in result["tool_calls"]]


def test_multi_part_corpus_question_retrieves_multiple_times():
    """A question spanning two corpus topics should retrieve at least once,
    and the answer should cover both topics."""
    # Property: a question that spans two distinct corpus topics (bat
    # echolocation and bee navigation) must (a) call `retrieve` to ground the
    # answer in the corpus rather than hallucinate, and (b) produce an answer
    # that actually addresses both topics. We assert on the outcome (keywords
    # in the answer) rather than the call count, because the agent may satisfy
    # both topics in a single retrieval if top-k results happen to span both
    # documents -- asserting len(retrieve_calls) >= 2 would be brittle.
    result = ask_agent("Compare the findings across the UK AISI Report and the Meridian governance report.")
    assert "retrieve" in [tc["tool"] for tc in result["tool_calls"]]
    answer_lower = result["answer"].lower()
    assert "governance" in answer_lower, "Answer should mention governance"
    assert "human oversight" in answer_lower, "Answer should mention human oversight"
