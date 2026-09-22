"""Offline tests for the graph agent's structured final answer."""

import unittest
from typing import Annotated, Literal, TypedDict

from graph_agent import (
    ITERATION_LIMIT_ANSWER,
    StructuredAnswer,
    structure_final_answer,
)


class FakeModel:
    def __init__(self, answer):
        self.answer = answer
        self.messages = None

    def with_structured_output(self, schema):
        assert schema is StructuredAnswer
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.answer


class StructuredAnswerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.run = {
            "question": "What are the Core functions?",
            "answer": "GOVERN, MAP, MEASURE, and MANAGE.",
            "tool_calls": [{"tool": "retrieve", "args": {"query": "AI RMF Core"}}],
            "contexts": ["[Source: nist-ai-rmf.txt]\nThe Core has four functions."],
            "iterations": 2,
        }

    async def test_adds_validated_answer_without_changing_existing_fields(self):
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="The four functions are GOVERN, MAP, MEASURE, and MANAGE. [1]",
        ))

        result = await structure_final_answer(self.run, model)

        self.assertEqual(
            list(result),
            ["question", "status", "answer", "citations", "tool_calls", "contexts", "iterations"],
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(
            result["citations"],
            [{"id": 1, "source": "nist-ai-rmf.txt", "type": "document"}],
        )
        self.assertEqual(result["tool_calls"], self.run["tool_calls"])
        self.assertEqual(result["contexts"], self.run["contexts"])
        self.assertIn("The Core has four functions", model.messages[-1].content)
        self.assertIn("[1] nist-ai-rmf.txt (document)", model.messages[-1].content)

    async def test_unknown_citation_number_returns_insufficient_evidence(self):
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="An answer. [2]",
        ))

        result = await structure_final_answer(self.run, model)

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])

    async def test_missing_citation_returns_insufficient_evidence(self):
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="An answer without a citation.",
        ))

        result = await structure_final_answer(self.run, model)

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])

    async def test_citations_are_renumbered_in_order_of_first_use(self):
        self.run["contexts"] = [
            "[Source: first.txt]\nFirst finding.\n\n"
            "[Source: second.txt]\nSecond finding."
        ]
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="Second finding [2]. First finding [1]. Second finding again [2].",
        ))

        result = await structure_final_answer(self.run, model)

        self.assertEqual(
            result["answer"],
            "Second finding [1]. First finding [2]. Second finding again [1].",
        )
        self.assertEqual(result["citations"], [
            {"id": 1, "source": "second.txt", "type": "document"},
            {"id": 2, "source": "first.txt", "type": "document"},
        ])

    async def test_web_source_has_web_type(self):
        self.run["contexts"] = ["[Source: https://example.org/news]\nCurrent news."]
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="Current news. [1]",
        ))

        result = await structure_final_answer(self.run, model)

        self.assertEqual(result["citations"][0]["type"], "web")

    async def test_no_evidence_needs_no_extra_model_call(self):
        self.run["contexts"] = ["No relevant passages found."]
        model = FakeModel(None)

        result = await structure_final_answer(self.run, model)

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(model.messages)

    async def test_iteration_limit_needs_no_extra_model_call(self):
        self.run["answer"] = ITERATION_LIMIT_ANSWER
        model = FakeModel(None)

        result = await structure_final_answer(self.run, model)

        self.assertEqual(result["status"], "iteration_limit")
        self.assertIsNone(model.messages)


if __name__ == "__main__":
    unittest.main()
