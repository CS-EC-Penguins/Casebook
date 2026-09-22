"""Offline tests for the graph agent's structured final answer."""

import unittest

from graph_agent import (
    Citation,
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
            citations=[Citation(id=1, source="nist-ai-rmf.txt", type="document")],
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

    async def test_rejects_citation_not_in_tool_evidence(self):
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="An answer. [1]",
            citations=[Citation(id=1, source="invented.txt", type="document")],
        ))

        with self.assertRaisesRegex(ValueError, "not returned by a tool"):
            await structure_final_answer(self.run, model)

    async def test_rejects_wrong_citation_number(self):
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="An answer. [2]",
            citations=[Citation(id=1, source="nist-ai-rmf.txt", type="document")],
        ))

        with self.assertRaisesRegex(ValueError, "do not match"):
            await structure_final_answer(self.run, model)

    async def test_web_source_has_web_type(self):
        self.run["contexts"] = ["[Source: https://example.org/news]\nCurrent news."]
        model = FakeModel(StructuredAnswer(
            status="answered",
            answer="Current news. [1]",
            citations=[Citation(id=1, source="https://example.org/news", type="web")],
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
