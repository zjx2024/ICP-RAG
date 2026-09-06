import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.answer_context_binding import (
    conversation_review_context,
    clarification_review_observations,
)
from backend.agent.clarification_contract import make_clarification_reason, make_clarification_identity, make_missing_input_requirement
from backend.agent.nodes import answer
from backend.agent.nodes.answer_text import strip_generated_footnote_definitions


class ConversationBindingTests(unittest.TestCase):
    def setUp(self):
        self.state = {"query": "你先前讲了哪些部分？", "messages": [
            {"role": "user", "content": "介绍组成"},
            {"role": "assistant", "content": "列出了液冷回路、信号变换单元和外壳组件。[^3]"},
        ]}
        self.review = {"kind": "conversation_reference", "asserted_engineering_fact": False,
                       "bindings": [], "history_bindings": [{"message_id": "h1", "quote": "液冷回路、信号变换单元和外壳组件"}]}

    def test_bounded_context_keeps_stable_ids_citations_and_explicit_truncation(self):
        messages = [{"role": "assistant", "content": "x" * 7000 + "[^9]"} for _ in range(12)]
        context = conversation_review_context({"messages": messages})
        self.assertLessEqual(sum(len(m["content"]) for m in context["messages"]), 18000)
        self.assertEqual("h11", context["messages"][-1]["message_id"])
        self.assertTrue(context["messages"][-1]["truncated"])
        self.assertIn("[^3]", conversation_review_context(self.state)["messages"][-1]["content"])




    def test_public_citation_compaction_keeps_reviewed_context_without_fabricating_a_reference(self):
        line = "上轮已经列出了液冷回路。"
        state = {"_reviewed_context_units": [line], "safety_decision": {"approved_evidence_ids": []}}
        compacted, citations, _ = answer._compact_answer_citations(line, [], state)
        self.assertEqual(line, compacted)
        self.assertEqual([], citations)


if __name__ == "__main__":
    unittest.main()
