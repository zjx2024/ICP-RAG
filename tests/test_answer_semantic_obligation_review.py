"""Discourse review and delivery status must not depend on fixed user words."""
import asyncio
import unittest
from unittest.mock import patch

from backend.agent.nodes import answer
from backend.agent.answer_context_binding import conversation_review_context


class AnswerSemanticObligationReviewTests(unittest.TestCase):





    def test_shared_history_retains_late_detail_and_namespace_without_live_markers(self):
        messages = [{"role": "assistant", "content": "甲" * 700 + "后文包含乙组件。[^9]"}]
        rendered = answer._format_history(messages)
        review = conversation_review_context({"messages": messages})
        self.assertIn("后文包含乙组件", rendered)
        self.assertIn("后文包含乙组件", review["messages"][0]["content"])
        self.assertIn("历史引用 h0:9", rendered)
        self.assertNotIn("[^9]", rendered)
        self.assertIn("[^9]", review["messages"][0]["content"])
        self.assertFalse(review["messages"][0]["truncated"])

    def test_history_budget_and_explicit_truncation_still_apply(self):
        messages = [{"role": "assistant", "content": "字" * 7000} for _ in range(30)]
        view = conversation_review_context({"messages": messages})
        self.assertLessEqual(len(view["messages"]), 16)
        self.assertLessEqual(sum(len(item["content"]) for item in view["messages"]), 18000)
        self.assertTrue(view["messages"][-1]["truncated"])
        self.assertEqual("h29", view["messages"][-1]["message_id"])
        self.assertIn('"truncated": true', answer._format_history(messages))
