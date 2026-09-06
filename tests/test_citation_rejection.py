"""Citation rejection is not a retrieval score that a fallback may overrule."""
import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import answer


class CitationRejectionTests(unittest.TestCase):
    def setUp(self):
        self.draft = "检查控制器并更换组件。[^1]"
        self.item = {"evidence_id": "proof-a", "doc_type": "mfl",
                     "content": "检查控制器，但未确认需要更换。", "authority": "authoritative_record"}
        self.state = {"query": "如何处置？", "evidence_registry": [self.item]}








    def test_missing_source_relation_is_not_a_negative_api_fact(self):
        self.state["hmc_mfl_binding"] = {"required": True, "bound": False,
                                         "hmc_codes": ["73120408**0007"]}
        prompt = answer._answer_task_instruction(self.state)
        self.assertIn("unconfirmed_source_relation", prompt)
        self.assertIn("不能据此认定MFL没有记录或没有API", prompt)
