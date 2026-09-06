import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.citation_binding import (
    MAX_QUOTE_RECORD_CHARS,
    quote_field_view,
    unit_record_requirements,
)
from backend.agent.nodes import answer


class CitationFieldBindingTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "record_id": "FUNC-731", "evidence_id": "record-a", "doc_type": "function_locator",
            "claim_type": "function_fact", "title": "事件历史记录",
            "content": "保存各状态快照并记录事件时间。",
            "metadata": {"answer_fields": {
                "function_id": "FUNC-731", "module_name": "事件快照存储模块",
                "description": "保存各状态快照并记录事件时间。",
            }},
        }
        self.ledger = {1: self.item}
        self.binding = {"citation_id": 1, "field": "/answer_fields/module_name", "quote": "事件快照存储模块"}


    def test_model_receives_closed_identity_requirements_without_question_keywords(self):
        ledger = {**self.ledger, 5: {**self.item, "record_id": "FUNC-947"}}
        self.assertEqual([{"record_id": "FUNC-731", "citation_ids": [1]}], unit_record_requirements("FUNC-731 对应这个名称。", ledger))
        self.assertEqual([], unit_record_requirements("FUNC-7319 或 revision-FUNC-731", ledger))
        self.assertEqual(2, len(unit_record_requirements("对比FUNC-731与FUNC-947", ledger)))

    def test_hmc_symptom_identity_does_not_offer_hidden_maintenance_fields(self):
        item = {**self.item, "doc_type": "hmc", "claim_type": "hmc_fault", "metadata": {"answer_fields": {
            "fault_symptom": "输入钳位异常", "maintenance_action": "检查连接", "hsc": "0007",
        }}}
        fields = quote_field_view(item)["fields"]
        self.assertNotIn("/answer_fields/maintenance_action", fields)
        self.assertEqual("0007", fields["/answer_fields/hsc"])

    def test_long_payload_cannot_displace_the_business_identity(self):
        item = {**self.item, "content": "正文" * 9000, "metadata": {"answer_fields": {"notes": "附注" * 10000}}}
        view = quote_field_view(item)
        self.assertEqual("FUNC-731", view["fields"]["/record_id"])
        self.assertTrue(view["truncated_fields"])
        self.assertLessEqual(sum(map(len, view["fields"].values())), MAX_QUOTE_RECORD_CHARS)

    def test_name_obligation_preserves_names_at_each_explicit_source_level(self):
        from backend.agent.nodes.answer_field_catalog import _TARGET_COMPLETENESS_FIELD_KEYS

        # The coarse semantic facet must not erase the concrete name the user
        # asked for; choosing that level belongs to the model, not a phrase rule.
        self.assertTrue({"system_name", "subsystem_name", "module_name", "function_name", "component_name"}.issubset(
            _TARGET_COMPLETENESS_FIELD_KEYS["record_name"]
        ))


if __name__ == "__main__":
    unittest.main()
