import copy
import unittest

from backend.agent.citation_binding import quote_field_view
from backend.agent.nodes import answer
from backend.agent.nodes.answer_source_fields import source_owned_field_view


class SourceOwnedFieldTests(unittest.TestCase):
    def item(self, doc_type="hmc"):
        return {"evidence_id": "source-1", "doc_id": "doc-1", "doc_type": doc_type,
                "record_id": "record-1", "claim_type": "maintenance_action",
                "content": "原文维修处置：核查后更换。", "metadata": {"answer_fields": {
                    "record_code": "record-1", "maintenance_action": "核查后更换。",
                    "repair_method": "复测端口。", "fault_symptom": "输入异常",
                    "api_name": "ReadState", "api_usage": "读取状态位",
                    "compensation_action": "切换备用通路",
                }}}

    def state(self, item):
        return {"query": "那份指定资料究竟提供了什么？", "evidence_registry": [item],
                "safety_decision": {"approved_evidence_ids": ["source-1"]},
                "query_frame": {"required_answer_fields": ["maintenance_action", "maintenance_api"],
                    "source_obligations": [{"doc_type": "mfl", "fields": ["maintenance_action", "maintenance_api"]}]}}

    def test_all_answer_consumers_exclude_nonowner_fields_without_mutating_source(self):
        item = self.item()
        original = copy.deepcopy(item)
        state = self.state(item)
        contexts = [answer._context_evidence(state)[0], answer._citation_ledger(state)[1]]
        for view in contexts:
            for key in ("maintenance_action", "repair_method", "api_name", "compensation_action"):
                self.assertNotIn(key, view["metadata"]["answer_fields"])
            self.assertEqual("record-1", view["metadata"]["answer_fields"]["record_code"])
            self.assertNotIn("/content", quote_field_view(view)["fields"])
        self.assertNotIn("核查后更换", answer._format_context(state))
        self.assertEqual(original, item)

    def test_all_seven_types_use_contract_not_question_keywords(self):
        types = ("function_locator", "hmc", "fmeca", "fta", "rbd", "mfl", "fault_history")
        for owner in types:
            for source in types:
                for query in ("出处在哪？", "What does that source say?", "请核对它的记载"):
                    item = self.item(source)
                    state = self.state(item)
                    state["query"] = query
                    state["query_frame"]["source_obligations"][0]["doc_type"] = owner
                    view = source_owned_field_view(item, state)
                    if "maintenance_action" in view["metadata"].get("source_field_exclusions", []):
                        # 字段被剥除只允许发生在“非属主”行上；MFL 合同字段的
                        # 属主集合永远包含 MFL 本身。
                        self.assertNotEqual("mfl", source)
                    if owner == source or (owner != "mfl" and source == "mfl"):
                        # 属主自留；或维修合同字段被窄化到非 MFL 属主时，
                        # MFL 行仍保留权威字段。
                        self.assertIn("maintenance_action", view["metadata"]["answer_fields"])

    def test_shared_alias_preserved_for_independent_dimension(self):
        item = self.item("fmeca")
        state = self.state(item)
        state["query_frame"]["required_answer_fields"].append("compensation_action")
        view = source_owned_field_view(item, state)
        self.assertIn("compensation_action", view["metadata"]["answer_fields"])
        self.assertNotIn("maintenance_action", view["metadata"]["answer_fields"])

    def test_mixed_sources_and_multiple_owners_are_not_whole_turn_locks(self):
        item = self.item()
        state = self.state(item)
        state["query_frame"]["source_obligations"] = [
            {"doc_type": "hmc", "fields": ["maintenance_action"]},
            {"doc_type": "mfl", "fields": ["maintenance_api"]},
        ]
        view = source_owned_field_view(item, state)
        self.assertIn("maintenance_action", view["metadata"]["answer_fields"])
        self.assertNotIn("api_name", view["metadata"]["answer_fields"])
        state["query_frame"]["source_obligations"].append({"doc_type": "hmc", "fields": ["maintenance_api"]})
        self.assertEqual(item, source_owned_field_view(item, state))

    def test_missing_invalid_and_empty_contract_does_not_invent_owner(self):
        item = self.item()
        for obligations in ([], [{"doc_type": "invented", "fields": ["maintenance_action"]}], [{"doc_type": "mfl", "fields": []}]):
            state = self.state(item)
            state["query_frame"]["source_obligations"] = obligations
            self.assertIs(item, source_owned_field_view(item, state))
