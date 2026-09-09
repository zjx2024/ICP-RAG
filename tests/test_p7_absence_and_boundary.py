"""Boundary review uses source context, not mandatory prompt numbering."""
import json
import unittest

from backend.agent.answer_review import AnswerReviewError, prepare_review, review_input, validate_review
from tests.test_answer_integrated_review import response, issue


class BoundaryReviewTests(unittest.TestCase):
    def inputs(self, draft):
        ledger = {1: {"record_id": "M-17", "doc_type": "mfl", "content": "人工检查连接；直接上报为否"}}
        observations = [{"observation_id": "relation", "category": "source_relation",
                         "fields": {"bound": "false", "meaning": "未确认编码对应关系，不证明接口不存在"}}]
        prepared = prepare_review(draft, ledger=ledger, conversation={}, observations=observations,
            task={"original_question": "维修清单有对应接口吗？"}, document_semantics="产品类别不替代维修程序")
        return dict(prepared=prepared, ledger=ledger, safety={}, calculation_requests=[])

    def test_method_and_limit_reach_one_record_view(self):
        args = self.inputs("说明")
        view = review_input(args["prepared"], args["ledger"])
        self.assertIn("人工检查连接；直接上报为否", json.dumps(view, ensure_ascii=False))
        self.assertNotIn("units", view)
        self.assertNotIn("sources", view)

    def test_unknown_relation_does_not_erase_method_or_become_absence(self):
        args = self.inputs("对应关系尚未确认，不等于没有人工检查方法。")
        view = review_input(args["prepared"], args["ledger"])
        self.assertEqual("source_relation", view["observations"][0]["category"])
        self.assertEqual(args["prepared"]["draft"], validate_review(json.dumps(response()), **args)["text"])
        invalid = self.inputs("维修清单没有任何处理方法。")
        with self.assertRaises(AnswerReviewError):
            validate_review(json.dumps(response(verdict="revise", issues=[
                issue(quote=invalid["prepared"]["draft"], reason="当前来源已有人工检查，不能否定") ])), **invalid)

    def test_policy_boundary_needs_no_invented_engineering_ref(self):
        args = self.inputs("产品类别说明不能代替维修程序。")
        result = validate_review(json.dumps(response()), **args)
        self.assertNotIn("[^", result["text"])
        self.assertEqual([], result["manifest"][0]["evidence_ids"])

    def test_revision_prompt_does_not_encode_a_protocol_error_dictionary(self):
        from backend.agent.nodes.answer import REVISION_PROMPT_HEADER
        for code in ("unsupported_unit", "unknown_source_handle", "wrong_nonengineering_provenance"):
            self.assertNotIn(code, REVISION_PROMPT_HEADER)
