"""Query observations carry scope; retired terminal decisions grant no authority."""
import json
import re
import unittest

from backend.agent.current_model_selection import model_candidates, selection_observation, validate_selection
from backend.agent.nodes.answer import _answer_task_instruction, _answer_review_material


def contract(state):
    instruction = _answer_task_instruction(state)
    return json.loads(instruction.split("\n", 1)[1].split("\n", 1)[0])


class SourceBoundaryTests(unittest.TestCase):
    def test_explicit_reference_is_not_a_derived_or_unconfirmed_relation(self):
        state = {"query": "这个报码有处理办法吗？", "hmc_mfl_binding": {
            "required": True, "bound": True, "basis": "explicit_authoritative_reference",
            "hmc_codes": ["87654321**0012"]}}
        value = contract(state)
        self.assertNotIn("derived_source_relation", value)
        self.assertNotIn("unconfirmed_source_relation", value)

    def test_unbound_relation_does_not_claim_mfl_or_api_absence(self):
        state = {"query": "它有接口吗？", "hmc_mfl_binding": {
            "required": True, "bound": False, "hmc_codes": ["87654321**0012"]}}
        value = contract(state)["unconfirmed_source_relation"]
        self.assertIn("不能据此认定", value["reason"])

    def test_old_terminal_state_and_manifest_cannot_restore_negative_fact(self):
        state = {"query": "现在改查另一个部件", "messages": [{
            "role": "assistant", "content": "旧对象没有直接模型。",
            "answer_delivery_manifest": [{"kind": "terminal_negation_anchor", "doc_type": "rbd",
                "verified_universe": ["revision_id:old"], "subject": "旧对象"}]}],
            "rbd_query_binding": {"required": True, "bound": False,
                "basis": "current_model_selection_not_represented",
                "current_selection": {"status": "not_represented"}}}
        value = contract(state)
        self.assertNotIn("rbd_not_represented", value)
        self.assertEqual([], value["model_selection"])
        _, _, observations = _answer_review_material(state)
        self.assertFalse(any(o.get("category") in {"rbd_representation", "model_selection"} for o in observations))

    def test_current_unmatched_view_can_explain_only_the_query_result(self):
        state = {"query": "这个部件有没有模型？"}
        rows = [{"doc_id": "source-a", "revision_id": "revision-a", "title": "装置模型"}]
        selection = validate_selection({"status": "not_represented", "question_basis": "这个部件",
            "selections": [], "reason": "未从候选中选中"}, model_candidates(rows, "rbd"), query=state["query"])
        state["rbd_query_binding"] = {"required": True, "bound": False,
            "selection_observation": selection_observation(selection, rows, doc_type="rbd", query=state["query"])}
        self.assertFalse(contract(state)["model_selection"][0]["exhaustive"])
        _, _, observations = _answer_review_material(state)
        self.assertTrue(any(o.get("category") == "model_selection" for o in observations))
        state["query"] = "换一个对象"
        self.assertEqual([], contract(state)["model_selection"])

    def test_explain_semantics_does_not_force_a_capability_answer(self):
        state = {"query": "为什么这条链的备用支路还能工作？",
                 "query_frame": {"question_semantics": "explain"},
                 "rbd_query_binding": {"bound": True, "basis": "current_model_selection"}}
        self.assertNotIn("rbd_answer_compactness", contract(state))
