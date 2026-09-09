"""Adversarial identities and query scopes, independent of the 19-case gold."""
import random
import asyncio
import unittest
from unittest.mock import patch

from backend.agent.current_model_selection import model_candidates, validate_selection, selection_observation
from backend.agent.rbd_contract import bind_rbd_models, validate_rbd_binding_contract
from backend.agent.tools.icp_diagnosis import _bind_mfl_rows_to_hmc
from backend.agent.nodes.evidence_gate import evidence_gate_node
from scripts.evaluate_icp_paraphrases import _judge_quote_supported


class EvidenceBoundaryRegressions(unittest.TestCase):
    def test_explicit_other_hmc_cannot_be_overridden_by_equal_description(self):
        rng = random.Random(8407)
        for _ in range(12):
            prefix = str(rng.randrange(10000000, 99999999))
            first, other = prefix + "**0001", prefix + "**0002"
            with self.subTest(code=first), patch(
                "backend.agent.tools.icp_diagnosis._hmc_structured_component_identity",
                return_value={"MOD": "测试模块"}, create=True,
            ):
                rows, binding = _bind_mfl_rows_to_hmc(
                    [{"hmc_code": first, "match_type": "exact_components",
                      "fault_desc": "MOD 状态反馈异常"}],
                    [{"mfl_code": "maintenance-row", "hmc_code": other,
                      "module_fault": "测试模块：状态反馈异常"}],
                )
                self.assertFalse(binding["bound"])
                self.assertTrue(rows[0]["query_binding_excluded"])

    def test_equal_text_without_explicit_relation_is_not_identity(self):
        with patch("backend.agent.tools.icp_diagnosis._hmc_structured_component_identity",
                   return_value={"MOD": "测试模块"}, create=True):
            _, binding = _bind_mfl_rows_to_hmc(
                [{"hmc_code": "87654321**0012", "match_type": "exact_components",
                  "fault_desc": "MOD 状态反馈异常"}],
                [{"mfl_code": "maintenance-row", "module_fault": "测试模块：状态反馈异常"}],
            )
        self.assertFalse(binding["bound"])

    def test_explicit_matching_relation_still_works(self):
        rows, binding = _bind_mfl_rows_to_hmc(
            [{"hmc_code": "87654321**0012", "match_type": "exact_components"}],
            [{"mfl_code": "maintenance-row", "hmc_code": "87654321**0012"}],
        )
        self.assertTrue(binding["bound"])
        self.assertNotIn("query_binding_excluded", rows[0])

    def test_negative_model_selection_is_never_formal_authority(self):
        row = {"doc_id": "doc-a", "revision_id": "rev-a", "title": "模型甲"}
        query = "这个滤波部件有没有直接模型？"
        selection = validate_selection(
            {"status": "not_represented", "question_basis": "滤波部件",
             "selections": [], "reason": "当前候选中未识别到"},
            model_candidates([row], "rbd"), query=query,
        )
        for current_query, rows in ((query, [row]), ("改查显示链", [row]), (query, [])):
            with self.subTest(query=current_query, rows=rows):
                selected, binding = bind_rbd_models(
                    rows, required=True, raw_user_query=current_query,
                    current_selection=selection,
                )
                self.assertEqual([], selected)
                self.assertFalse(validate_rbd_binding_contract(binding)["valid"])
                self.assertNotEqual("current_model_selection_not_represented", binding["basis"])

    def test_judge_preserves_business_identifier_characters(self):
        for quote, answer in (
            ("HMC 87654321**0012", "HMC 876543210012"),
            ("Get_Diagnostic_Status", "GetDiagnosticStatus"),
        ):
            with self.subTest(quote=quote):
                self.assertFalse(_judge_quote_supported(quote, answer))

    def test_negative_observation_is_current_and_bounded_to_offered_view(self):
        row = {"doc_id": "doc-random", "revision_id": "rev-31", "title": "装置模型"}
        query = "这个部件有没有模型？"
        selection = validate_selection(
            {"status": "not_represented", "question_basis": "这个部件",
             "selections": [], "reason": "候选里没选中"},
            model_candidates([row], "rbd"), query=query,
        )
        observation = selection_observation(selection, [row], doc_type="rbd", query=query)
        self.assertEqual("no_model_selected", observation["status"])
        self.assertFalse(observation["exhaustive"])
        for current_query, rows in (
            ("换成另一个对象", [row]),
            (query, []),
            (query, [{**row, "title": "内容已经修订"}]),
            (query, [row, {**row, "revision_id": "rev-32"}]),
        ):
            with self.subTest(query=current_query, rows=rows):
                self.assertEqual({}, selection_observation(selection, rows, doc_type="rbd", query=current_query))

    def test_judge_cannot_skip_negation_or_scope_with_ellipsis(self):
        self.assertFalse(_judge_quote_supported(
            "可以...更换模块", "可以查询状态，但不能更换模块"))

    def test_unmatched_query_is_not_forced_identity_input_or_formal_authority(self):
        row = {"doc_id": "doc-a", "revision_id": "rev-a", "title": "装置模型"}
        query = "这个部件有没有直接模型？"
        for status in ("not_represented", "ambiguous"):
            selection = validate_selection({"status": status, "question_basis": "这个部件",
                "selections": [], "reason": "候选选择结果"}, model_candidates([row], "rbd"), query=query)
            _, binding = bind_rbd_models([row], required=True, raw_user_query=query, current_selection=selection)
            result = asyncio.run(evidence_gate_node({"query": query, "answer_mode": "knowledge_explanation",
                "allowed_tools": ["rbd_query"], "rbd_query_binding": binding, "evidence_registry": []}))
            decision = result["safety_decision"]
            slots = [item["slot"] for item in decision["missing_input_requirements"]]
            self.assertEqual(status == "ambiguous", "rbd_model_identity" in slots)
            self.assertIn("formal_calculation", decision["blocked_claims"])
            self.assertIn("formal_topology", decision["blocked_claims"])


if __name__ == "__main__":
    unittest.main()
