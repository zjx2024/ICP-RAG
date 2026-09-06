import asyncio
import json
import unittest
from unittest.mock import patch
from backend.agent.nodes import plan


class TargetContractRetryTests(unittest.TestCase):
    def test_closed_focus_labels_do_not_create_a_second_semantic_retry(self):
        bad = {"primary_intent": "exact_code", "targets": ["exact_code"],
            "required_answer_fields": ["hmc_identity"], "response_focus": ["maintenance_action"]}
        with patch.object(plan.llm_service, "call_model", return_value=json.dumps(bad)) as call:
            result = self.classify()
        call.assert_called_once()
        self.assertEqual(["hmc_identity"], result["required_answer_fields"])
        self.assertEqual([], result["calculation_authorizations"])

    def test_source_limitation_contract_is_not_forced_to_claim_missing_fields(self):
        decision = {"primary_intent": "reliability_query", "targets": ["reliability_query"],
            "required_answer_fields": ["rbd_path_structure"], "response_focus": ["cause", "source_limitation"]}
        with patch.object(plan.llm_service, "call_model", return_value=json.dumps(decision)) as call:
            result = self.classify()
        call.assert_called_once()
        self.assertNotIn("fault_cause", result["required_answer_fields"])

    def test_structural_cause_is_compatible_without_requiring_an_individual_cause_field(self):
        decision = {"primary_intent": "cause_analysis", "targets": ["cause_analysis"],
            "required_answer_fields": ["structural_cause_chain"], "response_focus": ["cause"]}
        with patch.object(plan.llm_service, "call_model", return_value=json.dumps(decision)) as call:
            result = self.classify()
        call.assert_called_once()
        self.assertEqual(["structural_cause_chain"], result["required_answer_fields"])

    def classify(self):
        return asyncio.run(plan._classify_explicit_targets(query="分析这张图", history="历史", conversation_anchors=[], model={"provider": "ollama"}))

    def test_truncated_contract_has_one_retry_with_original_deadline(self):
        good = json.dumps({"primary_intent": "reliability_query", "targets": ["reliability_query"]})
        with patch.object(plan.llm_service, "call_model", side_effect=['{"targets":[', good]) as call, patch.object(
            plan.asyncio, "wait_for", wraps=asyncio.wait_for) as bounded:
            result = self.classify()
        self.assertEqual("reliability_query", result["primary_intent"])
        self.assertEqual(2, call.call_count)
        self.assertEqual([], result["calculation_authorizations"])
        self.assertLess(bounded.call_args_list[1].kwargs["timeout"], bounded.call_args_list[0].kwargs["timeout"])

    def test_semantic_rejection_empty_or_transport_failure_is_not_retried(self):
        for output in ("", "not JSON"):
            with self.subTest(output=output), patch.object(plan.llm_service, "call_model", return_value=output) as call:
                self.assertIsNone(self.classify())
            call.assert_called_once()

    def test_unknown_ontology_value_gets_one_model_correction_not_a_code_alias(self):
        good = json.dumps({"primary_intent": "reliability_query", "targets": ["reliability_query"]})
        with patch.object(plan.llm_service, "call_model", side_effect=['{"targets":["calculation_authorizations"]}', good]) as call:
            result = self.classify()
        self.assertEqual("reliability_query", result["primary_intent"])
        self.assertEqual([], result["calculation_authorizations"])
        self.assertIn("字段名称、计算操作名称不是意图值", call.call_args.args[0][0]["content"])
