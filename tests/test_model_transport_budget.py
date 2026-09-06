import unittest

from backend.agent.model_runtime_policy import bounded_model_transport
from backend.agent.nodes import plan, answer, scope_router, adaptive_tools


class ModelTransportBudgetTests(unittest.TestCase):
    def test_role_output_and_transport_limits_apply_to_both_providers(self):
        roles = [(plan._prepare_plan_model, plan.PLAN_LLM_TIMEOUT_SECONDS),
                 (plan._prepare_target_model, plan.TARGET_LLM_TIMEOUT_SECONDS),
                 (answer._prepare_answer_model, answer.ANSWER_LLM_TIMEOUT_SECONDS),
                 (answer._prepare_answer_review_model, answer.ANSWER_REVIEW_TIMEOUT_SECONDS),
                 (scope_router._prepare_scope_model, scope_router.SCOPE_ROUTER_TIMEOUT_SECONDS),
                 (adaptive_tools._prepare_model, adaptive_tools.ADAPTIVE_TOOL_TIMEOUT_SECONDS)]
        for provider in ("ollama", "qwen"):
            for prepare, budget in roles:
                with self.subTest(provider=provider, role=prepare.__name__):
                    original = {"provider": provider, "timeout": 3000, "max_tokens": 262144}
                    prepared = prepare(original)
                    self.assertEqual(budget, prepared["timeout"])
                    self.assertEqual(prepared["num_predict"], prepared["max_tokens"])
                    self.assertEqual(3000, original["timeout"])

    def test_shorter_provider_deadline_is_not_widened(self):
        self.assertEqual(2, bounded_model_transport({"timeout": 2}, 20)["timeout"])
        for value in (None, "bad", float("inf"), 0, -1):
            self.assertEqual(10, bounded_model_transport({"timeout": value}, 10)["timeout"])
        self.assertIsNone(bounded_model_transport(None, 10))
