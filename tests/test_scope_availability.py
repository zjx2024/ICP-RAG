"""Classifier unavailability must not masquerade as missing user context."""
import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import scope_router
from backend.agent.api.chat_router import _scope_stage_event
from scripts import evaluate_icp_paraphrases as evaluator


class ScopeAvailabilityTests(unittest.TestCase):
    def test_model_failures_have_typed_availability_and_no_retrieval_authority(self):
        scenarios = (("invalid_output", "not JSON"),
                     ("timeout", asyncio.TimeoutError()),
                     ("model_error", RuntimeError("private server detail")))
        for kind, response in scenarios:
            with self.subTest(kind=kind), patch.object(
                scope_router, "_model_for_state", return_value={},
            ), patch.object(scope_router.llm_service, "call_model") as model, patch(
                "backend.agent.tracing.record_degradation",
            ) as degradation:
                if isinstance(response, Exception):
                    model.side_effect = response
                else:
                    model.return_value = response
                result = asyncio.run(scope_router.scope_router_node({"query": "那为什么？"}))
            self.assertEqual(kind, result["scope_failure_kind"])
            self.assertEqual([], result["allowed_tools"])
            self.assertEqual("safe_fallback", result["scope_route_source"])
            self.assertIn("入口模型", result["clarification_question"])
            self.assertNotIn("请补充", result["clarification_question"])
            self.assertNotIn("private server detail", result["clarification_question"])
            degradation.assert_called_once_with("scope_router", detail=kind)

    def test_telemetry_failure_cannot_turn_safe_fallback_into_crash(self):
        with patch.object(scope_router, "_model_for_state", return_value={}), patch.object(
            scope_router.llm_service, "call_model", return_value="invalid",
        ), patch("backend.agent.tracing.record_degradation", side_effect=OSError("trace disk unavailable")):
            result = asyncio.run(scope_router.scope_router_node({"query": "当前问题"}))
        self.assertEqual("invalid_output", result["scope_failure_kind"])

    def test_real_model_clarification_is_not_an_availability_failure(self):
        response = json.dumps({"route": "clarification", "clarification_question": "请说明所指对象。"})
        with patch.object(scope_router, "_model_for_state", return_value={}), patch.object(
            scope_router.llm_service, "call_model", return_value=response,
        ):
            result = asyncio.run(scope_router.scope_router_node({"query": "它呢？"}))
        event = _scope_stage_event(result)
        self.assertEqual("completed", event["status"])
        self.assertNotIn("failure_kind", event)

    def test_public_event_keeps_failure_type_without_private_reason(self):
        event = _scope_stage_event({
            "scope_route": "clarification", "scope_route_source": "safe_fallback",
            "scope_failure_kind": "timeout", "scope_route_reason": "private exception and credentials",
        })
        self.assertEqual("degraded", event["status"])
        self.assertEqual("timeout", event["failure_kind"])
        self.assertNotIn("private", json.dumps(event))

    def test_graceful_scope_failure_is_an_evaluation_runtime_error(self):
        case = evaluator.EvaluationCase(1, "case", "group", "base", "它为什么？", {}, {})
        turn = evaluator.EvaluationTurn(1, "turn", "它为什么？", {}, {})
        result = evaluator._build_turn_result(
            case, turn, include_raw_events=False,
            events=[{"type": "stage", "stage": "scope_routing", "status": "degraded",
                     "route": "clarification", "failure_kind": "timeout"},
                    {"type": "answer"}, {"type": "done"}],
            runtime_errors=[], answer="入口模型暂时不可用，请重试。",
            citations=[], evidence_catalog=[], used_evidence_ids=[],
            done_received=True, answer_received=True, started_at="start", duration_seconds=20,
        )
        self.assertFalse(result["passed"])
        self.assertEqual("scope_router_timeout", result["runtime_errors"][0]["kind"])
        summary = evaluator._evaluation_breakdowns([{"turns": [result]}])["overall"]
        self.assertEqual(0, summary["evaluable_turns"])
        self.assertEqual(1, summary["runtime_error_turns"])


if __name__ == "__main__":
    unittest.main()
