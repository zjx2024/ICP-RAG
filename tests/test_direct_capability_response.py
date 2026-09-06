import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import direct_response as node


class CapabilityResponseTests(unittest.TestCase):
    def test_natural_followups_use_current_question_and_bounded_history(self):
        for query in ("那些记录在什么时候有用？", "你会在什么条件下做定量分析？", "And what needs human review?"):
            state = {"query": query, "scope_route": "agent_capability", "agent_id": "agent",
                     "messages": [{"role": "assistant", "content": "可查历史与可靠性资料。"}]}
            with self.subTest(query=query), patch.object(node.agent_model_role_service, "get_model_by_role", return_value={"provider": "ollama"}), patch.object(
                node.agent_model_role_service, "apply_agent_llm_params", side_effect=lambda s, m: m), patch.object(
                node.llm_service, "call_model", return_value="根据问题需要使用已有资料，正式程序仍须人工复核。") as call:
                result = asyncio.run(node.direct_response_node(state))
            self.assertEqual("根据问题需要使用已有资料，正式程序仍须人工复核。", result["final_answer"])
            payload = json.loads(call.call_args.args[0][1]["content"])
            self.assertEqual(query, payload["current_question"])
            self.assertIn("可靠性", payload["conversation"]["messages"][0]["content"])
            self.assertFalse(call.call_args.args[1]["reasoning"])
            for key in ("tools_plan", "tool_calls", "citations", "evidence_registry", "authorized_tools"):
                self.assertEqual([], result[key])

    def test_non_capability_routes_do_not_call_answer_model(self):
        for route in ("out_of_scope", "clarification"):
            with patch.object(node.llm_service, "call_model") as call:
                result = asyncio.run(node.direct_response_node({"scope_route": route, "query": "问句", "clarification_question": "哪个对象？"}))
            call.assert_not_called()
            self.assertTrue(result["final_answer"])

    def test_failure_returns_observable_product_fallback(self):
        with patch.object(node.agent_model_role_service, "get_model_by_role", side_effect=RuntimeError("private detail")), patch(
            "backend.agent.tracing.record_degradation") as trace:
            result = asyncio.run(node.direct_response_node({"scope_route": "agent_capability", "query": "能做什么？"}))
        self.assertIn("暂未能生成", result["final_answer"])
        self.assertNotIn("private detail", result["final_answer"])
        trace.assert_called_once_with("capability_response", detail="RuntimeError")

    def test_observability_failure_cannot_break_product_fallback(self):
        with patch.object(node.agent_model_role_service, "get_model_by_role", side_effect=RuntimeError("private")), patch(
            "backend.agent.tracing.record_degradation", side_effect=OSError("trace unavailable")):
            result = asyncio.run(node.direct_response_node({"scope_route": "agent_capability", "query": "能做什么？"}))
        self.assertIn("暂未能生成", result["final_answer"])
        self.assertEqual([], result["tool_calls"])

    def test_static_product_facts_preserve_authority_and_calculation_boundaries(self):
        for text in ("最小割集", "provisional", "人工复核", "没有检索", "PostgreSQL"):
            self.assertIn(text, node.CAPABILITY_FACTS)
