import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.answer_contract import (
    RBD_ANSWER_FIELDS,
    ensure_intent_answer_fields,
)
from backend.agent.nodes import plan as plan_module
from backend.agent.tool_policy import resolve_tool_policy


def _run(coroutine):
    return asyncio.run(coroutine)


def _target_payload(
    query,
    subject,
    subject_kind,
    failure_state,
    aspect,
    *,
    primary="impact_analysis",
):
    return {
        "primary_intent": primary,
        "targets": [primary],
        "contextualized_query": query,
        "subjects": [subject],
        "requested_facets": ["影响" if aspect == "impact" else "继续工作能力"],
        "required_answer_fields": ["local_impact", "final_impact"],
        "topology_dependency": {
            "required": True,
            "subject": subject,
            "subject_kind": subject_kind,
            "failure_state": failure_state,
            "question_aspects": [aspect],
        },
        "coverage_scope": "focused_detail",
        "collection_subject": "",
        "collection_member_kind": "unknown",
        "context_used": False,
        "context_status": "standalone",
        "selected_anchor_ids": [],
    }


class StructuredTopologyTargetTests(unittest.TestCase):
    def test_named_failed_path_keeps_impact_and_reliability_targets(self):
        cases = (
            (
                "若A路电源损坏，会有什么影响",
                "A路电源",
                "supply_path",
                "损坏",
                "impact",
            ),
            (
                "某通信通道断开后会造成什么后果",
                "某通信通道",
                "channel",
                "断开",
                "consequence",
            ),
            (
                "这条数据支路失联后设备还能工作吗",
                "这条数据支路",
                "branch",
                "失联",
                "continuity",
            ),
            (
                "主供电路径中断后还能维持任务吗",
                "主供电路径",
                "power_path",
                "中断",
                "mission_success",
            ),
        )
        for query, subject, kind, failure, aspect in cases:
            with self.subTest(query=query):
                parsed = plan_module._parse_target_response(
                    json.dumps(
                        _target_payload(
                            query,
                            subject,
                            kind,
                            failure,
                            aspect,
                            primary=(
                                "reliability_query"
                                if aspect in {"continuity", "mission_success"}
                                else "impact_analysis"
                            ),
                        ),
                        ensure_ascii=False,
                    )
                )

                self.assertIsNotNone(parsed)
                self.assertEqual(
                    {"impact_analysis", "reliability_query"},
                    {
                        target["intent"]
                        for target in parsed["query_targets"]
                    },
                )
                self.assertTrue(parsed["topology_dependency"]["required"])
                self.assertIn(
                    "rbd_single_failure_boundary",
                    parsed["required_answer_fields"],
                )
                self.assertIn(
                    "rbd_success_criterion",
                    parsed["required_answer_fields"],
                )

    def test_incomplete_topology_claim_cannot_promote_component_impact(self):
        incomplete = _target_payload(
            "滤波电容老化会有什么影响",
            "滤波电容",
            "branch",
            "",
            "impact",
        )
        parsed = plan_module._parse_target_response(
            json.dumps(incomplete, ensure_ascii=False)
        )

        self.assertEqual(
            ["impact_analysis"],
            [target["intent"] for target in parsed["query_targets"]],
        )
        self.assertFalse(parsed["topology_dependency"]["required"])
        self.assertFalse(
            RBD_ANSWER_FIELDS.intersection(parsed["required_answer_fields"])
        )

    def test_explicit_non_topology_module_impact_stays_fmeca_only(self):
        payload = {
            **_target_payload(
                "显示处理模块性能下降会影响什么功能",
                "显示处理模块",
                "path",
                "性能下降",
                "impact",
            ),
            "topology_dependency": {
                "required": False,
                "subject": "显示处理模块",
                "subject_kind": "unknown",
                "failure_state": "性能下降",
                "question_aspects": ["impact"],
            },
        }
        parsed = plan_module._parse_target_response(
            json.dumps(payload, ensure_ascii=False)
        )
        decision = resolve_tool_policy(
            intent=parsed["primary_intent"],
            query=payload["contextualized_query"],
            raw_user_query=payload["contextualized_query"],
            query_targets=parsed["query_targets"],
            required_answer_fields=parsed["required_answer_fields"],
            coverage_scope=parsed["coverage_scope"],
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        self.assertEqual(
            ["fmeca_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertNotIn("rbd_query", decision["authorized_tools"])


class ReliabilityContractPolicyTests(unittest.TestCase):
    def test_reliability_target_repairs_focused_impact_only_contract(self):
        fields = ensure_intent_answer_fields(
            ["local_impact", "final_impact"],
            intents=["impact_analysis", "reliability_query"],
        )
        self.assertIn("rbd_success_criterion", fields)

        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="若A路电源损坏，会有什么影响",
            raw_user_query="若A路电源损坏，会有什么影响",
            query_targets=[
                {"intent": "impact_analysis"},
                {"intent": "reliability_query"},
            ],
            required_answer_fields=["local_impact", "final_impact"],
            coverage_scope="focused_detail",
            # Deliberately omit RBD to reproduce an impact-first planner.
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        self.assertIn("fmeca_query", decision["allowed_tools"])
        self.assertIn("rbd_query", decision["allowed_tools"])
        self.assertIn("rbd_query", decision["authorized_tools"])
        self.assertIn(
            "rbd_success_criterion",
            decision["query_targets"][0]["required_fields"],
        )

    def test_non_numeric_topology_question_never_authorizes_calculator(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="某供电支路断开后设备还能工作吗",
            raw_user_query="某供电支路断开后设备还能工作吗",
            query_targets=[
                {"intent": "impact_analysis"},
                {"intent": "reliability_query"},
            ],
            required_answer_fields=[
                "local_impact",
                "rbd_single_failure_boundary",
                "rbd_success_criterion",
            ],
            coverage_scope="focused_detail",
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "rbd_query"},
                {"tool": "rbd_calculator"},
            ],
        )

        self.assertIn("rbd_query", decision["allowed_tools"])
        self.assertIn("function_locator", decision["allowed_tools"])
        self.assertNotIn("rbd_calculator", decision["allowed_tools"])
        self.assertIn("rbd_calculator", decision["rejected_tools"])
        self.assertEqual([], decision["calculation_requests"])


class ImplicitTopologyPlanIntegrationTests(unittest.TestCase):
    def test_impact_first_planner_cannot_drop_structured_reliability_target(self):
        query = "若A路电源损坏，会有什么影响"
        target_response = json.dumps(
            _target_payload(
                query,
                "A路电源",
                "supply_path",
                "损坏",
                "impact",
            ),
            ensure_ascii=False,
        )
        planner_response = json.dumps(
            {
                "intent": "impact_analysis",
                "query_targets": [{"intent": "impact_analysis"}],
                "relation_expansion": {
                    "needed": False,
                    "reason": "先读取直接影响与成功判据",
                },
                "answer_mode": "knowledge_explanation",
                "rewritten_query": query,
                "entity_keywords": ["A路电源"],
                "type_keywords": ["影响"],
                "query_frame": {
                    "entities": ["A路电源"],
                    "symptoms": ["损坏"],
                    "coverage_scope": "focused_detail",
                    "requested_facets": ["影响"],
                    "required_answer_fields": ["local_impact", "final_impact"],
                },
                "tool_queries": {
                    "fmeca_query": ["A路电源损坏的影响"],
                },
                "tools_plan": [
                    {"tool": "fmeca_query", "reason": "查询直接影响"},
                ],
                "reasoning": "先查询故障影响。",
            },
            ensure_ascii=False,
        )
        state = {
            "query": query,
            "messages": [{"role": "user", "content": query}],
            "memory_enabled": False,
            "iteration": 0,
        }
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[target_response, planner_response],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(plan_module.plan_node(state))

        self.assertEqual(
            {"impact_analysis", "reliability_query"},
            {target["intent"] for target in result["query_targets"]},
        )
        self.assertIn("fmeca_query", result["allowed_tools"])
        self.assertIn("function_locator", result["allowed_tools"])
        self.assertIn("rbd_query", result["allowed_tools"])
        self.assertIn("function_locator", result["tool_queries"])
        self.assertIn("rbd_query", result["tool_queries"])
        self.assertIn(
            "rbd_single_failure_boundary",
            result["query_frame"]["required_answer_fields"],
        )
        self.assertIn(
            "rbd_success_criterion",
            result["query_frame"]["required_answer_fields"],
        )
        self.assertEqual(
            {
                "required": True,
                "subject": "A路电源",
                "subject_kind": "supply_path",
                "failure_state": "损坏",
                "question_aspects": ["impact"],
            },
            result["query_frame"]["topology_dependency"],
        )
        self.assertNotIn("rbd_calculator", result["allowed_tools"])


if __name__ == "__main__":
    unittest.main()
