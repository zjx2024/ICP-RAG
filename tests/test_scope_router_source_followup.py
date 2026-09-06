import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import scope_router as scope_module
from backend.agent.nodes import plan as plan_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _knowledge_followup_state(query):
    return {
        "query": query,
        "messages": [
            {"role": "user", "content": "若 A 路电源损坏，会有什么影响"},
            {
                "role": "assistant",
                "content": "现有 FMECA 记录说明可能影响启动控制。[^1]",
                "conversation_anchors": [{"doc_type": "fmeca", "record_id": "FM-1"}],
            },
            # 生产接口会先保存本轮消息，再装载历史；覆盖该重复形态。
            {"role": "user", "content": query},
        ],
    }


class NamedSourceFollowupRoutingTests(unittest.TestCase):
    @staticmethod
    def _model_route(route):
        return (
            '{"route":"' + route + '","confidence":0.95,'
            '"reason":"模型识别的本轮范围","clarification_question":""}'
        )

    def test_required_retrieval_corrections_continue_the_knowledge_turn(self):
        for query in ("你不查RBD文档吗", "请补查RBD依据", "为什么没查FTA"):
            with self.subTest(query=query), patch.object(
                scope_module.llm_service,
                "call_model",
                return_value=self._model_route("knowledge_base"),
            ) as model_call, patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ):
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("knowledge_base", result["scope_route"])
                self.assertEqual("model", result["scope_route_source"])
                self.assertNotIn("tool_calls", result)
                self.assertNotIn("tools_plan", result)
                model_call.assert_called_once()

    def test_source_followup_covers_all_seven_document_types(self):
        sources = (
            "系统架构与功能说明",
            "HMC",
            "FTA",
            "RBD",
            "FMECA",
            "MFL",
            "故障历史",
        )
        for source in sources:
            query = f"请补查{source}依据"
            with self.subTest(source=source), patch.object(
                scope_module.llm_service,
                "call_model",
                return_value=self._model_route("knowledge_base"),
            ), patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ):
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("knowledge_base", result["scope_route"])

    def test_pure_source_capability_questions_do_not_start_retrieval(self):
        for query in ("你能查RBD吗", "你能查哪些资料", "你是否支持FTA文档吗"):
            with self.subTest(query=query), patch.object(
                scope_module.llm_service,
                "call_model",
                return_value=self._model_route("agent_capability"),
            ) as model_call, patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ):
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("agent_capability", result["scope_route"])
                self.assertEqual([], result["tools_plan"])
                self.assertEqual([], result["allowed_tools"])
                model_call.assert_called_once()

    def test_content_request_with_capability_wording_still_uses_model_route(self):
        query = "你能查一下RBD里的A路电源影响吗"
        response = (
            '{"route":"knowledge_base","confidence":0.95,'
            '"reason":"要求查询具体工程内容","clarification_question":""}'
        )
        with (
            patch.object(
                scope_module.llm_service, "_call_model", return_value=response
            ) as model_call,
            patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(scope_module.scope_router_node({"query": query, "messages": []}))

        self.assertEqual("knowledge_base", result["scope_route"])
        self.assertEqual("model", result["scope_route_source"])
        model_call.assert_called_once()

    def test_omission_challenge_without_a_prior_knowledge_turn_is_not_forced(self):
        query = "你不查RBD文档吗"
        state = {
            "query": query,
            "messages": [
                {"role": "user", "content": "你的功能"},
                {"role": "assistant", "content": "我可以查询七类工程资料。"},
                {"role": "user", "content": query},
            ],
        }
        response = (
            '{"route":"agent_capability","confidence":0.9,'
            '"reason":"询问资料范围","clarification_question":""}'
        )
        with (
            patch.object(scope_module.llm_service, "_call_model", return_value=response),
            patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(scope_module.scope_router_node(state))

        self.assertEqual("agent_capability", result["scope_route"])
        self.assertEqual("model", result["scope_route_source"])


class ContinuationSourceHelperEdgeCaseTests(unittest.TestCase):
    """来源延续帮助函数必须对无助手消息的历史安全返回空集。

    首个知识库轮次（history=[]）以及只有用户消息的会话都会走到
    _prior_answer_doc_types／_prior_answer_anchor_subjects；它们必须先
    初始化返回值，否则循环体一次未执行时抛 UnboundLocalError，导致
    所有知识库轮次在 plan 节点整体失败。
    """

    def test_empty_history_returns_empty_doc_types(self):
        for messages in ([], None, {"role": "user", "content": "x"}, "not-a-list"):
            with self.subTest(messages=messages):
                self.assertEqual(
                    [], plan_module._prior_answer_doc_types(messages)
                )

    def test_empty_history_returns_empty_anchor_subjects(self):
        for messages in ([], None, "not-a-list"):
            with self.subTest(messages=messages):
                self.assertEqual(
                    [], plan_module._prior_answer_anchor_subjects(messages)
                )

    def test_history_without_assistant_message_returns_empty(self):
        messages = [
            {"role": "user", "content": "第一个问题"},
            {"role": "user", "content": "第二个问题"},
        ]
        self.assertEqual([], plan_module._prior_answer_doc_types(messages))
        self.assertEqual([], plan_module._prior_answer_anchor_subjects(messages))

    def test_prior_cited_doc_types_are_extracted_from_anchors(self):
        messages = [
            {"role": "user", "content": "哪些结论能说？"},
            {
                "role": "assistant",
                "content": "已根据 RBD 资料说明。[^1]",
                "conversation_anchors": [
                    {"doc_type": "rbd", "record_id": "RBD-SYN-001"},
                    {"doc_type": "rbd", "record_id": "RBD-SYN-002"},
                    {"doc_type": "mfl", "record_id": "MFL-SYN-049"},
                ],
            },
        ]
        self.assertEqual(
            ["rbd", "mfl"], plan_module._prior_answer_doc_types(messages)
        )

    def test_boundary_followup_does_not_reconstruct_source_from_prior_prose(self):
        query = "只根据已经查到的资料回答：哪些结论能说，哪些不能说？"
        messages = [
            {
                "role": "user",
                "content": "请查 RBD：输入滤波器是否在成功路径中？",
            },
            {
                "role": "assistant",
                "content": "RBD 未找到对应记录，不能以其他资料替代。",
            },
            {"role": "user", "content": query},
        ]

        tools = plan_module._continuation_source_tools(
            query=query,
            messages=messages,
            prior_doc_types=[],
            response_focus=["source_capability", "source_limitation"],
        )

        self.assertEqual([], tools)

    def test_source_boundary_wording_alone_does_not_drive_routing(self):
        query = "只根据已经查到的资料回答：哪些结论能说，哪些不能说？"
        messages = [
            {"role": "user", "content": "请查 RBD：某部件是否存在？"},
            {"role": "assistant", "content": "没有找到对应记录。"},
        ]

        self.assertEqual(
            [],
            plan_module._continuation_source_tools(
                query=query,
                messages=messages,
                prior_doc_types=[],
                response_focus=[],
            ),
        )

    def test_non_boundary_followup_does_not_inherit_empty_source_request(self):
        messages = [
            {"role": "user", "content": "请查 RBD：某部件是否存在？"},
            {"role": "assistant", "content": "没有找到对应记录。"},
        ]

        self.assertEqual(
            [],
            plan_module._continuation_source_tools(
                query="那应该怎么维修？",
                messages=messages,
                prior_doc_types=[],
            ),
        )

    def test_boundary_followup_inherits_only_explicitly_selected_anchor_sources(self):
        selected = [{
            "anchor_id": "ev-rbd",
            "doc_type": "rbd",
            "record_id": "RBD-SYN-001",
        }]

        tools = plan_module._continuation_source_tools(
            query="这些资料能证明什么",
            messages=[],
            # An old caller may still pass the nearest answer's whole set; it is
            # compatibility data only and must not authorize MFL.
            prior_doc_types=["rbd", "mfl"],
            selected_anchors=selected,
            context_used=True,
            response_focus=["source_capability"],
        )

        self.assertEqual(["rbd_query"], tools)

    def test_boundary_focus_without_selected_context_cannot_inherit_prior_sources(self):
        anchor = {
            "anchor_id": "ev-rbd",
            "doc_type": "rbd",
            "record_id": "RBD-SYN-001",
        }
        for selected, context_used in (([], True), ([anchor], False)):
            with self.subTest(selected=selected, context_used=context_used):
                self.assertEqual(
                    [],
                    plan_module._continuation_source_tools(
                        query="这些资料能证明什么",
                        messages=[],
                        prior_doc_types=["rbd", "mfl"],
                        selected_anchors=selected,
                        context_used=context_used,
                        response_focus=["source_capability", "source_limitation"],
                    ),
                )

    def test_current_source_contract_prevents_selected_anchor_source_stacking(self):
        self.assertEqual(
            [],
            plan_module._continuation_source_tools(
                query="再按 FMECA 核对",
                messages=[],
                selected_anchors=[{
                    "anchor_id": "ev-rbd",
                    "doc_type": "rbd",
                    "record_id": "RBD-SYN-001",
                }],
                context_used=True,
                response_focus=["source_capability"],
                current_requested_sources=[{
                    "doc_type": "fmeca",
                    "basis": "FMECA",
                    "question_role": "evidence",
                }],
            ),
        )

    def test_plan_locks_source_boundary_to_only_the_selected_anchor_family(self):
        query = "这些已查资料能证明什么，不能证明什么？"
        target = {
            "primary_intent": "entity_profile",
            "targets": ["entity_profile"],
            "contextualized_query": "RBD-SYN-001 能证明什么，不能证明什么？",
            "subjects": ["RBD-SYN-001"],
            "required_answer_fields": [],
            "question_semantics": "explain",
            "response_focus": ["source_capability", "source_limitation"],
            "response_scope": "incremental",
            "response_scope_basis": "这些已查资料能证明什么，不能证明什么",
            "coverage_scope": "focused_detail",
            "context_used": True,
            "context_status": "resolved",
            "selected_anchor_ids": ["ev-rbd"],
            "topology_dependency": {"required": False},
            "relation_expansion": {"needed": False},
        }
        planner = {
            "intent": "entity_profile",
            "query_targets": [{"intent": "entity_profile"}],
            "rewritten_query": "RBD-SYN-001 能证明什么，不能证明什么？",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [],
            },
            # The planner sees both old source names, but they carry no source
            # authority unless the target selected their citation anchor.
            "tools_plan": [
                {"tool": "rbd_query"},
                {"tool": "mfl_query"},
            ],
        }
        messages = [
            {"role": "user", "content": "检查这个对象的资料边界"},
            {
                "role": "assistant",
                "message_id": "answer-1",
                "content": "上一答同时引用了两个资料族。",
                "conversation_anchors": [
                    {
                        "anchor_id": "ev-rbd",
                        "doc_type": "rbd",
                        "doc_id": "doc-rbd",
                        "record_id": "RBD-SYN-001",
                    },
                    {
                        "anchor_id": "ev-mfl",
                        "doc_type": "mfl",
                        "doc_id": "doc-mfl",
                        "record_id": "MFL-SYN-001",
                    },
                ],
            },
            {"role": "user", "content": query},
        ]
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[
                    json.dumps(target, ensure_ascii=False),
                    json.dumps(planner, ensure_ascii=False),
                ],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(plan_module.plan_node({
                "query": query,
                "messages": messages,
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertEqual(["rbd"], result["query_frame"]["locked_doc_types"])
        self.assertTrue(result["query_frame"]["evidence_domain_locked"])
        self.assertIn("rbd_query", result["allowed_tools"])
        self.assertNotIn("mfl_query", result["allowed_tools"])
        self.assertEqual(
            ["rbd_query"], result["query_frame"]["requested_source_tools"]
        )


if __name__ == "__main__":
    unittest.main()
