import asyncio
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
    def test_required_retrieval_corrections_continue_the_knowledge_turn(self):
        for query in ("你不查RBD文档吗", "请补查RBD依据", "为什么没查FTA"):
            with self.subTest(query=query), patch.object(
                scope_module.llm_service, "_call_model"
            ) as model_call:
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("knowledge_base", result["scope_route"])
                self.assertEqual(
                    "deterministic_source_followup", result["scope_route_source"]
                )
                self.assertNotIn("tool_calls", result)
                self.assertNotIn("tools_plan", result)
                model_call.assert_not_called()

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
            with self.subTest(source=source):
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("knowledge_base", result["scope_route"])

    def test_pure_source_capability_questions_do_not_start_retrieval(self):
        for query in ("你能查RBD吗", "你能查哪些资料", "你是否支持FTA文档吗"):
            with self.subTest(query=query), patch.object(
                scope_module.llm_service, "_call_model"
            ) as model_call:
                result = _run(
                    scope_module.scope_router_node(_knowledge_followup_state(query))
                )
                self.assertEqual("agent_capability", result["scope_route"])
                self.assertEqual([], result["tools_plan"])
                self.assertEqual([], result["allowed_tools"])
                model_call.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
