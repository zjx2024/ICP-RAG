import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import answer_mode_for_intent, make_evidence, make_tool_result
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes import plan as plan_module
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)


class KnowledgeQuestionFlowTests(unittest.TestCase):
    def test_answer_prompts_require_evidence_backed_complete_but_non_mechanical_answers(self):
        self.assertIn("只展开与本轮分问相关的内容", answer_module.ANSWER_PROMPT)
        self.assertIn("逐条抄录字段", answer_module.ANSWER_PROMPT)
        self.assertFalse(hasattr(answer_module, "MAINTENANCE_ANSWER_PROMPT"))

    def test_exact_hmc_prompt_can_use_only_explicitly_linked_complementary_evidence(self):
        prompt = answer_module._build_answer_prompt({"query": "这个报码有对应接口吗？",
            "answer_mode": "exact_code_lookup"}, "本轮证据")
        self.assertIn("这个报码有对应接口吗？", prompt)
        self.assertIn("相似描述不能证明记录相同或跨文档关联", prompt)
        self.assertIn("未确认关联均不等于原文没有", prompt)
        self.assertFalse(hasattr(answer_module, "EXACT_HMC_ANSWER_PROMPT"))

    def test_other_question_focus_keeps_direct_result_and_actionable_boundary(self):
        reliability_focus = answer_module._answer_task_instruction({"intent": "reliability_query"})
        history_focus = answer_module._answer_task_instruction({"intent": "history_query"})

        self.assertEqual(reliability_focus, history_focus)
        self.assertIn("原问题是本轮目标的依据", reliability_focus)
        self.assertNotIn("发生时间", history_focus)

        contracted = answer_module._answer_task_instruction(
            {"query_frame": {"required_answer_fields": ["rbd_missing_parameters"]}},
        )
        self.assertEqual(reliability_focus, contracted)
        self.assertIn("参数缺项和阻断原因", contracted)

    def test_multi_target_focus_keeps_cause_maintenance_and_history(self):
        focus = answer_module._answer_task_instruction(
            {
                "query_targets": [
                    {"intent": "cause_analysis"},
                    {"intent": "maintenance_query"},
                    {"intent": "history_query"},
                ]
            },
        )

        self.assertIn("原问题是本轮目标的依据", focus)
        self.assertNotIn("原因与作用机制", focus)
        self.assertNotIn("发生时间", focus)

        explicit_focus = answer_module._answer_task_instruction(
            {
                "query_targets": [
                    {"intent": "architecture_query"},
                ],
                "query_frame": {
                    "required_answer_fields": [
                        "fault_cause",
                        "maintenance_action",
                        "history_result",
                    ]
                },
            },
        )
        self.assertEqual(focus, explicit_focus)

    def test_component_condition_impact_is_a_direct_knowledge_question(self):
        model_response = json.dumps(
            {
                "intent": "impact_analysis",
                "query_targets": [{"intent": "impact_analysis", "reason": "用户询问后果"}],
                "required_answer_fields": ["local_impact", "final_impact"],
                "rewritten_query": "滤波电容老化对系统的影响",
                "entity_keywords": ["滤波电容"],
                "query_frame": {
                    "entities": ["滤波电容"],
                    "symptoms": ["老化"],
                    "normalized_terms": ["滤波电容老化"],
                },
                "tool_queries": {
                    "fmeca_query": ["滤波电容老化 故障影响 严酷度"],
                    "hybrid_search": ["滤波电容老化会对系统造成什么影响"],
                },
                "tools_plan": [{"tool": "fmeca_query", "reason": "查询明确影响记录"}],
                "reasoning": "用户询问已知部件状态的影响，不进行唯一故障诊断",
            },
            ensure_ascii=False,
        )
        with (
            patch.object(plan_module.llm_service, "_call_model", return_value=model_response),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            parsed = _run(
                plan_module.plan_node(
                    {
                        "query": "滤波电容有点老化了，会有什么影响吗",
                        "messages": [],
                        "memory_enabled": False,
                        "iteration": 0,
                    }
                )
            )

        self.assertEqual("impact_analysis", parsed["intent"])
        self.assertEqual("knowledge_explanation", answer_mode_for_intent(parsed["intent"]))
        self.assertEqual(
            ["fmeca_query", "hybrid_search"],
            [item["tool"] for item in parsed["tools_plan"]],
        )
        self.assertEqual(
            ["滤波电容老化 故障影响 严酷度"],
            parsed["tool_queries"]["fmeca_query"],
        )

    def test_impact_retrieval_does_not_generate_diagnostic_questionnaire(self):
        fmeca_rows = [
            {
                "doc_id": "doc-fmeca",
                "doc_name": "FMECA分析报告.docx",
                "code": "FM-SYN-002",
                "product_flag": "ICU / FUNC-110 / 外部电源输入调理",
                "fault_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化、连接阻抗增大或屏蔽接地不良。",
                "local_impact": "输入纹波和传导干扰增大。",
                "final_impact": "多个子系统出现间歇性异常。",
                "severity": "III",
                "source_ref": "FMECA分析报告.docx#row=3",
            }
        ]
        state = {
            "intent": "impact_analysis",
            "answer_mode": "knowledge_explanation",
            "query": "滤波电容有点老化了，会有什么影响吗",
            "rewritten_query": "滤波电容老化的影响",
            "entity_keywords": ["滤波电容", "老化"],
            "allowed_tools": ["fmeca_query", "hybrid_search", "evidence_synthesis"],
            "hybrid_doc_types": ["fmeca"],
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }

        with (
            patch.object(diagnosis_module, "_query_fmeca", return_value=fmeca_rows),
            patch.object(diagnosis_module, "_hybrid_search", return_value=([], {"reranked": False})),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: list(rows),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual("", result["clarification_question"])
        self.assertNotIn(
            "fault_reasoning",
            [item["tool"] for item in result["tool_calls"]],
        )

    def test_gate_does_not_treat_impact_explanation_as_fault_narrowing(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="局部影响为输入纹波增大；最终影响为多个子系统间歇性异常。",
        )
        result = _run(
            evidence_gate_node(
                {
                    "intent": "impact_analysis",
                    "query": "滤波电容有点老化了，会有什么影响吗",
                    "allowed_tools": ["fmeca_query", "evidence_synthesis"],
                    "evidence_registry": [evidence],
                    "tool_results": [
                        make_tool_result(
                            tool="fmeca_query",
                            status="ok",
                            backend="postgres",
                            evidence_ids=[evidence["evidence_id"]],
                        )
                    ],
                    "candidate_faults": [
                        {
                            "fault": "输入滤波性能退化",
                            "score": 0.25,
                            "exact_hmc_match": False,
                            "structured_hmc_match": False,
                        }
                    ],
                    "diagnostic_support": 0.48,
                }
            )
        )

        decision = result["safety_decision"]
        self.assertEqual("answer", decision["status"])
        self.assertFalse(decision["requires_clarification"])
        self.assertEqual([], decision["questions"])
        self.assertNotIn("unique_diagnosis", decision["blocked_claims"])



if __name__ == "__main__":
    unittest.main()
