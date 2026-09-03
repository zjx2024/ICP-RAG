import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _state():
    values = (
        ("电源保障链可靠性模型", "0.99951007"),
        ("核心处理功能可靠性模型", "0.99929014"),
        ("外部通信功能可靠性模型", "0.99910032"),
        ("显示输出功能可靠性模型", "0.99888052"),
    )
    evidence = []
    for index, (name, reliability) in enumerate(values, start=1):
        evidence.append(
            make_evidence(
                source_kind="calculation",
                authority="deterministic_derivation",
                evidence_level="L1_DIRECT",
                claim_type="rbd_calculation",
                doc_type="rbd",
                doc_id="doc-rbd",
                record_id=f"RBD-{index}",
                title=f"RBD 确定性计算：{name}",
                content=f"RBD 确定性计算：{name}；系统可靠度：{reliability}。",
            )
        )
    ids = [item["evidence_id"] for item in evidence]
    return {
        "query": "十小时任务中四条链路哪条成功概率最低？",
        "intent": "reliability_query",
        "calculation_requests": ["rbd_reliability"],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
    }


def _fta_state():
    evidence = make_evidence(
        source_kind="calculation",
        authority="deterministic_derivation",
        evidence_level="L1_DIRECT",
        claim_type="fta_calculation",
        doc_type="fta",
        doc_id="doc-fta",
        record_id="FTA-1",
        title="FTA 确定性计算：系统无法启动",
        content=(
            "FTA 确定性计算：系统无法启动；顶事件：系统无法启动；"
            "最小割集 2 个（完整列表：1. 供电中断；2. 时钟丢失 + 复位失效）；"
            "顶事件概率：0.0012。"
        ),
    )
    return {
        "query": "系统无法启动的最小割集有哪些？",
        "intent": "reliability_query",
        "calculation_requests": ["fta_cut_sets"],
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence["evidence_id"]],
        "safety_decision": {
            "approved_evidence_ids": [evidence["evidence_id"]],
            "context_evidence_ids": [evidence["evidence_id"]],
        },
    }


def _fta_probability_state(*, blocked=False):
    result = {
        "model": "FTA",
        "top_event": "任务数据输出中断",
        "minimal_cut_sets": [["A通道丢失"], ["B通道丢失"]],
        "minimal_cut_set_count": 2,
        "top_event_probability": None if blocked else 0.00125,
        "mission_time": 10.0,
        "missing_probability_nodes": ["B通道丢失"] if blocked else [],
        "shared_dependency_nodes": [],
        "probability_block_reason": "缺少B通道基本事件概率" if blocked else "",
        "assumption": "基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算",
    }
    evidence = make_evidence(
        source_kind="calculation",
        authority="deterministic_derivation",
        evidence_level="L2_CORROBORATED",
        claim_type="fta_calculation",
        doc_type="fta",
        doc_id="doc-fta-probability",
        record_id="FTA-PROBABILITY",
        title="FTA 确定性计算：任务数据输出中断",
        content=json.dumps(result, ensure_ascii=False),
        limitations=[result["assumption"]],
    )
    return {
        "query": "十小时内任务数据输出中断的概率是多少？",
        "intent": "reliability_query",
        "calculation_requests": ["fta_probability"],
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence["evidence_id"]],
        "safety_decision": {
            "approved_evidence_ids": [evidence["evidence_id"]],
            "context_evidence_ids": [evidence["evidence_id"]],
        },
    }


class AnswerCalculationReviewTests(unittest.TestCase):
    def test_calculation_answer_drops_generic_followup_questions(self):
        state = _fta_state()
        draft = (
            "FTA 确定性计算：最小割集 2 个。[^1]\n\n"
            "### 下一步需要确认\n"
            "- 请补充故障发生阶段和告警文字。\n\n"
            "实际处置以适用构型、有效版本和正式维修程序为准。"
        )

        result = answer_module._strip_diagnostic_followup_for_calculation(
            draft, state
        )

        self.assertNotIn("下一步需要确认", result)
        self.assertNotIn("故障发生阶段", result)
        self.assertIn("FTA 确定性计算", result)
        self.assertIn("实际处置以适用构型", result)

        plain = (
            "FTA 确定性计算：最小割集 2 个。[^1]\n"
            "当前信息不足以唯一定位故障。请补充：故障阶段和告警文字。\n"
            "实际处置以适用构型、有效版本和正式维修程序为准。"
        )
        plain_result = answer_module._strip_diagnostic_followup_for_calculation(
            plain, state
        )
        self.assertNotIn("当前信息不足以唯一定位故障", plain_result)

    def test_calculation_followup_does_not_require_inherited_structural_prose(self):
        state = _fta_state()
        state["query_frame"] = {
            "coverage_scope": "focused_detail",
            "required_answer_fields": ["structural_cause_chain"],
        }
        missing, manifest = answer_module._target_completion_manifest("", state)

        self.assertEqual([], missing)
        self.assertEqual([], manifest)

    def test_manifest_keeps_all_deterministic_rbd_results(self):
        manifest = answer_module._rbd_calculation_manifest(_state())

        self.assertEqual(4, len(manifest))
        self.assertEqual(
            {"0.99951007", "0.99929014", "0.99910032", "0.99888052"},
            {item["reliability"] for item in manifest},
        )

    def test_topology_dependency_does_not_remove_deterministic_rbd_results(self):
        state = _state()
        topology_dependency = {
            "required": True,
            "subject": "A路电源",
            "subject_kind": "supply_path",
            "failure_state": "失效",
            "question_aspects": ["mission_success"],
        }
        state["query_frame"] = {"topology_dependency": topology_dependency}
        state["dialogue_context"] = {
            "topology_dependency": topology_dependency,
        }

        context = answer_module._context_evidence(state)
        manifest = answer_module._rbd_calculation_manifest(state)

        self.assertEqual(4, len(context))
        self.assertTrue(
            all(item["claim_type"] == "rbd_calculation" for item in context)
        )
        self.assertEqual(4, len(manifest))
        self.assertEqual(
            {"0.99951007", "0.99929014", "0.99910032", "0.99888052"},
            {item["reliability"] for item in manifest},
        )

    def test_missing_rbd_model_triggers_llm_complete_rewrite(self):
        state = _state()
        draft = "电源 0.99951007，处理 0.99929014，通信 0.99910032。[^1][^2][^3]"
        complete = (
            "电源保障 0.99951007，核心处理 0.99929014，对外通信 0.99910032，"
            "显示输出 0.99888052，因此显示输出最低。[^1][^2][^3][^4]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_calculations(draft, state, None))

        call.assert_called_once()
        self.assertEqual(complete, result)
        self.assertIn("0.99888052", call.call_args.args[0][0]["content"])

    def test_complete_rbd_answer_skips_review(self):
        state = _state()
        answer = (
            "电源保障链为 0.99951007[^1]，核心处理链为 0.99929014[^2]，"
            "外部通信链为 0.99910032[^3]，显示输出链为 0.99888052[^4]，"
            "因此显示输出链最低。"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("完整计算答案不应再次调用模型"),
        ):
            result = _run(answer_module._review_answer_calculations(answer, state, None))

        self.assertEqual(answer, result)

    def test_rbd_values_without_model_identity_or_citations_are_not_complete(self):
        state = _state()
        anonymous_values = "、".join(
            ("0.99951007", "0.99929014", "0.99910032", "0.99888052")
        )
        complete = (
            "电源保障链为 0.99951007[^1]，核心处理链为 0.99929014[^2]，"
            "外部通信链为 0.99910032[^3]，显示输出链为 0.99888052[^4]，"
            "因此显示输出链最低。"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(
                answer_module._review_answer_calculations(anonymous_values, state, None)
            )

        call.assert_called_once()
        self.assertEqual(complete, result)

    def test_missing_fta_cut_set_list_triggers_complete_rewrite(self):
        state = _fta_state()
        draft = "系统无法启动可能与供电或时钟有关。[^1]"
        complete = (
            "该顶事件共有 2 个最小割集：`供电中断`，以及"
            "`时钟丢失 + 复位失效`。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_calculations(draft, state, None))

        call.assert_called_once()
        self.assertEqual(complete, result)

    def test_fta_count_without_cut_set_items_is_not_complete(self):
        state = _fta_state()
        draft = "该顶事件共有 2 个最小割集。[^1]"
        complete = (
            "该顶事件共有 2 个最小割集：`供电中断`，以及"
            "`时钟丢失 + 复位失效`。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_calculations(draft, state, None))

        call.assert_called_once()
        self.assertEqual(complete, result)

    def test_complete_fta_cut_set_answer_skips_review(self):
        state = _fta_state()
        answer = (
            "该顶事件共有 2 个最小割集：`供电中断`，以及"
            "`时钟丢失 + 复位失效`。[^1]"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("完整 FTA 计算答案不应再次调用模型"),
        ):
            result = _run(answer_module._review_answer_calculations(answer, state, None))

        self.assertEqual(answer, result)

    def test_fta_probability_omissions_trigger_bounded_complete_rewrite(self):
        state = _fta_probability_state()
        draft = "顶事件概率为 0.00125。[^1]"
        complete = (
            "任务数据输出中断在任务时间 10.0 小时内的顶事件概率为 0.00125；"
            "概率输入参数完整。[^1]\n\n"
            "计算假设为基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_calculations(draft, state, None))

        call.assert_called_once()
        self.assertEqual(complete, result)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("任务时间：10.0 小时", prompt)
        self.assertIn("概率输入参数：完整", prompt)
        self.assertNotIn("完整列表", prompt)

    def test_complete_fta_probability_skips_review_without_forcing_cut_sets(self):
        state = _fta_probability_state()
        answer = (
            "任务数据输出中断在任务时间 10.0 小时内的顶事件概率为 0.00125；"
            "概率输入参数完整。[^1]\n\n"
            "计算假设为基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算。[^1]"
        )

        self.assertFalse(
            answer_module._missing_fta_calculation_values(
                answer, answer_module._fta_calculation_manifest(state)
            )
        )
        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("完整 FTA 概率回答不应再次调用模型"),
        ):
            result = _run(answer_module._review_answer_calculations(answer, state, None))

        self.assertEqual(answer, result)

    def test_blocked_fta_probability_requires_missing_input_and_limit_not_a_number(self):
        state = _fta_probability_state(blocked=True)
        draft = "当前不能给出顶事件概率。[^1]"
        complete = (
            "任务数据输出中断在任务时间 10.0 小时内未形成顶事件概率结论："
            "缺失概率参数为 B通道丢失，限制原因是缺少B通道基本事件概率。[^1]\n\n"
            "计算假设为基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_calculations(draft, state, None))

        call.assert_called_once()
        self.assertEqual(complete, result)
        manifest = answer_module._fta_calculation_manifest(state)
        self.assertEqual("", manifest[0]["top_event_probability"])
        self.assertNotIn("最小割集总数", "\n".join(
            answer_module._missing_fta_calculation_values(result, manifest)
        ))

    def test_fta_block_reason_is_not_repeated_as_a_generic_limitation(self):
        state = _fta_probability_state(blocked=True)
        evidence = state["evidence_registry"][0]
        evidence["limitations"] = [
            "基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算",
            "缺少B通道基本事件概率",
        ]

        manifest = answer_module._fta_calculation_manifest(state)

        self.assertEqual("缺少B通道基本事件概率", manifest[0]["probability_block_reason"])
        self.assertNotIn("缺少B通道基本事件概率", manifest[0]["limitations"])

    def test_fta_cut_set_explanation_rejects_other_channel_fmeca(self):
        for claimed_channel, evidence_channel in (("A", "B"), ("B", "A")):
            with self.subTest(claimed=claimed_channel, evidence=evidence_channel):
                calculation = make_evidence(
                    source_kind="calculation",
                    authority="deterministic_derivation",
                    evidence_level="L1_DIRECT",
                    claim_type="fta_calculation",
                    doc_type="fta",
                    doc_id="doc-fta",
                    record_id=f"FTA-{claimed_channel}",
                    title="FTA 确定性计算：外部数据中断",
                    content=(
                        "最小割集 1 个（完整列表：1. "
                        f"{claimed_channel} 通道输入丢失）。"
                    ),
                )
                other_mode = make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type="fmeca_fact",
                    doc_type="fmeca",
                    doc_id="doc-fmeca",
                    record_id=f"FM-{evidence_channel}",
                    title=f"{evidence_channel} 通道输入丢失",
                    content="最终影响：业务数据输出中断。",
                    metadata={
                        "answer_fields": {
                            "failure_mode": f"{evidence_channel} 通道输入丢失",
                            "final_impact": "业务数据输出中断",
                        }
                    },
                )
                ids = [calculation["evidence_id"], other_mode["evidence_id"]]
                state = {
                    "query": "列出外部数据中断的最小割集",
                    "intent": "reliability_query",
                    "calculation_requests": ["fta_cut_sets"],
                    "evidence_registry": [calculation, other_mode],
                    "context_evidence_ids": ids,
                    "safety_decision": {
                        "approved_evidence_ids": ids,
                        "context_evidence_ids": ids,
                    },
                }
                ledger = answer_module._citation_ledger(state)
                calculation_citation = next(
                    number
                    for number, item in ledger.items()
                    if item["evidence_id"] == calculation["evidence_id"]
                )
                fmeca_citation = next(
                    number
                    for number, item in ledger.items()
                    if item["evidence_id"] == other_mode["evidence_id"]
                )
                draft = (
                    "该顶事件有 1 个最小割集："
                    f"`{claimed_channel} 通道输入丢失`。"
                    f"[^{calculation_citation}]\n\n"
                    f"### 最小割集 1：{claimed_channel} 通道输入丢失\n"
                    f"- 影响：业务数据输出中断。[^{fmeca_citation}]"
                )
                repair = json.dumps(
                    {
                        "units": [
                            {
                                "unit_id": "u1",
                                "bindings": [
                                    {
                                        "citation_id": calculation_citation,
                                        "quote": calculation["content"],
                                    }
                                ],
                            },
                            {
                                "unit_id": "u2",
                                "bindings": [
                                    {
                                        "citation_id": fmeca_citation,
                                        "quote": "最终影响：业务数据输出中断。",
                                    }
                                ],
                            },
                        ]
                    },
                    ensure_ascii=False,
                )

                with patch.object(
                    answer_module.llm_service,
                    "_call_model",
                    return_value=repair,
                ):
                    result = _run(
                        answer_module._repair_answer_citations(
                            draft,
                            state,
                            None,
                            citation_count=2,
                        )
                    )

                self.assertIn(f"{claimed_channel} 通道输入丢失", result)
                self.assertIn(f"[^{calculation_citation}]", result)
                self.assertNotIn("业务数据输出中断", result)
                self.assertNotIn(f"[^{fmeca_citation}]", result)


if __name__ == "__main__":
    unittest.main()
