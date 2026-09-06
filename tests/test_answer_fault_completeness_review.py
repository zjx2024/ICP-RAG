import backend.agent.nodes.answer_domain_scope as _source_answer_domain_scope
import backend.agent.nodes.answer_fault_rules as _source_answer_fault_rules
import asyncio
import json
import time
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _state():
    fmeca = make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="fmeca_fact",
        doc_type="fmeca",
        doc_id="doc-fmeca",
        record_id="FM-SYN-041",
        title="帧转换校验错误",
        content="算法异常会导致有效帧无法输出，对应外部接口数据中断。",
        metadata={
            "answer_fields": {
                "fault_cause": "算法异常或数据存储损坏。",
                "local_impact": "有效帧无法输出。",
                "final_impact": "对应外部接口数据中断。",
                "detection_method": "对比输入输出帧和校验过程。",
            }
        },
    )
    mfl = make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="maintenance_locator",
        doc_type="mfl",
        doc_id="doc-mfl",
        record_id="MFL-SYN-041",
        title="帧转换故障维修定位",
        content="检查转换队列和配置，确认模块异常后更换协议转换模块。",
        metadata={
            "answer_fields": {
                "maintenance_action": "检查转换队列和配置，确认模块异常后更换协议转换模块。"
            }
        },
    )
    evidence = [fmeca, mfl]
    ids = [item["evidence_id"] for item in evidence]
    return {
        "query": "内部报文已经产生，但外接设备收不到，哪些环节可能出错？",
        "intent": "cause_analysis",
        "query_targets": [{"intent": "cause_analysis"}],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
        "query_frame": {
            "required_answer_fields": [
                "fault_cause",
                "local_impact",
                "final_impact",
                "detection_method",
                "maintenance_action",
            ],
        },
    }


def _parallel_state(*, comparison=True):
    specifications = (
        {
            "mode": "前端信号通道开路",
            "cause": "连接器松动或输入线路断开。",
            "local": "该路输入值不可用。",
            "final": "依赖该输入的状态判断失去依据。",
            "detect": "检查线路连续性和开路状态。",
            "action": "重新固定连接器并修复输入线路。",
            "sur": "状态字第21位异常",
        },
        {
            "mode": "转换级输入饱和",
            "cause": "输入超量程或增益配置错误。",
            "local": "转换结果固定在上限。",
            "final": "后续监测可能发生误判。",
            "detect": "核对输入电平、量程和原始转换值。",
            "action": "校正量程与增益，确认器件异常后更换转换模块。",
            "sur": "状态字第22位异常",
        },
    )
    evidence = []
    for index, spec in enumerate(specifications, start=1):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-compare-fmeca",
            record_id=f"FM-COMP-{index}",
            title=spec["mode"],
            content=(
                f"故障模式：{spec['mode']}\n故障原因：{spec['cause']}\n"
                f"局部影响：{spec['local']}\n最终影响：{spec['final']}\n"
                f"检测方法：{spec['detect']}"
            ),
            metadata={
                "answer_fields": {
                    "product_flag": "SIG / FUNC-880",
                    "failure_mode": spec["mode"],
                    "fault_cause": spec["cause"],
                    "local_impact": spec["local"],
                    "final_impact": spec["final"],
                    "detection_method": spec["detect"],
                    "explicit_references": {"mfl": [f"MFL-COMP-{index}"]},
                }
            },
        )
        fmeca["match_type"] = "structured_query" if index == 1 else "semantic_id_hydration"
        fmeca["retrieval_score"] = 0.82 if index == 1 else 0.41
        mfl = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_locator",
            doc_type="mfl",
            doc_id="doc-compare-mfl",
            record_id=f"MFL-COMP-{index}",
            title=f"{spec['mode']}维修定位",
            content=(
                f"模块故障模式：采集单元：{spec['mode']}\n"
                f"SUR板故障模式：{spec['sur']}\n"
                f"API使用方法：{spec['detect']}\n维修措施：{spec['action']}"
            ),
            metadata={
                "answer_fields": {
                    "module_failure": f"采集单元：{spec['mode']}",
                    "sur_failure": spec["sur"],
                    "api_usage": spec["detect"],
                    "maintenance_action": spec["action"],
                }
            },
        )
        mfl["match_type"] = "structured_query" if index == 1 else "semantic_id_hydration"
        mfl["retrieval_score"] = 0.79 if index == 1 else 0.38
        evidence.extend((fmeca, mfl))
    ids = [item["evidence_id"] for item in evidence]
    query = (
        "现场怎样区分前端线路异常和转换级问题？"
        if comparison
        else "前端线路异常会有什么原因和影响，应该怎么处理？"
    )
    return {
        "query": query,
        "intent": "maintenance_query",
        "query_targets": [{"intent": "maintenance_query"}],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
        "query_frame": {
            "original_query": query,
            "required_answer_fields": [
                "fault_cause",
                "local_impact",
                "final_impact",
                "detection_method",
                "maintenance_action",
            ],
            "comparison_subjects": (
                [
                    {
                        "kind": "failure_mode",
                        "value": "前端线路异常",
                        "basis": "前端线路异常",
                    },
                    {
                        "kind": "failure_mode",
                        "value": "转换级问题",
                        "basis": "转换级问题",
                    },
                ]
                if comparison
                else []
            ),
        },
    }


def _single_fault_state():
    specifications = (
        {
            "record_id": "FM-SYN-038",
            "mode": "突发丢包",
            "cause": "缓存不足、端口拥塞或处理性能下降。",
            "local": "网络转发完整性下降。",
            "final": "任务数据间歇缺失或延迟。",
            "detect": "检查端口统计、队列占用和丢包时间分布。",
            "compensation": "切换冗余链路或使用最近有效数据。",
        },
        {
            "record_id": "FM-SYN-039",
            "mode": "端口链路抖动",
            "cause": "连接器接触不良或物理层器件退化。",
            "local": "网络端口连续性下降。",
            "final": "外部数据通信出现间歇中断。",
            "detect": "检查链路状态时间戳、误码和连接器状态。",
            "compensation": "切换冗余链路或使用最近有效数据。",
        },
    )
    evidence = []
    for spec in specifications:
        evidence.append(
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-single-fmeca",
                record_id=spec["record_id"],
                title=spec["mode"],
                content=(
                    f"故障模式：{spec['mode']}\n故障原因：{spec['cause']}\n"
                    f"局部影响：{spec['local']}\n最终影响：{spec['final']}\n"
                    f"检测方法：{spec['detect']}"
                ),
                metadata={
                    "answer_fields": {
                        "failure_mode": spec["mode"],
                        "fault_cause": spec["cause"],
                        "local_impact": spec["local"],
                        "final_impact": spec["final"],
                        "detection_method": spec["detect"],
                        "compensation_action": spec["compensation"],
                    }
                },
            )
        )
        if spec["record_id"] == "FM-SYN-038":
            evidence.append(
                make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type="maintenance_locator",
                    doc_type="mfl",
                    doc_id="doc-single-mfl",
                    record_id="MFL-SYN-038",
                    title="突发丢包维修定位",
                    content="模块故障模式：网络网关模块：突发丢包",
                    metadata={
                        "answer_fields": {
                            "module_failure": "网络网关模块：突发丢包",
                            "maintenance_action": "限制流量并检查队列配置。",
                        }
                    },
                )
            )
    ids = [item["evidence_id"] for item in evidence]
    return {
        "query": "网络一忙就有成批报文到不了，优先怀疑哪里？",
        "intent": "cause_analysis",
        "query_targets": [{"intent": "cause_analysis"}],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
        "query_frame": {
            "required_answer_fields": [
                "fault_cause",
                "local_impact",
                "final_impact",
                "detection_method",
                "maintenance_action",
            ],
        },
    }


def _broad_multi_mode_state():
    state = _single_fault_state()
    state["query"] = (
        "内部已经产生业务报文，但接在外面的设备始终收不到有效内容，"
        "可能经过哪些环节出错？"
    )
    return state


def _broad_fta_state():
    state = _broad_multi_mode_state()
    state["query_frame"] = {
        "required_answer_fields": ["structural_cause_chain"],
    }
    for item in state["evidence_registry"]:
        if item["doc_type"] == "fmeca":
            item["match_type"] = "exact_fta_reference"
    fta_summary = (
        "FTA 结构：系统无法与外部设备交换有效业务数据；"
        "建模说明：A、B总线作为冗余通道;网关与协议转换模块为串联单点。"
    )
    fta = make_evidence(
        source_kind="postgres",
        authority="validated_relation",
        evidence_level="L2_CORROBORATED",
        claim_type="fta_structure",
        doc_type="fta",
        doc_id="doc-fta",
        record_id="FTA-SYN-003",
        title="系统无法与外部设备交换有效业务数据",
        content=fta_summary,
        display_summary=fta_summary,
        review_status="auto_validated",
    )
    state["evidence_registry"].append(fta)
    ids = [item["evidence_id"] for item in state["evidence_registry"]]
    state["context_evidence_ids"] = ids
    state["safety_decision"]["approved_evidence_ids"] = ids
    state["safety_decision"]["context_evidence_ids"] = ids
    return state


def _fault_fit_state(*, stage="运行中", temporal="持续发生"):
    specifications = (
        {
            "record_id": "FM-FIT-DIRECT",
            "mode": "帧转换校验错误",
            "phase": "正常运行",
            "operating_stage": "运行中",
            "temporal_pattern": "持续发生",
            "impact": "对应外部接口数据中断。",
            "score": 0.01,
        },
        {
            "record_id": "FM-FIT-CONDITIONAL",
            "mode": "转换队列溢出",
            "phase": "高通信负载",
            "operating_stage": "高负载",
            "temporal_pattern": "持续发生",
            "impact": "实时数据延迟或缺失。",
            "score": 0.5,
        },
        {
            "record_id": "FM-FIT-CONTRADICTORY",
            "mode": "突发丢包",
            "phase": "高通信负载",
            "operating_stage": "高负载",
            "temporal_pattern": "间歇发生",
            "impact": "任务数据间歇缺失或延迟。",
            "score": 0.99,
        },
    )
    evidence = []
    for spec in specifications:
        item = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fit",
            record_id=spec["record_id"],
            title=spec["mode"],
            content=f"故障模式：{spec['mode']}；最终影响：{spec['impact']}",
            match_type="exact_fta_reference",
            metadata={
                "answer_fields": {
                    "failure_mode": spec["mode"],
                    "fault_cause": "结构化原因。",
                    "mission_phase": spec["phase"],
                    "operating_stage": spec["operating_stage"],
                    "temporal_pattern": spec["temporal_pattern"],
                    "local_impact": spec["impact"],
                    "final_impact": spec["impact"],
                    "detection_method": "执行结构化检查。",
                }
            },
        )
        item["answer_rerank_score"] = spec["score"]
        evidence.append(item)
    ids = [item["evidence_id"] for item in evidence]
    return {
        "query": "测试问题",
        "intent": "cause_analysis",
        "query_targets": [{"intent": "cause_analysis"}],
        "query_frame": {
            "operating_stage": stage,
            "temporal_pattern": temporal,
        },
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
    }


def _parallel_citations(state):
    return {
        item["record_id"]: number
        for number, item in answer_module._citation_ledger(state).items()
    }


def _first_parallel_branch_answer(state):
    citations = _parallel_citations(state)
    return (
        "前端信号通道开路的原因是连接器松动或输入线路断开；"
        "局部影响是该路输入值不可用，最终会使依赖该输入的状态判断失去依据。"
        f"[^{citations['FM-COMP-1']}]\n\n"
        "判别时检查线路连续性和开路状态；处置时重新固定连接器并修复输入线路。"
        f"[^{citations['FM-COMP-1']}][^{citations['MFL-COMP-1']}]"
    )


def _complete_parallel_answer(state):
    citations = _parallel_citations(state)
    return (
        _first_parallel_branch_answer(state)
        + "\n\n转换级输入饱和的原因是输入超量程或增益配置错误；"
        "局部影响是转换结果固定在上限，最终会使后续监测发生误判。"
        f"[^{citations['FM-COMP-2']}]\n\n"
        "判别时核对输入电平、量程和原始转换值；处置时校正量程与增益，"
        "确认器件异常后更换转换模块。"
        f"[^{citations['FM-COMP-2']}][^{citations['MFL-COMP-2']}]"
    )


class AnswerFaultCompletenessReviewTests(unittest.TestCase):

    def test_explicit_fta_structure_request_excludes_unrelated_fmeca_rows(self):
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L1_DIRECT",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-RESET",
            title="系统运行过程中重复复位",
            content="FTA结构：顶事件通过OR门连接复位信号卡滞与过温保护异常。",
            metadata={"answer_fields": {"structural_cause_chain": "OR门连接下层事件"}},
        )
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-COMM",
            title="A通道发送占用",
            content="发送状态机卡滞会导致业务数据中断。",
            metadata={"answer_fields": {"failure_mode": "A通道发送占用"}},
        )
        state = {
            "query": "运行后设备反复重启，FTA文档把它关联到哪些下层条件？先只解释结构。",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "requested_source_tools": ["fta_query"],
            "query_frame": {
                "required_answer_fields": ["structural_cause_chain"],
                "requested_source_tools": ["fta_query"],
                "response_mode": {
                    "mode": "structure_only",
                    "basis": "FTA文档把它关联到哪些下层条件",
                },
                "source_obligations": [
                    {
                        "tool": "fta_query",
                        "doc_type": "fta",
                        "fields": ["structural_cause_chain"],
                    }
                ],
            },
            "evidence_registry": [fta, fmeca],
        }

        result = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [fta, fmeca]
        )

        self.assertEqual(["FTA-RESET"], [item["record_id"] for item in result])









    def test_task_context_does_not_promote_ungrounded_conditions_or_fta_anchor(self):
        state = _broad_fta_state()
        state["query_frame"] = {
            "operating_stage": "高负载",
            "temporal_pattern": "间歇发生",
        }
        citations = _parallel_citations(state)

        instruction = answer_module._answer_task_instruction(state)
        self.assertIn('"operating_stage":"未知"', instruction)
        self.assertNotIn("发生阶段=高负载", instruction)
        self.assertNotIn(f"[^{citations['FTA-SYN-003']}]", instruction)


    def test_fault_fit_filters_explicit_temporal_conflicts_and_sorts_small_window(self):
        state = _fault_fit_state()

        context = answer_module._context_evidence(state)

        self.assertEqual(
            ["FM-FIT-DIRECT", "FM-FIT-CONDITIONAL"],
            [item["record_id"] for item in context],
        )
        contradictory = next(
            item
            for item in state["evidence_registry"]
            if item["record_id"] == "FM-FIT-CONTRADICTORY"
        )
        self.assertEqual(
            "contradictory",
            _source_answer_fault_rules._fault_candidate_fit(state, contradictory),
        )

    def test_fault_fit_restores_high_load_intermittent_candidate(self):
        state = _fault_fit_state(stage="高负载", temporal="间歇发生")
        burst = next(
            item
            for item in state["evidence_registry"]
            if item["record_id"] == "FM-FIT-CONTRADICTORY"
        )

        self.assertEqual("direct", _source_answer_fault_rules._fault_candidate_fit(state, burst))
        self.assertIn(
            "FM-FIT-CONTRADICTORY",
            [item["record_id"] for item in answer_module._context_evidence(state)],
        )

    def test_fault_fit_unknown_constraints_do_not_remove_exact_fta_rows(self):
        state = _fault_fit_state(stage="未知", temporal="未知")

        self.assertEqual(
            {
                "FM-FIT-DIRECT",
                "FM-FIT-CONDITIONAL",
                "FM-FIT-CONTRADICTORY",
            },
            {item["record_id"] for item in answer_module._context_evidence(state)},
        )

















    def test_incident_fta_window_keeps_only_structure_with_bound_mode(self):
        function = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="function_fact",
            doc_type="function_locator",
            doc_id="doc-function",
            record_id="FUNC-210",
            title="系统启动与复位控制",
            content="启动控制模块负责系统启动与复位。",
            metadata={"answer_fields": {"function_id": "FUNC-210"}},
        )
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-018",
            title="复位信号卡滞",
            content="复位信号卡滞。",
            metadata={
                "answer_fields": {
                    "product_flag": "BCM / FUNC-210",
                    "failure_mode": "复位信号卡滞",
                }
            },
        )
        matching_fta = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta-reset",
            record_id="FTA-RESET",
            title="反复复位故障树",
            content="基本事件：复位信号卡滞。",
        )
        unrelated_fta = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta-data",
            record_id="FTA-DATA",
            title="外部数据输出中断故障树",
            content="基本事件：帧转换校验错误。",
        )
        evidence = [function, fmeca, matching_fta, unrelated_fta]
        state = {
            "query": "设备运行中反复复位，优先排查哪里？",
            "query_frame": {
                "diagnostic_closure": {"required": True, "reason": "当前异常"}
            },
        }

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(state, evidence)

        # An invalid/legacy closure shape carries no current-query authority.
        # Free-text leaf overlap must therefore not delete either FTA record.
        self.assertIn("FTA-RESET", {item["record_id"] for item in filtered})
        self.assertIn("FTA-DATA", {item["record_id"] for item in filtered})










    def test_retiring_field_checklist_keeps_authoritative_actions_in_evidence(self):
        context = answer_module._format_context(_state())
        self.assertIn("MFL-SYN-041", context)
        self.assertIn("检查转换队列和配置", context)

    def test_multi_candidate_task_does_not_require_a_fixed_report_shape(self):
        instruction = answer_module._answer_task_instruction(
            _broad_multi_mode_state()
        )

        self.assertNotIn("最终回答必须全部覆盖", instruction)
        self.assertNotIn("替代排查方向", instruction)
        self.assertNotIn("1～3", instruction)

    def test_planner_comparison_wording_does_not_turn_broad_question_into_explicit_comparison(self):
        state = _broad_multi_mode_state()
        state["rewritten_query"] = "比较多个候选模式并区分各自原因"
        state["query_frame"] = {
            "constraints": ["逐项对比候选证据"],
            "original_query": state["query"],
        }

        self.assertFalse(_source_answer_fault_rules._query_requests_parallel_failure_modes(state))
        self.assertNotIn(state["rewritten_query"], answer_module._answer_task_instruction(state))






    def test_comparison_evidence_keeps_every_mode_branch(self):
        context = answer_module._format_context(_parallel_state())
        self.assertIn("前端信号通道开路", context)
        self.assertIn("转换级输入饱和", context)


if __name__ == "__main__":
    unittest.main()
