import asyncio
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
    return {
        "query": (
            "现场怎样区分前端线路异常和转换级问题？"
            if comparison
            else "前端线路异常会有什么原因和影响，应该怎么处理？"
        ),
        "intent": "maintenance_query",
        "query_targets": [{"intent": "maintenance_query"}],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
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
            "impact": "对应外部接口数据中断。",
            "score": 0.01,
        },
        {
            "record_id": "FM-FIT-CONDITIONAL",
            "mode": "转换队列溢出",
            "phase": "高通信负载",
            "impact": "实时数据延迟或缺失。",
            "score": 0.5,
        },
        {
            "record_id": "FM-FIT-CONTRADICTORY",
            "mode": "突发丢包",
            "phase": "高通信负载",
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
            },
            "evidence_registry": [fta, fmeca],
        }

        result = answer_module._filter_answer_domain_evidence(
            state, [fta, fmeca]
        )

        self.assertEqual(["FTA-RESET"], [item["record_id"] for item in result])

    def test_broad_multi_stage_answer_is_not_expanded_into_repeated_mode_blocks(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            "以下是替代排查方向，不表示两个故障同时发生。\n\n"
            "- **队列与转发**：缓存不足、端口拥塞或处理性能下降时，"
            "会出现任务数据间歇缺失或延迟；先检查队列占用和丢包分布。"
            f"[^{citations['FM-SYN-038']}]\n"
            "- **物理链路**：仅当链路状态也随之波动时，再检查连接器接触不良或物理层退化；"
            "它会造成外部数据通信间歇中断，先核对误码和链路时间戳。"
            f"[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertEqual(draft, result)
        self.assertNotIn("### 故障模式的原因、影响与处理", result)
        self.assertEqual(1, result.count("缓存不足、端口拥塞或处理性能下降"))
        self.assertEqual(1, result.count("连接器接触不良或物理层退化"))

    def test_broad_multi_stage_answer_is_not_reexpanded_by_one_history_link(self):
        state = _broad_multi_mode_state()
        history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-SYN-013",
            title="HIS-SYN-013",
            content="高通信负载下突发丢包，恢复队列配置后正常。",
            metadata={
                "answer_fields": {
                    "explicit_references": {"fmeca": ["FM-SYN-038"]},
                }
            },
        )
        history["match_type"] = "semantic_id_hydration"
        history["retrieval_score"] = 0.9
        state["evidence_registry"].append(history)
        ids = [item["evidence_id"] for item in state["evidence_registry"]]
        state["context_evidence_ids"] = ids
        state["safety_decision"]["approved_evidence_ids"] = ids
        state["safety_decision"]["context_evidence_ids"] = ids
        citations = _parallel_citations(state)
        draft = (
            "- 队列与转发：突发丢包由缓存不足、端口拥塞或处理性能下降引起。"
            f"[^{citations['FM-SYN-038']}]\n"
            "- 物理链路：端口链路抖动由连接器接触不良或物理层器件退化引起。"
            f"[^{citations['FM-SYN-039']}]\n"
            "- 历史线索：曾有队列配置问题。"
            f"[^{citations['HIS-SYN-013']}]"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertEqual(draft, result)
        self.assertNotIn("故障模式的原因、影响与处理", result)

    def test_complete_broad_multi_stage_answer_skips_review_and_mode_expansion(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            "候选是替代排查方向，不表示同时发生。\n\n"
            "- **原因与影响**：缓存不足或端口拥塞会导致任务数据间歇缺失；先检查队列占用和丢包分布。"
            f"[^{citations['FM-SYN-038']}]\n"
            "- **原因与影响**：连接器接触不良会导致外部通信间歇中断；先检查误码和链路时间戳。"
            f"[^{citations['FM-SYN-039']}]"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("完整的多阶段回答不应再触发补写"),
        ):
            result = _run(
                answer_module._review_answer_fault_completeness(draft, state, None)
            )

        self.assertEqual(draft, result)
        self.assertNotIn("### 故障模式的原因、影响与处理", result)

    def test_multiple_cited_modes_receive_non_simultaneous_candidate_framing(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            f"缓存不足会导致数据间歇缺失。[^{citations['FM-SYN-038']}]\n\n"
            f"连接器接触不良会导致通信中断。[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_multi_candidate_framing(draft, state)

        self.assertTrue(
            result.startswith("以下是待排查的候选方向，不表示这些故障同时发生。")
        )
        self.assertEqual(1, result.count("缓存不足会导致数据间歇缺失"))
        self.assertEqual(1, result.count("连接器接触不良会导致通信中断"))

    def test_multi_candidate_framing_survives_unapproved_assertion_guard(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            f"缓存不足会导致数据间歇缺失。[^{citations['FM-SYN-038']}]\n\n"
            f"连接器接触不良会导致通信中断。[^{citations['FM-SYN-039']}]"
        )

        framed = answer_module._ensure_multi_candidate_framing(draft, state)
        result = answer_module._downgrade_unapproved_assertions(framed, state)

        self.assertTrue(result.startswith(answer_module._MULTI_CANDIDATE_FRAMING))
        self.assertEqual(1, result.count(answer_module._MULTI_CANDIDATE_FRAMING))
        self.assertIn("缓存不足会导致数据间歇缺失", result)
        self.assertIn("连接器接触不良会导致通信中断", result)

    def test_numbered_candidate_manifest_detects_branch_local_gaps(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            "以下是待排查的候选方向，不表示这些故障同时发生。\n\n"
            f"1. **队列与转发**[^{citations['FM-SYN-038']}]\n"
            f"   - **影响**：任务数据间歇缺失或延迟。[^{citations['FM-SYN-038']}]\n\n"
            f"2. **物理链路**[^{citations['FM-SYN-039']}]\n\n"
            f"* **网络层**：检查端口统计、队列占用和丢包时间分布。[^{citations['FM-SYN-038']}]"
        )

        missing, manifest = answer_module._fault_completion_manifest(draft, state)

        self.assertEqual({"cause", "impact", "resolution"}, set(missing))
        labels = {item.get("branch_label") for item in manifest if item.get("branch_label")}
        self.assertEqual({"队列与转发", "物理链路"}, labels)

    def test_terminal_guard_rebuilds_hollow_numbered_candidates_in_place(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            "以下是待排查的候选方向，不表示这些故障同时发生。\n\n"
            f"1. **队列与转发**[^{citations['FM-SYN-038']}]\n"
            f"   - **影响**：任务数据间歇缺失或延迟。[^{citations['FM-SYN-038']}]\n\n"
            f"2. **物理链路**[^{citations['FM-SYN-039']}]\n\n"
            f"* **网络层**：检查端口统计、队列占用和丢包时间分布。[^{citations['FM-SYN-038']}]"
        )

        result = answer_module._ensure_numbered_fault_candidate_completeness(
            draft, state
        )
        blocks = answer_module._numbered_fault_candidate_blocks(result, state)

        self.assertEqual(2, len(blocks))
        self.assertTrue(all(not answer_module._numbered_fault_branch_missing_facets(block) for block in blocks))
        self.assertEqual(2, result.count("**为什么可能**"))
        self.assertEqual(2, result.count("**可能影响**"))
        self.assertEqual(2, result.count("**如何确认**"))
        self.assertNotIn("* **网络层**", result)
        physical = next(block["text"] for block in blocks if block["title"] == "物理链路")
        self.assertIn("连接器接触不良或物理层器件退化", physical)
        self.assertNotIn("缓存不足、端口拥塞", physical)

    def test_complete_numbered_candidate_branches_are_left_unchanged(self):
        state = _broad_multi_mode_state()
        citations = _parallel_citations(state)
        draft = (
            f"1. **队列与转发**[^{citations['FM-SYN-038']}]\n"
            "   - 原因是缓存不足、端口拥塞或处理性能下降；"
            "会导致任务数据间歇缺失或延迟；检查端口统计、队列占用和丢包时间分布。"
            f"[^{citations['FM-SYN-038']}]\n\n"
            f"2. **物理链路**[^{citations['FM-SYN-039']}]\n"
            "   - 原因是连接器接触不良或物理层器件退化；"
            "会导致外部数据通信出现间歇中断；检查链路状态时间戳、误码和连接器状态。"
            f"[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_numbered_fault_candidate_completeness(
            draft, state
        )

        self.assertEqual(draft, result)

    def test_broad_path_preflight_preserves_constraints_and_fta_anchor(self):
        state = _broad_fta_state()
        state["query_frame"] = {
            "operating_stage": "高负载",
            "temporal_pattern": "间歇发生",
        }
        citations = _parallel_citations(state)

        instruction = answer_module._fault_preflight_instruction(state)

        self.assertIn("发生阶段=高负载", instruction)
        self.assertIn("发生规律=间歇发生", instruction)
        self.assertIn("与约束不一致的记录", instruction)
        self.assertIn("FTA 结构依据", instruction)
        self.assertIn(f"[^{citations['FTA-SYN-003']}]", instruction)
        self.assertIn("A、B总线作为冗余通道", instruction)

    def test_broad_path_terminal_guard_keeps_validated_fta_structure_visible(self):
        state = _broad_fta_state()
        citations = _parallel_citations(state)
        draft = (
            "以下是待排查的候选方向，不表示这些故障同时发生。\n\n"
            f"1. **队列与转发**[^{citations['FM-SYN-038']}]\n"
            "   - 原因是缓存不足、端口拥塞或处理性能下降；"
            "会导致任务数据间歇缺失或延迟；检查端口统计、队列占用和丢包时间分布。"
            f"[^{citations['FM-SYN-038']}]\n\n"
            f"2. **物理链路**[^{citations['FM-SYN-039']}]\n"
            "   - 原因是连接器接触不良或物理层器件退化；"
            "会导致外部数据通信出现间歇中断；检查链路状态时间戳、误码和连接器状态。"
            f"[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_broad_fta_structure_context(draft, state)

        self.assertIn("**先看结构**", result)
        self.assertIn("A、B总线作为冗余通道；网关与协议转换模块为串联单点", result)
        self.assertEqual(1, result.count(f"[^{citations['FTA-SYN-003']}]"))
        self.assertLess(result.index("**先看结构**"), result.index("1. **队列与转发**"))

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
            answer_module._fault_candidate_fit(state, contradictory),
        )

    def test_fault_fit_restores_high_load_intermittent_candidate(self):
        state = _fault_fit_state(stage="高负载", temporal="间歇发生")
        burst = next(
            item
            for item in state["evidence_registry"]
            if item["record_id"] == "FM-FIT-CONTRADICTORY"
        )

        self.assertEqual("direct", answer_module._fault_candidate_fit(state, burst))
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

    def test_conditional_candidate_exposes_its_source_phase(self):
        state = _fault_fit_state()
        citations = _parallel_citations(state)
        draft = (
            f"1. **协议转换**[^{citations['FM-FIT-DIRECT']}]\n"
            "   - 结构化原因会导致对应外部接口数据中断；执行结构化检查。"
            f"[^{citations['FM-FIT-DIRECT']}]\n\n"
            f"2. **处理队列**[^{citations['FM-FIT-CONDITIONAL']}]\n"
            "   - 结构化原因会导致实时数据延迟或缺失；执行结构化检查。"
            f"[^{citations['FM-FIT-CONDITIONAL']}]"
        )

        result = answer_module._ensure_numbered_fault_candidate_conditions(
            draft,
            state,
        )

        self.assertIn("转换队列溢出仅在高通信负载时再查", result)
        self.assertEqual(1, result.count("**适用条件**"))

    def test_exact_repeated_cited_facts_are_coalesced_without_merging_contexts(self):
        answer = (
            "- 检查队列占用。[^1]\n"
            "- 检查队列占用。[^1]\n\n"
            "### 相似历史\n"
            "- 检查队列占用。[^1]\n"
            "- 检查队列占用。[^1]"
        )

        result = answer_module._coalesce_exact_repeated_cited_lines(answer)

        # 当前建议与历史操作属于不同语境，各保留一份；同一语境内去重。
        self.assertEqual(2, result.count("检查队列占用"))
        self.assertIn("### 相似历史", result)

    def test_citation_repair_drops_children_when_their_entity_parent_is_removed(self):
        lines = ["", "    * 状态机卡滞。[^1]", "* 独立候选。[^2]"]
        units = [
            {
                "unit_id": "u-child",
                "parent_line_indexes": [0],
            },
            {
                "unit_id": "u-independent",
                "parent_line_indexes": [],
            },
        ]
        indexes = {"u-child": 1, "u-independent": 2}

        result = answer_module._drop_units_with_removed_parents(
            lines, units, indexes
        )

        self.assertEqual("", result[1])
        self.assertEqual("* 独立候选。[^2]", result[2])

    def test_single_anchored_fmeca_restores_its_missing_impacts_only(self):
        state = _single_fault_state()
        citations = _parallel_citations(state)
        draft = (
            "优先怀疑突发丢包，原因是缓存不足、端口拥塞或处理性能下降。"
            f"[^{citations['FM-SYN-038']}]\n\n"
            "处置时限制流量并检查队列配置。"
            f"[^{citations['MFL-SYN-038']}]\n\n"
            "另可检查链路状态时间戳、误码和连接器状态。"
            f"[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertIn("网络转发完整性下降", result)
        self.assertIn("任务数据间歇缺失或延迟", result)
        self.assertIn(f"[^{citations['FM-SYN-038']}]", result)
        self.assertNotIn("连接器接触不良或物理层器件退化", result)
        self.assertNotIn("网络端口连续性下降", result)
        self.assertNotIn("外部数据通信出现间歇中断", result)

    def test_single_fallback_does_not_promote_an_unanchored_fmeca_row(self):
        state = _single_fault_state()
        citations = _parallel_citations(state)
        draft = (
            "先限制流量并检查队列配置。"
            f"[^{citations['MFL-SYN-038']}]\n\n"
            "同时检查链路状态时间戳、误码和连接器状态。"
            f"[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertEqual(draft, result)

    def test_cited_history_exact_fmeca_reference_closes_only_that_mode(self):
        state = _single_fault_state()
        history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-SYN-013",
            title="HIS-SYN-013",
            content="高通信负载下突发丢包，恢复队列配置后正常。",
            metadata={
                "answer_fields": {
                    "root_cause": "配置不当而非网络网关硬件退化。",
                    "explicit_references": {
                        "fmeca": ["FM-SYN-038", "FMECA-SYN-038"]
                    },
                }
            },
        )
        history["match_type"] = "semantic_id_hydration"
        history["retrieval_score"] = 0.9
        state["evidence_registry"].append(history)
        ids = [item["evidence_id"] for item in state["evidence_registry"]]
        state["context_evidence_ids"] = ids
        state["safety_decision"]["approved_evidence_ids"] = ids
        state["safety_decision"]["context_evidence_ids"] = ids
        citations = _parallel_citations(state)
        draft = (
            "另可检查链路状态时间戳、误码和连接器状态。"
            f"[^{citations['FM-SYN-039']}]\n\n"
            "历史案例确认是队列配置问题，恢复配置后复测正常。"
            f"[^{citations['HIS-SYN-013']}]"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertIn("突发丢包", result)
        self.assertIn("网络转发完整性下降", result)
        self.assertIn("任务数据间歇缺失或延迟", result)
        self.assertNotIn("连接器接触不良或物理层器件退化", result)
        self.assertNotIn("网络端口连续性下降", result)
        self.assertNotIn("外部数据通信出现间歇中断", result)

    def test_single_anchored_fmeca_is_completed_when_review_times_out(self):
        state = _single_fault_state()
        citations = _parallel_citations(state)
        draft = (
            "优先怀疑突发丢包，原因是缓存不足、端口拥塞或处理性能下降。"
            f"[^{citations['FM-SYN-038']}]"
        )

        def slow_review(*_args, **_kwargs):
            time.sleep(0.03)
            return draft

        with (
            patch.object(
                answer_module,
                "ANSWER_HISTORY_REVIEW_TIMEOUT_SECONDS",
                0.001,
            ),
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=slow_review,
            ),
        ):
            result = _run(
                answer_module._review_answer_fault_completeness(draft, state, None)
            )

        self.assertIn("网络转发完整性下降", result)
        self.assertIn("任务数据间歇缺失或延迟", result)
        self.assertNotIn("网络端口连续性下降", result)

    def test_section_titles_and_bare_mode_labels_do_not_fake_cause_or_impact(self):
        answer = (
            "### 故障原因、影响及处理依据\n\n"
            "* **传感器通道开路**\n"
            "* **A/D 采样饱和**\n\n"
            "建议检查转换队列和配置，确认模块异常后更换协议转换模块。[^2]"
        )

        missing, _manifest = answer_module._fault_completion_manifest(answer, _state())

        self.assertEqual(["cause", "impact"], missing)

    def test_cause_and_later_action_words_do_not_fake_an_impact(self):
        answer = (
            "传感器线路开路会导致采集值固定为满量程码，是当前候选原因。[^1]\n\n"
            "建议检查连接器是否损坏，并按维修记录完成校准。[^2]"
        )

        missing, _manifest = answer_module._fault_completion_manifest(answer, _state())

        self.assertEqual(["impact"], missing)

    def test_each_named_fmeca_mode_needs_its_own_cited_impact(self):
        state = _state()
        saturation = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-059",
            title="A/D 采样饱和",
            content="输入超量程会造成采样结果失真，并可能使健康监测误判。",
            metadata={
                "answer_fields": {
                    "failure_mode": "A/D 采样饱和",
                    "fault_cause": "输入超量程、增益配置错误或转换器异常。",
                    "local_impact": "采样结果失真。",
                    "final_impact": "健康监测或保护可能误判。",
                }
            },
        )
        state["evidence_registry"].append(saturation)
        ids = [item["evidence_id"] for item in state["evidence_registry"]]
        state["context_evidence_ids"] = ids
        state["safety_decision"]["approved_evidence_ids"] = ids
        state["safety_decision"]["context_evidence_ids"] = ids
        answer = (
            "算法异常会导致有效帧无法输出和外部接口数据中断。[^1]\n\n"
            "A/D 采样饱和的原因包括输入超量程、增益配置错误或转换器异常。[^3]\n\n"
            "建议检查转换队列和配置，确认模块异常后更换协议转换模块。[^2]"
        )

        missing, manifest = answer_module._fault_completion_manifest(answer, state)

        self.assertEqual(["impact"], missing)
        saturation_manifest = next(
            item for item in manifest if item["record_id"] == "FM-SYN-059"
        )
        self.assertEqual(["impact"], saturation_manifest["missing_supports"])

    def test_history_root_cause_does_not_replace_current_cause_analysis(self):
        state = _state()
        history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-SYN-013",
            title="HIS-SYN-013",
            content="历史案例根因为软件升级后队列配置错误。",
            metadata={
                "answer_fields": {
                    "occurred_at": "2025-09-09 11:05",
                    "root_cause": "软件升级后队列配置错误。",
                    "final_solution": "恢复受控队列参数。",
                    "verification_result": "高负载回放无丢包。",
                }
            },
        )
        history["match_type"] = "semantic_id_hydration"
        history["retrieval_score"] = 0.9
        state["evidence_registry"].append(history)
        ids = [item["evidence_id"] for item in state["evidence_registry"]]
        state["context_evidence_ids"] = ids
        state["safety_decision"]["approved_evidence_ids"] = ids
        state["safety_decision"]["context_evidence_ids"] = ids
        draft = (
            "影响：有效帧无法输出，对应外部接口数据中断。[^1]\n\n"
            "建议检查转换队列和配置。[^2]\n\n"
            "历史案例的根因是软件升级后队列配置错误。[^3]"
        )

        missing, _manifest = answer_module._fault_completion_manifest(draft, state)

        self.assertEqual(["cause"], missing)

    def test_manifest_detects_evidence_backed_missing_resolution(self):
        answer = "算法异常会导致有效帧无法输出和外部接口数据中断。[^1]"

        missing, manifest = answer_module._fault_completion_manifest(answer, _state())

        self.assertEqual(["resolution"], missing)
        self.assertTrue(any("resolution" in item["supports"] for item in manifest))

    def test_current_incident_requires_documented_action_not_only_detection(self):
        state = _state()
        state["query_frame"] = {
            "diagnostic_closure": {
                "required": True,
                "reason": "用户报告当前异常并询问排查处置",
            }
        }
        mfl = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "mfl"
        )
        fmeca = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "fmeca"
        )
        fmeca["metadata"]["answer_fields"]["failure_mode"] = "帧转换校验错误"
        mfl["metadata"]["answer_fields"]["module_failure"] = "协议转换模块：帧转换校验错误"
        draft = (
            "帧转换校验错误可能由算法异常引起，并导致有效帧无法输出和外部接口数据中断。[^1] "
            "可先对比输入输出帧和校验过程。[^1]"
        )

        missing, _manifest = answer_module._fault_completion_manifest(draft, state)
        completed = answer_module._ensure_current_incident_actions(draft, state)

        self.assertIn("action", missing)
        self.assertIn("帧转换校验错误的文档处置", completed)
        self.assertIn("检查转换队列和配置，确认模块异常后更换协议转换模块", completed)
        self.assertIn("[^2]", completed)

    def test_abstract_explanation_does_not_append_repair_contract(self):
        state = _state()
        mfl = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "mfl"
        )
        mfl["metadata"]["answer_fields"]["module_failure"] = "协议转换模块：帧转换校验错误"
        draft = "算法异常会导致有效帧无法输出和外部接口数据中断。[^1]"

        self.assertEqual(
            draft,
            answer_module._ensure_current_incident_actions(draft, state),
        )

    def test_incident_action_can_extend_bounded_prompt_ledger(self):
        state = _state()
        state["query_frame"] = {
            "diagnostic_closure": {
                "required": True,
                "reason": "用户报告当前异常并询问排查处置",
            }
        }
        fmeca = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "fmeca"
        )
        fmeca["metadata"]["answer_fields"]["failure_mode"] = "帧转换校验错误"
        mfl = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "mfl"
        )
        mfl["metadata"]["answer_fields"]["module_failure"] = "协议转换模块：帧转换校验错误"
        prompt = [{"id": 1, **answer_module._evidence_payload(fmeca)}]
        draft = "帧转换校验错误可能由算法异常引起。[^1]"

        completed, extended = answer_module._extend_incident_action_ledger(
            draft,
            prompt,
            state,
        )

        self.assertEqual(2, len(extended))
        self.assertEqual("MFL-SYN-041", extended[-1]["record_id"])
        self.assertIn("帧转换校验错误的文档处置", completed)
        self.assertIn("[^2]", completed)

    def test_public_compaction_keeps_current_incident_action(self):
        state = _state()
        state["query_frame"] = {
            "diagnostic_closure": {
                "required": True,
                "reason": "用户报告当前异常并询问排查处置",
            }
        }
        fmeca = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "fmeca"
        )
        fmeca["metadata"]["answer_fields"]["failure_mode"] = "帧转换校验错误"
        mfl = next(
            item for item in state["evidence_registry"] if item["doc_type"] == "mfl"
        )
        mfl["metadata"]["answer_fields"]["module_failure"] = "协议转换模块：帧转换校验错误"
        prompt = [{"id": 1, **answer_module._evidence_payload(fmeca)}]
        draft = "帧转换校验错误可能由算法异常引起。[^1]"
        completed, extended = answer_module._extend_incident_action_ledger(
            draft, prompt, state
        )

        final, citations, _catalog = answer_module._compact_answer_citations(
            completed, extended, state
        )

        self.assertIn("帧转换校验错误的文档处置", final)
        self.assertIn("检查转换队列和配置，确认模块异常后更换协议转换模块", final)
        self.assertEqual("MFL-SYN-041", citations[-1]["record_id"])

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

        filtered = answer_module._filter_answer_domain_evidence(state, evidence)

        self.assertIn("FTA-RESET", {item["record_id"] for item in filtered})
        self.assertNotIn("FTA-DATA", {item["record_id"] for item in filtered})

    def test_current_incident_hmc_heading_is_completed_as_priority_card(self):
        """An HMC-cited candidate may not survive as an empty ``1.`` item."""

        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-reset-fmeca",
            record_id="FM-SYN-018",
            title="复位信号卡滞",
            content="复位信号卡滞。",
            metadata={
                "answer_fields": {
                    "failure_mode": "复位信号卡滞",
                    "fault_cause": "复位驱动器件失效、控制逻辑异常或噪声耦合。",
                    "final_impact": "系统无法启动或出现反复自动重启。",
                    "detection_method": "监测复位线、看门狗状态和复位源记录。",
                }
            },
        )
        hmc = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_action",
            doc_type="hmc",
            doc_id="doc-reset-hmc",
            record_id="91820001**0003",
            title="91820001**0003 维修方法",
            content="BCM 复位信号卡滞。",
            metadata={
                "answer_fields": {
                    "fault_symptom": "BCM 复位信号卡滞",
                    "maintenance_action": "隔离复位源并检查驱动电路，确认后更换启动控制模块。",
                }
            },
        )
        evidence = [fmeca, hmc]
        ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "设备正常运行时反复自动复位，优先查哪里？",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query_frame": {"diagnostic_closure": {"required": True}},
            "evidence_registry": evidence,
            "context_evidence_ids": ids,
            "safety_decision": {
                "approved_evidence_ids": ids,
                "context_evidence_ids": ids,
            },
        }
        citations = _parallel_citations(state)
        draft = (
            f"1. **启动控制模块（BCM）的复位信号卡滞**[^{citations['91820001**0003']}]\n\n"
            "1. **其他供电候选**\n"
        )

        result = answer_module._ensure_approved_single_fault_completeness(
            draft, state
        )

        self.assertIn("**优先定位：启动控制模块（BCM）的复位信号卡滞**", result)
        self.assertNotIn("1. **启动控制模块", result)
        self.assertIn("复位驱动器件失效、控制逻辑异常或噪声耦合", result)
        self.assertIn("系统无法启动或出现反复自动重启", result)
        self.assertIn("监测复位线、看门狗状态和复位源记录", result)
        self.assertIn("隔离复位源并检查驱动电路，确认后更换启动控制模块", result)
        self.assertEqual(1, result.count("隔离复位源并检查驱动电路"))

    def test_current_incident_removes_fta_only_detail_card_and_closes_numbers(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-DIRECT",
            title="B路间歇停机",
            content="B路间歇停机。",
            metadata={"answer_fields": {"failure_mode": "B路间歇停机"}},
        )
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-RESET",
            title="反复复位故障树",
            content="基本事件：A路间歇停机。",
        )
        evidence = [fmeca, fta]
        ids = [item["evidence_id"] for item in evidence]
        state = {
            "query_frame": {"diagnostic_closure": {"required": True}},
            "evidence_registry": evidence,
            "context_evidence_ids": ids,
            "safety_decision": {
                "approved_evidence_ids": ids,
                "context_evidence_ids": ids,
            },
        }
        citations = _parallel_citations(state)
        draft = (
            f"2. **B路稳压供电间歇停机**[^{citations['FM-DIRECT']}]\n"
            "   - **原因**：内部连接松动。"
            f"[^{citations['FM-DIRECT']}]\n\n"
            f"3. **A路间歇停机**[^{citations['FTA-RESET']}]\n"
            "   - **处置**：隔离复位源并更换模块。"
            f"[^{citations['FTA-RESET']}]"
        )

        result = answer_module._strip_fta_only_current_incident_candidate_cards(
            draft, state
        )
        result = answer_module._renumber_current_incident_candidate_cards(
            result, state
        )

        self.assertIn("1. **B路稳压供电间歇停机**", result)
        self.assertNotIn("A路间歇停机", result)
        self.assertNotIn("隔离复位源并更换模块", result)

    def test_current_incident_deduplicates_terminal_document_action(self):
        state = {
            "query_frame": {"diagnostic_closure": {"required": True}}
        }
        action = "隔离复位源并检查驱动电路，确认后更换启动控制模块。"
        draft = (
            f"  - **文档处置**：{action}[^2]\n\n"
            f"- **复位信号卡滞的文档处置**：{action}[^6]"
        )

        result = answer_module._deduplicate_current_incident_action_lines(draft, state)

        self.assertEqual(1, result.count("隔离复位源并检查驱动电路"))

    def test_current_incident_merges_same_mode_supplement_into_card(self):
        state = _single_fault_state()
        state["query_frame"] = {"diagnostic_closure": {"required": True}}
        citations = _parallel_citations(state)
        fmeca_citation = citations["FM-SYN-038"]
        draft = (
            f"**优先定位：网络网关模块突发丢包**[^{fmeca_citation}]\n"
            "  - **原因**：缓存不足、端口拥塞或处理性能下降。"
            f"[^{fmeca_citation}]\n\n"
            f"- **故障模式：突发丢包**[^{fmeca_citation}]\n"
            "  - **检测/判别**：检查端口统计、队列占用和丢包时间分布。"
            f"[^{fmeca_citation}]\n"
            "  - **使用补偿**：切换冗余链路或使用最近有效数据。"
            f"[^{fmeca_citation}]\n\n"
            "实际处置以适用构型、有效版本和正式维修程序为准。\n\n"
            "- **突发丢包的使用补偿**：切换冗余链路或使用最近有效数据。"
            f"[^{fmeca_citation}]"
        )

        result = answer_module._merge_current_incident_mode_supplements(draft, state)

        self.assertNotIn("**故障模式：突发丢包**", result)
        self.assertNotIn("**突发丢包的使用补偿**", result)
        self.assertIn("**如何确认**：检查端口统计、队列占用和丢包时间分布", result)
        self.assertIn("**可采取处置**：切换冗余链路或使用最近有效数据", result)
        self.assertIn("**可选补偿**：切换冗余链路或使用最近有效数据", result)

    def test_current_incident_merges_same_mode_hmc_card_into_fmeca_card(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-BCA",
            title="A通道发送占用",
            content="A通道发送占用。",
            metadata={
                "answer_fields": {
                    "failure_mode": "A通道发送占用",
                    "fault_cause": "发送状态机卡滞。",
                    "final_impact": "可能阻塞总线。",
                }
            },
        )
        hmc = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_action",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="HMC-BCA",
            title="HMC-BCA",
            content="A通道发送占用。",
            metadata={
                "answer_fields": {
                    "fault_symptom": "BCA A通道发送占用",
                    "maintenance_action": "隔离A通道并执行控制器复位。",
                }
            },
        )
        state = {
            "query": "外部设备收不到数据，区分并给出维修动作",
            "intent": "cause_analysis",
            "query_frame": {"diagnostic_closure": {"required": True}},
            "evidence_registry": [fmeca, hmc],
            "context_evidence_ids": [fmeca["evidence_id"], hmc["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [fmeca["evidence_id"], hmc["evidence_id"]],
                "context_evidence_ids": [fmeca["evidence_id"], hmc["evidence_id"]],
            },
        }
        draft = (
            "1. **A通道发送占用**[^1]\n"
            "   - **原因**：发送状态机卡滞。[^1]\n\n"
            "2. **A通道发送占用（对应发送异常现象）**[^2]\n"
            "   - **文档处置**：隔离A通道并执行控制器复位。[^2]"
        )

        result = answer_module._deduplicate_current_incident_candidate_modes(
            draft, state
        )

        self.assertEqual(1, result.count("A通道发送占用"))
        self.assertIn("隔离A通道并执行控制器复位", result)

    def test_current_incident_fmeca_card_drops_mixed_fta_sibling_detail(self):
        state = _single_fault_state()
        state["query_frame"] = {"diagnostic_closure": {"required": True}}
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-SYN-001",
            title="网络故障树",
            content="基本事件：端口链路抖动。",
        )
        state["evidence_registry"].append(fta)
        ids = [item["evidence_id"] for item in state["evidence_registry"]]
        state["context_evidence_ids"] = ids
        state["safety_decision"]["approved_evidence_ids"] = ids
        state["safety_decision"]["context_evidence_ids"] = ids
        citations = _parallel_citations(state)
        draft = (
            f"2. **网络网关模块突发丢包**[^{citations['FM-SYN-038']}]\n"
            "   - **原因**：端口链路抖动导致本事件。"
            f"[^{citations['FTA-SYN-001']}]"
        )

        result = answer_module._normalize_current_incident_fmeca_cards(draft, state)

        self.assertNotIn("端口链路抖动导致本事件", result)
        self.assertNotIn(f"[^{citations['FTA-SYN-001']}]", result)
        self.assertIn("缓存不足、端口拥塞或处理性能下降", result)
        self.assertIn("任务数据间歇缺失或延迟", result)
        self.assertIn("检查端口统计、队列占用和丢包时间分布", result)
        self.assertIn("切换冗余链路或使用最近有效数据", result)

    def test_current_incident_candidate_lead_count_matches_visible_cards(self):
        state = _single_fault_state()
        state["query_frame"] = {"diagnostic_closure": {"required": True}}
        citations = _parallel_citations(state)
        draft = (
            "应优先定位以下三个板级环节：\n\n"
            f"**优先定位：突发丢包**[^{citations['FM-SYN-038']}]\n\n"
            f"1. **端口链路抖动**[^{citations['FM-SYN-039']}]"
        )

        result = answer_module._align_current_incident_candidate_count(draft, state)

        self.assertIn("以下两个板级环节", result)
        self.assertNotIn("以下三个板级环节", result)

    def test_llm_review_adds_documented_check_and_action(self):
        draft = "算法异常会导致有效帧无法输出和外部接口数据中断。[^1]"
        complete = (
            "算法异常会导致有效帧无法输出和外部接口数据中断。[^1]\n\n"
            "建议对比输入输出帧并检查转换队列和配置；确认模块异常后再更换协议转换模块。[^1][^2]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(answer_module._review_answer_fault_completeness(draft, _state(), None))

        call.assert_called_once()
        self.assertEqual(complete, result)

    def test_complete_fault_answer_skips_review(self):
        answer = (
            "算法异常会导致有效帧无法输出和外部接口数据中断。[^1] "
            "建议检查转换队列和配置，确认模块异常后更换协议转换模块。[^2]"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("完整回答不应重复校对"),
        ):
            result = _run(answer_module._review_answer_fault_completeness(answer, _state(), None))

        self.assertEqual(answer, result)

    def test_first_answer_prompt_gets_evidence_backed_closure_checklist(self):
        instruction = answer_module._fault_preflight_instruction(_state())

        self.assertIn("原因、影响、检查/处置", instruction)
        self.assertIn("MFL-SYN-041", instruction)
        self.assertIn("检查转换队列和配置", instruction)

    def test_multi_candidate_preflight_requests_compact_diagnostic_path(self):
        instruction = answer_module._fault_preflight_instruction(
            _broad_multi_mode_state()
        )

        self.assertIn("替代排查方向", instruction)
        self.assertIn("不表示这些故障同时发生", instruction)
        self.assertIn("1～3 个编号主分支", instruction)
        self.assertIn("按实际流程排序", instruction)
        self.assertIn("最有区分力的检查", instruction)
        self.assertIn("禁止在分阶段解释后再追加故障模式清单", instruction)

    def test_planner_comparison_wording_does_not_turn_broad_question_into_explicit_comparison(self):
        state = _broad_multi_mode_state()
        state["rewritten_query"] = "比较多个候选模式并区分各自原因"
        state["query_frame"] = {
            "constraints": ["逐项对比候选证据"],
            "original_query": state["query"],
        }

        self.assertFalse(answer_module._query_requests_parallel_failure_modes(state))
        self.assertIn(
            "不表示这些故障同时发生",
            answer_module._fault_preflight_instruction(state),
        )

    def test_explicit_comparison_requires_every_approved_mode_pair(self):
        state = _parallel_state()

        missing, manifest = answer_module._fault_completion_manifest(
            _first_parallel_branch_answer(state), state
        )

        self.assertEqual(["cause", "impact", "resolution"], missing)
        second_branch = [
            item for item in manifest if item.get("branch_mode") == "转换级输入饱和"
        ]
        self.assertEqual(
            {"FM-COMP-2", "MFL-COMP-2"},
            {item["record_id"] for item in second_branch},
        )
        self.assertEqual(
            ["cause", "impact", "resolution"],
            next(
                item["missing_supports"]
                for item in second_branch
                if item["doc_type"] == "fmeca"
            ),
        )

    def test_comparison_review_restores_mode_omitted_by_initial_draft(self):
        state = _parallel_state()
        draft = _first_parallel_branch_answer(state)
        complete = _complete_parallel_answer(state)

        with patch.object(
            answer_module.llm_service, "_call_model", return_value=complete
        ) as call:
            result = _run(
                answer_module._review_answer_fault_completeness(draft, state, None)
            )

        call.assert_called_once()
        self.assertEqual(complete, result)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("前端信号通道开路", prompt)
        self.assertIn("转换级输入饱和", prompt)
        self.assertIn("校正量程与增益", prompt)

    def test_comparison_fallback_restores_branch_after_two_llm_omissions(self):
        state = _parallel_state()
        draft = _first_parallel_branch_answer(state)

        with patch.object(
            answer_module.llm_service, "_call_model", return_value=draft
        ) as call:
            result = _run(
                answer_module._review_answer_fault_completeness(draft, state, None)
            )

        self.assertEqual(2, call.call_count)
        self.assertIn("**故障模式：转换级输入饱和**", result)
        self.assertIn("输入超量程或增益配置错误", result)
        self.assertIn("转换结果固定在上限", result)
        self.assertIn("后续监测可能发生误判", result)
        self.assertIn("核对输入电平、量程和原始转换值", result)
        self.assertIn("校正量程与增益，确认器件异常后更换转换模块", result)
        self.assertEqual(1, result.count("连接器松动或输入线路断开"))
        self.assertNotIn("answer_fields", result)
        self.assertNotIn("fault_cause", result)

    def test_comparison_fallback_restores_branch_when_llm_review_times_out(self):
        state = _parallel_state()
        draft = _first_parallel_branch_answer(state)

        def slow_review(*_args, **_kwargs):
            time.sleep(0.03)
            return draft

        with (
            patch.object(
                answer_module,
                "ANSWER_HISTORY_REVIEW_TIMEOUT_SECONDS",
                0.001,
            ),
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=slow_review,
            ) as call,
        ):
            result = _run(
                answer_module._review_answer_fault_completeness(draft, state, None)
            )

        call.assert_called_once()
        self.assertIn("**故障模式：转换级输入饱和**", result)
        self.assertIn("输入超量程或增益配置错误", result)
        self.assertIn("转换结果固定在上限", result)
        self.assertIn("后续监测可能发生误判", result)
        self.assertIn("核对输入电平、量程和原始转换值", result)
        self.assertIn("校正量程与增益，确认器件异常后更换转换模块", result)
        self.assertNotIn("answer_fields", result)
        self.assertNotIn("fault_cause", result)

    def test_noncomparison_does_not_force_every_retrieved_mode_pair(self):
        state = _parallel_state(comparison=False)

        missing, manifest = answer_module._fault_completion_manifest(
            _first_parallel_branch_answer(state), state
        )

        self.assertEqual([], missing)
        self.assertEqual([], manifest)

    def test_comparison_preflight_names_every_mode_branch(self):
        instruction = answer_module._fault_preflight_instruction(_parallel_state())

        self.assertIn("必须逐支回答", instruction)
        self.assertIn("前端信号通道开路", instruction)
        self.assertIn("转换级输入饱和", instruction)


if __name__ == "__main__":
    unittest.main()
