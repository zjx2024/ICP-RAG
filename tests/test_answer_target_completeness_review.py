import backend.agent.answer_contract as _source_answer_contract
import backend.agent.nodes.answer_domain_scope as _source_answer_domain_scope
import backend.agent.nodes.answer_evidence_window as _source_answer_evidence_window
import backend.agent.nodes.answer_history_rules as _source_answer_history_rules
import backend.agent.nodes.answer_payload as _source_answer_payload
import backend.agent.nodes.answer_record_scope as _source_answer_record_scope
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from backend.agent.contracts import make_evidence
from backend.agent.answer_contract import incremental_followup_contract
from backend.agent.nodes import answer as answer_module
from backend.agent.rbd_contract import (
    RBD_SINGLE_FAILURE_RELATION_VERSION,
    bind_rbd_models,
    selected_rbd_anchor_selectors,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def _approved_state(*evidence, intent, targets, query="用户问题"):
    items = list(evidence)
    ids = [item["evidence_id"] for item in items]
    return {
        "query": query,
        "intent": intent,
        "query_targets": [{"intent": value} for value in targets],
        "evidence_registry": items,
        "context_evidence_ids": list(ids),
        "safety_decision": {
            "approved_evidence_ids": list(ids),
            "context_evidence_ids": list(ids),
        },
    }


def _with_required_fields(state, *fields):
    state["query_frame"] = {
        "coverage_scope": "focused_detail",
        "required_answer_fields": list(fields),
    }
    return state


def _data_output_parallel_vertices():
    return [
        {
            "id": "data-a",
            "properties": {
                "connection_type": "parallel",
                "branch_id": "root.data:branch:1",
                "module_id": "MOD-DATA-A",
                "name": "A路数据输出",
            },
        },
        {
            "id": "data-b",
            "properties": {
                "connection_type": "parallel",
                "branch_id": "root.data:branch:2",
                "module_id": "MOD-DATA-B",
                "name": "B路数据输出",
            },
        },
    ]


def _data_output_single_failure_relation():
    return {
        "contract_version": RBD_SINGLE_FAILURE_RELATION_VERSION,
        "relation_type": "single_failure_survivability",
        "revision_id": "RBD-DATA-OUTPUT",
        "validation_status": "auto_validated",
        "derivation": "deterministic_topology_validation",
        "outcome": "success_criterion_satisfied",
        "group_id": "root.data",
        "failed_business_ids": ["MOD-DATA-A"],
        "surviving_business_ids": ["MOD-DATA-B"],
        "statement": "任一路失效时，另一条路径仍可维持数据输出",
    }


def _authorize_data_output_topology(state):
    state["topology_dependency"] = {
        "required": True,
        "subject": "A路数据输出",
        "subject_kind": "data_path",
        "failure_state": "失效",
        "question_aspects": ["continued_operation"],
        "canonical_business_ids": ["MOD-DATA-A"],
    }
    rbd_items = [
        item
        for item in state.get("evidence_registry") or []
        if item.get("doc_type") == "rbd" and item.get("claim_type") == "rbd_structure"
    ]
    anchors = [
        {
            "doc_type": "rbd",
            "doc_id": item.get("doc_id"),
            "record_id": item.get("record_id"),
            "revision_id": (item.get("metadata") or {}).get("revision_id"),
            "diagram_key": (item.get("metadata") or {}).get("diagram_key"),
        }
        for item in rbd_items
    ]
    _, state["rbd_query_binding"] = bind_rbd_models(
        rbd_items,
        required=True,
        raw_user_query=state.get("query"),
        anchor_selectors=selected_rbd_anchor_selectors(anchors),
        allow_multiple=True,
    )
    state["selected_conversation_anchors"] = anchors
    return state


def _evidence(
    *,
    doc_type,
    record_id,
    claim_type,
    answer_fields=None,
    content="仅供测试的来源正文",
    source_kind="postgres",
    authority="authoritative_record",
):
    metadata = {"answer_fields": answer_fields or {}}
    review_status = "auto_validated" if source_kind == "hugegraph" else "not_required"
    if doc_type == "rbd" and claim_type == "rbd_structure":
        metadata.update(
            {
                "revision_id": record_id,
                "diagram_key": record_id,
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            }
        )
        review_status = "auto_validated"
    return make_evidence(
        source_kind=source_kind,
        authority=authority,
        evidence_level="L1_DIRECT" if source_kind == "postgres" else "L2_CORROBORATED",
        claim_type=claim_type,
        doc_type=doc_type,
        doc_id=f"doc-{doc_type}",
        record_id=record_id,
        title=record_id,
        content=content,
        display_summary=content,
        review_status=review_status,
        metadata=metadata,
    )


class AnswerTargetCompletenessReviewTests(unittest.TestCase):

    def test_incremental_contract_allows_bounded_multi_source_followup(self):
        state = {
            "query": "先如何检查？旧案例在什么条件出现并如何验证恢复？",
            "query_targets": [
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "calculation_requests": [],
            "query_frame": {
                "original_query": "先如何检查？旧案例在什么条件出现并如何验证恢复？",
                "coverage_scope": "focused_detail",
                "prior_delivered_fields": ["fault_cause"],
                "question_semantics": "act",
                "response_scope": "incremental",
                "required_answer_fields": [
                    "detection_method",
                    "maintenance_action",
                    "history_steps",
                    "history_result",
                ],
                "diagnostic_closure": {"required": True, "reason": "当前异常"},
            },
        }

        self.assertTrue(incremental_followup_contract(state))
        self.assertTrue(_source_answer_contract.focused_answer_contract(state))

        state["query"] = "请从头完整重述诊断和全部处置。"
        state["query_frame"]["original_query"] = state["query"]
        state["query_frame"]["response_scope"] = "full_restatement"
        self.assertFalse(incremental_followup_contract(state))

    def test_focused_structured_binding_removes_sibling_locator_and_fmeca_rows(self):
        target_locator = _evidence(
            doc_type="function_locator",
            record_id="MOD-NAV-A",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-NAV-A",
                "component_name": "导航处理单元",
                "description": "保存航路参数并执行完整性校验。",
            },
        )
        sibling_locator = _evidence(
            doc_type="function_locator",
            record_id="MOD-NAV-B",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-NAV-B",
                "component_name": "备用导航单元",
                "description": "承担备用计算。",
            },
        )
        target_mode = _evidence(
            doc_type="fmeca",
            record_id="FM-NAV-1",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "NAV-A / MOD-NAV-A",
                "failure_mode": "参数读出失效",
                "local_impact": "当前参数无法读取。",
            },
        )
        sibling_mode = _evidence(
            doc_type="fmeca",
            record_id="FM-NAV-2",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "NAV-B / MOD-NAV-B",
                "failure_mode": "备用计算超时",
                "local_impact": "备用结果延迟。",
            },
        )

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            {
                "query": "它停用后先影响什么？",
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["local_impact"],
                },
                "structured_entity_binding": {
                    "bound": True,
                    "business_ids": ["MOD-NAV-A"],
                },
            },
            [target_locator, sibling_locator, target_mode, sibling_mode],
        )

        self.assertEqual(
            {target_locator["evidence_id"], target_mode["evidence_id"]},
            {item["evidence_id"] for item in filtered},
        )

    def test_exact_abbreviation_bridges_component_binding_without_name_similarity(self):
        target = _evidence(
            doc_type="fmeca",
            record_id="FM-NAV-1",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "NIU / FUNC-710",
                "failure_mode": "输入滤波退化",
                "local_impact": "输入纹波增大。",
            },
        )
        sibling = _evidence(
            doc_type="fmeca",
            record_id="FM-NAV-2",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "NOU / FUNC-720",
                "failure_mode": "输出驱动退化",
                "local_impact": "输出幅值降低。",
            },
        )

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            {
                "query": "该输入单元失效后有什么影响？",
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["local_impact"],
                },
                "structured_entity_binding": {
                    "bound": True,
                    "business_ids": ["MOD-NAV-NIU"],
                    "terms": ["MOD-NAV-NIU", "NIU"],
                },
            },
            [target, sibling],
        )

        self.assertEqual([target["evidence_id"]], [item["evidence_id"] for item in filtered])

    def test_selected_fmeca_anchor_does_not_bind_mfl_by_equal_fault_text(self):
        target = _evidence(
            doc_type="fmeca",
            record_id="FM-COM-1",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "CRX / FUNC-810",
                "failure_mode": "接收链路热间歇",
                "detection_method": "关联温度与接收计数。",
            },
        )
        target["match_type"] = "exact_conversation_reference"
        sibling = _evidence(
            doc_type="fmeca",
            record_id="FM-COM-9",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "CTX / FUNC-820",
                "failure_mode": "发送队列拥塞",
                "detection_method": "检查发送队列深度。",
            },
        )
        sibling["match_type"] = "semantic_id_hydration"
        good_mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-COM-1",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "接收单元：接收链路热间歇",
                "maintenance_action": "检查连接并执行热循环。",
            },
        )
        bad_mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-COM-9",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "发送单元：发送队列拥塞",
                "maintenance_action": "清理发送队列。",
            },
        )

        state = {
            "query": "这个现象先如何验证和处置？",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["detection_method", "maintenance_action"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-COM-1"}
            ],
        }
        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [target, sibling, good_mfl, bad_mfl]
        )

        self.assertEqual(
            {target["evidence_id"]},
            {item["evidence_id"] for item in filtered},
        )

    def test_unique_focused_history_links_override_prior_candidate_anchors(self):
        linked_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-101",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "UNIT-X / FUNC-101",
                "failure_mode": "输入状态间歇丢失",
                "detection_method": "对照输入计数与状态快照。",
            },
        )
        prior_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-202",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "UNIT-Y / FUNC-202",
                "failure_mode": "输出队列持续阻塞",
                "detection_method": "检查输出队列深度。",
            },
        )
        linked_mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-GEN-101",
            claim_type="maintenance_locator",
            answer_fields={
                "record_code": "MFL-GEN-101",
                "module_failure": "输入单元：输入状态间歇丢失",
                "maintenance_action": "检查输入连接并复测计数。",
            },
        )
        prior_mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-GEN-202",
            claim_type="maintenance_locator",
            answer_fields={
                "record_code": "MFL-GEN-202",
                "module_failure": "输出单元：输出队列持续阻塞",
                "maintenance_action": "检查队列配置并复测输出。",
            },
        )
        for item in (linked_fmeca, prior_fmeca, linked_mfl, prior_mfl):
            item["match_type"] = "exact_conversation_reference"
        history = _evidence(
            doc_type="fault_history",
            record_id="HIS-GEN-101",
            claim_type="historical_case",
            answer_fields={
                "case_id": "HIS-GEN-101",
                "fault_symptom": "输入计数停止变化后状态短暂丢失。",
                "steps": ["保存快照并检查输入连接。"],
                "verification_result": "连续复测后输入计数保持变化。",
                "explicit_references": {
                    "fmeca": ["FM-GEN-101"],
                    "mfl": ["MFL-GEN-101"],
                },
            },
        )
        history["match_type"] = "structured_query"
        state = {
            "query": "现场先怎么检查处置？以前相近案例在什么条件出现并如何验证恢复？",
            "intent": "maintenance_query",
            "query_targets": [
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "original_query": "现场先怎么检查处置？以前相近案例在什么条件出现并如何验证恢复？",
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "detection_method",
                    "maintenance_action",
                    "history_steps",
                    "history_result",
                ],
            },
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["his-gen-101"],
                "row_ids": [],
                "source": "current_turn_selection",
                "authority": "identity",
                "comparison": False,
                "ambiguous": False,
                "conflict": False,
                "authorizes_exact_links": True,
            },
            "selected_conversation_anchors": [
                {
                    "anchor_id": "ev-history-101",
                    "doc_type": "fault_history",
                    "record_id": "HIS-GEN-101",
                },
                {"doc_type": "fmeca", "record_id": "FM-GEN-101"},
                {"doc_type": "fmeca", "record_id": "FM-GEN-202"},
                {"doc_type": "mfl", "record_id": "MFL-GEN-202"},
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state,
            [prior_mfl, prior_fmeca, history, linked_mfl, linked_fmeca],
        )

        retained_ids = {item["record_id"] for item in retained}
        self.assertEqual(
            {"HIS-GEN-101", "FM-GEN-101", "MFL-GEN-101"},
            retained_ids,
        )

    def test_current_explicit_ids_are_preserved_as_history_link_conflicts(self):
        def table_row(doc_type, record_id, mode):
            fields = (
                {
                    "product_flag": f"UNIT-{record_id[-3:]} / FUNC-{record_id[-3:]}",
                    "failure_mode": mode,
                    "detection_method": f"验证 {mode}。",
                }
                if doc_type == "fmeca"
                else {
                    "record_code": record_id,
                    "module_failure": mode,
                    "maintenance_action": f"处理 {mode}。",
                }
            )
            item = _evidence(
                doc_type=doc_type,
                record_id=record_id,
                claim_type="fmeca_fact" if doc_type == "fmeca" else "maintenance_locator",
                answer_fields=fields,
            )
            item["match_type"] = "exact_conversation_reference"
            return item

        linked_fmeca = table_row("fmeca", "FM-GEN-301", "输入计数冻结")
        explicit_fmeca = table_row("fmeca", "FM-GEN-302", "输入采样偏移")
        stale_fmeca = table_row("fmeca", "FM-GEN-303", "输出时序异常")
        linked_mfl = table_row("mfl", "MFL-GEN-301", "输入计数冻结")
        explicit_mfl = table_row("mfl", "MFL-GEN-302", "输入采样偏移")
        stale_mfl = table_row("mfl", "MFL-GEN-303", "输出时序异常")
        history = _evidence(
            doc_type="fault_history",
            record_id="HIS-GEN-301",
            claim_type="historical_case",
            answer_fields={
                "case_id": "HIS-GEN-301",
                "fault_symptom": "输入计数在复测时停止变化。",
                "explicit_references": {
                    "fmeca": ["FM-GEN-301"],
                    "mfl": ["MFL-GEN-301"],
                },
            },
        )
        history["match_type"] = "structured_query"
        query = (
            "请同时核对 FM-GEN-302 和 MFL-GEN-302；"
            "以前相近案例如何检查并验证？"
        )
        state = {
            "query": query,
            "intent": "maintenance_query",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "original_query": query,
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "detection_method",
                    "maintenance_action",
                    "history_steps",
                ],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-301"},
                {"doc_type": "fmeca", "record_id": "FM-GEN-303"},
                {"doc_type": "mfl", "record_id": "MFL-GEN-301"},
                {"doc_type": "mfl", "record_id": "MFL-GEN-303"},
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state,
            [
                history,
                stale_fmeca,
                explicit_fmeca,
                linked_fmeca,
                stale_mfl,
                explicit_mfl,
                linked_mfl,
            ],
        )

        retained_ids = {item["record_id"] for item in retained}
        self.assertEqual(
            {
                "HIS-GEN-301",
                "FM-GEN-302",
                "MFL-GEN-302",
            },
            retained_ids,
        )

    def test_multiple_history_objects_do_not_create_single_parent_override(self):
        rows = []
        for suffix in ("401", "402"):
            fmeca = _evidence(
                doc_type="fmeca",
                record_id=f"FM-GEN-{suffix}",
                claim_type="fmeca_fact",
                answer_fields={
                    "product_flag": f"UNIT-{suffix} / FUNC-{suffix}",
                    "failure_mode": f"模式 {suffix}",
                },
            )
            fmeca["match_type"] = "exact_conversation_reference"
            history = _evidence(
                doc_type="fault_history",
                record_id=f"HIS-GEN-{suffix}",
                claim_type="historical_case",
                answer_fields={
                    "case_id": f"HIS-GEN-{suffix}",
                    "fault_symptom": f"对象 {suffix} 的记录现象。",
                    "explicit_references": {"fmeca": [f"FM-GEN-{suffix}"]},
                },
            )
            history["match_type"] = "exact_record_identity"
            rows.extend((fmeca, history))
        query = "比较 HIS-GEN-401 与 HIS-GEN-402 两个历史案例。"
        state = {
            "query": query,
            "intent": "history_query",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "original_query": query,
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["history_steps"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-401"},
                {"doc_type": "fmeca", "record_id": "FM-GEN-402"},
            ],
        }

        self.assertIsNone(
            _source_answer_history_rules._focused_history_explicit_references(state, rows)
        )
        retained = _source_answer_record_scope._narrow_semantic_table_families(state, rows)
        self.assertEqual(
            {"FM-GEN-401", "FM-GEN-402"},
            {item["record_id"] for item in retained if item["doc_type"] == "fmeca"},
        )

    def test_history_without_links_does_not_replace_prior_table_targets(self):
        fmeca_rows = []
        for suffix in ("501", "502"):
            item = _evidence(
                doc_type="fmeca",
                record_id=f"FM-GEN-{suffix}",
                claim_type="fmeca_fact",
                answer_fields={
                    "product_flag": f"UNIT-{suffix} / FUNC-{suffix}",
                    "failure_mode": f"模式 {suffix}",
                },
            )
            item["match_type"] = "exact_conversation_reference"
            fmeca_rows.append(item)
        history = _evidence(
            doc_type="fault_history",
            record_id="HIS-GEN-501",
            claim_type="historical_case",
            answer_fields={
                "case_id": "HIS-GEN-501",
                "fault_symptom": "输入状态暂时中断。",
                "steps": ["保存状态后检查连接。"],
            },
        )
        history["match_type"] = "structured_query"
        state = {
            "query": "以前相近案例怎么检查？",
            "intent": "history_query",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "original_query": "以前相近案例怎么检查？",
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["history_steps"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-501"},
                {"doc_type": "fmeca", "record_id": "FM-GEN-502"},
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state, [history, *fmeca_rows]
        )

        self.assertEqual(
            {"FM-GEN-501", "FM-GEN-502"},
            {item["record_id"] for item in retained if item["doc_type"] == "fmeca"},
        )

    def test_implicit_history_row_does_not_override_current_table_target(self):
        linked = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-601",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "UNIT-601 / FUNC-601",
                "failure_mode": "旁路状态异常",
            },
        )
        current = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-602",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "UNIT-602 / FUNC-602",
                "failure_mode": "输入状态异常",
            },
        )
        current["match_type"] = "exact_conversation_reference"
        history = _evidence(
            doc_type="fault_history",
            record_id="HIS-GEN-601",
            claim_type="historical_case",
            answer_fields={
                "case_id": "HIS-GEN-601",
                "fault_symptom": "旁路状态曾出现异常。",
                "explicit_references": {"fmeca": ["FM-GEN-601"]},
            },
        )
        history.update(match_type="semantic_id_hydration", retrieval_score=0.9)
        state = {
            "query": "这个现象先怎么检查？",
            "intent": "maintenance_query",
            "query_targets": [{"intent": "maintenance_query"}],
            "query_frame": {
                "original_query": "这个现象先怎么检查？",
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["detection_method", "history_root"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-602"}
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state, [history, linked, current]
        )

        self.assertEqual(
            {"FM-GEN-602"},
            {item["record_id"] for item in retained if item["doc_type"] == "fmeca"},
        )

    def test_exact_fmeca_mode_does_not_fuzzy_match_undelimited_mfl_text(self):
        target = _evidence(
            doc_type="fmeca",
            record_id="FM-SENSE-1",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "SNS / FUNC-910",
                "failure_mode": "压力采样漂移",
                "detection_method": "对比基准压力源。",
            },
        )
        target["match_type"] = "exact_conversation_reference"
        fuzzy_neighbor = _evidence(
            doc_type="mfl",
            record_id="MFL-SENSE-9",
            claim_type="maintenance_locator",
            answer_fields={
                # Contains the FMECA text but is neither the same stored value
                # nor an explicitly delimited table tail.
                "module_failure": "压力采样漂移监控告警",
                "maintenance_action": "检查监控告警状态。",
            },
        )
        state = {
            "query": "这个故障模式如何验证和处置？",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "detection_method",
                    "maintenance_action",
                ],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-SENSE-1"}
            ],
        }

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [target, fuzzy_neighbor]
        )

        self.assertEqual(
            [target["evidence_id"]],
            [item["evidence_id"] for item in filtered],
        )

    def test_exact_fmeca_mode_rejects_ambiguous_duplicate_mfl_tails(self):
        target = _evidence(
            doc_type="fmeca",
            record_id="FM-SENSE-2",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "SNS / FUNC-920",
                "failure_mode": "高度输入冻结",
                "detection_method": "检查输入计数变化。",
            },
        )
        target["match_type"] = "exact_conversation_reference"
        first = _evidence(
            doc_type="mfl",
            record_id="MFL-SENSE-1",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "主采集单元：高度输入冻结",
                "maintenance_action": "检查主采集链路。",
            },
        )
        second = _evidence(
            doc_type="mfl",
            record_id="MFL-SENSE-2",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "备用采集单元：高度输入冻结",
                "maintenance_action": "检查备用采集链路。",
            },
        )
        state = {
            "query": "这个故障模式如何验证和处置？",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "detection_method",
                    "maintenance_action",
                ],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-SENSE-2"}
            ],
        }

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [target, first, second]
        )

        self.assertEqual(
            [target["evidence_id"]],
            [item["evidence_id"] for item in filtered],
        )

    def test_impact_focus_does_not_admit_unrelated_exact_fta_leaf(self):
        fta = _evidence(
            doc_type="fta",
            record_id="FTA-NAV-1",
            claim_type="fta_structure",
            answer_fields={"basic_events": ["{'name': '系统时钟丢失'}"]},
            source_kind="hugegraph",
            authority="validated_relation",
        )
        unrelated = _evidence(
            doc_type="fmeca",
            record_id="FM-TIME-1",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "CLK / FUNC-990",
                "failure_mode": "系统时钟丢失",
                "local_impact": "统一时基丧失。",
            },
        )
        unrelated["match_type"] = "exact_fta_reference"
        state = _approved_state(
            fta,
            unrelated,
            intent="impact_analysis",
            targets=["impact_analysis"],
            query="导航输入单元停用后最先影响什么？",
        )
        state["query_frame"] = {
            "coverage_scope": "focused_detail",
            "required_answer_fields": ["local_impact"],
        }
        state["structured_entity_binding"] = {
            "bound": True,
            "business_ids": ["MOD-NAV-NIU"],
            "terms": ["MOD-NAV-NIU", "NIU"],
        }

        filtered = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [fta, unrelated]
        )

        self.assertNotIn(unrelated["evidence_id"], {item["evidence_id"] for item in filtered})































    def test_evidence_window_reserves_complete_architecture_inventory(self):
        generic = [
            _evidence(
                doc_type="function_locator",
                record_id=f"FUNC-{index:03d}",
                claim_type="function_fact",
                answer_fields={
                    "function_id": f"FUNC-{index:03d}",
                    "function_name": f"一般功能{index}",
                    "description": "普通功能记录",
                },
            )
            for index in range(20)
        ]
        inventory = [
            _evidence(
                doc_type="function_locator",
                record_id=f"SUB-{index}",
                claim_type="component_fact",
                answer_fields={
                    "component_id": f"SUB-{index}",
                    "component_name": f"第{index}子系统",
                    "component_type": "subsystem",
                    "parent_component_id": "SYS-1",
                    "collection_closed": True,
                    "collection_parent_id": "SYS-1",
                    "collection_parent_name": "测试系统",
                    "collection_member_kind": "subsystem",
                    "collection_member_count": 6,
                    "collection_role": "member",
                    "collection_member_id": f"SUB-{index}",
                    "collection_member_name": f"第{index}子系统",
                },
            )
            for index in range(1, 7)
        ]
        for item in generic:
            item["answer_rerank_score"] = 1.0
        for item in inventory:
            item["answer_rerank_score"] = -1.0
        state = _approved_state(
            *generic,
            *inventory,
            intent="architecture_query",
            targets=["architecture_query"],
        )
        state["query_frame"] = {
            "coverage_scope": "complete_collection",
            "collection_subject": "测试系统",
            "collection_member_kind": "subsystem",
            "required_answer_fields": ["closed_collection_members"],
        }

        selected = _source_answer_evidence_window._select_evidence_window(
            state, [*generic, *inventory], limit=10
        )

        selected_ids = {item["record_id"] for item in selected}
        self.assertTrue({f"SUB-{index}" for index in range(1, 7)} <= selected_ids)



    def test_evidence_retains_supported_fields_without_expanding_the_task(self):
        hmc = _evidence(
            doc_type="hmc",
            record_id="91810001**0001",
            claim_type="hmc_fault",
            answer_fields={
                "hmc_code": "91810001**0001",
                "fault_symptom": "ICU 输入过压钳位失效",
                "maintenance_action": "检查输入采样与钳位器件",
            },
        )
        function = _evidence(
            doc_type="function_locator",
            record_id="FUNC-330",
            claim_type="function_fact",
            answer_fields={
                "module_name": "IFC",
                "function_name": "外部总线适配",
            },
        )
        state = _approved_state(
            hmc,
            function,
            intent="architecture_query",
            targets=["architecture_query", "exact_code"],
        )
        _with_required_fields(
            state,
            "hmc_identity",
            "hmc_symptom",
            "maintenance_action",
            "architecture_hierarchy",
            "architecture_function",
        )

        instruction = answer_module._answer_task_instruction(state)
        self.assertNotIn("required_answer_fields", instruction)
        evidence = answer_module._format_context(state)
        self.assertIn("91810001**0001", evidence)
        self.assertIn("ICU 输入过压钳位失效", evidence)
        # A symptom claim must not smuggle a maintenance field into the
        # approved view; that needs a separately approved maintenance claim.
        self.assertNotIn("检查输入采样与钳位器件", evidence)
        self.assertIn("FUNC-330", evidence)
        self.assertIn("外部总线适配", evidence)
        maintenance = _evidence(
            doc_type="hmc", record_id="91810001**0001", claim_type="maintenance_action",
            answer_fields={"maintenance_action": "检查输入采样与钳位器件"},
        )
        state["evidence_registry"].append(maintenance)
        state["context_evidence_ids"].append(maintenance["evidence_id"])
        state["safety_decision"]["approved_evidence_ids"].append(maintenance["evidence_id"])
        self.assertIn("检查输入采样与钳位器件", answer_module._format_context(state))













    def test_filter_keeps_fmeca_leaf_of_approved_fta_tree(self):
        """已批准 FTA 树的叶子故障模式即使没有对象 ID 交集也不被 fmeca_ids
        边界检查删除；未关联的 FMECA 行仍按原规则过滤。"""
        fta_item = _evidence(
            doc_type="fta",
            record_id="FTA-SYN-003",
            claim_type="fta_structure",
            answer_fields={
                "basic_events": [
                    "{'name': 'A通道接收超时', 'description': '接收器件失效。'}"
                ]
            },
            source_kind="hugegraph",
            authority="validated_relation",
            content="FTA 结构：系统无法与外部设备交换有效业务数据",
        )
        leaf_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-031",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "BCA",
                "failure_mode": "A通道接收超时",
                "fault_cause": "接收器件失效、线路异常或时序配置错误。",
            },
        )
        unrelated_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-999",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "OTHER",
                "failure_mode": "温度采集满量程",
                "fault_cause": "采集前端损坏。",
            },
        )
        state = _approved_state(
            fta_item,
            leaf_fmeca,
            unrelated_fmeca,
            intent="cause_analysis",
            targets=["cause_analysis"],
            query="外部设备拿不到数据，按可能断开的环节说明原因",
        )
        state["topology_dependency"] = {
            "required": True,
            "subject": "外部数据链路",
            "subject_kind": "data_path",
            "failure_state": "失效",
            "canonical_business_ids": ["FUNC-999"],
        }
        state["query_frame"] = {
            "coverage_scope": "relation_path",
            "required_answer_fields": ["structural_cause_chain", "fault_cause"],
            "diagnostic_closure": {"required": True},
        }

        retained = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [fta_item, leaf_fmeca, unrelated_fmeca]
        )
        retained_ids = {item["evidence_id"] for item in retained}

        self.assertIn(leaf_fmeca["evidence_id"], retained_ids)
        self.assertNotIn(unrelated_fmeca["evidence_id"], retained_ids)

    def test_plain_english_words_are_not_structured_identity_codes(self):
        self.assertEqual(
            set(),
            _source_answer_payload._exact_structured_identity_codes(
                ["input conditioning module", "power supply board"]
            ),
        )
        self.assertEqual(
            {"ICU", "FUNC-110"},
            _source_answer_payload._exact_structured_identity_codes(["ICU / FUNC-110"]),
        )

    def test_exact_structured_fmeca_target_beats_semantic_neighbors(self):
        exact = _evidence(
            doc_type="fmeca",
            record_id="FM-EXACT",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "UNIT-A / FUNC-101",
                "failure_mode": "基准信号漂移",
            },
        )
        exact["match_type"] = "exact_structured_identity"
        semantic_a = _evidence(
            doc_type="fmeca",
            record_id="FM-NEIGHBOR-A",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "UNIT-B / FUNC-202",
                "failure_mode": "邻近模式一",
            },
        )
        semantic_a.update(match_type="semantic_id_hydration", answer_rerank_score=0.99)
        semantic_b = _evidence(
            doc_type="fmeca",
            record_id="FM-NEIGHBOR-B",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "UNIT-C / FUNC-303",
                "failure_mode": "邻近模式二",
            },
        )
        semantic_b.update(match_type="semantic_id_hydration", answer_rerank_score=0.98)

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            {"query_frame": {"coverage_scope": "focused_detail"}},
            [semantic_a, exact, semantic_b],
        )

        self.assertEqual([exact["evidence_id"]], [item["evidence_id"] for item in retained])

    def test_ambiguous_structured_record_binding_disables_single_row_narrowing(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-701",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "UNIT-701 / FUNC-701",
                "failure_mode": "输入状态丢失",
            },
        )
        first.update(match_type="semantic_id_hydration", answer_rerank_score=0.95)
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-702",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "UNIT-702 / FUNC-702",
                "failure_mode": "输出状态丢失",
            },
        )
        second.update(match_type="semantic_id_hydration", answer_rerank_score=0.41)

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            {
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["failure_mode"],
                },
                "structured_record_binding": {
                    "bound": False,
                    "ambiguous": True,
                    "candidate_record_ids": ["FM-GEN-701", "FM-GEN-702"],
                },
            },
            [first, second],
        )

        self.assertEqual(
            {first["evidence_id"], second["evidence_id"]},
            {item["evidence_id"] for item in retained},
        )



if __name__ == "__main__":
    unittest.main()
