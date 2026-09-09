import backend.agent.nodes.answer_collections as _source_answer_collections
import backend.agent.nodes.answer_domain_scope as _source_answer_domain_scope
import backend.agent.nodes.answer_evidence_window as _source_answer_evidence_window
import backend.agent.nodes.answer_fault_rules as _source_answer_fault_rules
import backend.agent.nodes.answer_record_scope as _source_answer_record_scope
import backend.agent.nodes.answer_request as _source_answer_request
import backend.agent.nodes.answer_target_presence as _source_answer_target_presence
"""Canaries for semantic answer contracts at the final-answer boundary.

These tests deliberately separate deterministic safety checks (approved source,
exact identity, citation and scalar validation) from open-language judgements.
The latter must be carried by the planner/semantic reviewer, not reconstructed
from a small dictionary of Chinese words or from coincidentally equal text in
different document families.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _evidence(
    *,
    doc_type,
    record_id,
    answer_fields,
    claim_type="structured_fact",
    content="用于通用契约测试的权威记录。",
):
    return make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type=claim_type,
        doc_type=doc_type,
        doc_id=f"doc-{doc_type}",
        record_id=record_id,
        title=record_id,
        content=content,
        display_summary=content,
        review_status="not_required",
        metadata={"answer_fields": dict(answer_fields)},
    )


def _approved_state(*items, query="当前问题", intent="cause_analysis"):
    evidence = list(items)
    evidence_ids = [item["evidence_id"] for item in evidence]
    return {
        "query": query,
        "intent": intent,
        "query_targets": [{"intent": intent}],
        "evidence_registry": evidence,
        "context_evidence_ids": evidence_ids,
        "safety_decision": {
            "approved_evidence_ids": evidence_ids,
            "context_evidence_ids": evidence_ids,
        },
    }


class EvidenceSelectionSemanticBoundaryTests(unittest.TestCase):
    def test_mixed_history_filter_is_invariant_to_intent_labels(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-MIX-1",
            claim_type="fmeca_fact",
            answer_fields={
                "failure_mode": "某当前模式",
                "fault_cause": "某当前原因",
            },
        )
        history = _evidence(
            doc_type="fault_history",
            record_id="HIS-MIX-1",
            claim_type="historical_case",
            answer_fields={
                "case_id": "HIS-MIX-1",
                "explicit_references": {"fmeca": ["FM-MIX-1"]},
                "verification_result": "复测满足记录标准。",
            },
        )
        history.update(match_type="semantic_id_hydration", retrieval_score=0.9)
        outputs = []
        for intent in ("cause_analysis", "architecture_query", "history_query"):
            state = _approved_state(fmeca, history, intent=intent)
            state["query_targets"] = [{"intent": intent}]
            state["query_frame"] = {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["fault_cause", "history_result"],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": ["fault_cause", "history_result"],
                    "basis": "当前模式与复测结果",
                },
            }
            outputs.append([
                item["evidence_id"]
                for item in answer_module._context_evidence(state)
            ])

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_intent_and_fields_do_not_reserve_or_drop_candidate_families(self):
        rows = [_evidence(doc_type=kind, record_id=f"ROW-{n}", claim_type="source_fact",
                          answer_fields={"description": "事实"})
                for n, kind in enumerate(("fmeca", "fta", "fault_history", "mfl"))]
        for intent in ("cause_analysis", "maintenance_query", "history_query", "architecture_query"):
            state = {"intent": intent, "query_frame": {"required_answer_fields": ["fault_cause"]}}
            self.assertEqual({r["record_id"] for r in rows},
                             {r["record_id"] for r in _source_answer_evidence_window._select_evidence_window(state, rows)})

    def test_architecture_intent_alone_does_not_expand_a_closed_group_window(self):
        members = []
        for index in range(8):
            members.append(
                _evidence(
                    doc_type="function_locator",
                    record_id=f"MOD-WIN-{index}",
                    claim_type="component_fact",
                    answer_fields={
                        "component_id": f"MOD-WIN-{index}",
                        "component_name": f"单元{index}",
                        "collection_closed": True,
                        "collection_role": "member",
                        "collection_parent_id": "SYS-WIN",
                        "collection_parent_name": "窗口测试系统",
                        "collection_member_kind": "module",
                        "collection_member_count": 8,
                        "collection_member_id": f"MOD-WIN-{index}",
                        "collection_member_name": f"单元{index}",
                    },
                )
            )
        state = _approved_state(*members, intent="architecture_query")
        state["query_frame"] = {"required_answer_fields": []}

        selected = _source_answer_evidence_window._select_evidence_window(state, members, limit=3)

        self.assertEqual(8, len(selected))

    def test_surface_domain_words_do_not_change_an_unbound_evidence_window(self):
        """A noun in the question is not an authority-bearing object binding."""

        reset = _evidence(
            doc_type="fta",
            record_id="TREE-GEN-A",
            claim_type="fta_structure",
            content="启动控制异常可由复位支路断开引起。",
            answer_fields={
                "top_event": "启动控制异常",
                "basic_events": ["复位支路断开"],
            },
        )
        transport = _evidence(
            doc_type="fta",
            record_id="TREE-GEN-B",
            claim_type="fta_structure",
            content="外部业务链路异常可由传输支路断开引起。",
            answer_fields={
                "top_event": "外部业务链路异常",
                "basic_events": ["传输支路断开"],
            },
        )
        presentation = _evidence(
            doc_type="fta",
            record_id="TREE-GEN-C",
            claim_type="fta_structure",
            content="画面输出异常可由图像支路断开引起。",
            answer_fields={
                "top_event": "画面输出异常",
                "basic_events": ["图像支路断开"],
            },
        )
        evidence = [reset, transport, presentation]
        expected = {item["evidence_id"] for item in evidence}

        # These are surface-form mutations only.  None provides an exact ID,
        # selected anchor, persisted binding, or planner-authored domain lock.
        for query in (
            "这个异常涉及哪些环节？",
            "这个数据异常涉及哪些环节？",
            "这个显示异常涉及哪些环节？",
            "这个收发异常涉及哪些环节？",
            "这个复位异常涉及哪些环节？",
        ):
            with self.subTest(query=query):
                state = _approved_state(
                    *evidence,
                    query=query,
                    intent="cause_analysis",
                )
                state["query_frame"] = {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["structural_cause_chain"],
                }
                retained = _source_answer_domain_scope._filter_answer_domain_evidence(
                    state, evidence
                )
                self.assertEqual(
                    expected,
                    {item["evidence_id"] for item in retained},
                )

    def test_exact_structured_binding_still_removes_a_sibling(self):
        """Removing evidence remains valid when an exact current object binds it."""

        selected = _evidence(
            doc_type="function_locator",
            record_id="MOD-GEN-A",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-GEN-A",
                "component_name": "甲处理单元",
            },
        )
        sibling = _evidence(
            doc_type="function_locator",
            record_id="MOD-GEN-B",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-GEN-B",
                "component_name": "乙处理单元",
            },
        )
        state = _approved_state(
            selected,
            sibling,
            query="MOD-GEN-A 的职责是什么？",
            intent="entity_profile",
        )
        state["query_frame"] = {
            "coverage_scope": "focused_detail",
            "required_answer_fields": ["record_name"],
        }
        state["structured_entity_binding"] = {
            "bound": True,
            "business_ids": ["MOD-GEN-A"],
        }

        retained = _source_answer_domain_scope._filter_answer_domain_evidence(
            state, [selected, sibling]
        )

        self.assertEqual(
            [selected["evidence_id"]],
            [item["evidence_id"] for item in retained],
        )

    def test_equal_fault_text_does_not_create_an_fmeca_to_mfl_branch(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-A",
            claim_type="fmeca_fact",
            answer_fields={
                "product_flag": "UNIT-A / FUNC-GEN-A",
                "failure_mode": "状态采样间歇丢失",
                "detection_method": "核对采样计数。",
            },
        )
        fmeca["match_type"] = "exact_conversation_reference"
        mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-GEN-A",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "接口单元：状态采样间歇丢失",
                "maintenance_action": "重新锁止接口连接。",
            },
        )
        state = {
            "query": "这个模式如何判别？",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["detection_method"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-A"}
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state, [fmeca, mfl]
        )

        self.assertEqual(
[fmeca["evidence_id"], mfl["evidence_id"]],
            [item["evidence_id"] for item in retained],
        )

    def test_explicit_record_anchors_can_keep_both_table_rows(self):
        """The negative text-join rule must not weaken exact record selection."""

        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-GEN-B",
            claim_type="fmeca_fact",
            answer_fields={"failure_mode": "状态采样间歇丢失"},
        )
        mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-GEN-B",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "接口单元：状态采样间歇丢失",
                "maintenance_action": "重新锁止接口连接。",
            },
        )
        state = {
            "query": "分别查看这两条记录。",
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["failure_mode", "maintenance_action"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "record_id": "FM-GEN-B"},
                {"doc_type": "mfl", "record_id": "MFL-GEN-B"},
            ],
        }

        retained = _source_answer_record_scope._narrow_semantic_table_families(
            state, [fmeca, mfl]
        )

        self.assertEqual(
            {fmeca["evidence_id"], mfl["evidence_id"]},
            {item["evidence_id"] for item in retained},
        )






class OpenLanguageSufficiencyTests(unittest.TestCase):
    def test_architecture_role_paraphrases_have_the_same_mechanical_status(self):
        manifest = [
            {
                "citation_id": 1,
                "doc_type": "function_locator",
                "record_id": "MOD-GEN-E",
                "supports": ["architecture_receivers"],
                "answer_fields": {
                    "component_id": "MOD-GEN-E",
                    "component_name": "接口处理单元",
                },
            }
        ]
        conventional = "业务报文由接口处理单元 MOD-GEN-E 接收。[^1]"
        paraphrase = "业务报文抵达接口处理单元 MOD-GEN-E。[^1]"

        self.assertEqual(
            _source_answer_target_presence._target_facet_is_present(
                conventional, "architecture_receivers", manifest
            ),
            _source_answer_target_presence._target_facet_is_present(
                paraphrase, "architecture_receivers", manifest
            ),
        )

    def test_comparison_mode_uses_the_planner_contract_not_a_phrase_list(self):
        subjects = [
            {"kind": "failure_mode", "value": "候选甲", "basis": "候选甲"},
            {"kind": "failure_mode", "value": "候选乙", "basis": "候选乙"},
        ]
        conventional = {
            "query": "请对比故障模式甲和故障模式乙的不同。",
            "query_frame": {
                "original_query": "请对比候选甲和候选乙的不同。",
                "question_semantics": "compare",
                "comparison_subjects": subjects,
            },
        }
        paraphrase = {
            "query": "候选甲、候选乙的异同何在？",
            "query_frame": {
                "original_query": "候选甲、候选乙的异同何在？",
                "question_semantics": "compare",
                "comparison_subjects": subjects,
            },
        }

        self.assertTrue(
            _source_answer_fault_rules._query_requests_parallel_failure_modes(conventional)
        )
        self.assertTrue(
            _source_answer_fault_rules._query_requests_parallel_failure_modes(paraphrase)
        )




    def test_locator_keyword_cannot_replace_exact_source_value(self):
        manifest = [
            {
                "citation_id": 1,
                "doc_type": "mfl",
                "record_id": "MFL-GEN-F",
                "supports": ["maintenance_locator"],
                "answer_fields": {
                    "record_code": "MFL-GEN-F",
                    "module_failure": "接口单元状态保持异常",
                },
            }
        ]

        self.assertFalse(
            _source_answer_target_presence._target_facet_is_present(
                "维修定位信息见该记录。[^1]",
                "maintenance_locator",
                manifest,
            )
        )
        self.assertTrue(
            _source_answer_target_presence._target_facet_is_present(
                "维修定位记录为 MFL-GEN-F。[^1]",
                "maintenance_locator",
                manifest,
            )
        )


class StructuredSemanticInputBoundaryTests(unittest.TestCase):
    def test_fault_fit_does_not_classify_free_text_stage_or_occurrence_words(self):
        row = _evidence(
            doc_type="fmeca",
            record_id="FM-STAGE-FREE",
            claim_type="fmeca_fact",
            answer_fields={
                "failure_mode": "状态保持异常",
                "mission_phase": "正常运行中偶发状态保持异常。",
                "final_impact": "状态会反复丢失。",
            },
        )
        state = {
            "query_frame": {
                "operating_stage": "上电",
                "temporal_pattern": "持续发生",
            }
        }

        self.assertEqual("unknown", _source_answer_fault_rules._fault_candidate_fit(state, row))

    def test_fault_fit_uses_only_explicit_source_enums(self):
        row = _evidence(
            doc_type="fmeca",
            record_id="FM-STAGE-ENUM",
            claim_type="fmeca_fact",
            answer_fields={
                "failure_mode": "状态保持异常",
                "mission_phase": "这段自由文本不承担分类职责。",
                "operating_stage": "运行中",
                "temporal_pattern": "重复发生",
            },
        )
        direct = {
            "query_frame": {
                "operating_stage": "运行中",
                "temporal_pattern": "重复发生",
            }
        }
        conflicting = {
            "query_frame": {
                "operating_stage": "上电",
                "temporal_pattern": "持续发生",
            }
        }

        self.assertEqual("direct", _source_answer_fault_rules._fault_candidate_fit(direct, row))
        self.assertEqual(
            "contradictory",
            _source_answer_fault_rules._fault_candidate_fit(conflicting, row),
        )

    def test_architecture_question_verbs_do_not_create_role_selection(self):
        receiver = _evidence(
            doc_type="function_locator",
            record_id="MOD-ROLE-A",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-ROLE-A",
                "component_name": "甲通道单元",
                "component_type": "module",
                "description": "处理甲通道输入。",
            },
        )

        for query in (
            "哪个模块负责接收？",
            "信息抵达后由哪些单元承接？",
            "后续交给谁？",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    {},
                    _source_answer_collections._architecture_role_indices(
                        {"query": query, "query_frame": {}}, [receiver]
                    ),
                )

    def test_grounded_architecture_roles_select_exact_structured_subjects(self):
        first = _evidence(
            doc_type="function_locator",
            record_id="MOD-ROLE-B1",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-ROLE-B1",
                "component_name": "甲通道单元",
                "component_type": "module",
            },
        )
        second = _evidence(
            doc_type="function_locator",
            record_id="MOD-ROLE-B2",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-ROLE-B2",
                "component_name": "乙通道单元",
                "component_type": "module",
            },
        )
        roles = [
            {"role": "receiver", "subject": "甲通道单元", "basis": "甲通道单元"},
            {"role": "receiver", "subject": "乙通道单元", "basis": "乙通道单元"},
        ]

        for query in ("核对这两个对象。", "请说明它们。"):
            with self.subTest(query=query):
                selected = _source_answer_collections._architecture_role_indices(
                    {
                        "query": query,
                        "query_frame": {"architecture_roles": roles},
                    },
                    [first, second],
                )
                self.assertEqual(
                    [0, 1], selected.get("architecture_receivers")
                )

    def test_payload_direction_parser_accepts_only_structured_enum(self):
        for prose in (
            "外部设备发送数据但系统接收不到",
            "内部发出报文而外部终端未收到",
        ):
            with self.subTest(prose=prose):
                self.assertEqual("", _source_answer_fault_rules._payload_direction(prose))

        self.assertEqual(
            "external_to_system",
            _source_answer_fault_rules._payload_direction(
                {"kind": "data_flow", "direction": "external_to_system"}
            ),
        )






    @staticmethod
    def _fta_structure_state():
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-FTA-A",
            claim_type="fmeca_fact",
            answer_fields={
                "failure_mode": "候选甲",
                "fault_cause": "甲源记录。",
                "final_impact": "甲影响。",
                "detection_method": "甲检查。",
            },
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-FTA-B",
            claim_type="fmeca_fact",
            answer_fields={
                "failure_mode": "候选乙",
                "fault_cause": "乙源记录。",
                "final_impact": "乙影响。",
                "detection_method": "乙检查。",
            },
        )
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta-contract",
            record_id="FTA-CONTRACT",
            title="通用结构记录",
            content="建模说明：结构化父节点包含两条分支。",
            display_summary="建模说明：结构化父节点包含两条分支。",
            review_status="auto_validated",
        )
        state = _approved_state(first, second, fta, intent="cause_analysis")
        state["query_frame"] = {
            "diagnostic_closure": {"required": True},
        }
        draft = (
            "1. **候选甲**[^1]\n"
            "   - 甲源记录；甲影响；甲检查。[^1]\n\n"
            "2. **候选乙**[^2]\n"
            "   - 乙源记录；乙影响；乙检查。[^2]"
        )
        return state, draft




class StructuredPostprocessingObligationTests(unittest.TestCase):


    def test_fta_structure_only_mode_never_comes_from_question_wording(self):
        state = {
            "query": "只解释 FTA 故障树结构，不要计算。",
            "intent": "cause_analysis",
            "requested_source_tools": ["fta_query"],
            "query_frame": {
                "required_answer_fields": ["structural_cause_chain"],
                "source_obligations": [
                    {
                        "tool": "fta_query",
                        "doc_type": "fta",
                        "fields": ["structural_cause_chain"],
                    }
                ],
            },
        }

        self.assertFalse(_source_answer_request._is_fta_structure_only_request(state))
        state["query_frame"]["response_mode"] = {
            "mode": "structure_only",
            "basis": "故障树结构",
        }
        self.assertTrue(_source_answer_request._is_fta_structure_only_request(state))

        for query in ("解释一下。", "给出层级。"):
            with self.subTest(query=query):
                state["query"] = query
                self.assertTrue(_source_answer_request._is_fta_structure_only_request(state))

    def test_draft_members_and_count_do_not_activate_collection_contract(self):
        manifest = [
            {
                "citation_id": index,
                "answer_fields": {
                    "collection_closed": True,
                    "collection_role": "member",
                    "collection_parent_id": "SYS-COL",
                    "collection_parent_name": "某系统",
                    "collection_member_kind": "module",
                    "collection_member_count": 2,
                    "collection_member_id": f"MOD-COL-{index}",
                    "collection_member_name": name,
                },
            }
            for index, name in enumerate(("甲单元", "乙单元"), start=1)
        ]
        draft = "甲单元和乙单元，总数为 2 个。[^1][^2]"

        self.assertFalse(
            _source_answer_request._closed_collection_obligation_active(
                draft,
                {"query_frame": {"coverage_scope": "focused_detail"}},
                manifest,
            )
        )
        self.assertTrue(
            _source_answer_request._closed_collection_obligation_active(
                "任意表达。",
                {"query_frame": {"coverage_scope": "complete_collection"}},
                manifest,
            )
        )




    def test_focused_scalar_and_status_values_need_no_preferred_labels(self):
        cases = (
            (
                "severity",
                {"severity": "III"},
                "该条记录归入 III。[^1]",
            ),
            (
                "maintenance_api",
                {"api_reportable": False},
                "该字段值为否。[^1]",
            ),
            (
                "rbd_missing_parameters",
                {"missing_parameters": ["输出网关故障率"]},
                "输出网关故障率。[^1]",
            ),
        )
        for facet, fields, answer in cases:
            with self.subTest(facet=facet):
                manifest = [
                    {
                        "citation_id": 1,
                        "record_id": "REC-SCALAR-1",
                        "record_name": "某记录",
                        "supports": [facet],
                        "answer_fields": fields,
                    }
                ]
                self.assertTrue(
                    _source_answer_target_presence._target_facet_is_present(answer, facet, manifest)
                )







class StructuredClarificationAnswerTests(unittest.TestCase):
    @staticmethod
    def _state():
        requirement = {
            "kind": "identity_slot",
            "slot": "selected_record_identity",
            "source": {
                "kind": "evidence_gate",
                "name": "identity_resolution",
                "evidence_ids": [],
            },
            "identity": {"doc_type": "hmc", "record_ids": [], "fields": {}},
        }
        return {
            "safety_decision": {
                "missing_input_requirements": [requirement],
                "unresolved_identity_slots": [requirement],
                "clarification_reasons": [],
                "clarification_fallback": {
                    "kind": "generic_clarification",
                    "required": False,
                },
                "questions": [],
            },
            "clarification_question": "",
        }

    def test_structured_gap_is_given_to_model_without_a_canonical_question(self):
        text = answer_module._clarification_text(self._state())

        self.assertIn("selected_record_identity", text)
        self.assertIn("没有预制问句", text)
        self.assertNotIn("请提供完整", text)




if __name__ == "__main__":
    unittest.main()
