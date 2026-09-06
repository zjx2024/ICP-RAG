import backend.agent.nodes.answer_evidence_window as _source_answer_evidence_window
import backend.agent.nodes.answer_target_presence as _source_answer_target_presence
import asyncio
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.agent.api.chat_router import ChatRequest
from backend.agent.db import AgentDB, _conversation_anchors
from backend.agent.graph import build_graph
from backend.agent.nodes import plan as plan_module
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes.direct_response import direct_response_node
from backend.agent.nodes.scope_router import parse_scope_response
from backend.agent.tool_policy import resolve_tool_policy
from backend.agent.tools import icp_diagnosis as diagnosis_module
from backend.agent.contracts import make_evidence


def _run(coroutine):
    return asyncio.run(coroutine)


def _failure_mode_collection_member(
    index,
    total,
    *,
    closed=True,
    parent_id="SUB-PWR",
):
    """Create a collection member without architecture-component aliases.

    A FMECA collection is not a component hierarchy.  Its stable public
    identity is therefore ``collection_member_id/name`` rather than a borrowed
    ``component_id/name`` pair.
    """

    names = {
        1: "A路间歇停机",
        2: "输入欠压保护误动作",
        3: "电源切换控制异常",
    }
    name = names.get(index, f"供电故障模式{index}")
    member_id = f"FM-PWR-{index:03d}"
    return make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="fmeca_fact",
        doc_type="fmeca",
        doc_id="doc-pwr-fmeca",
        record_id=member_id,
        title=name,
        content=f"{name}是电源与配电子系统的故障模式。",
        metadata={
            "answer_fields": {
                "record_code": member_id,
                "failure_mode": name,
                "fault_cause": f"记录原因{index}",
                "collection_closed": closed,
                "collection_parent_id": parent_id,
                "collection_parent_name": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
                "collection_member_count": total,
                "collection_role": "member",
                "collection_member_id": member_id,
                "collection_member_name": name,
                "collection_member_description": f"{name}的权威记录",
            }
        },
    )


class ScopeRoutingTests(unittest.TestCase):
    def test_raw_architecture_inventory_words_do_not_promote_closed_contract(self):
        decision = {
            "primary_intent": "architecture_query",
            "query_targets": [{"intent": "architecture_query"}],
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "subjects": ["SYN-ICP-9000", "功能域"],
            "required_answer_fields": ["architecture_hierarchy"],
        }
        repaired = plan_module._downgrade_unrequested_collection_contract(
            decision,
            query="SYN-ICP-9000 总体有几块功能域？请列出全部",
        )
        self.assertEqual("focused_detail", repaired["coverage_scope"])
        self.assertNotIn("closed_collection_members", repaired["required_answer_fields"])

    def test_scope_contract_accepts_only_closed_routes(self):
        parsed = parse_scope_response(json.dumps({
            "route": "knowledge_base",
            "confidence": 0.92,
            "reason": "需要查询模块资料",
        }, ensure_ascii=False))
        self.assertEqual("knowledge_base", parsed["route"])
        self.assertIsNone(parse_scope_response('{"route":"search_everything"}'))

    def test_direct_responses_never_return_tools_or_evidence(self):
        result = _run(direct_response_node({
            "scope_route": "agent_capability",
            "agent_name": "测试智能体",
        }))
        self.assertIn("测试智能体", result["final_answer"])
        self.assertIn("七类资料", result["final_answer"])
        for field in (
            "citations", "evidence_catalog", "used_evidence_ids",
            "tool_calls", "tool_results", "evidence_registry",
        ):
            self.assertEqual([], result[field])

    def test_icp_graph_has_scope_and_context_clarification_shortcuts(self):
        compiled = _run(build_graph(["icp_diagnosis", "graph_traversal"]))
        edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
        self.assertIn(("scope_router", "plan"), edges)
        self.assertIn(("scope_router", "direct_response"), edges)
        self.assertIn(("plan", "icp_diagnosis"), edges)
        self.assertIn(("plan", "direct_response"), edges)
        self.assertNotIn(("direct_response", "icp_diagnosis"), edges)

    def test_chat_query_validation_rejects_blank_and_control_characters(self):
        with self.assertRaises(ValidationError):
            ChatRequest(agent_id="a", session_id="s", query="   ")
        with self.assertRaises(ValidationError):
            ChatRequest(agent_id="a", session_id="s", query="正常\x00问题")

    def test_cross_type_business_ids_keep_one_prefix_agnostic_entity_contract(self):
        cases = (
            ("FM-SYN-002 是什么", "FM-SYN-002"),
            ("MFL-SYN-041 是什么", "MFL-SYN-041"),
            ("HIS-SYN-020 是什么", "HIS-SYN-020"),
        )
        identity_probes = {
            "function_locator",
            "hmc_query",
            "fmeca_query",
            "mfl_query",
            "fault_history_query",
        }

        for query, business_id in cases:
            with self.subTest(query=query):
                target_response = json.dumps(
                    {
                        "primary_intent": "entity_profile",
                        "targets": ["entity_profile"],
                        "contextualized_query": query,
                        "subjects": [business_id],
                        "requested_facets": ["业务记录身份"],
                        "required_answer_fields": ["record_identity"],
                        "coverage_scope": "focused_detail",
                        "context_used": False,
                        "context_status": "standalone",
                        "selected_anchor_ids": [],
                    },
                    ensure_ascii=False,
                )
                planner_response = json.dumps(
                    {
                        "intent": "entity_profile",
                        "query_targets": [{"intent": "entity_profile"}],
                        "rewritten_query": query,
                        "entity_keywords": [business_id],
                        "type_keywords": ["业务记录"],
                        "query_frame": {"coverage_scope": "focused_detail"},
                        # Deliberately reproduce a planner that assumes every
                        # explicit code belongs to Function Locator.  Policy
                        # must still probe all authoritative identity columns.
                        "tool_queries": {"function_locator": [business_id]},
                        "tools_plan": [
                            {
                                "tool": "function_locator",
                                "reason": "按业务 ID 查询架构记录",
                            }
                        ],
                        "relation_expansion": {
                            "needed": False,
                            "reason": "单条记录身份无需关系扩展",
                        },
                        "reasoning": "查询业务记录身份",
                    },
                    ensure_ascii=False,
                )
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
                    result = _run(
                        plan_module.plan_node(
                            {
                                "query": query,
                                "messages": [{"role": "user", "content": query}],
                                "memory_enabled": False,
                                "iteration": 0,
                            }
                        )
                    )

                self.assertEqual("entity_profile", result["intent"])
                self.assertEqual(
                    ["record_identity"],
                    result["query_frame"]["required_answer_fields"],
                )
                self.assertEqual(
                    ["record_identity"],
                    result["query_targets"][0]["required_fields"],
                )
                self.assertEqual(
                    identity_probes | {"evidence_synthesis"},
                    set(result["allowed_tools"]),
                )
                for tool in identity_probes:
                    self.assertTrue(
                        any(
                            business_id in tool_query
                            for tool_query in result["tool_queries"][tool]
                        ),
                        tool,
                    )

        self.assertIn(
            "FMECA、MFL、故障历史等业务记录",
            plan_module.TARGET_PROMPT,
        )


class ConversationAnchorTests(unittest.TestCase):
    def test_delivered_fields_only_carry_into_a_resolved_continuation(self):
        messages = [
            {
                "role": "assistant",
                "delivered_answer_fields": [
                    "local_impact",
                    "final_impact",
                    "local_impact",
                ],
            }
        ]

        self.assertEqual(
            ["local_impact", "final_impact"],
            plan_module._contextual_prior_delivered_fields(
                messages, {"context_used": True, "status": "resolved"}
            ),
        )
        self.assertEqual(
            [],
            plan_module._contextual_prior_delivered_fields(
                messages, {"context_used": False, "status": "standalone"}
            ),
        )

    def test_delivery_fields_follow_selected_older_object_not_latest_turn(self):
        messages = [
            {
                "id": "answer-a",
                "role": "assistant",
                "answer_delivery_manifest": [{
                    "field": "local_impact",
                    "evidence_ids": ["ev-a"],
                    "record_ids": ["FM-A"],
                    "doc_types": ["fmeca"],
                    "subject_ids": ["MOD-A"],
                }],
            },
            {
                "id": "answer-b",
                "role": "assistant",
                "answer_delivery_manifest": [{
                    "field": "maintenance_action",
                    "evidence_ids": ["ev-b"],
                    "record_ids": ["MFL-B"],
                    "doc_types": ["mfl"],
                    "subject_ids": ["MOD-B"],
                }],
            },
        ]
        selected = [{
            "anchor_id": "ev-a",
            "evidence_id": "ev-a",
            "record_id": "FM-A",
            "origin_message_id": "answer-a",
            "answer_fields": {"module_id": "MOD-A"},
        }]

        contracts = plan_module._selected_anchor_delivery_contracts(
            messages,
            {"context_used": True, "status": "resolved"},
            selected,
        )

        self.assertEqual(["local_impact"], [item["field"] for item in contracts])

    def test_field_only_delivery_is_not_applied_when_selected_object_has_no_manifest(self):
        messages = [{
            "role": "assistant",
            "delivered_answer_fields": ["final_impact"],
        }]

        self.assertEqual(
            [],
            plan_module._contextual_prior_delivered_fields(
                messages,
                {"context_used": True, "status": "resolved"},
                [{"anchor_id": "ev-current", "record_id": "FM-CURRENT"}],
            ),
        )

    def test_structured_target_preserves_multiple_response_obligations(self):
        raw_query = "已消解对象的现场处置与历史恢复验证"
        decision = plan_module._parse_target_response(json.dumps({
            "primary_intent": "maintenance_query",
            "targets": ["maintenance_query", "history_query"],
            "contextualized_query": "已消解对象的现场处置与历史恢复验证",
            "subjects": ["对象-X"],
            "requested_facets": ["现场处置", "历史恢复"],
            "required_answer_fields": [
                "detection_method", "maintenance_action", "history_steps",
                "history_result",
            ],
            "question_semantics": "act",
            "response_focus": [
                "first_action", "maintenance_action", "history_conditions",
                "recovery_verification",
            ],
            "response_scope": "incremental",
            "response_scope_basis": "现场处置与历史恢复验证",
            "coverage_scope": "focused_detail",
            "context_used": True,
            "context_status": "resolved",
        }), raw_user_query=raw_query)

        self.assertIsNotNone(decision)
        self.assertEqual("act", decision["question_semantics"])
        self.assertEqual(
            [
                "first_action", "maintenance_action", "history_conditions",
                "recovery_verification",
            ],
            decision["response_focus"],
        )
        self.assertEqual("incremental", decision["response_scope"])

    def test_unknown_response_obligation_is_rejected_not_keyword_repaired(self):
        decision = plan_module._parse_target_response(json.dumps({
            "primary_intent": "impact_analysis",
            "targets": ["impact_analysis"],
            "required_answer_fields": ["local_impact"],
            "question_semantics": "invented-shape",
            "response_focus": ["immediate_impact", "magic-keyword"],
            "response_scope": "invented-scope",
            "coverage_scope": "focused_detail",
            "context_used": False,
        }))

        self.assertIsNotNone(decision)
        self.assertEqual("unknown", decision["question_semantics"])
        self.assertEqual(["immediate_impact"], decision["response_focus"])
        self.assertEqual("unknown", decision["response_scope"])

    def test_diagram_surface_count_never_adds_unselected_offered_ids(self):
        offered = [f"ev-rbd-{index}" for index in range(1, 5)]
        for query in (
            "重新计算这四个模型",
            "把先前展示的全体结构再算一遍",
            "repeat the complete set shown above",
        ):
            with self.subTest(query=query):
                decision = plan_module._parse_target_response(
                    json.dumps({
                        "primary_intent": "reliability_query",
                        "targets": ["reliability_query"],
                        "contextualized_query": query,
                        "required_answer_fields": ["failure_probability"],
                        "selected_anchor_ids": [
                            "ev-rbd-1", "ev-rbd-2", "ev-rbd-3",
                        ],
                        "context_used": True,
                        "context_status": "resolved",
                    }, ensure_ascii=False),
                    offered_anchor_ids=offered,
                    raw_user_query=query,
                )

                self.assertEqual(
                    ["ev-rbd-1", "ev-rbd-2", "ev-rbd-3"],
                    decision["selected_anchor_ids"],
                )

    def test_diagram_selection_does_not_jump_to_latest_answer_origin(self):
        messages = [
            {
                "role": "assistant",
                "message_id": "answer-old",
                "conversation_anchors": [{
                    "anchor_id": "ev-rbd-old",
                    "doc_type": "rbd",
                    "doc_id": "doc-old",
                    "record_id": "RBD:OLD",
                }],
            },
            {
                "role": "assistant",
                "message_id": "answer-new",
                "conversation_anchors": [
                    {
                        "anchor_id": f"ev-rbd-new-{index}",
                        "doc_type": "rbd",
                        "doc_id": "doc-new",
                        "record_id": f"RBD:NEW:{index}",
                    }
                    for index in range(1, 5)
                ],
            },
        ]
        offered = plan_module._offered_conversation_anchors(messages)
        decision = plan_module._parse_target_response(
            json.dumps({
                "primary_intent": "reliability_query",
                "targets": ["reliability_query"],
                "contextualized_query": "把那组结构重新计算",
                "required_answer_fields": ["failure_probability"],
                "selected_anchor_ids": ["ev-rbd-old"],
                "context_used": True,
                "context_status": "resolved",
            }, ensure_ascii=False),
            offered_anchor_ids=[anchor["anchor_id"] for anchor in offered],
            raw_user_query="把那组结构重新计算",
        )

        selected = [
            anchor for anchor in offered
            if anchor["anchor_id"] in set(decision["selected_anchor_ids"])
        ]
        self.assertEqual(["ev-rbd-old"], decision["selected_anchor_ids"])
        self.assertEqual({"answer-old"}, {
            anchor.get("origin_message_id") for anchor in selected
        })

    def test_architecture_count_followup_has_no_raw_wording_promotion(self):
        anchor = {
            "anchor_id": "ev-subsystem",
            "record_id": "SUB-PWR",
            "answer_fields": {
                "collection_closed": True,
                "collection_parent_id": "SYN-ICP-9000",
                "collection_parent_name": "综合信息处理与显示系统",
                "collection_member_kind": "subsystem",
                "collection_member_count": 6,
            },
        }
        context = {
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "context_used": False,
            "selected_anchor_ids": [],
            "subjects": [],
            "requested_facets": [],
            "required_answer_fields": [],
            "status": "standalone",
            "contextualized_query": "所以总数是几个？刚才有没有把供电和显示相关部分漏掉？",
        }

        self.assertFalse(hasattr(plan_module, "_apply_collection_followup_safety"))
        self.assertEqual("focused_detail", context["coverage_scope"])

    def test_complete_collection_prefers_exact_component_subject_over_context_rows(self):
        rows = [
            {
                "doc_id": "doc-arch",
                "content_type": "function_component",
                "component_id": "SUB-PWR",
                "name": "电源与配电子系统",
                "abbreviation": "PWR",
            },
            {
                "doc_id": "doc-arch",
                "function_id": "FUNC-110",
                "module_id": "MOD-PWR-ICU",
                "function_name": "外部电源输入调理",
                "subsystem_name": "电源与配电子系统",
            },
            {
                "doc_id": "doc-arch",
                "function_id": "FUNC-120",
                "module_id": "MOD-PWR-PSUA",
                "function_name": "A路稳压供电",
                "subsystem_name": "电源与配电子系统",
            },
        ]
        state = {
            "query": "电源与配电子系统的故障模式有哪些？",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "电源与配电子系统",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "故障模式有哪些",
                },
                "collection_subject": "电源与配电子系统的故障模式",
                "entities": ["电源与配电子系统", "PDS"],
            }
        }

        component_ids, function_ids, doc_id = (
            diagnosis_module._closed_collection_scope_seeds(rows, state)
        )

        self.assertEqual(["SUB-PWR"], component_ids)
        self.assertEqual([], function_ids)
        self.assertEqual("doc-arch", doc_id)

    def test_unresolved_name_variant_cannot_authorize_closed_collection(self):
        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "SUB-PWR",
                "collection_entity_name": "电源与配电子系统",
                "collection_entity_abbreviation": "PWR",
            },
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "MOD-PWR-PDU",
                "collection_entity_name": "配电模块",
                "collection_entity_abbreviation": "PDU",
            },
            {
                "collection_entity_kind": "function",
                "doc_id": "doc-arch",
                "collection_entity_id": "FUNC-120",
                "collection_entity_name": "A路稳压供电",
            },
        ]
        state = {
            "query": "电源与配电模块的故障模式有哪些？",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "电源与配电模块",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "故障模式有哪些",
                },
                "collection_subject": "电源与配电模块的故障模式",
                "entities": ["电源与配电系统", "PDU", "PSU"],
            }
        }

        component_ids, function_ids, doc_id = (
            diagnosis_module._closed_collection_scope_seeds(rows, state)
        )

        # A lexical variant could denote the subsystem or the PDU.  Retrieval
        # may propose either, but this authority-only helper cannot choose it.
        self.assertEqual([], component_ids)
        self.assertEqual([], function_ids)
        self.assertEqual("", doc_id)

    def test_complete_collection_rejects_duplicate_active_name_outside_search_window(self):
        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-config-a",
                "collection_entity_id": "SUB-PWR-A",
                "collection_entity_name": "电源与配电子系统",
            },
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-config-b",
                "collection_entity_id": "SUB-PWR-B",
                "collection_entity_name": "电源与配电子系统",
            },
        ]
        state = {
            "query_frame": {
                "collection_subject": "电源与配电子系统的全部故障模式",
                "entities": ["电源与配电子系统"],
            }
        }

        self.assertEqual(
            ([], [], ""),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_complete_collection_rejects_same_label_across_entity_types(self):
        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "MOD-X",
                "collection_entity_name": "状态采集",
            },
            {
                "collection_entity_kind": "function",
                "doc_id": "doc-arch",
                "collection_entity_id": "FUNC-X",
                "collection_entity_name": "状态采集",
            },
        ]
        state = {
            "query_frame": {
                "collection_subject": "状态采集的所有故障模式",
                "entities": ["状态采集"],
            }
        }

        self.assertEqual(
            ([], [], ""),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_complete_collection_uses_resolved_conversation_business_id(self):
        rows = [{
            "collection_entity_kind": "function",
            "doc_id": "doc-arch",
            "collection_entity_id": "FUNC-530",
            "collection_entity_name": "故障记录存储",
        }]
        state = {
            "query": "它的所有故障模式有哪些？",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "它",
                    "member_kind": "failure_mode", "reference_mode": "selected_anchor",
                    "selected_anchor_ids": ["anchor-function"], "basis": "所有故障模式",
                },
                "collection_subject": "它的所有故障模式",
                "entities": ["FUNC-530"],
            },
            "selected_conversation_anchors": [{
                "anchor_id": "anchor-function",
                "doc_id": "doc-arch",
                "record_id": "FUNC-530",
                "answer_fields": {"function_id": "FUNC-530"},
            }],
        }

        self.assertEqual(
            ([], ["FUNC-530"], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_collection_parent_anchor_outvotes_multiple_member_function_ids(self):
        """Member citations may carry many FUNC IDs but share one SUB parent."""

        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "SUB-PWR",
                "collection_entity_name": "电源与配电子系统",
            },
            *[
                {
                    "collection_entity_kind": "function",
                    "doc_id": "doc-arch",
                    "collection_entity_id": function_id,
                    "collection_entity_name": f"供电功能{index}",
                }
                for index, function_id in enumerate(
                    ("FUNC-110", "FUNC-120", "FUNC-121", "FUNC-130"),
                    start=1,
                )
            ],
        ]
        anchors = [
            {
                "anchor_id": f"anchor-member-{index}",
                "record_id": f"FM-PWR-{index:03d}",
                "answer_fields": {
                    "collection_parent_id": "SUB-PWR",
                    "collection_parent_name": "电源与配电子系统",
                    "collection_member_kind": "failure_mode",
                    # This mirrors the FMECA product flag identities that are
                    # legitimately present in the same citation set.
                    "product_flag": f"PSU / {function_id}",
                },
            }
            for index, function_id in enumerate(
                ("FUNC-110", "FUNC-120", "FUNC-121", "FUNC-130"),
                start=1,
            )
        ]
        state = {
            "query": "刚才的集合还有遗漏的故障模式吗？",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "刚才的集合",
                    "member_kind": "failure_mode", "reference_mode": "selected_anchor",
                    "selected_anchor_ids": [anchor["anchor_id"] for anchor in anchors],
                    "basis": "还有遗漏的故障模式吗",
                },
                "collection_subject": "电源与配电子系统的故障模式",
                "entities": [
                    "电源与配电子系统", "SUB-PWR", "FUNC-110",
                    "FUNC-120", "FUNC-121", "FUNC-130",
                ],
            },
            "selected_conversation_anchors": anchors,
        }

        self.assertEqual(
            (["SUB-PWR"], [], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_collection_subject_exact_label_precedes_member_entity_ids(self):
        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "SUB-PWR",
                "collection_entity_name": "电源与配电子系统",
            },
            {
                "collection_entity_kind": "function",
                "doc_id": "doc-arch",
                "collection_entity_id": "FUNC-120",
                "collection_entity_name": "A路稳压供电",
            },
        ]
        state = {
            "query": "电源与配电子系统有哪些故障模式？",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "电源与配电子系统",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "有哪些故障模式",
                },
                "collection_subject": "电源与配电子系统",
                "entities": ["SUB-PWR", "FUNC-120"],
            }
        }

        self.assertEqual(
            (["SUB-PWR"], [], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_current_business_id_overrides_stale_conversation_anchor(self):
        rows = [
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "MOD-A",
                "collection_entity_name": "旧模块",
            },
            {
                "collection_entity_kind": "component",
                "doc_id": "doc-arch",
                "collection_entity_id": "MOD-B",
                "collection_entity_name": "新模块",
            },
        ]
        state = {
            "query": "MOD-B 的所有故障模式",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "MOD-B",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "所有故障模式",
                },
                "collection_subject": "MOD-B 的所有故障模式",
                "entities": ["MOD-B"],
            },
            "selected_conversation_anchors": [{
                "record_id": "MOD-A",
                "answer_fields": {"component_id": "MOD-A"},
            }],
        }

        self.assertEqual(
            (["MOD-B"], [], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_complete_collection_keeps_two_current_exact_paths_instead_of_stale_single_anchor(self):
        rows = [
            {
                "collection_entity_kind": "function",
                "doc_id": "doc-arch",
                "collection_entity_id": "FUNC-120",
                "collection_entity_name": "A路稳压供电",
            },
            {
                "collection_entity_kind": "function",
                "doc_id": "doc-arch",
                "collection_entity_id": "FUNC-121",
                "collection_entity_name": "B路稳压供电",
            },
        ]
        state = {
            "query": "FUNC-120 和 FUNC-121 的全部故障模式",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "FUNC-120 和 FUNC-121",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "全部故障模式",
                },
                "collection_subject": "两路稳压供电的全部故障模式",
                # These are semantic-model outputs for the current turn, not
                # a rule-generated A/B expansion.
                "entities": ["A路稳压供电", "B路稳压供电"],
            },
            "selected_conversation_anchors": [{
                "record_id": "FUNC-120",
                "answer_fields": {"collection_parent_id": "FUNC-120"},
            }],
        }

        self.assertEqual(
            ([], ["FUNC-120", "FUNC-121"], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_business_id_embedded_in_subject_binds_without_entity_slot(self):
        rows = [{
            "collection_entity_kind": "component",
            "doc_id": "doc-arch",
            "collection_entity_id": "MOD-B",
            "collection_entity_name": "新模块",
        }]
        state = {
            "query": "请找 MOD-B 的所有故障模式。",
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "请找 MOD-B 的所有故障模式。",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "所有故障模式",
                },
                "collection_subject": "请找 MOD-B 的所有故障模式。",
                "entities": [],
            }
        }

        self.assertEqual(
            (["MOD-B"], [], "doc-arch"),
            diagnosis_module._closed_collection_scope_seeds(rows, state),
        )

    def test_state_history_keeps_answer_citations_but_not_candidate_catalog(self):
        database = AgentDB()
        messages = [
            {"role": "user", "content": "某模块是什么"},
            {
                "role": "assistant",
                "content": "它是电源模块。[^1]",
                "citations": [{
                    "evidence_id": "ev-used",
                    "doc_type": "function_locator",
                    "record_id": "MOD-PWR-PSUA",
                    "document_metadata": {
                        "answer_fields": {"component_id": "MOD-PWR-PSUA"}
                    },
                }],
                "evidence_catalog": [{
                    "evidence_id": "ev-candidate",
                    "status": "candidate",
                }],
            },
        ]
        with patch.object(database, "list_messages", return_value=messages):
            result = database.list_messages_for_state("session-1")

        anchors = result[1]["conversation_anchors"]
        self.assertEqual(["ev-used"], [item["anchor_id"] for item in anchors])
        self.assertNotIn("evidence_catalog", result[1])

    def test_history_keeps_only_answer_used_identity_fields(self):
        anchors = _conversation_anchors([
            {
                "id": 1,
                "evidence_id": "ev-fm-2",
                "doc_type": "fmeca",
                "doc_id": "doc-fmeca",
                "record_id": "FM-SYN-002",
                "title": "输入滤波性能退化",
                "source_ref": "FMECA.docx#row=3",
                "document_metadata": {
                    "answer_fields": {
                        "record_code": "FM-SYN-002",
                        "failure_mode": "输入滤波性能退化",
                        "collection_parent_id": "SUB-PWR",
                        "collection_parent_name": "电源与配电子系统",
                        "collection_member_kind": "failure_mode",
                        "collection_member_count": 15,
                        "collection_member_id": "FM-SYN-002",
                        "collection_member_name": "输入滤波性能退化",
                        "collection_closed": True,
                        "severity": "III",
                        "final_impact": "多个子系统出现间歇性异常",
                    }
                },
            }
        ])
        self.assertEqual(1, len(anchors))
        self.assertEqual("FM-SYN-002", anchors[0]["record_id"])
        self.assertEqual("FM-SYN-002", anchors[0]["answer_fields"]["record_code"])
        self.assertEqual("SUB-PWR", anchors[0]["answer_fields"]["collection_parent_id"])
        self.assertEqual(
            "failure_mode", anchors[0]["answer_fields"]["collection_member_kind"]
        )
        self.assertEqual(15, anchors[0]["answer_fields"]["collection_member_count"])
        self.assertNotIn("severity", anchors[0]["answer_fields"])
        self.assertNotIn("final_impact", anchors[0]["answer_fields"])

    def test_history_keeps_twelve_answer_used_anchors_for_bounded_followup(self):
        citations = [
            {
                "evidence_id": f"ev-{index}",
                "doc_type": "rbd",
                "record_id": f"RBD:{index}",
            }
            for index in range(1, 14)
        ]

        anchors = _conversation_anchors(citations)

        self.assertEqual(12, len(anchors))
        self.assertEqual("ev-12", anchors[-1]["anchor_id"])

    def test_target_parser_rejects_model_invented_anchor_id(self):
        result = plan_module._parse_target_response(
            json.dumps({
                "primary_intent": "impact_analysis",
                "targets": ["impact_analysis"],
                "contextualized_query": "ICU 输入异常的严酷度",
                "subjects": ["ICU 输入异常"],
                "requested_facets": ["严酷度"],
                "context_used": True,
                "context_status": "resolved",
                "selected_anchor_ids": ["ev-real", "ev-invented"],
            }, ensure_ascii=False),
            offered_anchor_ids=["ev-real"],
        )
        self.assertEqual(["ev-real"], result["selected_anchor_ids"])

    def test_target_parser_accepts_complete_collection_metadata(self):
        raw_query = "电源与配电子系统的故障模式中，A路间歇停机是否为唯一原因"
        result = plan_module._parse_target_response(
            json.dumps({
                "primary_intent": "cause_analysis",
                "targets": ["cause_analysis"],
                "contextualized_query": (
                    "电源与配电子系统的故障模式中，A路间歇停机是否为唯一原因"
                ),
                "subjects": ["电源与配电子系统", "A路间歇停机"],
                "requested_facets": ["故障模式", "原因", "完整性"],
                "coverage_request": {
                    "scope": "complete_collection",
                    "subject": "电源与配电子系统",
                    "member_kind": "failure_mode",
                    "reference_mode": "current_query",
                    "basis": "是否为唯一原因",
                },
                "context_used": True,
                "context_status": "resolved",
                "selected_anchor_ids": ["ev-pwr"],
            }, ensure_ascii=False),
            offered_anchor_ids=["ev-pwr"],
            raw_user_query=raw_query,
        )

        self.assertEqual("complete_collection", result["coverage_scope"])
        self.assertEqual("电源与配电子系统", result["collection_subject"])
        self.assertEqual("failure_mode", result["collection_member_kind"])
        self.assertEqual(["ev-pwr"], result["selected_anchor_ids"])
        self.assertIn(
            "closed_collection_members", result["required_answer_fields"]
        )

    def test_target_parser_preserves_semantic_location_and_incident_contracts(self):
        raw_query = "运行中反复复位时应优先定位哪个板级环节"
        result = plan_module._parse_target_response(
            json.dumps({
                "primary_intent": "function_location",
                "targets": ["function_location"],
                "contextualized_query": "运行中反复复位时应优先定位哪个板级环节",
                "subjects": ["运行中反复复位"],
                "requested_facets": ["故障定位", "维修定位"],
                "required_answer_fields": [
                    "fault_location", "maintenance_locator",
                ],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": [
                        "fault_location", "maintenance_locator",
                    ],
                    "basis": raw_query,
                    "reason": "用户描述正在发生的异常并要求排查",
                },
                "context_used": False,
                "context_status": "standalone",
            }, ensure_ascii=False),
            raw_user_query=raw_query,
        )

        self.assertEqual("function_location", result["primary_intent"])
        self.assertEqual(
            ["function_location", "maintenance_query"],
            [item["intent"] for item in result["query_targets"]],
        )
        self.assertEqual(
            {"fault_location", "maintenance_locator"},
            set(result["required_answer_fields"]),
        )
        self.assertEqual(
            {
                "required": True,
                "required_dimensions": [
                    "fault_location", "maintenance_locator",
                ],
                "basis": raw_query,
                "reason": "用户描述正在发生的异常并要求排查",
            },
            result["diagnostic_closure"],
        )

    def test_legacy_complete_collection_contract_fails_closed_without_repair(self):
        initial = json.dumps({
            "primary_intent": "impact_analysis",
            "targets": ["impact_analysis"],
            "contextualized_query": "MOD-PWR-ICU可能出现哪些故障模式",
            "subjects": ["MOD-PWR-ICU"],
            "requested_facets": ["故障模式"],
            "required_answer_fields": ["fault_cause", "final_impact"],
            "coverage_scope": "complete_collection",
            "collection_subject": "MOD-PWR-ICU",
            "collection_member_kind": "unknown",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
        }, ensure_ascii=False)
        repaired = json.dumps({
            "primary_intent": "cause_analysis",
            "targets": ["cause_analysis"],
            "contextualized_query": "MOD-PWR-ICU的全部故障模式",
            "subjects": ["MOD-PWR-ICU"],
            "requested_facets": ["故障模式"],
            "required_answer_fields": ["closed_collection_members"],
            "coverage_scope": "complete_collection",
            "collection_subject": "MOD-PWR-ICU",
            "collection_member_kind": "failure_mode",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
        }, ensure_ascii=False)

        with patch.object(
            plan_module.llm_service,
            "_call_model",
            side_effect=[initial, repaired],
        ) as call:
            result = _run(plan_module._classify_explicit_targets(
                query="MOD-PWR-ICU可能出现哪些故障模式？请完整列出。",
                history="无",
                conversation_anchors=[],
                model={"provider": "ollama"},
            ))

        self.assertEqual(1, call.call_count)
        self.assertEqual("unknown", result["coverage_scope"])
        self.assertEqual("unknown", result["collection_member_kind"])
        self.assertNotIn("closed_collection_members", result["required_answer_fields"])

    def test_plain_cause_question_cannot_be_promoted_to_complete_collection(self):
        model_response = json.dumps({
            "primary_intent": "cause_analysis",
            "targets": ["cause_analysis"],
            "contextualized_query": "什么会导致电源不可用",
            "subjects": ["电源不可用"],
            "requested_facets": ["原因"],
            "required_answer_fields": ["fault_cause"],
            # Simulate the exact overreach observed in a live query: the
            # model sees many possible rows and mistakes that for a user
            # request for an exhaustive, closed list.
            "coverage_scope": "complete_collection",
            "collection_subject": "电源不可用",
            "collection_member_kind": "cause",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
        }, ensure_ascii=False)

        with patch.object(
            plan_module.llm_service,
            "_call_model",
            return_value=model_response,
        ) as call:
            result = _run(plan_module._classify_explicit_targets(
                query="什么会导致电源不可用",
                history="无",
                conversation_anchors=[],
                model={"provider": "ollama"},
            ))

        call.assert_called_once()
        self.assertEqual("unknown", result["coverage_scope"])
        self.assertEqual("", result["collection_subject"])
        self.assertEqual("unknown", result["collection_member_kind"])
        self.assertEqual(["fault_cause"], result["required_answer_fields"])
        self.assertNotIn("closed_collection_members", result["required_answer_fields"])

    def test_complete_collection_target_scope_overrides_planner_detail_scope(self):
        target_response = json.dumps({
            "primary_intent": "cause_analysis",
            "targets": ["cause_analysis"],
            "contextualized_query": (
                "电源与配电子系统的故障模式中，A路间歇停机是否为唯一原因"
            ),
            "subjects": ["电源与配电子系统", "A路间歇停机"],
            "requested_facets": ["故障模式", "原因", "完整性"],
            "coverage_scope": "complete_collection",
            "collection_subject": "电源与配电子系统",
            "collection_member_kind": "failure_mode",
            "coverage_request": {
                "scope": "complete_collection",
                "subject": "电源与配电子系统",
                "member_kind": "failure_mode",
                "reference_mode": "selected_anchor",
                "basis": "只有 A路间歇停机 这一个原因吗",
                "selected_anchor_ids": ["ev-pwr"],
            },
            "context_used": True,
            "context_status": "resolved",
            "selected_anchor_ids": ["ev-pwr"],
        }, ensure_ascii=False)
        # The second planner is allowed to optimize tool calls, but it must not
        # narrow an already resolved exhaustive follow-up back to one record.
        planner_response = json.dumps({
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "rewritten_query": "A路间歇停机的原因",
            "query_frame": {"coverage_scope": "focused_detail"},
            "tool_queries": {
                "fmeca_query": ["电源与配电子系统 故障模式 原因"],
            },
            "tools_plan": [{"tool": "fmeca_query", "reason": "查询故障模式"}],
            "relation_expansion": {"needed": False, "reason": "直接记录可回答"},
        }, ensure_ascii=False)
        messages = [
            {"role": "user", "content": "电源与配电子系统如果故障，原因有哪些？"},
            {
                "role": "assistant",
                "content": "资料中有 A路间歇停机这一条记录。",
                "conversation_anchors": [{
                    "anchor_id": "ev-pwr",
                    "doc_type": "function_locator",
                    "record_id": "SUB-PWR",
                    "answer_fields": {
                        "component_id": "SUB-PWR",
                        "component_name": "电源与配电子系统",
                    },
                }],
            },
            {"role": "user", "content": "只有 A路间歇停机 这一个原因吗？"},
        ]
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
            result = _run(plan_module.plan_node({
                "query": "只有 A路间歇停机 这一个原因吗？",
                "messages": messages,
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertEqual("complete_collection", result["query_frame"]["coverage_scope"])
        self.assertEqual("电源与配电子系统", result["query_frame"]["collection_subject"])
        self.assertEqual(
            "failure_mode", result["query_frame"]["collection_member_kind"]
        )
        self.assertEqual("complete_collection", result["task_profile"]["coverage_scope"])
        self.assertEqual("SUB-PWR", result["selected_conversation_anchors"][0]["record_id"])

    def test_complete_collection_target_blocks_planner_answer_scope_expansion(self):
        query = "MOD-PWR-ICU 有哪些故障模式？请完整列出。"
        contextualized_query = "完整列出 MOD-PWR-ICU 的故障模式"
        target_response = json.dumps(
            {
                "primary_intent": "cause_analysis",
                "targets": ["cause_analysis"],
                "contextualized_query": contextualized_query,
                "subjects": ["MOD-PWR-ICU"],
                # The collection kind already says what the members are.  The
                # only output obligation emitted by the target stage is N-of-N
                # closure; no cause/effect detail was requested.
                "requested_facets": ["完整列举"],
                "required_answer_fields": ["closed_collection_members"],
                "coverage_scope": "complete_collection",
                "collection_subject": "MOD-PWR-ICU",
                "collection_member_kind": "failure_mode",
                "coverage_request": {
                    "scope": "complete_collection",
                    "subject": "MOD-PWR-ICU",
                    "member_kind": "failure_mode",
                    "reference_mode": "current_query",
                    "basis": "请完整列出",
                },
                "context_used": False,
                "context_status": "standalone",
                "selected_anchor_ids": [],
            },
            ensure_ascii=False,
        )
        planner_response = json.dumps(
            {
                "intent": "cause_analysis",
                "query_targets": [{"intent": "cause_analysis"}],
                "rewritten_query": (
                    "完整列出 MOD-PWR-ICU 的故障模式，并逐项说明原因、"
                    "局部影响和最终影响"
                ),
                "entity_keywords": ["MOD-PWR-ICU"],
                "query_frame": {
                    "coverage_scope": "complete_collection",
                    "requested_facets": ["原因", "局部影响", "最终影响"],
                    "required_answer_fields": [
                        "fault_cause", "local_impact", "final_impact"
                    ],
                },
                "tool_queries": {
                    "fmeca_query": [
                        "MOD-PWR-ICU 故障模式 原因 局部影响 最终影响"
                    ]
                },
                "tools_plan": [
                    {"tool": "fmeca_query", "reason": "查询故障模式记录"}
                ],
                "relation_expansion": {
                    "needed": False,
                    "reason": "结构化记录可以回答",
                },
            },
            ensure_ascii=False,
        )

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
            result = _run(
                plan_module.plan_node(
                    {
                        "query": query,
                        "messages": [{"role": "user", "content": query}],
                        "memory_enabled": False,
                        "iteration": 0,
                    }
                )
            )

        self.assertEqual(contextualized_query, result["rewritten_query"])
        self.assertEqual(
            contextualized_query,
            result["query_frame"]["rewritten_query"],
        )
        self.assertEqual(
            ["完整列举"], result["query_frame"]["requested_facets"]
        )
        self.assertEqual(
            ["closed_collection_members"],
            result["query_frame"]["required_answer_fields"],
        )
        self.assertEqual(
            ["closed_collection_members"],
            result["query_targets"][0]["required_fields"],
        )
        self.assertEqual(
            ["closed_collection_members"],
            result["task_profile"]["required_answer_fields"],
        )
        self.assertNotIn("原因", result["rewritten_query"])
        self.assertNotIn("影响", result["rewritten_query"])

    def test_exhaustive_followup_does_not_bypass_missing_typed_contract(self):
        """Raw wording and prior prose cannot manufacture an N-of-N contract."""

        target_response = json.dumps({
            "primary_intent": "cause_analysis",
            "targets": ["cause_analysis"],
            "contextualized_query": "运行一阵子A路自己停掉是否为唯一一种",
            "subjects": ["A路自己停掉"],
            "requested_facets": ["原因"],
            # Reproduce the observed local-model miss.
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
        }, ensure_ascii=False)
        planner_response = json.dumps({
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "rewritten_query": "A路自己停掉的原因",
            "entity_keywords": ["A路自己停掉"],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "collection_member_kind": "unknown",
            },
            "tool_queries": {"fmeca_query": ["A路自己停掉"]},
            "tools_plan": [{"tool": "fmeca_query", "reason": "查询该条记录"}],
            "relation_expansion": {"needed": False, "reason": "单条记录"},
        }, ensure_ascii=False)
        cited_members = [
            {
                "anchor_id": f"ev-pwr-{index}",
                "doc_type": "fmeca",
                "record_id": f"FM-PWR-{index:03d}",
                "answer_fields": {
                    "record_code": f"FM-PWR-{index:03d}",
                    "failure_mode": f"供电故障模式{index}",
                    "collection_parent_id": "SUB-PWR",
                    "collection_parent_name": "电源与配电子系统",
                    "collection_member_kind": "failure_mode",
                    "collection_member_count": 15,
                    "collection_member_id": f"FM-PWR-{index:03d}",
                    "collection_member_name": f"供电故障模式{index}",
                    "collection_closed": True,
                },
            }
            for index in range(1, 4)
        ]
        query = "所以运行一阵子A路自己停掉，就是这个范围里唯一的一种吗？"
        messages = [
            {"role": "user", "content": "电源与配电这一块都有哪些故障表现和成因？"},
            {
                "role": "assistant",
                "content": "资料列出了多条故障模式。",
                "conversation_anchors": cited_members,
            },
            {"role": "user", "content": query},
        ]
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
            result = _run(plan_module.plan_node({
                "query": query,
                "messages": messages,
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertEqual("unknown", result["query_frame"]["coverage_scope"])
        self.assertEqual("", result["query_frame"]["collection_subject"])
        self.assertEqual("unknown", result["query_frame"]["collection_member_kind"])
        self.assertEqual([], result["selected_conversation_anchors"])
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"], result["allowed_tools"]
        )
        self.assertNotIn("fmeca_query", result["allowed_tools"])

    def test_exhaustive_followup_has_no_raw_collection_helper(self):
        anchors = [
            {
                "anchor_id": "ev-a",
                "answer_fields": {
                    "collection_parent_id": "SUB-A",
                    "collection_parent_name": "子系统A",
                    "collection_member_kind": "failure_mode",
                },
            },
            {
                "anchor_id": "ev-b",
                "answer_fields": {
                    "collection_parent_id": "SUB-B",
                    "collection_parent_name": "子系统B",
                    "collection_member_kind": "failure_mode",
                },
            },
        ]
        context = {
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "context_used": False,
            "status": "standalone",
            "subjects": [],
            "requested_facets": [],
            "selected_anchor_ids": [],
            "contextualized_query": "这个范围里只有一种吗",
        }

        self.assertFalse(hasattr(plan_module, "_apply_collection_followup_safety"))
        self.assertEqual("focused_detail", context["coverage_scope"])

    def test_power_subsystem_only_a_route_followup_is_marked_complete_collection(self):
        target_response = json.dumps({
            "primary_intent": "cause_analysis",
            "targets": ["cause_analysis"],
            "contextualized_query": (
                "电源与配电子系统的所有故障模式中，A路间歇停机是否为唯一原因"
            ),
            "subjects": ["电源与配电子系统", "A路间歇停机"],
            "requested_facets": ["故障模式", "原因", "完整性"],
            "coverage_scope": "complete_collection",
            "collection_subject": "电源与配电子系统",
            "collection_member_kind": "failure_mode",
            "coverage_request": {
                "scope": "complete_collection",
                "subject": "电源与配电子系统",
                "member_kind": "failure_mode",
                "reference_mode": "selected_anchor",
                "basis": "只有 A路间歇停机 这一个原因吗",
                "selected_anchor_ids": ["ev-pwr"],
            },
            "context_used": True,
            "context_status": "resolved",
            "selected_anchor_ids": ["ev-pwr"],
        }, ensure_ascii=False)
        planner_response = json.dumps({
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "rewritten_query": "电源与配电子系统故障模式是否只有 A路间歇停机",
            "query_frame": {"coverage_scope": "complete_collection"},
            "tool_queries": {"fmeca_query": ["SUB-PWR 故障模式"]},
            "tools_plan": [{"tool": "fmeca_query", "reason": "核验完整集合"}],
            "relation_expansion": {"needed": False, "reason": "先核验权威记录"},
        }, ensure_ascii=False)
        state = {
            "query": "只有 A路间歇停机 这一个原因吗？",
            "messages": [
                {"role": "user", "content": "电源与配电子系统如果故障，原因有哪些？"},
                {
                    "role": "assistant",
                    "content": "其中一个方向是 A路间歇停机。",
                    "conversation_anchors": [{
                        "anchor_id": "ev-pwr",
                        "doc_type": "function_locator",
                        "record_id": "SUB-PWR",
                        "answer_fields": {"component_id": "SUB-PWR"},
                    }],
                },
                {"role": "user", "content": "只有 A路间歇停机 这一个原因吗？"},
            ],
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
            planned = _run(plan_module.plan_node(state))

        state.update(planned)
        self.assertEqual("complete_collection", state["query_frame"]["coverage_scope"])
        self.assertEqual("电源与配电子系统", state["query_frame"]["collection_subject"])
        self.assertEqual(
            "failure_mode", state["query_frame"]["collection_member_kind"]
        )

    def test_followup_severity_resolves_subject_and_selects_previous_fmeca(self):
        target_response = json.dumps({
            "primary_intent": "impact_analysis",
            "targets": ["impact_analysis"],
            "contextualized_query": "ICU 在电源波动时出现关联状态异常的严酷度是什么",
            "subjects": ["ICU", "电源波动时关联状态异常"],
            # Retrieval facets may remain broad, but the explicit answer
            # contract below must keep this follow-up focused on severity.
            "requested_facets": ["严酷度", "局部影响", "最终影响"],
            "required_answer_fields": ["severity"],
            "context_used": True,
            "context_status": "resolved",
            "selected_anchor_ids": ["ev-fm-2"],
        }, ensure_ascii=False)
        planner_response = json.dumps({
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "rewritten_query": "这种异常什么级别",
            "entity_keywords": [],
            "type_keywords": ["严酷度"],
            "query_frame": {"coverage_scope": "focused_detail"},
            "tool_queries": {
                "fmeca_query": ["ICU 电源波动 关联状态异常 严酷度"],
            },
            "tools_plan": [{"tool": "fmeca_query", "reason": "查询严酷度"}],
            "relation_expansion": {"needed": False, "reason": "直接记录可回答"},
            "reasoning": "承接上一轮对象查询其严酷度",
        }, ensure_ascii=False)
        state = {
            "query": "这种异常什么级别",
            "messages": [
                {"role": "user", "content": "ICU关联状态异常：电源一波动就出错"},
                {
                    "role": "assistant",
                    "content": "检索到输入滤波性能退化。",
                    "conversation_anchors": [{
                        "anchor_id": "ev-fm-2",
                        "evidence_id": "ev-fm-2",
                        "doc_type": "fmeca",
                        "doc_id": "doc-fmeca",
                        "record_id": "FM-SYN-002",
                        "title": "输入滤波性能退化",
                        "answer_fields": {"record_code": "FM-SYN-002"},
                    }],
                },
                {"role": "user", "content": "这种异常什么级别"},
            ],
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

        self.assertEqual("impact_analysis", result["intent"])
        self.assertEqual("knowledge_base", result["scope_route"])
        self.assertIn("严酷度", result["query_frame"]["requested_facets"])
        self.assertEqual(
            ["severity"], result["query_frame"]["required_answer_fields"]
        )
        self.assertIn("fmeca_query", result["allowed_tools"])
        self.assertEqual(
            "FM-SYN-002",
            result["selected_conversation_anchors"][0]["record_id"],
        )
        self.assertIn("ICU", result["rewritten_query"])

    def test_unresolved_followup_stops_before_retrieval(self):
        target_response = json.dumps({
            "primary_intent": "impact_analysis",
            "targets": ["impact_analysis"],
            "contextualized_query": "它的重要性",
            "subjects": [],
            "requested_facets": ["重要性"],
            "context_used": True,
            "context_status": "missing",
            "selected_anchor_ids": [],
        }, ensure_ascii=False)
        planner_response = json.dumps({
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "rewritten_query": "它的重要性",
            "query_frame": {},
            "tools_plan": [{"tool": "fmeca_query"}],
        }, ensure_ascii=False)
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
            result = _run(plan_module.plan_node({
                "query": "它重要吗",
                "messages": [{"role": "user", "content": "它重要吗"}],
                "memory_enabled": False,
                "iteration": 0,
            }))
        self.assertEqual("knowledge_base", result["scope_route"])
        self.assertEqual([], result["allowed_tools"])
        self.assertEqual("", result["clarification_question"])
        self.assertEqual(
            "unresolved_identity", result["clarification_reasons"][0]["kind"]
        )
        self.assertTrue(result["clarification_fallback"]["required"])
        self.assertEqual("icp_diagnosis", plan_module.route_after_plan(result))

    def test_module_importance_followup_keeps_object_and_combines_goal_facets(self):
        target_response = json.dumps({
            "primary_intent": "impact_analysis",
            "targets": ["impact_analysis", "architecture_query"],
            "contextualized_query": "MOD-PWR-PSUA 为什么重要",
            "subjects": ["MOD-PWR-PSUA"],
            "requested_facets": ["重要性", "职责", "上下游依赖", "失效影响"],
            "required_answer_fields": [
                "architecture_function", "local_impact", "final_impact", "relation_path",
            ],
            "relation_expansion": {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "selected_anchor",
                "basis": "它重要吗",
                "selected_anchor_ids": ["ev-component"],
                "reason": "需要核对所选模块的上下游依赖",
            },
            "context_used": True,
            "context_status": "resolved",
            "selected_anchor_ids": ["ev-component"],
        }, ensure_ascii=False)
        planner_response = json.dumps({
            "intent": "impact_analysis",
            "query_targets": [
                {"intent": "impact_analysis"},
                {"intent": "architecture_query"},
            ],
            "rewritten_query": "它重要吗",
            "query_frame": {"coverage_scope": "focused_detail"},
            "tool_queries": {
                "function_locator": ["MOD-PWR-PSUA 职责 上下游"],
                "fmeca_query": ["MOD-PWR-PSUA 失效影响"],
            },
            "tools_plan": [
                {"tool": "function_locator", "reason": "查询职责与依赖"},
                {"tool": "fmeca_query", "reason": "查询失效影响"},
            ],
            "relation_expansion": {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "selected_anchor",
                "basis": "它重要吗",
                "selected_anchor_ids": ["ev-component"],
                "reason": "需要核对上下游关系",
            },
        }, ensure_ascii=False)
        messages = [
            {"role": "user", "content": "MOD-PWR-PSUA 是什么"},
            {
                "role": "assistant",
                "content": "它是电源 A 模块。",
                "conversation_anchors": [{
                    "anchor_id": "ev-component",
                    "doc_type": "function_locator",
                    "record_id": "MOD-PWR-PSUA",
                    "answer_fields": {"component_id": "MOD-PWR-PSUA"},
                }],
            },
            {"role": "user", "content": "它重要吗"},
        ]
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
            result = _run(plan_module.plan_node({
                "query": "它重要吗",
                "messages": messages,
                "memory_enabled": False,
                "iteration": 0,
            }))

        intents = [item["intent"] for item in result["query_targets"]]
        self.assertEqual(["impact_analysis", "architecture_query"], intents)
        self.assertIn("function_locator", result["allowed_tools"])
        self.assertIn("fmeca_query", result["allowed_tools"])
        self.assertTrue(result["relation_expansion"]["needed"])
        self.assertEqual(["MOD-PWR-PSUA"], result["query_frame"]["entities"])
        self.assertIn("重要性", result["query_frame"]["requested_facets"])

    def test_entity_profile_reads_component_table_not_only_function_rows(self):
        state = {
            "scope_route": "knowledge_base",
            "intent": "entity_profile",
            "query_targets": [{"intent": "entity_profile"}],
            "query": "MOD-PWR-PSUA 是什么",
            "rewritten_query": "MOD-PWR-PSUA 的名称、类型、位置和职责",
            "allowed_tools": ["function_locator", "evidence_synthesis"],
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }
        component = {
            "doc_id": "doc-function",
            "component_id": "MOD-PWR-PSUA",
            "component_type": "模块",
            "name": "电源 A 模块",
            "parent_component_id": "SUB-PWR",
            "description": "为输入调理链路提供主路供电。",
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=[]),
            patch.object(diagnosis_module.icp_repository, "search_components", return_value=[component]),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: list(rows),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        component_evidence = [
            item for item in result["evidence_registry"]
            if item.get("claim_type") == "component_fact"
        ]
        self.assertEqual(1, len(component_evidence))
        fields = component_evidence[0]["metadata"]["answer_fields"]
        self.assertEqual("电源 A 模块", fields["component_name"])
        self.assertEqual("模块", fields["component_type"])

    def test_business_record_entity_profile_executes_its_structured_query(self):
        cases = (
            {
                "business_id": "91810001**0001",
                "tool": "hmc_query",
                "query_owner": diagnosis_module,
                "query_attr": "_query_hmc",
                "doc_type": "hmc",
                "claim_type": "hmc_fault",
                "row": {
                    "doc_id": "doc-hmc",
                    "doc_name": "HMC故障代码手册.docx",
                    "hmc_code": "91810001**0001",
                    "lcn": "91810001",
                    "hamc": "**",
                    "hsc": "0001",
                    "fault_desc": "ICU 输入过压钳位失效",
                    "match_type": "exact_code",
                },
            },
            {
                "business_id": "FM-SYN-002",
                "tool": "fmeca_query",
                "query_owner": diagnosis_module,
                "query_attr": "_query_fmeca",
                "doc_type": "fmeca",
                "claim_type": "fmeca_fact",
                "row": {
                    "doc_id": "doc-fmeca",
                    "doc_name": "FMECA分析报告.docx",
                    "code": "FM-SYN-002",
                    "fault_mode": "输入滤波性能退化",
                    "fault_cause": "滤波电容老化",
                    "final_impact": "输入质量下降",
                },
            },
            {
                "business_id": "MFL-SYN-041",
                "tool": "mfl_query",
                "query_owner": diagnosis_module,
                "query_attr": "_query_mfl",
                "doc_type": "mfl",
                "claim_type": "maintenance_locator",
                "row": {
                    "doc_id": "doc-mfl",
                    "doc_name": "MFL维修故障清单.xlsx",
                    "mfl_code": "MFL-SYN-041",
                    "module_fault": "输入调理模块异常",
                    "subcircuit_fault": "滤波支路退化",
                    "maintenance_action": "检查滤波支路并复测",
                },
            },
            {
                "business_id": "HIS-SYN-020",
                "tool": "fault_history_query",
                "query_owner": diagnosis_module.icp_repository,
                "query_attr": "search_history",
                "doc_type": "fault_history",
                "claim_type": "historical_case",
                "row": {
                    "doc_id": "doc-history",
                    "doc_name": "故障历史查询文档.docx",
                    "case_id": "HIS-SYN-020",
                    "occurred_at": "2026-01-29 14:05:00",
                    "fault_symptom": "输入状态间歇异常",
                    "root_cause": "连接器针脚后退",
                    "final_solution": "重新锁止针脚",
                    "verification_result": "复测正常",
                },
            },
        )

        for case in cases:
            business_id = case["business_id"]
            expected_tool = case["tool"]
            state = {
                "scope_route": "knowledge_base",
                "intent": "entity_profile",
                "query_targets": [
                    {
                        "intent": "entity_profile",
                        "required_fields": ["record_identity"],
                    }
                ],
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["record_identity"],
                    "entities": [business_id],
                },
                "query": f"{business_id} 是什么",
                "rewritten_query": f"查询业务记录 {business_id} 的身份",
                "entity_keywords": [business_id],
                "tool_queries": {expected_tool: [business_id]},
                "allowed_tools": [expected_tool, "evidence_synthesis"],
                "sql_limit": 8,
                "retrieval_top_k": 8,
            }
            with self.subTest(tool=expected_tool, business_id=business_id):
                with (
                    patch.object(
                        case["query_owner"],
                        case["query_attr"],
                        return_value=[case["row"]],
                    ) as structured_query,
                    patch.object(
                        diagnosis_module.icp_repository,
                        "search_functions",
                        side_effect=AssertionError(
                            "业务记录身份查询不得进入 Function Locator"
                        ),
                    ),
                    patch.object(
                        diagnosis_module.icp_repository,
                        "search_components",
                        side_effect=AssertionError(
                            "业务记录身份查询不得进入组件查询"
                        ),
                    ),
                    patch.object(
                        diagnosis_module,
                        "_enrich_function_sources",
                        side_effect=lambda rows, errors=None: [dict(row) for row in rows],
                    ),
                ):
                    result = _run(diagnosis_module.icp_diagnosis_node(state))

                structured_query.assert_called_once()
                self.assertIn(business_id, str(structured_query.call_args))
                self.assertEqual(
                    [expected_tool, "evidence_synthesis"],
                    [item["tool"] for item in result["tool_results"]],
                )
                matched_evidence = [
                    item
                    for item in result["evidence_registry"]
                    if item.get("doc_type") == case["doc_type"]
                    and item.get("record_id") == business_id
                ]
                self.assertEqual(1, len(matched_evidence))
                self.assertEqual(
                    case["claim_type"], matched_evidence[0]["claim_type"]
                )

    def test_explicit_business_id_keeps_only_exact_stored_identity_across_probes(self):
        # The deliberately crossed storage families prove that no prefix table
        # is being asserted by this test.  Only the exact value found in an
        # authoritative identity column decides which evidence survives.
        cases = (
            ("FM-SYN-002", "mfl_query", "mfl", "maintenance_locator"),
            (
                "MFL-SYN-041",
                "fault_history_query",
                "fault_history",
                "historical_case",
            ),
            ("HIS-SYN-020", "fmeca_query", "fmeca", "fmeca_fact"),
            (
                "MOD-PWR-PSUA",
                "function_locator",
                "function_locator",
                "component_fact",
            ),
        )
        probe_tools = {
            "function_locator",
            "hmc_query",
            "fmeca_query",
            "mfl_query",
            "fault_history_query",
        }

        for business_id, stored_tool, doc_type, claim_type in cases:
            with self.subTest(business_id=business_id, stored_tool=stored_tool):
                policy = resolve_tool_policy(
                    intent="entity_profile",
                    query=f"{business_id} 是什么",
                    raw_user_query=f"{business_id} 是什么",
                    query_targets=[
                        {
                            "intent": "entity_profile",
                            "required_fields": ["record_identity"],
                        }
                    ],
                    required_answer_fields=["record_identity"],
                    coverage_scope="focused_detail",
                    proposed_plan=[{"tool": "function_locator"}],
                )
                self.assertTrue(probe_tools.issubset(policy["allowed_tools"]))

                function_rows = [
                    {
                        "doc_id": "doc-function-noise",
                        "function_id": "FUNC-NOISE-001",
                        "function_name": f"文字中出现 {business_id}",
                        "description": f"{business_id} 的宽关键词噪声",
                    }
                ]
                component_rows = [
                    {
                        "doc_id": "doc-component-noise",
                        "component_id": "MOD-NOISE-001",
                        "component_type": "module",
                        "name": f"文字中出现 {business_id}",
                        "description": f"{business_id} 的组件关键词噪声",
                    }
                ]
                hmc_rows = [
                    {
                        "doc_id": "doc-hmc-noise",
                        "hmc_code": "91810001**9999",
                        "fault_desc": f"描述包含 {business_id}",
                    }
                ]
                fmeca_rows = [
                    {
                        "doc_id": "doc-fmeca-noise",
                        "code": "FM-NOISE-001",
                        "fault_mode": f"文字中出现 {business_id}",
                    }
                ]
                mfl_rows = [
                    {
                        "doc_id": "doc-mfl-noise",
                        "mfl_code": "MFL-NOISE-001",
                        "module_fault": f"文字中出现 {business_id}",
                    }
                ]
                history_rows = [
                    {
                        "doc_id": "doc-history-noise",
                        "case_id": "HIS-NOISE-001",
                        "fault_symptom": f"文字中出现 {business_id}",
                    }
                ]

                if stored_tool == "function_locator":
                    component_rows.append(
                        {
                            "doc_id": "doc-function-exact",
                            "component_id": business_id,
                            "component_type": "module",
                            "name": "精确身份记录",
                            "description": "由组件身份列精确命中",
                        }
                    )
                elif stored_tool == "fmeca_query":
                    fmeca_rows.append(
                        {
                            "doc_id": "doc-fmeca-exact",
                            "code": business_id,
                            "fault_mode": "精确身份记录",
                            "fault_cause": "由 FMECA 身份列精确命中",
                        }
                    )
                elif stored_tool == "mfl_query":
                    mfl_rows.append(
                        {
                            "doc_id": "doc-mfl-exact",
                            "mfl_code": business_id,
                            "module_fault": "精确身份记录",
                            "maintenance_action": "由 MFL 身份列精确命中",
                        }
                    )
                else:
                    history_rows.append(
                        {
                            "doc_id": "doc-history-exact",
                            "case_id": business_id,
                            "fault_symptom": "精确身份记录",
                            "root_cause": "由历史案例身份列精确命中",
                        }
                    )

                state = {
                    "scope_route": "knowledge_base",
                    "intent": "entity_profile",
                    "query_targets": policy["query_targets"],
                    "query_frame": {
                        "coverage_scope": "focused_detail",
                        "required_answer_fields": ["record_identity"],
                        "entities": [business_id],
                    },
                    "query": f"{business_id} 是什么",
                    "rewritten_query": f"{business_id} 是什么",
                    "entity_keywords": [business_id],
                    "tool_queries": {"function_locator": [business_id]},
                    "allowed_tools": policy["allowed_tools"],
                    "sql_limit": 8,
                    "retrieval_top_k": 8,
                }
                with (
                    patch.object(
                        diagnosis_module.icp_repository,
                        "search_functions",
                        return_value=function_rows,
                    ) as function_query,
                    patch.object(
                        diagnosis_module.icp_repository,
                        "search_components",
                        return_value=component_rows,
                    ) as component_query,
                    patch.object(
                        diagnosis_module,
                        "_query_hmc",
                        return_value=hmc_rows,
                    ) as hmc_query,
                    patch.object(
                        diagnosis_module,
                        "_query_fmeca",
                        return_value=fmeca_rows,
                    ) as fmeca_query,
                    patch.object(
                        diagnosis_module,
                        "_query_mfl",
                        return_value=mfl_rows,
                    ) as mfl_query,
                    patch.object(
                        diagnosis_module.icp_repository,
                        "search_history",
                        return_value=history_rows,
                    ) as history_query,
                    patch.object(
                        diagnosis_module,
                        "_enrich_function_sources",
                        side_effect=lambda rows, errors=None: [dict(row) for row in rows],
                    ),
                ):
                    result = _run(diagnosis_module.icp_diagnosis_node(state))

                for query_mock in (
                    function_query,
                    component_query,
                    hmc_query,
                    fmeca_query,
                    mfl_query,
                    history_query,
                ):
                    query_mock.assert_called_once()

                self.assertEqual(
                    probe_tools | {"evidence_synthesis"},
                    {item["tool"] for item in result["tool_results"]},
                )
                self.assertEqual(1, len(result["evidence_registry"]))
                surviving = result["evidence_registry"][0]
                self.assertEqual(business_id, surviving["record_id"])
                self.assertEqual(doc_type, surviving["doc_type"])
                self.assertEqual(claim_type, surviving["claim_type"])
                self.assertTrue(
                    all(
                        item.get("record_id") == business_id
                        for item in result["evidence_registry"]
                    )
                )

    def test_complete_failure_mode_collection_survives_architecture_misclassification(self):
        """A closed collection is a data contract, not an architecture-only answer.

        The target model may retain ``architecture_query`` while resolving the
        collection subject through Function Locator.  That must not trigger the
        function-only early return before the authoritative FMECA collection is
        queried and registered as evidence.
        """

        state = {
            "scope_route": "knowledge_base",
            "intent": "architecture_query",
            "query_targets": [{"intent": "architecture_query"}],
            "query": "电源与配电子系统只有 A 路间歇停机这一种吗？",
            "rewritten_query": "电源与配电子系统的全部故障模式",
            "allowed_tools": [
                "function_locator",
                "fmeca_query",
                "evidence_synthesis",
            ],
            "query_frame": {
                "coverage_request": {
                    "scope": "complete_collection", "subject": "电源与配电子系统",
                    "member_kind": "failure_mode", "reference_mode": "current_query",
                    "basis": "只有 A 路间歇停机这一种吗",
                },
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
                "entities": ["电源与配电子系统"],
            },
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }
        subsystem = {
            "doc_id": "doc-function",
            "doc_name": "系统架构与功能说明.docx",
            "component_id": "SUB-PWR",
            "component_type": "subsystem",
            "name": "电源与配电子系统",
            "abbreviation": "PWR",
            "description": "负责系统供电与配电。",
            "source_ref": "系统架构与功能说明.docx#SUB-PWR",
        }
        collection_rows = [
            {
                "doc_id": "doc-fmeca",
                "doc_name": "FMECA分析报告.docx",
                "code": f"FM-PWR-{index:03d}",
                "fault_mode": mode,
                "fault_cause": cause,
                "final_impact": impact,
                "source_ref": f"FMECA分析报告.docx#row={index}",
                "collection_closed": True,
                "collection_parent_id": "SUB-PWR",
                "collection_parent_name": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
                "collection_member_count": 3,
                "collection_role": "member",
                "collection_member_id": f"FM-PWR-{index:03d}",
                "collection_member_name": mode,
                "collection_member_description": cause,
            }
            for index, (mode, cause, impact) in enumerate(
                (
                    ("输入过压钳位失效", "钳位器件失效", "输入调理功能中断"),
                    ("输入滤波性能退化", "滤波电容老化", "供电质量下降"),
                    ("A路间歇停机", "A路连接器接触不良", "主路供电间歇中断"),
                ),
                start=1,
            )
        ]

        with (
            patch.object(
                diagnosis_module.icp_repository,
                "search_functions",
                return_value=[],
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "search_components",
                return_value=[subsystem],
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "expand_function_context",
                return_value=[],
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "expand_architecture_inventory",
                return_value=[],
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "expand_fmeca_fault_mode_collection",
                return_value=collection_rows,
            ) as expand_collection,
            patch.object(
                diagnosis_module.icp_repository,
                "list_collection_scope_candidates",
                return_value=[{
                    "collection_entity_kind": "component",
                    "doc_id": "doc-function",
                    "collection_entity_id": "SUB-PWR",
                    "collection_entity_name": "电源与配电子系统",
                    "collection_entity_abbreviation": "PWR",
                }],
            ),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: [dict(row) for row in rows],
            ),
            patch.object(
                diagnosis_module,
                "_query_fmeca",
                side_effect=AssertionError("closed collection must not use broad FMECA lookup"),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        expand_collection.assert_called_once_with(
            "doc-function",
            component_ids=["SUB-PWR"],
            function_ids=[],
            limit=25,
        )
        resolution = result["collection_resolution"]
        self.assertTrue(resolution["requested"])
        self.assertTrue(resolution["closed"])
        self.assertEqual(3, resolution["member_count"])
        self.assertEqual("SUB-PWR", resolution["parent_id"])

        fmeca_call = next(
            item for item in result["tool_calls"] if item["tool"] == "fmeca_query"
        )
        self.assertEqual(3, fmeca_call["count"])
        self.assertTrue(fmeca_call["collection_resolution"]["closed"])
        fmeca_result = next(
            item for item in result["tool_results"] if item["tool"] == "fmeca_query"
        )
        self.assertEqual(3, fmeca_result["data"]["count"])

        fmeca_evidence = [
            item
            for item in result["evidence_registry"]
            if item.get("doc_type") == "fmeca"
        ]
        self.assertEqual(
            {"FM-PWR-001", "FM-PWR-002", "FM-PWR-003"},
            {item["record_id"] for item in fmeca_evidence},
        )
        self.assertTrue(
            all(
                item["metadata"]["answer_fields"]["collection_closed"] is True
                for item in fmeca_evidence
            )
        )

    def test_selected_fmeca_anchor_is_re_read_by_exact_business_id(self):
        state = {
            "scope_route": "knowledge_base",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "query": "这种异常什么级别",
            "rewritten_query": "ICU 电源波动关联状态异常的严酷度",
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "selected_conversation_anchors": [{
                "anchor_id": "ev-fm-2",
                "doc_type": "fmeca",
                "doc_id": "doc-fmeca",
                "record_id": "FM-SYN-002",
                "answer_fields": {"record_code": "FM-SYN-002"},
            }],
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }
        row = {
            "doc_id": "doc-fmeca",
            "code": "FM-SYN-002",
            "fault_mode": "输入滤波性能退化",
            "severity": "III",
        }
        with (
            patch.object(
                diagnosis_module.icp_repository,
                "get_records_by_business_selectors",
                return_value={"rows": [row], "reads": []},
            ) as exact_query,
            patch.object(diagnosis_module, "_query_fmeca") as broad_query,
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: list(rows),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        exact_query.assert_called_once()
        self.assertEqual(("fmeca", [{"doc_id": "doc-fmeca", "record_id": "FM-SYN-002"}]), exact_query.call_args.args)
        broad_query.assert_not_called()
        fmeca = [item for item in result["evidence_registry"] if item["doc_type"] == "fmeca"]
        self.assertEqual("III", fmeca[0]["metadata"]["answer_fields"]["severity"])

    def test_selected_hmc_anchor_is_re_read_instead_of_keyword_search(self):
        state = {
            "scope_route": "knowledge_base",
            "intent": "maintenance_query",
            "query_targets": [{"intent": "maintenance_query"}],
            "query": "那应该怎么处理",
            "rewritten_query": "上一轮 HMC 对应记录的处理方法",
            "allowed_tools": ["hmc_query", "evidence_synthesis"],
            "selected_conversation_anchors": [{
                "anchor_id": "ev-hmc-1",
                "doc_type": "hmc",
                "doc_id": "doc-hmc",
                "record_id": "91810001**0001",
                "answer_fields": {"hmc_code": "91810001**0001"},
            }],
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }
        row = {
            "doc_id": "doc-hmc",
            "hmc_code": "91810001**0001",
            "lcn": "91810001",
            "hamc": "**",
            "hsc": "0001",
            "fault_desc": "ICU 输入过压钳位失效",
            "repair_method": "检查输入采样与钳位器件，确认异常后更换输入调理模块。",
        }
        selector = {
            "kind": "components", "doc_id": "doc-hmc",
            "lcn": "91810001", "hamc": "**", "hsc": "0001",
        }
        row.update(match_type="exact_conversation_reference", hmc_anchor_selector=selector)
        with (
            patch.object(diagnosis_module.icp_repository, "get_records_by_business_selectors", return_value={"rows": [row], "reads": []}) as exact_query,
            patch.object(diagnosis_module, "_query_hmc") as keyword_query,
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual(("hmc", [selector]), exact_query.call_args.args)
        keyword_query.assert_not_called()
        hmc = [item for item in result["evidence_registry"] if item["doc_type"] == "hmc"]
        self.assertEqual("exact_conversation_reference", hmc[0]["match_type"])


class FinalOutputGateTests(unittest.TestCase):

    def test_public_citation_compaction_cannot_promote_blocked_evidence(self):
        approved = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-ok",
            record_id="FM-OK",
            title="已批准记录",
            content="资料记载局部影响。",
        )
        blocked = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L4_UNRESOLVED",
            claim_type="search_hit",
            doc_type="fmeca",
            doc_id="doc-clue",
            record_id="candidate",
            title="候选线索",
            content="未经核验的候选结论。",
        )
        state = {
            "query": "会有什么影响",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "evidence_registry": [approved, blocked],
            "context_evidence_ids": [approved["evidence_id"], blocked["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [approved["evidence_id"]],
                "blocked_evidence_ids": [blocked["evidence_id"]],
                "context_evidence_ids": [approved["evidence_id"], blocked["evidence_id"]],
            },
        }
        prompt_citations = answer_module._extract_citations(state)
        # Rendering rejects the invalid publication instead of silently
        # deleting prose and reporting a successful partial interpretation.
        with self.assertRaises(answer_module.AnswerGenerationError):
            answer_module._compact_answer_citations(
                "- 资料记载局部影响。 [^1]\n- 已确定发生未经核验的候选故障。 [^2]",
                prompt_citations,
                state,
            )


class CollectionAnswerSafetyTests(unittest.TestCase):
    def test_closed_collection_count_followup_is_left_to_the_answer_model(self):
        members = []
        for member_id, name, abbreviation, description in (
            (
                "SUB-PWR",
                "电源与配电子系统",
                "PWR",
                "完成外部电源调理和短时保持供电",
            ),
            (
                "SUB-DSP",
                "显示处理与输出子系统",
                "DSP",
                "完成画面合成与显示输出",
            ),
        ):
            members.append(
                make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type="component_fact",
                    doc_type="function_locator",
                    doc_id="doc-arch",
                    record_id=member_id,
                    title=name,
                    content=description,
                    metadata={
                        "answer_fields": {
                            "collection_closed": True,
                            "collection_parent_id": "SYS-1",
                            "collection_parent_name": "综合系统",
                            "collection_member_kind": "subsystem",
                            "collection_member_count": 2,
                            "collection_role": "member",
                            "collection_member_id": member_id,
                            "collection_member_name": name,
                            "component_name": name,
                            "abbreviation": abbreviation,
                            "description": description,
                        }
                    },
                )
            )
        evidence_ids = [item["evidence_id"] for item in members]
        state = {
            "query": "所以总数是几个？有没有把供电和显示相关部分漏掉？",
            "query_frame": {
                "original_query": "所以总数是几个？有没有把供电和显示相关部分漏掉？",
                "coverage_scope": "complete_collection",
                "collection_subject": "SYS-1",
                "collection_member_kind": "subsystem",
                "required_answer_fields": ["closed_collection_members"],
                "prior_delivered_fields": ["closed_collection_members"],
                "question_semantics": "verify",
                "response_scope": "incremental",
                "response_focus": ["count", "completeness_check"],
            },
            "collection_resolution": {
                "closed": True,
                "parent_id": "SYS-1",
                "member_kind": "subsystem",
                "member_count": 2,
            },
            "evidence_registry": members,
            "context_evidence_ids": evidence_ids,
            "safety_decision": {
                "approved_evidence_ids": evidence_ids,
                "context_evidence_ids": evidence_ids,
            },
        }

        prompt = answer_module._build_answer_prompt(state, answer_module._format_context(state))
        self.assertIn(state["query"], prompt)
        self.assertFalse(hasattr(answer_module, "_render_closed_collection_identity_only"))

    def test_healthy_collection_is_not_replaced_by_identity_renderer(self):
        inventory = [
            _failure_mode_collection_member(index, 2)
            for index in range(1, 3)
        ]
        for index, item in enumerate(inventory, start=1):
            fields = item["metadata"]["answer_fields"]
            fields.update(
                {
                    "local_impact": f"不应展示的局部影响{index}",
                    "final_impact": f"不应展示的最终影响{index}",
                    "severity": f"不应展示的严酷度{index}",
                    "detection_method": f"不应展示的检测方法{index}",
                    "design_improvement": f"不应展示的设计改进{index}",
                    "compensation_action": f"不应展示的使用补偿{index}",
                    "maintenance_action": f"不应展示的维修措施{index}",
                }
            )
        evidence_ids = [item["evidence_id"] for item in inventory]
        state = {
            "query": "这个子系统一共有哪几种故障模式？",
            "intent": "entity_profile",
            "query_targets": [
                {
                    "intent": "entity_profile",
                    "required_fields": [
                        "closed_collection_members",
                        "failure_mode",
                    ],
                }
            ],
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
                "required_answer_fields": [
                    "closed_collection_members",
                    "failure_mode",
                ],
            },
            "collection_resolution": {
                "closed": True,
                "parent_id": "SUB-PWR",
                "parent_name": "电源与配电子系统",
                "member_kind": "failure_mode",
                "member_count": 2,
            },
            "evidence_registry": inventory,
            "context_evidence_ids": evidence_ids,
            "safety_decision": {
                "approved_evidence_ids": evidence_ids,
                "context_evidence_ids": evidence_ids,
            },
        }

        # 集合字段是可用证据，不另建一个按字段决定正文的渲染入口。
        prompt = answer_module._build_answer_prompt(state, answer_module._format_context(state))
        self.assertIn(state["query"], prompt)
        self.assertFalse(hasattr(answer_module, "_closed_collection_identity_only"))
        self.assertFalse(hasattr(answer_module, "_render_closed_collection_identity_only"))




    def test_closed_failure_mode_members_use_collection_identity_not_component_fields(self):
        evidence = [
            _failure_mode_collection_member(index, 3)
            for index in range(1, 4)
        ]
        manifest = [
            {
                "citation_id": index,
                "record_id": item["record_id"],
                "answer_fields": item["metadata"]["answer_fields"],
                "supports": ["closed_collection_members"],
            }
            for index, item in enumerate(evidence, start=1)
        ]
        self.assertTrue(all(
            "component_id" not in item["answer_fields"]
            and "component_name" not in item["answer_fields"]
            for item in manifest
        ))
        title_only = "\n".join(
            f"- {item['answer_fields']['collection_member_name']}"
            f"（{item['answer_fields']['collection_member_id']}）。[^{item['citation_id']}]"
            for item in manifest
        )
        # 故障模式集合的内容完整性按字段核对：纯标题清单不算“已列全”，
        # 否则权威富文本清单会被静默压制。
        self.assertFalse(
            _source_answer_target_presence._closed_collection_members_present(title_only, manifest)
        )

        answered = "\n".join(
            f"- {item['answer_fields']['collection_member_name']}"
            f"（{item['answer_fields']['collection_member_id']}）"
            f"原文原因：{item['answer_fields']['fault_cause']}。"
            f"[^{item['citation_id']}]"
            for item in manifest
        )
        self.assertTrue(
            _source_answer_target_presence._closed_collection_members_present(answered, manifest)
        )

    def test_evidence_window_keeps_all_fifteen_members_of_closed_collection(self):
        inventory = [
            _failure_mode_collection_member(index, 15)
            for index in range(1, 16)
        ]
        for item in inventory:
            # These records must win as a closed whole, not one by one on score.
            item["answer_rerank_score"] = -1.0
        distractors = []
        for index in range(20):
            item = make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-other-fmeca",
                record_id=f"FM-OTHER-{index:03d}",
                title=f"无关故障模式{index}",
                content="仅用于验证证据窗口排序。",
                metadata={
                    "answer_fields": {
                        "record_code": f"FM-OTHER-{index:03d}",
                        "failure_mode": f"无关故障模式{index}",
                    }
                },
            )
            item["answer_rerank_score"] = 1.0
            distractors.append(item)
        evidence = [*distractors, *inventory]
        evidence_ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "电源与配电子系统是否只有这些故障模式？",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
            },
            "evidence_registry": evidence,
            "context_evidence_ids": evidence_ids,
            "safety_decision": {
                "approved_evidence_ids": evidence_ids,
                "context_evidence_ids": evidence_ids,
            },
        }

        # Twelve is the normal soft model-context budget.  A PostgreSQL-closed
        # 15-member collection is indivisible and must expand that budget up
        # to the global hard cap instead of silently dropping three members.
        selected = _source_answer_evidence_window._select_evidence_window(state, evidence, limit=12)

        selected_ids = {item["record_id"] for item in selected}
        inventory_ids = {item["record_id"] for item in inventory}
        self.assertTrue(inventory_ids <= selected_ids)
        self.assertGreaterEqual(len(selected), len(inventory))

    def test_evidence_window_selects_requested_fmeca_group_not_first_architecture_group(self):
        architecture = []
        for index in range(1, 7):
            architecture.append(make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="component_fact",
                doc_type="function_locator",
                doc_id="doc-arch",
                record_id=f"SUB-{index}",
                title=f"子系统{index}",
                content=f"子系统{index}",
                metadata={"answer_fields": {
                    "collection_closed": True,
                    "collection_parent_id": "SYS-9000",
                    "collection_parent_name": "综合系统",
                    "collection_member_kind": "subsystem",
                    "collection_member_count": 6,
                    "collection_role": "member",
                    "collection_member_id": f"SUB-{index}",
                    "collection_member_name": f"子系统{index}",
                }},
            ))
        inventory = [
            _failure_mode_collection_member(index, 15)
            for index in range(1, 16)
        ]
        distractors = []
        for index in range(20):
            item = make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-other",
                record_id=f"OTHER-{index}",
                title=f"高分干扰{index}",
                content=f"高分干扰{index}",
            )
            item["answer_rerank_score"] = 1.0
            distractors.append(item)
        evidence = [*architecture, *inventory, *distractors]
        state = {
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电子系统",
                "collection_member_kind": "failure_mode",
            },
            "collection_resolution": {"parent_id": "SUB-PWR"},
            "safety_decision": {
                "approved_evidence_ids": [item["evidence_id"] for item in evidence],
            },
        }

        selected = _source_answer_evidence_window._select_evidence_window(state, evidence, limit=12)

        selected_ids = {item["record_id"] for item in selected}
        self.assertTrue(
            {item["record_id"] for item in inventory} <= selected_ids
        )


    def test_answer_context_restores_approved_closed_members_from_focused_gate(self):
        """A stale Top-K gate window must not truncate an approved closed set."""

        inventory = [
            _failure_mode_collection_member(index, 15)
            for index in range(1, 16)
        ]
        state = {
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电模块的故障模式",
                "collection_member_kind": "failure_mode",
            },
            "collection_resolution": {"parent_id": "SUB-PWR"},
            "evidence_registry": inventory,
            # Reproduce the live failure shape: all rows were approved, while
            # an earlier focus pass exposed only the first three to Answer.
            "context_evidence_ids": [
                item["evidence_id"] for item in inventory[:3]
            ],
            "safety_decision": {
                "approved_evidence_ids": [
                    item["evidence_id"] for item in inventory
                ],
                "context_evidence_ids": [
                    item["evidence_id"] for item in inventory[:3]
                ],
            },
        }

        context = answer_module._context_evidence(state)

        self.assertEqual(15, len(context))
        self.assertEqual(
            {item["record_id"] for item in inventory},
            {item["record_id"] for item in context},
        )











if __name__ == "__main__":
    unittest.main()
