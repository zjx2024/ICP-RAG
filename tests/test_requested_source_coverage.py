import unittest

from backend.agent.answer_contract import focused_answer_contract
from backend.agent.rbd_contract import bind_rbd_models, selected_rbd_anchor_selectors
from backend.agent.nodes.answer_request import _NO_SUPPORTED_FACTS_MESSAGE
from backend.agent.nodes.answer import _extract_citations
from backend.agent.nodes.answer_domain_scope import _filter_answer_domain_evidence
from backend.agent.nodes.answer import _answer_task_instruction
from backend.agent.tools.icp_diagnosis import (
    _filter_diagrams_by_explicit_object_ids,
    _filter_requested_rbd_rows,
)


def _authorize_selected_rbd(state, rbd):
    revision_id = f"revision-{rbd['evidence_id']}"
    rbd.update(
        {
            "claim_type": "rbd_structure",
            "source_kind": "postgres",
            "authority": "validated_relation",
            "doc_id": "doc-rbd",
            "review_status": "auto_validated",
        }
    )
    metadata = dict(rbd.get("metadata") or {})
    metadata.update(
        {
            "revision_id": revision_id,
            "diagram_key": rbd["record_id"],
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
        }
    )
    rbd["metadata"] = metadata
    anchor = {
        "doc_type": "rbd",
        "doc_id": "doc-rbd",
        "record_id": rbd["record_id"],
        "revision_id": revision_id,
        "diagram_key": rbd["record_id"],
    }
    _, binding = bind_rbd_models(
        [rbd],
        required=True,
        raw_user_query=state.get("query"),
        anchor_selectors=selected_rbd_anchor_selectors([anchor]),
    )
    state["selected_conversation_anchors"] = [anchor]
    state["rbd_query_binding"] = binding


class RequestedSourceCoverageTests(unittest.TestCase):
    def test_source_question_survives_an_unavailable_field_without_authorizing_other_sources(self):
        from backend.agent.nodes.plan import _parse_requested_sources
        from backend.agent.tool_policy import resolve_tool_policy

        for source in ("rbd", "function_locator", "hmc"):
            for wording in ("请向资料甲核实检测依据", "Does source A provide a verification basis?", "资料甲能证明到哪一步？"):
                with self.subTest(source=source, wording=wording):
                    request = [{"doc_type": source, "basis": wording, "question_role": "capability", "requested_fields": ["detection_method"]}]
                    self.assertEqual(request, _parse_requested_sources(request, raw_user_query=wording))
                    decision = resolve_tool_policy(intent="entity_profile", raw_user_query=wording, requested_sources=request)
                    self.assertIn("detection_method", decision["source_coverage_fields"])
                    self.assertEqual([source], decision["authorized_doc_types"])
                    self.assertNotIn("fmeca_query", decision["authorized_tools"])

    def test_explicit_source_applies_to_maintenance_fields_too(self):
        """MFL's usual field ownership cannot override the user's chosen source."""
        from backend.agent.tool_policy import resolve_tool_policy

        wording = "只根据 FMECA 中的检测和处置措施回答，不要检索 MFL。"
        request = [{
            "doc_type": "fmeca",
            "basis": "只根据 FMECA 中的检测和处置措施回答",
            "scope": "exclusive",
            "scope_basis": "不要检索 MFL",
            "question_role": "evidence",
            "requested_fields": ["detection_method", "maintenance_action"],
        }]
        decision = resolve_tool_policy(
            intent="maintenance_query",
            raw_user_query=wording,
            requested_sources=request,
            required_answer_fields=["detection_method", "maintenance_action"],
        )
        self.assertNotIn("mfl_query", decision["authorized_tools"])
        self.assertNotIn("mfl", decision["authorized_doc_types"])
        self.assertIn("fmeca", decision["authorized_doc_types"])

    def test_explicit_mfl_source_keeps_contract_scoping(self):
        from backend.agent.tool_policy import resolve_tool_policy

        wording = "维修清单对这类失效建议怎么处理？"
        request = [{
            "doc_type": "mfl",
            "basis": "维修清单对这类失效建议怎么处理",
            "question_role": "evidence",
            "requested_fields": ["maintenance_action"],
        }]
        decision = resolve_tool_policy(
            intent="maintenance_query",
            raw_user_query=wording,
            requested_sources=request,
            required_answer_fields=["maintenance_action"],
        )
        self.assertIn("mfl_query", decision["authorized_tools"])
        self.assertIn("mfl", decision["authorized_doc_types"])

    def test_non_maintenance_source_scoping_is_unchanged(self):
        """不含 MFL 合同字段的来源请求维持原有窄边界。"""
        from backend.agent.tool_policy import resolve_tool_policy

        wording = "请向资料甲核实检测依据"
        request = [{
            "doc_type": "fmeca",
            "basis": "请向资料甲核实检测依据",
            "question_role": "capability",
            "requested_fields": ["detection_method"],
        }]
        decision = resolve_tool_policy(
            intent="entity_profile",
            raw_user_query=wording,
            requested_sources=request,
            required_answer_fields=["detection_method"],
        )
        self.assertEqual(["fmeca"], decision["authorized_doc_types"])
        self.assertNotIn("mfl_query", decision["authorized_tools"])


    def test_source_boundary_window_uses_selected_rbd_not_locator_neighbors(self):
        rbd = {
            "evidence_id": "rbd-selected",
            "doc_type": "rbd",
            "record_id": "RBD:demo-power-path",
            "title": "演示供电路径",
        }
        locator = {
            "evidence_id": "function-neighbor",
            "doc_type": "function_locator",
            "record_id": "FUNC-OTHER",
            "metadata": {"answer_fields": {"function_id": "FUNC-OTHER"}},
        }
        state = {
            "requested_source_tools": ["rbd_query"],
            "selected_conversation_anchors": [
                {"doc_type": "rbd", "record_id": "RBD:demo-power-path"}
            ],
            "safety_decision": {
                "approved_evidence_ids": ["rbd-selected", "function-neighbor"]
            },
            "query_frame": {
                "requested_source_tools": ["rbd_query"],
                "response_focus": ["source_capability", "source_limitation"],
            },
        }
        _authorize_selected_rbd(state, rbd)

        filtered = _filter_answer_domain_evidence(state, [rbd, locator])

        self.assertEqual(["rbd-selected"], [item["evidence_id"] for item in filtered])

    def test_source_boundary_does_not_drop_an_independent_second_goal(self):
        rbd = {
            "evidence_id": "rbd-selected",
            "doc_type": "rbd",
            "record_id": "RBD:demo-path",
            "authority": "validated_relation",
        }
        history = {
            "evidence_id": "history-selected",
            "doc_type": "fault_history",
            "record_id": "HIS-DEMO",
            "authority": "authoritative_record",
            "match_type": "exact_conversation_reference",
            "metadata": {"answer_fields": {"verification_result": "连续复测通过"}},
        }
        state = {
            "requested_source_tools": ["rbd_query"],
            "query_targets": [
                {"intent": "reliability_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "requested_source_tools": ["rbd_query"],
                "source_obligations": [{
                    "tool": "rbd_query",
                    "doc_type": "rbd",
                    "fields": ["rbd_success_criterion", "rbd_path_structure"],
                }],
                "response_focus": ["source_capability", "recovery_verification"],
                "required_answer_fields": [
                    "rbd_success_criterion",
                    "rbd_path_structure",
                    "history_result",
                ],
            },
        }
        _authorize_selected_rbd(state, rbd)

        filtered = _filter_answer_domain_evidence(state, [rbd, history])

        self.assertEqual(
            {"rbd-selected", "history-selected"},
            {item["evidence_id"] for item in filtered},
        )







    def test_explicit_source_boundary_keeps_only_requested_evidence_dimensions(self):
        instruction = _answer_task_instruction(
            {
                "query_frame": {
                    "source_obligations": [{
                        "tool": "rbd_query",
                        "doc_type": "rbd",
                        "fields": ["rbd_single_failure_boundary"],
                    }]
                }
            }
        )

        self.assertIn('"fields":["rbd_single_failure_boundary"]', instruction)
        self.assertNotIn("rbd_success_criterion", instruction)
        self.assertNotIn("rbd_path_structure", instruction)



    def test_explicit_rbd_coverage_keeps_cause_answer_narrative(self):
        self.assertFalse(
            focused_answer_contract(
                {
                    "intent": "cause_analysis",
                    "requested_source_tools": ["rbd_query"],
                    "query_frame": {
                        "coverage_scope": "focused_detail",
                        "required_answer_fields": [
                            "fault_cause",
                            "rbd_path_structure",
                        ],
                        "diagnostic_closure": {
                            "required": True,
                            "required_dimensions": [
                                "fault_cause",
                                "rbd_path_structure",
                            ],
                            "basis": "当前原因与结构边界",
                        },
                    },
                }
            )
        )

    def test_structured_subject_without_rbd_authority_contract_drops_graphs(self):
        function_item = {
            "evidence_id": "function-msm",
            "doc_type": "function_locator",
            "metadata": {
                "answer_fields": {"component_id": "MOD-GPP-MSM"}
            },
        }
        matching_rbd = {
            "evidence_id": "rbd-msm",
            "doc_type": "rbd",
            "content": (
                '{"vertices":[{"business_id":"MOD-GPP-MSM",'
                '"properties":{"module_id":"MOD-GPP-MSM"}}]}'
            ),
        }
        unrelated_rbd = {
            "evidence_id": "rbd-pdu",
            "doc_type": "rbd",
            "content": (
                '{"vertices":[{"business_id":"MOD-PWR-PDU",'
                '"properties":{"module_id":"MOD-PWR-PDU"}}]}'
            ),
        }

        filtered = _filter_answer_domain_evidence(
            {
                "intent": "cause_analysis",
                "requested_source_tools": ["rbd_query"],
                "query_frame": {"requested_source_tools": ["rbd_query"]},
                "structured_entity_binding": {
                    "bound": True,
                    "business_ids": ["MOD-GPP-MSM"],
                    "terms": ["MOD-GPP-MSM", "MSM"],
                },
            },
            [function_item, matching_rbd, unrelated_rbd],
        )

        self.assertEqual(
            ["function-msm"],
            [item["evidence_id"] for item in filtered],
        )

    def test_explicit_rbd_tool_filter_rejects_a_broad_diagram_hit(self):
        core = {
            "diagram_key": "RBD-SYN-002",
            "graph": {
                "vertices": [
                    {"properties": {"module_id": "MOD-GPP-MSM"}},
                ]
            },
        }
        power = {
            "diagram_key": "RBD-SYN-001",
            "graph": {
                "vertices": [
                    {"properties": {"module_id": "MOD-PWR-PDU"}},
                ]
            },
        }

        filtered = _filter_diagrams_by_explicit_object_ids(
            [core, power], ["MOD-GPP-MSM"]
        )

        self.assertEqual(["RBD-SYN-002"], [row["diagram_key"] for row in filtered])

    def test_natural_rbd_subject_cannot_select_a_model(self):
        rows = [
            {
                "diagram_type": "rbd",
                "revision_id": "rev-power",
                "diagram_key": "RBD:电源保障链可靠性模型",
                "title": "电源保障链可靠性模型",
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
                "graph": {
                    "vertices": [
                        {"business_id": "MOD-PWR-ICU", "name": "输入调理模块"}
                    ]
                },
            },
            {
                "diagram_type": "rbd",
                "revision_id": "rev-display",
                "diagram_key": "RBD:显示输出功能可靠性模型",
                "title": "显示输出功能可靠性模型",
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
                "graph": {"vertices": [{"name": "显示处理模块"}]},
            },
        ]

        filtered = _filter_requested_rbd_rows(
            rows,
            requested=True,
            literal_subject="显示处理模块",
        )
        self.assertEqual([], filtered)

    def test_selected_rbd_citation_identity_outvotes_meta_question_text(self):
        rows = [
            {
                "diagram_type": "rbd",
                "revision_id": "rev-task",
                "diagram_key": "RBD:任务计算保障链",
                "title": "任务计算保障链",
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
            },
            {
                "diagram_type": "rbd",
                "revision_id": "rev-display",
                "diagram_key": "RBD:显示输出保障链",
                "title": "显示输出保障链",
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
            },
        ]

        filtered = _filter_requested_rbd_rows(
            rows,
            requested=True,
            anchor_identities=["RBD:任务计算保障链"],
            literal_subject="可靠性框图的结论边界",
        )

        self.assertEqual(
            ["RBD:任务计算保障链"],
            [item["diagram_key"] for item in filtered],
        )

    def test_explicit_rbd_boundary_is_reusable_after_semantic_hydration(self):
        unrelated = {
            "diagram_key": "RBD:电源保障链可靠性模型",
            "title": "电源保障链可靠性模型",
            "graph": {"vertices": [{"name": "输入调理模块"}]},
            "match_type": "semantic_locator_hydration",
        }

        self.assertEqual(
            [],
            _filter_requested_rbd_rows(
                [unrelated],
                requested=True,
                literal_subject="输入端滤波器",
            ),
        )

    def test_explicit_rbd_business_id_miss_drops_all_other_rbd_evidence(self):
        function_item = {
            "evidence_id": "function-msm",
            "doc_type": "function_locator",
            "metadata": {
                "answer_fields": {"component_id": "MOD-GPP-MSM"}
            },
        }
        unrelated_rbd = {
            "evidence_id": "rbd-pdu",
            "doc_type": "rbd",
            "content": (
                '{"vertices":[{"business_id":"MOD-PWR-PDU",'
                '"properties":{"module_id":"MOD-PWR-PDU"}}]}'
            ),
        }

        filtered = _filter_answer_domain_evidence(
            {
                "requested_source_tools": ["rbd_query"],
                "query_frame": {"requested_source_tools": ["rbd_query"]},
                "structured_entity_binding": {
                    "bound": True,
                    "business_ids": ["MOD-GPP-MSM"],
                    "terms": ["MOD-GPP-MSM", "MSM"],
                },
            },
            [function_item, unrelated_rbd],
        )

        self.assertEqual(
            ["function-msm"],
            [item["evidence_id"] for item in filtered],
        )


if __name__ == "__main__":
    unittest.main()
