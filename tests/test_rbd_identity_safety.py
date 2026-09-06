import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.tools import icp_calculation as calculation
from backend.agent.tools.icp_diagnosis import (
    _bind_requested_rbd_rows,
    _filter_diagrams_by_explicit_object_ids,
    _filter_requested_rbd_rows,
    _selected_rbd_anchor_identities,
    _selected_rbd_anchor_selectors,
)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _graph(block_name="处理单元"):
    return {
        "vertices": [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {
                "id": "block",
                "label": "ReliabilityBlock",
                "properties": {"name": block_name, "reliability": 0.98},
            },
            {"id": "end", "label": "End", "properties": {"name": "END"}},
        ],
        "edges": [
            {"source": "start", "target": "block"},
            {"source": "block", "target": "end"},
        ],
    }


def _row(
    revision_id,
    *,
    diagram_key=None,
    diagram_id=None,
    doc_id="doc-rbd",
    title="共用可靠性模型",
    configuration="",
):
    row = {
        "diagram_type": "rbd",
        "revision_id": revision_id,
        "diagram_id": diagram_id or f"diagram-{revision_id}",
        "diagram_key": diagram_key or f"RBD-{revision_id}",
        "doc_id": doc_id,
        "title": title,
        "image_title": title,
        "validation_status": "auto_validated",
        "projection_status": "published",
        "is_active": True,
        "graph": _graph(revision_id),
    }
    if configuration:
        row["scope"] = {"configuration": configuration}
    return row


def _state(rows, **overrides):
    state = {
        "query": "计算 RBD 可靠度",
        "calculation_requests": ["rbd_reliability"],
        "allowed_tools": ["rbd_calculator"],
        "confirmed_graphs": list(rows),
        "evidence_registry": [],
    }
    state.update(overrides)
    state.setdefault(
        "calculation_authorizations",
        [{
            "operation": "rbd_reliability",
            "mode": "calculate",
            "target": "rbd",
            "basis": (
                "可靠度" if "可靠度" in state["query"] else state["query"]
            ),
        }],
    )
    return state


class RbdSubjectIdentitySafetyTests(unittest.TestCase):
    def test_multiple_unbound_models_fail_closed(self):
        rows = [_row("rev-a"), _row("rev-b")]

        filtered, binding = _bind_requested_rbd_rows(rows, requested=True)

        self.assertEqual([], filtered)
        self.assertTrue(binding["ambiguous"])
        self.assertEqual("ambiguous_unresolved", binding["basis"])

    def test_typed_natural_subject_cannot_select_a_formal_model(self):
        matching = _row("rev-a")
        matching["graph"] = _graph("电源支路甲")
        other = _row("rev-b")
        other["graph"] = _graph("显示输出支路")
        filtered, binding = _bind_requested_rbd_rows(
            [matching, other],
            requested=True,
            literal_subject="电源支路甲",
        )

        self.assertEqual([], filtered)
        self.assertFalse(binding["bound"])
        self.assertTrue(binding["ambiguous"])
        self.assertNotEqual("exact_typed_subject", binding["basis"])

    def test_free_text_business_code_does_not_pass_typed_id_boundary(self):
        row = {
            "canonical_graph": {
                "vertices": [
                    {
                        # Code-shaped local vertex IDs are not typed business
                        # IDs and must not be promoted across documents.
                        "id": "MOD-PWR-A",
                        "properties": {
                            "name": "处理单元",
                            "description": "参考 MOD-PWR-A",
                        },
                    }
                ]
            }
        }

        self.assertEqual(
            [],
            _filter_diagrams_by_explicit_object_ids([row], ["MOD-PWR-A"]),
        )

    def test_reused_anchor_key_is_scoped_to_cited_document(self):
        rows = [
            _row("rev-a", diagram_key="RBD-SHARED", doc_id="doc-a"),
            _row("rev-b", diagram_key="RBD-SHARED", doc_id="doc-b"),
        ]
        state = {
            "selected_conversation_anchors": [
                {
                    "doc_type": "rbd",
                    "doc_id": "doc-b",
                    "record_id": "RBD-SHARED",
                    "answer_fields": {"diagram_key": "RBD-SHARED"},
                }
            ]
        }

        filtered = _filter_requested_rbd_rows(
            rows,
            requested=True,
            anchor_identities=_selected_rbd_anchor_identities(state),
            anchor_selectors=_selected_rbd_anchor_selectors(state),
        )

        self.assertEqual(["rev-b"], [row["revision_id"] for row in filtered])

    def test_anchor_stable_id_preserves_separators(self):
        rows = [
            _row("rev-a", diagram_key="RBD-A"),
            _row("rev-b", diagram_key="RBDA"),
        ]
        state = {
            "selected_conversation_anchors": [
                {
                    "doc_type": "rbd",
                    "doc_id": "doc-rbd",
                    "record_id": "RBD-A",
                }
            ]
        }

        filtered = _filter_requested_rbd_rows(
            rows,
            requested=True,
            anchor_identities=_selected_rbd_anchor_identities(state),
            anchor_selectors=_selected_rbd_anchor_selectors(state),
        )

        self.assertEqual(["rev-a"], [row["revision_id"] for row in filtered])

    def test_reused_anchor_key_is_scoped_to_cited_configuration(self):
        rows = [
            _row(
                "rev-a",
                diagram_key="RBD-SHARED",
                doc_id="doc-rbd",
                configuration="CONFIG-A",
            ),
            _row(
                "rev-b",
                diagram_key="RBD-SHARED",
                doc_id="doc-rbd",
                configuration="CONFIG-B",
            ),
        ]
        state = {
            "selected_conversation_anchors": [
                {
                    "doc_type": "rbd",
                    "doc_id": "doc-rbd",
                    "record_id": "RBD-SHARED",
                    "scope": {"configuration": "CONFIG-B"},
                }
            ]
        }

        filtered = _filter_requested_rbd_rows(
            rows,
            requested=True,
            anchor_identities=_selected_rbd_anchor_identities(state),
            anchor_selectors=_selected_rbd_anchor_selectors(state),
        )

        self.assertEqual(["rev-b"], [row["revision_id"] for row in filtered])


class RbdCalculationIdentitySafetyTests(unittest.TestCase):
    def test_planner_recall_terms_cannot_mint_a_calculation_selector(self):
        selectors = calculation._requested_graph_selectors(
            {
                "query": "计算当前模型的可靠度",
                "query_frame": {
                    "entities": ["RBD-INVENTED"],
                    "normalized_terms": ["RBD-INVENTED"],
                },
                "normalized_terms": ["RBD-INVENTED"],
            },
            "rbd",
        )

        self.assertEqual([], selectors)

    def test_single_formal_graph_is_not_calculated_while_query_binding_is_open(self):
        state = _state(
            [_row("rev-a")],
            rbd_query_binding={
                "required": True,
                "bound": False,
                "ambiguous": False,
                "basis": "single_unbound_candidate",
                "record_ids": ["rev-a"],
                "candidate_record_ids": ["rev-a"],
            },
        )

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual([], result["calculation_results"])
        self.assertTrue(
            any(
                "未通过精确稳定 ID" in warning
                for warning in result["tool_results"][0]["diagnostics"]["warnings"]
            )
        )

    def test_planner_invented_selector_cannot_pick_one_of_two_models(self):
        rows = [
            _row("rev-a", diagram_key="RBD-INVENTED", doc_id="doc-a"),
            _row("rev-b", diagram_key="RBD-REAL", doc_id="doc-b"),
        ]
        state = _state(
            rows,
            query="计算当前模型的可靠度",
            query_frame={"normalized_terms": ["RBD-INVENTED"]},
            normalized_terms=["RBD-INVENTED"],
            rbd_query_binding={
                "required": True,
                "bound": False,
                "ambiguous": True,
                "basis": "ambiguous_unresolved",
                "record_ids": [],
                "candidate_record_ids": ["RBD-INVENTED", "RBD-REAL"],
            },
        )

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual([], result["calculation_results"])

    def test_graph_identity_excludes_human_readable_titles(self):
        values = calculation._graph_identity_values(
            {
                "revision_id": "rev-1",
                "diagram_key": "RBD-ONE",
                "title": "共用标题",
                "image_title": "另一标题",
                "name": "模型名",
            }
        )

        self.assertEqual({"rev-1", "rbd-one"}, values)

    def test_anchor_title_alone_is_not_a_requested_graph_key(self):
        keys = calculation._requested_graph_keys(
            {
                "query": "按刚才的图计算",
                "selected_conversation_anchors": [
                    {"doc_type": "rbd", "title": "共用可靠性模型"}
                ],
            },
            "rbd",
        )

        self.assertEqual(set(), keys)

    def test_same_title_cannot_bind_calculation_parent(self):
        parent = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-b",
            record_id="rev-b",
            title="共用标题",
            review_status="auto_validated",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )

        self.assertEqual(
            [],
            calculation._calculation_parent_evidence_ids(
                _row("rev-a", doc_id="doc-a", title="共用标题"),
                "rbd",
                [parent],
            ),
        )

    def test_unbound_single_handoff_is_not_authoritative(self):
        selected = _row("rev-a", doc_id="doc-a", title="共用标题")
        sibling = _row("rev-b", doc_id="doc-b", title="共用标题")

        with patch.object(calculation, "_query_graphs", return_value=[sibling]):
            result = _run(calculation.icp_calculation_node(_state([selected])))

        self.assertEqual([], result["calculation_results"])

    def test_two_same_title_handoffs_fail_closed_without_stable_selector(self):
        rows = [
            _row("rev-a", doc_id="doc-a", title="共用标题"),
            _row("rev-b", doc_id="doc-b", title="共用标题"),
        ]

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(_state(rows)))

        self.assertEqual([], result["calculation_results"])
        self.assertTrue(
            any(
                "未通过精确稳定 ID" in warning
                for warning in result["tool_results"][0]["diagnostics"]["warnings"]
            )
        )

    def test_distinct_current_handoffs_remain_available_for_comparison(self):
        rows = [
            _row("rev-a", title="电源保障链"),
            _row("rev-b", title="显示输出链"),
        ]

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(
                calculation.icp_calculation_node(
                    _state(
                        rows,
                        query="比较并计算 RBD-rev-a 与 RBD-rev-b 的可靠度",
                    )
                )
            )

        self.assertEqual(
            ["rev-a", "rev-b"],
            [item["revision_id"] for item in result["calculation_results"]],
        )

    def test_revision_selector_picks_one_of_two_same_title_graphs(self):
        rows = [
            _row("rev-a", doc_id="doc-a", title="共用标题"),
            _row("rev-b", doc_id="doc-b", title="共用标题"),
        ]
        state = _state(
            rows,
            selected_conversation_anchors=[
                {
                    "doc_type": "rbd",
                    "record_id": "RBD-BUSINESS",
                    "answer_fields": {"revision_id": "rev-b"},
                }
            ],
        )

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual(
            ["rev-b"],
            [item["revision_id"] for item in result["calculation_results"]],
        )

    def test_selector_reads_stable_id_from_structured_document_metadata(self):
        selected = _row("rev-a")
        selected.pop("diagram_key")
        selected["document_metadata"] = {
            "answer_fields": {"diagram_key": "RBD-METADATA"}
        }
        state = _state(
            [selected],
            query="按已选图计算可靠度",
            selected_conversation_anchors=[
                {
                    "doc_type": "rbd",
                    "doc_id": "doc-rbd",
                    "record_id": "RBD-METADATA",
                    "answer_fields": {"diagram_key": "RBD-METADATA"},
                }
            ],
        )

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual(
            ["rev-a"],
            [item["revision_id"] for item in result["calculation_results"]],
        )

    def test_explicit_labelled_revision_id_is_a_stable_selector(self):
        rows = [
            _row("rev-a", doc_id="doc-a", title="共用标题"),
            _row("rev-b", doc_id="doc-b", title="共用标题"),
        ]
        state = _state(rows, query="计算 revision_id=rev-a 的可靠度")

        with patch.object(calculation, "_query_graphs", return_value=[]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual(
            ["rev-a"],
            [item["revision_id"] for item in result["calculation_results"]],
        )

    def test_reused_diagram_key_needs_document_or_configuration_scope(self):
        rows = [
            _row(
                "rev-a",
                diagram_key="RBD-SHARED",
                doc_id="doc-a",
                configuration="CONFIG-A",
            ),
            _row(
                "rev-b",
                diagram_key="RBD-SHARED",
                doc_id="doc-b",
                configuration="CONFIG-B",
            ),
        ]
        ambiguous = _state(rows, query="计算 RBD-SHARED 可靠度")
        scoped = _state(
            rows,
            query="按已选图计算可靠度",
            selected_conversation_anchors=[
                {
                    "doc_type": "rbd",
                    "doc_id": "doc-b",
                    "record_id": "RBD-SHARED",
                    "scope": {"configuration": "CONFIG-B"},
                }
            ],
        )

        with patch.object(calculation, "_query_graphs", return_value=[]):
            ambiguous_result = _run(
                calculation.icp_calculation_node(ambiguous)
            )
            scoped_result = _run(calculation.icp_calculation_node(scoped))

        self.assertEqual([], ambiguous_result["calculation_results"])
        self.assertEqual(
            ["rev-b"],
            [item["revision_id"] for item in scoped_result["calculation_results"]],
        )

    def test_repository_title_match_without_current_stable_handoff_is_not_calculated(self):
        recalled = _row("rev-title-only", title="只有标题命中")
        state = _state([], query="计算只有标题命中的 RBD 可靠度")

        with patch.object(calculation, "_query_graphs", return_value=[recalled]):
            result = _run(calculation.icp_calculation_node(state))

        self.assertEqual([], result["calculation_results"])
        self.assertTrue(
            any(
                "未通过精确稳定 ID" in warning
                for warning in result["tool_results"][0]["diagnostics"]["warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
