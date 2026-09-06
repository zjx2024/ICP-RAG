from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.rbd_contract import (
    RBD_AUTHORITY_CONTRACT_VERSION,
    bind_rbd_models,
    merge_rbd_query_bindings,
    selected_rbd_anchor_selectors,
    validate_rbd_authority,
    validate_rbd_binding_contract,
)
from backend.agent.tools import icp_calculation


def _run(coro):
    return asyncio.run(coro)


def _graph(*, business_id="MOD-PWR-A", local_id="block-local"):
    return {
        "vertices": [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {
                "id": local_id,
                "label": "ReliabilityBlock",
                "properties": {
                    "name": "自然语言模块名",
                    "module_id": business_id,
                    "reliability": 0.98,
                },
            },
            {"id": "end", "label": "End", "properties": {"name": "END"}},
        ],
        "edges": [
            {"source": "start", "target": local_id},
            {"source": local_id, "target": "end"},
        ],
    }


def _row(
    revision_id="rev-a",
    *,
    diagram_key="RBD-A",
    doc_id="doc-a",
    status="auto_validated",
    active=True,
    projection="published",
    business_id="MOD-PWR-A",
    local_id="block-local",
):
    return {
        "diagram_type": "rbd",
        "revision_id": revision_id,
        "diagram_id": f"diagram-{revision_id}",
        "diagram_key": diagram_key,
        "doc_id": doc_id,
        "validation_status": status,
        "projection_status": projection,
        "is_active": active,
        "graph": _graph(business_id=business_id, local_id=local_id),
        "image_title": "共享显示名",
    }


def _evidence(row):
    return make_evidence(
        source_kind="postgres",
        authority="validated_relation",
        evidence_level="L2_CORROBORATED",
        claim_type="rbd_structure",
        doc_type="rbd",
        doc_id=row["doc_id"],
        record_id=row["diagram_key"],
        review_status=row["validation_status"],
        content="正式结构",
        metadata={
            "revision_id": row["revision_id"],
            "diagram_id": row["diagram_id"],
            "diagram_key": row["diagram_key"],
            "validation_status": row["validation_status"],
            "projection_status": row["projection_status"],
            "is_active": row["is_active"],
            "rbd_block_business_ids": ["MOD-PWR-A"],
        },
    )


def _binding_for_id(row, query=None):
    query = query or f"检查 {row['diagram_key']}"
    selected, binding = bind_rbd_models(
        [row],
        required=True,
        raw_user_query=query,
        explicit_rbd_ids=[row["diagram_key"]],
    )
    assert selected
    return binding


class RbdAuthoritySelectionTests(unittest.TestCase):
    def test_natural_name_cannot_select_a_formal_model(self):
        rows = [
            _row("rev-a", diagram_key="RBD-A"),
            _row("rev-b", diagram_key="RBD-B", business_id="MOD-DSP-B"),
        ]
        selected, binding = bind_rbd_models(
            rows,
            required=True,
            raw_user_query="计算自然语言模块名的可靠度",
        )

        self.assertEqual([], selected)
        self.assertFalse(binding["bound"])
        self.assertTrue(binding["ambiguous"])

    def test_natural_name_cannot_be_smuggled_as_explicit_id(self):
        row = _row(diagram_key="共享显示名")
        selected, binding = bind_rbd_models(
            [row],
            required=True,
            raw_user_query="计算共享显示名的可靠度",
            explicit_rbd_ids=["共享显示名"],
        )

        self.assertEqual([row], selected)
        self.assertFalse(binding["bound"])
        self.assertEqual("single_unbound_candidate", binding["basis"])

    def test_exact_stable_rbd_id_is_authoritative(self):
        row = _row()
        query = "请读取 RBD-A 的正式结构"
        selected, binding = bind_rbd_models(
            [row],
            required=True,
            raw_user_query=query,
            explicit_rbd_ids=["RBD-A"],
        )

        self.assertEqual(["rev-a"], [item["revision_id"] for item in selected])
        self.assertTrue(binding["bound"])
        self.assertEqual("exact_rbd_id", binding["basis"])
        self.assertTrue(
            validate_rbd_authority(
                binding,
                models=[row],
                raw_user_query=query,
            )["valid"]
        )

    def test_selected_citation_identity_is_authoritative_with_scope(self):
        row = _row(diagram_key="RBD-SHARED", doc_id="doc-b")
        anchors = [
            {
                "doc_type": "rbd",
                "doc_id": "doc-b",
                "record_id": "RBD-SHARED",
                "answer_fields": {"revision_id": "rev-a"},
            }
        ]
        selected, binding = bind_rbd_models(
            [row],
            required=True,
            raw_user_query="按刚才引用的模型继续",
            anchor_selectors=selected_rbd_anchor_selectors(anchors),
        )

        self.assertEqual([row], selected)
        self.assertEqual("selected_rbd_identity", binding["basis"])
        self.assertTrue(
            validate_rbd_authority(
                binding,
                models=[row],
                raw_user_query="按刚才引用的模型继续",
                selected_anchors=anchors,
            )["valid"]
        )

    def test_exact_formal_block_business_id_intersection_is_authoritative(self):
        row = _row()
        query = "计算 MOD-PWR-A 所在正式 RBD"
        selected, binding = bind_rbd_models(
            [row],
            required=True,
            raw_user_query=query,
            explicit_block_business_ids=["MOD-PWR-A"],
        )

        self.assertEqual([row], selected)
        self.assertEqual("exact_block_business_id", binding["basis"])
        self.assertTrue(
            validate_rbd_authority(
                binding,
                models=[row],
                raw_user_query=query,
            )["valid"]
        )

    def test_local_block_id_cannot_cross_the_business_identity_boundary(self):
        row = _row(local_id="BLOCK-LOCAL-7")
        selected, binding = bind_rbd_models(
            [row],
            required=True,
            raw_user_query="计算 BLOCK-LOCAL-7 所在 RBD",
            explicit_block_business_ids=["BLOCK-LOCAL-7"],
        )

        self.assertEqual([], selected)
        self.assertFalse(binding["bound"])

    def test_formal_lifecycle_and_revision_are_rechecked(self):
        source = _row()
        query = "读取 RBD-A"
        binding = _binding_for_id(source, query)
        mutations = [
            _row("rev-other"),
            _row(status="provisional"),
            _row(active=False),
            _row(projection="unpublished"),
            {
                **source,
                "metadata": {"revision_id": "conflicting-revision"},
            },
        ]
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                self.assertFalse(
                    validate_rbd_authority(
                        binding,
                        models=[mutated],
                        raw_user_query=query,
                    )["valid"]
                )

    def test_legacy_or_forged_bound_true_is_not_a_contract(self):
        forged = {
            "required": True,
            "bound": True,
            "basis": "exact_typed_subject",
            "record_ids": ["revision_id:rev-a"],
        }
        self.assertFalse(validate_rbd_binding_contract(forged)["valid"])


class RbdAuthorityConsumerTests(unittest.TestCase):
    def test_gate_rejects_forged_bound_true_and_removes_formal_rbd(self):
        row = _row()
        source = _evidence(row)
        result = _run(
            evidence_gate_node(
                {
                    "query": "读取 RBD-A",
                    "answer_mode": "knowledge_explanation",
                    "allowed_tools": ["rbd_query"],
                    "rbd_query_binding": {
                        "required": True,
                        "bound": True,
                        "basis": "exact_typed_subject",
                        "record_ids": ["revision_id:rev-a"],
                    },
                    "evidence_registry": [source],
                }
            )
        )

        self.assertNotIn(source["evidence_id"], result["context_evidence_ids"])
        self.assertIn("rbd_model_identity", result["safety_decision"]["blocked_claims"])
        self.assertFalse(
            result["safety_decision"]["rbd_authority_validation"]["valid"]
        )

    def test_gate_accepts_same_contract_only_for_same_formal_revision(self):
        row = _row()
        query = "读取 RBD-A"
        binding = _binding_for_id(row, query)
        source = _evidence(row)
        result = _run(
            evidence_gate_node(
                {
                    "query": query,
                    "answer_mode": "knowledge_explanation",
                    "allowed_tools": ["rbd_query"],
                    "rbd_query_binding": binding,
                    "evidence_registry": [source],
                }
            )
        )

        self.assertIn(source["evidence_id"], result["approved_evidence_ids"])
        self.assertTrue(
            result["safety_decision"]["rbd_authority_validation"]["valid"]
        )

    def test_calculator_rejects_natural_name_and_accepts_exact_contract(self):
        row = _row()
        parent = _evidence(row)
        natural_state = {
            "query": "计算自然语言模块名的可靠度",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "calculation_authorizations": [
                {
                    "operation": "rbd_reliability",
                    "mode": "calculate",
                    "target": "rbd",
                    "basis": "计算自然语言模块名的可靠度",
                }
            ],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
        }
        query = "计算 RBD-A 的可靠度"
        exact_state = {
            **natural_state,
            "query": query,
            "calculation_authorizations": [
                {
                    "operation": "rbd_reliability",
                    "mode": "calculate",
                    "target": "rbd",
                    "basis": query,
                }
            ],
            "rbd_query_binding": _binding_for_id(row, query),
        }
        with patch.object(icp_calculation, "_query_graphs", return_value=[]):
            natural = _run(icp_calculation.icp_calculation_node(natural_state))
            exact = _run(icp_calculation.icp_calculation_node(exact_state))

        self.assertEqual([], natural["calculation_results"])
        self.assertEqual(["rev-a"], [item["revision_id"] for item in exact["calculation_results"]])

    def test_adaptive_merge_rejects_weak_or_revision_conflicts(self):
        first_row = _row()
        first = _binding_for_id(first_row, "读取 RBD-A")
        weak = {
            "required": True,
            "bound": True,
            "basis": "exact_typed_subject",
            "record_ids": ["revision_id:rev-a"],
            "candidate_record_ids": ["revision_id:rev-a"],
        }
        self.assertFalse(merge_rbd_query_bindings(first, weak)["bound"])

        second_row = _row("rev-b", diagram_key="RBD-B")
        second = _binding_for_id(second_row, "读取 RBD-B")
        conflict = merge_rbd_query_bindings(first, second)
        self.assertFalse(conflict["bound"])
        self.assertTrue(conflict["ambiguous"])


if __name__ == "__main__":
    unittest.main()
