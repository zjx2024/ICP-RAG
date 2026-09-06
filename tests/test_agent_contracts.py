import unittest

from backend.agent.answer_contract import (
    focused_answer_contract,
    normalize_answer_fields,
    required_answer_fields,
)
from backend.agent.contracts import (
    EXACT_MATCH_TYPES,
    answer_mode_for_intent,
    binding_basis_rank,
    deduplicate_evidence,
    evidence_identity,
    make_evidence,
    make_tool_result,
    match_type_is_exact,
    match_type_precision,
)
from backend.agent.state import create_initial_state


class EvidenceContractTests(unittest.TestCase):
    def test_free_form_facets_never_expand_answer_or_tool_obligations(self):
        for facets in (
            ["可靠性"],
            ["严酷度与影响"],
            ["成功路径和冗余"],
            ["换一种说法询问同一件事"],
        ):
            with self.subTest(facets=facets):
                self.assertEqual(
                    [],
                    normalize_answer_fields([], requested_facets=facets),
                )
                self.assertEqual(
                    [],
                    required_answer_fields(
                        {"query_frame": {"requested_facets": facets}}
                    ),
                )

    def test_initial_symptom_diagnosis_is_not_narrowed_by_one_field(self):
        state = {
            "intent": "symptom_diagnosis",
            "query_targets": [
                {"intent": "symptom_diagnosis", "required_fields": ["failure_mode"]}
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["failure_mode"],
                "response_mode": {
                    "mode": "diagnostic",
                    "basis": "当前故障现象",
                },
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": ["failure_mode"],
                    "basis": "当前故障现象",
                },
            },
        }

        self.assertFalse(focused_answer_contract(state))

    def test_multi_dimension_diagnosis_is_not_a_focused_field_contract(self):
        state = {
            "intent": "cause_analysis",
            "query_targets": [
                {"intent": "cause_analysis"},
                {"intent": "impact_analysis"},
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "failure_mode",
                    "fault_cause",
                    "local_impact",
                    "final_impact",
                    "severity",
                    "detection_method",
                    "maintenance_action",
                    "history_time",
                    "history_root",
                    "history_steps",
                    "history_solution",
                    "history_result",
                ],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": [
                        "failure_mode",
                        "fault_cause",
                        "local_impact",
                        "maintenance_action",
                        "history_result",
                    ],
                    "basis": "当前多维诊断",
                },
            },
        }

        self.assertFalse(focused_answer_contract(state))

    def test_single_field_followups_remain_focused(self):
        for intent, field in (
            ("impact_analysis", "severity"),
            ("history_query", "history_solution"),
            ("entity_profile", "record_identity"),
        ):
            with self.subTest(intent=intent, field=field):
                state = {
                    "intent": intent,
                    "query_targets": [
                        {"intent": intent, "required_fields": [field]}
                    ],
                    "query_frame": {
                        "coverage_scope": "focused_detail",
                        "required_answer_fields": [field],
                    },
                }

                self.assertTrue(focused_answer_contract(state))

    def test_answer_mode_contract_separates_knowledge_from_diagnosis(self):
        self.assertEqual("knowledge_explanation", answer_mode_for_intent("impact_analysis"))
        self.assertEqual("maintenance_guidance", answer_mode_for_intent("maintenance_query"))
        self.assertEqual("fault_diagnosis", answer_mode_for_intent("symptom_diagnosis"))
        self.assertEqual("exact_code_lookup", answer_mode_for_intent("exact_code"))
        self.assertEqual("architecture_knowledge", answer_mode_for_intent("function_location"))

    def test_identity_is_stable_and_claim_specific(self):
        common = {
            "source_kind": "postgres",
            "doc_type": "hmc",
            "doc_id": "doc-1",
            "record_id": "HMC-001",
        }
        diagnosis_id = evidence_identity(**common, claim_type="diagnosis")
        self.assertEqual(diagnosis_id, evidence_identity(**common, claim_type="diagnosis"))
        self.assertNotEqual(diagnosis_id, evidence_identity(**common, claim_type="maintenance"))

    def test_make_evidence_normalizes_optional_fields(self):
        evidence = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="fmeca",
            doc_id=" doc-1 ",
            doc_name=" FMECA.xlsx ",
            chunk_id="chunk-1",
            content=" fault mode ",
            display_summary=" 可读故障摘要 ",
            scope={"configuration": " CONFIG-A ", "ignored": "value"},
            retrieval_score="0.82",
            rerank_score="invalid",
            derived_from=["ev-a", "ev-a", ""],
            limitations=[" 仅为检索线索 ", "仅为检索线索"],
            metadata={"index": "icp_chunks"},
        )
        self.assertEqual("doc-1", evidence["doc_id"])
        self.assertEqual("FMECA.xlsx", evidence["doc_name"])
        self.assertEqual({"configuration": "CONFIG-A"}, evidence["scope"])
        self.assertEqual(0.82, evidence["retrieval_score"])
        self.assertIsNone(evidence["rerank_score"])
        self.assertEqual(["ev-a"], evidence["derived_from"])
        self.assertEqual(["仅为检索线索"], evidence["limitations"])
        self.assertEqual({"index": "icp_chunks"}, evidence["metadata"])
        self.assertEqual("可读故障摘要", evidence["display_summary"])

    def test_make_evidence_rejects_unknown_enum_values(self):
        with self.assertRaises(ValueError):
            make_evidence(
                source_kind="unknown",  # type: ignore[arg-type]
                authority="search_projection",
                evidence_level="L3_INFERRED",
                claim_type="retrieval_hit",
            )

    def test_machine_validated_graph_contract_states_are_explicit(self):
        validated = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            review_status="auto_validated",
            metadata={"validation_status": "auto_validated"},
        )
        provisional = make_evidence(
            source_kind="postgres",
            authority="unconfirmed_extraction",
            evidence_level="L4_UNRESOLVED",
            claim_type="rbd_structure",
            review_status="provisional",
        )
        quarantined = make_evidence(
            source_kind="postgres",
            authority="unconfirmed_extraction",
            evidence_level="L4_UNRESOLVED",
            claim_type="rbd_structure",
            review_status="quarantined",
        )

        self.assertEqual("validated_relation", validated["authority"])
        self.assertEqual("auto_validated", validated["review_status"])
        self.assertEqual("provisional", provisional["review_status"])
        self.assertEqual("quarantined", quarantined["review_status"])

    def test_deduplicate_keeps_order_and_latest_payload(self):
        items = [
            {"evidence_id": "ev-1", "content": "old"},
            {"evidence_id": "ev-2", "content": "second"},
            {"evidence_id": "ev-1", "content": "new"},
            {"content": "missing identity"},
        ]
        result = deduplicate_evidence(items)
        self.assertEqual(["ev-1", "ev-2"], [item["evidence_id"] for item in result])
        self.assertEqual("new", result[0]["content"])

    def test_deduplicate_preserves_exact_match_identity_from_earlier_query(self):
        result = deduplicate_evidence(
            [
                {
                    "evidence_id": "ev-hmc",
                    "content": "精确查询内容",
                    "match_type": "exact_components",
                },
                {
                    "evidence_id": "ev-hmc",
                    "content": "后续水合的完整内容",
                    "match_type": "semantic_id_hydration",
                },
            ]
        )

        self.assertEqual("后续水合的完整内容", result[0]["content"])
        self.assertEqual("exact_components", result[0]["match_type"])

    def test_every_registered_exact_identity_survives_semantic_hydration(self):
        for exact_match_type in EXACT_MATCH_TYPES:
            with self.subTest(match_type=exact_match_type):
                result = deduplicate_evidence(
                    [
                        {
                            "evidence_id": "ev-shared",
                            "content": "direct",
                            "match_type": exact_match_type,
                        },
                        {
                            "evidence_id": "ev-shared",
                            "content": "hydrated",
                            "match_type": "semantic_id_hydration",
                        },
                    ]
                )

                self.assertEqual("hydrated", result[0]["content"])
                self.assertEqual(exact_match_type, result[0]["match_type"])

    def test_match_type_and_binding_strength_are_central_fail_closed_contracts(self):
        self.assertTrue(match_type_is_exact("exact_structured_identity"))
        self.assertTrue(match_type_is_exact("exact_fta_reference"))
        self.assertTrue(match_type_is_exact("exact_record_identity"))
        self.assertFalse(match_type_is_exact("exact_unreviewed_plugin_label"))
        self.assertGreater(
            match_type_precision("exact_record_identity"),
            match_type_precision("semantic_id_hydration"),
        )
        self.assertGreater(
            binding_basis_rank("exact_business_id"),
            binding_basis_rank("identity_label_containment"),
        )
        self.assertEqual(0, binding_basis_rank("unregistered_basis"))


class ToolResultContractTests(unittest.TestCase):
    def test_empty_and_failed_are_distinct_auditable_states(self):
        empty = make_tool_result(tool="hmc_query", status="empty", backend="postgres")
        failed = make_tool_result(
            tool="hmc_query",
            status="failed",
            backend="postgres",
            errors=["connection unavailable"],
        )
        self.assertEqual("empty", empty["status"])
        self.assertEqual("failed", failed["status"])
        self.assertEqual(["connection unavailable"], failed["diagnostics"]["errors"])

    def test_initial_state_contains_new_fields_and_keeps_legacy_fields(self):
        state = create_initial_state("query", "agent", "")
        for key in (
            "allowed_tools",
            "hybrid_doc_types",
            "retrieval_scope",
            "tool_results",
            "evidence_registry",
            "context_evidence_ids",
            "approved_evidence_ids",
            "safety_decision",
            "diagnostic_support",
            "answer_mode",
            "evidence_catalog",
            "used_evidence_ids",
        ):
            self.assertIn(key, state)
        self.assertIn("diagnostic_confidence", state)
        self.assertIn("domain_results", state)


if __name__ == "__main__":
    unittest.main()
