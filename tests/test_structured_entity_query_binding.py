import asyncio
import unittest
from unittest.mock import patch

from backend.agent.query_understanding import bind_structured_function_entity
from backend.agent.tools import icp_diagnosis as diagnosis


FUNCTION_ROWS = [
    {
        "doc_id": "doc-function",
        "content_type": "function_component",
        "component_id": "MOD-PWR-PSUA",
        "name": "A路电源模块",
        "abbreviation": "PSU-A",
    },
    {
        "doc_id": "doc-function",
        "content_type": "function_record",
        "function_id": "FUNC-120",
        "module_id": "MOD-PWR-PSUA",
        "module_name": "A路电源模块",
        "function_name": "A路稳压供电",
    },
    {
        "doc_id": "doc-function",
        "content_type": "function_record",
        "function_id": "FUNC-210",
        "module_id": "MOD-GPP-BCM",
        "module_name": "启动控制模块",
        "function_name": "BCM启动时序控制",
    },
]


class StructuredEntityBindingTests(unittest.TestCase):
    def test_same_authoritative_row_tie_binds_module_function_chain(self):
        row = {
            "doc_id": "doc-function",
            "content_type": "function_record",
            "module_id": "MOD-IFC-BCA",
            "module_name": "外部通信链路",
            "function_id": "FUNC-530",
            "function_name": "外部通信链路",
        }

        binding = bind_structured_function_entity("外部通信链路中断", [row])

        self.assertTrue(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertEqual(
            ["MOD-IFC-BCA", "FUNC-530"], binding["business_ids"]
        )
        self.assertIn("FUNC-530", binding["terms"])
        self.assertEqual("same_row_module_function_chain", binding["basis"])

    def test_explicit_function_id_bridges_to_same_row_module_id(self):
        row = {
            "doc_id": "doc-function",
            "content_type": "function_record",
            "module_id": "MOD-IFC-GW",
            "module_name": "网络网关模块",
            "function_id": "FUNC-310",
            "function_name": "外部业务数据转发",
        }

        binding = bind_structured_function_entity("FUNC-310是什么", [row])

        self.assertTrue(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertEqual(
            {"FUNC-310", "MOD-IFC-GW"}, set(binding["business_ids"])
        )
        self.assertIn("MOD-IFC-GW", binding["terms"])
        self.assertIn("网络网关模块", binding["terms"])

    def test_equal_module_and_function_matches_on_different_rows_fail_closed(self):
        rows = [
            {
                "doc_id": "doc-function",
                "content_type": "function_record",
                "module_id": "MOD-IFC-BCA",
                "module_name": "外部通信链路",
            },
            {
                "doc_id": "doc-function",
                "content_type": "function_record",
                "function_id": "FUNC-530",
                "function_name": "外部通信链路",
            },
        ]

        binding = bind_structured_function_entity("外部通信链路中断", rows)

        self.assertFalse(binding["bound"])
        self.assertTrue(binding["ambiguous"])
        self.assertEqual([], binding["business_ids"])

    def test_explicit_branch_binds_authoritative_module_and_linked_function(self):
        binding = bind_structured_function_entity("A路电源损坏", FUNCTION_ROWS)

        self.assertTrue(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertEqual(
            ["MOD-PWR-PSUA", "FUNC-120"], binding["business_ids"]
        )
        self.assertEqual([0, 1], binding["row_indexes"])
        for expected in (
            "MOD-PWR-PSUA",
            "A路电源模块",
            "PSU-A",
            "FUNC-120",
            "A路稳压供电",
        ):
            self.assertIn(expected, binding["terms"])
        self.assertNotIn("FUNC-210", binding["terms"])

    def test_generic_power_word_does_not_choose_one_branch(self):
        rows = [
            *FUNCTION_ROWS[:2],
            {
                "doc_id": "doc-function",
                "content_type": "function_component",
                "component_id": "MOD-PWR-PSUB",
                "name": "B路电源模块",
                "abbreviation": "PSU-B",
            },
        ]

        binding = bind_structured_function_entity("电源损坏", rows)

        self.assertFalse(binding["bound"])
        self.assertEqual([], binding["business_ids"])
        self.assertEqual([], binding["terms"])

    def test_handoff_prioritizes_structured_terms_for_fmeca_and_rbd(self):
        state = {
            "query": "A路电源损坏会有什么影响",
            "rewritten_query": "A路电源损坏的影响",
            "query_frame": {
                "entities": ["A路电源"],
                "normalized_terms": ["A路电源损坏"],
            },
        }
        terms_by_tool = {
            "function_locator": ["A路电源", "电源"],
            "fmeca_query": ["A路电源", "电源", "Power A Failure"],
            "rbd_query": ["A路电源", "电源"],
            "hmc_query": ["A路电源", "电源"],
            "mfl_query": ["A路电源", "电源"],
            "fault_history_query": ["A路电源", "电源"],
        }

        selected, binding = diagnosis._apply_structured_entity_binding(
            state, FUNCTION_ROWS, terms_by_tool
        )

        self.assertTrue(binding["bound"])
        self.assertEqual(
            ["MOD-PWR-PSUA", "FUNC-120"],
            [
                row.get("component_id") or row.get("function_id")
                for row in selected
            ],
        )
        for tool in ("fmeca_query", "rbd_query"):
            self.assertEqual("MOD-PWR-PSUA", terms_by_tool[tool][0])
            self.assertIn("PSU-A", terms_by_tool[tool])
            self.assertIn("FUNC-120", terms_by_tool[tool])
            self.assertNotIn("FUNC-210", terms_by_tool[tool])

        # The extension is still exact-ID based. MFL/HMC/history get the
        # authoritative Function Locator IDs for a bounded cross-document
        # lookup, but no similar-text entity merge is introduced.
        for tool in ("hmc_query", "mfl_query", "fault_history_query"):
            self.assertEqual("MOD-PWR-PSUA", terms_by_tool[tool][0])
            self.assertIn("FUNC-120", terms_by_tool[tool])
            self.assertNotIn("FUNC-210", terms_by_tool[tool])

    def test_generic_locator_containment_cannot_steer_relation_query_to_one_child(self):
        state = {
            "query": "两路稳压供电为什么会一起不可用",
            "relation_expansion": {
                "needed": True,
                "reason": "需要共同上游和影响链",
            },
            "query_frame": {
                "entities": ["两路稳压供电", "供电模块"],
                "required_answer_fields": ["structural_cause_chain"],
            },
        }
        terms_by_tool = {
            "function_locator": ["两路稳压供电", "供电模块"],
            "fmeca_query": ["供电模块"],
            "rbd_query": ["供电模块"],
            "hmc_query": ["供电模块"],
            "mfl_query": ["供电模块"],
            "fault_history_query": ["供电模块"],
        }
        hum_only = [{
            "doc_id": "doc-function",
            "content_type": "function_component",
            "component_id": "MOD-PWR-HUM",
            "name": "保持供电模块",
            "abbreviation": "HUM",
        }]

        selected, binding = diagnosis._apply_structured_entity_binding(
            state, hum_only, terms_by_tool
        )

        self.assertEqual(hum_only, selected)
        self.assertFalse(binding["bound"])
        self.assertEqual(
            "generic_locator_match_not_used_as_cross_document_seed",
            binding["basis"],
        )
        self.assertEqual(["供电模块"], terms_by_tool["fmeca_query"])
        self.assertEqual(["供电模块"], terms_by_tool["rbd_query"])


class StructuredEntityRankingTests(unittest.TestCase):
    def test_multiple_fmeca_rows_share_explicit_product_and_function_identity(self):
        rows = [
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出失效",
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出欠压",
            },
        ]

        handoff = diagnosis._shared_fmeca_identity_terms(rows)

        self.assertEqual(
            {"PSU-A", "FUNC-120", "A路稳压供电"},
            set(handoff),
        )

    def test_fmeca_identity_consensus_ignores_one_unrelated_weak_hit(self):
        rows = [
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "retrieval_identity_score": 4,
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "retrieval_identity_score": 4,
            },
            {
                "code": "FM-SYN-016",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
                "retrieval_identity_score": 0,
            },
        ]

        self.assertEqual(
            {"PSU-A", "FUNC-120", "A路稳压供电"},
            set(diagnosis._shared_fmeca_identity_terms(rows)),
        )

    def test_positive_identity_minority_outranks_zero_score_majority(self):
        strong = [
            {
                "code": f"FM-SYN-00{index}",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "retrieval_identity_score": 16,
            }
            for index in (4, 5)
        ]
        zero_score_majority = [
            {
                "code": f"FM-SYN-01{index}",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
                "retrieval_identity_score": 0,
            }
            for index in range(5)
        ]

        self.assertEqual(
            {"PSU-A", "FUNC-120", "A路稳压供电"},
            set(
                diagnosis._shared_fmeca_identity_terms(
                    [*strong, *zero_score_majority]
                )
            ),
        )

    def test_structured_binding_rejects_non_intersecting_fmeca_cluster(self):
        rows = [
            {
                "code": "FM-SYN-032",
                "product_flag": "GW / FUNC-310",
                "func_management": "外部业务数据转发",
                "retrieval_identity_score": 4,
            },
            {
                "code": "FM-SYN-033",
                "product_flag": "GW / FUNC-310",
                "func_management": "外部业务数据转发",
                "retrieval_identity_score": 4,
            },
            {
                "code": "FM-SYN-015",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
                "retrieval_identity_score": 100,
            },
            {
                "code": "FM-SYN-016",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
                "retrieval_identity_score": 100,
            },
        ]

        self.assertEqual(
            {"FUNC-310", "外部业务数据转发"},
            set(
                diagnosis._shared_fmeca_identity_terms(
                    rows,
                    structured_business_ids=["MOD-IFC-GW", "FUNC-310"],
                )
            ),
        )

    def test_fmeca_identity_consensus_fails_closed_on_equal_clusters(self):
        rows = [
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
            },
            {
                "code": "FM-SYN-015",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
            },
            {
                "code": "FM-SYN-016",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
            },
        ]

        self.assertEqual([], diagnosis._shared_fmeca_identity_terms(rows))

    def test_fmeca_identity_handoff_requires_multiple_rows_with_shared_codes(self):
        single = [
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
            }
        ]
        disjoint = [
            *single,
            {
                "code": "FM-SYN-016",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动时序控制",
            },
        ]
        missing_codes = [
            {
                "code": "FM-SYN-004",
                "product_flag": "",
                "func_management": "A路稳压供电",
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "",
                "func_management": "A路稳压供电",
            },
        ]

        self.assertEqual([], diagnosis._shared_fmeca_identity_terms(single))
        self.assertEqual([], diagnosis._shared_fmeca_identity_terms(disjoint))
        self.assertEqual([], diagnosis._shared_fmeca_identity_terms(missing_codes))

    def test_fmeca_business_fields_keep_all_psua_rows_ahead_of_bcm(self):
        binding = bind_structured_function_entity("A路电源损坏", FUNCTION_ROWS)
        terms = [*binding["terms"], "A路电源", "电源", "Power A Failure"]
        rows = [
            {
                "code": "FM-SYN-016",
                "product_flag": "BCM / FUNC-210",
                "func_management": "启动控制",
                "fault_mode": "BCM启动时序超时",
                "fault_cause": "电源良好信号延迟",
            },
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出失效",
                "fault_cause": "功率器件损坏",
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出欠压",
                "fault_cause": "调压环路异常",
            },
            {
                "code": "FM-SYN-006",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出中断",
                "fault_cause": "支路开路",
            },
        ]

        ranked = diagnosis._rank_table_rows(
            rows,
            (
                "code",
                "product_flag",
                "func_management",
                "fault_mode",
                "fault_cause",
            ),
            terms,
            3,
            priority_columns=(
                "code",
                "product_flag",
                "func_management",
                "fault_mode",
            ),
            priority_terms=binding["terms"],
        )

        self.assertEqual(
            {"FM-SYN-004", "FM-SYN-005", "FM-SYN-006"},
            {row["code"] for row in ranked},
        )
        self.assertNotIn("FM-SYN-016", [row["code"] for row in ranked])
        self.assertTrue(all(row["retrieval_identity_score"] > 0 for row in ranked))

    def test_rbd_vertices_business_fields_outrank_generic_power_text(self):
        binding = bind_structured_function_entity("A路电源损坏", FUNCTION_ROWS)
        wrong = {
            "revision_id": "rev-bcm",
            "diagram_type": "rbd",
            "title": "电源启动控制路径",
            "diagram_key": "RBD-BCM",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {
                "vertices": [
                    {
                        "id": "BCM",
                        "properties": {
                            "module_id": "MOD-GPP-BCM",
                            "name": "启动控制模块",
                        },
                    }
                ]
            },
        }
        correct = {
            "revision_id": "rev-psua",
            "diagram_type": "rbd",
            "title": "A路供电成功路径",
            "diagram_key": "RBD-PWR-A",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {
                "vertices": [
                    {
                        "id": "PSU-A",
                        "properties": {
                            "module_id": "MOD-PWR-PSUA",
                            "function_id": "FUNC-120",
                            "name": "PSU-A",
                        },
                    }
                ]
            },
        }
        with patch.object(
            diagnosis.diagram_repository,
            "list_active_revisions",
            return_value=[wrong, correct],
        ):
            rows = diagnosis._query_diagrams(
                "rbd",
                [*binding["terms"], "电源"],
                2,
                entity_terms=binding["terms"],
            )

        self.assertEqual(["rev-psua", "rev-bcm"], [row["revision_id"] for row in rows])
        self.assertGreater(rows[0]["retrieval_identity_score"], 0)
        self.assertEqual(0, rows[1]["retrieval_identity_score"])

    def test_rbd_optional_nodes_shape_keeps_identity_ranking_compatible(self):
        binding = bind_structured_function_entity("A路电源损坏", FUNCTION_ROWS)
        legacy_nodes = {
            "revision_id": "rev-legacy-nodes",
            "diagram_type": "rbd",
            "title": "A路供电成功路径",
            "diagram_key": "RBD-PWR-A",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {
                "nodes": [
                    {
                        "id": "PSU-A",
                        "properties": {
                            "module_id": "MOD-PWR-PSUA",
                            "function_id": "FUNC-120",
                            "name": "PSU-A",
                        },
                    }
                ]
            },
        }
        with patch.object(
            diagnosis.diagram_repository,
            "list_active_revisions",
            return_value=[legacy_nodes],
        ):
            rows = diagnosis._query_diagrams(
                "rbd",
                binding["terms"],
                1,
                entity_terms=binding["terms"],
            )

        self.assertEqual(["rev-legacy-nodes"], [row["revision_id"] for row in rows])
        self.assertGreater(rows[0]["retrieval_identity_score"], 0)
        self.assertIn("MOD-PWR-PSUA", rows[0]["retrieval_identity_terms"])


class StructuredEntityDiagnosisFlowTests(unittest.TestCase):
    def test_hybrid_hydrated_function_chain_enriches_final_topology_identity(self):
        hydrated_function = {
            "id": 310,
            "doc_id": "doc-function",
            "content_type": "function_record",
            "module_id": "MOD-IFC-GW",
            "module_name": "外部通信链路",
            "function_id": "FUNC-310",
            "function_name": "外部通信链路",
        }
        linked_fmeca = {
            "id": 32,
            "doc_id": "doc-fmeca",
            "code": "FM-SYN-032",
            "product_flag": "GW / FUNC-310",
            "func_management": "外部通信链路",
            "fault_mode": "外部业务数据转发失效",
            "fault_cause": "网络网关模块异常",
            "final_impact": "外部设备无法收到业务数据",
        }
        state = {
            "scope_route": "knowledge_base",
            "query": "外部通信链路中断后还能工作吗",
            "rewritten_query": "外部通信链路中断后的剩余能力",
            "intent": "reliability_query",
            "query_targets": [{"intent": "reliability_query"}],
            "query_frame": {
                "entities": ["外部通信链路"],
                "normalized_terms": ["外部通信链路中断"],
                "required_answer_fields": [
                    "rbd_success_criterion",
                    "rbd_single_failure_boundary",
                ],
                "topology_dependency": {
                    "required": True,
                    "subject": "外部通信链路",
                    "subject_kind": "data_path",
                    "failure_state": "中断",
                    "question_aspects": ["continued_operation"],
                },
            },
            "tool_queries": {
                "rbd_query": ["外部通信链路"],
                "hybrid_search": ["外部通信链路中断"],
            },
            # Mirror the adaptive second batch: only hybrid_search executes
            # now, while exact structured hydration may use tools that the
            # first batch had already authorized.
            "allowed_tools": ["hybrid_search"],
            "authorized_tools": ["fmeca_query", "rbd_query", "hybrid_search"],
            "structured_hydration_tools": ["fmeca_query", "rbd_query"],
            "hybrid_doc_types": ["function_locator", "rbd"],
            "sql_limit": 8,
        }
        locator = {
            "doc_type": "function_locator",
            "doc_id": "doc-function",
            "record_id": "FUNC-310",
            "chunk_id": "function_record:310",
        }

        def fmeca_query(terms, _limit, _errors, *, entity_terms=()):
            normalized = {
                diagnosis.normalize_search_term(value) for value in terms
            }
            if "func310" in normalized:
                return [linked_fmeca]
            return []

        with (
            patch.object(diagnosis, "_query_diagrams", return_value=[]),
            patch.object(diagnosis, "_query_fmeca", side_effect=fmeca_query),
            patch.object(
                diagnosis,
                "_hybrid_search",
                return_value=([locator], {"backend": "opensearch"}),
            ),
            patch.object(
                diagnosis,
                "_search_locator_handoffs",
                return_value=[locator],
            ),
            patch.object(
                diagnosis,
                "_hydrate_diagram_search_hits",
                return_value={},
            ),
            patch.object(
                diagnosis,
                "_hydrate_record_search_hits",
                return_value={"function_locator": [hydrated_function]},
            ),
            patch.object(
                diagnosis,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: list(rows),
            ),
        ):
            result = asyncio.run(diagnosis.icp_diagnosis_node(state))

        self.assertEqual(
            ["MOD-IFC-GW", "FUNC-310"],
            result["topology_dependency"]["canonical_business_ids"],
        )
        self.assertIn(
            "FM-SYN-032",
            {
                item.get("record_id")
                for item in result["evidence_registry"]
                if item.get("doc_type") == "fmeca"
            },
        )
        self.assertFalse(
            any(
                call["tool"] in {"fmeca_query", "rbd_query"}
                for call in result["tool_calls"]
            )
        )
        hybrid_call = next(
            call for call in result["tool_calls"] if call["tool"] == "hybrid_search"
        )
        self.assertEqual(
            "postgres_hydration",
            hybrid_call["structured_entity_binding"]["stage"],
        )
        self.assertEqual(
            {
                "tool": "fmeca_query",
                "count": 1,
                "business_ids": ["MOD-IFC-GW", "FUNC-310"],
            },
            hybrid_call["structured_identity_hydrated"],
        )

    def test_topology_identity_survives_retrieval_only_followup(self):
        dependency = {
            "required": True,
            "subject": "A路供电",
            "subject_kind": "supply_path",
            "failure_state": "失去输出",
            "question_aspects": ["continuity"],
            "canonical_business_ids": ["PSU-A", "FUNC-120"],
            "identity_basis": "PostgreSQL 结构化业务ID精确交接",
        }
        state = {
            "scope_route": "knowledge_base",
            "query": "A路供电失去输出后还能工作吗",
            "rewritten_query": "A路供电失去输出后的连续工作能力",
            "intent": "impact_analysis",
            "query_targets": [
                {"intent": "impact_analysis"},
                {"intent": "reliability_query"},
            ],
            "query_frame": {
                "entities": ["A路供电"],
                "topology_dependency": {
                    key: value
                    for key, value in dependency.items()
                    if key not in {"canonical_business_ids", "identity_basis"}
                },
            },
            "topology_dependency": dependency,
            "allowed_tools": ["hybrid_search"],
            "authorized_doc_types": ["fmeca", "rbd"],
            "hybrid_doc_types": ["fmeca", "rbd"],
        }

        with patch.object(diagnosis, "_hybrid_search", return_value=([], {})):
            result = asyncio.run(diagnosis.icp_diagnosis_node(state))

        self.assertEqual(
            ["PSU-A", "FUNC-120"],
            result["topology_dependency"]["canonical_business_ids"],
        )
        self.assertEqual(
            "PostgreSQL 结构化业务ID精确交接",
            result["topology_dependency"]["identity_basis"],
        )

    def test_topology_query_hands_shared_fmeca_identity_to_rbd_query(self):
        fmeca_rows = [
            {
                "code": "FM-SYN-004",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出失效",
                "local_effect": "A路供电中断",
                "final_effect": "可用供电路径减少",
            },
            {
                "code": "FM-SYN-005",
                "product_flag": "PSU-A / FUNC-120",
                "func_management": "A路稳压供电",
                "fault_mode": "A路输出欠压",
                "local_effect": "A路输出电压偏低",
                "final_effect": "下游负载供电裕度降低",
            },
        ]
        state = {
            "scope_route": "knowledge_base",
            "query": "若A路电源损坏，会有什么影响",
            "rewritten_query": "A路电源损坏的影响与剩余供电能力",
            "intent": "impact_analysis",
            "query_targets": [
                {"intent": "impact_analysis"},
                {"intent": "reliability_query"},
            ],
            "query_frame": {
                "entities": ["A路电源"],
                "normalized_terms": ["A路电源损坏"],
                "required_answer_fields": [
                    "local_impact",
                    "final_impact",
                    "rbd_success_criterion",
                    "rbd_single_failure_boundary",
                ],
                "topology_dependency": {
                    "required": True,
                    "subject": "A路电源",
                    "subject_kind": "supply_path",
                    "failure_state": "损坏",
                    "question_aspects": ["impact"],
                },
            },
            "tool_queries": {
                "fmeca_query": ["A路电源损坏"],
                "rbd_query": ["A路电源"],
            },
            "allowed_tools": ["fmeca_query", "rbd_query"],
            "sql_limit": 8,
        }
        captured = {}

        def fmeca_query(_terms, _limit, _errors, *, entity_terms=()):
            captured["fmeca_entity_terms"] = list(entity_terms)
            return fmeca_rows

        def diagram_query(doc_type, terms, _limit, _errors, *, entity_terms=()):
            captured["doc_type"] = doc_type
            captured["rbd_terms"] = list(terms)
            captured["rbd_entity_terms"] = list(entity_terms)
            return []

        with (
            patch.object(diagnosis, "_query_fmeca", side_effect=fmeca_query),
            patch.object(diagnosis, "_query_diagrams", side_effect=diagram_query),
        ):
            result = asyncio.run(diagnosis.icp_diagnosis_node(state))

        self.assertEqual("rbd", captured["doc_type"])
        for expected in ("PSU-A", "FUNC-120", "A路稳压供电"):
            self.assertIn(expected, captured["rbd_terms"])
            self.assertIn(expected, captured["rbd_entity_terms"])
        rbd_call = next(
            call for call in result["tool_calls"] if call["tool"] == "rbd_query"
        )
        self.assertEqual(
            {"PSU-A", "FUNC-120", "A路稳压供电"},
            set(rbd_call["structured_fmeca_handoff"]["terms"]),
        )
        self.assertIn(
            "PostgreSQL FMECA",
            rbd_call["structured_fmeca_handoff"]["basis"],
        )
        self.assertEqual(
            {"PSU-A", "FUNC-120"},
            set(result["topology_dependency"]["canonical_business_ids"]),
        )
        self.assertEqual(
            "PostgreSQL 结构化业务ID精确交接",
            result["topology_dependency"]["identity_basis"],
        )

    def test_diagnosis_reads_component_identity_and_hands_it_to_both_tools(self):
        state = {
            "scope_route": "knowledge_base",
            "query": "A路电源损坏会造成什么后果",
            "rewritten_query": "A路电源损坏的影响",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "query_frame": {
                "entities": ["A路电源"],
                "normalized_terms": ["A路电源损坏"],
                "required_answer_fields": ["impact"],
            },
            "tool_queries": {
                "function_locator": ["A路电源", "电源"],
                "fmeca_query": ["A路电源", "电源", "Power A Failure"],
                "rbd_query": ["A路电源", "电源"],
            },
            "allowed_tools": ["function_locator", "fmeca_query", "rbd_query"],
            "sql_limit": 8,
        }
        captured = {}

        def fmeca_query(terms, _limit, _errors, *, entity_terms=()):
            captured["fmeca_terms"] = list(terms)
            captured["fmeca_entity_terms"] = list(entity_terms)
            return []

        def diagram_query(doc_type, terms, _limit, _errors, *, entity_terms=()):
            captured[f"{doc_type}_terms"] = list(terms)
            captured[f"{doc_type}_entity_terms"] = list(entity_terms)
            return []

        with (
            patch.object(
                diagnosis.icp_repository,
                "search_functions",
                return_value=[FUNCTION_ROWS[2]],
            ),
            patch.object(
                diagnosis.icp_repository,
                "search_components",
                return_value=[FUNCTION_ROWS[0]],
            ) as component_search,
            patch.object(diagnosis, "_query_fmeca", side_effect=fmeca_query),
            patch.object(diagnosis, "_query_diagrams", side_effect=diagram_query),
            patch.object(
                diagnosis,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: list(rows),
            ),
        ):
            result = asyncio.run(diagnosis.icp_diagnosis_node(state))

        component_search.assert_called_once()
        for key in (
            "fmeca_terms",
            "fmeca_entity_terms",
            "rbd_terms",
            "rbd_entity_terms",
        ):
            self.assertIn("MOD-PWR-PSUA", captured[key])
            self.assertIn("PSU-A", captured[key])
        self.assertNotIn("FUNC-210", captured["fmeca_terms"])
        function_call = next(
            call for call in result["tool_calls"] if call["tool"] == "function_locator"
        )
        self.assertEqual(
            ["MOD-PWR-PSUA"],
            function_call["structured_entity_binding"]["business_ids"],
        )


if __name__ == "__main__":
    unittest.main()
