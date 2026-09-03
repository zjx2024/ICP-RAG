import unittest
from unittest.mock import MagicMock, patch

from backend.agent.tools import icp_diagnosis as diagnosis_module


class HistoryReferenceHydrationTests(unittest.TestCase):
    def test_only_semantic_identity_hydrated_history_can_seed_references(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 11,
                    "case_id": "HIS-011",
                    "match_type": "keyword",
                    "retrieval_match_score": 0.99,
                    "fmeca_code": "FM-KEYWORD-MUST-NOT-EXPAND",
                },
                {
                    "id": 12,
                    "case_id": "HIS-012",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.61,
                    "fmeca_code": "FM-SYN-006",
                    "hmc_code": "91810002**0003",
                    "document_links": {
                        "fmeca_refs": ["FM-SYN-006"],
                        "mfl_refs": ["MFL-SYN-006"],
                    },
                },
                {
                    "id": 13,
                    "case_id": "HIS-013",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.83,
                    "document_links": {"fmeca_refs": ["FM-SYN-006"]},
                },
                {
                    "id": 14,
                    "case_id": "HIS-014",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.15,
                    "document_links": {
                        "fmeca_refs": ["FM-WEAK-MUST-NOT-EXPAND"]
                    },
                },
            ]
        )

        self.assertNotIn("FM-KEYWORD-MUST-NOT-EXPAND", context.get("fmeca", {}))
        self.assertNotIn("FM-WEAK-MUST-NOT-EXPAND", context.get("fmeca", {}))
        self.assertEqual(0.83, context["fmeca"]["FM-SYN-006"]["retrieval_match_score"])
        self.assertEqual(
            ["12", "13"],
            context["fmeca"]["FM-SYN-006"]["history_record_ids"],
        )
        self.assertEqual(
            ["HIS-012", "HIS-013"],
            context["fmeca"]["FM-SYN-006"]["history_case_ids"],
        )
        self.assertIn("91810002**0003", context["hmc"])
        self.assertIn("MFL-SYN-006", context["mfl"])

    def test_fmeca_history_reference_uses_bound_code_and_carries_parent_score(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.__exit__.return_value = False
        cursor.description = [("id",), ("code",), ("fault_mode",), ("doc_id",)]
        cursor.fetchall.return_value = [
            (6, "FM-SYN-006", "PSU-A 输出中断", "doc-fmeca")
        ]
        connection = MagicMock()
        connection.cursor.return_value = cursor
        context = {
            "FM-SYN-006": {
                "retrieval_match_score": 0.83,
                "history_record_ids": ["13"],
                "history_case_ids": ["HIS-013"],
            }
        }

        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            rows = diagnosis_module._query_fmeca_by_codes(
                ["FM-SYN-006"],
                8,
                match_type="exact_history_reference",
                reference_context=context,
            )

        sql, params = cursor.execute.call_args.args
        self.assertIn("f.code = ANY(%s)", sql)
        self.assertNotIn("FM-SYN-006", sql)
        self.assertEqual((["FM-SYN-006"], 8), params)
        self.assertEqual("exact_history_reference", rows[0]["match_type"])
        self.assertEqual(0.83, rows[0]["retrieval_match_score"])
        self.assertEqual("fault_history", rows[0]["reference_source"])
        self.assertEqual(["HIS-013"], rows[0]["linked_history_case_ids"])

    def test_hmc_and_mfl_history_references_are_exact_parameterized_queries(self):
        cases = (
            (
                "hmc",
                "hmc_code",
                "91810002**0003",
                (31, "91810002**0003", "PSU-A 输出中断", "doc-hmc"),
                [("id",), ("hmc_code",), ("fault_desc",), ("doc_id",)],
            ),
            (
                "mfl",
                "mfl_code",
                "MFL-SYN-006",
                (41, "MFL-SYN-006", "检查 PSU-A", "doc-mfl"),
                [("id",), ("mfl_code",), ("maintenance_action",), ("doc_id",)],
            ),
        )
        for ref_type, code_column, code, db_row, description in cases:
            with self.subTest(ref_type=ref_type):
                cursor = MagicMock()
                cursor.__enter__.return_value = cursor
                cursor.__exit__.return_value = False
                cursor.description = description
                cursor.fetchall.return_value = [db_row]
                connection = MagicMock()
                connection.cursor.return_value = cursor
                context = {
                    code: {
                        "retrieval_match_score": "0.74",
                        "history_record_ids": ["21"],
                        "history_case_ids": ["HIS-021"],
                    }
                }

                with patch.object(diagnosis_module.postgres_client, "conn", connection):
                    rows = diagnosis_module._query_history_linked_rows(
                        ref_type, context, 5
                    )

                sql, params = cursor.execute.call_args.args
                self.assertIn(f"t.{code_column} = ANY(%s)", sql)
                self.assertNotIn(code, sql)
                self.assertEqual(([code], 5), params)
                self.assertEqual(code, rows[0][code_column])
                self.assertEqual("exact_history_reference", rows[0]["match_type"])
                self.assertEqual("0.74", rows[0]["retrieval_match_score"])
                self.assertEqual(["21"], rows[0]["linked_history_record_ids"])

    def test_linked_score_and_history_provenance_reach_direct_evidence(self):
        rows_by_tool = {
            "hmc_query": {
                "doc_id": "doc-hmc",
                "hmc_code": "91810002**0003",
                "fault_desc": "PSU-A 输出中断",
                "repair_method": "检查 PSU-A 供电支路",
            },
            "fmeca_query": {
                "doc_id": "doc-fmeca",
                "code": "FM-SYN-006",
                "fault_mode": "PSU-A 输出中断",
                "fault_cause": "PSU-A 支路开路",
            },
            "mfl_query": {
                "doc_id": "doc-mfl",
                "mfl_code": "MFL-SYN-006",
                "maintenance_action": "检查并恢复 PSU-A 支路",
            },
        }
        for tool, row in rows_by_tool.items():
            with self.subTest(tool=tool):
                row.update(
                    {
                        "match_type": "exact_history_reference",
                        "retrieval_match_score": 0.83,
                        "reference_source": "fault_history",
                        "linked_history_record_ids": ["13"],
                        "linked_history_case_ids": ["HIS-013"],
                    }
                )
                evidence = diagnosis_module._direct_evidence(tool, [row])
                self.assertTrue(evidence)
                for item in evidence:
                    self.assertEqual(0.83, item["retrieval_score"])
                    self.assertEqual(
                        "fault_history", item["metadata"]["reference_source"]
                    )
                    self.assertEqual(
                        ["HIS-013"],
                        item["metadata"]["linked_history_case_ids"],
                    )


class HistoryReferenceNodeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_literal_history_row_keeps_semantic_handoff_and_expands(self):
        history_row = {
            "id": 12,
            "doc_id": "doc-history",
            "case_id": "HIS-012",
            "fault_symptom": "PSU-A 支路间歇中断",
            "fmeca_code": "FM-SYN-006",
            "hmc_code": "91810002**0003",
            "mfl_code": "MFL-SYN-006",
        }
        hydrated_history = {
            **history_row,
            "match_type": "semantic_id_hydration",
            "retrieval_match_score": 0.83,
            "retrieval_source_chunk_id": "history-chunk-12",
        }
        hydrated_records = {
            "function_locator": [],
            "hmc_query": [],
            "fmeca_query": [],
            "mfl_query": [],
            "fault_history_query": [hydrated_history],
        }
        linked_fmeca = {
            "id": 6,
            "doc_id": "doc-fmeca",
            "code": "FM-SYN-006",
            "fault_mode": "PSU-A 输出中断",
            "fault_cause": "PSU-A 支路开路",
            "match_type": "exact_history_reference",
            "retrieval_match_score": 0.83,
            "reference_source": "fault_history",
            "linked_history_record_ids": ["12"],
            "linked_history_case_ids": ["HIS-012"],
        }

        def history_linked(ref_type, *_args, **_kwargs):
            if ref_type == "hmc":
                return [
                    {
                        "id": 7,
                        "doc_id": "doc-hmc",
                        "hmc_code": "91810002**0003",
                        "fault_desc": "PSU-A 输出中断",
                        "repair_method": "检查 PSU-A 支路",
                        "match_type": "exact_history_reference",
                        "retrieval_match_score": 0.83,
                        "reference_source": "fault_history",
                        "linked_history_record_ids": ["12"],
                        "linked_history_case_ids": ["HIS-012"],
                    }
                ]
            return [
                {
                    "id": 8,
                    "doc_id": "doc-mfl",
                    "mfl_code": "MFL-SYN-006",
                    "maintenance_action": "恢复 PSU-A 支路",
                    "match_type": "exact_history_reference",
                    "retrieval_match_score": 0.83,
                    "reference_source": "fault_history",
                    "linked_history_record_ids": ["12"],
                    "linked_history_case_ids": ["HIS-012"],
                }
            ]

        locator = {
            "doc_type_kwd": "fault_history",
            "doc_id": "doc-history",
            "pg_id_kwd": "12",
            "chunk_id": "history-chunk-12",
            "content": "PSU-A 支路间歇中断",
            "score": 0.83,
        }
        state = {
            "query": "A 路电源偶尔断掉，原因和之前怎么处理的？",
            "rewritten_query": "A 路电源偶尔断掉，原因和之前怎么处理的？",
            "intent": "cause_analysis",
            "allowed_tools": [
                "hmc_query",
                "fmeca_query",
                "mfl_query",
                "fault_history_query",
                "hybrid_search",
            ],
            "hybrid_doc_types": ["fault_history", "fmeca", "hmc", "mfl"],
            "normalized_terms": ["A 路电源", "间歇中断"],
            "query_targets": [
                {"intent": "cause_analysis"},
                {"intent": "history_lookup"},
                {"intent": "maintenance_guidance"},
            ],
        }

        with (
            patch.object(diagnosis_module, "_query_hmc", return_value=[]),
            patch.object(diagnosis_module, "_query_fmeca", return_value=[]),
            patch.object(diagnosis_module, "_query_mfl", return_value=[]),
            patch.object(
                diagnosis_module.icp_repository,
                "search_history",
                return_value=[dict(history_row)],
            ),
            patch.object(diagnosis_module, "_hybrid_search", return_value=([locator], {})),
            patch.object(
                diagnosis_module,
                "_hydrate_record_search_hits",
                return_value=hydrated_records,
            ),
            patch.object(
                diagnosis_module,
                "_query_fmeca_by_codes",
                return_value=[linked_fmeca],
            ) as fmeca_lookup,
            patch.object(
                diagnosis_module,
                "_query_history_linked_rows",
                side_effect=history_linked,
            ),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: list(rows),
            ),
        ):
            result = await diagnosis_module.icp_diagnosis_node(state)

        fmeca_lookup.assert_called_once()
        self.assertEqual(
            "exact_history_reference", fmeca_lookup.call_args.kwargs["match_type"]
        )
        record_ids = {item["record_id"] for item in result["evidence_registry"]}
        self.assertTrue(
            {"HIS-012", "FM-SYN-006", "91810002**0003", "MFL-SYN-006"}
            <= record_ids
        )
        history_evidence = next(
            item
            for item in result["evidence_registry"]
            if item["record_id"] == "HIS-012"
        )
        self.assertEqual("semantic_id_hydration", history_evidence["match_type"])
        calls = {item["tool"]: item for item in result["tool_calls"]}
        self.assertEqual(1, calls["fmeca_query"]["history_linked_count"])
        self.assertEqual(1, calls["hmc_query"]["history_linked_count"])
        self.assertEqual(1, calls["mfl_query"]["history_linked_count"])


if __name__ == "__main__":
    unittest.main()
