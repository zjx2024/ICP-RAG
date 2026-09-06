import unittest
from unittest.mock import MagicMock, patch

from backend.agent.tools import icp_diagnosis as diagnosis_module


def _identity_history_binding(case_id, row_id):
    return {
        "version": 1,
        "case_ids": [case_id],
        "row_ids": [str(row_id)],
        "source": "persisted_exact_parent",
        "authority": "identity",
        "comparison": False,
        "ambiguous": False,
        "conflict": False,
        "authorizes_exact_links": True,
    }


class HistoryReferenceHydrationTests(unittest.TestCase):
    def test_only_explicitly_selected_history_parents_can_seed_multiple_references(self):
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
                    "fmeca_code": "FM-GEN-012",
                    "hmc_code": "91810002**0003",
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-012"],
                        "mfl_refs": ["MFL-GEN-012"],
                    },
                },
                {
                    "id": 13,
                    "case_id": "HIS-013",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.83,
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-013"],
                        "mfl_refs": ["MFL-GEN-013"],
                    },
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
            ],
            state={
                "query": "比较 HIS-012 与 HIS-013 两个历史案例的处理。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "比较 HIS-012 与 HIS-013 两个历史案例的处理。",
                    "question_semantics": "compare",
                },
            },
        )

        self.assertNotIn("FM-KEYWORD-MUST-NOT-EXPAND", context.get("fmeca", {}))
        self.assertNotIn("FM-WEAK-MUST-NOT-EXPAND", context.get("fmeca", {}))
        self.assertEqual(
            {"FM-GEN-012", "FM-GEN-013"}, set(context.get("fmeca", {}))
        )
        self.assertEqual(
            ["12"], context["fmeca"]["FM-GEN-012"]["history_record_ids"]
        )
        self.assertEqual(
            ["HIS-013"], context["fmeca"]["FM-GEN-013"]["history_case_ids"]
        )
        self.assertIn("91810002**0003", context["hmc"])
        self.assertEqual(
            {"MFL-GEN-012", "MFL-GEN-013"}, set(context.get("mfl", {}))
        )

    def test_competing_history_parents_fail_closed_before_reference_expansion(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 21,
                    "case_id": "HIS-GEN-021",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.91,
                    "document_links": {"mfl_refs": ["MFL-GEN-021"]},
                },
                {
                    "id": 22,
                    "case_id": "HIS-GEN-022",
                    "match_type": "structured_query",
                    "document_links": {"mfl_refs": ["MFL-GEN-022"]},
                },
            ],
            state={
                "query": "以前相近案例怎样处置？",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {"original_query": "以前相近案例怎样处置？"},
            },
        )

        self.assertEqual({}, context)

    def test_explicit_history_identity_selects_one_parent_among_competitors(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 31,
                    "case_id": "HIS-GEN-031",
                    "match_type": "structured_query",
                    "document_links": {"fmeca_refs": ["FM-GEN-031"]},
                },
                {
                    "id": 32,
                    "case_id": "HIS-GEN-032",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.98,
                    "document_links": {"fmeca_refs": ["FM-GEN-032"]},
                },
            ],
            state={
                "query": "只核对历史案例 HIS-GEN-031 的恢复方法。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "只核对历史案例 HIS-GEN-031 的恢复方法。"
                },
            },
        )

        self.assertEqual({"FM-GEN-031"}, set(context.get("fmeca", {})))

    def test_single_structured_history_parent_cannot_seed_exact_child_links(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 41,
                    "case_id": "HIS-GEN-041",
                    "match_type": "structured_query",
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-041"],
                        "mfl_refs": ["MFL-GEN-041"],
                    },
                }
            ],
            state={
                "query": "以前相近案例如何检查和恢复？",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "以前相近案例如何检查和恢复？"
                },
            },
        )

        self.assertEqual({}, context)

    def test_conflicting_current_record_id_blocks_history_family_rebinding(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 51,
                    "case_id": "HIS-GEN-051",
                    "match_type": "structured_query",
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-051"],
                        "mfl_refs": ["MFL-GEN-051"],
                    },
                }
            ],
            state={
                "query": "核对 FM-GEN-052 和 MFL-GEN-052，并参考以前的案例。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "核对 FM-GEN-052 和 MFL-GEN-052，并参考以前的案例。"
                },
            },
        )

        self.assertNotIn("fmeca", context)
        self.assertNotIn("mfl", context)

    def test_persisted_parent_binding_beats_higher_scoring_semantic_neighbor(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 61,
                    "case_id": "HIS-GEN-061",
                    "match_type": "structured_query",
                    "retrieval_match_score": 0.31,
                    "fault_symptom": "外部源持续发送，但输入接收计数不变，旁路接口正常",
                    "root_cause": "连接端接异常导致接收方向开路",
                    "final_solution": "恢复连接与端接后复测输入计数",
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-061"],
                        "mfl_refs": ["MFL-GEN-061"],
                    },
                },
                {
                    "id": 62,
                    "case_id": "HIS-GEN-062",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.97,
                    "fault_symptom": "任务处理中图形画面停止更新并触发心跳超时",
                    "root_cause": "内部调度互锁",
                    "final_solution": "替换处理单元后恢复画面",
                    "document_links": {
                        "fmeca_refs": ["FM-GEN-062"],
                        "mfl_refs": ["MFL-GEN-062"],
                    },
                },
            ],
            state={
                "query": "外部源发送后输入接收计数不变，但旁路接口正常；应替换接口还是检查连接端接？",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "相近案例在什么条件下出现，最后怎样验证恢复？",
                    "rewritten_query": "外部源发送后输入接收计数不变，但旁路接口正常；应替换接口还是检查连接端接？",
                },
                "history_parent_binding": _identity_history_binding(
                    "HIS-GEN-061", 61
                ),
                "selected_conversation_anchors": [{
                    "doc_type": "fault_history",
                    "record_id": "HIS-GEN-061",
                }],
            },
        )

        self.assertEqual({"FM-GEN-061"}, set(context.get("fmeca", {})))
        self.assertEqual({"MFL-GEN-061"}, set(context.get("mfl", {})))

    def test_selected_history_anchor_wins_vague_followup(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 71,
                    "case_id": "HIS-GEN-071",
                    "match_type": "structured_query",
                    "document_links": {"mfl_refs": ["MFL-GEN-071"]},
                },
                {
                    "id": 72,
                    "case_id": "HIS-GEN-072",
                    "match_type": "semantic_id_hydration",
                    "retrieval_match_score": 0.99,
                    "document_links": {"mfl_refs": ["MFL-GEN-072"]},
                },
            ],
            state={
                "query": "这个案例最终如何验证？",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {"original_query": "这个案例最终如何验证？"},
                "selected_conversation_anchors": [
                    {"doc_type": "fault_history", "record_id": "HIS-GEN-071"}
                ],
            },
        )

        self.assertEqual({"MFL-GEN-071"}, set(context.get("mfl", {})))

    def test_multiple_explicit_case_ids_need_compare_semantics(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 81,
                    "case_id": "HIS-GEN-081",
                    "match_type": "structured_query",
                    "document_links": {"fmeca_refs": ["FM-GEN-081"]},
                },
                {
                    "id": 82,
                    "case_id": "HIS-GEN-082",
                    "match_type": "structured_query",
                    "document_links": {"fmeca_refs": ["FM-GEN-082"]},
                },
            ],
            state={
                "query": "核对 HIS-GEN-081 和 HIS-GEN-082 的恢复记录。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "核对 HIS-GEN-081 和 HIS-GEN-082 的恢复记录。",
                    "question_semantics": "verify",
                },
            },
        )

        self.assertEqual({}, context)

    def test_single_history_row_does_not_expand_without_current_history_target(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 91,
                    "case_id": "HIS-GEN-091",
                    "match_type": "structured_query",
                    "document_links": {"mfl_refs": ["MFL-GEN-091"]},
                }
            ],
            state={
                "query": "说明当前故障模式的原因。",
                "query_targets": [{"intent": "cause_analysis"}],
                "query_frame": {"original_query": "说明当前故障模式的原因。"},
            },
        )

        self.assertEqual({}, context)

    def test_unknown_current_history_id_fails_closed(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 101,
                    "case_id": "HIS-GEN-101",
                    "match_type": "structured_query",
                    "document_links": {"mfl_refs": ["MFL-GEN-101"]},
                }
            ],
            state={
                "query": "只查 HIS-GEN-199 的处理。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {"original_query": "只查 HIS-GEN-199 的处理。"},
            },
        )

        self.assertEqual({}, context)

    def test_fm_and_fmeca_prefixes_share_one_conflict_family(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 111,
                    "case_id": "HIS-GEN-111",
                    "match_type": "structured_query",
                    "document_links": {"fmeca_refs": ["FMECA-GEN-111"]},
                }
            ],
            state={
                "query": "核对 FM-GEN-112，并参考以前的案例。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "核对 FM-GEN-112，并参考以前的案例。"
                },
            },
        )

        self.assertNotIn("fmeca", context)

    def test_equivalent_hmc_format_does_not_create_false_conflict(self):
        context = diagnosis_module._semantic_history_reference_context(
            [
                {
                    "id": 121,
                    "case_id": "HIS-GEN-121",
                    "match_type": "structured_query",
                    "hmc_code": "91810002**0003",
                }
            ],
            state={
                "query": "参考历史记录核对 HMC 91810002-**-0003。",
                "query_targets": [{"intent": "history_query"}],
                "query_frame": {
                    "original_query": "参考历史记录核对 HMC 91810002-**-0003。"
                },
                "history_parent_binding": _identity_history_binding(
                    "HIS-GEN-121", 121
                ),
                "selected_conversation_anchors": [{
                    "doc_type": "fault_history",
                    "record_id": "HIS-GEN-121",
                }, {
                    "doc_type": "hmc",
                    "record_id": "91810002**0003",
                    "answer_fields": {"hmc_code": "91810002**0003"},
                }],
            },
        )

        self.assertEqual({"91810002**0003"}, set(context.get("hmc", {})))

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
    async def test_explicit_case_id_duplicate_row_keeps_handoff_and_expands(self):
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
            "query": "针对历史案例 HIS-012，核对原因和之前怎么处理的？",
            "rewritten_query": "针对历史案例 HIS-012，核对原因和之前怎么处理的？",
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
                {"intent": "history_query"},
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
