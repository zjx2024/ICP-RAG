import asyncio
import unittest
from unittest.mock import patch

from backend.agent.nodes import adaptive_tools
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.tool_policy import resolve_tool_policy
from backend.agent.tools import icp_diagnosis as diagnosis


def _run(coro):
    return asyncio.run(coro)


def _row(code, mode, cause, function=""):
    return {
        "id": code,
        "doc_id": "doc-fmeca",
        "code": code,
        "fault_mode": mode,
        "fault_cause": cause,
        "func_management": function,
        "local_impact": f"{mode}的局部影响",
        "final_impact": f"{mode}的最终影响",
        "detect_method": f"检查{mode}",
    }


class FmecaRecordIdentityBindingTests(unittest.TestCase):
    def test_partial_literal_overlap_is_relevance_not_record_identity(self):
        rows = [
            _row(
                "FM-A",
                "接收链路性能退化",
                "接收连接器接触阻抗上升",
            ),
            _row(
                "FM-B",
                "发送队列拥塞",
                "发送缓存配置不当",
            ),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "接收链路正在慢慢变差，会先影响什么？",
            rows,
        )

        self.assertFalse(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertTrue(binding["relevance_only"])
        self.assertEqual([], binding["record_ids"])
        self.assertNotIn("match_type", bound_rows[0])
        self.assertNotIn("query_binding_excluded", bound_rows[1])
        self.assertEqual(
            "接收链路",
            binding["candidates"][0]["matches"][0]["span"],
        )

    def test_multiple_partial_fields_still_do_not_create_identity(self):
        rows = [
            _row(
                "FM-A",
                "入口滤波性能退化",
                "滤波器件容量下降",
            ),
            _row(
                "FM-B",
                "入口滤波旁路失效",
                "旁路开关触点粘连",
            ),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "入口滤波器件逐渐变差后为什么会波动？",
            rows,
        )

        self.assertFalse(binding["bound"])
        self.assertTrue(binding["relevance_only"])
        self.assertEqual(2, binding["candidates"][0]["rank"][1])
        focused = diagnosis._fmeca_rows_for_bound_query(bound_rows, binding)
        self.assertEqual(["FM-A", "FM-B"], [row["code"] for row in focused])

    def test_equal_partial_spans_remain_open_relevance_candidates(self):
        rows = [
            _row("FM-A", "接口链路性能退化", "接插件污染"),
            _row("FM-B", "接口链路性能退化", "收发器偏置异常"),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "接口链路变差以后有什么后果？",
            rows,
        )

        self.assertFalse(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertTrue(binding["relevance_only"])
        self.assertEqual([], binding["record_ids"])
        self.assertTrue(
            all("query_binding_excluded" not in row for row in bound_rows)
        )
        self.assertEqual(
            ["FM-A", "FM-B"],
            [
                row["code"]
                for row in diagnosis._fmeca_rows_for_bound_query(
                    bound_rows, binding
                )
            ],
        )

    def test_typed_complete_field_equality_binds_one_authoritative_row(self):
        rows = [
            _row("FM-A", "接收链路性能退化", "接插件污染"),
            _row("FM-B", "发送队列拥塞", "缓存配置不当"),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "接收链路性能退化会带来什么影响？",
            rows,
            typed_fields={"failure_mode": "接收链路性能退化"},
        )

        self.assertTrue(binding["bound"])
        self.assertEqual("exact_typed_field_equality", binding["basis"])
        self.assertEqual(["FM-A"], binding["record_ids"])
        self.assertEqual("exact_record_identity", bound_rows[0]["match_type"])
        self.assertTrue(bound_rows[1]["query_binding_excluded"])

    def test_typed_value_not_grounded_in_current_question_cannot_bind(self):
        rows = [
            _row("FM-A", "接收链路性能退化", "接插件污染"),
            _row("FM-B", "发送队列拥塞", "缓存配置不当"),
        ]

        _bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "它会带来什么影响？",
            rows,
            typed_fields={"failure_mode": "接收链路性能退化"},
        )

        self.assertFalse(binding["bound"])
        self.assertEqual("typed_field_not_grounded", binding["basis"])

    def test_explicit_record_code_is_stable_identity(self):
        rows = [
            _row("FM-A", "接收链路性能退化", "接插件污染"),
            _row("FM-B", "发送队列拥塞", "缓存配置不当"),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "请说明 FM-B 的影响",
            rows,
        )

        self.assertTrue(binding["bound"])
        self.assertEqual("exact_record_code", binding["basis"])
        self.assertEqual(["FM-B"], binding["record_ids"])
        self.assertTrue(bound_rows[0]["query_binding_excluded"])

    def test_short_generic_substring_cannot_authorize_record_identity(self):
        rows = [
            _row("FM-A", "驱动机构卡滞", "轴承磨损"),
            _row("FM-B", "转台速度漂移", "轴承润滑不足"),
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "轴承异常会怎样？",
            rows,
        )

        self.assertFalse(binding["bound"])
        self.assertFalse(binding["ambiguous"])
        self.assertTrue(all("match_type" not in row for row in bound_rows))

    def test_deictic_followup_keeps_the_reread_conversation_record(self):
        rows = [
            {
                **_row("FM-A", "接收链路性能退化", "接插件污染"),
                "match_type": "exact_conversation_reference",
            },
            {
                **_row("FM-B", "发送队列拥塞", "缓存配置不当"),
                "match_type": "semantic_id_hydration",
            },
        ]

        bound_rows, binding = diagnosis._bind_unique_fmeca_record_identity(
            "那它应当怎么验证？",
            rows,
        )

        self.assertTrue(binding["bound"])
        self.assertEqual("exact_conversation_reference", binding["basis"])
        self.assertEqual(["FM-A"], binding["record_ids"])
        self.assertEqual(
            ["FM-A"],
            [
                row["code"]
                for row in diagnosis._fmeca_rows_for_bound_query(
                    bound_rows, binding
                )
            ],
        )

    def test_partial_overlap_cannot_authorize_unique_fact_or_graph_request(self):
        rows = [
            _row("FM-A", "接收链路性能退化", "接收连接器接触阻抗上升"),
            _row("FM-B", "发送链路性能退化", "发送连接器接触阻抗上升"),
        ]
        state = {
            "intent": "impact_analysis",
            "query": "接收链路慢慢变差后会带来什么影响？",
            "rewritten_query": "链路退化影响",
            "query_targets": [{"intent": "impact_analysis"}],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["local_impact", "final_impact"],
            },
            "allowed_tools": ["fmeca_query", "graph_traversal", "evidence_synthesis"],
            "authorized_tools": ["fmeca_query", "graph_traversal", "evidence_synthesis"],
        }
        with (
            patch.object(diagnosis, "_query_fmeca", return_value=rows),
            patch.object(
                diagnosis,
                "_enrich_function_sources",
                side_effect=lambda values, _errors=None: list(values),
            ),
        ):
            result = _run(diagnosis.icp_diagnosis_node(state))

        self.assertEqual([], result["structured_record_binding"]["record_ids"])
        self.assertTrue(result["structured_record_binding"]["relevance_only"])
        fmeca_evidence = [
            item for item in result["evidence_registry"]
            if item.get("doc_type") == "fmeca"
        ]
        self.assertEqual(2, len(fmeca_evidence))
        self.assertEqual(
            {"接收链路性能退化", "发送链路性能退化"},
            {item["fault"] for item in result["candidate_faults"]},
        )
        fmeca_seeds = [
            item for item in result["graph_seed_bindings"]
            if item.get("source_tool") == "fmeca_query"
        ]
        self.assertEqual(
            {"FM-A", "FM-B"}, {item["business_id"] for item in fmeca_seeds}
        )
        self.assertTrue(
            all(item["binding_status"] == "candidate" for item in fmeca_seeds)
        )
        self.assertEqual({}, result["graph_traversal_request"])

        gate = _run(evidence_gate_node({**state, **result}))
        target_id = next(
            item["evidence_id"]
            for item in fmeca_evidence
            if item.get("record_id") == "FM-A"
        )
        neighbour_id = next(
            item["evidence_id"]
            for item in fmeca_evidence
            if item.get("record_id") == "FM-B"
        )
        self.assertIn(target_id, gate["context_evidence_ids"])
        self.assertIn(neighbour_id, gate["context_evidence_ids"])

    def test_ambiguous_rows_remain_candidates_and_require_disambiguation(self):
        rows = [
            _row("FM-A", "接口链路性能退化", "接插件污染"),
            _row("FM-B", "接口链路性能退化", "收发器偏置异常"),
        ]
        state = {
            "intent": "impact_analysis",
            "query": "接口链路性能退化以后有什么后果？",
            "rewritten_query": "接口链路性能退化的影响",
            "query_targets": [{"intent": "impact_analysis"}],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["local_impact", "final_impact"],
                "typed_record_fields": {
                    "fmeca": {"failure_mode": "接口链路性能退化"}
                },
            },
            "allowed_tools": [
                "fmeca_query",
                "graph_traversal",
                "evidence_synthesis",
            ],
            "authorized_tools": [
                "fmeca_query",
                "graph_traversal",
                "evidence_synthesis",
            ],
        }
        with (
            patch.object(diagnosis, "_query_fmeca", return_value=rows),
            patch.object(
                diagnosis,
                "_enrich_function_sources",
                side_effect=lambda values, _errors=None: list(values),
            ),
        ):
            result = _run(diagnosis.icp_diagnosis_node(state))

        self.assertTrue(result["structured_record_binding"]["ambiguous"])
        self.assertEqual(
            {"FM-A", "FM-B"},
            {
                item["business_id"]
                for item in result["graph_seed_bindings"]
                if item.get("source_tool") == "fmeca_query"
            },
        )
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertEqual(
            {"接口链路性能退化"},
            {item["fault"] for item in result["candidate_faults"]},
        )
        self.assertEqual("", result["clarification_question"])
        self.assertEqual(
            ["fmeca_record_identity"],
            [item["slot"] for item in result["unresolved_identity_slots"]],
        )
        self.assertIn(
            "ambiguous_identity",
            {item["kind"] for item in result["clarification_reasons"]},
        )
        gate = _run(evidence_gate_node({**state, **result}))
        self.assertEqual("clarify", gate["safety_decision"]["status"])
        self.assertIn(
            "exact_record_identity",
            gate["safety_decision"]["blocked_claims"],
        )


class FmecaAdaptiveBindingMergeTests(unittest.TestCase):
    def test_empty_later_batch_cannot_erase_relevance_only_graph_boundary(self):
        relevance = {
            "doc_type": "fmeca",
            "bound": False,
            "ambiguous": False,
            "relevance_only": True,
            "basis": "literal_field_relevance_only",
            "record_ids": [],
            "candidates": [{"record_id": "FM-A", "row_index": 0}],
        }
        empty = {
            "doc_type": "fmeca",
            "bound": False,
            "ambiguous": False,
            "relevance_only": False,
            "basis": "",
            "record_ids": [],
            "candidates": [],
        }

        merged = adaptive_tools._merge_structured_record_binding(
            relevance, empty
        )
        eligible = adaptive_tools._eligible_graph_seed_bindings(
            {
                "intent": "impact_analysis",
                "query_targets": [{"intent": "impact_analysis"}],
                "structured_record_binding": merged,
                "graph_seed_bindings": [
                    {
                        "logical_id": "fmeca:FM-A",
                        "source_tool": "fmeca_query",
                        "binding_status": "candidate",
                    }
                ],
            }
        )

        self.assertTrue(merged["relevance_only"])
        self.assertEqual([], eligible)

    def test_later_subset_cannot_resolve_an_earlier_ambiguity(self):
        ambiguous = {
            "doc_type": "fmeca",
            "bound": False,
            "ambiguous": True,
            "basis": "ambiguous_structured_field_identity",
            "record_ids": ["FM-A", "FM-B"],
            "candidates": [
                {"record_id": "FM-A", "basis": "structured_field"},
                {"record_id": "FM-B", "basis": "structured_field"},
            ],
        }
        subset = {
            "doc_type": "fmeca",
            "bound": True,
            "ambiguous": False,
            "basis": "unique_high_specificity_structured_field",
            "record_ids": ["FM-A"],
        }

        merged = adaptive_tools._merge_structured_record_binding(
            ambiguous, subset
        )

        self.assertFalse(merged["bound"])
        self.assertTrue(merged["ambiguous"])
        self.assertEqual(
            {"FM-A", "FM-B"},
            {
                record_id
                for item in merged["candidates"]
                for record_id in item["record_ids"]
            },
        )

    def test_new_ambiguity_cannot_overwrite_an_existing_exact_target(self):
        exact = {
            "doc_type": "fmeca",
            "bound": True,
            "ambiguous": False,
            "basis": "exact_record_code",
            "record_ids": ["FM-A"],
        }
        ambiguous = {
            "doc_type": "fmeca",
            "bound": False,
            "ambiguous": True,
            "basis": "ambiguous_structured_field_identity",
            "record_ids": ["FM-A", "FM-B"],
            "candidates": [
                {"record_id": "FM-A"},
                {"record_id": "FM-B"},
            ],
        }

        merged = adaptive_tools._merge_structured_record_binding(
            exact, ambiguous
        )

        self.assertFalse(merged["bound"])
        self.assertTrue(merged["ambiguous"])

    def test_ambiguous_record_candidates_cannot_authorize_graph_traversal(self):
        eligible = adaptive_tools._eligible_graph_seed_bindings(
            {
                "intent": "impact_analysis",
                "query_targets": [{"intent": "impact_analysis"}],
                "structured_record_binding": {
                    "doc_type": "fmeca",
                    "bound": False,
                    "ambiguous": True,
                },
                "graph_seed_bindings": [
                    {
                        "logical_id": "fmeca:FM-A",
                        "source_tool": "fmeca_query",
                        "binding_status": "exact",
                    },
                    {
                        "logical_id": "fmeca:FM-B",
                        "source_tool": "fmeca_query",
                        "binding_status": "candidate",
                    },
                ],
            }
        )

        self.assertEqual([], eligible)


class FmecaFollowupRereadTests(unittest.TestCase):
    def test_policy_forces_selected_anchor_owner_and_required_maintenance_source(self):
        decision = resolve_tool_policy(
            intent="maintenance_query",
            proposed_plan=[{"tool": "mfl_query", "reason": "维修定位"}],
            query="如何验证并处置？",
            raw_user_query="如何验证并处置？",
            query_targets=[{"intent": "maintenance_query"}],
            coverage_scope="focused_detail",
            required_answer_fields=["detection_method", "maintenance_action"],
            diagnostic_closure={"required": True},
            selected_anchor_doc_types=["fmeca"],
        )

        self.assertIn("fmeca_query", decision["allowed_tools"])
        self.assertIn("mfl_query", decision["allowed_tools"])
        self.assertTrue(
            any(
                item["tool"] == "fmeca_query"
                and "同一权威记录" in item["reason"]
                for item in decision["tools_plan"]
            )
        )

    def test_followup_rereads_selected_fmeca_code_instead_of_old_prose(self):
        fmeca = _row(
            "FM-A",
            "接收链路性能退化",
            "接收连接器接触阻抗上升",
        )
        mfl = {
            "id": "MFL-A",
            "doc_id": "doc-mfl",
            "mfl_code": "MFL-A",
            "module_fault": "接收单元：接收链路性能退化",
            "maintenance_action": "检查连接后再决定更换",
        }
        state = {
            "intent": "maintenance_query",
            "query": "这种情况下如何验证并处置？",
            "rewritten_query": "验证并处置当前对象",
            "query_targets": [{"intent": "maintenance_query"}],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["detection_method", "maintenance_action"],
            },
            "selected_conversation_anchors": [
                {"doc_type": "fmeca", "doc_id": fmeca["doc_id"], "record_id": "FM-A"}
            ],
            "allowed_tools": ["fmeca_query", "mfl_query", "evidence_synthesis"],
            "authorized_tools": ["fmeca_query", "mfl_query", "evidence_synthesis"],
        }
        with (
            patch.object(
                diagnosis.icp_repository,
                "get_records_by_business_selectors",
                return_value={"rows": [
                    {**fmeca, "match_type": "exact_conversation_reference"}
                ], "reads": []},
            ) as fmeca_lookup,
            patch.object(diagnosis, "_query_mfl", return_value=[mfl]) as mfl_lookup,
            patch.object(
                diagnosis,
                "_enrich_function_sources",
                side_effect=lambda values, _errors=None: list(values),
            ),
        ):
            result = _run(diagnosis.icp_diagnosis_node(state))

        fmeca_lookup.assert_called_once()
        self.assertEqual(("fmeca", [{"doc_id": fmeca["doc_id"], "record_id": "FM-A"}]), fmeca_lookup.call_args.args)
        mfl_lookup.assert_called_once()
        exact_rows = [
            item for item in result["evidence_registry"]
            if item.get("record_id") == "FM-A"
        ]
        self.assertTrue(exact_rows)
        self.assertTrue(
            all(item["match_type"] == "exact_conversation_reference" for item in exact_rows)
        )


if __name__ == "__main__":
    unittest.main()
