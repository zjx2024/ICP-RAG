"""HMC/MFL 身份边界与 RBD 结构补引的回归测试。"""
import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.rbd_contract import bind_rbd_models, selected_rbd_anchor_selectors
from backend.agent.tools import icp_diagnosis as icp_module


def _run(coro):
    return asyncio.run(coro)






class HmcTypedIdentityTests(unittest.TestCase):
    def test_current_code_correction_supersedes_selected_old_code(self):
        old_anchor = {"anchor_id": "old", "doc_type": "hmc", "doc_id": "doc-controller", "record_id": "73120408**0007"}
        for query in ("现在查 63120408**0009", "现在查 LCN 63120408 HSC 9"):
            with self.subTest(query=query), patch.object(icp_module, "_query_hmc", return_value=[]) as current, patch.object(icp_module.icp_repository, "get_records_by_business_selectors") as previous:
                _run(icp_module.icp_diagnosis_node({
                    "query": query, "allowed_tools": ["hmc_query"],
                    "selected_conversation_anchors": [old_anchor],
                }))
                previous.assert_not_called()
                self.assertEqual(query, current.call_args.args[0])

    @staticmethod
    def _stored_row(hamc="**"):
        return {
            "id": hamc, "doc_id": "doc-controller",
            "lcn": "73120408", "hamc": hamc, "hsc": "0007",
            "hmc_code": f"73120408{hamc}0007", "fault_desc": "记录的异常",
            "repair_method": "记录的检查方法",
        }

    def test_limit_one_still_detects_ambiguous_partial_code(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.description = [(key,) for key in self._stored_row()]
        cursor.fetchall.return_value = [tuple(self._stored_row().values()), tuple(self._stored_row("02").values())]
        with patch.object(icp_module.postgres_client, "conn", connection):
            rows = icp_module._query_hmc("LCN 73120408 HSC 7", [], 1)
        self.assertEqual(("73120408", "0007", 2), cursor.execute.call_args.args[1])
        self.assertEqual({"ambiguous_components"}, {row["match_type"] for row in rows})
        evidence = icp_module._direct_evidence("hmc_query", rows)
        self.assertEqual(2, sum(item["claim_type"] == "hmc_fault" for item in evidence))
        decision = _run(evidence_gate_node({
            "query": "LCN 73120408 HSC 7", "allowed_tools": ["hmc_query"],
            "evidence_registry": evidence,
        }))["safety_decision"]
        self.assertIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertEqual(["hamc"], [item["slot"] for item in decision["unresolved_identity_slots"]])
        self.assertIn("ambiguous_identity", [item["kind"] for item in decision["clarification_reasons"]])

    def test_unique_partial_code_preserves_postgres_hamc(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.description = [(key,) for key in self._stored_row()]
        cursor.fetchall.return_value = [tuple(self._stored_row("02").values())]
        with patch.object(icp_module.postgres_client, "conn", connection):
            rows = icp_module._query_hmc("核对 LCN 73120408 HSC 0007", [], 1)
        self.assertEqual("exact_components", rows[0]["match_type"])
        self.assertEqual("02", rows[0]["hmc_component_selector"]["resolved_hamc"])
        decision = _run(evidence_gate_node({
            "query": "LCN 73120408 HSC 0007", "allowed_tools": ["hmc_query"],
            "evidence_registry": icp_module._direct_evidence("hmc_query", rows),
        }))["safety_decision"]
        self.assertNotIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertEqual([], decision["unresolved_identity_slots"])

    def test_selected_code_is_document_scoped_and_reread_before_neighbor_filter(self):
        state = {
            "query": "这条怎么处理？", "allowed_tools": ["hmc_query"],
            "selected_conversation_anchors": [{
                "anchor_id": "selected-code", "doc_type": "hmc",
                "doc_id": "doc-controller", "record_id": "73120408**0007",
            }],
        }
        selectors = icp_module._conversation_record_selectors(state, "hmc")
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.description = [(key,) for key in self._stored_row()]
        cursor.fetchall.return_value = [tuple(self._stored_row().values())]
        with patch.object(icp_module.postgres_client, "conn", connection):
            rows = icp_module.icp_repository.get_records_by_business_selectors("hmc", selectors)["rows"]
        self.assertEqual(("doc-controller", "73120408", "**", "0007"), cursor.execute.call_args.args[1])
        self.assertEqual("exact_conversation_reference", rows[0]["match_type"])
        neighbor = {**self._stored_row("02"), "match_type": "keyword"}
        evidence = icp_module._direct_evidence("hmc_query", [*rows, neighbor])
        decision = _run(evidence_gate_node({**state, "evidence_registry": evidence}))["safety_decision"]
        for item in evidence:
            if item["match_type"] == "exact_conversation_reference":
                self.assertIn(item["evidence_id"], decision["approved_evidence_ids"])
            else:
                self.assertIn(item["evidence_id"], decision["blocked_evidence_ids"])
        cursor.fetchall.return_value = [tuple(self._stored_row().values())] * 2
        with patch.object(icp_module.postgres_client, "conn", connection):
            self.assertEqual([], icp_module.icp_repository.get_records_by_business_selectors("hmc", selectors)["rows"])

    def _query(self, query, terms, *, typed_components=None):
        connection = _Connection()
        with patch.object(icp_module.postgres_client, "conn", connection):
            icp_module._query_hmc(
                query,
                terms,
                5,
                typed_components=typed_components,
            )
        return connection.calls[0]

    def test_labelled_planner_terms_do_not_authorize_exact_components(self):
        sql, _params = self._query(
            "查询当前异常",
            ["LCN 91810001", "HSC 0001", "91810001", "0001"],
        )

        self.assertIn("ILIKE", sql)
        self.assertNotIn("h.lcn=%s AND h.hsc=%s", sql)

    def test_task_time_and_voltage_numbers_are_not_guessed_as_hmc(self):
        sql, _params = self._query(
            "任务持续 10 小时，输入为 28V",
            ["91810001", "10", "28", "任务时间", "输入电压"],
        )

        self.assertIn("ILIKE", sql)
        self.assertNotIn("h.lcn=%s AND h.hsc=%s", sql)

    def test_reordered_untyped_numbers_are_not_hmc_components(self):
        sql, _params = self._query(
            "核对这些测量值",
            ["0001", "42", "91810001"],
        )

        self.assertIn("ILIKE", sql)
        self.assertNotIn("h.lcn=%s AND h.hsc=%s", sql)

    def test_bare_eight_digit_number_is_not_split_into_hmc_components(self):
        sql, _params = self._query(
            "任务批次号 91810001，持续 10 小时，输入 28V",
            ["91810001", "10", "28"],
        )

        self.assertIn("ILIKE", sql)
        self.assertNotIn("h.lcn=%s AND h.hamc=%s AND h.hsc=%s", sql)

    def test_naked_typed_components_cannot_authorize_exact_query(self):
        sql, _params = self._query(
            "核对当前报码",
            ["任务时间 10 小时", "输入 28V"],
            typed_components={"lcn": "91810001", "hsc": "1"},
        )

        self.assertIn("ILIKE", sql)
        self.assertNotIn("h.lcn=%s AND h.hsc=%s", sql)

    def test_raw_labelled_components_authorize_exact_query(self):
        sql, params = self._query(
            "核对 LCN 91810001 HSC 1",
            ["任务时间 10 小时", "输入 28V"],
        )

        self.assertIn("h.lcn=%s AND h.hsc=%s", sql)
        self.assertEqual(("91810001", "0001", 5), params)


class HmcMflRelationshipSafetyTests(unittest.TestCase):
    def test_semantically_similar_mfl_row_is_audit_only_without_explicit_link(self):
        hmc_rows = [
            {
                "hmc_code": "91810001**0001",
                "match_type": "exact_code",
            }
        ]
        mfl_rows = [
            {
                "mfl_code": "MFL-001",
                "module_fault": "输入异常",
                "maintenance_action": "检查输入器件",
            }
        ]

        rows, binding = icp_module._bind_mfl_rows_to_hmc(
            hmc_rows,
            mfl_rows,
            requested_components={
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "hmc_code": "91810001**0001",
            },
        )

        self.assertTrue(binding["required"])
        self.assertFalse(binding["bound"])
        self.assertEqual("no_explicit_authoritative_relation", binding["basis"])
        self.assertTrue(rows[0]["query_binding_excluded"])
        self.assertEqual([], icp_module._mfl_rows_for_hmc_binding(rows, binding))

    def test_authoritative_explicit_hmc_reference_keeps_mfl_row(self):
        hmc_rows = [
            {
                "hmc_code": "91810001**0001",
                "match_type": "exact_code",
            }
        ]
        mfl_rows = [
            {
                "mfl_code": "MFL-001",
                "document_links": {"hmc_codes": ["91810001-**-0001"]},
            }
        ]

        rows, binding = icp_module._bind_mfl_rows_to_hmc(hmc_rows, mfl_rows)

        self.assertTrue(binding["bound"])
        self.assertEqual("explicit_authoritative_reference", binding["basis"])
        self.assertNotIn("query_binding_excluded", rows[0])
        self.assertEqual(["MFL-001"], binding["mfl_codes"])

    def test_diagnosis_and_gate_keep_unlinked_mfl_out_of_answer_context(self):
        hmc = {
            "id": "hmc-row",
            "doc_id": "doc-hmc",
            "hmc_code": "91810001**0001",
            "lcn": "91810001",
            "hamc": "**",
            "hsc": "0001",
            "fault_desc": "输入异常",
            "repair_method": "按 HMC 检查输入",
            "match_type": "exact_code",
        }
        mfl = {
            "id": "mfl-row",
            "doc_id": "doc-mfl",
            "mfl_code": "MFL-001",
            "module_fault": "输入异常",
            "maintenance_action": "检查相似输入故障",
        }
        state = {
            "intent": "exact_code",
            "query": "针对 91810001-**-0001，维修清单是否有处理办法？",
            "rewritten_query": "查询该 HMC 和维修清单",
            "query_targets": [
                {"intent": "exact_code"},
                {"intent": "maintenance_query"},
            ],
            "query_frame": {
                "required_answer_fields": [
                    "hmc_identity",
                    "maintenance_action",
                ]
            },
            "allowed_tools": ["hmc_query", "mfl_query", "evidence_synthesis"],
            "authorized_tools": [
                "hmc_query",
                "mfl_query",
                "evidence_synthesis",
            ],
        }
        with (
            patch.object(icp_module, "_query_hmc", return_value=[hmc]),
            patch.object(icp_module, "_query_mfl", return_value=[mfl]),
            patch.object(
                icp_module,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: list(rows),
            ),
        ):
            result = _run(icp_module.icp_diagnosis_node(state))

        self.assertFalse(result["hmc_mfl_binding"]["bound"])
        mfl_evidence = [
            item
            for item in result["evidence_registry"]
            if item.get("doc_type") == "mfl"
        ]
        self.assertTrue(mfl_evidence)
        self.assertTrue(
            all(
                item.get("metadata", {}).get("query_binding_excluded") is True
                for item in mfl_evidence
            )
        )

        gated = _run(evidence_gate_node({**state, **result}))
        self.assertTrue(
            all(
                item["evidence_id"] not in gated["context_evidence_ids"]
                for item in mfl_evidence
            )
        )
        self.assertIn(
            "hmc_mfl_relation", gated["safety_decision"]["blocked_claims"]
        )


_RBD_LEDGER = {
    1: {
        "doc_type": "rbd",
        "claim_type": "rbd_structure",
        "display_summary": "RBD 结构：核心处理功能可靠性模型；成功判据：电源链正常,GPP-A/B至少一路正常。",
        "evidence_id": "ev-rbd-1",
    },
    2: {"doc_type": "fmeca", "claim_type": "fmeca_fact", "evidence_id": "ev-fmeca-1"},
}




class _Cursor:
    description = [("hmc_code",)]

    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        self.calls.append((sql, params))

    def fetchall(self):
        return []

class _Connection:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return _Cursor(self.calls)


if __name__ == "__main__":
    unittest.main()
