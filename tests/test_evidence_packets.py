"""Record/context contracts use identities and read states, not question words."""
import copy
import json
import subprocess
import sys
import unittest

from backend.agent.citation_binding import quote_field_view
from backend.agent.contracts import deduplicate_evidence, evidence_identity, make_evidence, make_tool_result
from backend.agent.evidence_packets import build_evidence_packets, query_coverage, returned_projection_read


def evidence(**overrides):
    return make_evidence(**{
        "source_kind": "postgres", "authority": "authoritative_record", "evidence_level": "L1_DIRECT",
        "doc_type": "hmc", "doc_id": "manual-alpha", "record_id": "0007", "claim_type": "hmc_fault",
        "title": "任意同名现象", "content": "资料原文",
        "metadata": {"answer_fields": {"fault_symptom": "显示更新停顿"}}, **overrides,
    })


def packets(items, **options):
    return build_evidence_packets(dict(enumerate(items, 1)), approved_ids=set(), blocked_ids=set(), **options)


class EvidencePacketTests(unittest.TestCase):
    def test_identity_stays_legacy_compatible_without_scope(self):
        row = evidence()
        self.assertEqual(row["evidence_id"], evidence_identity(
            source_kind="postgres", doc_type="hmc", doc_id="manual-alpha", record_id="0007", claim_type="hmc_fault",
        ))

    def test_claim_ids_separate_scopes_versions_revisions_and_variants_before_dedup(self):
        items = [evidence()]
        for scope in ({"configuration": "A"}, {"configuration": "B"}, {"document_version": "01"}, {"document_version": "1"}):
            items.append(evidence(scope=scope))
        for metadata in ({"revision_id": "01"}, {"revision_id": "1"}, {"variant_id": "001"}, {"variant_id": "002"}):
            items.append(evidence(metadata=metadata))
        self.assertEqual(len(items), len({item["evidence_id"] for item in items}))
        self.assertEqual(len(items), len(deduplicate_evidence(items)))
        self.assertEqual(len(items), len(packets(items)))

    def test_scope_key_order_does_not_change_identity(self):
        self.assertEqual(evidence(scope={"system": "X", "configuration": "Y"})["evidence_id"],
                         evidence(scope={"configuration": "Y", "system": "X"})["evidence_id"])

    def test_one_record_groups_claims_without_promoting_their_permission(self):
        symptom = evidence()
        repair = evidence(claim_type="maintenance_action", content="先检查连接状态")
        result = build_evidence_packets({4: symptom, 9: repair},
            approved_ids={symptom["evidence_id"]}, blocked_ids={repair["evidence_id"]})
        self.assertEqual(1, len(result))
        self.assertEqual([4, 9], [claim["citation_id"] for claim in result[0]["claims"]])
        self.assertEqual(["[^4]", "[^9]"], [claim["citation_label"] for claim in result[0]["claims"]])
        self.assertEqual(["approved", "blocked"], [claim["usage"] for claim in result[0]["claims"]])
        self.assertEqual("not_asserted", result[0]["cross_record_relation"])

    def test_same_labels_other_records_documents_and_unknown_parents_stay_separate(self):
        for other in (evidence(record_id="7"), evidence(doc_id="manual-beta"),
                      evidence(source_kind="opensearch", authority="search_projection", chunk_id="chunk-a")):
            self.assertEqual(2, len(packets([evidence(), other])))
        for overrides in ({"record_id": ""}, {"doc_id": ""}):
            first = evidence(**overrides)
            second = evidence(**overrides, claim_type="maintenance_action")
            self.assertEqual(2, len(packets([first, second])))

    def test_blocked_takes_precedence_if_gate_sets_are_inconsistent(self):
        item = evidence()
        result = build_evidence_packets({1: item}, approved_ids={item["evidence_id"]}, blocked_ids={item["evidence_id"]})
        self.assertEqual("blocked", result[0]["claims"][0]["usage"])

    def test_projection_keeps_explicit_empty_separate_from_false_zero_and_unread(self):
        read = returned_projection_read({"null": None, "blank": "  ", "list": [], "false": False, "zero": 0})
        self.assertEqual(["blank", "list", "null"], read["empty_fields"])
        self.assertIn("false", read["returned_fields"])
        self.assertIn("zero", read["returned_fields"])
        self.assertNotIn("unread", read["returned_fields"])
        self.assertEqual("unknown", read["record_completeness"])

    def test_complete_view_is_not_a_complete_source_read(self):
        claim = packets([evidence()])[0]["claims"][0]
        self.assertEqual("complete_selected_view", claim["view_state"]["status"])
        self.assertEqual({"status": "unknown", "record_completeness": "unknown"}, claim["source_read"])

    def test_truncation_is_explicit_and_quote_coordinates_stay_identical(self):
        item = evidence(claim_type="fmeca_fact", metadata={"answer_fields": {"detection_method": "测" * 2000}})
        original = copy.deepcopy(item)
        claim = packets([item])[0]["claims"][0]
        self.assertEqual("truncated", claim["view_state"]["status"])
        self.assertEqual(quote_field_view(item), claim["quote_view"])
        self.assertIn("/answer_fields/detection_method", claim["quote_view"]["truncated_fields"])
        self.assertEqual(original, item)

    def test_source_restricted_view_does_not_reintroduce_hidden_content(self):
        item = evidence(metadata={"answer_fields": {"fault_symptom": "延迟"}, "source_field_exclusions": ["maintenance_action"]},
                        content="必须禁止重新泄露的原始维修文本")
        item["_citation_fields_only"] = True
        result = packets([item])
        claim = result[0]["claims"][0]
        self.assertTrue(claim["view_state"]["source_restricted"])
        self.assertNotIn("必须禁止重新泄露", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("/content", claim["quote_view"]["fields"])

    def test_packet_module_does_not_import_runtime_or_ranking(self):
        result = subprocess.run([sys.executable, "-c", "import backend.agent.evidence_packets, sys; "
            "assert not any(x.startswith(('backend.agent.nodes', 'backend.agent.tools', 'backend.storage')) for x in sys.modules); "
            "assert 'backend.agent.candidate_ranking' not in sys.modules"], capture_output=True, text=True, timeout=20)
        self.assertEqual(0, result.returncode, result.stderr)


class QueryCoverageTests(unittest.TestCase):
    def test_not_run_empty_error_partial_and_success_are_distinct(self):
        results = [make_tool_result(tool=f"query-{status}", status=status,
                    params={"limit": 8}, data={"count": 0 if status in {"empty", "failed"} else 1})
                   for status in ("ok", "empty", "failed", "partial", "blocked")]
        actual = query_coverage(results, requested_tools=["unqueried", "unqueried", "query-ok"])
        self.assertEqual(["ok", "empty", "failed", "partial", "blocked", "not_queried"],
                         [item["execution_status"] for item in actual])
        self.assertTrue(all(item["matching_set_completeness"] == "not_established" for item in actual))

    def test_retries_remain_separate_and_counts_do_not_prove_exhaustiveness(self):
        results = [make_tool_result(tool="hmc_query", status=status, params={"limit": 12, "terms": ["0007"]},
                    data={"count": count}, errors=["unavailable"] if status == "failed" else [])
                   for status, count in (("failed", 0), ("ok", 2), ("ok", 12))]
        actual = query_coverage(results)
        self.assertEqual(3, len(actual))
        self.assertTrue(actual[0]["errors_present"])
        self.assertTrue(all(row["query_scope"]["terms"] == ["0007"] for row in actual))
        self.assertTrue(all(row["matching_set_completeness"] == "not_established" for row in actual))


class EvidencePacketIntegrationTests(unittest.TestCase):
    def test_pg_identity_reaches_registry_without_needing_a_unique_source_ref(self):
        from backend.agent.tools.icp_diagnosis import _direct_evidence
        for tool, record in (("hmc_query", {"hmc_code": "009"}), ("fmeca_query", {"code": "009"})):
            for key in ("revision_id", "variant_id"):
                rows = [{"doc_id": "manual", **record, key: value} for value in ("01", "1")]
                items = _direct_evidence(tool, rows)
                self.assertEqual(2, len(deduplicate_evidence(items)))
                self.assertEqual(2, len(packets(items)))

    def test_direct_hmc_preserves_empty_vs_unread_repair_column(self):
        from backend.agent.tools.icp_diagnosis import _direct_evidence
        for extra in ({}, {"repair_method": None}, {"repair_method": ""}, {"repair_method": "检查"}):
            items = _direct_evidence("hmc_query", [{"doc_id": "manual", "hmc_code": "009", **extra}])
            read = items[0]["metadata"]["source_read"]
            self.assertEqual("repair_method" in extra, "repair_method" in read["returned_fields"])
            self.assertEqual(extra.get("repair_method", "unread") in (None, ""), "repair_method" in read["empty_fields"])
            self.assertEqual("unknown", read["record_completeness"])

    def test_context_replaces_flat_list_and_retains_all_case_dimensions(self):
        from backend.agent.nodes.answer import _format_context
        from backend.agent.tools.icp_diagnosis import _direct_evidence
        items = _direct_evidence("fault_history_query", [{"doc_id": "history", "case_id": "CASE-Q",
            "fault_symptom": "升温后出现断续", "root_cause": "接触不良", "final_solution": "修复连接器",
            "verification_result": "复测恢复", "mission_phase": "温度循环试验"}])
        ids = [item["evidence_id"] for item in items]
        text = _format_context({"query": "这个案例是怎样验证的？", "intent": "history_query",
            "evidence_registry": items, "context_evidence_ids": ids, "safety_decision": {"approved_evidence_ids": ids},
            "query_frame": {"required_answer_fields": ["record_identity"]},
            "requested_source_tools": ["mfl_query"]})
        for value in ("升温后出现断续", "接触不良", "修复连接器", "复测恢复", "温度循环试验", '"citation_id": 1', '"execution_status": "not_queried"'):
            self.assertIn(value, text)
        self.assertIn("按精确记录成包", text)

    def test_query_errors_are_available_even_without_evidence(self):
        from backend.agent.nodes.answer import _format_context
        text = _format_context({"tool_results": [make_tool_result(tool="hmc_query", status="failed", errors=["offline"])]})
        self.assertIn('"execution_status": "failed"', text)


if __name__ == "__main__":
    unittest.main()
