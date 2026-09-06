"""Publication is a mechanical identity mapping, not a second answer editor."""
import ast
import copy
import itertools
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

from backend.agent.citation_rendering import CitationRenderingError, render_citations


def citation(number, **values):
    return {"id": number, "evidence_id": f"e-{number}", "doc_id": "doc", "record_id": f"R-{number}", **values}


class CitationRenderingTests(unittest.TestCase):
    def test_arbitrary_order_keeps_record_bindings_and_every_other_character(self):
        ledger = [citation(number) for number in (3, 8, 21)]
        for order in itertools.permutations((3, 8, 21)):
            original = "  **正文**\r\n\n\n" + "\n".join(f"记录-{number} [^{number}]" for number in (*order, order[0])) + "  "
            actual, items = render_citations(original, ledger)
            self.assertEqual(re.sub(r"\[\^\d+\]", "", original), re.sub(r"\[\^\d+\]", "", actual))
            self.assertEqual([f"e-{number}" for number in order], [item["evidence_id"] for item in items])
            self.assertEqual([1, 2, 3], [item["id"] for item in items])
            self.assertEqual(["1", "2", "3", "1"], re.findall(r"\[\^(\d+)\]", actual))

    def test_context_history_words_and_code_wildcards_do_not_trigger_edits(self):
        for label in ("健康监控与历史记录", "历史缓存", "连接记录", "previous-case buffer", "任意名字 Ω"):
            original = f"- {label}负责事件记录，接口名 GetHistory()；报码 0007**0001。[^8]"
            actual, _ = render_citations(original, [citation(8)], approved_evidence_ids={"e-8"})
            self.assertEqual(original.replace("[^8]", "[^1]"), actual)

    def test_unknown_markers_are_errors_not_silently_removed_claim_support(self):
        for value in ("0", "4", "999"):
            with self.assertRaises(CitationRenderingError):
                render_citations(f"事实。[^{value}]", [citation(3)])

    def test_blocked_or_missing_identity_cannot_become_a_public_source(self):
        for item in (citation(1), {"id": 1}, citation(1, evidence_id="other")):
            with self.assertRaises(CitationRenderingError):
                render_citations("事实。[^1]", [item], approved_evidence_ids=set())

    def test_unused_blocked_evidence_is_not_published_and_does_not_block_valid_text(self):
        actual, items = render_citations("事实。[^3]", [citation(1), citation(3)], approved_evidence_ids={"e-3"})
        self.assertEqual("事实。[^1]", actual)
        self.assertEqual(["e-3"], [item["evidence_id"] for item in items])

    def test_ambiguous_and_malformed_ledger_numbers_fail_closed(self):
        for ledger in ([citation(1), citation(1)], [citation(True)], [citation(0)], [citation("1")]):
            with self.assertRaises(CitationRenderingError):
                render_citations("事实。[^1]", ledger)

    def test_renderer_is_idempotent_and_does_not_mutate_source_payloads(self):
        ledger = [citation(8, document_metadata={"answer_fields": {"value": "原文"}})]
        before = copy.deepcopy(ledger)
        result = render_citations("原文。[^8]", ledger, approved_evidence_ids={"e-8"})
        self.assertEqual(result, render_citations(*result, approved_evidence_ids={"e-8"}))
        self.assertEqual(before, ledger)

    def test_reviewed_no_citation_context_stays_verbatim(self):
        value = "上轮已经列过这些对象。\n\n请补充缺少的标识。"
        self.assertEqual((value, []), render_citations(value, [], approved_evidence_ids=set()))

    def test_legacy_citations_without_evidence_ids_remain_supported(self):
        self.assertEqual(("原文。[^1]", [{"id": 1, "doc_id": "legacy"}]),
                         render_citations("原文。[^9]", [{"id": 9, "doc_id": "legacy"}]))

    def test_module_loads_without_runtime_or_storage(self):
        result = subprocess.run([sys.executable, "-c", "import backend.agent.citation_rendering, sys; "
            "assert not any(x.startswith(('backend.agent.nodes', 'backend.agent.tools', 'backend.storage')) for x in sys.modules)"],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(0, result.returncode, result.stderr)


class PublicationBoundaryTests(unittest.TestCase):
    def test_compaction_no_longer_calls_semantic_editors_or_reconstructs_state(self):
        from backend.agent.nodes import answer
        tree = ast.parse(Path(answer.__file__).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_compact_answer_citations")
        calls = {node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("render_citations", calls)
        self.assertFalse(any(name.startswith(("_strip_", "_ensure_", "_downgrade_", "_normalize_")) for name in calls))
        self.assertNotIn("_terminal_public_ledger", ast.unparse(function))
        self.assertNotIn("_strip_fta_only_current_incident_candidate_cards", Path(answer.__file__).read_text())

    def test_compaction_retains_all_reviewed_collection_members_and_catalog_identity(self):
        from backend.agent.nodes import answer
        from backend.agent.contracts import make_evidence
        items = [make_evidence(source_kind="postgres", authority="authoritative_record", evidence_level="L1_DIRECT",
            doc_type="function_locator", doc_id="architecture-x", record_id=f"C-{index}", title=label,
            claim_type="component_fact", content=f"{label}承担对应功能。")
            for index, label in enumerate(("输入", "输出", "事件历史记录", "时间基准"), 1)]
        ids = [item["evidence_id"] for item in items]
        state = {"query": "请列出全部组成", "intent": "architecture_query", "evidence_registry": items,
                 "safety_decision": {"approved_evidence_ids": ids}}
        ledger = [{"id": index, **answer._evidence_payload(item)} for index, item in enumerate(items, 1)]
        draft = "\n".join(f"- {item['content']}[^{index}]" for index, item in enumerate(items, 1))
        state["_field_reviewed_units"] = draft.splitlines()
        before = copy.deepcopy(state)
        with patch.object(answer, "_review_answer", side_effect=AssertionError("No second semantic pass")):
            final, citations, catalog = answer._compact_answer_citations(draft, ledger, state)
        self.assertEqual(draft, final)
        self.assertEqual(ids, [item["evidence_id"] for item in citations])
        self.assertTrue(all(item["status"] == "answer_used" for item in catalog))
        self.assertEqual(before, state)

    def test_numbered_fta_structure_is_not_mistaken_for_a_repair_candidate(self):
        from backend.agent.nodes import answer
        item = citation(7, doc_type="fta", type="fta_structure")
        draft = "1. 顶事件由两个中间事件的 OR 门连接。[^7]"
        state = {"query_frame": {"diagnostic_closure": {"required": True}},
                 "safety_decision": {"approved_evidence_ids": ["e-7"]}}
        final, citations, _ = answer._compact_answer_citations(draft, [item], state)
        self.assertEqual(draft.replace("[^7]", "[^1]"), final)
        self.assertEqual(["e-7"], [item["evidence_id"] for item in citations])

    def test_missing_gate_cannot_promote_unified_evidence(self):
        from backend.agent.nodes import answer
        with self.assertRaises(answer.AnswerGenerationError):
            answer._compact_answer_citations("事实。[^1]", [citation(1)], {"evidence_registry": [{"evidence_id": "e-1"}]})


if __name__ == "__main__":
    unittest.main()
