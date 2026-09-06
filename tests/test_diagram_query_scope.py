import asyncio
import unittest
from unittest.mock import patch

from backend.agent import diagram_query_scope as scope
from backend.agent.current_model_selection import model_candidates, validate_selection
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.tools.icp_diagnosis import _direct_evidence


def row(key, title="故障树", **extra):
    return {"doc_id": "d", "revision_id": f"rev-{key}", "diagram_id": f"diagram-{key}",
            "diagram_key": f"FTA-{key}", "title": title, "image_title": title, "diagram_type": "fta",
            "validation_status": "auto_validated", "projection_status": "published", "is_active": True,
            "graph": {"vertices": [], "edges": []}, **extra}


def choice(query, rows, chosen):
    return validate_selection({"status": "selected", "question_basis": query, "selections": [
        {"doc_id": item["doc_id"], "revision_id": item["revision_id"], "matched_title": item["title"]}
        for item in chosen]}, model_candidates(rows, "fta"), query=query)


class DiagramScopeTests(unittest.TestCase):
    def test_semantic_choice_is_not_validity_or_top_one(self):
        rows = [row("101", "通信异常"), row("702", "运行重启")]
        for query in ("运行时一遍遍重新起来", "Please inspect the restart tree", "解释该现象的下层条件"):
            with self.subTest(query=query), patch.object(scope, "select_current_models", return_value=choice(query, rows, rows[1:])):
                selected, binding = asyncio.run(scope.bind_current_diagrams({"query": query}, rows, doc_type="fta"))
            self.assertEqual([rows[1]], selected)
            self.assertFalse(scope.diagram_in_scope(rows[0], binding, query=query))
            self.assertNotIn("calculation_authorizations", binding)

    def test_current_explicit_id_replaces_prior_citation(self):
        rows = [row("101"), row("702")]
        state = {"query": "改查 FTA-702", "selected_conversation_anchors": [
            {"anchor_id": "a1", "doc_type": "fta", "doc_id": "d", "record_id": "FTA-101"}]}
        with patch.object(scope, "select_current_models") as call:
            selected, binding = asyncio.run(scope.bind_current_diagrams(state, rows, doc_type="fta"))
        call.assert_not_called()
        self.assertEqual([rows[1]], selected)
        self.assertEqual("current_identity", binding["basis"])

    def test_same_business_key_across_documents_requires_disambiguation(self):
        rows = [row("101"), row("101", doc_id="other", revision_id="other-revision")]
        selected, binding = asyncio.run(scope.bind_current_diagrams({"query": "FTA-101"}, rows, doc_type="fta"))
        self.assertEqual([], selected)
        self.assertEqual("ambiguous", binding["status"])

    def test_selected_anchor_is_reread_but_neighbour_cannot_join_adaptive_batch(self):
        rows = [row("101"), row("702")]
        query = "继续解释这个模型"
        state = {"query": query, "selected_conversation_anchors": [{"anchor_id": "a", "doc_type": "fta", "doc_id": "d", "record_id": "FTA-101"}]}
        selected, binding = asyncio.run(scope.bind_current_diagrams(state, rows, doc_type="fta"))
        later, later_binding = asyncio.run(scope.bind_current_diagrams({**state, "fta_query_binding": binding}, rows, doc_type="fta"))
        self.assertEqual([rows[0]], later)
        self.assertEqual(binding, later_binding)
        self.assertEqual(binding, scope.merge_diagram_scopes(binding, later_binding))

    def test_query_revision_or_configuration_change_invalidates_scope(self):
        original = row("101", configuration="C1")
        _, binding = asyncio.run(scope.bind_current_diagrams({"query": "FTA-101"}, [original], doc_type="fta"))
        for change in ({"revision_id": "changed"}, {"configuration": "C2"}, {"doc_id": "another"}):
            self.assertFalse(scope.diagram_in_scope({**original, **change}, binding, query="FTA-101"))
        self.assertFalse(scope.diagram_in_scope(original, binding, query="另一轮"))

    def test_ambiguous_or_unavailable_selection_never_falls_back_to_top_one(self):
        for decision in ({}, {"status": "ambiguous", "selections": []}):
            with patch.object(scope, "select_current_models", return_value=decision):
                selected, _ = asyncio.run(scope.bind_current_diagrams({"query": "这张图"}, [row("1"), row("2")], doc_type="fta"))
            self.assertEqual([], selected)

    def test_conflicting_batches_fail_closed(self):
        _, first = asyncio.run(scope.bind_current_diagrams({"query": "FTA-1"}, [row("1")], doc_type="fta"))
        self.assertEqual("conflict", scope.merge_diagram_scopes(first, {**first, "models": [row("2")]})["status"])

    def test_evidence_gate_does_not_restore_a_formal_neighbour(self):
        rows = [row("1"), row("2")]
        _, binding = asyncio.run(scope.bind_current_diagrams({"query": "FTA-1"}, rows, doc_type="fta"))
        evidence = _direct_evidence("fta_query", rows)
        result = asyncio.run(evidence_gate_node({"query": "FTA-1", "intent": "cause_analysis",
            "answer_mode": "knowledge_explanation", "allowed_tools": ["fta_query"],
            "fta_query_binding": binding, "evidence_registry": evidence}))
        self.assertIn(evidence[0]["evidence_id"], result["approved_evidence_ids"])
        self.assertNotIn(evidence[1]["evidence_id"], result["context_evidence_ids"])
