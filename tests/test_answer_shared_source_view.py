import json
import unittest

from backend.agent.citation_binding import quote_field_view
from backend.agent.nodes import answer
from tests.test_answer_rbd_success_criterion_projection import _evidence, _bound_state


class SharedAnswerSourceViewTests(unittest.TestCase):
    def test_facts_outside_a_long_summary_are_visible_to_generation_and_citation(self):
        item = {"evidence_id": "fm-x", "record_id": "FM-ALPHA", "doc_id": "doc-x",
                "doc_type": "fmeca", "claim_type": "fmeca_fact", "authority": "authoritative_record",
                "content": "背景说明。" * 1000, "metadata": {"answer_fields": {
                    "failure_mode": "发送缓冲区堵塞", "local_impact": "后续帧无法入队。",
                    "final_impact": "相关状态更新滞后。", "detection_method": "比较队列计数和接收时间。",
                }}}
        state = {"query": "会影响什么，怎样验证？", "evidence_registry": [item],
                 "safety_decision": {"approved_evidence_ids": ["fm-x"]}}
        prompt = answer._format_context(state)
        ledger = answer._citation_ledger(state)
        for field in ("local_impact", "final_impact", "detection_method"):
            value = item["metadata"]["answer_fields"][field]
            self.assertIn(value, prompt)
            self.assertEqual(value, quote_field_view(ledger[1])["fields"][f"/answer_fields/{field}"])

    def test_rbd_uses_same_authorized_expression_in_draft_proof_and_public_source(self):
        item = _evidence()
        graph = json.loads(item["content"])
        expression = "ENTRY → (CHANNEL-X ∥ CHANNEL-Y) → SINK"
        graph["metadata"]["structure_expression"] = expression
        item["content"] = json.dumps(graph, ensure_ascii=False)
        state = _bound_state(item)
        state["query_frame"].pop("topology_dependency")
        state["query_frame"]["required_answer_fields"] = ["rbd_path_structure"]
        ledger = answer._citation_ledger(state)
        self.assertEqual(expression, quote_field_view(ledger[1])["fields"]["/answer_fields/structure_expression"])
        self.assertIn(expression, answer._format_context(state))
        public = answer._extract_citations(state)[0]
        self.assertEqual(expression, public["document_metadata"]["answer_fields"]["structure_expression"])
        self.assertNotIn("/content", quote_field_view(ledger[1])["fields"])
        self.assertEqual(graph, json.loads(ledger[1]["content"]))
        self.assertEqual(graph, json.loads(item["content"]))

    def test_shared_view_does_not_authorize_an_unbound_or_provisional_graph(self):
        for item in (_evidence(), _evidence(validation_status="provisional")):
            state = _bound_state(item)
            state["rbd_query_binding"] = {}
            self.assertEqual({}, answer._citation_ledger(state))
            self.assertNotIn("success_criterion", answer._format_context(state))


if __name__ == "__main__":
    unittest.main()
