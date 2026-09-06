import copy
import unittest

from backend.agent.evidence_display import explicit_fta_gate_relations
from backend.agent.tools.icp_diagnosis import _direct_evidence
from backend.agent.citation_binding import quote_field_view


class FtaStructureProjectionTests(unittest.TestCase):
    def graph(self):
        return {"root_fault_id": "top", "vertices": [
            {"id": "top", "properties": {"name": "任务异常", "gate_type": "OR"}},
            {"id": "g", "properties": {"name": "双条件组合", "gate_type": "AND"}},
            *[{"id": i, "label": "BasicEvent", "properties": {"name": name}} for i, name in [("a", "能源丢失"), ("b", "温控失效"), ("c", "反馈丢失")]]],
            "edges": [{"source": s, "target": t} for s, t in [("top", "a"), ("top", "g"), ("g", "b"), ("g", "c")]]}

    def test_gate_operands_remain_grouped_in_source_field_view(self):
        graph = self.graph()
        original = copy.deepcopy(graph)
        relations = explicit_fta_gate_relations(graph)
        self.assertEqual(["OR", "AND"], [r["gate_type"] for r in relations])
        self.assertEqual(["b", "c"], [r["event_id"] for r in relations[1]["inputs"]])
        self.assertEqual(original, graph)
        evidence = _direct_evidence("fta_query", [{"doc_id": "doc", "revision_id": "rev", "graph": graph, "validation_status": "auto_validated"}])[0]
        fields = quote_field_view(evidence)["fields"]
        self.assertEqual("AND", fields["/answer_fields/gate_relations/1/gate_type"])
        self.assertEqual("温控失效", fields["/answer_fields/gate_relations/1/inputs/0/event_name"])
        self.assertIn("双条件组合 ← AND(温控失效；反馈丢失)", fields["/answer_fields/structural_cause_chain"])
        self.assertEqual("能源丢失", evidence["metadata"]["answer_fields"]["basic_events"][0]["name"])
        self.assertNotIn("minimal_cut_sets", evidence["metadata"]["answer_fields"])

    def test_unknown_conflicting_or_dangling_topology_does_not_guess_or_gate(self):
        for variant in ("unknown", "conflict", "dangling"):
            graph = self.graph()
            if variant == "unknown": graph["vertices"][0]["properties"]["gate_type"] = "UNKNOWN"
            elif variant == "conflict": graph["edges"][0]["properties"] = {"gate_type": "AND"}
            else: graph["edges"][0]["target"] = "absent"
            with self.subTest(variant=variant):
                self.assertEqual([], explicit_fta_gate_relations(graph))

    def test_provisional_cannot_gain_formal_gate_projection(self):
        evidence = _direct_evidence("fta_query", [{"doc_id": "doc", "revision_id": "rev", "graph": self.graph(), "validation_status": "provisional"}])[0]
        self.assertNotIn("gate_relations", evidence["metadata"]["answer_fields"])
        self.assertEqual("unconfirmed_extraction", evidence["authority"])
