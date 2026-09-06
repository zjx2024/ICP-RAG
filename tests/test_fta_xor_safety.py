import unittest

from backend.icp.reliability import _gate_type, calculate_fta, ReliabilityModelError


class XorSafetyTests(unittest.TestCase):
    def graph(self, gate):
        return {"root_fault_id": "t", "vertices": [
            {"id": "t", "properties": {"gate_type": gate}},
            {"id": "a", "properties": {"name": "条件甲", "probability": .1}},
            {"id": "b", "properties": {"name": "条件乙", "probability": .2}}],
            "edges": [{"source": "t", "target": "a"}, {"source": "t", "target": "b"}]}

    def test_xor_is_not_inclusive_or_and_cannot_use_positive_only_cut_sets(self):
        self.assertEqual("XOR", _gate_type({"properties": {"gate_type": "XOR"}}, []))
        with self.assertRaisesRegex(ReliabilityModelError, "XOR"):
            calculate_fta(self.graph("XOR"))

    def test_or_and_probabilities_unchanged(self):
        self.assertAlmostEqual(.28, calculate_fta(self.graph("OR"))["top_event_probability"])
        self.assertAlmostEqual(.02, calculate_fta(self.graph("AND"))["top_event_probability"])

    def test_unreachable_xor_does_not_block_a_selected_or_subtree(self):
        graph = self.graph("OR")
        graph["vertices"].append({"id": "unrelated", "properties": {"gate_type": "XOR"}})
        graph["edges"].append({"source": "unrelated", "target": "a"})
        self.assertAlmostEqual(.28, calculate_fta(graph)["top_event_probability"])
