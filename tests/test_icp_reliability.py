import math
import unittest

from backend.icp.reliability import calculate_fta, calculate_rbd


class FtaCalculationTests(unittest.TestCase):
    def test_or_gate_probability_and_minimal_cut_sets(self):
        graph = {
            "root_fault_id": "top",
            "vertices": [
                {"id": "top", "label": "TopEvent", "properties": {"name": "ICP功能丧失", "gate_type": "OR"}},
                {"id": "a", "label": "BasicEvent", "properties": {"name": "GPP失效", "probability": 0.1}},
                {"id": "b", "label": "BasicEvent", "properties": {"name": "电源失效", "probability": 0.2}},
            ],
            "edges": [
                {"id": "e1", "source": "top", "target": "a", "properties": {"gate_type": "OR"}},
                {"id": "e2", "source": "top", "target": "b", "properties": {"gate_type": "OR"}},
            ],
        }

        result = calculate_fta(graph)

        self.assertEqual(result["top_event"], "ICP功能丧失")
        self.assertEqual(result["minimal_cut_set_count"], 2)
        self.assertEqual(
            {frozenset(item) for item in result["minimal_cut_sets"]},
            {frozenset({"GPP失效"}), frozenset({"电源失效"})},
        )
        self.assertAlmostEqual(result["top_event_probability"], 0.28, places=10)
        self.assertEqual(result["missing_probability_nodes"], [])

    def test_and_gate_builds_joint_cut_set(self):
        graph = {
            "root_fault_id": "top",
            "vertices": [
                {"id": "top", "label": "TopEvent", "properties": {"name": "双通道失效", "gate_type": "AND"}},
                {"id": "a", "label": "BasicEvent", "properties": {"name": "A通道失效", "probability": 0.1}},
                {"id": "b", "label": "BasicEvent", "properties": {"name": "B通道失效", "probability": 0.2}},
            ],
            "edges": [
                {"id": "e1", "source": "top", "target": "a", "properties": {"gate_type": "AND"}},
                {"id": "e2", "source": "top", "target": "b", "properties": {"gate_type": "AND"}},
            ],
        }

        result = calculate_fta(graph)

        self.assertEqual(result["minimal_cut_sets"], [["A通道失效", "B通道失效"]])
        self.assertAlmostEqual(result["top_event_probability"], 0.02, places=10)

    def test_failure_rate_uses_mission_time(self):
        graph = {
            "root_fault_id": "top",
            "vertices": [
                {"id": "top", "label": "TopEvent", "properties": {"name": "失效", "gate_type": "OR"}},
                {"id": "a", "label": "BasicEvent", "properties": {"name": "基本事件", "failure_rate": 0.001}},
            ],
            "edges": [{"id": "e1", "source": "top", "target": "a", "properties": {"gate_type": "OR"}}],
        }

        result = calculate_fta(graph, mission_time=100)
        self.assertAlmostEqual(result["top_event_probability"], 1 - math.exp(-0.1), places=10)


class RbdCalculationTests(unittest.TestCase):
    def test_parallel_paths(self):
        graph = {
            "vertices": [
                {"id": "start", "label": "Start", "properties": {"name": "START"}},
                {"id": "a", "label": "ReliabilityBlock", "properties": {"name": "GPP-A", "reliability": 0.9}},
                {"id": "b", "label": "ReliabilityBlock", "properties": {"name": "GPP-B", "reliability": 0.8}},
                {"id": "end", "label": "End", "properties": {"name": "END"}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "a"},
                {"id": "e2", "source": "a", "target": "end"},
                {"id": "e3", "source": "start", "target": "b"},
                {"id": "e4", "source": "b", "target": "end"},
            ],
        }

        result = calculate_rbd(graph)

        self.assertEqual(result["success_path_count"], 2)
        self.assertAlmostEqual(result["system_reliability"], 0.98, places=10)
        self.assertAlmostEqual(result["system_unreliability"], 0.02, places=10)
        self.assertEqual(result["missing_reliability_blocks"], [])

    def test_series_path(self):
        graph = {
            "vertices": [
                {"id": "start", "label": "Start", "properties": {"name": "START"}},
                {"id": "power", "label": "ReliabilityBlock", "properties": {"name": "电源", "reliability": 0.99}},
                {"id": "gpp", "label": "ReliabilityBlock", "properties": {"name": "GPP", "reliability": 0.98}},
                {"id": "end", "label": "End", "properties": {"name": "END"}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "power"},
                {"id": "e2", "source": "power", "target": "gpp"},
                {"id": "e3", "source": "gpp", "target": "end"},
            ],
        }

        result = calculate_rbd(graph)
        self.assertEqual(result["success_paths"], [["电源", "GPP"]])
        self.assertAlmostEqual(result["system_reliability"], 0.99 * 0.98, places=10)

    def test_missing_parameter_does_not_invent_value(self):
        graph = {
            "vertices": [
                {"id": "start", "label": "Start", "properties": {"name": "START"}},
                {"id": "a", "label": "ReliabilityBlock", "properties": {"name": "未知可靠度模块"}},
                {"id": "end", "label": "End", "properties": {"name": "END"}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "a"},
                {"id": "e2", "source": "a", "target": "end"},
            ],
        }

        result = calculate_rbd(graph)
        self.assertIsNone(result["system_reliability"])
        self.assertEqual(result["missing_reliability_blocks"], ["未知可靠度模块"])


if __name__ == "__main__":
    unittest.main()
