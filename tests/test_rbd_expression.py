import unittest

from backend.core.parsers.rbd_expression import (
    RbdExpressionError,
    parse_rbd_expression,
    rbd_expression_to_graph,
)
from backend.icp.reliability import calculate_rbd


class RbdExpressionTests(unittest.TestCase):
    def test_nested_series_parallel_builds_two_success_paths(self):
        graph = rbd_expression_to_graph(
            "结构表达式：SERIES(ICU, PARALLEL(PSU-A, PSU-B), PDU)",
            [
                {"abbreviation": "ICU", "reliability": 0.99},
                {"abbreviation": "PSU-A", "reliability": 0.9},
                {"abbreviation": "PSU-B", "reliability": 0.8},
                {"abbreviation": "PDU", "reliability": 0.95},
            ],
            source_ref="doc#section-2/expression",
        )
        result = calculate_rbd(graph)
        self.assertEqual(result["success_path_count"], 2)
        self.assertAlmostEqual(result["system_reliability"], 0.99 * (1 - 0.1 * 0.2) * 0.95)
        self.assertEqual(graph["metadata"]["topology_source"], "structured_expression")

    def test_fullwidth_punctuation_and_whitespace_are_supported(self):
        expr = parse_rbd_expression("PARALLEL（ A ， SERIES（B，C） ）；")
        self.assertEqual(expr.canonical(), "PARALLEL(A,SERIES(B,C))")

    def test_invalid_or_unknown_expression_fails_closed(self):
        with self.assertRaises(RbdExpressionError):
            parse_rbd_expression("PARALLEL(A)")
        with self.assertRaises(RbdExpressionError):
            parse_rbd_expression("MAGIC(A,B)")
        with self.assertRaises(RbdExpressionError):
            parse_rbd_expression("SERIES(A,B")

    def test_graph_identity_is_stable_for_same_expression(self):
        first = rbd_expression_to_graph("SERIES(A,PARALLEL(B,C),D)")
        second = rbd_expression_to_graph(" SERIES（A， PARALLEL（B，C），D） ")
        self.assertEqual(first["vertices"], second["vertices"])
        self.assertEqual(first["edges"], second["edges"])


if __name__ == "__main__":
    unittest.main()
