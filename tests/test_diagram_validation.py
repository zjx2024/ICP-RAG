import unittest

from backend.icp.diagram_validation import (
    VALIDATION_AUTO_VALIDATED,
    VALIDATION_PROVISIONAL,
    VALIDATION_QUARANTINED,
    deterministic_candidate_allows_formal_topology,
    evaluate_automatic_validation,
)


def _candidate(source_kind, graph, confidence=None):
    candidate_graph = dict(graph)
    metadata = dict(candidate_graph.get("metadata") or {})
    if confidence is not None:
        metadata["confidence"] = confidence
    candidate_graph["metadata"] = metadata
    return {
        "candidate_id": f"candidate-{source_kind}",
        "source_kind": source_kind,
        "merge_role": "included",
        "extraction_status": "succeeded",
        "normalized_graph": candidate_graph,
    }


def _fta(probability=0.1):
    return {
        "diagram_type": "fta",
        "root_fault_id": "top",
        "vertices": [
            {
                "id": "top",
                "label": "TopEvent",
                "properties": {"name": "系统失效", "gate_type": "OR"},
            },
            {
                "id": "basic",
                "label": "BasicEvent",
                "properties": {"name": "电源失效", "probability": probability},
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "label": "causes",
                "source": "top",
                "target": "basic",
                "properties": {"gate_type": "OR"},
            }
        ],
    }


def _rbd(*, start_name="START", end_name="END"):
    return {
        "diagram_type": "rbd",
        "metadata": {"topology_source": "structured_expression"},
        "vertices": [
            {"id": "start", "label": "Start", "properties": {"name": start_name}},
            {
                "id": "block",
                "label": "ReliabilityBlock",
                "properties": {"name": "PSU-A", "reliability": 0.99},
            },
            {"id": "end", "label": "End", "properties": {"name": end_name}},
        ],
        "edges": [
            {"id": "e1", "label": "series", "source": "start", "target": "block"},
            {"id": "e2", "label": "series", "source": "block", "target": "end"},
        ],
    }


class DiagramAutomaticValidationTests(unittest.TestCase):
    def test_explicit_table_fta_is_auto_validated_for_formal_use(self):
        graph = _fta()
        result = evaluate_automatic_validation(
            diagram_type="fta",
            graph=graph,
            candidates=[_candidate("table_edges", graph)],
        )

        self.assertEqual(VALIDATION_AUTO_VALIDATED, result.status)
        self.assertTrue(result.publishable)
        self.assertTrue(result.formal_claims_allowed)

    def test_high_confidence_visual_fta_is_provisional_only(self):
        graph = _fta()
        result = evaluate_automatic_validation(
            diagram_type="fta",
            graph=graph,
            candidates=[_candidate("image_vision", graph, confidence=0.92)],
        )

        self.assertEqual(VALIDATION_PROVISIONAL, result.status)
        self.assertTrue(result.publishable)
        self.assertFalse(result.formal_claims_allowed)

    def test_low_confidence_visual_graph_is_quarantined(self):
        graph = _fta()
        result = evaluate_automatic_validation(
            diagram_type="fta",
            graph=graph,
            candidates=[_candidate("image_vision", graph, confidence=0.45)],
        )

        self.assertEqual(VALIDATION_QUARANTINED, result.status)
        self.assertFalse(result.publishable)
        self.assertFalse(result.formal_claims_allowed)

    def test_invalid_probability_quarantines_deterministic_fta(self):
        graph = _fta(probability=1.2)
        result = evaluate_automatic_validation(
            diagram_type="fta",
            graph=graph,
            candidates=[_candidate("table_edges", graph)],
        )

        self.assertEqual(VALIDATION_QUARANTINED, result.status)
        self.assertTrue(
            any(check["code"] == "engineering_parameter_domains" and not check["passed"]
                for check in result.checks)
        )

    def test_rbd_detached_block_is_quarantined(self):
        graph = {
            "diagram_type": "rbd",
            "vertices": [
                {"id": "start", "label": "Start", "properties": {"name": "START"}},
                {"id": "block", "label": "ReliabilityBlock", "properties": {"name": "A"}},
                {"id": "detached", "label": "ReliabilityBlock", "properties": {"name": "B"}},
                {"id": "end", "label": "End", "properties": {"name": "END"}},
            ],
            "edges": [
                {"id": "e1", "label": "series", "source": "start", "target": "block"},
                {"id": "e2", "label": "series", "source": "block", "target": "end"},
            ],
        }
        result = evaluate_automatic_validation(
            diagram_type="rbd",
            graph=graph,
            candidates=[_candidate("structured_expression", graph)],
        )

        self.assertEqual(VALIDATION_QUARANTINED, result.status)
        self.assertTrue(
            any(check["code"] == "rbd_all_nodes_on_path" and not check["passed"]
                for check in result.checks)
        )

    def test_isolated_visual_candidate_does_not_downgrade_table_graph(self):
        graph = _fta()
        visual = _candidate("image_vision", graph, confidence=0.2)
        visual["merge_role"] = "isolated"
        result = evaluate_automatic_validation(
            diagram_type="fta",
            graph=graph,
            candidates=[_candidate("table_edges", graph), visual],
        )

        self.assertEqual(VALIDATION_AUTO_VALIDATED, result.status)
        self.assertEqual(("table_edges",), result.source_kinds)

    def test_complete_rbd_expression_anchors_formal_graph_while_visual_stays_evidence(self):
        deterministic = _rbd()
        visual = _candidate(
            "image_vision",
            _rbd(start_name="输入", end_name="成功"),
            confidence=0.95,
        )
        visual["merge_role"] = "isolated"

        self.assertTrue(
            deterministic_candidate_allows_formal_topology(
                diagram_type="rbd",
                source_kind="structured_expression",
                graph=deterministic,
            )
        )
        self.assertFalse(
            deterministic_candidate_allows_formal_topology(
                diagram_type="rbd",
                source_kind="image_vision",
                graph=visual["normalized_graph"],
            )
        )
        result = evaluate_automatic_validation(
            diagram_type="rbd",
            graph=deterministic,
            candidates=[_candidate("structured_expression", deterministic), visual],
        )

        self.assertEqual(VALIDATION_AUTO_VALIDATED, result.status)
        self.assertTrue(result.formal_claims_allowed)
        self.assertEqual(("structured_expression",), result.source_kinds)
        checks = {item["code"]: item["passed"] for item in result.checks}
        self.assertTrue(checks["rbd_single_start"])
        self.assertTrue(checks["rbd_single_end"])

    def test_invalid_deterministic_source_cannot_anchor_canonical_selection(self):
        invalid = _rbd()
        invalid["vertices"].append(
            {"id": "end-2", "label": "End", "properties": {"name": "另一个终点"}}
        )
        self.assertFalse(
            deterministic_candidate_allows_formal_topology(
                diagram_type="rbd",
                source_kind="structured_expression",
                graph=invalid,
            )
        )


if __name__ == "__main__":
    unittest.main()
