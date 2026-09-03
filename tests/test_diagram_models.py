import copy
import unittest

from backend.icp.diagram_models import (
    DiagramStateTransitionError,
    PROJECTION_NOT_PROJECTED,
    PROJECTION_PUBLISHING,
    REVIEW_CONFIRMED,
    REVIEW_CONFLICTED,
    REVIEW_PENDING,
    canonical_graph_hash,
    detect_graph_conflicts,
    ensure_projection_transition,
    ensure_review_transition,
    merge_candidates,
    normalize_graph,
    source_fingerprint,
)


def _fta_graph(*, probability=0.1, reverse=False):
    edge = {"source": "top", "target": "basic"}
    if reverse:
        edge = {"source": "basic", "target": "top"}
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
                "properties": {
                    "event_id": "EVT-001",
                    "name": "电源失效",
                    "probability": probability,
                },
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "label": "causes",
                **edge,
                "properties": {"gate_type": "OR"},
            }
        ],
    }


class DiagramNormalizationTests(unittest.TestCase):
    def test_normalization_is_idempotent_and_hash_ignores_provenance(self):
        graph = _fta_graph()
        first = normalize_graph(
            graph,
            "diagram-1",
            diagram_type="fta",
            source_ref={"candidate_id": "candidate-a", "source_kind": "image_vision"},
        )
        second = normalize_graph(first, "diagram-1", diagram_type="fta")
        with_other_source = normalize_graph(
            graph,
            "diagram-1",
            diagram_type="fta",
            source_ref={"candidate_id": "candidate-b", "source_kind": "table_nodes"},
        )

        self.assertEqual(first["vertices"][0]["id"], second["vertices"][0]["id"])
        self.assertEqual(canonical_graph_hash(first), canonical_graph_hash(second))
        self.assertEqual(canonical_graph_hash(first), canonical_graph_hash(with_other_source))

    def test_hash_is_independent_of_input_order(self):
        graph = _fta_graph()
        reversed_graph = copy.deepcopy(graph)
        reversed_graph["vertices"].reverse()
        normalized_left = normalize_graph(graph, "diagram-1", diagram_type="fta")
        normalized_right = normalize_graph(reversed_graph, "diagram-1", diagram_type="fta")
        self.assertEqual(canonical_graph_hash(normalized_left), canonical_graph_hash(normalized_right))

    def test_source_fingerprint_is_stable_for_table_objects_and_bytes(self):
        self.assertEqual(
            source_fingerprint({"rows": [["事件ID", "名称"], ["01", "欠压"]]}),
            source_fingerprint({"rows": [["事件ID", "名称"], ["01", "欠压"]]}),
        )
        self.assertEqual(source_fingerprint(b"image"), source_fingerprint(bytearray(b"image")))


class DiagramMergeTests(unittest.TestCase):
    def test_table_and_image_exact_identity_merge_once_with_both_sources(self):
        image = _fta_graph()
        table = {
            "diagram_type": "fta",
            "root_fault_id": "T",
            "vertices": [
                {"id": "T", "label": "顶事件", "properties": {"name": "系统失效", "gate_type": "OR"}},
                {
                    "id": "row-1",
                    "label": "基本事件",
                    "properties": {"event_id": "EVT-001", "name": "电源失效", "probability": 0.1},
                },
            ],
            "edges": [{"source": "T", "target": "row-1", "label": "导致", "properties": {"gate_type": "OR"}}],
        }
        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "image", "source_kind": "image_vision", "graph": image},
                {"candidate_id": "table", "source_kind": "table_nodes", "graph": table},
            ],
        )

        self.assertEqual(result.review_status, REVIEW_PENDING)
        self.assertEqual(result.conflict_count, 0)
        self.assertEqual(len(result.graph["vertices"]), 2)
        self.assertEqual(len(result.graph["edges"]), 1)
        event = next(
            vertex for vertex in result.graph["vertices"]
            if vertex["properties"].get("event_id") == "EVT-001"
        )
        self.assertEqual(
            {ref["candidate_id"] for ref in event["properties"]["source_refs"]},
            {"image", "table"},
        )

    def test_critical_parameter_difference_is_blocking(self):
        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "image", "source_kind": "image_vision", "graph": _fta_graph(probability=0.1)},
                {"candidate_id": "table", "source_kind": "table_nodes", "graph": _fta_graph(probability=0.2)},
            ],
        )
        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertTrue(any(item.code == "node_property_mismatch" for item in result.conflicts))

    def test_table_explicit_id_and_image_name_only_merge_by_unique_alias(self):
        table = _fta_graph()
        table["vertices"][0]["id"] = "TOP-EVENT-001"
        table["edges"][0]["source"] = "TOP-EVENT-001"
        table["root_fault_id"] = "TOP-EVENT-001"
        image = _fta_graph()
        del image["vertices"][1]["properties"]["event_id"]

        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "table", "source_kind": "table_nodes", "graph": table},
                {"candidate_id": "image", "source_kind": "image_vision", "graph": image},
            ],
        )

        self.assertEqual(result.review_status, REVIEW_PENDING)
        self.assertEqual(len(result.graph["vertices"]), 2)
        self.assertEqual(len(result.graph["edges"]), 1)
        self.assertEqual(
            {"image", "table"},
            {
                ref["candidate_id"]
                for vertex in result.graph["vertices"]
                for ref in vertex["properties"].get("source_refs", [])
            },
        )

    def test_same_name_with_two_explicit_ids_is_not_merged(self):
        left = {
            "diagram_type": "fta",
            "vertices": [
                {
                    "id": "EVENT-001",
                    "label": "BasicEvent",
                    "properties": {"name": "电源失效"},
                }
            ],
            "edges": [],
        }
        right = copy.deepcopy(left)
        right["vertices"][0]["id"] = "EVENT-002"

        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "left", "source_kind": "table_nodes", "graph": left},
                {"candidate_id": "right", "source_kind": "image_vision", "graph": right},
            ],
        )

        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertEqual(len(result.graph["vertices"]), 2)
        self.assertTrue(
            any(item.code == "same_name_distinct_identity" for item in result.conflicts)
        )

    def test_opposite_edge_directions_are_not_silently_combined(self):
        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "image", "source_kind": "image_vision", "graph": _fta_graph()},
                {"candidate_id": "table", "source_kind": "table_edges", "graph": _fta_graph(reverse=True)},
            ],
        )
        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertTrue(any(item.code == "edge_direction_mismatch" for item in result.conflicts))

    def test_same_edge_with_different_gate_type_is_conflicted(self):
        left = _fta_graph()
        right = _fta_graph()
        right["edges"][0]["properties"]["gate_type"] = "AND"

        result = merge_candidates(
            "diagram-1",
            "fta",
            [
                {"candidate_id": "left", "source_kind": "table_edges", "graph": left},
                {"candidate_id": "right", "source_kind": "image_vision", "graph": right},
            ],
        )

        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertTrue(any(item.code == "edge_property_mismatch" for item in result.conflicts))

    def test_same_name_with_different_types_requires_review(self):
        architecture = {
            "diagram_type": "function_locator",
            "vertices": [{"id": "v1", "label": "Module", "properties": {"name": "DPM"}}],
            "edges": [],
        }
        other = copy.deepcopy(architecture)
        other["vertices"][0]["label"] = "Function"
        result = merge_candidates(
            "diagram-2",
            "function_locator",
            [
                {"candidate_id": "a", "source_kind": "image_vision", "graph": architecture},
                {"candidate_id": "b", "source_kind": "table_nodes", "graph": other},
            ],
        )
        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertTrue(any(item.code == "ambiguous_name_types" for item in result.conflicts))

    def test_empty_automatic_candidate_is_never_confirmed(self):
        result = merge_candidates(
            "diagram-3",
            "rbd",
            [{"candidate_id": "image", "source_kind": "image_vision", "graph": {"vertices": [], "edges": []}}],
        )
        self.assertEqual(result.review_status, REVIEW_CONFLICTED)
        self.assertNotEqual(result.review_status, REVIEW_CONFIRMED)

    def test_dangling_edge_is_detected(self):
        graph = normalize_graph(_fta_graph(), "diagram-1", diagram_type="fta")
        graph["edges"][0]["target"] = "missing"
        conflicts = detect_graph_conflicts(graph)
        self.assertTrue(any(item.code == "dangling_edge" for item in conflicts))


class DiagramStateTests(unittest.TestCase):
    def test_automatic_transition_cannot_confirm(self):
        with self.assertRaises(DiagramStateTransitionError):
            ensure_review_transition(REVIEW_PENDING, REVIEW_CONFIRMED, automatic=True)

    def test_projection_transition_is_explicit(self):
        ensure_projection_transition(PROJECTION_NOT_PROJECTED, PROJECTION_PUBLISHING)
        with self.assertRaises(DiagramStateTransitionError):
            ensure_projection_transition(PROJECTION_NOT_PROJECTED, "published")


if __name__ == "__main__":
    unittest.main()
