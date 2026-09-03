import importlib.util
import asyncio
import sys
import types
import unittest


if importlib.util.find_spec("psycopg2") is None:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)
    psycopg2_stub.connect = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("psycopg2 unavailable")
    )
    sys.modules["psycopg2"] = psycopg2_stub

from backend.agent.tools.icp_diagnosis import _graph_request, _graph_seed_bindings
from backend.agent.graph import build_graph
from backend.icp.structured_graph import business_node_id, document_text_node_id


class AgentGraphSeedHandoffTests(unittest.TestCase):
    def test_complete_hmc_code_is_exact_while_semantic_rows_are_candidates(self):
        bindings = _graph_seed_bindings(
            hmc_rows=[
                {
                    "doc_id": "doc-hmc",
                    "hmc_code": "91810001**0001",
                    "match_type": "keyword",
                },
                {
                    "doc_id": "doc-hmc",
                    "hmc_code": "91810001**0001",
                    "match_type": "exact_code",
                },
            ],
            fmeca_rows=[
                {"doc_id": "doc-fmeca", "code": "FM-SYN-001"},
            ],
        )
        by_id = {item["logical_id"]: item for item in bindings}
        hmc = by_id[business_node_id("hmc_code", "91810001**0001")]
        fmeca = by_id[business_node_id("fmeca_record", "FM-SYN-001")]
        self.assertEqual("exact", hmc["binding_status"])
        self.assertEqual("candidate", fmeca["binding_status"])

    def test_graph_request_is_bounded_and_always_requires_pg_reauthorization(self):
        bindings = _graph_seed_bindings(
            history_rows=[{"doc_id": "doc-history", "case_id": "HIS-SYN-024"}],
        )
        request = _graph_request(bindings, ["graph_traversal"])
        self.assertEqual("subgraph", request["mode"])
        self.assertEqual(["icp_structured"], request["kb_ids"])
        self.assertEqual(3, request["max_depth"])
        self.assertLessEqual(request["max_nodes"], 200)
        self.assertLessEqual(request["max_edges"], 400)
        self.assertTrue(request["require_relation_reauthorization"])

    def test_non_formal_diagram_cannot_become_graph_seed(self):
        bindings = _graph_seed_bindings(
            fta_rows=[
                {
                    "doc_id": "doc-fta",
                    "diagram_key": "FTA-SYN-001",
                    "validation_status": "provisional",
                    "projection_status": "published",
                    "is_active": True,
                }
            ]
        )
        self.assertEqual([], bindings)
        self.assertEqual({}, _graph_request(bindings, ["graph_traversal"]))

    def test_formal_rbd_without_business_id_uses_revision_scoped_occurrence(self):
        row = {
            "doc_id": "doc-rbd",
            "diagram_id": "diagram-1",
            "revision_id": "revision-2",
            "diagram_key": "RBD:任务链路",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
        }
        bindings = _graph_seed_bindings(rbd_rows=[row])
        self.assertEqual(1, len(bindings))
        self.assertEqual(
            document_text_node_id(
                "doc-rbd",
                "rbd_diagram_occurrence",
                "revision-2:diagram-1",
            ),
            bindings[0]["logical_id"],
        )
        self.assertEqual("candidate", bindings[0]["binding_status"])

    def test_icp_graph_routes_through_adaptive_review_before_optional_traversal(self):
        compiled = asyncio.run(build_graph(["icp_diagnosis", "graph_traversal"]))
        edges = {
            (edge.source, edge.target)
            for edge in compiled.get_graph().edges
        }
        self.assertIn(("icp_diagnosis", "adaptive_tools"), edges)
        self.assertIn(("adaptive_tools", "graph_traversal"), edges)
        self.assertIn(("adaptive_tools", "icp_calculation"), edges)
        self.assertIn(("adaptive_tools", "adaptive_retrieval"), edges)
        self.assertNotIn(("icp_diagnosis", "graph_traversal"), edges)
        self.assertIn(("graph_traversal", "icp_calculation"), edges)


if __name__ == "__main__":
    unittest.main()
