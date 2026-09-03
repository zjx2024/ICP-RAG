import copy
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from backend.icp import structured_projection as projection
from backend.icp.diagram_models import content_hash
from backend.icp.document_types import DOCUMENT_TYPES
from backend.icp.structured_graph import business_node_id
from backend.storage.hugegraph_client import (
    DYNAMIC_EDGE_PROPERTIES,
    ENTITY_VERTEX_PROPERTIES,
)


def _logical_graph(doc_id, *, shared_code="HMC-001", include_shared=True):
    document_id = business_node_id("document", doc_id)
    vertices = [
        {
            "id": document_id,
            "label": "Document",
            "properties": {"business_id": doc_id, "name": f"{doc_id}.docx"},
        }
    ]
    edges = []
    if include_shared:
        shared_id = business_node_id("hmc", shared_code)
        vertices.insert(
            0,
            {
                "id": shared_id,
                "label": "HMCCode",
                "properties": {"business_id": shared_code, "name": shared_code},
            },
        )
        edges.append(
            {
                "id": f"sg:e:explicit:{doc_id}",
                "label": "extracted_from",
                "source": shared_id,
                "target": document_id,
                "properties": {"explicit": True},
            }
        )
    return {
        "schema_version": 1,
        "graph_type": "structured_explicit",
        "doc_type": "hmc",
        "doc_id": doc_id,
        "vertices": vertices,
        "edges": edges,
        "warnings": [],
    }


def _shared_relation_graph(doc_id, *, edge_suffix="initial"):
    source = business_node_id("hmc", "HMC-SHARED")
    target = business_node_id("fmeca_record", "FM-SHARED")
    return {
        "schema_version": 1,
        "graph_type": "structured_explicit",
        "doc_type": "hmc",
        "doc_id": doc_id,
        "vertices": [
            {
                "id": source,
                "label": "HMCCode",
                "properties": {"business_id": "HMC-SHARED", "name": "共享 HMC"},
            },
            {
                "id": target,
                "label": "FMECARecord",
                "properties": {"business_id": "FM-SHARED", "name": "共享 FMECA"},
            },
        ],
        "edges": [
            {
                "id": f"sg:e:{doc_id}:{edge_suffix}",
                "label": "references_fmeca",
                "source": source,
                "target": target,
                "properties": {
                    "source_refs": [f"{doc_id}.docx#row=1"],
                    "occurrence_qualifiers": [f"{doc_id}:{edge_suffix}"],
                },
            }
        ],
        "warnings": [],
    }


def _canonical_id(logical_id, *, scope="CONFIG-A"):
    return "entity-" + content_hash({"logical_id": logical_id, "scope": scope})[:24]


def _entity_snapshot(graph, *, scope="CONFIG-A"):
    return {
        "logical_entity_ids": {
            item["id"]: _canonical_id(item["id"], scope=scope)
            for item in graph.get("vertices", [])
        },
        "identity_scope": {
            "system_code": "SYN-ICP-9000",
            "configuration": scope,
        },
        "skipped_edges": [],
    }


def _physical(graph, *, doc_id, doc_type, source_hash, scope="CONFIG-A"):
    snapshot = _entity_snapshot(graph, scope=scope)
    return projection._physical_graph(
        graph,
        doc_id=doc_id,
        doc_type=doc_type,
        source_hash=source_hash,
        logical_entity_ids=snapshot["logical_entity_ids"],
        identity_scope=snapshot["identity_scope"],
    )


class _EntityRepository:
    def sync_structured_graph(self, *, graph, **_kwargs):
        return _entity_snapshot(graph)


class _MemoryHugeGraph:
    _use_mock = False

    def __init__(
        self,
        *,
        fail_batch=False,
        fail_once=False,
        partial_batch=False,
        server_edge_ids=False,
    ):
        self.vertices = {}
        self.edges = {}
        self.fail_batch = fail_batch
        self.fail_once = fail_once
        self.partial_batch = partial_batch
        self.server_edge_ids = server_edge_ids
        self.events = []

    def get_vertices_by_kb_with_details(self, kb_id):
        return [
            copy.deepcopy(item)
            for item in self.vertices.values()
            if (item.get("properties") or {}).get("kb_id") == kb_id
        ]

    def get_edges_by_kb_with_details(self, kb_id):
        return [
            copy.deepcopy(item)
            for item in self.edges.values()
            if (item.get("properties") or {}).get("kb_id") == kb_id
        ]

    def get_vertex_by_id(self, vertex_id):
        item = self.vertices.get(vertex_id)
        return copy.deepcopy(item) if item else None

    def batch_insert(self, vertices, edges):
        self.events.append(("insert", len(vertices), len(edges)))
        should_fail = self.fail_batch
        if should_fail and self.fail_once:
            self.fail_batch = False
        if should_fail and not self.partial_batch:
            return False
        self.vertices.update({item["id"]: copy.deepcopy(item) for item in vertices})
        for item in edges:
            stored = copy.deepcopy(item)
            edge_id = item["id"]
            if self.server_edge_ids:
                edge_id = f"server-edge:{len(self.edges) + 1}"
                stored["id"] = edge_id
            self.edges[edge_id] = stored
        if should_fail:
            return False
        return True

    def replace_vertex_properties(self, vertex_id, properties):
        self.events.append(("replace_vertex", vertex_id))
        if vertex_id not in self.vertices:
            return False
        self.vertices[vertex_id]["properties"] = copy.deepcopy(properties)
        return True

    def replace_edge_properties(self, edge_id, properties):
        self.events.append(("replace_edge", edge_id))
        if edge_id not in self.edges:
            return False
        self.edges[edge_id]["properties"] = copy.deepcopy(properties)
        return True

    def delete_edge(self, edge_id):
        self.events.append(("delete_edge", edge_id))
        return self.edges.pop(edge_id, None) is not None

    def delete_vertex(self, vertex_id):
        self.events.append(("delete_vertex", vertex_id))
        if self.vertices.pop(vertex_id, None) is None:
            return False
        for edge_id in [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.get("outV") == vertex_id or edge.get("inV") == vertex_id
        ]:
            del self.edges[edge_id]
        return True


class _ProjectionState:
    def __init__(self):
        self.rows = {}
        self.marks = []

    def get(self, doc_id):
        value = self.rows.get(doc_id)
        return copy.deepcopy(value) if value else None

    def mark(
        self,
        doc_id,
        doc_type,
        source_hash,
        status,
        *,
        vertex_count=0,
        edge_count=0,
        warnings=(),
        error="",
    ):
        self.marks.append((doc_id, status, str(error)))
        self.rows[doc_id] = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_hash": source_hash,
            "projector_version": projection.PROJECTOR_VERSION,
            "status": status,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "warnings": list(warnings),
            "error": str(error),
        }


class _SqlCursor:
    description = None

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, params=None):
        compact = " ".join(str(sql).split())
        self.conn.statements.append((compact, params))
        if self.conn.fail_insert and compact.startswith("INSERT INTO"):
            raise RuntimeError("insert failed")


class _SqlConnection:
    dsn = ""

    def __init__(self, *, fail_insert=False):
        self.fail_insert = fail_insert
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _SqlCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class StructuredProjectionConversionTests(unittest.TestCase):
    def test_v5_physical_properties_match_hugegraph_schema(self):
        graph = _logical_graph("doc-a")
        vertices, edges = _physical(
            graph,
            doc_id="doc-a",
            doc_type="hmc",
            source_hash="a" * 64,
        )

        self.assertTrue(vertices)
        self.assertTrue(edges)
        for vertex in vertices:
            self.assertTrue(
                set(vertex["properties"]).issubset(set(ENTITY_VERTEX_PROPERTIES))
            )
            self.assertIsInstance(vertex["properties"]["identity_scope"], str)
            self.assertIsInstance(
                json.loads(vertex["properties"]["identity_scope"]), dict
            )
        for edge in edges:
            self.assertTrue(
                set(edge["properties"]).issubset(set(DYNAMIC_EDGE_PROPERTIES))
            )

    def test_canonical_collisions_merge_physical_vertices_and_single_edges(self):
        graph = {
            "vertices": [
                {"id": "sg:o:event:1", "label": "FTAEvent", "properties": {"name": "事件A"}},
                {"id": "sg:o:event:2", "label": "FTAEvent", "properties": {"name": "事件A"}},
                {"id": "sg:o:gate:1", "label": "FTAGate", "properties": {"name": "OR"}},
                {"id": "sg:o:gate:2", "label": "FTAGate", "properties": {"name": "OR"}},
            ],
            "edges": [
                {
                    "id": "sg:e:fta:1",
                    "label": "decomposed_by",
                    "source": "sg:o:event:1",
                    "target": "sg:o:gate:1",
                    "properties": {"occurrence_qualifiers": ["row-1"]},
                },
                {
                    "id": "sg:e:fta:2",
                    "label": "decomposed_by",
                    "source": "sg:o:event:2",
                    "target": "sg:o:gate:2",
                    "properties": {"occurrence_qualifiers": ["row-2"]},
                },
            ],
        }
        mapping = {
            "sg:o:event:1": "entity-event",
            "sg:o:event:2": "entity-event",
            "sg:o:gate:1": "entity-gate",
            "sg:o:gate:2": "entity-gate",
        }

        vertices, edges = projection._physical_graph(
            graph,
            doc_id="doc-fta",
            doc_type="fta",
            source_hash="a" * 64,
            logical_entity_ids=mapping,
            identity_scope={"configuration": "CONFIG-A"},
        )

        self.assertEqual(2, len(vertices))
        self.assertEqual(1, len(edges))
        event = next(item for item in vertices if item["id"] == "icp:ce:entity-event")
        self.assertEqual(
            {"logical:sg:o:event:1", "logical:sg:o:event:2"},
            {
                marker
                for marker in event["properties"]["source_id"]
                if marker.startswith("logical:")
            },
        )
        self.assertEqual(
            {"sg:o:event:1", "sg:o:event:2"},
            {
                part["logical_id"]
                for part in projection._parts(
                    event["properties"]["description_parts"]
                )
            },
        )
        relation = edges[0]
        self.assertEqual(2, relation["properties"]["weight"])
        edge_parts = projection._parts(
            relation["properties"]["description_parts"]
        )
        self.assertEqual({"sg:e:fta:1", "sg:e:fta:2"}, {part["logical_id"] for part in edge_parts})
        self.assertEqual(
            {"row-1", "row-2"},
            {
                qualifier
                for part in edge_parts
                for qualifier in part["occurrence_qualifiers"]
            },
        )

    def test_all_seven_types_have_document_vertex_and_diagrams_use_formal_loader(self):
        current_type = {"value": ""}

        class Postgres:
            def get_icp_document(self, doc_id):
                return {
                    "doc_id": doc_id,
                    "doc_type": current_type["value"],
                    "file_name": f"{doc_id}.docx",
                    "minio_path": f"source/icp/{doc_id}/original.docx",
                    "metadata": {},
                }

        dispatched = []
        diagram_dispatched = []

        def build(doc_type, doc_id, rows):
            dispatched.append((doc_type, doc_id, rows))
            return {
                "schema_version": 1,
                "doc_type": doc_type,
                "doc_id": doc_id,
                "vertices": [],
                "edges": [],
                "warnings": [],
            }

        def build_diagram(doc_type, doc_id, revisions):
            diagram_dispatched.append((doc_type, doc_id, revisions))
            return {
                "schema_version": 1,
                "doc_type": doc_type,
                "doc_id": doc_id,
                "vertices": [],
                "edges": [],
                "warnings": [],
            }

        with patch.object(projection, "postgres_client", Postgres()), patch.object(
            projection, "load_structured_rows", return_value=[]
        ), patch.object(
            projection, "load_formal_diagram_revisions", return_value=[]
        ), patch.object(
            projection, "build_structured_graph", side_effect=build
        ), patch.object(
            projection, "build_formal_diagram_graph", side_effect=build_diagram
        ):
            for doc_type in sorted(projection.PROJECTABLE_DOC_TYPES):
                current_type["value"] = doc_type
                graph = projection.build_document_graph(f"doc-{doc_type}", doc_type)
                documents = [
                    item for item in graph["vertices"] if item["label"] == "Document"
                ]
                self.assertEqual(1, len(documents), doc_type)
                self.assertEqual(
                    f"source/icp/doc-{doc_type}/original.docx",
                    documents[0]["properties"]["storage_key"],
                )

        self.assertEqual(
            set(projection.SUPPORTED_DOC_TYPES),
            {item[0] for item in dispatched},
        )
        self.assertEqual(5, len(dispatched))
        self.assertEqual({"fta", "rbd"}, {item[0] for item in diagram_dispatched})
        self.assertEqual(2, len(diagram_dispatched))
        self.assertEqual(
            {item[0] for item in DOCUMENT_TYPES},
            set(projection.PROJECTABLE_DOC_TYPES),
        )

    def test_physical_ids_do_not_change_with_source_hash(self):
        graph = _logical_graph("doc-a")
        first_vertices, first_edges = _physical(
            graph, doc_id="doc-a", doc_type="hmc", source_hash="a" * 64
        )
        second_vertices, second_edges = _physical(
            graph, doc_id="doc-a", doc_type="hmc", source_hash="b" * 64
        )

        self.assertEqual(
            [item["id"] for item in first_vertices],
            [item["id"] for item in second_vertices],
        )
        self.assertEqual(
            [item["id"] for item in first_edges],
            [item["id"] for item in second_edges],
        )
        self.assertTrue(
            all(
                item["properties"]["kb_id"] == projection.STRUCTURED_KB_ID
                for item in first_vertices + first_edges
            )
        )
        self.assertTrue(
            all(
                "doc-a" in item["properties"]["doc_id"]
                for item in first_vertices + first_edges
            )
        )

    def test_same_business_code_in_different_configurations_has_distinct_physical_id(self):
        graph = _logical_graph("doc-a")
        config_a, _ = _physical(
            graph,
            doc_id="doc-a",
            doc_type="hmc",
            source_hash="a" * 64,
            scope="CONFIG-A",
        )
        config_b, _ = _physical(
            graph,
            doc_id="doc-b",
            doc_type="hmc",
            source_hash="b" * 64,
            scope="CONFIG-B",
        )
        logical_id = business_node_id("hmc", "HMC-001")
        config_a_id = next(
            item["id"]
            for item in config_a
            if f"logical:{logical_id}" in item["properties"]["source_id"]
        )
        config_b_id = next(
            item["id"]
            for item in config_b
            if f"logical:{logical_id}" in item["properties"]["source_id"]
        )
        self.assertNotEqual(config_a_id, config_b_id)

    def test_diagram_document_graph_keeps_revision_provenance(self):
        class Postgres:
            @staticmethod
            def get_icp_document(doc_id):
                return {
                    "doc_id": doc_id,
                    "doc_type": "fta",
                    "file_name": "fta.docx",
                    "metadata": {},
                }

        revision = {
            "doc_id": "doc-fta",
            "diagram_type": "fta",
            "diagram_id": "diagram-1",
            "diagram_key": "FTA-SYN-001",
            "title": "启动故障树",
            "revision_id": "revision-1",
            "revision_no": 1,
            "canonical_hash": "a" * 64,
            "review_status": "pending",
            "validation_status": "auto_validated",
            "validation_tier": "deterministic_structure",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {
                "root_fault_id": "top",
                "metadata": {"business_key": "FTA-SYN-001", "source_refs": ["table=2"]},
                "vertices": [
                    {
                        "id": "top",
                        "label": "TopEvent",
                        "properties": {"name": "无法启动", "event_id": "FTA-SYN-001"},
                    },
                    {
                        "id": "basic",
                        "label": "BasicEvent",
                        "properties": {"name": "电源失效", "event_id": "EVT-SYN-001-BE-01"},
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "label": "causes",
                        "source": "top",
                        "target": "basic",
                        "properties": {"gate_type": "OR"},
                    }
                ],
            },
        }
        with patch.object(projection, "postgres_client", Postgres()), patch.object(
            projection, "load_formal_diagram_revisions", return_value=[revision]
        ):
            graph = projection.build_document_graph("doc-fta", "fta")

        formal_vertices = [
            item
            for item in graph["vertices"]
            if item.get("properties", {}).get("revision_id") == "revision-1"
        ]
        self.assertTrue(formal_vertices)
        extracted = [
            item
            for item in graph["edges"]
            if item["label"] == "extracted_from"
            and item["source"] == formal_vertices[0]["id"]
        ][0]
        self.assertEqual("revision-1", extracted["properties"]["revision_id"])
        self.assertEqual("auto_validated", extracted["properties"]["validation_status"])
        self.assertEqual("published", extracted["properties"]["projection_status"])

        snapshot = _entity_snapshot(graph)
        _vertices, physical_edges = projection._physical_graph(
            graph,
            doc_id="doc-fta",
            doc_type="fta",
            source_hash=content_hash(graph),
            logical_entity_ids=snapshot["logical_entity_ids"],
            identity_scope=snapshot["identity_scope"],
        )
        topology = next(item for item in physical_edges if item["label"] == "decomposed_by")
        self.assertIn("revision-1", topology["properties"]["description_parts"])
        self.assertIn("table=2", topology["properties"]["description_parts"])

    def test_server_generated_edge_id_is_url_encoded_for_delete(self):
        class Response:
            status_code = 204

        class Session:
            def __init__(self):
                self.urls = []

            def delete(self, url):
                self.urls.append(url)
                return Response()

        graph = type(
            "Graph",
            (),
            {
                "session": Session(),
                "_get_graph_url": staticmethod(lambda path: f"http://hugegraph{path}"),
            },
        )()

        with patch.object(projection, "hugegraph_client", graph):
            self.assertTrue(projection._delete_edge("S>1>causes>>T"))

        self.assertEqual(
            ["http://hugegraph/edges/S%3E1%3Ecauses%3E%3ET"],
            graph.session.urls,
        )


class StructuredProjectionPostgresStateTests(unittest.TestCase):
    def test_mark_projection_commits_schema_and_state(self):
        conn = _SqlConnection()
        postgres = type("Postgres", (), {"conn": conn})()

        with patch.object(projection, "postgres_client", postgres):
            projection._mark_projection(
                "doc-a",
                "hmc",
                "a" * 64,
                "published",
                vertex_count=2,
                edge_count=1,
            )

        self.assertEqual(2, conn.commits)
        self.assertEqual(0, conn.rollbacks)
        insert = next(item for item in conn.statements if item[0].startswith("INSERT INTO"))
        self.assertEqual("published", insert[1][4])
        self.assertEqual((2, 1), insert[1][5:7])

    def test_mark_projection_rolls_back_failed_state_write(self):
        conn = _SqlConnection(fail_insert=True)
        postgres = type("Postgres", (), {"conn": conn})()

        with patch.object(projection, "postgres_client", postgres):
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                projection._mark_projection(
                    "doc-a", "hmc", "a" * 64, "running"
                )

        self.assertEqual(1, conn.commits)
        self.assertEqual(1, conn.rollbacks)


class StructuredProjectionLifecycleTests(unittest.TestCase):
    def _patches(self, graph, state, graphs):
        return (
            patch.object(projection, "hugegraph_client", graph),
            patch.object(
                projection,
                "build_document_graph",
                side_effect=lambda doc_id, doc_type: copy.deepcopy(graphs[doc_id]),
            ),
            patch.object(projection, "get_structured_projection", side_effect=state.get),
            patch.object(projection, "_mark_projection", side_effect=state.mark),
            patch.object(
                projection,
                "entity_resolution_repository",
                _EntityRepository(),
            ),
        )

    def test_same_hash_retry_is_readback_only(self):
        # HugeGraph generates edge IDs from endpoints/label; it does not retain
        # the caller's logical edge ID.
        hugegraph = _MemoryHugeGraph(server_edge_ids=True)
        state = _ProjectionState()
        graphs = {"doc-a": _logical_graph("doc-a")}
        contexts = self._patches(hugegraph, state, graphs)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            first = projection.project_structured_document("doc-a", "hmc")
            writes_after_first = list(hugegraph.events)
            marks_after_first = list(state.marks)
            second = projection.project_structured_document("doc-a", "hmc")

        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(writes_after_first, hugegraph.events)
        self.assertEqual(marks_after_first, state.marks)
        self.assertEqual(["running", "published"], [item[1] for item in state.marks])

    def test_cleanup_is_structured_kb_scoped_and_retains_other_doc_contribution(self):
        hugegraph = _MemoryHugeGraph(server_edge_ids=True)
        state = _ProjectionState()
        graphs = {
            "doc-a": _logical_graph("doc-a"),
            "doc-b": _logical_graph("doc-b"),
        }
        revision_id = "icp:revision-1:top"
        hugegraph.vertices[revision_id] = {
            "id": revision_id,
            "label": "Entity",
            "properties": {"kb_id": "doc-a", "doc_id": ["doc-a"]},
        }
        revision_edge_id = "S>revision-causes>>T"
        hugegraph.edges[revision_edge_id] = {
            "id": revision_edge_id,
            "label": "causes",
            "outV": revision_id,
            "inV": revision_id,
            "properties": {"kb_id": "doc-a", "doc_id": ["doc-a"]},
        }
        contexts = self._patches(hugegraph, state, graphs)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            projection.project_structured_document("doc-a", "hmc")
            projection.project_structured_document("doc-b", "hmc")

            shared_id = f"icp:ce:{_canonical_id(business_node_id('hmc', 'HMC-001'))}"
            self.assertEqual(
                ["doc-a", "doc-b"],
                hugegraph.vertices[shared_id]["properties"]["doc_id"],
            )

            event_start = len(hugegraph.events)
            graphs["doc-a"] = _logical_graph("doc-a", include_shared=False)
            projection.project_structured_document("doc-a", "hmc")

        self.assertIn(revision_id, hugegraph.vertices)
        self.assertIn(revision_edge_id, hugegraph.edges)
        self.assertEqual(
            ["doc-b"], hugegraph.vertices[shared_id]["properties"]["doc_id"]
        )
        self.assertEqual(
            ["doc-b"],
            [
                part["doc_id"]
                for part in projection._parts(
                    hugegraph.vertices[shared_id]["properties"]["description_parts"]
                )
            ],
        )
        self.assertTrue(
            any(
                "logical:sg:e:explicit:doc-b"
                in item.get("properties", {}).get("source_id", [])
                for item in hugegraph.edges.values()
            )
        )

        cleanup_events = hugegraph.events[event_start:]
        edge_positions = [
            index
            for index, event in enumerate(cleanup_events)
            if event[0] in {"delete_edge", "replace_edge"}
        ]
        vertex_positions = [
            index
            for index, event in enumerate(cleanup_events)
            if event[0] in {"delete_vertex", "replace_vertex"}
        ]
        self.assertTrue(edge_positions)
        self.assertTrue(vertex_positions)
        self.assertLess(max(edge_positions), min(vertex_positions))

    def test_shared_single_edge_retains_both_docs_when_one_is_reprojected(self):
        hugegraph = _MemoryHugeGraph(server_edge_ids=True)
        state = _ProjectionState()
        graphs = {
            "doc-a": _shared_relation_graph("doc-a"),
            "doc-b": _shared_relation_graph("doc-b"),
        }
        contexts = self._patches(hugegraph, state, graphs)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            projection.project_structured_document("doc-a", "hmc")
            projection.project_structured_document("doc-b", "hmc")
            self.assertEqual(1, len(hugegraph.edges))
            shared = next(iter(hugegraph.edges.values()))
            self.assertEqual(["doc-a", "doc-b"], shared["properties"]["doc_id"])

            graphs["doc-a"] = _shared_relation_graph(
                "doc-a", edge_suffix="replacement"
            )
            projection.project_structured_document("doc-a", "hmc")

        self.assertEqual(1, len(hugegraph.edges))
        shared = next(iter(hugegraph.edges.values()))
        parts = projection._parts(shared["properties"]["description_parts"])
        self.assertEqual({"doc-a", "doc-b"}, {part["doc_id"] for part in parts})
        self.assertIn(
            "logical:sg:e:doc-b:initial", shared["properties"]["source_id"]
        )
        self.assertIn(
            "logical:sg:e:doc-a:replacement", shared["properties"]["source_id"]
        )
        self.assertNotIn(
            "logical:sg:e:doc-a:initial", shared["properties"]["source_id"]
        )

    def test_global_projection_lock_serializes_shared_vertex_updates(self):
        hugegraph = _MemoryHugeGraph(server_edge_ids=True)
        state = _ProjectionState()
        graphs = {
            "doc-a": _logical_graph("doc-a"),
            "doc-b": _logical_graph("doc-b"),
        }
        activity_lock = threading.Lock()
        activity = {"current": 0, "maximum": 0}

        def build(doc_id, doc_type):
            with activity_lock:
                activity["current"] += 1
                activity["maximum"] = max(activity["maximum"], activity["current"])
            time.sleep(0.01)
            with activity_lock:
                activity["current"] -= 1
            return copy.deepcopy(graphs[doc_id])

        with patch.object(projection, "hugegraph_client", hugegraph), patch.object(
            projection, "build_document_graph", side_effect=build
        ), patch.object(
            projection, "get_structured_projection", side_effect=state.get
        ), patch.object(projection, "_mark_projection", side_effect=state.mark), patch.object(
            projection, "entity_resolution_repository", _EntityRepository()
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda doc_id: projection.project_structured_document(
                            doc_id, "hmc"
                        ),
                        ("doc-a", "doc-b"),
                    )
                )

        shared_id = f"icp:ce:{_canonical_id(business_node_id('hmc', 'HMC-001'))}"
        self.assertEqual(1, activity["maximum"])
        self.assertEqual(["published", "published"], [item["status"] for item in results])
        self.assertEqual(
            ["doc-a", "doc-b"],
            hugegraph.vertices[shared_id]["properties"]["doc_id"],
        )

    def test_batch_failure_is_recorded_as_failed_after_running(self):
        hugegraph = _MemoryHugeGraph(fail_batch=True, partial_batch=True)
        state = _ProjectionState()
        graphs = {"doc-a": _logical_graph("doc-a")}
        contexts = self._patches(hugegraph, state, graphs)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            with self.assertRaisesRegex(RuntimeError, "批量写入失败"):
                projection.project_structured_document("doc-a", "hmc")

        self.assertEqual(["running", "failed"], [item[1] for item in state.marks])
        self.assertEqual("failed", state.rows["doc-a"]["status"])
        self.assertIn("批量写入失败", state.rows["doc-a"]["error"])
        self.assertFalse(
            any(
                "doc-a" in item.get("properties", {}).get("doc_id", [])
                for item in list(hugegraph.vertices.values())
                + list(hugegraph.edges.values())
            )
        )

    def test_partial_update_failure_restores_previous_published_snapshot(self):
        hugegraph = _MemoryHugeGraph(server_edge_ids=True)
        state = _ProjectionState()
        graphs = {"doc-a": _logical_graph("doc-a", shared_code="HMC-001")}
        contexts = self._patches(hugegraph, state, graphs)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            projection.project_structured_document("doc-a", "hmc")
            previous_vertices = copy.deepcopy(hugegraph.vertices)
            previous_edges = copy.deepcopy(hugegraph.edges)

            graphs["doc-a"] = _logical_graph("doc-a", shared_code="HMC-002")
            hugegraph.fail_batch = True
            hugegraph.fail_once = True
            hugegraph.partial_batch = True
            with self.assertRaisesRegex(RuntimeError, "批量写入失败"):
                projection.project_structured_document("doc-a", "hmc")

        self.assertEqual(previous_vertices, hugegraph.vertices)
        self.assertEqual(previous_edges, hugegraph.edges)
        self.assertEqual("failed", state.rows["doc-a"]["status"])

    def test_readback_checks_ids_not_only_counts(self):
        hugegraph = _MemoryHugeGraph()
        graph = _logical_graph("doc-a")
        vertices, edges = _physical(
            graph,
            doc_id="doc-a",
            doc_type="hmc",
            source_hash=content_hash(graph),
        )
        for item in vertices:
            hugegraph.vertices[item["id"]] = copy.deepcopy(item)
        for item in edges:
            hugegraph.edges[item["id"]] = copy.deepcopy(item)
        removed = hugegraph.vertices.pop(vertices[0]["id"])
        removed["id"] = "icp:sg:different-id"
        hugegraph.vertices[removed["id"]] = removed

        with patch.object(projection, "hugegraph_client", hugegraph):
            matches, actual_vertices, actual_edges = projection._readback_matches(
                "doc-a", vertices, edges
            )

        self.assertFalse(matches)
        self.assertEqual(len(vertices), actual_vertices)
        self.assertEqual(len(edges), actual_edges)

    def test_readback_rejects_same_ids_with_stale_source_hash(self):
        hugegraph = _MemoryHugeGraph()
        graph = _logical_graph("doc-a")
        vertices, edges = _physical(
            graph,
            doc_id="doc-a",
            doc_type="hmc",
            source_hash=content_hash(graph),
        )
        for item in vertices:
            hugegraph.vertices[item["id"]] = copy.deepcopy(item)
        for item in edges:
            hugegraph.edges[item["id"]] = copy.deepcopy(item)
        first_vertex = hugegraph.vertices[vertices[0]["id"]]
        first_vertex["properties"]["source_id"] = [
            value
            for value in first_vertex["properties"]["source_id"]
            if not value.startswith("projection:doc-a:")
        ]

        with patch.object(projection, "hugegraph_client", hugegraph):
            matches, actual_vertices, actual_edges = projection._readback_matches(
                "doc-a", vertices, edges
            )

        self.assertFalse(matches)
        self.assertEqual(len(vertices), actual_vertices)
        self.assertEqual(len(edges), actual_edges)

    def test_readback_recovers_expected_vertex_omitted_by_index_scan(self):
        hugegraph = _MemoryHugeGraph()
        graph = _logical_graph("doc-a")
        vertices, edges = _physical(
            graph,
            doc_id="doc-a",
            doc_type="hmc",
            source_hash=content_hash(graph),
        )
        hugegraph.vertices = {
            item["id"]: copy.deepcopy(item) for item in vertices
        }
        hugegraph.edges = {item["id"]: copy.deepcopy(item) for item in edges}
        complete_lookup = hugegraph.get_vertex_by_id
        omitted_id = vertices[0]["id"]

        def stale_index_scan(kb_id):
            return [
                copy.deepcopy(item)
                for item_id, item in hugegraph.vertices.items()
                if item_id != omitted_id
                and (item.get("properties") or {}).get("kb_id") == kb_id
            ]

        hugegraph.get_vertices_by_kb_with_details = stale_index_scan
        hugegraph.get_vertex_by_id = complete_lookup
        with patch.object(projection, "hugegraph_client", hugegraph):
            matches, actual_vertices, actual_edges = projection._readback_matches(
                "doc-a", vertices, edges
            )

        self.assertTrue(matches)
        self.assertEqual(len(vertices), actual_vertices)
        self.assertEqual(len(edges), actual_edges)

    def test_readback_rejects_missing_logical_contribution_with_same_counts(self):
        graph = {
            "vertices": [
                {"id": "sg:o:a1", "label": "FTAEvent", "properties": {"name": "A"}},
                {"id": "sg:o:a2", "label": "FTAEvent", "properties": {"name": "A"}},
                {"id": "sg:o:b", "label": "FTAGate", "properties": {"name": "B"}},
            ],
            "edges": [
                {"id": "sg:e:1", "label": "decomposed_by", "source": "sg:o:a1", "target": "sg:o:b"},
                {"id": "sg:e:2", "label": "decomposed_by", "source": "sg:o:a2", "target": "sg:o:b"},
            ],
        }
        mapping = {
            "sg:o:a1": "entity-a",
            "sg:o:a2": "entity-a",
            "sg:o:b": "entity-b",
        }
        vertices, edges = projection._physical_graph(
            graph,
            doc_id="doc-fta",
            doc_type="fta",
            source_hash="a" * 64,
            logical_entity_ids=mapping,
            identity_scope={"configuration": "CONFIG-A"},
        )
        hugegraph = _MemoryHugeGraph()
        hugegraph.vertices = {item["id"]: copy.deepcopy(item) for item in vertices}
        hugegraph.edges = {item["id"]: copy.deepcopy(item) for item in edges}
        edge = next(iter(hugegraph.edges.values()))
        parts = projection._parts(edge["properties"]["description_parts"])
        edge["properties"]["description_parts"] = json.dumps(
            parts[:1], ensure_ascii=False
        )

        with patch.object(projection, "hugegraph_client", hugegraph):
            matches, actual_vertices, actual_edges = projection._readback_matches(
                "doc-fta", vertices, edges
            )

        self.assertFalse(matches)
        self.assertEqual(len(vertices), actual_vertices)
        self.assertEqual(len(edges), actual_edges)


if __name__ == "__main__":
    unittest.main()
