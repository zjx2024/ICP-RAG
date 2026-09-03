import json
import asyncio
import unittest

from backend.agent.tools.graph_traversal import (
    HARD_MAX_DEPTH,
    STRUCTURED_KB_ID,
    graph_traversal_node,
    run_graph_traversal,
)


def _structured_vertex(logical_id, name, *, doc_id="doc-1", source_ref=""):
    # Structured business nodes are projected through the canonical entity
    # registry.  The logical ID remains in ``source_id`` for exact resolution,
    # while the HugeGraph vertex ID is the canonical-entity namespace.
    physical_id = f"icp:ce:{logical_id}"
    summary = {
        "business_id": logical_id.rsplit(":", 1)[-1],
        "identity_scope": "business_id",
        "doc_ids": [doc_id],
        "source_refs": [source_ref] if source_ref else [],
    }
    contribution = {
        "doc_id": doc_id,
        "doc_type": "function_locator",
        "source_hash": f"hash-{doc_id}",
        "name": name,
        "entity_type": "Component",
        "summary": json.dumps(summary, ensure_ascii=False),
    }
    return {
        "id": physical_id,
        "label": "Entity",
        "properties": {
            "name": name,
            "entity_type": "Component",
            "description": name,
            "summary": json.dumps(summary, ensure_ascii=False),
            "doc_id": [doc_id],
            "kb_id": STRUCTURED_KB_ID,
            "project_id": STRUCTURED_KB_ID,
            "tenant_id": STRUCTURED_KB_ID,
            "source_id": [
                f"structured-doc:{doc_id}",
                f"logical:{logical_id}",
                f"projection:{doc_id}:hash-{doc_id}",
            ],
            "description_parts": json.dumps([contribution], ensure_ascii=False),
        },
    }


def _structured_edge(
    logical_id,
    source_logical,
    target_logical,
    relation,
    *,
    doc_id="doc-1",
    source_ref="",
):
    contribution = {
        "doc_id": doc_id,
        "doc_type": "function_locator",
        "source_hash": f"hash-{doc_id}",
        "text": relation,
        "source_refs": [source_ref] if source_ref else [],
    }
    return {
        "id": f"icp:sg:{logical_id}",
        "label": relation,
        "outV": f"icp:ce:{source_logical}",
        "inV": f"icp:ce:{target_logical}",
        "properties": {
            "description": relation,
            "doc_id": [doc_id],
            "kb_id": STRUCTURED_KB_ID,
            "project_id": "structured:function_locator",
            "tenant_id": STRUCTURED_KB_ID,
            "source_id": [
                f"structured-doc:{doc_id}",
                f"logical:{logical_id}",
                f"projection:{doc_id}:hash-{doc_id}",
            ],
            "description_parts": json.dumps([contribution], ensure_ascii=False),
        },
    }


def _engineering_vertex(logical_id, name, *, revision="rev-1", status="auto_validated"):
    return {
        "id": f"icp:{revision}:v:{logical_id}",
        "label": "Entity",
        "properties": {
            "name": name,
            "entity_type": "FTAEvent",
            "doc_id": ["doc-fta"],
            "kb_id": "fta",
            "project_id": "diagram-1",
            "tenant_id": revision,
            "source_id": [
                "doc-fta",
                "diagram:diagram-1",
                f"revision:{revision}",
                "canonical:hash-fta",
                f"validation:{status}",
                "validation_tier:deterministic",
            ],
        },
    }


def _engineering_edge(
    logical_id,
    source_logical,
    target_logical,
    *,
    revision="rev-1",
    status="auto_validated",
):
    return {
        "id": f"icp:{revision}:e:{logical_id}",
        "label": "has_input_event",
        "outV": f"icp:{revision}:v:{source_logical}",
        "inV": f"icp:{revision}:v:{target_logical}",
        "properties": {
            "doc_id": ["doc-fta"],
            "kb_id": "fta",
            "project_id": "diagram-1",
            "tenant_id": revision,
            "source_id": [
                "doc-fta",
                "diagram:diagram-1",
                f"revision:{revision}",
                "canonical:hash-fta",
                f"validation:{status}",
                "validation_tier:deterministic",
            ],
        },
    }


class _GraphClient:
    _use_mock = False

    def __init__(self, by_kb, *, reverse=False, fail=False):
        self.by_kb = by_kb
        self.reverse = reverse
        self.fail = fail

    def get_vertices_by_kb_with_details(self, kb_id):
        if self.fail:
            raise RuntimeError("HugeGraph offline")
        values = list(self.by_kb.get(kb_id, {}).get("vertices", []))
        return list(reversed(values)) if self.reverse else values

    def get_edges_by_kb_with_details(self, kb_id):
        if self.fail:
            raise RuntimeError("HugeGraph offline")
        values = list(self.by_kb.get(kb_id, {}).get("edges", []))
        return list(reversed(values)) if self.reverse else values


class _MockGraphClient(_GraphClient):
    _use_mock = True


def _structured_client(*, reverse=False):
    vertices = [
        _structured_vertex("sg:b:component:a", "系统", source_ref="架构.docx#A"),
        _structured_vertex("sg:b:component:b", "子系统", source_ref="架构.docx#B"),
        _structured_vertex("sg:b:component:c", "模块", source_ref="架构.docx#C"),
        _structured_vertex("sg:b:component:d", "备用模块", source_ref="架构.docx#D"),
    ]
    edges = [
        _structured_edge(
            "sg:e:contains:a-b",
            "sg:b:component:a",
            "sg:b:component:b",
            "contains_component",
            source_ref="架构.docx#table=1;row=1",
        ),
        _structured_edge(
            "sg:e:contains:b-c",
            "sg:b:component:b",
            "sg:b:component:c",
            "contains_component",
            source_ref="架构.docx#table=1;row=2",
        ),
        _structured_edge(
            "sg:e:contains:a-d",
            "sg:b:component:a",
            "sg:b:component:d",
            "contains_component",
            source_ref="架构.docx#table=1;row=3",
        ),
        _structured_edge(
            "sg:e:backup:d-c",
            "sg:b:component:d",
            "sg:b:component:c",
            "backup_for",
            source_ref="架构.docx#table=1;row=4",
        ),
    ]
    return _GraphClient(
        {STRUCTURED_KB_ID: {"vertices": vertices, "edges": edges}},
        reverse=reverse,
    )


def _provenance_bridge_client():
    vertices = [
        _structured_vertex("sg:b:component:a", "故障模块"),
        _structured_vertex("sg:b:document:doc-1", "来源文档"),
        _structured_vertex("sg:b:component:unrelated", "无关模块"),
    ]
    edges = [
        _structured_edge(
            "sg:e:extracted:a-doc",
            "sg:b:component:a",
            "sg:b:document:doc-1",
            "extracted_from",
        ),
        _structured_edge(
            "sg:e:extracted:unrelated-doc",
            "sg:b:component:unrelated",
            "sg:b:document:doc-1",
            "extracted_from",
        ),
    ]
    return _GraphClient(
        {STRUCTURED_KB_ID: {"vertices": vertices, "edges": edges}}
    )


def _merged_single_edge_client(*, tampered_marker=False):
    def vertex(physical_id, logical_ids, name):
        contributions = [
            {
                "doc_id": "doc-1",
                "doc_type": "function_locator",
                "source_hash": "hash-doc-1",
                "logical_id": logical_id,
                "name": name,
                "entity_type": "Component",
                "summary": json.dumps({"name": name}, ensure_ascii=False),
            }
            for logical_id in logical_ids
        ]
        return {
            "id": physical_id,
            "label": "Entity",
            "properties": {
                "name": name,
                "entity_type": "Component",
                "description": name,
                "doc_id": ["doc-1"],
                "kb_id": STRUCTURED_KB_ID,
                "project_id": STRUCTURED_KB_ID,
                "tenant_id": STRUCTURED_KB_ID,
                "source_id": [
                    "structured-doc:doc-1",
                    "projection:doc-1:hash-doc-1",
                    *[f"logical:{logical_id}" for logical_id in logical_ids],
                ],
                "description_parts": json.dumps(contributions, ensure_ascii=False),
            },
        }

    edge_logical_ids = ["sg:e:merged:1", "sg:e:merged:2"]
    source_ids = [
        "structured-doc:doc-1",
        "projection:doc-1:hash-doc-1",
        *[f"logical:{logical_id}" for logical_id in edge_logical_ids],
    ]
    if tampered_marker:
        source_ids.append("logical:sg:e:forged")
    edge_contributions = [
        {
            "doc_id": "doc-1",
            "doc_type": "function_locator",
            "source_hash": "hash-doc-1",
            "logical_id": logical_id,
            "source_logical_id": f"sg:o:source:{index}",
            "target_logical_id": f"sg:o:target:{index}",
            "source_relation": "contains_component",
            "text": "contains_component",
        }
        for index, logical_id in enumerate(edge_logical_ids, start=1)
    ]
    return _GraphClient(
        {
            STRUCTURED_KB_ID: {
                "vertices": [
                    vertex(
                        "icp:ce:entity-source",
                        ["sg:o:source:1", "sg:o:source:2"],
                        "源实体",
                    ),
                    vertex(
                        "icp:ce:entity-target",
                        ["sg:o:target:1", "sg:o:target:2"],
                        "目标实体",
                    ),
                ],
                "edges": [
                    {
                        # HugeGraph SINGLE edges use a server-composed ID; it
                        # deliberately has no icp:sg prefix.
                        "id": "S1:entity-source>contains_component>>S1:entity-target",
                        "label": "contains_component",
                        "outV": "icp:ce:entity-source",
                        "inV": "icp:ce:entity-target",
                        "properties": {
                            "description": "contains_component",
                            "doc_id": ["doc-1"],
                            "kb_id": STRUCTURED_KB_ID,
                            "project_id": STRUCTURED_KB_ID,
                            "tenant_id": STRUCTURED_KB_ID,
                            "source_id": source_ids,
                            "description_parts": json.dumps(
                                edge_contributions, ensure_ascii=False
                            ),
                        },
                    }
                ],
            }
        }
    )


class GraphTraversalTests(unittest.TestCase):
    def test_server_generated_merged_edge_is_visible_and_reauthorized_as_one_set(self):
        captured = []

        def authorize(requests):
            captured.extend(requests)
            return {
                request["authorization_key"]: {"authorized": True}
                for request in requests
            }

        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["icp:ce:entity-source"],
                "direction": "out",
                "require_relation_reauthorization": True,
            },
            client=_merged_single_edge_client(),
            reauthorize_relations=authorize,
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(1, len(result["edges"]))
        self.assertEqual(1, len(captured))
        self.assertEqual(
            ["sg:e:merged:1", "sg:e:merged:2"],
            captured[0]["logical_edge_ids"],
        )
        self.assertEqual(
            {"doc-1": ["sg:e:merged:1", "sg:e:merged:2"]},
            captured[0]["logical_edge_ids_by_doc"],
        )
        self.assertEqual(
            "icp:ce:entity-source", captured[0]["source_physical_id"]
        )
        self.assertEqual(
            "icp:ce:entity-target", captured[0]["target_physical_id"]
        )

    def test_unattributed_logical_marker_is_rejected_before_authorization(self):
        captured = []
        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["icp:ce:entity-source"],
                "direction": "out",
                "require_relation_reauthorization": True,
            },
            client=_merged_single_edge_client(tampered_marker=True),
            reauthorize_relations=lambda requests: captured.extend(requests) or {},
        )

        self.assertEqual("empty", result["status"])
        self.assertEqual([], captured)

    def test_default_inference_excludes_document_provenance_bridges(self):
        captured = []
        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["sg:b:component:a"],
                "direction": "both",
                "max_depth": 2,
                "require_relation_reauthorization": True,
            },
            client=_provenance_bridge_client(),
            reauthorize_relations=lambda requests: captured.extend(requests) or {},
        )

        self.assertEqual("empty", result["status"])
        self.assertEqual([], result["paths"])
        self.assertEqual([], result["edges"])
        self.assertNotIn("无关模块", {item["name"] for item in result["nodes"]})
        self.assertEqual(["extracted_from"], result["request"]["excluded_relations"])
        self.assertEqual([], captured)

        provenance_query = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["sg:b:component:a"],
                "direction": "both",
                "relations": ["extracted_from"],
                "max_depth": 2,
            },
            client=_provenance_bridge_client(),
        )
        self.assertEqual("ok", provenance_query["status"])
        self.assertIn(
            "无关模块", {item["name"] for item in provenance_query["nodes"]}
        )

    def test_neighbors_returns_stable_multihop_paths_and_provenance(self):
        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["icp:ce:sg:b:component:a"],
                "direction": "out",
                "max_depth": 2,
            },
            client=_structured_client(),
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(4, len(result["nodes"]))
        self.assertEqual(3, len(result["paths"]))
        module_path = next(
            path for path in result["paths"] if path["end_id"].endswith("component:c")
        )
        self.assertEqual(2, module_path["depth"])
        self.assertEqual(
            ["sg:e:contains:a-b", "sg:e:contains:b-c"],
            [step["logical_edge_id"] for step in module_path["steps"]],
        )
        self.assertEqual([], module_path["source_refs"])
        self.assertIn(
            "架构.docx#table=1;row=2",
            module_path["projection_source_refs"],
        )
        node = next(item for item in result["nodes"] if item["name"] == "模块")
        self.assertEqual("sg:b:component:c", node["logical_id"])
        self.assertEqual("c", node["business_id"])
        evidence = next(
            item
            for item in result["evidence"]
            if item["record_id"] == module_path["path_id"]
        )
        self.assertIn("系统 —[contains_component]→ 子系统", evidence["display_summary"])
        self.assertEqual("validated_relation", evidence["authority"])
        self.assertEqual("auto_validated", evidence["review_status"])

    def test_path_output_is_stable_when_hugegraph_readback_order_changes(self):
        request = {
            "mode": "path",
            "seed_ids": ["icp:ce:sg:b:component:a"],
            "target_ids": ["icp:ce:sg:b:component:c"],
            "direction": "out",
            "max_depth": 3,
        }
        first = run_graph_traversal(request, client=_structured_client())
        second = run_graph_traversal(request, client=_structured_client(reverse=True))

        self.assertEqual("ok", first["status"])
        self.assertEqual(
            [path["path_id"] for path in first["paths"]],
            [path["path_id"] for path in second["paths"]],
        )
        self.assertEqual(
            [path["node_ids"] for path in first["paths"]],
            [path["node_ids"] for path in second["paths"]],
        )
        self.assertEqual(2, len(first["paths"]))

    def test_unambiguous_logical_seed_and_target_ids_are_resolved(self):
        result = run_graph_traversal(
            {
                "mode": "path",
                "seed_ids": ["sg:b:component:a"],
                "target_ids": ["sg:b:component:c"],
                "direction": "out",
                "max_depth": 3,
            },
            client=_structured_client(),
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(["icp:ce:sg:b:component:a"], result["request"]["seed_ids"])
        self.assertEqual(["icp:ce:sg:b:component:c"], result["request"]["target_ids"])

    def test_subgraph_includes_non_tree_edges_and_obeys_edge_limit(self):
        seed = "icp:ce:sg:b:component:a"
        full = run_graph_traversal(
            {
                "mode": "subgraph",
                "seed_ids": [seed],
                "direction": "out",
                "max_depth": 2,
            },
            client=_structured_client(),
        )
        self.assertEqual(4, len(full["edges"]))
        self.assertIn("backup_for", {edge["relation"] for edge in full["edges"]})

        limited = run_graph_traversal(
            {
                "mode": "subgraph",
                "seed_ids": [seed],
                "direction": "out",
                "max_depth": 2,
                "max_edges": 2,
            },
            client=_structured_client(),
        )
        self.assertEqual("partial", limited["status"])
        self.assertTrue(limited["truncated"])
        self.assertEqual(2, len(limited["edges"]))

    def test_relation_and_direction_filters_are_applied_before_traversal(self):
        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["icp:ce:sg:b:component:c"],
                "direction": "in",
                "relations": ["contains_component"],
                "max_depth": 2,
            },
            client=_structured_client(),
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(
            {"系统", "子系统", "模块"}, {item["name"] for item in result["nodes"]}
        )
        self.assertNotIn("backup_for", {edge["relation"] for edge in result["edges"]})
        self.assertTrue(
            all(
                step["traversal_direction"] == "reverse"
                for path in result["paths"]
                for step in path["steps"]
            )
        )

    def test_engineering_edges_require_current_published_pg_reauthorization(self):
        vertices = [
            _engineering_vertex("top", "顶事件"),
            _engineering_vertex("basic", "基本事件"),
        ]
        edge = _engineering_edge("edge-1", "top", "basic")
        client = _GraphClient({"fta": {"vertices": vertices, "edges": [edge]}})
        request = {
            "mode": "neighbors",
            "kb_ids": ["fta"],
            "seed_ids": ["icp:rev-1:v:top"],
            "direction": "out",
        }

        excluded = run_graph_traversal(request, client=client)
        self.assertEqual("empty", excluded["status"])
        self.assertEqual([], excluded["edges"])

        captured = []

        def authorize(assertions):
            captured.extend(assertions)
            assertion = assertions[0]
            return {
                assertion["authorization_key"]: {
                    "authorized": True,
                    "validation_status": "auto_validated",
                    "projection_status": "published",
                    "is_active": True,
                    "source_refs": ["04_FTA.docx#figure=1"],
                }
            }

        accepted = run_graph_traversal(
            request, client=client, reauthorize_relations=authorize
        )
        self.assertEqual("ok", accepted["status"])
        self.assertEqual("edge-1", captured[0]["logical_edge_id"])
        self.assertEqual("top", captured[0]["source_logical_id"])
        self.assertEqual("basic", captured[0]["target_logical_id"])
        self.assertTrue(accepted["edges"][0]["reauthorized"])
        self.assertTrue(accepted["nodes"][0]["is_active"])
        self.assertIn("04_FTA.docx#figure=1", accepted["paths"][0]["source_refs"])

    def test_provisional_engineering_relation_never_reaches_reauthorizer(self):
        vertices = [
            _engineering_vertex("top", "顶事件", status="provisional"),
            _engineering_vertex("basic", "基本事件", status="provisional"),
        ]
        edge = _engineering_edge(
            "edge-1", "top", "basic", status="provisional"
        )
        client = _GraphClient({"fta": {"vertices": vertices, "edges": [edge]}})
        called = []

        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "kb_ids": ["fta"],
                "seed_ids": ["icp:rev-1:v:top"],
            },
            client=client,
            reauthorize_relations=lambda assertions: called.extend(assertions) or {},
        )
        self.assertEqual("empty", result["status"])
        self.assertEqual([], called)
        self.assertGreaterEqual(result["diagnostics"]["rejected_items"], 3)

    def test_callback_failure_closes_query_without_partial_structured_data(self):
        def fail(_assertions):
            raise RuntimeError("PostgreSQL reauthorization unavailable")

        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["icp:ce:sg:b:component:a"],
                "require_relation_reauthorization": True,
            },
            client=_structured_client(),
            reauthorize_relations=fail,
        )
        self.assertEqual("failed", result["status"])
        self.assertEqual([], result["nodes"])
        self.assertEqual([], result["paths"])
        self.assertIn("reauthorization unavailable", result["diagnostics"]["errors"][0])

    def test_structured_assertions_can_be_reauthorized_by_logical_edge_id(self):
        captured = []

        def authorize(assertions):
            captured.extend(assertions)
            return {
                item["logical_edge_id"]: {
                    "authorized": True,
                    "source_refs": [f"PG:{item['logical_edge_id']}"],
                }
                for item in assertions
            }

        result = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["sg:b:component:a"],
                "direction": "out",
                "max_depth": 1,
                "require_relation_reauthorization": True,
            },
            client=_structured_client(),
            reauthorize_relations=authorize,
        )
        self.assertEqual("ok", result["status"])
        self.assertTrue(captured)
        self.assertTrue(all(item["logical_edge_id"].startswith("sg:e:") for item in captured))
        self.assertTrue(all(edge["reauthorized"] for edge in result["edges"]))
        self.assertTrue(any(ref.startswith("PG:sg:e:") for ref in result["paths"][0]["source_refs"]))

    def test_three_hop_formal_evidence_excludes_endpoint_and_projection_sources(self):
        client = _structured_client()
        graph = client.by_kb[STRUCTURED_KB_ID]
        # Force the deterministic three-hop path A -> B -> C <- D.
        graph["edges"] = [
            edge
            for edge in graph["edges"]
            if "logical:sg:e:contains:a-d"
            not in edge["properties"]["source_id"]
        ]

        target = next(
            vertex
            for vertex in graph["vertices"]
            if vertex["id"] == "icp:ce:sg:b:component:d"
        )
        props = target["properties"]
        props["doc_id"].append("doc-fta-unrelated")
        props["source_id"].extend(
            [
                "structured-doc:doc-fta-unrelated",
                "projection:doc-fta-unrelated:hash-fta-unrelated",
            ]
        )
        contributions = json.loads(props["description_parts"])
        contributions.append(
            {
                "doc_id": "doc-fta-unrelated",
                "doc_type": "fta",
                "source_hash": "hash-fta-unrelated",
                "name": "备用模块",
                "entity_type": "Component",
                "source_refs": ["04_FTA无关节点贡献.docx#figure=9"],
                "summary": json.dumps(
                    {
                        "doc_ids": ["doc-fta-unrelated"],
                        "source_refs": ["04_FTA无关节点贡献.docx#figure=9"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
        props["description_parts"] = json.dumps(
            contributions, ensure_ascii=False
        )

        def authorize(requests):
            return {
                request["authorization_key"]: {
                    "authorized": True,
                    "doc_ids": ["doc-1"],
                    "source_refs": [f"PG:{request['logical_edge_id']}"],
                }
                for request in requests
            }

        result = run_graph_traversal(
            {
                "mode": "path",
                "seed_ids": ["sg:b:component:a"],
                "target_ids": ["sg:b:component:d"],
                "direction": "both",
                "relations": ["contains_component", "backup_for"],
                "max_depth": 3,
                "require_relation_reauthorization": True,
            },
            client=client,
            reauthorize_relations=authorize,
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(1, len(result["paths"]))
        path = result["paths"][0]
        self.assertEqual(3, path["depth"])
        self.assertTrue(path["reauthorized"])
        self.assertEqual(["doc-1"], path["doc_ids"])
        self.assertTrue(path["source_refs"])
        self.assertTrue(all(ref.startswith("PG:") for ref in path["source_refs"]))
        self.assertNotIn(
            "04_FTA无关节点贡献.docx#figure=9", path["source_refs"]
        )
        self.assertIn("doc-fta-unrelated", path["endpoint_doc_ids"])
        self.assertIn(
            "04_FTA无关节点贡献.docx#figure=9",
            path["endpoint_source_refs"],
        )
        self.assertTrue(
            any(
                ref.startswith("架构.docx#table=")
                for ref in path["projection_source_refs"]
            )
        )
        self.assertTrue(
            all(
                all(ref.startswith("PG:") for ref in edge["source_refs"])
                for edge in result["edges"]
            )
        )

        evidence = result["evidence"][0]
        self.assertNotIn("04_FTA无关节点贡献", evidence["source_ref"])
        self.assertEqual(["doc-1"], evidence["metadata"]["doc_ids"])
        self.assertIn(
            "doc-fta-unrelated",
            evidence["metadata"]["endpoint_doc_ids"],
        )
        self.assertIn(
            "04_FTA无关节点贡献.docx#figure=9",
            evidence["metadata"]["endpoint_source_refs"],
        )
        answer_fields = evidence["metadata"]["answer_fields"]
        self.assertEqual("系统", answer_fields["source_name"])
        self.assertEqual("备用模块", answer_fields["target_name"])
        self.assertEqual(
            ["系统", "子系统", "模块", "备用模块"],
            answer_fields["path_nodes"],
        )
        self.assertEqual(3, len(answer_fields["path_relations"]))
        self.assertIn("系统", answer_fields["path_summary"])
        self.assertIn("备用模块", answer_fields["path_summary"])

    def test_missing_required_callback_backend_failure_and_mock_fail_closed(self):
        request = {
            "mode": "neighbors",
            "seed_ids": ["icp:ce:sg:b:component:a"],
            "require_relation_reauthorization": True,
        }
        missing = run_graph_traversal(request, client=_structured_client())
        self.assertEqual("failed", missing["status"])
        self.assertEqual([], missing["edges"])

        backend = run_graph_traversal(
            {**request, "require_relation_reauthorization": False},
            client=_GraphClient({}, fail=True),
        )
        self.assertEqual("failed", backend["status"])
        self.assertEqual([], backend["evidence"])

        mocked = run_graph_traversal(
            {**request, "require_relation_reauthorization": False},
            client=_MockGraphClient({}),
        )
        self.assertEqual("failed", mocked["status"])
        self.assertIn("内存 Mock", mocked["diagnostics"]["errors"][0])

    def test_invalid_mode_and_limit_are_blocked_before_backend_read(self):
        invalid_mode = run_graph_traversal(
            {"mode": "free_gremlin", "seed_ids": ["v"]},
            client=_structured_client(),
        )
        self.assertEqual("blocked", invalid_mode["status"])

        invalid_limit = run_graph_traversal(
            {
                "mode": "neighbors",
                "seed_ids": ["v"],
                "max_depth": HARD_MAX_DEPTH + 1,
            },
            client=_structured_client(),
        )
        self.assertEqual("blocked", invalid_limit["status"])
        self.assertEqual([], invalid_limit["nodes"])

    def test_langgraph_adapter_emits_tool_result_and_clickable_evidence(self):
        update = asyncio.run(
            graph_traversal_node(
                {
                    "graph_traversal_request": {
                        "mode": "neighbors",
                        "seed_ids": ["sg:b:component:a"],
                        "direction": "out",
                        "max_depth": 1,
                    }
                },
                client=_structured_client(),
            )
        )
        self.assertEqual("graph_traversal", update["tool_results"][0]["tool"])
        self.assertEqual("ok", update["tool_results"][0]["status"])
        self.assertTrue(update["evidence_registry"])
        self.assertEqual(
            update["evidence_registry"][0]["evidence_id"],
            update["tool_calls"][0]["results_preview"][0]["evidence_id"],
        )

    def test_langgraph_adapter_skips_safely_without_entity_seed(self):
        update = asyncio.run(
            graph_traversal_node(
                {
                    "allowed_tools": ["graph_traversal"],
                    "graph_seed_bindings": [],
                    "graph_traversal_request": {},
                },
                client=_structured_client(),
            )
        )
        self.assertEqual("empty", update["tool_results"][0]["status"])
        self.assertEqual([], update["evidence_registry"])
        self.assertEqual([], update["graph_paths"])

    def test_candidate_seed_downgrades_path_but_not_formal_edge_status(self):
        update = asyncio.run(
            graph_traversal_node(
                {
                    "allowed_tools": ["graph_traversal"],
                    "graph_seed_bindings": [
                        {
                            "logical_id": "sg:b:component:a",
                            "binding_status": "candidate",
                            "match_type": "semantic_candidate",
                        }
                    ],
                    "graph_traversal_request": {
                        "mode": "neighbors",
                        "seed_ids": ["sg:b:component:a"],
                        "direction": "out",
                        "max_depth": 1,
                    },
                },
                client=_structured_client(),
            )
        )
        evidence = update["evidence_registry"][0]
        self.assertEqual("L3_INFERRED", evidence["evidence_level"])
        self.assertEqual("validated_relation", evidence["authority"])
        self.assertEqual("candidate", evidence["metadata"]["seed_binding_status"])
        self.assertEqual("candidate", update["graph_paths"][0]["seed_binding_status"])


if __name__ == "__main__":
    unittest.main()
