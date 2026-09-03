import unittest
from unittest.mock import patch

from fastapi import HTTPException

# Importing the API aggregate first follows the application's real import order
# and avoids the legacy kg_service <-> api package cycle during isolated tests.
import backend.api  # noqa: F401
from backend.api.routers import types as types_router
from backend.services import kg_service


def _revision(*, revision_id="revision-2", status="not_projected", active=False):
    return {
        "revision_id": revision_id,
        "revision_no": 2,
        "diagram_id": "diagram-1",
        "doc_id": "doc-1",
        "diagram_type": "fta",
        "canonical_hash": "a" * 64,
        "review_status": "confirmed",
        "projection_status": status,
        "is_active": active,
        "created_at": "2026-08-24T00:00:00",
        "canonical_graph": {
            "diagram_type": "fta",
            "root_fault_id": "top",
            "vertices": [
                {
                    "id": "top",
                    "label": "TopEvent",
                    "properties": {
                        "name": "系统失效",
                        "event_id": "TOP-1",
                        "source_refs": [{"artifact_ref_id": "artifact-1"}],
                    },
                },
                {
                    "id": "event-a",
                    "label": "BasicEvent",
                    "properties": {
                        "name": "电源失效",
                        "probability": 0.1,
                        "business_id": "EVENT-A",
                    },
                },
            ],
            "edges": [
                {
                    "id": "edge-a",
                    "label": "causes",
                    "source": "top",
                    "target": "event-a",
                    "properties": {"gate_type": {"value": "OR"}},
                }
            ],
        },
    }


class _SelectionRepository:
    def list_diagrams_by_document(self, doc_id):
        return [
            {"diagram_id": "diagram-1", "diagram_type": "fta"},
            {"diagram_id": "diagram-2", "diagram_type": "fta"},
        ]

    def list_revisions(self, diagram_id):
        if diagram_id == "diagram-1":
            return [
                {
                    "revision_id": "new-pending",
                    "review_status": "pending",
                    "projection_status": "not_projected",
                    "revision_no": 2,
                },
                {
                    "revision_id": "old-confirmed",
                    "review_status": "confirmed",
                    "projection_status": "not_projected",
                    "revision_no": 1,
                },
            ]
        return [
            {
                "revision_id": "latest-confirmed",
                "review_status": "confirmed",
                "projection_status": "failed",
                "revision_no": 1,
            }
        ]


class _TaskManager:
    def __init__(self):
        self.statuses = []
        self.progress = []
        self.result = None
        self.error = None

    async def update_status(self, task_id, status):
        self.statuses.append(status)

    async def update_progress(self, task_id, progress, message, **kwargs):
        self.progress.append((progress, message))

    async def set_result(self, task_id, result):
        self.result = result

    async def set_error(self, task_id, error):
        self.error = error


class _ProjectionRepository:
    def __init__(self, revision, *, already_active=False):
        self.revision = dict(revision)
        self.already_active = already_active
        self.activations = []
        self.failures = []
        self.invalidations = []

    def get_revision(self, revision_id):
        return dict(self.revision) if revision_id == self.revision["revision_id"] else None

    def start_projection(self, revision_id, **kwargs):
        return {
            "run_id": "run-1",
            "revision_id": revision_id,
            "already_active": self.already_active,
            "reused": self.already_active,
        }

    def activate_projection(self, revision_id, **kwargs):
        self.activations.append((revision_id, kwargs))
        self.revision["projection_status"] = "published"
        self.revision["is_active"] = True
        return {"run_id": kwargs["run_id"], "status": "succeeded"}

    def fail_projection(self, revision_id, **kwargs):
        self.failures.append((revision_id, kwargs))
        self.revision["projection_status"] = "failed"
        return {"run_id": kwargs["run_id"], "status": "failed"}

    def invalidate_active_projection(self, revision_id, **kwargs):
        self.invalidations.append((revision_id, kwargs))
        self.revision["projection_status"] = "failed"
        self.revision["is_active"] = False
        return {"run_id": kwargs["run_id"], "status": "failed"}

    def list_diagrams_by_document(self, doc_id):
        return [{"diagram_id": self.revision["diagram_id"], "diagram_type": "fta"}]

    def list_revisions(self, diagram_id):
        return [dict(self.revision)]


class _HugeGraph:
    _use_mock = False

    def __init__(self, *, drop_readback_vertex=False, fail_initial_read=False):
        self.vertices = {}
        self.edges = {}
        self.drop_readback_vertex = drop_readback_vertex
        self.fail_initial_read = fail_initial_read
        self.vertex_read_count = 0
        self.events = []

    def batch_insert(self, vertices, edges):
        self.events.append(("insert", len(vertices), len(edges)))
        self.vertices.update({item["id"]: dict(item) for item in vertices})
        self.edges.update({item["id"]: dict(item) for item in edges})
        return True

    def get_vertices_by_kb_with_details(self, kb_id):
        self.vertex_read_count += 1
        if self.fail_initial_read and self.vertex_read_count == 1:
            raise RuntimeError("HugeGraph initial read unavailable")
        values = list(self.vertices.values())
        return values[1:] if self.drop_readback_vertex and values else values

    def get_edges_by_kb_with_details(self, kb_id):
        return list(self.edges.values())

    def delete_edge(self, edge_id):
        self.events.append(("delete_edge", edge_id))
        return self.edges.pop(edge_id, None) is not None

    def delete_vertex(self, vertex_id):
        self.events.append(("delete_vertex", vertex_id))
        return self.vertices.pop(vertex_id, None) is not None


class _Postgres:
    def __init__(self):
        self.statuses = []

    def update_icp_document_status(self, doc_id, status):
        self.statuses.append((doc_id, status))


class DiagramProjectionPureTests(unittest.TestCase):
    def test_candidate_evidence_urls_ignore_storage_keys_and_unsafe_values(self):
        urls = types_router._candidate_evidence_image_urls(
            [
                {
                    "source_locator": {
                        "image_url": "/images/doc-1/evidence-1.png",
                        "storage_key": "source/icp/doc-1/original.docx",
                    }
                },
                {
                    "source_locator": '{"image":{"url":"https://evidence.local/diagram.png"},'
                    '"storage_key":"source/secret.png"}'
                },
                {"source_locator": {"image_url": "source/icp/private.png"}},
                {"source_locator": {"image_url": "data:image/png;base64,secret"}},
                {"source_locator": {"image_url": "/images/../source/private.png"}},
                {"source_locator": {"image_url": "/images/%2e%2e/source/private.png"}},
            ]
        )

        self.assertEqual(
            {
                "/images/doc-1/evidence-1.png",
                "https://evidence.local/diagram.png",
            },
            urls,
        )
        self.assertFalse(any("storage" in value or "secret" in value for value in urls))

    def test_current_revision_candidate_urls_are_collected_per_active_diagram(self):
        class Repository:
            def list_diagrams_by_document(self, doc_id):
                return [{"diagram_id": "diagram-1"}]

            def list_revisions(self, diagram_id):
                return [
                    {"revision_id": "latest"},
                    {"revision_id": "older"},
                ]

            def get_revision_candidates(self, revision_id):
                self.requested_revision_id = revision_id
                return [
                    {
                        "source_locator": {
                            "image_url": "/images/doc-1/current.png",
                            "storage_key": "source/private/current.png",
                        }
                    }
                ]

        repository = Repository()
        with patch.object(types_router, "diagram_repository", repository):
            urls = types_router._revision_candidate_evidence_urls("doc-1")

        self.assertEqual({"/images/doc-1/current.png"}, urls)
        self.assertEqual("latest", repository.requested_revision_id)

    def test_conversion_uses_revision_namespace_and_existing_hugegraph_schema(self):
        vertices, edges = kg_service._revision_to_hugegraph(_revision())

        self.assertEqual(2, len(vertices))
        self.assertEqual(1, len(edges))
        self.assertTrue(all(item["id"].startswith("icp:revision-2:") for item in vertices + edges))
        vertex_props = vertices[0]["properties"]
        self.assertEqual("revision-2", vertex_props["tenant_id"])
        self.assertEqual("diagram-1", vertex_props["project_id"])
        self.assertEqual(["doc-1"], vertex_props["doc_id"])
        self.assertIn("revision:revision-2", vertex_props["source_id"])
        self.assertNotIn("revision_id", vertex_props)
        self.assertNotIn("canonical_hash", vertex_props)
        self.assertNotIn("event_id", vertex_props)
        self.assertEqual("OR", edges[0]["properties"]["gate_type"])

    def test_selection_never_falls_back_from_latest_pending_to_old_confirmed(self):
        repository = _SelectionRepository()

        selected = kg_service.select_latest_confirmed_revisions(
            repository,
            doc_id="doc-1",
            doc_type="fta",
        )

        self.assertEqual(["latest-confirmed"], [item["revision_id"] for item in selected])
        with self.assertRaisesRegex(ValueError, "不是当前最新版本"):
            kg_service.select_latest_confirmed_revisions(
                repository,
                doc_id="doc-1",
                doc_type="fta",
                requested_ids=["old-confirmed"],
            )
        with self.assertRaisesRegex(ValueError, "未确认"):
            kg_service.select_latest_confirmed_revisions(
                repository,
                doc_id="doc-1",
                doc_type="fta",
                requested_ids=["new-pending"],
            )

    def test_projection_marker_does_not_accept_unmarked_legacy_items(self):
        self.assertEqual(
            "revision-2",
            kg_service.projection_item_revision_id(
                {"properties": {"tenant_id": "revision-2"}}
            ),
        )
        self.assertEqual(
            "revision-3",
            kg_service.projection_item_revision_id(
                {"properties": {"source_id": ["doc-1", "revision:revision-3"]}}
            ),
        )
        self.assertEqual("", kg_service.projection_item_revision_id({"properties": {"doc_id": ["doc-1"]}}))


class DiagramProjectionBackgroundTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_pipeline_refreshes_common_graph_after_activation(self):
        revision = _revision()
        repository = _ProjectionRepository(revision)
        graph = _HugeGraph()
        tasks = _TaskManager()
        postgres = _Postgres()
        refreshed = {
            "doc_id": "doc-1",
            "doc_type": "fta",
            "status": "published",
            "vertices": 8,
            "edges": 9,
        }

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ), patch(
            "backend.icp.structured_projection.project_structured_document",
            return_value=refreshed,
        ) as refresh:
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
                structured_result={"vertices": 1, "edges": 0},
            )

        refresh.assert_called_once_with("doc-1", "fta")
        self.assertEqual(refreshed, tasks.result["structured_projection"])
        self.assertTrue(repository.revision["is_active"])

    async def test_success_requires_write_readback_before_activation(self):
        revision = _revision()
        repository = _ProjectionRepository(revision)
        graph = _HugeGraph()
        tasks = _TaskManager()
        postgres = _Postgres()

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ):
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
            )

        self.assertEqual(["revision-2"], tasks.result["revision_ids"])
        self.assertEqual("kg_done", tasks.result["status"])
        self.assertEqual("completed", tasks.statuses[-1])
        self.assertEqual(1, len(repository.activations))
        self.assertEqual([], repository.failures)
        self.assertEqual(("doc-1", "kg_done"), postgres.statuses[-1])

    async def test_readback_count_mismatch_fails_projection_and_keeps_pending(self):
        revision = _revision()
        repository = _ProjectionRepository(revision)
        graph = _HugeGraph(drop_readback_vertex=True)
        tasks = _TaskManager()
        postgres = _Postgres()

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ):
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
            )

        self.assertIsNone(tasks.result)
        self.assertIn("写后读回数量不一致", tasks.error)
        self.assertEqual("failed", tasks.statuses[-1])
        self.assertEqual(1, len(repository.failures))
        self.assertEqual([], repository.activations)
        self.assertEqual(("doc-1", "kg_pending"), postgres.statuses[-1])

    async def test_initial_hugegraph_read_failure_closes_started_projection(self):
        revision = _revision()
        repository = _ProjectionRepository(revision)
        graph = _HugeGraph(fail_initial_read=True)
        tasks = _TaskManager()
        postgres = _Postgres()

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ):
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
            )

        self.assertIsNone(tasks.result)
        self.assertIn("initial read unavailable", tasks.error)
        self.assertEqual("failed", tasks.statuses[-1])
        self.assertEqual(1, len(repository.failures))
        self.assertEqual("revision-2", repository.failures[0][0])
        self.assertEqual([], repository.activations)
        self.assertEqual(("doc-1", "kg_pending"), postgres.statuses[-1])

    async def test_active_retry_is_readback_only_and_does_not_insert_again(self):
        revision = _revision(status="published", active=True)
        repository = _ProjectionRepository(revision, already_active=True)
        graph = _HugeGraph()
        vertices, edges = kg_service._revision_to_hugegraph(revision)
        graph.vertices.update({item["id"]: item for item in vertices})
        graph.edges.update({item["id"]: item for item in edges})
        tasks = _TaskManager()
        postgres = _Postgres()

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ):
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
            )

        self.assertFalse(any(event[0] == "insert" for event in graph.events))
        self.assertEqual([], repository.activations)
        self.assertEqual("completed", tasks.statuses[-1])

    async def test_active_retry_readback_drift_closes_active_gate(self):
        revision = _revision(status="published", active=True)
        repository = _ProjectionRepository(revision, already_active=True)
        graph = _HugeGraph(drop_readback_vertex=True)
        vertices, edges = kg_service._revision_to_hugegraph(revision)
        graph.vertices.update({item["id"]: item for item in vertices})
        graph.edges.update({item["id"]: item for item in edges})
        tasks = _TaskManager()
        postgres = _Postgres()

        with patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "hugegraph_client", graph
        ), patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ):
            await kg_service.generate_icp_kg_background(
                "task-1",
                "doc-1",
                "fta",
                "fta.docx",
                revision_ids=["revision-2"],
            )

        self.assertIsNone(tasks.result)
        self.assertIn("已发布图修订", tasks.error)
        self.assertEqual("failed", tasks.statuses[-1])
        self.assertEqual(1, len(repository.invalidations))
        self.assertEqual([], repository.failures)
        self.assertFalse(repository.revision["is_active"])
        self.assertEqual(("doc-1", "kg_pending"), postgres.statuses[-1])


class DiagramGraphDataGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_data_excludes_stale_and_unmarked_items(self):
        class Repository:
            def list_active_revisions(self, **kwargs):
                return [
                    {
                        "revision_id": "active-revision",
                        "review_status": "confirmed",
                        "projection_status": "published",
                        "is_active": True,
                    }
                ]

        class Graph:
            def get_vertices_by_kb_with_details(self, kb_id):
                return [
                    {"id": "active-v", "properties": {"doc_id": ["doc-1"], "tenant_id": "active-revision"}},
                    {"id": "stale-v", "properties": {"doc_id": ["doc-1"], "tenant_id": "stale-revision"}},
                    {"id": "legacy-v", "properties": {"doc_id": ["doc-1"]}},
                ]

            def get_edges_by_kb_with_details(self, kb_id):
                return [
                    {"id": "active-e", "properties": {"doc_id": ["doc-1"], "tenant_id": "active-revision"}},
                    {"id": "stale-e", "properties": {"doc_id": ["doc-1"], "tenant_id": "stale-revision"}},
                ]

        with patch.object(types_router, "diagram_repository", Repository()), patch.object(
            types_router, "hugegraph_client", Graph()
        ), patch.object(types_router, "_get_icp_document", return_value={"doc_id": "doc-1"}):
            result = await types_router.get_icp_graph_data("fta", "doc-1")

        self.assertEqual(["active-revision"], result["active_revision_ids"])
        self.assertEqual(["active-v"], [item["id"] for item in result["vertices"]])
        self.assertEqual(["active-e"], [item["id"] for item in result["edges"]])

    async def test_graph_data_failure_returns_service_unavailable(self):
        class Repository:
            def list_active_revisions(self, **kwargs):
                return [{"revision_id": "active-revision"}]

        class Graph:
            def get_vertices_by_kb_with_details(self, kb_id):
                raise RuntimeError("HugeGraph unavailable")

            def get_edges_by_kb_with_details(self, kb_id):
                return []

        with patch.object(types_router, "diagram_repository", Repository()), patch.object(
            types_router, "hugegraph_client", Graph()
        ), patch.object(
            types_router, "_get_icp_document", return_value={"doc_id": "doc-1"}
        ), patch.object(types_router.logger, "exception"):
            with self.assertRaises(HTTPException) as raised:
                await types_router.get_icp_graph_data("fta", "doc-1")

        self.assertEqual(503, raised.exception.status_code)
        self.assertIn("图谱服务读取失败", raised.exception.detail)

    async def test_graph_data_document_backend_failure_returns_service_unavailable(self):
        with patch.object(
            types_router,
            "_get_icp_document",
            side_effect=RuntimeError("PostgreSQL unavailable"),
        ), patch.object(types_router.logger, "exception"):
            with self.assertRaises(HTTPException) as raised:
                await types_router.get_icp_graph_data("fta", "doc-1")

        self.assertEqual(503, raised.exception.status_code)


class DiagramReviewApiSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_locator_keeps_optional_review_mutations_disabled(self):
        calls = [
            lambda: types_router.update_diagram_revision(
                "function_locator",
                "doc-1",
                "revision-1",
                {"canonical_graph": {}, "lock_version": 1},
            ),
            lambda: types_router.confirm_diagram_revision(
                "function_locator",
                "doc-1",
                "revision-1",
                {"reviewed_by": "reviewer", "lock_version": 1},
            ),
            lambda: types_router.reject_diagram_revision(
                "function_locator",
                "doc-1",
                "revision-1",
                {"reviewed_by": "reviewer", "review_note": "invalid", "lock_version": 1},
            ),
        ]

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(HTTPException) as raised:
                    await call()
                self.assertEqual(400, raised.exception.status_code)
                self.assertIn("FTA/RBD", raised.exception.detail)

    async def test_all_seven_types_allow_graph_projection_and_architecture_listing(self):
        class Repository:
            def list_diagrams_by_document(self, doc_id):
                return []

            def list_active_revisions(self, **kwargs):
                return []

        class Graph:
            def get_vertices_by_kb_with_details(self, kb_id):
                return []

            def get_edges_by_kb_with_details(self, kb_id):
                return []

        with patch.object(types_router, "diagram_repository", Repository()), patch.object(
            types_router, "hugegraph_client", Graph()
        ), patch.object(
            types_router,
            "_get_icp_document",
            return_value={"doc_id": "doc-1", "doc_type": "function_locator"},
        ):
            self.assertEqual([], await types_router.get_fault_trees("function_locator", "doc-1"))
            self.assertEqual(
                [], await types_router.list_reviewable_diagrams("function_locator", "doc-1")
            )
            result = await types_router.get_icp_graph_data("function_locator", "doc-1")
            self.assertEqual(0, result["total_vertices"])
            self.assertEqual(0, result["total_edges"])

        for doc_type in (
            "function_locator",
            "hmc",
            "fta",
            "rbd",
            "fmeca",
            "mfl",
            "fault_history",
        ):
            with self.subTest(doc_type=doc_type):
                types_router._require_graph_type(doc_type)

    async def test_confirm_and_reject_require_lock_version(self):
        class Repository:
            def __init__(self):
                self.review_called = False

            def get_revision(self, revision_id):
                return {
                    "revision_id": revision_id,
                    "doc_id": "doc-1",
                    "diagram_type": "fta",
                }

            def confirm_revision(self, *args, **kwargs):
                self.review_called = True

            def reject_revision(self, *args, **kwargs):
                self.review_called = True

        repository = Repository()
        with patch.object(types_router, "diagram_repository", repository), patch.object(
            types_router,
            "_get_icp_document",
            return_value={"doc_id": "doc-1", "doc_type": "fta"},
        ):
            for call in (
                lambda: types_router.confirm_diagram_revision(
                    "fta", "doc-1", "revision-1", {"reviewed_by": "reviewer"}
                ),
                lambda: types_router.reject_diagram_revision(
                    "fta",
                    "doc-1",
                    "revision-1",
                    {"reviewed_by": "reviewer", "review_note": "invalid"},
                ),
            ):
                with self.subTest(call=call):
                    with self.assertRaises(HTTPException) as raised:
                        await call()
                    self.assertEqual(422, raised.exception.status_code)
                    self.assertIn("lock_version", raised.exception.detail)

        self.assertFalse(repository.review_called)

    async def test_explicit_empty_revision_list_never_means_bulk_publish(self):
        with patch.object(
            types_router,
            "_get_icp_document",
            return_value={
                "doc_id": "doc-1",
                "doc_type": "fta",
                "file_name": "fta.docx",
                "status": "kg_pending",
            },
        ), patch.object(types_router, "select_latest_confirmed_revisions") as selector:
            with self.assertRaises(HTTPException) as raised:
                await types_router.generate_kg(
                    "fta",
                    "doc-1",
                    {"revision_ids": []},
                )

        self.assertEqual(422, raised.exception.status_code)
        selector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
