import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts import smoke_icp_ingestion as smoke


def _stats(doc_type):
    common = {"vectors": 2, "projection_failed": []}
    if doc_type == "function_locator":
        return {**common, "components_written": 2, "functions_written": 3}
    if doc_type in {"hmc", "fmeca", "mfl"}:
        return {**common, "rows_ok": 3, "rows_failed": 0}
    if doc_type == "fault_history":
        return {**common, "cases": 2, "steps": 4}
    return {
        **common,
        "images": 1,
        "graph_count": 1,
        "present_diagrams": 1,
        "review_pending": True,
        "projection_status_counts": {"not_projected": 1},
    }


def _diagram_revisions(doc_type, doc_id):
    statuses = {
        "function_locator": "auto_validated",
        "fta": "auto_validated",
        "rbd": "provisional",
    }
    validation_status = statuses.get(doc_type)
    if not validation_status:
        return []
    return [
        {
            "diagram_id": f"diagram-{doc_id}",
            "revision_id": f"revision-{doc_id}",
            "validation_status": validation_status,
            "projection_status": "published",
            "is_active": True,
        }
    ]


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.documents = {doc_type: [] for doc_type in smoke.EXPECTED_TYPES}
        self.tasks = {}
        self.next_id = 1

    def add_existing(self, fixture, *, status="kg_done", sha256=None):
        doc_id = f"existing-{fixture.doc_type}-{len(self.documents[fixture.doc_type]) + 1}"
        document = {
            "doc_id": doc_id,
            "doc_type": fixture.doc_type,
            "file_name": fixture.file_name,
            "status": status,
            "parse_count": 1 if status in smoke.READY_DOCUMENT_STATUSES else 0,
            "last_result": _stats(fixture.doc_type) if status in smoke.READY_DOCUMENT_STATUSES else None,
            "metadata": {"sha256": sha256} if sha256 else {},
        }
        self.documents[fixture.doc_type].append(document)
        return document

    def get(self, path):
        self.calls.append(("GET", path))
        if path == "/readyz":
            return {
                "status": "ready",
                "checks": {
                    "postgresql": {"ok": True},
                    "opensearch": {"ok": True},
                    "artifact_store": {"ok": True},
                    "hugegraph": {"ok": True},
                },
            }
        if path == "/api/v1/document-types":
            return [{"doc_type": doc_type} for doc_type in smoke.EXPECTED_TYPES]
        if path == "/openapi.json":
            return {
                "paths": {
                    "/api/v1/dify/server-info": {"get": {}},
                    "/api/v1/dify/retrieval": {"post": {}},
                }
            }
        if path == "/api/v1/dify/server-info":
            return {"lan_ip": "127.0.0.1", "source": "test"}
        prefix = "/api/v1/document-types/"
        if path.startswith(prefix) and path.endswith("/documents"):
            doc_type = path[len(prefix) : -len("/documents")]
            return [dict(item) for item in self.documents[doc_type]]
        if path.startswith(prefix) and path.endswith("/parse-result"):
            parts = path.split("/")
            doc_type = parts[4]
            doc_id = parts[6]
            document = self._document(doc_type, doc_id)
            return {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "status": document["status"],
                "parse_count": document["parse_count"],
                "last_result": document["last_result"],
            }
        if path.startswith(prefix) and path.endswith("/diagrams"):
            parts = path.split("/")
            doc_type = parts[4]
            doc_id = parts[6]
            self._document(doc_type, doc_id)
            return _diagram_revisions(doc_type, doc_id)
        task_prefix = "/api/v1/knowledge-bases/tasks/"
        if path.startswith(task_prefix) and path.endswith("/status"):
            task_id = path[len(task_prefix) : -len("/status")]
            task = self.tasks[task_id]
            status = task["states"].pop(0) if len(task["states"]) > 1 else task["states"][0]
            if status == "completed":
                document = self._document(task["doc_type"], task["doc_id"])
                if task["task_type"] == "parse":
                    graph_task_id = f"graph-{task['doc_id']}"
                    self.tasks.setdefault(
                        graph_task_id,
                        {
                            "task_type": "generate_kg",
                            "doc_type": task["doc_type"],
                            "doc_id": task["doc_id"],
                            "states": ["pending", "generating_kg", "completed"],
                        },
                    )
                    document.update(
                        {
                            "status": "generating_kg",
                            "parse_count": 1,
                            "last_result": _stats(task["doc_type"]),
                        }
                    )
                    result = {
                        "doc_id": task["doc_id"],
                        "doc_type": task["doc_type"],
                        "status": "generating_kg",
                        "graph_task_id": graph_task_id,
                    }
                else:
                    document["status"] = "kg_done"
                    result = {
                        "doc_id": task["doc_id"],
                        "doc_type": task["doc_type"],
                        "status": "kg_done",
                        "structured_projection": {
                            "status": "published",
                            "vertices": 2,
                            "edges": 1,
                        },
                        "revision_ids": [
                            item["revision_id"]
                            for item in _diagram_revisions(task["doc_type"], task["doc_id"])
                        ],
                    }
            else:
                result = None
            return {
                "task_id": task_id,
                "task_type": task["task_type"],
                "status": status,
                "progress": 100 if status == "completed" else 50,
                "result": result,
                "error": "",
            }
        raise AssertionError(f"unexpected GET {path}")

    def upload_file(self, path, file_path):
        self.calls.append(("UPLOAD", path))
        doc_type = path.split("/")[4]
        doc_id = f"created-{self.next_id}"
        self.next_id += 1
        document = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "file_name": file_path.name,
            "status": "uploaded",
            "parse_count": 0,
            "last_result": None,
            "metadata": {},
        }
        self.documents[doc_type].insert(0, document)
        return [{"doc_id": doc_id, "file_name": file_path.name, "status": "uploaded"}]

    def post_json(self, path, payload):
        self.calls.append(("POST", path))
        if not path.endswith("/parse"):
            raise AssertionError(f"unexpected POST {path}")
        parts = path.split("/")
        doc_type = parts[4]
        doc_id = parts[6]
        task_id = f"task-{doc_id}"
        self.tasks[task_id] = {
            "task_type": "parse",
            "doc_type": doc_type,
            "doc_id": doc_id,
            "states": ["pending", "parsing", "completed"],
        }
        return {"task_id": task_id, "doc_id": doc_id, "status": "parsing"}

    def delete(self, path):
        self.calls.append(("DELETE", path))
        parts = path.split("/")
        doc_type = parts[4]
        doc_id = parts[6]
        document = self._document(doc_type, doc_id)
        self.documents[doc_type].remove(document)
        return {"status": "success", "doc_id": doc_id, "cleanup_errors": []}

    def _document(self, doc_type, doc_id):
        return next(item for item in self.documents[doc_type] if item["doc_id"] == doc_id)


class SyntheticIngestionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        for index, (doc_type, file_name) in enumerate(smoke.EXPECTED_DOCUMENTS, start=1):
            (root / file_name).write_bytes(f"synthetic-{index}-{doc_type}".encode("utf-8"))
        self.root = root
        self.fixtures = smoke.load_fixtures(root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _run(self, client, *, cleanup=False):
        runner = smoke.SmokeRunner(
            client,
            self.fixtures,
            poll_interval=0.001,
            task_timeout=2,
            cleanup_created=cleanup,
            sleep=lambda _: None,
        )
        return runner.run(base_url="http://test.invalid", test_docs=str(self.root))

    def test_uploads_parses_and_waits_for_all_seven_automatic_graph_tasks(self):
        client = _FakeClient()

        summary = self._run(client)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(len(summary["documents"]), 7)
        self.assertEqual(sum(method == "UPLOAD" for method, _ in client.calls), 7)
        self.assertEqual(sum(method == "POST" for method, _ in client.calls), 7)
        task_gets = [path for method, path in client.calls if method == "GET" and "/tasks/" in path]
        self.assertEqual(len(task_gets), 42)
        self.assertTrue(all(document["task_status"] == "completed" for document in summary["documents"]))
        self.assertTrue(
            all(document["graph_task_status"] == "completed" for document in summary["documents"])
        )
        self.assertTrue(all(document["document_status"] == "kg_done" for document in summary["documents"]))
        self.assertFalse(
            any(
                token in path
                for _, path in client.calls
                for token in ("/confirm", "/reject", "/generate_kg")
            )
        )
        self.assertEqual(summary["safety"]["manual_graph_review_actions"], 0)
        self.assertEqual(summary["safety"]["explicit_graph_generation_actions"], 0)
        self.assertEqual(summary["safety"]["automatic_graph_tasks_observed"], 7)

        fta = next(item for item in summary["documents"] if item["doc_type"] == "fta")
        rbd = next(item for item in summary["documents"] if item["doc_type"] == "rbd")
        self.assertEqual(fta["diagram_validation_status_counts"], {"auto_validated": 1})
        self.assertEqual(rbd["diagram_validation_status_counts"], {"provisional": 1})
        self.assertEqual(fta["diagram_projection_status_counts"], {"published": 1})
        self.assertEqual(rbd["diagram_projection_status_counts"], {"published": 1})

    def test_unique_ready_filename_matches_are_reused_without_upload_or_reparse(self):
        client = _FakeClient()
        for fixture in self.fixtures:
            client.add_existing(fixture)

        summary = self._run(client)

        self.assertTrue(summary["ok"], summary)
        self.assertFalse(any(method == "UPLOAD" for method, _ in client.calls))
        self.assertFalse(any(method == "POST" for method, _ in client.calls))
        self.assertTrue(all(item["origin"] == "reused" for item in summary["documents"]))
        self.assertTrue(
            all(item["duplicate_basis"] == "file_name_server_sha_unavailable" for item in summary["documents"])
        )
        self.assertEqual(summary["safety"]["automatic_graph_tasks_observed"], 0)

    def test_quarantined_revision_is_terminal_only_while_isolated(self):
        isolated = {
            "revision_id": "revision-quarantined",
            "validation_status": "quarantined",
            "projection_status": "not_projected",
            "is_active": False,
        }

        issues, validation_counts, projection_counts = smoke.SmokeRunner._validate_diagram_revisions(
            "fta", [isolated], newly_created=True
        )

        self.assertEqual(issues, [])
        self.assertEqual(validation_counts, {"quarantined": 1})
        self.assertEqual(projection_counts, {"not_projected": 1})

        leaked = {**isolated, "projection_status": "published", "is_active": True}
        issues, _, _ = smoke.SmokeRunner._validate_diagram_revisions(
            "fta", [leaked], newly_created=True
        )
        self.assertIn("leaked into the active projection", issues[0])

    def test_pending_diagram_revision_fails_the_automatic_terminal_policy(self):
        issues, _, _ = smoke.SmokeRunner._validate_diagram_revisions(
            "rbd",
            [
                {
                    "revision_id": "revision-pending",
                    "validation_status": "pending",
                    "projection_status": "not_projected",
                    "is_active": False,
                }
            ],
            newly_created=True,
        )

        self.assertIn("non-terminal validation status pending", issues[0])

    def test_filename_only_unparsed_match_is_reported_and_not_modified(self):
        client = _FakeClient()
        hmc_fixture = next(item for item in self.fixtures if item.doc_type == "hmc")
        existing = client.add_existing(hmc_fixture, status="uploaded")

        summary = self._run(client)

        self.assertFalse(summary["ok"])
        hmc = next(item for item in summary["documents"] if item["doc_type"] == "hmc")
        self.assertIn("not modified", hmc["error"])
        self.assertEqual(existing["status"], "uploaded")
        hmc_paths = [path for method, path in client.calls if method in {"UPLOAD", "POST"} and "/hmc/" in path]
        self.assertEqual(hmc_paths, [])

    def test_ambiguous_filename_matches_are_reported_without_an_extra_upload(self):
        client = _FakeClient()
        fixture = self.fixtures[0]
        client.add_existing(fixture)
        client.add_existing(fixture)

        summary = self._run(client)

        self.assertFalse(summary["ok"])
        item = next(row for row in summary["documents"] if row["doc_type"] == fixture.doc_type)
        self.assertIn("reuse is ambiguous", item["error"])
        self.assertFalse(
            any(method == "UPLOAD" and f"/{fixture.doc_type}/" in path for method, path in client.calls)
        )

    def test_cleanup_deletes_only_ids_created_by_this_run(self):
        client = _FakeClient()
        existing_fixture = self.fixtures[0]
        existing = client.add_existing(existing_fixture)

        summary = self._run(client, cleanup=True)

        self.assertTrue(summary["ok"], summary)
        deleted_ids = {path.split("/")[6] for method, path in client.calls if method == "DELETE"}
        self.assertEqual(deleted_ids, set(summary["safety"]["created_doc_ids"]))
        self.assertNotIn(existing["doc_id"], deleted_ids)
        self.assertEqual(len(deleted_ids), 6)
        self.assertIs(client._document(existing_fixture.doc_type, existing["doc_id"]), existing)

    def test_server_sha_mismatch_blocks_same_filename_reuse(self):
        client = _FakeClient()
        fixture = self.fixtures[0]
        client.add_existing(fixture, sha256="f" * 64)

        summary = self._run(client)

        self.assertFalse(summary["ok"])
        item = next(row for row in summary["documents"] if row["doc_type"] == fixture.doc_type)
        self.assertIn("different SHA-256", item["error"])
        self.assertFalse(
            any(method == "UPLOAD" and f"/{fixture.doc_type}/" in path for method, path in client.calls)
        )

    def test_multipart_upload_preserves_utf8_fixture_filename(self):
        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b"[]"

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response()

        fixture = self.fixtures[0]
        client = smoke.JsonApiClient("http://127.0.0.1:19999", timeout=3)
        with patch.object(client._opener, "open", fake_urlopen):
            client.upload_file("/upload", fixture.path)

        request = captured["request"]
        self.assertIn(f'filename="{fixture.file_name}"'.encode("utf-8"), request.data)
        self.assertNotIn(b"filename*=", request.data)
        self.assertIn("charset=utf-8", request.headers["Content-type"])
        self.assertEqual(captured["timeout"], 3)

    def test_client_bypasses_environment_proxy_for_explicit_service_url(self):
        client = smoke.JsonApiClient("http://127.0.0.1:19999")
        # ``build_opener(ProxyHandler({}))`` deliberately omits the default
        # environment-backed proxy handler; an empty handler has no protocol
        # methods and therefore is not retained in the final handler list.
        self.assertFalse(
            any(
                isinstance(handler, smoke.ProxyHandler)
                for handler in client._opener.handlers
            )
        )


if __name__ == "__main__":
    unittest.main()
