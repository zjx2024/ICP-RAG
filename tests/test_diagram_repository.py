import hashlib
import unittest
from unittest.mock import Mock

from backend.icp.diagram_models import content_hash
from backend.icp.diagram_repository import (
    DiagramConcurrencyError,
    DiagramRepository,
    DiagramStateError,
)


def _valid_graph():
    return {
        "diagram_type": "fta",
        "root_fault_id": "top",
        "vertices": [
            {"id": "top", "label": "TopEvent", "properties": {"name": "系统失效", "gate_type": "OR"}},
            {"id": "a", "label": "BasicEvent", "properties": {"event_id": "E-01", "name": "电源失效"}},
        ],
        "edges": [{"id": "e1", "label": "causes", "source": "top", "target": "a", "properties": {"gate_type": "OR"}}],
    }


def _reviewable_revision(**overrides):
    revision = {
        "revision_id": "revision-1",
        "diagram_id": "diagram-1",
        "diagram_type": "fta",
        "document_status": "kg_pending",
        "lifecycle_status": "active",
        "revision_no": 1,
        "latest_revision_no": 1,
        "review_status": "pending",
        "projection_status": "not_projected",
        "conflict_count": 0,
        "lock_version": 1,
        "canonical_graph": _valid_graph(),
    }
    revision.update(overrides)
    return revision


class _ScriptedCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._rows = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.connection.executions.append((" ".join(str(sql).split()), params))
        response = self.connection.responses.pop(0) if self.connection.responses else {}
        self._rows = list(response.get("rows", []))
        self.description = response.get("description")
        self.rowcount = response.get("rowcount", len(self._rows))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _ScriptedConnection:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.executions = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _ScriptedCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class DiagramRepositorySchemaTests(unittest.TestCase):
    def test_init_tables_is_idempotent_and_contains_review_projection_guards(self):
        connection = _ScriptedConnection()
        repository = DiagramRepository(connection)

        repository.init_tables()
        first_count = len(connection.executions)
        repository.init_tables()

        all_sql = "\n".join(sql for sql, _ in connection.executions)
        self.assertGreater(first_count, 5)
        self.assertEqual(len(connection.executions), first_count)
        self.assertEqual(connection.commits, 1)
        self.assertIn("icp_diagram_revisions", all_sql)
        self.assertIn("review_status IN ('pending','conflicted','confirmed','rejected')", all_sql)
        self.assertIn("validation_status IN ('pending','auto_validated','provisional','quarantined','failed','manual_confirmed')", all_sql)
        self.assertIn(
            "merge_status IN ('included','duplicate','conflict','isolated','rejected')",
            all_sql,
        )
        self.assertIn("uq_icp_revision_one_published", all_sql)
        self.assertIn("idx_icp_revisions_hash", all_sql)


class DiagramRepositoryOperationTests(unittest.TestCase):
    def test_failed_rollback_is_logged_without_masking_original_error(self):
        connection = Mock()
        connection.rollback.side_effect = RuntimeError("connection closed")
        repository = DiagramRepository(connection)
        with self.assertLogs("backend.icp.diagram_repository", level="WARNING") as logs:
            repository._rollback()
        self.assertIn("connection closed", logs.output[0])

    def _repository(self, responses):
        connection = _ScriptedConnection(responses)
        ids = iter(["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"])
        repository = DiagramRepository(connection, id_factory=lambda: next(ids))
        repository._initialized = True
        return repository, connection

    def test_get_revision_ends_read_transaction(self):
        repository, connection = self._repository(
            [{"rows": [{"revision_id": "revision-1", "canonical_graph": {}}]}]
        )

        result = repository.get_revision("revision-1")

        self.assertEqual("revision-1", result["revision_id"])
        self.assertEqual(1, connection.commits)
        self.assertEqual(0, connection.rollbacks)

    def test_candidate_upsert_normalizes_before_persistence(self):
        returned = {
            "candidate_id": "candidate-1",
            "diagram_id": "diagram-1",
            "source_kind": "image_vision",
            "candidate_hash": "a" * 64,
        }
        repository, connection = self._repository(
            [
                {"rows": [("fta", "active")]},
                {"rows": [returned]},
            ]
        )
        result = repository.upsert_candidate(
            diagram_id="diagram-1",
            source_kind="image_vision",
            source_fingerprint=hashlib.sha256(b"image").hexdigest(),
            raw_graph=_valid_graph(),
            extractor_name="vision",
            extractor_version="v1",
        )

        self.assertEqual(result["candidate_id"], "candidate-1")
        insert_sql, params = connection.executions[1]
        self.assertIn("ON CONFLICT", insert_sql)
        self.assertIn("normalized_graph", insert_sql)
        self.assertNotIn("confirmed", str(params))
        self.assertEqual(connection.commits, 1)

    def test_deterministic_candidate_is_auto_validated_without_review_state_change(self):
        current = _reviewable_revision(
            canonical_hash="a" * 64,
            conflicts=[],
            validation_report={},
        )
        candidate = {
            "candidate_id": "candidate-1",
            "source_kind": "table_edges",
            "merge_role": "included",
            "extraction_status": "succeeded",
            "normalized_graph": _valid_graph(),
        }
        updated = {
            **current,
            "validation_status": "auto_validated",
            "validation_tier": "deterministic_source",
            "review_status": "pending",
            "lock_version": 2,
        }
        repository, connection = self._repository(
            [{"rows": [current]}, {"rows": [candidate]}, {"rows": [updated]}]
        )

        result = repository.auto_validate_revision("revision-1")

        self.assertEqual("auto_validated", result["validation_status"])
        self.assertEqual("pending", result["review_status"])
        update_sql, update_params = connection.executions[2]
        self.assertIn("validation_status=%s", update_sql)
        self.assertEqual("auto_validated", update_params[0])
        self.assertEqual(connection.commits, 1)

    def test_create_revision_writes_only_pending_or_conflicted(self):
        candidate = {
            "candidate_id": "candidate-1",
            "diagram_id": "diagram-1",
            "source_kind": "image_vision",
            "normalized_graph": _valid_graph(),
            "source_fingerprint": "f" * 64,
        }
        inserted_revision = {
            "revision_id": "revision-1",
            "diagram_id": "diagram-1",
            "revision_no": 1,
            "review_status": "pending",
            "projection_status": "not_projected",
            "lock_version": 1,
        }
        repository, connection = self._repository(
            [
                {"rows": [{"diagram_id": "diagram-1", "diagram_type": "fta"}]},
                {"rows": [candidate]},
                {"rows": []},
                {"rows": [(1,)]},
                {"rows": [inserted_revision]},
                {},
                {},
            ]
        )
        result = repository.create_or_reuse_revision(
            diagram_id="diagram-1", candidate_ids=["candidate-1"]
        )

        self.assertEqual(result["review_status"], "pending")
        insert = next(
            (sql, params) for sql, params in connection.executions
            if sql.startswith("INSERT INTO icp_diagram_revisions")
        )
        self.assertIn(insert[1][6], {"pending", "conflicted"})
        self.assertNotEqual(insert[1][6], "confirmed")
        reuse_sql, reuse_params = connection.executions[2]
        self.assertIn("icp_diagram_revision_candidates", reuse_sql)
        self.assertEqual(reuse_params[2], ["candidate-1"])

    def test_confirmed_revision_is_reused_only_for_identical_candidate_set(self):
        candidate = {
            "candidate_id": "candidate-1",
            "diagram_id": "diagram-1",
            "source_kind": "image_vision",
            "normalized_graph": _valid_graph(),
            "source_fingerprint": "f" * 64,
        }
        confirmed = {
            "revision_id": "revision-1",
            "diagram_id": "diagram-1",
            "revision_no": 1,
            "review_status": "confirmed",
            "projection_status": "published",
            "lock_version": 3,
        }
        repository, connection = self._repository(
            [
                {"rows": [{"diagram_id": "diagram-1", "diagram_type": "fta"}]},
                {"rows": [candidate]},
                {"rows": [confirmed]},
                {},
                {},
            ]
        )

        result = repository.create_or_reuse_revision(
            diagram_id="diagram-1", candidate_ids=["candidate-1"]
        )

        self.assertTrue(result["reused"])
        self.assertEqual(result["review_status"], "confirmed")
        self.assertFalse(
            any(
                sql.startswith("INSERT INTO icp_diagram_revisions")
                for sql, _ in connection.executions
            )
        )

    def test_conflicted_revision_cannot_be_confirmed(self):
        repository, connection = self._repository(
            [
                {
                    "rows": [
                        _reviewable_revision(
                            review_status="conflicted",
                            conflict_count=1,
                        )
                    ]
                }
            ]
        )
        with self.assertRaises(DiagramStateError):
            repository.confirm_revision(
                "revision-1",
                reviewed_by="reviewer",
                expected_lock_version=1,
            )
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)

    def test_explicit_confirmation_records_reviewer(self):
        current = _reviewable_revision(lock_version=2)
        confirmed = {**current, "review_status": "confirmed", "reviewed_by": "reviewer", "lock_version": 3}
        repository, connection = self._repository(
            [{"rows": [current]}, {"rows": [confirmed]}]
        )
        result = repository.confirm_revision(
            "revision-1", reviewed_by="reviewer", expected_lock_version=2
        )
        self.assertEqual(result["review_status"], "confirmed")
        update_sql, update_params = connection.executions[1]
        self.assertIn("review_status='confirmed'", update_sql)
        self.assertEqual(update_params[0], "reviewer")
        lock_sql, _ = connection.executions[0]
        self.assertIn("JOIN icp_documents doc", lock_sql)
        self.assertIn("latest_revision_no", lock_sql)
        self.assertIn("FOR UPDATE OF doc,d,r", lock_sql)

    def test_update_review_requires_ready_document(self):
        repository, connection = self._repository(
            [{"rows": [_reviewable_revision(document_status="failed")]}]
        )

        with self.assertRaisesRegex(DiagramStateError, "kg_pending or kg_done"):
            repository.update_draft_revision(
                "revision-1",
                _valid_graph(),
                expected_lock_version=1,
            )

        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_update_latest_draft_in_ready_document_succeeds(self):
        current = _reviewable_revision(lock_version=3, document_status="kg_done")
        updated = {**current, "lock_version": 4}
        repository, connection = self._repository(
            [{"rows": [current]}, {"rows": [updated]}]
        )

        result = repository.update_draft_revision(
            "revision-1",
            _valid_graph(),
            expected_lock_version=3,
        )

        self.assertEqual(4, result["lock_version"])
        self.assertEqual(connection.commits, 1)

    def test_confirmation_rejects_stale_lock_version(self):
        repository, connection = self._repository(
            [{"rows": [_reviewable_revision(lock_version=2)]}]
        )

        with self.assertRaises(DiagramConcurrencyError):
            repository.confirm_revision(
                "revision-1",
                reviewed_by="reviewer",
                expected_lock_version=1,
            )

        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_confirmation_requires_active_diagram(self):
        repository, connection = self._repository(
            [{"rows": [_reviewable_revision(lifecycle_status="archived")]}]
        )

        with self.assertRaisesRegex(DiagramStateError, "archived"):
            repository.confirm_revision(
                "revision-1",
                reviewed_by="reviewer",
                expected_lock_version=1,
            )

        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_rejection_requires_latest_revision(self):
        repository, connection = self._repository(
            [{"rows": [_reviewable_revision(revision_no=1, latest_revision_no=2)]}]
        )

        with self.assertRaisesRegex(DiagramStateError, "latest"):
            repository.reject_revision(
                "revision-1",
                reviewed_by="reviewer",
                review_note="证据不一致",
                expected_lock_version=1,
            )

        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_rejection_records_required_reviewer_note_and_lock(self):
        current = _reviewable_revision(review_status="conflicted", conflict_count=1)
        rejected = {
            **current,
            "review_status": "rejected",
            "reviewed_by": "reviewer",
            "review_note": "证据不一致",
            "lock_version": 2,
        }
        repository, connection = self._repository(
            [{"rows": [current]}, {"rows": [rejected]}]
        )

        result = repository.reject_revision(
            "revision-1",
            reviewed_by="reviewer",
            review_note="证据不一致",
            expected_lock_version=1,
        )

        self.assertEqual("rejected", result["review_status"])
        self.assertEqual(connection.commits, 1)

    def test_review_repository_rejects_function_locator(self):
        repository, connection = self._repository(
            [{"rows": [_reviewable_revision(diagram_type="function_locator")]}]
        )

        with self.assertRaisesRegex(DiagramStateError, "FTA/RBD"):
            repository.confirm_revision(
                "revision-1",
                reviewed_by="reviewer",
                expected_lock_version=1,
            )

        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_active_read_filters_confirmed_and_published(self):
        active = {
            "revision_id": "revision-1",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
        }
        repository, connection = self._repository([{"rows": [active]}])
        result = repository.get_active_revision("diagram-1")
        self.assertEqual(result["revision_id"], "revision-1")
        sql, _ = connection.executions[0]
        self.assertIn("r.review_status='confirmed'", sql)
        self.assertIn("r.projection_status='published'", sql)
        self.assertIn("d.lifecycle_status='active'", sql)
        self.assertIn("JOIN icp_documents doc", sql)
        self.assertIn("doc.status IN ('indexed','kg_pending','kg_done')", sql)

    def test_active_revision_list_fails_closed_for_processing_or_failed_documents(self):
        repository, connection = self._repository([{"rows": []}])

        self.assertEqual(
            [],
            repository.list_active_revisions(doc_id="doc-1", diagram_type="fta"),
        )

        sql, params = connection.executions[0]
        self.assertIn("JOIN icp_documents doc", sql)
        self.assertIn("doc.status IN ('indexed','kg_pending','kg_done')", sql)
        self.assertNotIn("'parsing'", sql)
        self.assertNotIn("'failed'", sql)
        self.assertEqual(("doc-1", "fta"), params)

    def test_both_active_revision_reads_include_authoritative_document_scope(self):
        metadata = {"configuration": "BUILD-Q", "document_version": "V8"}
        active = {"revision_id": "revision-3", "document_metadata": metadata, "doc_name": "source.docx"}
        repository, connection = self._repository([{"rows": [active]}, {"rows": [active]}])
        self.assertEqual(metadata, repository.get_active_revision("diagram-3")["document_metadata"])
        self.assertEqual(metadata, repository.list_active_revisions(diagram_type="rbd")[0]["document_metadata"])
        for sql, _ in connection.executions:
            self.assertIn("doc.metadata AS document_metadata", sql)
            self.assertIn("doc.file_name AS doc_name", sql)

    def test_active_read_keeps_last_confirmed_revision_while_new_draft_is_pending(self):
        active = {
            "revision_id": "revision-1",
            "revision_no": 1,
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
        }
        repository, connection = self._repository([{"rows": [active]}])

        result = repository.list_active_revisions(doc_id="doc-1", diagram_type="fta")

        self.assertEqual(["revision-1"], [item["revision_id"] for item in result])
        sql, _ = connection.executions[0]
        self.assertNotIn("latest_revision_no", sql)
        self.assertNotIn("MAX(candidate.revision_no)", sql)

    def test_document_diagram_list_hides_archived_by_default(self):
        repository, connection = self._repository([{"rows": []}, {"rows": []}])

        repository.list_diagrams_by_document("doc-1")
        repository.list_diagrams_by_document("doc-1", include_archived=True)

        self.assertIn("d.lifecycle_status='active'", connection.executions[0][0])
        self.assertNotIn("d.lifecycle_status='active'", connection.executions[1][0])

    def test_successful_reparse_archives_missing_and_stales_active_revision(self):
        repository, connection = self._repository(
            [
                {"rows": [{"doc_id": "doc-1"}]},
                {"rows": [{"diagram_id": "diagram-a"}, {"diagram_id": "diagram-b"}]},
                {"rows": [{"revision_id": "revision-b"}]},
                {"rows": [{"diagram_id": "diagram-b"}]},
                {"rows": []},
            ]
        )

        result = repository.archive_missing_diagrams(
            doc_id="doc-1",
            present_diagram_ids=["diagram-a"],
        )

        self.assertEqual(result["archived_diagram_ids"], ["diagram-b"])
        self.assertEqual(result["stale_revision_ids"], ["revision-b"])
        sql = [statement for statement, _ in connection.executions]
        stale_index = next(
            i for i, statement in enumerate(sql) if "projection_status='stale'" in statement
        )
        archive_index = next(
            i for i, statement in enumerate(sql) if "active_revision_id=NULL" in statement
        )
        self.assertLess(stale_index, archive_index)
        self.assertEqual(connection.commits, 1)

    def test_reparse_snapshot_rejects_foreign_or_unknown_diagram_id(self):
        repository, connection = self._repository(
            [
                {"rows": [{"doc_id": "doc-1"}]},
                {"rows": [{"diagram_id": "diagram-a"}]},
            ]
        )

        with self.assertRaisesRegex(DiagramStateError, "do not belong"):
            repository.archive_missing_diagrams(
                doc_id="doc-1",
                present_diagram_ids=["diagram-other-doc"],
            )
        self.assertEqual(len(connection.executions), 2)
        self.assertEqual(connection.rollbacks, 1)

    def test_reparse_snapshot_requires_explicit_empty_confirmation(self):
        repository, connection = self._repository([])

        with self.assertRaisesRegex(ValueError, "allow_empty"):
            repository.archive_missing_diagrams(
                doc_id="doc-1",
                present_diagram_ids=[],
            )
        self.assertEqual(connection.executions, [])

    def test_projection_refuses_unconfirmed_revision(self):
        repository, connection = self._repository(
            [
                {
                    "rows": [
                        {
                            "revision_id": "revision-1",
                            "diagram_id": "diagram-1",
                            "review_status": "pending",
                            "projection_status": "not_projected",
                            "active_revision_id": None,
                            "canonical_hash": "a" * 64,
                            "revision_no": 1,
                            "latest_revision_no": 1,
                        }
                    ]
                },
            ]
        )
        with self.assertRaises(DiagramStateError):
            repository.start_projection(
                "revision-1",
                idempotency_key="run-key",
                projector_version="v1",
                expected_vertex_count=1,
                expected_edge_count=0,
            )
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)

    def test_projection_accepts_auto_validated_pending_revision(self):
        revision = {
            "revision_id": "revision-1",
            "diagram_id": "diagram-1",
            "review_status": "pending",
            "validation_status": "auto_validated",
            "projection_status": "not_projected",
            "active_revision_id": None,
            "canonical_hash": "a" * 64,
            "revision_no": 1,
            "latest_revision_no": 1,
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-1",
            "canonical_hash": "a" * 64,
            "projector_version": "v1",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
            "status": "running",
        }
        repository, connection = self._repository(
            [
                {"rows": [revision]},
                {"rows": []},
                {"rows": [("revision-1",)]},
                {"rows": [run]},
            ]
        )

        result = repository.start_projection(
            "revision-1",
            idempotency_key="run-key",
            projector_version="v1",
            expected_vertex_count=2,
            expected_edge_count=1,
        )

        self.assertEqual("running", result["status"])
        self.assertEqual(connection.commits, 1)

    def test_projection_refuses_non_latest_confirmed_revision(self):
        repository, connection = self._repository(
            [
                {
                    "rows": [
                        {
                            "revision_id": "revision-1",
                            "diagram_id": "diagram-1",
                            "review_status": "confirmed",
                            "projection_status": "not_projected",
                            "active_revision_id": None,
                            "canonical_hash": "a" * 64,
                            "revision_no": 1,
                            "latest_revision_no": 2,
                        }
                    ]
                }
            ]
        )

        with self.assertRaisesRegex(DiagramStateError, "latest"):
            repository.start_projection(
                "revision-1",
                idempotency_key="run-key",
                projector_version="v1",
                expected_vertex_count=1,
                expected_edge_count=0,
            )
        self.assertEqual(len(connection.executions), 1)
        self.assertIn("latest_revision_no", connection.executions[0][0])

    def test_running_projection_retry_is_reused_after_latest_guard(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "publishing",
            "active_revision_id": "revision-1",
            "canonical_hash": "a" * 64,
            "revision_no": 2,
            "latest_revision_no": 2,
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "projector_version": "v1",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
            "status": "running",
        }
        repository, connection = self._repository(
            [{"rows": [revision]}, {"rows": [run]}]
        )

        result = repository.start_projection(
            "revision-2",
            idempotency_key="run-key",
            projector_version="v1",
            expected_vertex_count=2,
            expected_edge_count=1,
        )

        self.assertTrue(result["reused"])
        self.assertFalse(result["already_active"])
        self.assertEqual(connection.commits, 1)
        self.assertEqual(len(connection.executions), 2)

    def test_succeeded_active_projection_retry_is_readback_only(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "published",
            "active_revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "revision_no": 2,
            "latest_revision_no": 2,
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "projector_version": "v1",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
            "status": "succeeded",
        }
        repository, connection = self._repository(
            [{"rows": [revision]}, {"rows": [run]}]
        )

        result = repository.start_projection(
            "revision-2",
            idempotency_key="run-key",
            projector_version="v1",
            expected_vertex_count=2,
            expected_edge_count=1,
        )

        self.assertTrue(result["already_active"])
        self.assertEqual(len(connection.executions), 2)

    def test_failed_same_revision_projection_reuses_run_for_rebuild(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "failed",
            "active_revision_id": None,
            "canonical_hash": "a" * 64,
            "revision_no": 2,
            "latest_revision_no": 2,
        }
        failed_run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "projector_version": "v1",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
            "status": "failed",
            "attempt_count": 1,
        }
        retried_run = {**failed_run, "status": "running", "attempt_count": 2}
        repository, connection = self._repository(
            [
                {"rows": [revision]},
                {"rows": [failed_run]},
                {"rows": [("revision-2",)]},
                {"rows": [retried_run]},
            ]
        )

        result = repository.start_projection(
            "revision-2",
            idempotency_key="run-key",
            projector_version="v1",
            expected_vertex_count=2,
            expected_edge_count=1,
        )

        self.assertTrue(result["reused"])
        self.assertFalse(result["already_active"])
        self.assertEqual("running", result["status"])
        self.assertEqual(2, result["attempt_count"])
        self.assertIn("attempt_count=attempt_count+1", connection.executions[3][0])
        self.assertEqual(connection.commits, 1)

    def test_activation_refuses_revision_when_newer_draft_appeared(self):
        repository, connection = self._repository(
            [
                {
                    "rows": [
                        {
                            "revision_id": "revision-1",
                            "diagram_id": "diagram-1",
                            "review_status": "confirmed",
                            "projection_status": "publishing",
                            "active_revision_id": None,
                            "canonical_hash": "a" * 64,
                            "revision_no": 1,
                            "latest_revision_no": 2,
                        }
                    ]
                }
            ]
        )

        with self.assertRaisesRegex(DiagramStateError, "newer"):
            repository.activate_projection(
                "revision-1",
                run_id="run-1",
                manifest={},
                actual_vertex_count=1,
                actual_edge_count=0,
            )
        self.assertEqual(len(connection.executions), 1)
        self.assertEqual(connection.rollbacks, 1)

    def test_activation_validates_manifest_before_switching_active(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "publishing",
            "active_revision_id": "revision-1",
            "canonical_hash": "a" * 64,
            "revision_no": 2,
            "latest_revision_no": 2,
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "status": "running",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
        }
        repository, connection = self._repository(
            [{"rows": [revision]}, {"rows": [run]}]
        )

        with self.assertRaisesRegex(DiagramStateError, "manifest hash"):
            repository.activate_projection(
                "revision-2",
                run_id="run-1",
                manifest={
                    "revision_id": "revision-2",
                    "canonical_hash": "a" * 64,
                    "vertex_ids": ["v1", "v2"],
                    "edge_ids": ["e1"],
                    "manifest_hash": "bad",
                },
                actual_vertex_count=2,
                actual_edge_count=1,
            )
        self.assertEqual(len(connection.executions), 2)
        self.assertEqual(connection.rollbacks, 1)

    def test_activation_switches_active_only_after_verified_manifest(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "publishing",
            "active_revision_id": "revision-1",
            "canonical_hash": "a" * 64,
            "revision_no": 2,
            "latest_revision_no": 2,
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "status": "running",
            "expected_vertex_count": 2,
            "expected_edge_count": 1,
        }
        completed = {**run, "status": "succeeded"}
        repository, connection = self._repository(
            [
                {"rows": [revision]},
                {"rows": [run]},
                {},
                {"rowcount": 1},
                {"rowcount": 1},
                {"rows": [completed]},
            ]
        )
        manifest_data = {
            "revision_id": "revision-2",
            "canonical_hash": "a" * 64,
            "vertex_ids": ["v1", "v2"],
            "edge_ids": ["e1"],
        }
        manifest = {**manifest_data, "manifest_hash": content_hash(manifest_data)}

        result = repository.activate_projection(
            "revision-2",
            run_id="run-1",
            manifest=manifest,
            actual_vertex_count=2,
            actual_edge_count=1,
        )

        self.assertEqual(result["active_revision_id"], "revision-2")
        self.assertEqual(result["superseded_revision_id"], "revision-1")
        sql = [statement for statement, _ in connection.executions]
        self.assertLess(
            next(i for i, statement in enumerate(sql) if "projection_status='stale'" in statement),
            next(i for i, statement in enumerate(sql) if "active_revision_id=%s" in statement),
        )
        self.assertEqual(connection.commits, 1)

    def test_readback_drift_invalidates_active_revision_and_succeeded_run_atomically(self):
        revision = {
            "revision_id": "revision-2",
            "diagram_id": "diagram-1",
            "review_status": "confirmed",
            "projection_status": "published",
            "active_revision_id": "revision-2",
            "lifecycle_status": "active",
        }
        run = {
            "run_id": "run-1",
            "revision_id": "revision-2",
            "status": "succeeded",
        }
        invalidated = {**run, "status": "failed", "error": "readback mismatch"}
        repository, connection = self._repository(
            [
                {"rows": [revision]},
                {"rows": [run]},
                {"rowcount": 1},
                {"rowcount": 1},
                {"rows": [invalidated]},
            ]
        )

        result = repository.invalidate_active_projection(
            "revision-2",
            run_id="run-1",
            error="readback mismatch",
        )

        self.assertEqual("failed", result["status"])
        self.assertIsNone(result["active_revision_id"])
        sql = [statement for statement, _ in connection.executions]
        self.assertTrue(any("projection_status='failed'" in statement for statement in sql))
        self.assertTrue(any("active_revision_id=NULL" in statement for statement in sql))
        self.assertTrue(any("status='failed'" in statement for statement in sql))
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)


if __name__ == "__main__":
    unittest.main()
