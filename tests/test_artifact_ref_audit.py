import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import audit_artifact_refs as audit


class _FakeCursor:
    def __init__(self, rows_by_table):
        self.rows_by_table = rows_by_table
        self.rows = []
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, query):
        self.queries.append(query)
        for table, rows in self.rows_by_table.items():
            if f'FROM "{table}"' in query:
                self.rows = list(rows)
                return
        raise AssertionError(f"unexpected query: {query}")

    def __iter__(self):
        return iter(self.rows)


class _FakeConnection:
    def __init__(self, rows_by_table=None):
        self.rows_by_table = {
            "documents": [],
            "icp_documents": [],
            "document_images": [],
        }
        self.rows_by_table.update(rows_by_table or {})
        self.cursor_instance = _FakeCursor(self.rows_by_table)
        self.session_calls = []
        self.rollback_calls = 0
        self.close_calls = 0
        self.commit_calls = 0

    def set_session(self, **kwargs):
        self.session_calls.append(kwargs)

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rollback_calls += 1

    def commit(self):
        self.commit_calls += 1

    def close(self):
        self.close_calls += 1


class ArtifactReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.artifact_root = Path(self._temporary_directory.name) / "artifacts"
        self.artifact_root.mkdir()

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write(self, storage_key, payload=b"artifact"):
        path = self.artifact_root / storage_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _run_main(self, connection, *extra_args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        args = ["--artifact-root", str(self.artifact_root), *extra_args]
        with mock.patch.object(audit, "connect_database", return_value=connection), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            return_code = audit.main(args)
        return return_code, stdout.getvalue(), stderr.getvalue()

    def test_all_three_tables_are_scanned_in_a_read_only_session(self):
        document_key = "documents/kb-1/doc-1/source.docx"
        icp_key = "icp/hmc/doc-2/source.xlsx"
        image_key = "images/doc-1/evidence.png"
        for storage_key in (document_key, icp_key, image_key):
            self._write(storage_key)
        connection = _FakeConnection(
            {
                "documents": [("doc-1", document_key)],
                "icp_documents": [("doc-2", icp_key)],
                "document_images": [("image-1", image_key)],
            }
        )

        result = audit.audit_references(connection, self.artifact_root)

        self.assertTrue(result.is_consistent)
        self.assertEqual(result.references_scanned, 3)
        self.assertEqual(result.unique_storage_keys, 3)
        self.assertEqual(result.reference_issues, [])
        self.assertEqual(
            connection.session_calls,
            [{"readonly": True, "autocommit": False}],
        )
        self.assertEqual(len(connection.cursor_instance.queries), 3)
        self.assertTrue(
            all(
                '"minio_path" IS NOT NULL' in query
                and 'BTRIM("minio_path")' in query
                for query in connection.cursor_instance.queries
            )
        )

    def test_missing_and_invalid_references_return_inconsistent_exit_code(self):
        connection = _FakeConnection(
            {
                "documents": [
                    ("doc-missing", "documents/kb/missing.pdf"),
                    ("doc-whitespace", "documents/kb/source.pdf "),
                ],
                "icp_documents": [("doc-invalid", "../outside.bin")],
            }
        )

        return_code, stdout, stderr = self._run_main(connection, "--json")

        self.assertEqual(return_code, audit.EXIT_INCONSISTENT)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertFalse(payload["consistent"])
        issues = {
            (issue["table"], issue["owner_id"]): issue["reason"]
            for issue in payload["reference_issues"]
        }
        self.assertEqual(issues[("documents", "doc-missing")], "not_found")
        self.assertEqual(issues[("documents", "doc-whitespace")], "invalid_storage_key")
        self.assertEqual(issues[("icp_documents", "doc-invalid")], "invalid_storage_key")
        self.assertEqual(connection.rollback_calls, 1)
        self.assertEqual(connection.close_calls, 1)
        self.assertEqual(connection.commit_calls, 0)

    def test_orphan_scan_excludes_staging_trash_and_reports_only_real_artifacts(self):
        referenced_key = "icp/hmc/doc-1/source.xlsx"
        orphan_key = "images/orphan/evidence.png"
        self._write(referenced_key)
        self._write(orphan_key)
        self._write(".staging/in-flight.part")
        self._write(".trash/20260101/deleted.bin")
        connection = _FakeConnection({"icp_documents": [("doc-1", referenced_key)]})

        return_code, stdout, stderr = self._run_main(connection, "--check-orphans", "--json")

        self.assertEqual(return_code, audit.EXIT_INCONSISTENT, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["orphan_keys"], [orphan_key])
        self.assertTrue(payload["orphan_scan_enabled"])

    def test_orphan_scan_is_opt_in(self):
        self._write("images/orphan/evidence.png")
        connection = _FakeConnection()

        return_code, stdout, stderr = self._run_main(connection, "--json")

        self.assertEqual(return_code, audit.EXIT_OK, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["consistent"])
        self.assertFalse(payload["orphan_scan_enabled"])
        self.assertEqual(payload["orphan_keys"], [])

    def test_main_does_not_modify_artifact_tree_or_commit_database(self):
        storage_key = "documents/kb/doc/source.pdf"
        path = self._write(storage_key, b"immutable evidence")
        connection = _FakeConnection({"documents": [("doc", storage_key)]})
        before_entries = sorted(
            item.relative_to(self.artifact_root).as_posix()
            for item in self.artifact_root.rglob("*")
        )
        before_stat = path.stat()

        return_code, _stdout, stderr = self._run_main(connection, "--check-orphans")

        after_entries = sorted(
            item.relative_to(self.artifact_root).as_posix()
            for item in self.artifact_root.rglob("*")
        )
        after_stat = path.stat()
        self.assertEqual(return_code, audit.EXIT_OK, stderr)
        self.assertEqual(before_entries, after_entries)
        self.assertEqual(before_stat.st_ino, after_stat.st_ino)
        self.assertEqual(before_stat.st_size, after_stat.st_size)
        self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)
        self.assertEqual(connection.commit_calls, 0)

    def test_missing_artifact_root_is_an_audit_error_and_is_not_created(self):
        missing_root = self.artifact_root / "not-mounted"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            audit,
            "connect_database",
            side_effect=AssertionError("database must not be opened"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = audit.main(["--artifact-root", str(missing_root), "--json"])

        self.assertEqual(return_code, audit.EXIT_AUDIT_ERROR)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("artifact root does not exist", stderr.getvalue())
        self.assertFalse(missing_root.exists())

    def test_database_failure_has_distinct_audit_error_exit_code(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            audit,
            "connect_database",
            side_effect=audit.AuditError("database unavailable"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = audit.main(
                ["--artifact-root", str(self.artifact_root), "--json"]
            )

        self.assertEqual(return_code, audit.EXIT_AUDIT_ERROR)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["audit_error"], "database unavailable")

    def test_database_settings_use_application_environment(self):
        environment = {
            "POSTGRES_HOST": "postgres.internal",
            "POSTGRES_PORT": "5544",
            "POSTGRES_USER": "audit-user",
            "POSTGRES_PASSWORD": "secret",
            "POSTGRES_DB": "icp",
            "POSTGRES_CONNECT_TIMEOUT": "7",
            "POSTGRES_SSLMODE": "require",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            settings = audit.database_settings_from_env()

        self.assertEqual(settings["host"], "postgres.internal")
        self.assertEqual(settings["port"], 5544)
        self.assertEqual(settings["user"], "audit-user")
        self.assertEqual(settings["password"], "secret")
        self.assertEqual(settings["dbname"], "icp")
        self.assertEqual(settings["connect_timeout"], 7)
        self.assertEqual(settings["sslmode"], "require")


if __name__ == "__main__":
    unittest.main()
