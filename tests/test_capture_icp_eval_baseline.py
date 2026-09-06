import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import capture_icp_eval_baseline as baseline


class CaptureIcpEvalBaselineTests(unittest.TestCase):
    def test_source_snapshot_includes_runtime_files_outside_curated_role_groups(self):
        self._complete_local_tree()
        with patch.object(baseline, "_collect_git", return_value={}), patch.object(
            baseline, "_collect_environment", return_value={},
        ):
            before = baseline.capture_baseline(self.root, offline=True)
            self._write("backend/icp/extra_repository.py", "CURRENT_SOURCE = 2\n")
            after = baseline.capture_baseline(self.root, offline=True)
        self.assertNotEqual(
            before["code_inputs"]["application_source"]["aggregate_sha256"],
            after["code_inputs"]["application_source"]["aggregate_sha256"],
        )

    def test_fingerprint_covers_every_answer_layer_after_decomposition(self):
        root = Path(__file__).resolve().parents[1]
        modules = {str(path.relative_to(root)) for path in (root / "backend/agent/nodes").glob("answer*.py")}
        self.assertTrue(modules.issubset(baseline.AGENT_FILES), modules - set(baseline.AGENT_FILES))

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write(self, relative_path, content=b"fixture"):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return path

    def _complete_local_tree(self):
        for index, (_doc_type, filename) in enumerate(baseline.EXPECTED_DOCUMENTS):
            self._write(f"test-docs/{filename}", f"document-{index}".encode())

        eval_payload = {
            "name": "synthetic suite",
            "version": "test-v1",
            "cases": [
                {"id": "one", "turns": [{"question": "a"}, {"question": "b"}]},
                {"id": "two", "turns": [{"question": "c"}]},
            ],
        }
        for _suite_id, relative_path in baseline.EVALUATION_FILES:
            self._write(relative_path, json.dumps(eval_payload, ensure_ascii=False))

        for relative_path in (
            *baseline.PARSER_RELIABILITY_FILES,
            *baseline.AGENT_FILES,
            *baseline.EVALUATOR_FILES,
        ):
            self._write(relative_path, f"# {relative_path}\n")

    def test_offline_manifest_contains_all_local_fingerprints(self):
        self._complete_local_tree()
        with (
            patch.object(baseline, "_collect_git", return_value={"available": True, "dirty": False}),
            patch.object(baseline, "_collect_environment", return_value={"python": "test"}),
        ):
            manifest = baseline.capture_baseline(self.root, offline=True)

        self.assertEqual(baseline.SCHEMA_VERSION, manifest["schema_version"])
        self.assertEqual("complete", manifest["status"])
        self.assertEqual([], manifest["errors"])
        self.assertEqual("offline", manifest["services"]["mode"])
        self.assertFalse(manifest["services"]["postgres_direct_connection"])
        self.assertEqual(7, len(manifest["fixtures"]["files"]))
        self.assertTrue(all(item["exists"] for item in manifest["fixtures"]["files"]))

        suites = {item["suite_id"]: item for item in manifest["evaluation_suites"]["files"]}
        self.assertEqual(2, suites["canary_19"]["cases"])
        self.assertEqual(3, suites["canary_19"]["turns"])
        self.assertRegex(manifest["fixtures"]["aggregate_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            manifest["evaluation_suites"]["regression_100_aggregate_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_missing_local_inputs_are_reported_without_aborting(self):
        with (
            patch.object(baseline, "_collect_git", return_value={"available": False}),
            patch.object(baseline, "_collect_environment", return_value={}),
        ):
            manifest = baseline.capture_baseline(self.root, offline=True)

        self.assertEqual("complete_with_errors", manifest["status"])
        self.assertTrue(manifest["errors"])
        self.assertTrue(
            any(item["component"] == "fixtures" for item in manifest["errors"])
        )

    def test_sensitive_values_are_recursively_redacted_but_max_tokens_remains(self):
        value = {
            "api_key": "key-value",
            "nested": {
                "password": "password-value",
                "accessToken": "token-value",
                "clientSecret": "secret-value",
                "max_tokens": 4096,
                "api_url": "https://user:pass@example.test/v1?q=hidden#fragment",
            },
        }

        scrubbed = baseline._scrub_sensitive(value)

        self.assertEqual(baseline.REDACTED, scrubbed["api_key"])
        self.assertEqual(baseline.REDACTED, scrubbed["nested"]["password"])
        self.assertEqual(baseline.REDACTED, scrubbed["nested"]["accessToken"])
        self.assertEqual(baseline.REDACTED, scrubbed["nested"]["clientSecret"])
        self.assertEqual(4096, scrubbed["nested"]["max_tokens"])
        self.assertEqual("https://example.test/v1", scrubbed["nested"]["api_url"])
        self.assertNotIn("key-value", json.dumps(scrubbed))
        self.assertNotIn("token-value", json.dumps(scrubbed))

    def test_application_probe_keeps_only_safe_document_and_dify_data(self):
        class FakeClient:
            def __init__(self, _base_url, _timeout_seconds):
                pass

            def get(self, path):
                if path == "/readyz":
                    return 200, {
                        "status": "ready",
                        "checks": {
                            "postgresql": {"ok": True, "detail": "secret DSN"},
                            "opensearch": {"ok": True},
                        },
                    }
                if path == "/api/v1/document-types":
                    return 200, [
                        {"doc_type": doc_type, "internal": "not retained"}
                        for doc_type, _filename in baseline.EXPECTED_DOCUMENTS
                    ]
                if path.endswith("/documents"):
                    doc_type = path.split("/")[4]
                    return 200, [
                        {
                            "doc_id": f"doc-{doc_type}",
                            "doc_type": doc_type,
                            "file_name": f"{doc_type}.docx",
                            "status": "kg_done",
                            "parse_count": 1,
                            "minio_path": "must-not-be-retained",
                            "metadata": {"api_key": "hidden"},
                        }
                    ]
                if path.endswith("/parse-result"):
                    return 200, {
                        "status": "kg_done",
                        "parse_count": 1,
                        "last_result": {
                            "rows_ok": 4,
                            "raw_content": "must-not-be-retained",
                        },
                    }
                if path == "/api/v1/models/current-config":
                    return 200, {
                        "chat_model": {
                            "model_name": "test-model",
                            "api_key": "top-secret",
                            "max_tokens": 2048,
                        }
                    }
                if path == "/openapi.json":
                    return 200, {
                        "paths": {"/api/v1/dify/retrieval": {"post": {}}}
                    }
                if path == "/api/v1/dify/server-info":
                    return 200, {"lan_ip": "192.0.2.8", "token": "hidden"}
                raise AssertionError(path)

        errors = []
        with patch.object(baseline, "JsonGetClient", FakeClient):
            result = baseline._probe_application("http://app.test", 1.0, errors)

        self.assertEqual([], errors)
        self.assertTrue(result["document_types"]["ok"])
        self.assertEqual({"ok": True}, result["readyz"]["checks"]["postgresql"])
        document = result["documents"]["hmc"][0]
        self.assertNotIn("minio_path", document)
        self.assertNotIn("raw_content", document["parse_result"]["stats"])
        self.assertEqual(4, document["parse_result"]["stats"]["rows_ok"])
        self.assertEqual(
            baseline.REDACTED,
            result["models"]["configuration"]["chat_model"]["api_key"],
        )
        self.assertEqual(2048, result["models"]["configuration"]["chat_model"]["max_tokens"])
        self.assertEqual(
            {
                "server_info_get": {"ok": True, "status_code": 200},
                "retrieval_post_registered": True,
                "response_body_retained": False,
            },
            result["dify"],
        )
        self.assertNotIn("192.0.2.8", json.dumps(result))

    def test_service_probe_failures_are_manifest_data(self):
        self._complete_local_tree()

        class UnreachableClient:
            def __init__(self, _base_url, _timeout_seconds):
                pass

            def get(self, _path):
                raise baseline.ProbeError("connection refused")

        with (
            patch.object(baseline, "JsonGetClient", UnreachableClient),
            patch.object(baseline, "_collect_git", return_value={"available": True, "dirty": False}),
            patch.object(baseline, "_collect_environment", return_value={}),
        ):
            manifest = baseline.capture_baseline(self.root, offline=False)

        self.assertEqual("complete_with_errors", manifest["status"])
        self.assertTrue(manifest["errors"])
        self.assertEqual("read_only_http", manifest["services"]["mode"])
        self.assertEqual(["GET"], manifest["services"]["http_methods_used"])
        self.assertFalse(manifest["services"]["postgres_direct_connection"])

    def test_opensearch_and_hugegraph_capture_only_required_identity(self):
        class FakeClient:
            def __init__(self, base_url, _timeout_seconds):
                self.base_url = base_url

            def get(self, path):
                if path == "/_cluster/health":
                    return 200, {"cluster_name": "test", "status": "yellow", "password": "x"}
                if path == "/_alias":
                    return 200, {
                        "knowledge_graph_default_v1": {
                            "aliases": {"knowledge_graph_default": {"filter": "not retained"}}
                        }
                    }
                if path.startswith("/_cat/indices"):
                    return 200, [
                        {
                            "index": "knowledge_graph_default_v1",
                            "uuid": "index-uuid",
                            "docs.count": "42",
                            "store.size": "not retained",
                        }
                    ]
                if path == "/versions":
                    return 200, {
                        "versions": {"version": "1.5.0", "core": "not retained"},
                        "token": "not retained",
                    }
                raise AssertionError(path)

        errors = []
        with patch.object(baseline, "JsonGetClient", FakeClient):
            opensearch = baseline._probe_opensearch("http://search.test", 1.0, errors)
            hugegraph = baseline._probe_hugegraph("http://graph.test", 1.0, errors)

        self.assertEqual([], errors)
        self.assertEqual(
            [{"alias": "knowledge_graph_default", "index": "knowledge_graph_default_v1"}],
            opensearch["aliases"]["items"],
        )
        self.assertEqual(42, opensearch["indices"]["items"][0]["document_count"])
        self.assertEqual(
            {"versions.version": "1.5.0"}, hugegraph["version"]["fields"]
        )
        self.assertNotIn("not retained", json.dumps({"opensearch": opensearch, "hugegraph": hugegraph}))

    @unittest.skipUnless(shutil.which("git"), "git executable is required")
    def test_git_snapshot_includes_dirty_status_and_tracked_diff_hash(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "baseline@example.invalid"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Baseline Test"], cwd=self.root, check=True
        )
        tracked = self._write("tracked.txt", "before\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=self.root, check=True)
        tracked.write_text("after\n", encoding="utf-8")
        self._write("untracked.txt", "new\n")

        errors = []
        snapshot = baseline._collect_git(self.root, errors)

        self.assertEqual([], errors)
        self.assertTrue(snapshot["available"])
        self.assertTrue(snapshot["dirty"])
        self.assertEqual(1, snapshot["status"]["tracked_changed"])
        self.assertEqual(1, snapshot["status"]["untracked"])
        self.assertEqual(0, snapshot["status"]["staged"])
        self.assertEqual(1, snapshot["status"]["worktree_changed"])
        self.assertIn(" M tracked.txt", snapshot["status"]["entries"])
        self.assertIn("?? untracked.txt", snapshot["status"]["entries"])
        self.assertRegex(snapshot["tracked_diff"]["sha256"], r"^[0-9a-f]{64}$")

    def test_atomic_output_is_valid_json_and_replaces_existing_file(self):
        destination = self.root / "nested" / "manifest.json"
        destination.parent.mkdir(parents=True)
        destination.write_text("old", encoding="utf-8")
        manifest = {"schema_version": baseline.SCHEMA_VERSION, "api_key": baseline.REDACTED}

        baseline._write_manifest(manifest, str(destination), compact=False)

        self.assertEqual(manifest, json.loads(destination.read_text(encoding="utf-8")))
        self.assertEqual([], list(destination.parent.glob(".manifest.json.*.tmp")))

    def test_aggregate_is_stable_across_input_order(self):
        entries = [
            {"path": "b", "exists": True, "size_bytes": 2, "sha256": "b" * 64},
            {"path": "a", "exists": True, "size_bytes": 1, "sha256": "a" * 64},
        ]
        self.assertEqual(
            baseline._aggregate_file_entries(entries),
            baseline._aggregate_file_entries(reversed(entries)),
        )


if __name__ == "__main__":
    unittest.main()
