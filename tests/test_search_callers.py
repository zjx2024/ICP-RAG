import unittest
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "deletion_helpers_for_search_tests",
    REPO_ROOT / "backend/api/routers/_deletion_helpers.py",
)
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
handle_community_reports = _HELPERS.handle_community_reports
CALLER_FILES = (
    "backend/core/parsers/diagram.py",
    "backend/core/parsers/excel_row.py",
    "backend/core/parsers/fault_history.py",
    "backend/core/parsers/function_architecture.py",
    "backend/services/document_service.py",
    "backend/services/kg_service.py",
    "backend/graphrag/graph_builder.py",
    "backend/api/routers/_deletion_helpers.py",
    "backend/api/routers/documents.py",
    "backend/api/routers/types.py",
    "backend/agent/tools/kg_search.py",
    "backend/storage/postgres_client.py",
)


class _FakePostgres:
    def __init__(self, events):
        self.events = events

    def get_community_reports(self, kb_id, doc_id=None):
        self.events.append(("pg", "read", kb_id, doc_id))
        return [
            {
                "projection_id": "kb_report_single",
                "report_id": "report_single",
                "kb_id": kb_id,
                "doc_id": doc_id,
                "doc_ids": [doc_id],
                "entities": ["A"],
                "report": {"title": "single"},
            },
            {
                "projection_id": "kb_report_shared",
                "report_id": "report_shared",
                "kb_id": kb_id,
                "doc_id": doc_id,
                "doc_ids": [doc_id, "doc-2"],
                "entities": ["A", "B"],
                "report": {"title": "shared"},
            },
        ]

    def delete_community_report(self, projection_id):
        self.events.append(("pg", "delete", projection_id))
        return True

    def mark_community_report_stale(
        self, projection_id, doc_ids, stale_kwd, reason, stale_at
    ):
        self.events.append(
            ("pg", "stale", projection_id, tuple(doc_ids), stale_kwd, reason)
        )
        return True

    def upsert_community_report(self, projection_id, report):
        self.events.append(("pg", "restore", projection_id, report.get("report_id")))
        return projection_id


class _FakeSearchStore:
    def __init__(self, events):
        self.events = events

    def delete_by_id(self, projection_id):
        self.events.append(("search", "delete", projection_id))
        return True

    def upsert(self, projection_id, document, vector=None):
        self.events.append(
            (
                "search",
                "upsert",
                projection_id,
                tuple(document.get("doc_ids", [])),
                document.get("stale_kwd"),
            )
        )


class _FailingDeleteSearchStore(_FakeSearchStore):
    def delete_by_id(self, projection_id):
        self.events.append(("search", "delete_failed", projection_id))
        raise RuntimeError("projection unavailable")


class SearchCallerMigrationTests(unittest.TestCase):
    def test_callers_do_not_reach_through_legacy_es_client(self):
        forbidden = (
            "backend.storage.es_client",
            "es_client.client",
            "transport.hosts",
            '"/_search"',
            "'/_search'",
        )
        for relative_path in CALLER_FILES:
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{relative_path} still contains {marker}")

    def test_community_report_truth_is_updated_before_projection(self):
        events = []
        result = handle_community_reports(
            _FakeSearchStore(events),
            _FakePostgres(events),
            doc_id="doc-1",
            kb_id="kb",
            fully_removed_entities=["B"],
        )

        self.assertEqual({"deleted": 1, "updated": 1}, result)
        self.assertLess(
            events.index(("pg", "delete", "kb_report_single")),
            events.index(("search", "delete", "kb_report_single")),
        )
        pg_stale_index = next(
            index
            for index, event in enumerate(events)
            if event[:3] == ("pg", "stale", "kb_report_shared")
        )
        search_upsert_index = next(
            index
            for index, event in enumerate(events)
            if event[:3] == ("search", "upsert", "kb_report_shared")
        )
        self.assertLess(pg_stale_index, search_upsert_index)
        self.assertEqual("topology", events[pg_stale_index][4])
        self.assertEqual(("doc-2",), events[search_upsert_index][3])

    def test_failed_projection_delete_restores_postgres_retry_marker(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "projection unavailable"):
            handle_community_reports(
                _FailingDeleteSearchStore(events),
                _FakePostgres(events),
                doc_id="doc-1",
                kb_id="kb",
                fully_removed_entities=[],
            )
        self.assertLess(
            events.index(("pg", "delete", "kb_report_single")),
            events.index(("search", "delete_failed", "kb_report_single")),
        )
        self.assertIn(
            ("pg", "restore", "kb_report_single", "report_single"), events
        )


if __name__ == "__main__":
    unittest.main()
