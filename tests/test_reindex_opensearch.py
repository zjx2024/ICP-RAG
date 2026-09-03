import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "reindex_opensearch.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("reindex_opensearch_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reindex = _load_script_module()


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [[float(index)] * 1024 for index, _ in enumerate(texts, start=1)]


class FakeSearchStore:
    def __init__(self):
        self.ready_calls = 0
        self.documents = {"old": ({"projection_source_kwd": "old"}, [0.0] * 1024)}
        self.reset_calls = 0

    def ensure_ready(self):
        self.ready_calls += 1

    def delete_by_filter(self, filters):
        self.reset_calls += 1
        if filters is not None:
            raise AssertionError("test reset expects an unfiltered delete")
        deleted = len(self.documents)
        self.documents.clear()
        return deleted

    def upsert(self, projection_id, document, vector=None):
        self.documents[projection_id] = (dict(document), list(vector))

    def count(self, filters=None):
        if not filters:
            return len(self.documents)
        return sum(
            1
            for document, _ in self.documents.values()
            if all(document.get(key) == value for key, value in filters.items())
        )


class MismatchingSearchStore(FakeSearchStore):
    def count(self, filters=None):
        count = super().count(filters)
        if filters == {"projection_source_kwd": "fmea"}:
            return count + 1
        return count


def metadata_tagger(metadata):
    return {
        "document_metadata": dict(metadata),
        **{f"{key}_kwd": value for key, value in metadata.items()},
    }


def fixture_rows():
    rows = {
        reindex.ICP_DOCUMENTS_QUERY: [
            {
                "doc_id": "icp-1",
                "doc_type": "hmc",
                "file_name": "HMC.docx",
                "metadata": {
                    "document_number": "SYN-ICP-DOC-002",
                    "applicable_configuration": "CONFIG-A",
                },
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-02T00:00:00",
            }
        ],
        reindex.LEGACY_DOCUMENTS_QUERY: [
            {
                "doc_id": "legacy-1",
                "kb_id": "kb-1",
                "file_name": "legacy.txt",
                "status": "parsed",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        ],
    }
    source_rows = {
        "document_chunks": [
            {
                "chunk_id": "legacy-chunk-1",
                "doc_id": "legacy-1",
                "section": "第一节",
                "title": "旧知识块",
                "content": "旧知识库正文",
                "content_type": "text",
                "chunk_index": 0,
                "selected_for_kg": True,
                "created_at": "2026-01-01T00:00:00",
            }
        ],
        "fault_hmc": [
            {
                "id": 11,
                "doc_id": "icp-1",
                "variant_id": "variant-hmc-001",
                "hmc_code": "1201**0007",
                "lcn": "1201",
                "hamc": "**",
                "hsc": "0007",
                "fault_desc": "显示单元无输出",
                "repair_method": "检查供电后更换模块",
                "source_ref": "HMC.docx#table=1;row=3",
            }
        ],
        "fmea": [
            {
                "id": 12,
                "doc_id": "icp-1",
                "code": "F-001",
                "product_flag": "显示模块",
                "func_management": "信息显示",
                "fault_mode": "黑屏",
                "fault_cause": "供电异常",
                "mission_phase": "全阶段",
                "local_impact": "本机无显示",
                "final_impact": "信息不可见",
                "severity": "II",
                "severity_raw": "Ⅱ类",
                "severity_normalized": "II",
                "probability": "0.01",
                "probability_raw": "1%",
                "probability_level": "",
                "detect_method": "BIT",
                "design_improve": "冗余供电",
                "compensate_action": "切换备份",
            }
        ],
        "mfl": [
            {
                "id": 13,
                "doc_id": "icp-1",
                "mfl_code": "MFL-001",
                "module_fault": "显示模块异常",
                "sur_fault": "SUR板无输出",
                "subcircuit_fault": "电源电路故障",
                "api_reportable": True,
                "api_name": "getDisplayState",
                "api_method": "GET",
                "api_usage": "读取状态",
                "platform_test": False,
                "platform_test_raw": "部分完成",
                "platform_test_status": "部分完成",
                "maintenance_action": "检查接口",
            }
        ],
        "icp_functions": [
            {
                "id": 14,
                "doc_id": "icp-1",
                "function_id": "FUNC-001",
                "system_name": "综合信息处理与显示系统",
                "subsystem_name": "显示子系统",
                "module_name": "显示模块",
                "function_name": "告警显示",
                "description": "显示告警信息",
                "aliases": ["告警页面"],
                "failure_symptoms": "黑屏",
                "source_ref": "HMC.docx#1",
            }
        ],
        "fault_history_cases": [
            {
                "id": 15,
                "doc_id": "icp-1",
                "case_id": "CASE-001",
                "occurred_at": "2025-01-01",
                "personnel": "测试员",
                "device_type": "CONFIG-A",
                "fault_symptom": "间歇黑屏",
                "related_module": "显示模块",
                "diagnosis_method": "检查供电",
                "root_cause": "接插件松动",
                "final_solution": "重新紧固",
                "preventive_action": "定期检查",
                "hmc_code": "1201**0007",
                "fmeca_code": "F-001",
                "fta_ref": "FTA-001",
                "mfl_code": "MFL-001",
                "source_ref": "history.docx#case-1",
                "steps": ["测量电压", "紧固接插件"],
            }
        ],
        "document_fault_trees": [
            {
                "id": "tree-1",
                "doc_id": "icp-1",
                "kb_id": "fta",
                "chunk_id": "chapter-1",
                "image_id": "fta-image-1",
                "image_title": "显示失效故障树",
                "graph": {
                    "diagram_type": "fta",
                    "root_fault_id": "top",
                    "metadata": {"description": "显示失效原因分析", "confidence": 0.9},
                    "vertices": [
                        {"id": "top", "label": "TopEvent", "properties": {"name": "显示失效"}},
                        {"id": "basic", "label": "BasicEvent", "properties": {"name": "供电故障"}},
                    ],
                    "edges": [
                        {"source": "basic", "target": "top", "label": "causes", "properties": {"gate_type": "OR"}}
                    ],
                },
            }
        ],
        "community_reports": [
            {
                "projection_id": "kb-1_community-1",
                "report_id": "community-1",
                "community_id": "1",
                "kb_id": "kb-1",
                "doc_id": "icp-1",
                "doc_ids": ["icp-1"],
                "entities": ["显示模块", "供电模块"],
                "report": {
                    "title": "显示链路社区",
                    "summary": "供电异常可导致显示失效",
                    "findings": [{"summary": "检查供电"}],
                },
                "stale_kwd": "N",
                "stale_reason": "",
                "stale_at": None,
            }
        ],
    }
    for source, query in reindex.SOURCE_QUERIES:
        rows[query] = source_rows[source]
    return rows


def fixture_reader(rows):
    def read(_connection, query, batch_size):
        if batch_size <= 0:
            raise AssertionError("batch size was not validated")
        return iter(rows[query])

    return read


class ReindexOpenSearchTests(unittest.TestCase):
    def test_dry_run_counts_every_source_without_models_or_search(self):
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError(f"dry-run called {name}")

        result = reindex.run_reindex(
            object(),
            Forbidden(),
            Forbidden(),
            metadata_tagger,
            dry_run=True,
            batch_size=3,
            row_reader=fixture_reader(fixture_rows()),
        )

        self.assertEqual(
            result["sources"],
            {
                "document_chunks": 1,
                "fault_hmc": 2,
                "fmea": 1,
                "mfl": 1,
                "icp_functions": 1,
                "fault_history_cases": 1,
                "document_fault_trees": 4,
                "community_reports": 1,
            },
        )
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["indexed"], {})

    def test_reset_rebuild_batches_embeddings_and_audits_metadata(self):
        embedder = FakeEmbedder()
        store = FakeSearchStore()

        result = reindex.run_reindex(
            object(),
            embedder,
            store,
            metadata_tagger,
            batch_size=3,
            reset=True,
            yes=True,
            row_reader=fixture_reader(fixture_rows()),
        )

        self.assertEqual(store.ready_calls, 1)
        self.assertEqual(store.reset_calls, 1)
        self.assertEqual(result["reset_deleted"], 1)
        self.assertEqual([len(call) for call in embedder.calls], [3, 3, 3, 3])
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["sources"], result["indexed"])
        self.assertEqual(result["sources"], result["audited"])
        self.assertEqual(result["index_total"], 12)

        diagnostic, diagnostic_vector = store.documents[
            "icp-1_hmc_variant-hmc-001_diagnostic"
        ]
        maintenance, _ = store.documents[
            "icp-1_hmc_variant-hmc-001_maintenance"
        ]
        self.assertEqual(diagnostic["content_type"], "hmc_diagnostic")
        self.assertEqual(maintenance["content_type"], "hmc_maintenance")
        self.assertEqual(diagnostic["document_number_kwd"], "SYN-ICP-DOC-002")
        self.assertEqual(
            diagnostic["document_metadata"]["applicable_configuration"], "CONFIG-A"
        )
        self.assertEqual(len(diagnostic_vector), 1024)
        self.assertEqual(diagnostic["variant_id_kwd"], "variant-hmc-001")

        mfl, _ = store.documents["icp-1_row_13_mfl"]
        self.assertEqual(mfl["platform_test_status_kwd"], "部分完成")
        self.assertIn("平台级摸底实验: 部分完成", mfl["content"])

        self.assertIn("fta-image-1", store.documents)
        self.assertIn("fta-image-1_node_0", store.documents)
        self.assertIn("fta-image-1_edge_0", store.documents)
        community, _ = store.documents["kb-1_community-1"]
        self.assertEqual(community["knowledge_graph_kwd"], "community_report")
        self.assertIn("供电异常", community["content"])

    def test_reset_requires_explicit_confirmation_even_programmatically(self):
        with self.assertRaisesRegex(ValueError, "--yes"):
            reindex.run_reindex(
                object(),
                FakeEmbedder(),
                FakeSearchStore(),
                metadata_tagger,
                reset=True,
                yes=False,
                row_reader=fixture_reader(fixture_rows()),
            )

    def test_audit_mismatch_is_a_hard_failure(self):
        with self.assertRaisesRegex(RuntimeError, "投影审计不一致"):
            reindex.run_reindex(
                object(),
                FakeEmbedder(),
                MismatchingSearchStore(),
                metadata_tagger,
                batch_size=20,
                reset=True,
                yes=True,
                row_reader=fixture_reader(fixture_rows()),
            )

    def test_main_returns_nonzero_when_runtime_dependency_fails(self):
        def fail_dependencies():
            raise RuntimeError("PostgreSQL unavailable")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = reindex.main([], dependency_loader=fail_dependencies)
        self.assertEqual(code, 1)
        self.assertIn("PostgreSQL unavailable", stderr.getvalue())

    def test_main_dry_run_prints_per_source_counts(self):
        class Connection:
            closed = False

            def close(self):
                self.closed = True

        connection = Connection()

        def dependencies():
            return connection, FakeEmbedder(), FakeSearchStore(), metadata_tagger

        original_reader = reindex.iter_query
        reindex.iter_query = fixture_reader(fixture_rows())
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                code = reindex.main(["--dry-run", "--batch-size", "2"], dependencies)
        finally:
            reindex.iter_query = original_reader

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["total"], 12)
        self.assertEqual(report["sources"]["fault_hmc"], 2)
        self.assertTrue(connection.closed)

    def test_import_does_not_load_application_storage_or_model_modules(self):
        isolated_name = "reindex_opensearch_import_safety_test"
        tracked = (
            "backend.storage.postgres_client",
            "backend.storage.search_store",
            "backend.core.embedding",
        )
        before = {name: sys.modules.get(name) for name in tracked}
        spec = importlib.util.spec_from_file_location(isolated_name, SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name in tracked:
            self.assertIs(sys.modules.get(name), before[name])


if __name__ == "__main__":
    unittest.main()
