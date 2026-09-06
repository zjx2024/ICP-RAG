import unittest
from unittest.mock import patch

from backend.icp.repository import IcpRepository
from backend.storage.postgres_client import PostgresClient


class _Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.queries = connection.queries
        self.description = []
        self.rowcount = 0
        self._one = None
        self._all = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        params = tuple(params or ())
        self.queries.append((" ".join(query.split()), params))
        self.rowcount = 1
        self._all = []
        self._one = (101,) if "RETURNING id" in query else None
        if query.count("%s") != len(params):
            raise AssertionError(
                f"SQL placeholder mismatch: {query.count('%s')} != {len(params)}"
            )

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._all)


class _Connection:
    def __init__(self):
        self.queries = []
        self.commits = 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class IcpRepositoryContractTests(unittest.TestCase):
    def _repository(self):
        connection = _Connection()
        repository = IcpRepository()
        repository._initialized = True
        return repository, connection

    def test_history_upsert_keeps_function_rbd_and_step_source(self):
        repository, connection = self._repository()
        item = {
            "case_id": "CASE-001",
            "fault_symptom": "显示异常",
            "related_function": "FUNC-01",
            "rbd_ref": "RBD-01",
            "document_links": {"rbd_refs": ["RBD-01"]},
            "source_ref": "history.docx#table=2;row=3",
        }
        steps = [{
            "step_no": 1,
            "content": "查询状态",
            "source_ref": "history.docx#seq=12",
        }]

        with patch("backend.icp.repository.postgres_client.conn", connection):
            row_id = repository.upsert_history_case("doc-1", item, steps)

        self.assertEqual(row_id, 101)
        history_query, history_params = connection.queries[0]
        self.assertIn("related_function", history_query)
        self.assertIn("rbd_ref", history_query)
        self.assertIn("FUNC-01", history_params)
        self.assertIn("RBD-01", history_params)
        step_query, step_params = connection.queries[-1]
        self.assertIn("source_ref", step_query)
        self.assertIn("history.docx#seq=12", step_params)

    def test_agent_facing_queries_exclude_non_ready_documents(self):
        repository, connection = self._repository()
        with patch("backend.icp.repository.postgres_client.conn", connection):
            repository.search_functions(["显示"])
            repository.search_components(["处理模块"])
            repository.search_interfaces(["视频流"])
            repository.search_history(["黑屏"])

        sql = "\n".join(query for query, _params in connection.queries)
        self.assertEqual(sql.count("JOIN icp_documents"), 4)
        self.assertEqual(sql.count("d.status IN ('indexed','kg_pending','kg_done')"), 4)
        history_sql, history_params = next(
            (query, params)
            for query, params in connection.queries
            if "FROM fault_history_cases h" in query
        )
        for column in (
            "h.occurred_at",
            "h.personnel",
            "h.mission_phase",
            "h.verification_result",
            "h.preventive_action",
        ):
            self.assertIn(column, history_sql)
        self.assertEqual(19, len(history_params))

    def test_function_locator_storage_hydration_uses_content_type_table_identity(self):
        class HydrationCursor:
            def __init__(self, connection):
                self.connection = connection
                self.description = []
                self.rows = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, query, params=()):
                sql = " ".join(query.split())
                self.connection.queries.append((sql, tuple(params or ())))
                if "FROM icp_functions" in sql:
                    self.description = [("id",), ("doc_id",), ("function_id",)]
                    self.rows = [(31, "doc-arch", "FUNC-330")]
                elif "FROM icp_components" in sql:
                    self.description = [("id",), ("doc_id",), ("component_id",)]
                    self.rows = [(12, "doc-arch", "SUB-IFC")]
                elif "FROM icp_interfaces" in sql:
                    self.description = [("id",), ("doc_id",), ("interface_id",)]
                    self.rows = [(9, "doc-arch", "IF-IFC-GPP")]
                else:
                    raise AssertionError(f"unexpected query: {sql}")

            def fetchall(self):
                return list(self.rows)

        class HydrationConnection:
            def __init__(self):
                self.queries = []

            def cursor(self):
                return HydrationCursor(self)

        repository = IcpRepository()
        connection = HydrationConnection()
        locators = [
            {"doc_id": "doc-arch", "pg_id_kwd": "31", "content_type": "function_record"},
            {"doc_id": "doc-arch", "pg_id_kwd": "12", "content_type": "function_component"},
            {"doc_id": "doc-arch", "pg_id_kwd": "9", "content_type": "function_interface"},
            {"doc_id": "doc-arch", "pg_id_kwd": "999", "content_type": "function_text"},
        ]

        with patch("backend.icp.repository.postgres_client.conn", connection):
            rows = repository.get_records_by_storage_locators(
                "function_locator", locators, limit=8
            )

        self.assertEqual(
            ["function_record", "function_component", "function_interface"],
            [row["content_type"] for row in rows],
        )
        sql = "\n".join(query for query, _params in connection.queries)
        self.assertIn("FROM icp_functions", sql)
        self.assertIn("FROM icp_components", sql)
        self.assertIn("FROM icp_interfaces", sql)
        self.assertEqual(sql.count("JOIN icp_documents"), 3)
        self.assertNotIn("function_text", sql)

    def test_function_context_expansion_follows_explicit_ids_and_is_bounded(self):
        class ExpansionCursor:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, query, params=()):
                sql = " ".join(query.split())
                self.connection.queries.append((sql, tuple(params or ())))
                if query.count("%s") != len(tuple(params or ())):
                    raise AssertionError("placeholder mismatch")

            def fetchall(self):
                return [
                    (
                        "function_component",
                        {
                            "id": 10,
                            "doc_id": "doc-flow",
                            "component_id": "SUB-DATA",
                            "component_type": "subsystem",
                            "name": "数据接口子系统",
                        },
                    ),
                    (
                        "function_component",
                        {
                            "id": 11,
                            "doc_id": "doc-flow",
                            "component_id": "MOD-CHANNEL-B",
                            "parent_component_id": "SUB-DATA",
                            "component_type": "module",
                            "name": "B通道模块",
                        },
                    ),
                    (
                        "function_interface",
                        {
                            "id": 20,
                            "doc_id": "doc-flow",
                            "interface_id": "IF-DATA-PROCESSOR",
                            "source_component_id": "SUB-DATA",
                            "target_component_id": "SUB-PROCESSOR",
                        },
                    ),
                    (
                        "function_record",
                        {
                            "id": 31,
                            "doc_id": "doc-flow",
                            "function_id": "FUNC-LEAF",
                            "parent_function_id": "FUNC-FAMILY",
                            "module_id": "MOD-PROTOCOL",
                        },
                    ),
                ]

        class ExpansionConnection:
            def __init__(self):
                self.queries = []

            def cursor(self):
                return ExpansionCursor(self)

        repository = IcpRepository()
        repository._initialized = True
        connection = ExpansionConnection()
        seed = {
            "doc_id": "doc-flow",
            "function_id": "FUNC-LEAF",
            "parent_function_id": "FUNC-FAMILY",
            "module_id": "MOD-PROTOCOL",
            "content_type": "function_record",
        }

        with patch("backend.icp.repository.postgres_client.conn", connection):
            rows = repository.expand_function_context([seed], limit=4)

        self.assertEqual(4, len(rows))
        self.assertEqual(
            ["function_component", "function_component", "function_interface", "function_record"],
            [row["content_type"] for row in rows],
        )
        sql, params = connection.queries[0]
        self.assertIn("status IN ('indexed','kg_pending','kg_done')", sql)
        self.assertIn("f.parent_function_id", sql)
        self.assertIn("i.source_component_id", sql)
        self.assertIn("i.target_component_id", sql)
        self.assertIn("FUNC-LEAF", params[1])
        self.assertIn("FUNC-FAMILY", params[3])
        self.assertIn("MOD-PROTOCOL", params[5])
        self.assertEqual(4, params[-1])

    def test_fmeca_collection_expansion_uses_exact_function_tokens_and_closed_metadata(self):
        class CollectionCursor:
            def __init__(self, connection):
                self.connection = connection
                self.description = [
                    ("id",), ("doc_id",), ("code",), ("product_flag",),
                    ("fault_mode",), ("fault_cause",),
                    ("matched_function_ids",), ("matched_component_ids",),
                    ("collection_function_ids",),
                    ("collection_component_ids",),
                    ("collection_seed_function_ids",),
                    ("collection_seed_component_ids",),
                    ("collection_seed_function_name",),
                    ("collection_seed_component_name",), ("collection_total",),
                    ("collection_candidate_document_count",),
                    ("collection_candidate_document_ids",),
                    ("collection_applicable_system",),
                    ("collection_applicable_configuration",),
                    ("collection_applicability_complete",),
                    ("collection_member_ids_unique",),
                    ("collection_unmapped_row_count",),
                ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, query, params=()):
                sql = " ".join(query.split())
                values = tuple(params or ())
                self.connection.queries.append((sql, values))
                if query.count("%s") != len(values):
                    raise AssertionError("placeholder mismatch")

            def fetchall(self):
                return [
                    (
                        1, "doc-fmeca", "FM-110-A", "ICU / FUNC-110",
                        "输入滤波性能退化", "滤波电容老化", ["FUNC-110"], [],
                        ["FUNC-110", "FUNC-111"], ["SUB-PWR", "MOD-PWR-ICU"],
                        [], ["MOD-PWR-ICU"], None, "输入调理模块", 2,
                        1, ["doc-fmeca"], "SYN-ICP-9000", "CONFIG-A", True, True, 0,
                    ),
                    (
                        2, "doc-fmeca", "FM-110-B", "FUNC-110；电源输入",
                        "输入过压保护误动作", "采样偏置", ["FUNC-110"], [],
                        ["FUNC-110", "FUNC-111"], ["SUB-PWR", "MOD-PWR-ICU"],
                        [], ["MOD-PWR-ICU"], None, "输入调理模块", 2,
                        1, ["doc-fmeca"], "SYN-ICP-9000", "CONFIG-A", True, True, 0,
                    ),
                ]

        class CollectionConnection:
            def __init__(self):
                self.queries = []

            def cursor(self):
                return CollectionCursor(self)

        repository = IcpRepository()
        repository._initialized = True
        connection = CollectionConnection()

        with patch("backend.icp.repository.postgres_client.conn", connection):
            rows = repository.expand_fmeca_fault_mode_collection(
                "doc-arch",
                component_ids=["mod-pwr-icu"],
            )

        self.assertEqual(["FM-110-A", "FM-110-B"], [row["code"] for row in rows])
        for row in rows:
            self.assertTrue(row["collection_closed"])
            self.assertEqual("MOD-PWR-ICU", row["collection_parent_id"])
            self.assertEqual("输入调理模块", row["collection_parent_name"])
            self.assertEqual("failure_mode", row["collection_member_kind"])
            self.assertEqual(row["code"], row["collection_member_id"])
            self.assertEqual(row["fault_mode"], row["collection_member_name"])
            self.assertEqual(row["fault_cause"], row["collection_member_description"])
            self.assertEqual(2, row["collection_member_count"])
            self.assertEqual(1, row["collection_candidate_document_count"])
            self.assertEqual(["doc-fmeca"], row["collection_candidate_document_ids"])
            self.assertTrue(row["collection_applicability_complete"])
            self.assertTrue(row["collection_member_ids_unique"])
            self.assertEqual(["FUNC-110", "FUNC-111"], row["collection_function_ids"])
            self.assertEqual(["SUB-PWR", "MOD-PWR-ICU"], row["collection_component_ids"])
            self.assertEqual(["FUNC-110"], row["collection_matched_function_ids"])
            self.assertEqual([], row["collection_matched_component_ids"])

        sql, params = connection.queries[0]
        self.assertIn("regexp_split_to_table", sql)
        self.assertIn("token.value=scope.entity_id", sql)
        self.assertIn("component_descendants", sql)
        self.assertIn("function_descendants", sql)
        self.assertNotIn("component_ancestors", sql)
        self.assertNotIn("function_ancestors", sql)
        self.assertIn("candidate.doc_type='fmeca'", sql)
        self.assertIn("candidate.metadata->>'applicable_system'", sql)
        self.assertIn("candidate.metadata->>'applicable_configuration'", sql)
        self.assertIn("COUNT(DISTINCT UPPER(BTRIM(code)))", sql)
        self.assertIn("NULLIF(BTRIM(child.parent_component_id), '') IS NULL", sql)
        self.assertIn("COUNT(DISTINCT candidate.edge_parent_id)", sql)
        self.assertNotIn("product_flag ILIKE", sql)
        self.assertEqual(
            ("doc-arch", [], ["MOD-PWR-ICU"], 64),
            params,
        )

    def test_fmeca_collection_never_marks_limited_rows_closed(self):
        class PartialCursor:
            def __init__(self, _connection):
                self.description = [
                    ("code",), ("fault_mode",), ("fault_cause",),
                    ("matched_function_ids",), ("matched_component_ids",),
                    ("collection_function_ids",),
                    ("collection_component_ids",),
                    ("collection_seed_function_ids",),
                    ("collection_seed_component_ids",),
                    ("collection_seed_function_name",),
                    ("collection_seed_component_name",), ("collection_total",),
                    ("collection_candidate_document_count",),
                    ("collection_candidate_document_ids",),
                    ("collection_applicable_system",),
                    ("collection_applicable_configuration",),
                    ("collection_applicability_complete",),
                    ("collection_member_ids_unique",),
                    ("collection_unmapped_row_count",),
                ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, _query, _params=()):
                return None

            def fetchall(self):
                return [
                    ("FM-1", "模式1", "原因1", ["FUNC-110"], [], ["FUNC-110"],
                     ["MOD-A"], ["FUNC-110"], [], "功能110", None, 3,
                     1, ["doc-fmeca"], "SYN-ICP-9000", "CONFIG-A", True, True, 0),
                    ("FM-2", "模式2", "原因2", ["FUNC-110"], [], ["FUNC-110"],
                     ["MOD-A"], ["FUNC-110"], [], "功能110", None, 3,
                     1, ["doc-fmeca"], "SYN-ICP-9000", "CONFIG-A", True, True, 0),
                ]

        class PartialConnection:
            def cursor(self):
                return PartialCursor(self)

        repository = IcpRepository()
        repository._initialized = True
        with patch("backend.icp.repository.postgres_client.conn", PartialConnection()):
            rows = repository.expand_fmeca_fault_mode_collection(
                "doc-arch", function_ids=["FUNC-110"], limit=2
            )

        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["collection_closed"] is False for row in rows))
        self.assertTrue(all(row["collection_member_count"] == 3 for row in rows))

    def test_fmeca_collection_requires_one_applicable_version_and_unique_member_ids(self):
        columns = [
            ("doc_id",), ("code",), ("fault_mode",), ("fault_cause",),
            ("matched_function_ids",), ("matched_component_ids",),
            ("collection_function_ids",),
            ("collection_component_ids",), ("collection_seed_function_ids",),
            ("collection_seed_component_ids",),
            ("collection_seed_function_name",),
            ("collection_seed_component_name",), ("collection_total",),
            ("collection_candidate_document_count",),
            ("collection_candidate_document_ids",),
            ("collection_applicable_system",),
            ("collection_applicable_configuration",),
            ("collection_applicability_complete",),
            ("collection_member_ids_unique",),
            ("collection_unmapped_row_count",),
        ]

        class SafetyCursor:
            def __init__(self, connection):
                self.connection = connection
                self.description = columns

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, _query, _params=()):
                return None

            def fetchall(self):
                return [self.connection.row]

        class SafetyConnection:
            def __init__(self, row):
                self.row = row

            def cursor(self):
                return SafetyCursor(self)

        base = (
            "doc-fmeca-v1", "FM-110", "输入故障", "采样偏置",
            ["FUNC-110"], [], ["FUNC-110"], ["MOD-A"], ["FUNC-110"], [],
            "功能110", None, 1,
        )
        unsafe_variants = (
            (
                "parallel applicable FMECA versions",
                base + (
                    2, ["doc-fmeca-v1", "doc-fmeca-v2"],
                    "SYN-ICP-9000", "CONFIG-A", False, False,
                    0,
                ),
            ),
            (
                "missing applicability metadata",
                base + (0, [], "", "", False, True, 0),
            ),
            (
                "duplicate business member IDs",
                base + (
                    1, ["doc-fmeca-v1"], "SYN-ICP-9000", "CONFIG-A", True, False, 0,
                ),
            ),
        )

        for label, row in unsafe_variants:
            with self.subTest(label=label):
                repository = IcpRepository()
                repository._initialized = True
                with patch(
                    "backend.icp.repository.postgres_client.conn",
                    SafetyConnection(row),
                ):
                    results = repository.expand_fmeca_fault_mode_collection(
                        "doc-arch", function_ids=["FUNC-110"]
                    )

                self.assertEqual(1, len(results))
                self.assertFalse(results[0]["collection_closed"])

    def test_snapshot_clear_is_scoped_and_table_name_is_closed(self):
        connection = _Connection()
        client = PostgresClient.__new__(PostgresClient)
        client.conn = connection

        deleted = client.clear_icp_authoritative_rows("doc-2", "fmeca")
        self.assertEqual(deleted, 1)
        self.assertIn("DELETE FROM fmea WHERE doc_id = %s", connection.queries[0][0])
        with self.assertRaises(ValueError):
            client.clear_icp_authoritative_rows("doc-2", "fmea; DROP TABLE fmea")

    def test_graph_result_patch_does_not_increment_parse_count(self):
        connection = _Connection()
        client = PostgresClient.__new__(PostgresClient)
        client.conn = connection

        client.patch_icp_document_result(
            "doc-graph",
            {
                "kg_pending": False,
                "graph_projection": {"status": "kg_done"},
            },
        )

        sql, params = connection.queries[0]
        self.assertIn("COALESCE(last_result, '{}'::jsonb) || %s::jsonb", sql)
        self.assertNotIn("parse_count", sql)
        self.assertEqual("doc-graph", params[1])
        self.assertEqual(1, connection.commits)

    def test_hmc_variant_and_fmeca_precision_reach_postgres_contract(self):
        connection = _Connection()
        client = PostgresClient.__new__(PostgresClient)
        client.conn = connection

        client.upsert_hmc(
            "doc-1",
            {
                "variant_id": "variant-source-row-2",
                "hmc_code": "00123456**0001",
                "lcn": "00123456",
                "hamc": "**",
                "hsc": "0001",
                "fault_desc": "显示异常",
                "repair_method": "检查显示链路",
                "source_ref": "hmc.docx#table=1;row=2",
            },
        )
        hmc_sql, hmc_params = connection.queries[0]
        self.assertIn("ON CONFLICT (doc_id, variant_id)", hmc_sql)
        self.assertEqual(hmc_params[1], "variant-source-row-2")

        client.upsert_fmea(
            "doc-1",
            {
                "code": "FM-SCI-1",
                "fault_mode": "漂移",
                "severity_raw": "Ⅱ类",
                "severity_normalized": "II",
                "probability_raw": "1.25E-12",
                "probability_value": "1.25E-12",
            },
        )
        fmea_sql, fmea_params = connection.queries[1]
        self.assertIn("severity_normalized", fmea_sql)
        self.assertEqual(fmea_params[12], "1.25E-12")


if __name__ == "__main__":
    unittest.main()
