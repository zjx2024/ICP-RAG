"""Protect the retained legacy PG facade without a live database connection."""
import unittest
from unittest.mock import MagicMock

from backend.storage.postgres_client import PostgresClient


class PostgresQueryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.client = object.__new__(PostgresClient)
        self.client.is_mock = False
        self.client.conn = MagicMock()
        self.cursor = self.client.conn.cursor.return_value.__enter__.return_value
        self.cursor.fetchone.return_value = (0,)
        self.cursor.fetchall.return_value = []

    def test_mock_flag_is_explicit_and_missing_connection_is_not_ready_mock(self):
        self.assertFalse(self.client.conn_is_mock())
        self.client.is_mock = True
        self.assertTrue(self.client.conn_is_mock())
        self.client.is_mock = False
        self.client.conn = None
        self.assertFalse(self.client.conn_is_mock())

    def test_legacy_mock_connection_compatibility_is_retained(self):
        legacy_mock_type = type("MockPostgres", (), {})
        self.client.conn = legacy_mock_type()
        self.assertTrue(self.client.conn_is_mock())

    def test_unknown_or_injected_table_is_rejected_before_cursor(self):
        for table in ("documents", "fault_hmc; DROP TABLE mfl", "fault_hmc --"):
            with self.subTest(table=table), self.assertRaises(ValueError):
                self.client._query_rows(table, None, 1, 10, {})
        self.client.conn.cursor.assert_not_called()

    def test_filter_identifiers_cannot_escape_the_registry(self):
        attacker_column = "hmc_code) OR 1=1 --"
        self.client._query_rows("fault_hmc", None, 1, 10, {attacker_column: "data"})
        for call in self.cursor.execute.call_args_list:
            self.assertNotIn(attacker_column, call.args[0])
            self.assertNotIn("data", call.args[1])

    def test_filter_values_and_document_identity_remain_bound_parameters(self):
        value = "' OR 1=1 --"
        document = "doc'; DELETE FROM mfl; --"
        self.client._query_rows("fault_hmc", document, 2, 7, {"hmc_code": value})
        calls = self.cursor.execute.call_args_list
        self.assertEqual(2, len(calls))
        for call in calls:
            sql, params = call.args
            self.assertNotIn(value, sql)
            self.assertNotIn(document, sql)
            self.assertEqual([document, value], params[:2])
        self.assertEqual([7, 7], calls[-1].args[1][-2:])

    def test_keyword_search_keeps_code_as_data_and_uses_only_registered_columns(self):
        keyword = "'); SELECT secret FROM models; --"
        self.client._query_rows("mfl", None, 1, 10, {"keyword": keyword})
        sql, params = self.cursor.execute.call_args_list[0].args
        self.assertNotIn(keyword, sql)
        self.assertEqual([keyword] * len(self.client._ROW_SEARCH_COLS["mfl"]), params)
        self.assertIn("api_name", sql)


if __name__ == "__main__":
    unittest.main()
