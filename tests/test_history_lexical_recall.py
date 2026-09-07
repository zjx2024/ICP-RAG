"""fault_history_query must use the shared modern SQL recall path.

Contracts only (fake connection): a spaced planner term reaches SQL in its
space-normalized form so stored unspaced CJK-mixed text matches; the query
carries the len² specificity ordering and bounded over-fetch instead of a
raw updated_at LIMIT; returned rows get their steps hydrated.
"""
import unittest
from unittest.mock import patch

from backend.agent.tools import icp_diagnosis as diagnosis_module


class _CapturingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        self.connection.calls.append((" ".join(query.split()), tuple(params or ())))

    def fetchall(self):
        return []


class _CapturingConnection:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return _CapturingCursor(self)

    def rollback(self):
        pass


class HistoryLexicalRecallTests(unittest.TestCase):
    def test_spaced_term_reaches_sql_in_space_normalized_form(self):
        # "A 通道" (planner phrasing) must be able to match stored "A通道"
        # (document phrasing): the normalized variant without spaces is one
        # of the SQL recall patterns.
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_history(["A 通道完全收不到"], 5)
        self.assertTrue(connection.calls)
        sql, params = connection.calls[0]
        patterns = [str(value) for value in params]
        normalized = [value for value in patterns if "a通道" in value]
        self.assertTrue(
            normalized,
            f"normalized space-free pattern missing from params: {patterns}",
        )
        self.assertIn("fault_history_cases", sql)

    def test_history_recall_uses_specificity_ordering_and_overfetch(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_history(["A通道接收故障"], 5)
        sql, _params = connection.calls[0]
        self.assertIn("ORDER BY (", sql)
        self.assertIn("t.updated_at DESC", sql)
        # Document-status boundary stays identical to the other direct tools.
        self.assertIn("JOIN icp_documents", sql)
        self.assertIn("d.status IN ('indexed','kg_pending','kg_done')", sql)
        for column in (
            "t.case_id", "t.fault_symptom", "t.root_cause",
            "t.final_solution", "t.verification_result",
        ):
            self.assertIn(column, sql)

    def test_returned_rows_have_steps_hydrated(self):
        rows = [{"id": 7, "case_id": "HIS-SYN-011", "fault_symptom": "A通道收不到"}]
        connection = _CapturingConnection()
        with patch.object(diagnosis_module, "_query_table", return_value=list(rows)), \
                patch.object(diagnosis_module.postgres_client, "conn", connection):
            result = diagnosis_module._query_history(["A通道"], 5)
        step_calls = [sql for sql, _params in connection.calls if "fault_history_steps" in sql]
        self.assertEqual(len(rows), len(step_calls))
        # steps key exists on every row (empty list with the fake cursor)
        self.assertTrue(all("steps" in row for row in result))

    def test_empty_recall_skips_step_hydration(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module, "_query_table", return_value=[]), \
                patch.object(diagnosis_module.postgres_client, "conn", connection):
            result = diagnosis_module._query_history(["无匹配词"], 5)
        self.assertEqual([], result)
        self.assertEqual([], connection.calls)


if __name__ == "__main__":
    unittest.main()
