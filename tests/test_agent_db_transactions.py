import json
import unittest
from unittest.mock import patch

from backend.agent.db import AgentDB
from backend.storage.postgres_client import PostgresClient


class _FakeCursor:
    def __init__(self, *, fail=False, executions=None):
        self.fail = fail
        self.description = []
        self.executions = executions if executions is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        if self.fail:
            raise RuntimeError("query failed")
        self.executions.append((_sql, _params))

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _FakeConnection:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.commits = 0
        self.rollbacks = 0
        self.executions = []

    def cursor(self):
        return _FakeCursor(fail=self.fail, executions=self.executions)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class AgentDBReadTransactionTests(unittest.TestCase):
    def test_list_sessions_commits_read_transaction(self):
        connection = _FakeConnection()
        with patch("backend.agent.db.postgres_client.conn", connection):
            result = AgentDB().list_sessions(agent_id="agent-1")

        self.assertEqual(result, [])
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)

    def test_list_sessions_rolls_back_failed_read(self):
        connection = _FakeConnection(fail=True)
        with patch("backend.agent.db.postgres_client.conn", connection):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                AgentDB().list_sessions(agent_id="agent-1")

        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)

    def test_assistant_message_persists_evidence_catalog_separately_from_citations(self):
        connection = _FakeConnection()
        citations = [{"id": 1, "evidence_id": "ev-used"}]
        evidence_catalog = [
            {
                "catalog_index": 1,
                "evidence_id": "ev-used",
                "status": "answer_used",
                "answer_citation_id": 1,
            },
            {
                "catalog_index": 2,
                "evidence_id": "ev-reviewed",
                "status": "approved",
                "answer_citation_id": None,
            },
        ]
        with patch("backend.agent.db.postgres_client.conn", connection):
            result = AgentDB().add_message(
                "session-1",
                "assistant",
                "结论 [^1]",
                citations=citations,
                evidence_catalog=evidence_catalog,
                thinking_events=[{"type": "tool_call", "evidence_ids": ["ev-used"]}],
            )

        insert = next(
            (sql, params)
            for sql, params in connection.executions
            if "INSERT INTO agent_messages" in sql
        )
        self.assertIn("evidence_catalog", insert[0])
        self.assertIn("delivered_answer_fields", insert[0])
        # 新增的 delivered_answer_fields 位于参数末尾；evidence_catalog 是倒数第二列
        self.assertEqual(evidence_catalog, json.loads(insert[1][-2]))
        self.assertIsNone(insert[1][-1])
        self.assertEqual(citations, result["citations"])
        self.assertEqual(evidence_catalog, result["evidence_catalog"])

    def test_init_tables_adds_evidence_catalog_column_idempotently(self):
        connection = _FakeConnection()
        with patch("backend.agent.db.postgres_client.conn", connection):
            database = AgentDB()
            database.init_tables()
            database.init_tables()

        migration_sql = [
            " ".join(sql.split())
            for sql, _params in connection.executions
            if "ADD COLUMN IF NOT EXISTS evidence_catalog" in sql
        ]
        self.assertEqual(1, len(migration_sql))

    def test_evidence_catalog_migration_failure_rolls_back_and_is_not_swallowed(self):
        class MigrationFailCursor(_FakeCursor):
            def execute(self, sql, params=None):
                if "ADD COLUMN IF NOT EXISTS evidence_catalog" in sql:
                    raise RuntimeError("evidence catalog migration failed")
                super().execute(sql, params)

        class MigrationFailConnection(_FakeConnection):
            def cursor(self):
                return MigrationFailCursor(executions=self.executions)

        connection = MigrationFailConnection()
        database = AgentDB()
        with patch("backend.agent.db.postgres_client.conn", connection):
            with self.assertRaisesRegex(RuntimeError, "evidence catalog migration failed"):
                database.init_tables()

        self.assertFalse(database._initialized)
        self.assertGreaterEqual(connection.rollbacks, 1)

    def test_list_messages_restores_evidence_catalog_for_history_ui(self):
        columns = [
            "id", "session_id", "role", "content", "tool_calls", "citations",
            "thinking_events", "evidence_catalog", "feedback", "created_at",
        ]
        catalog = [{"evidence_id": "ev-1", "status": "answer_used"}]

        class MessageCursor(_FakeCursor):
            def __init__(self):
                super().__init__()
                self.description = [(column,) for column in columns]

            def fetchall(self):
                return [(
                    "message-1", "session-1", "assistant", "结果 [^1]", None,
                    json.dumps([{"id": 1, "evidence_id": "ev-1"}]),
                    json.dumps([{"type": "tool_call", "evidence_ids": ["ev-1"]}]),
                    json.dumps(catalog), None, None,
                )]

        class MessageConnection(_FakeConnection):
            def cursor(self):
                return MessageCursor()

        connection = MessageConnection()
        with patch("backend.agent.db.postgres_client.conn", connection):
            messages = AgentDB().list_messages("session-1")

        self.assertEqual(catalog, messages[0]["evidence_catalog"])
        self.assertEqual("ev-1", messages[0]["thinking_events"][0]["evidence_ids"][0])

    def test_list_messages_selects_recent_limit_then_restores_chronological_order(self):
        columns = [
            "id", "session_id", "role", "content", "tool_calls", "citations",
            "thinking_events", "evidence_catalog", "feedback", "created_at",
        ]

        class RecentCursor(_FakeCursor):
            def __init__(self, executions):
                super().__init__(executions=executions)
                self.description = [(column,) for column in columns]

            def fetchall(self):
                # PostgreSQL returns the selected recent window in DESC order.
                return [
                    ("m3", "s1", "assistant", "最新", None, None, None, None, None, "2026-01-03"),
                    ("m2", "s1", "user", "较新", None, None, None, None, None, "2026-01-02"),
                ]

        class RecentConnection(_FakeConnection):
            def cursor(self):
                return RecentCursor(self.executions)

        connection = RecentConnection()
        with patch("backend.agent.db.postgres_client.conn", connection):
            messages = AgentDB().list_messages("s1", limit=2)

        query = next(sql for sql, _params in connection.executions if "FROM agent_messages" in sql)
        self.assertIn("ORDER BY created_at DESC, id DESC", " ".join(query.split()))
        self.assertEqual(["m2", "m3"], [item["id"] for item in messages])


class PostgresClientReadTransactionTests(unittest.TestCase):
    @staticmethod
    def _client(connection):
        client = object.__new__(PostgresClient)
        client.conn = connection
        return client

    def test_get_model_commits_empty_read_transaction(self):
        connection = _FakeConnection()

        result = self._client(connection).get_model("missing-model")

        self.assertIsNone(result)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)

    def test_get_model_rolls_back_failed_read(self):
        connection = _FakeConnection(fail=True)

        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self._client(connection).get_model("model-1")

        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
