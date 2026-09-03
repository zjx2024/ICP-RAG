import importlib
import unittest
from unittest.mock import Mock, patch

from backend.agent.db import AgentDB
session_router = importlib.import_module("backend.agent.api.session_router")


class _Cursor:
    def __init__(self):
        self.calls = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))


class _Connection:
    def __init__(self):
        self.cursor_instance = _Cursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


class SessionTitleFormattingTests(unittest.TestCase):
    def test_first_question_is_normalized_for_sidebar(self):
        title = session_router._make_session_title(
            "  ###   LCN 91810001 HSC 0001 \n\n 故障如何修复？  "
        )

        self.assertEqual(title, "LCN 91810001 HSC 0001 故障如何修复")

    def test_long_title_is_bounded_and_uses_ellipsis(self):
        title = session_router._make_session_title("故障现象" * 20)

        self.assertEqual(len(title), session_router.AUTO_TITLE_MAX_LENGTH)
        self.assertTrue(title.endswith("…"))


class SessionTitleEndpointTests(unittest.TestCase):
    def test_manual_title_is_never_regenerated(self):
        fake_db = Mock()
        session = {"id": "s1", "title": "用户命名", "title_origin": "manual"}
        fake_db.get_session.return_value = session

        with patch.object(session_router, "agent_db", fake_db):
            result = session_router.generate_title("s1")

        self.assertEqual(result, session)
        fake_db.list_messages.assert_not_called()
        fake_db.generate_session_title_once.assert_not_called()

    def test_already_generated_title_is_never_regenerated(self):
        fake_db = Mock()
        session = {"id": "s1", "title": "既有自动标题", "title_origin": "auto"}
        fake_db.get_session.return_value = session

        with patch.object(session_router, "agent_db", fake_db):
            result = session_router.generate_title("s1")

        self.assertEqual(result, session)
        fake_db.list_messages.assert_not_called()

    def test_default_title_is_generated_from_first_user_message_once(self):
        fake_db = Mock()
        fake_db.get_session.return_value = {
            "id": "s1",
            "title": "新诊断会话",
            "title_origin": "default",
        }
        fake_db.list_messages.return_value = [
            {"role": "assistant", "content": "欢迎"},
            {"role": "user", "content": "HMC 91810001 ** 0001 故障查询？"},
        ]
        generated = {
            "id": "s1",
            "title": "HMC 91810001 ** 0001 故障查询",
            "title_origin": "auto",
        }
        fake_db.generate_session_title_once.return_value = generated

        with patch.object(session_router, "agent_db", fake_db):
            result = session_router.generate_title("s1")

        self.assertEqual(result, generated)
        fake_db.generate_session_title_once.assert_called_once_with(
            "s1", "HMC 91810001 ** 0001 故障查询"
        )


class SessionTitlePersistenceTests(unittest.TestCase):
    def test_automatic_update_is_guarded_by_default_origin(self):
        connection = _Connection()
        database = AgentDB()
        current = {"id": "s1", "title": "用户在竞态中命名", "title_origin": "manual"}

        with patch("backend.agent.db.postgres_client.conn", connection), patch.object(
            database, "get_session", return_value=current
        ):
            result = database.generate_session_title_once("s1", "自动标题")

        sql, params = connection.cursor_instance.calls[0]
        self.assertIn("title_origin = 'default'", sql)
        self.assertEqual(params, ("自动标题", "s1"))
        self.assertEqual(connection.commits, 1)
        self.assertEqual(result, current)

    def test_manual_update_marks_title_origin(self):
        connection = _Connection()
        database = AgentDB()

        with patch("backend.agent.db.postgres_client.conn", connection), patch.object(
            database, "get_session", return_value={"id": "s1"}
        ):
            database.update_session("s1", title="手动标题")

        sql, params = connection.cursor_instance.calls[0]
        self.assertIn("title_origin = %s", sql)
        self.assertEqual(params, ["手动标题", "manual", "s1"])

    def test_legacy_placeholder_with_messages_remains_eligible_for_repair(self):
        source = __import__("inspect").getsource(AgentDB.init_tables)

        self.assertIn("WHEN title IN ('新会话', '新诊断会话')", source)
        self.assertNotIn("COALESCE(message_count", source)
        self.assertIn("ALTER COLUMN title SET DEFAULT '新诊断会话'", source)

    def test_first_user_message_triggers_server_side_title(self):
        connection = _Connection()
        database = AgentDB()

        with patch("backend.agent.db.postgres_client.conn", connection), patch.object(
            database, "touch_session"
        ), patch.object(database, "generate_session_title_once") as generate:
            database.add_message("s1", "user", "  HMC 0001 故障如何修复？ ")

        generate.assert_called_once_with("s1", "HMC 0001 故障如何修复")

    def test_assistant_message_does_not_trigger_title(self):
        connection = _Connection()
        database = AgentDB()

        with patch("backend.agent.db.postgres_client.conn", connection), patch.object(
            database, "touch_session"
        ), patch.object(database, "generate_session_title_once") as generate:
            database.add_message("s1", "assistant", "诊断结果")

        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
