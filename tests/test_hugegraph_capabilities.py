import asyncio
import json
import unittest
from unittest.mock import patch

from backend.storage.hugegraph_client import HugeGraphClient, HugeGraphUnavailable


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, *, gremlin=None, vertices=None, edges=None):
        self.gremlin = gremlin or _Response(
            payload={
                "status": {"code": 200},
                "result": {"data": [0]},
            }
        )
        self.vertices = vertices or _Response(payload={"vertices": []})
        self.edges = edges or _Response(payload={"edges": []})
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None, timeout=None):
        self.get_calls.append((url, params, timeout))
        return self.vertices if url.endswith("/vertices") else self.edges

    def post(self, url, json=None, timeout=None):
        self.post_calls.append((url, json, timeout))
        return self.gremlin


def _client(session, *, use_mock=False):
    client = object.__new__(HugeGraphClient)
    client._use_mock = use_mock
    client.base_url = "http://hugegraph:8080"
    client.graph_name = "hugegraph"
    client.graphspace = "DEFAULT"
    client.session = session
    return client


class HugeGraphProtocolTests(unittest.TestCase):
    def test_gremlin_uses_graphspace_qualified_aliases_and_status_response(self):
        session = _Session()
        client = _client(session)

        self.assertEqual([0], client.execute_gremlin("g.V().limit(0).count()"))

        url, payload, timeout = session.post_calls[0]
        self.assertEqual("http://hugegraph:8080/gremlin", url)
        self.assertEqual(
            {
                "graph": "DEFAULT-hugegraph",
                "g": "__g_DEFAULT-hugegraph",
            },
            payload["aliases"],
        )
        self.assertNotIn("session", payload)
        self.assertEqual(30, timeout)

    def test_gremlin_graphspace_is_configurable_and_empty_keeps_legacy_alias(self):
        client = _client(_Session())
        client.graphspace = "MISSION-A"
        self.assertEqual(
            {
                "graph": "MISSION-A-hugegraph",
                "g": "__g_MISSION-A-hugegraph",
            },
            client._gremlin_aliases(),
        )

        client.graphspace = ""
        self.assertEqual(
            {"graph": "hugegraph", "g": "__g_hugegraph"},
            client._gremlin_aliases(),
        )

    def test_gremlin_rejects_unsuccessful_protocol_status(self):
        session = _Session(
            gremlin=_Response(
                payload={"status": {"code": 500}, "result": {"data": []}}
            )
        )
        client = _client(session)

        with self.assertRaises(HugeGraphUnavailable):
            client.execute_gremlin("g.V().count()")

    def test_projection_reads_are_required_but_gremlin_is_optional(self):
        session = _Session(
            gremlin=_Response(
                status_code=400,
                payload={"message": "missing global binding"},
                text="missing global binding",
            )
        )
        status = _client(session).capability_status(timeout=1.25)

        self.assertTrue(status["ok"])
        self.assertEqual(
            {"ok": True, "required": True}, status["rest_projection_read"]
        )
        self.assertFalse(status["gremlin_query"]["ok"])
        self.assertFalse(status["gremlin_query"]["required"])
        self.assertEqual(2, len(session.get_calls))
        self.assertTrue(all(call[2] == 1.25 for call in session.get_calls))

    def test_invalid_edge_collection_fails_required_capability(self):
        session = _Session(edges=_Response(payload={"not_edges": []}))
        status = _client(session).capability_status()

        self.assertFalse(status["ok"])
        self.assertFalse(status["rest_projection_read"]["ok"])
        self.assertIn("edges", status["rest_projection_read"]["error"])

    def test_in_memory_fallback_never_reports_real_traversal_ready(self):
        status = _client(_Session(), use_mock=True).capability_status()

        self.assertFalse(status["ok"])
        self.assertTrue(status["rest_projection_read"]["required"])
        self.assertFalse(status["gremlin_query"]["required"])


class BoundedShortestPathTests(unittest.TestCase):
    def test_projection_reader_follows_hugegraph_page_cursor(self):
        class CursorSession(_Session):
            def get(self, url, params=None, timeout=None):
                self.get_calls.append((url, params, timeout))
                token = (params or {}).get("page")
                if token == "":
                    return _Response(
                        payload={
                            "vertices": [{"id": "v-1"}, {"id": "v-2"}],
                            "page": "next-page",
                        }
                    )
                return _Response(
                    payload={"vertices": [{"id": "v-3"}], "page": None}
                )

        session = CursorSession()
        client = _client(session)

        rows = client._read_projection_collection_bounded(
            "/vertices", "vertices", max_items=10
        )

        self.assertEqual(["v-1", "v-2", "v-3"], [row["id"] for row in rows])
        self.assertEqual("", session.get_calls[0][1]["page"])
        self.assertEqual("next-page", session.get_calls[1][1]["page"])

    def test_user_names_are_compared_as_literals_and_never_enter_gremlin(self):
        injected_name = "source'); g.V().drop(); //"
        session = _Session(
            vertices=_Response(
                payload={
                    "vertices": [
                        {
                            "id": "v-source",
                            "label": "Entity",
                            "properties": {"kb_id": "kb-1", "name": injected_name},
                        },
                        {
                            "id": "v-target",
                            "label": "Entity",
                            "properties": {"kb_id": "kb-1", "name": "target"},
                        },
                    ]
                }
            ),
            edges=_Response(
                payload={
                    "edges": [
                        {
                            "id": "e-1",
                            "label": "related_to",
                            "outV": "v-source",
                            "inV": "v-target",
                            "properties": {"kb_id": "kb-1"},
                        }
                    ]
                }
            ),
        )
        client = _client(session)

        paths = client.get_shortest_path("kb-1", injected_name, "target")

        self.assertEqual(1, len(paths))
        self.assertEqual([injected_name, "target"], [item["name"] for item in paths[0]])
        self.assertEqual("related_to", paths[0][1]["relation"])
        self.assertEqual([], session.post_calls)
        self.assertEqual(2, len(session.get_calls))

    def test_projection_size_limit_fails_closed_before_traversal(self):
        session = _Session(
            vertices=_Response(
                payload={
                    "vertices": [
                        {
                            "id": "v-1",
                            "properties": {"kb_id": "kb-1", "name": "one"},
                        },
                        {
                            "id": "v-2",
                            "properties": {"kb_id": "kb-1", "name": "two"},
                        },
                    ]
                }
            )
        )
        client = _client(session)
        with patch(
            "backend.storage.hugegraph_client._SHORTEST_PATH_MAX_VERTICES", 1
        ):
            with self.assertRaises(HugeGraphUnavailable):
                client.get_shortest_path("kb-1", "one", "two")

        self.assertEqual([], session.post_calls)


class ReadinessContractTests(unittest.TestCase):
    def test_optional_gremlin_degradation_does_not_change_ready_contract(self):
        # Import lazily so protocol tests remain independent from application
        # startup and its configured external clients.
        from backend import main

        checks = {
            "postgresql": {"ok": True},
            "opensearch": {"ok": True},
            "artifact_store": {"ok": True},
            "hugegraph": {
                "ok": True,
                "rest_projection_read": {"ok": True, "required": True},
                "gremlin_query": {
                    "ok": False,
                    "required": False,
                    "error": "missing global binding",
                },
            },
        }
        with patch.object(main, "_dependency_status", return_value=checks):
            response = asyncio.run(main.readiness())

        self.assertEqual(200, response.status_code)
        payload = json.loads(response.body)
        self.assertEqual("ready", payload["status"])
        self.assertEqual(checks["hugegraph"], payload["checks"]["hugegraph"])


if __name__ == "__main__":
    unittest.main()
