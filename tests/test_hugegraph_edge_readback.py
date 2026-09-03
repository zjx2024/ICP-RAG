import unittest

from backend.storage.hugegraph_client import HugeGraphClient


class _Response:
    status_code = 200

    def __init__(self, items):
        self._items = items

    def json(self):
        return {"edges": self._items, "vertices": self._items}


class _Session:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        offset = int((params or {}).get("offset", 0))
        return _Response(self.pages.get(offset, []))


class _JsonResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _BatchSession:
    def __init__(self, item):
        self.item = item
        self.get_calls = []
        self.put_calls = []

    def get(self, url, params=None):
        self.get_calls.append((url, params))
        return _JsonResponse(200, self.item)

    def put(self, url, json=None):
        self.put_calls.append((url, json))
        return _JsonResponse(200, {})


class HugeGraphEdgeReadbackTests(unittest.TestCase):
    def _client(self, pages):
        client = object.__new__(HugeGraphClient)
        client._use_mock = False
        client.base_url = "http://hugegraph:8080"
        client.graph_name = "hugegraph"
        client.session = _Session(pages)
        return client

    def test_edge_readback_scans_without_property_index_and_filters_client_side(self):
        client = self._client(
            {
                0: [
                    {"id": "a", "label": "dynamic_a", "properties": {"kb_id": "wanted"}},
                    {"id": "b", "label": "dynamic_b", "properties": {"kb_id": "other"}},
                ],
                2: [],
            }
        )

        result = client.get_edges_by_kb("wanted", limit=2)

        self.assertEqual([edge["id"] for edge in result], ["a"])
        self.assertEqual([call[1]["offset"] for call in client.session.calls], [0, 2])
        self.assertTrue(
            all("properties" not in params for _, params in client.session.calls),
            "dynamic edge labels must not rely on a per-label kb_id index",
        )

    def test_detailed_readback_uses_the_same_index_independent_path(self):
        client = self._client(
            {0: [{"id": "a", "properties": {"kb_id": "wanted"}}]}
        )

        result = client.get_edges_by_kb_with_details("wanted")

        self.assertEqual([edge["id"] for edge in result], ["a"])
        self.assertNotIn("properties", client.session.calls[0][1])

    def test_detailed_vertex_readback_stops_after_last_partial_page(self):
        client = self._client(
            {
                0: [
                    {"id": "v1", "properties": {"kb_id": "wanted"}},
                    {"id": "v2", "properties": {"kb_id": "wanted"}},
                ],
                # HugeGraph may repeat this page for every later offset.  It
                # must be consumed once and recognized as the terminal page.
                2: [{"id": "v3", "properties": {"kb_id": "wanted"}}],
                4: [{"id": "v3", "properties": {"kb_id": "wanted"}}],
            }
        )

        original = client.get_vertices_by_kb
        client.get_vertices_by_kb = lambda kb_id, label=None: original(
            kb_id, label=label, limit=2
        )
        result = client.get_vertices_by_kb_with_details("wanted")

        self.assertEqual([vertex["id"] for vertex in result], ["v1", "v2", "v3"])
        self.assertEqual([call[1]["offset"] for call in client.session.calls], [0, 2])

    def test_vertex_exact_replace_uses_batch_override_for_list_properties(self):
        client = self._client({})
        client.session = _BatchSession({"id": "v1", "label": "Entity"})

        self.assertTrue(
            client.replace_vertex_properties(
                "v1", {"doc_id": ["d1", "d2"], "name": "node"}
            )
        )

        url, payload = client.session.put_calls[0]
        self.assertTrue(url.endswith("/vertices/batch"))
        self.assertEqual(payload["vertices"][0]["properties"]["doc_id"], ["d1", "d2"])
        self.assertEqual(payload["update_strategies"]["doc_id"], "OVERRIDE")

    def test_edge_exact_replace_encodes_server_id_and_uses_batch_override(self):
        edge_id = "Sv1>1>>Sv2"
        client = self._client({})
        client.session = _BatchSession(
            {
                "id": edge_id,
                "label": "relation",
                "outV": "v1",
                "inV": "v2",
                "outVLabel": "Entity",
                "inVLabel": "Entity",
            }
        )

        self.assertTrue(
            client.replace_edge_properties(edge_id, {"doc_id": ["d1"]})
        )

        self.assertIn("%3E", client.session.get_calls[0][0])
        url, payload = client.session.put_calls[0]
        self.assertTrue(url.endswith("/edges/batch"))
        self.assertEqual(payload["edges"][0]["id"], edge_id)
        self.assertEqual(payload["update_strategies"]["doc_id"], "OVERRIDE")


if __name__ == "__main__":
    unittest.main()
