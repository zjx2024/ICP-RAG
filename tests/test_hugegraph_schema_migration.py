import unittest
from unittest.mock import patch

from backend.storage.hugegraph_client import (
    DYNAMIC_EDGE_NULLABLE_KEYS,
    DYNAMIC_EDGE_PROPERTIES,
    ENTITY_VERTEX_NULLABLE_KEYS,
    ENTITY_VERTEX_PROPERTIES,
    HugeGraphClient,
    PROPERTY_KEY_DEFINITIONS,
)


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, get_response):
        self.get_response = get_response
        self.posts = []
        self.puts = []
        self.deletes = []

    def get(self, url):
        return self.get_response(url)

    def post(self, url, json):
        self.posts.append((url, json))
        return _Response(201)

    def put(self, url, json):
        self.puts.append((url, json))
        return _Response(200)

    def delete(self, url):
        self.deletes.append(url)
        return _Response(204)


def _client(session):
    client = object.__new__(HugeGraphClient)
    client.session = session
    client._get_schema_url = lambda path: path
    return client


class HugeGraphSchemaMigrationTests(unittest.TestCase):
    def test_missing_structured_identity_property_keys_are_created_idempotently(self):
        missing = {"canonical_entity_id", "identity_scope"}
        session = _Session(
            lambda url: _Response(
                404 if url.rsplit("/", 1)[-1] in missing else 200
            )
        )
        client = _client(session)

        with patch("backend.storage.hugegraph_client.time.sleep"):
            client._create_property_keys()

        created = {payload["name"]: payload for _url, payload in session.posts}
        self.assertEqual(missing, set(created))
        self.assertEqual(
            {"name": "identity_scope", "data_type": "TEXT", "cardinality": "SINGLE"},
            created["identity_scope"],
        )
        self.assertEqual(
            {"name": "canonical_entity_id", "data_type": "TEXT", "cardinality": "SINGLE"},
            created["canonical_entity_id"],
        )

        defined = {name for name, _type, _cardinality in PROPERTY_KEY_DEFINITIONS}
        self.assertTrue(set(ENTITY_VERTEX_PROPERTIES).issubset(defined))
        self.assertTrue(set(DYNAMIC_EDGE_PROPERTIES).issubset(defined))

    def test_existing_entity_label_appends_only_new_nullable_fields(self):
        old_properties = [
            value
            for value in ENTITY_VERTEX_PROPERTIES
            if value not in {"canonical_entity_id", "identity_scope"}
        ]
        session = _Session(
            lambda _url: _Response(200, {"properties": old_properties})
        )
        client = _client(session)

        client._create_vertex_label_if_not_exists(
            "Entity",
            list(ENTITY_VERTEX_PROPERTIES),
            list(ENTITY_VERTEX_NULLABLE_KEYS),
        )

        self.assertEqual([], session.posts)
        self.assertEqual([], session.deletes)
        self.assertEqual(1, len(session.puts))
        url, payload = session.puts[0]
        self.assertEqual("/schema/vertexlabels/Entity?action=append", url)
        self.assertEqual(
            {"canonical_entity_id", "identity_scope"}, set(payload["properties"])
        )
        self.assertEqual(
            {"canonical_entity_id", "identity_scope"},
            set(payload["nullable_keys"]),
        )

    def test_incomplete_old_edge_label_is_never_deleted_automatically(self):
        session = _Session(
            lambda _url: _Response(200, {"properties": [], "source_label": None})
        )
        client = _client(session)

        migrated = client._update_edge_label_if_needed(
            "contains_component",
            "Entity",
            "Entity",
            list(DYNAMIC_EDGE_PROPERTIES),
            list(DYNAMIC_EDGE_NULLABLE_KEYS),
        )

        self.assertFalse(migrated)
        self.assertEqual([], session.deletes)
        self.assertEqual([], session.posts)
        self.assertEqual([], session.puts)

    def test_existing_dynamic_edge_label_appends_missing_properties_only(self):
        old_properties = [
            value
            for value in DYNAMIC_EDGE_PROPERTIES
            if value not in {
                "description_parts",
                "keywords_parts",
                "weight_parts",
            }
        ]
        session = _Session(
            lambda _url: _Response(
                200,
                {
                    "properties": old_properties,
                    "source_label": "Entity",
                    "target_label": "Entity",
                },
            )
        )
        client = _client(session)

        migrated = client.create_edge_label_dynamic("contains_component")

        self.assertTrue(migrated)
        self.assertEqual([], session.posts)
        self.assertEqual([], session.deletes)
        self.assertEqual(1, len(session.puts))
        url, payload = session.puts[0]
        self.assertEqual(
            "/schema/edgelabels/contains_component?action=append", url
        )
        expected = {"description_parts", "keywords_parts", "weight_parts"}
        self.assertEqual(expected, set(payload["properties"]))
        self.assertEqual(expected, set(payload["nullable_keys"]))


if __name__ == "__main__":
    unittest.main()
