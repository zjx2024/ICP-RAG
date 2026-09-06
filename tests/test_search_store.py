import copy
import unittest

from backend.storage.opensearch_client import OpenSearchClient, filters_to_query
from backend.storage.search_store import (
    InMemorySearchBackend,
    SearchHit,
    SearchStore,
    SearchStoreConnectionError,
    SearchStoreSchemaError,
    VECTOR_DIMENSION,
    VECTOR_FIELD,
    weighted_rrf,
)


def vector_at(position, value=1.0):
    vector = [0.0] * VECTOR_DIMENSION
    vector[position] = value
    return vector


class SearchStoreTests(unittest.TestCase):
    def setUp(self):
        self.backend = InMemorySearchBackend()
        self.store = SearchStore(self.backend)

    def test_facade_is_lazy_and_initializes_backend_once(self):
        created = []

        def factory():
            created.append(True)
            return InMemorySearchBackend()

        store = SearchStore(backend_factory=factory)
        self.assertEqual(created, [])

        self.assertEqual(store.count(), 0)
        self.assertEqual(store.stats()["backend"], "memory")

        self.assertEqual(created, [True])
        self.assertEqual(store.backend.ensure_ready_calls, 1)

    def test_connection_failure_is_not_replaced_by_memory_fallback(self):
        failure = SearchStoreConnectionError("OpenSearch unavailable")
        backend = InMemorySearchBackend(fail_on_ready=failure)
        store = SearchStore(backend)

        with self.assertRaisesRegex(SearchStoreConnectionError, "unavailable"):
            store.count()

        self.assertFalse(backend.ready)
        self.assertEqual(backend.ensure_ready_calls, 1)

    def test_upsert_extracts_legacy_vector_and_never_returns_it_as_document_data(self):
        source = {
            "content": "HMC 液压泵故障",
            "kb_id": "kb-a",
            VECTOR_FIELD: vector_at(0),
        }

        self.store.upsert(id="chunk-1", document=source)

        self.assertIn(VECTOR_FIELD, source)
        hit = self.store.get_many(ids=["chunk-1"])[0]
        self.assertEqual(hit.id, "chunk-1")
        self.assertNotIn(VECTOR_FIELD, hit.document)
        self.assertEqual(self.store.stats()["vectors"], 1)

    def test_vector_dimension_and_values_are_validated(self):
        with self.assertRaisesRegex(ValueError, "dimension must be 1024"):
            self.store.upsert("bad", {"content": "bad"}, [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "not finite"):
            vector = vector_at(0)
            vector[5] = float("nan")
            self.store.vector_search(vector)
        with self.assertRaisesRegex(ValueError, "not numeric"):
            vector = vector_at(0)
            vector[5] = True
            self.store.vector_search(vector)

    def test_filters_support_scalar_in_and_boolean_without_string_coercion(self):
        documents = {
            "a": {
                "kb_id": "kb-a",
                "doc_type_kwd": "hmc",
                "api_reportable": True,
                "tags": ["hydraulic", "pump"],
            },
            "b": {
                "kb_id": "kb-a",
                "doc_type_kwd": "fmeca",
                "api_reportable": False,
                "tags": ["electrical"],
            },
            "c": {
                "kb_id": "kb-b",
                "doc_type_kwd": "fta",
                "api_reportable": True,
                "tags": ["pump"],
            },
        }
        for record_id, document in documents.items():
            self.store.upsert(record_id, document)

        hits = self.store.query(
            {
                "kb_id": "kb-a",
                "doc_type_kwd": ["hmc", "fta"],
                "api_reportable": True,
                "tags": "pump",
            }
        )
        self.assertEqual([hit.id for hit in hits], ["a"])
        self.assertEqual(self.store.count({"doc_type_kwd": []}), 0)

    def test_crud_query_count_update_delete_and_ordered_get_many(self):
        self.store.upsert("b", {"kb_id": "kb", "status": "new"}, vector_at(1))
        self.store.upsert("a", {"kb_id": "kb", "status": "new"}, vector_at(0))
        self.store.upsert("c", {"kb_id": "other", "status": "new"})

        self.assertEqual([hit.id for hit in self.store.query({"kb_id": "kb"})], ["a", "b"])
        self.assertEqual(self.store.count({"status": "new"}), 3)
        self.assertEqual(
            self.store.update_by_filter({"kb_id": "kb"}, {"status": "parsed"}),
            2,
        )
        self.assertEqual(self.store.count({"status": "parsed"}), 2)
        self.assertEqual(
            [hit.id for hit in self.store.get_many(["b", "missing", "a"])],
            ["b", "a"],
        )
        self.assertTrue(self.store.delete_by_id("a"))
        self.assertFalse(self.store.delete_by_id("a"))
        self.assertEqual(self.store.delete_by_filter({"kb_id": ["kb", "other"]}), 2)
        self.assertEqual(self.store.count(), 0)

    def test_keyword_vector_and_hybrid_search_use_independent_rankings(self):
        self.store.upsert(
            "keyword-first",
            {"content": "泵 泵 泵 压力异常", "kb_id": "kb"},
            vector_at(1),
        )
        self.store.upsert(
            "vector-first",
            {"content": "泵 压力异常", "kb_id": "kb"},
            vector_at(0),
        )
        self.store.upsert(
            "filtered-out",
            {"content": "泵 泵 泵 泵", "kb_id": "other"},
            vector_at(0),
        )

        keyword = self.store.keyword_search("泵", {"kb_id": "kb"}, top_k=2)
        vector = self.store.vector_search(vector_at(0), {"kb_id": "kb"}, limit=2)
        hybrid = self.store.hybrid_search(
            "泵",
            query_vector=vector_at(0),
            filters={"kb_id": "kb"},
            top_k=2,
            keyword_weight=0.5,
            vector_weight=0.5,
        )

        self.assertEqual([hit.id for hit in keyword], ["keyword-first", "vector-first"])
        self.assertEqual(vector[0].id, "vector-first")
        self.assertEqual({hit.id for hit in hybrid}, {"keyword-first", "vector-first"})
        self.assertNotEqual(hybrid[0].score, keyword[0].score)

    def test_rrf_deduplicates_each_ranking_and_respects_weights(self):
        document = {"content": "evidence"}
        first = [SearchHit("a", 100, document), SearchHit("b", 10, document)]
        second = [SearchHit("b", 0.9, document), SearchHit("a", 0.8, document)]

        fused = weighted_rrf([(first, 0.2), (second, 0.8)], top_k=2, rrf_k=0)

        self.assertEqual([hit.id for hit in fused], ["b", "a"])
        self.assertAlmostEqual(fused[0].score, 0.8 / 1 + 0.2 / 2)

    def test_community_report_search_filters_doc_and_stale_topology(self):
        common = {"kb_id": "kb", "knowledge_graph_kwd": "community_report"}
        self.store.upsert(
            "active",
            {
                **common,
                "doc_ids": ["doc-1", "doc-2"],
                "entities": ["液压泵"],
                "report": {"summary": "压力故障"},
                "stale_kwd": "N",
            },
        )
        self.store.upsert(
            "topology-stale",
            {
                **common,
                "doc_ids": ["doc-1"],
                "entities": ["液压泵"],
                "report": {"summary": "压力故障"},
                "stale_kwd": "topology",
            },
        )
        self.store.upsert(
            "wrong-doc",
            {
                **common,
                "doc_ids": ["doc-9"],
                "entities": ["液压泵"],
                "report": {"summary": "压力故障"},
            },
        )

        hits = self.store.search_community_reports(
            "kb", doc_id="doc-1", query="压力", entity_names=["液压泵"]
        )

        self.assertEqual([hit.id for hit in hits], ["active"])


class FakeIndices:
    def __init__(self):
        self.definitions = {}
        self.aliases = {}

    def exists(self, index):
        return index in self.definitions

    def create(self, index, body):
        self.definitions[index] = copy.deepcopy(body)
        return {"acknowledged": True}

    def get_mapping(self, index):
        return {index: {"mappings": copy.deepcopy(self.definitions[index]["mappings"])}}

    def exists_alias(self, name):
        return name in self.aliases

    def put_alias(self, index, name, body=None):
        self.aliases[name] = {index}
        return {"acknowledged": True}

    def get_alias(self, name):
        return {index: {"aliases": {name: {}}} for index in self.aliases[name]}

    def stats(self, index):
        return {
            "indices": {
                index: {
                    "primaries": {
                        "docs": {"count": 3},
                        "store": {"size_in_bytes": 4096},
                    }
                }
            }
        }


class FakeOpenSearchSDK:
    def __init__(self):
        self.indices = FakeIndices()
        self.search_calls = []
        self.index_calls = []

    def ping(self):
        return True

    def index(self, **kwargs):
        self.index_calls.append(copy.deepcopy(kwargs))
        return {"result": "created"}

    def search(self, **kwargs):
        self.search_calls.append(copy.deepcopy(kwargs))
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "result-1",
                        "_score": 0.75,
                        "_source": {
                            "content": "证据",
                            VECTOR_FIELD: vector_at(0),
                        },
                    }
                ]
            }
        }


class OpenSearchClientContractTests(unittest.TestCase):
    def make_client(self, sdk=None):
        return OpenSearchClient(
            client=sdk or FakeOpenSearchSDK(),
            index_alias="knowledge_graph_default",
            index_version="v1",
            embedding_dim=VECTOR_DIMENSION,
        )

    def test_index_mapping_has_cjk_bm25_knn_and_dynamic_metadata(self):
        definition = OpenSearchClient.index_definition()
        settings = definition["settings"]
        mappings = definition["mappings"]
        vector_mapping = mappings["properties"][VECTOR_FIELD]

        self.assertIs(settings["index"]["knn"], True)
        self.assertEqual(mappings["properties"]["content"]["analyzer"], "icp_cjk")
        self.assertIn("cjk_bigram", str(settings["analysis"]))
        self.assertIs(mappings["dynamic"], True)
        self.assertTrue(mappings["dynamic_templates"])
        self.assertEqual(vector_mapping["type"], "knn_vector")
        self.assertEqual(vector_mapping["dimension"], 1024)
        self.assertEqual(vector_mapping["method"]["name"], "hnsw")
        self.assertEqual(vector_mapping["method"]["engine"], "lucene")
        self.assertEqual(vector_mapping["method"]["space_type"], "cosinesimil")

    def test_ensure_ready_creates_fixed_index_and_stable_alias(self):
        sdk = FakeOpenSearchSDK()
        client = self.make_client(sdk)

        client.ensure_ready()
        client.ensure_ready()

        self.assertIn("knowledge_graph_default_v1", sdk.indices.definitions)
        self.assertEqual(
            sdk.indices.aliases["knowledge_graph_default"],
            {"knowledge_graph_default_v1"},
        )

    def test_existing_alias_on_another_version_fails_explicitly(self):
        sdk = FakeOpenSearchSDK()
        sdk.indices.aliases["knowledge_graph_default"] = {"knowledge_graph_default_v0"}
        client = self.make_client(sdk)

        with self.assertRaisesRegex(SearchStoreSchemaError, "migrate data"):
            client.ensure_ready()

    def test_filter_translation_preserves_bool_and_uses_terms_for_in(self):
        query = filters_to_query(
            {"kb_id": "kb", "doc_type_kwd": ["hmc", "fta"], "enabled": False}
        )
        clauses = query["bool"]["filter"]

        self.assertIn({"term": {"kb_id": "kb"}}, clauses)
        self.assertIn({"terms": {"doc_type_kwd": ["hmc", "fta"]}}, clauses)
        self.assertIn({"term": {"enabled": False}}, clauses)
        self.assertEqual(filters_to_query({"kb_id": []}), {"match_none": {}})

    def test_upsert_maps_vector_and_search_results_do_not_expose_it(self):
        sdk = FakeOpenSearchSDK()
        client = self.make_client(sdk)
        source = {"content": "HMC", VECTOR_FIELD: vector_at(0)}

        client.upsert("chunk", source)
        hits = client.query()

        indexed = sdk.index_calls[0]["body"]
        self.assertEqual(len(indexed[VECTOR_FIELD]), VECTOR_DIMENSION)
        self.assertIn(VECTOR_FIELD, source)
        self.assertNotIn(VECTOR_FIELD, hits[0].document)

    def test_knn_filter_is_inside_opensearch_3_vector_clause(self):
        sdk = FakeOpenSearchSDK()
        client = self.make_client(sdk)

        client.vector_search(
            vector_at(0),
            filters={"kb_id": "kb", "api_reportable": True},
            top_k=7,
        )

        body = sdk.search_calls[-1]["body"]
        options = body["query"]["knn"][VECTOR_FIELD]
        self.assertEqual(options["k"], 7)
        self.assertEqual(options["filter"]["bool"]["filter"][-1], {"term": {"api_reportable": True}})
        self.assertEqual(body["_source"], {"excludes": [VECTOR_FIELD]})

    def test_long_keyword_query_uses_bounded_clause_safe_projection(self):
        sdk = FakeOpenSearchSDK()
        client = self.make_client(sdk)
        client.keyword_search("受热高负荷支路掉线" * 80, top_k=12)
        body = sdk.search_calls[-1]["body"]
        match = body["query"]["bool"]["must"][0]["multi_match"]
        self.assertLessEqual(len(match["query"]), 220)
        self.assertEqual(match["fields"], ["content^4", "title^3", "summary^2", "entities"])

    def test_configured_dimension_other_than_1024_is_rejected(self):
        with self.assertRaisesRegex(SearchStoreSchemaError, "requires EMBEDDING_DIM=1024"):
            OpenSearchClient(
                client=FakeOpenSearchSDK(),
                index_alias="knowledge_graph_default",
                index_version="v1",
                embedding_dim=768,
            )


if __name__ == "__main__":
    unittest.main()
