"""OpenSearch 3.x implementation of the unified search backend.

One versioned index stores BM25 text, filterable metadata, and the project's
1024-dimensional embedding.  A stable alias is the only name used by runtime
operations so an index migration can switch versions explicitly.
"""

from __future__ import annotations

import copy
import os
import re
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.storage.search_store import (
    SearchHit,
    SearchStoreConnectionError,
    SearchStoreSchemaError,
    VECTOR_DIMENSION,
    VECTOR_FIELD,
    validate_vector,
    weighted_rrf,
)


_INDEX_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _config_value(name: str, default: Any) -> Any:
    """Read new config attributes while remaining import-safe during migration."""

    try:
        from backend.config.config import config

        return getattr(config, name, os.getenv(name, default))
    except (ImportError, AttributeError):
        return os.getenv(name, default)


def _validate_field_name(field: Any) -> str:
    value = str(field)
    if not _FIELD_NAME_RE.fullmatch(value):
        raise ValueError(f"invalid filter/update field name: {value!r}")
    return value


def filters_to_query(filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Translate facade filters to exact OpenSearch filter DSL.

    Scalars become ``term`` queries, sequences become ``terms`` (IN), and
    booleans stay JSON booleans.  An empty IN list deliberately matches no
    documents.
    """

    if not filters:
        return {"match_all": {}}
    clauses: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []
    for raw_field, value in filters.items():
        field = _validate_field_name(raw_field)
        if isinstance(value, (list, tuple, set, frozenset)):
            values = list(value)
            if not values:
                return {"match_none": {}}
            clauses.append({"terms": {field: values}})
        elif value is None:
            must_not.append({"exists": {"field": field}})
        else:
            clauses.append({"term": {field: value}})
    bool_query: Dict[str, Any] = {}
    if clauses:
        bool_query["filter"] = clauses
    if must_not:
        bool_query["must_not"] = must_not
    return {"bool": bool_query} if bool_query else {"match_all": {}}


class OpenSearchClient:
    """Lazy, explicit OpenSearch backend.

    ``client`` is accepted for unit tests.  When omitted, importing
    ``opensearchpy`` and opening a network connection are both deferred until
    :meth:`ensure_ready`.
    """

    DEFAULT_ALIAS = "knowledge_graph_default"
    DEFAULT_VERSION = "v1"
    TEXT_FIELDS = [
        "content^4",
        "title^3",
        "section^2",
        "summary^2",
        "doc_name",
        "chapter",
        "report.title^3",
        "report.summary^2",
        "report.findings.summary",
        "entities",
    ]

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        scheme: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_certs: Optional[bool] = None,
        index_alias: Optional[str] = None,
        index_version: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        request_timeout: Optional[int] = None,
        refresh_policy: Any = "wait_for",
    ) -> None:
        self.host = str(host or _config_value("OPENSEARCH_HOST", "localhost"))
        self.port = int(port or _config_value("OPENSEARCH_PORT", 9200))
        self.scheme = str(
            scheme or _config_value("OPENSEARCH_SCHEME", "http")
        ).lower()
        self.username = (
            str(username)
            if username is not None
            else str(_config_value("OPENSEARCH_USERNAME", ""))
        )
        self.password = (
            str(password)
            if password is not None
            else str(_config_value("OPENSEARCH_PASSWORD", ""))
        )
        configured_verify = _config_value("OPENSEARCH_VERIFY_CERTS", False)
        self.verify_certs = (
            _as_bool(configured_verify) if verify_certs is None else bool(verify_certs)
        )
        self.index_alias = str(
            index_alias
            or _config_value("OPENSEARCH_INDEX_ALIAS", self.DEFAULT_ALIAS)
        ).lower()
        self.index_version = str(
            index_version
            or _config_value("OPENSEARCH_INDEX_VERSION", self.DEFAULT_VERSION)
        ).lower()
        self.embedding_dim = int(
            embedding_dim
            if embedding_dim is not None
            else _config_value("EMBEDDING_DIM", VECTOR_DIMENSION)
        )
        self.request_timeout = int(
            request_timeout
            if request_timeout is not None
            else _config_value("OPENSEARCH_REQUEST_TIMEOUT", 60)
        )
        self.index_name = f"{self.index_alias}_{self.index_version}"
        self.refresh_policy = refresh_policy
        self._client = client
        self._ready = False
        self._lock = threading.RLock()

        for label, value in (
            ("OpenSearch alias", self.index_alias),
            ("OpenSearch versioned index", self.index_name),
        ):
            if not _INDEX_NAME_RE.fullmatch(value):
                raise ValueError(f"invalid {label}: {value!r}")
        if self.embedding_dim != VECTOR_DIMENSION:
            raise SearchStoreSchemaError(
                f"ICP-RAG requires EMBEDDING_DIM={VECTOR_DIMENSION}, "
                f"configured {self.embedding_dim}"
            )
        if self.request_timeout <= 0:
            raise ValueError("OPENSEARCH_REQUEST_TIMEOUT must be positive")

    @staticmethod
    def index_definition() -> Dict[str, Any]:
        keyword_fields = (
            "project_id",
            "kb_id",
            "doc_id",
            "chunk_id",
            "source_id",
            "report_id",
            "community_id",
            "content_type",
            "knowledge_graph_kwd",
            "removed_kwd",
            "stale_kwd",
            "language",
            "status",
            "category",
            "chapter",
            "doc_type_kwd",
            "business_id_kwd",
            "pg_id_kwd",
            "hmc_code_kwd",
            "lcn_kwd",
            "hamc_kwd",
            "hsc_kwd",
            "severity_kwd",
            "product_kwd",
            "api_reportable_kwd",
            "platform_test_kwd",
            "platform_test_status_kwd",
            "parent_kwd",
            "node_type_kwd",
            "diagram_id_kwd",
            "projection_role_kwd",
        )
        properties: Dict[str, Any] = {
            field: {"type": "keyword", "ignore_above": 2048}
            for field in keyword_fields
        }
        for field in ("content", "title", "section", "summary", "stale_reason"):
            properties[field] = {
                "type": "text",
                "analyzer": "icp_cjk",
                "search_analyzer": "icp_cjk_search",
            }
        properties.update(
            {
                "doc_name": {
                    "type": "text",
                    "analyzer": "icp_cjk",
                    "search_analyzer": "icp_cjk_search",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}},
                },
                "entities": {"type": "keyword", "ignore_above": 2048},
                "doc_ids": {"type": "keyword", "ignore_above": 2048},
                "tags": {"type": "keyword", "ignore_above": 2048},
                "available_int": {"type": "integer"},
                "weight_int": {"type": "integer"},
                "page_number": {"type": "integer"},
                "table_count": {"type": "integer"},
                "image_count": {"type": "integer"},
                "rank_flt": {"type": "float"},
                "probability_flt": {"type": "float"},
                "created_at": {"type": "date", "ignore_malformed": True},
                "updated_at": {"type": "date", "ignore_malformed": True},
                "stale_at": {"type": "date", "ignore_malformed": True},
                "report": {
                    "type": "object",
                    "dynamic": True,
                    "properties": {
                        "title": {"type": "text", "analyzer": "icp_cjk"},
                        "summary": {"type": "text", "analyzer": "icp_cjk"},
                        "rating": {"type": "float"},
                        "rating_explanation": {
                            "type": "text",
                            "analyzer": "icp_cjk",
                        },
                        "findings": {
                            "type": "object",
                            "dynamic": True,
                            "properties": {
                                "summary": {
                                    "type": "text",
                                    "analyzer": "icp_cjk",
                                },
                                "explanation": {
                                    "type": "text",
                                    "analyzer": "icp_cjk",
                                },
                            },
                        },
                    },
                },
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        )
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "analysis": {
                    "filter": {
                        "icp_cjk_bigram": {
                            "type": "cjk_bigram",
                            "output_unigrams": True,
                        }
                    },
                    "analyzer": {
                        "icp_cjk": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["cjk_width", "lowercase", "icp_cjk_bigram"],
                        },
                        "icp_cjk_search": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["cjk_width", "lowercase", "icp_cjk_bigram"],
                        },
                    },
                },
            },
            "mappings": {
                "dynamic": True,
                "dynamic_templates": [
                    {
                        "keyword_suffix": {
                            "match": "*_kwd",
                            "mapping": {"type": "keyword", "ignore_above": 2048},
                        }
                    },
                    {
                        "id_suffix": {
                            "match": "*_id",
                            "mapping": {"type": "keyword", "ignore_above": 2048},
                        }
                    },
                    {
                        "dynamic_strings": {
                            "match_mapping_type": "string",
                            "mapping": {"type": "keyword", "ignore_above": 2048},
                        }
                    },
                ],
                "properties": properties,
            },
        }

    def _create_sdk_client(self) -> Any:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise SearchStoreConnectionError(
                "opensearch-py is required for the configured OpenSearch backend"
            ) from exc

        kwargs: Dict[str, Any] = {
            "hosts": [
                {"host": self.host, "port": self.port, "scheme": self.scheme}
            ],
            "use_ssl": self.scheme == "https",
            "verify_certs": self.verify_certs,
            "timeout": self.request_timeout,
            "max_retries": 5,
            "retry_on_timeout": True,
        }
        if self.username or self.password:
            kwargs["http_auth"] = (self.username, self.password)
        if not self.verify_certs:
            kwargs.update({"ssl_assert_hostname": False, "ssl_show_warn": False})
        return OpenSearch(**kwargs)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._create_sdk_client()
        return self._client

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                client = self.client
                if hasattr(client, "ping") and not client.ping():
                    raise SearchStoreConnectionError(
                        f"OpenSearch at {self.scheme}://{self.host}:{self.port} did not respond"
                    )
                if not client.indices.exists(index=self.index_name):
                    client.indices.create(
                        index=self.index_name, body=self.index_definition()
                    )
                self._validate_mapping()
                self._ensure_alias()
            except (SearchStoreConnectionError, SearchStoreSchemaError):
                raise
            except Exception as exc:
                raise SearchStoreConnectionError(
                    f"failed to initialize OpenSearch index {self.index_name}: {exc}"
                ) from exc
            self._ready = True

    def _validate_mapping(self) -> None:
        response = self.client.indices.get_mapping(index=self.index_name)
        index_mapping = response.get(self.index_name, response)
        mappings = index_mapping.get("mappings", index_mapping)
        vector_mapping = mappings.get("properties", {}).get(VECTOR_FIELD, {})
        method = vector_mapping.get("method", {})
        expected = {
            "type": "knn_vector",
            "dimension": VECTOR_DIMENSION,
            "name": "hnsw",
            "engine": "lucene",
            "space_type": "cosinesimil",
        }
        actual = {
            "type": vector_mapping.get("type"),
            "dimension": int(vector_mapping.get("dimension", 0) or 0),
            "name": method.get("name"),
            "engine": method.get("engine"),
            "space_type": method.get("space_type"),
        }
        if actual != expected:
            raise SearchStoreSchemaError(
                f"OpenSearch index {self.index_name} has incompatible vector mapping: "
                f"expected {expected}, got {actual}"
            )

    def _ensure_alias(self) -> None:
        indices = self.client.indices
        if not indices.exists_alias(name=self.index_alias):
            indices.put_alias(
                index=self.index_name,
                name=self.index_alias,
                body={"is_write_index": True},
            )
            return
        targets = set(indices.get_alias(name=self.index_alias).keys())
        if targets != {self.index_name}:
            raise SearchStoreSchemaError(
                f"alias {self.index_alias} points to {sorted(targets)}, not "
                f"{self.index_name}; migrate data and switch the alias explicitly"
            )

    def _ready_client(self) -> Any:
        self.ensure_ready()
        return self.client

    @staticmethod
    def _clean_document(source: Mapping[str, Any]) -> Dict[str, Any]:
        document = copy.deepcopy(dict(source))
        document.pop(VECTOR_FIELD, None)
        return document

    @classmethod
    def _hits(cls, response: Mapping[str, Any]) -> List[SearchHit]:
        return [
            SearchHit(
                id=str(hit.get("_id", "")),
                score=float(hit.get("_score", 0.0) or 0.0),
                document=cls._clean_document(hit.get("_source", {})),
            )
            for hit in response.get("hits", {}).get("hits", [])
        ]

    @staticmethod
    def _source_filter() -> Dict[str, Any]:
        return {"excludes": [VECTOR_FIELD]}

    @staticmethod
    def _is_not_found(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "meta", None), "status", None)
        return status == 404

    def upsert(
        self,
        id: str,
        document: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        record_id = str(id)
        if not record_id:
            raise ValueError("id must not be empty")
        payload = copy.deepcopy(dict(document))
        embedded_vector = payload.pop(VECTOR_FIELD, None)
        vector = vector if vector is not None else embedded_vector
        if vector is not None:
            payload[VECTOR_FIELD] = validate_vector(vector)
        self._ready_client().index(
            index=self.index_alias,
            id=record_id,
            body=payload,
            refresh=self.refresh_policy,
        )

    def delete_by_id(self, id: str) -> bool:
        try:
            response = self._ready_client().delete(
                index=self.index_alias,
                id=str(id),
                refresh=self.refresh_policy,
            )
            return response.get("result") == "deleted"
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise

    def delete_by_filter(self, filters: Optional[Mapping[str, Any]]) -> int:
        response = self._ready_client().delete_by_query(
            index=self.index_alias,
            body={"query": filters_to_query(filters)},
            conflicts="proceed",
            refresh=True,
        )
        return int(response.get("deleted", 0))

    def update_by_filter(
        self,
        filters: Optional[Mapping[str, Any]],
        updates: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> int:
        payload = copy.deepcopy(dict(updates))
        embedded_vector = payload.pop(VECTOR_FIELD, None)
        vector = vector if vector is not None else embedded_vector
        for field in payload:
            _validate_field_name(field)
        if vector is not None:
            payload[VECTOR_FIELD] = validate_vector(vector)
        if not payload:
            return 0
        response = self._ready_client().update_by_query(
            index=self.index_alias,
            body={
                "query": filters_to_query(filters),
                "script": {
                    "lang": "painless",
                    "source": (
                        "for (entry in params.updates.entrySet()) { "
                        "ctx._source[entry.getKey()] = entry.getValue(); }"
                    ),
                    "params": {"updates": payload},
                },
            },
            conflicts="proceed",
            refresh=True,
        )
        return int(response.get("updated", 0))

    def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SearchHit]:
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset non-negative")
        response = self._ready_client().search(
            index=self.index_alias,
            body={
                "query": filters_to_query(filters),
                "from": offset,
                "size": limit,
                "_source": self._source_filter(),
            },
        )
        return self._hits(response)

    def count(self, filters: Optional[Mapping[str, Any]] = None) -> int:
        response = self._ready_client().count(
            index=self.index_alias, body={"query": filters_to_query(filters)}
        )
        return int(response.get("count", 0))

    def keyword_search(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]:
        filter_query = filters_to_query(filters)
        keyword_query = str(query).strip()
        if len(keyword_query) > 220:
            # CJK analysis expands each character sequence across every text
            # field. Keep long model-authored queries bounded so a valid
            # semantic request cannot exceed OpenSearch's Boolean clause cap.
            keyword_query = keyword_query[:220]
        if keyword_query:
            fields = self.TEXT_FIELDS
            if len(keyword_query) > 140:
                fields = ["content^4", "title^3", "summary^2", "entities"]
            bool_query: Dict[str, Any] = {
                "must": [
                    {
                        "multi_match": {
                            "query": keyword_query,
                            "fields": fields,
                            "type": "best_fields",
                            "analyzer": "icp_cjk_search",
                        }
                    }
                ]
            }
            if filter_query != {"match_all": {}}:
                bool_query["filter"] = [filter_query]
            query_dsl: Dict[str, Any] = {"bool": bool_query}
        else:
            query_dsl = filter_query
        response = self._ready_client().search(
            index=self.index_alias,
            body={
                "query": query_dsl,
                "size": top_k,
                "_source": self._source_filter(),
            },
        )
        return self._hits(response)

    def vector_search(
        self,
        vector: Sequence[float],
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]:
        knn_options: Dict[str, Any] = {
            "vector": validate_vector(vector),
            "k": top_k,
        }
        filter_query = filters_to_query(filters)
        if filter_query != {"match_all": {}}:
            # OpenSearch 3.x efficient filtering for Lucene HNSW belongs inside
            # the k-NN field clause, so restrictive filters can still return k.
            knn_options["filter"] = filter_query
        response = self._ready_client().search(
            index=self.index_alias,
            body={
                "query": {"knn": {VECTOR_FIELD: knn_options}},
                "size": top_k,
                "_source": self._source_filter(),
            },
        )
        return self._hits(response)

    def hybrid_search(
        self,
        query: str,
        vector: Optional[Sequence[float]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
        *,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[SearchHit]:
        candidate_k = min(max(top_k * 4, top_k), 10_000)
        rankings = []
        if str(query).strip():
            rankings.append(
                (self.keyword_search(query, filters, candidate_k), keyword_weight)
            )
        if vector is not None:
            rankings.append(
                (self.vector_search(vector, filters, candidate_k), vector_weight)
            )
        return weighted_rrf(rankings, top_k, rrf_k) if rankings else []

    def get_many(self, ids: Sequence[str]) -> List[SearchHit]:
        ordered_ids = [str(record_id) for record_id in ids]
        if not ordered_ids:
            return []
        response = self._ready_client().mget(
            index=self.index_alias,
            body={
                "docs": [
                    {
                        "_id": record_id,
                        "_source": {"excludes": [VECTOR_FIELD]},
                    }
                    for record_id in ordered_ids
                ]
            },
        )
        hits_by_id = {
            str(item.get("_id")): SearchHit(
                str(item.get("_id")),
                1.0,
                self._clean_document(item.get("_source", {})),
            )
            for item in response.get("docs", [])
            if item.get("found")
        }
        return [hits_by_id[record_id] for record_id in ordered_ids if record_id in hits_by_id]

    def stats(self) -> Dict[str, Any]:
        response = self._ready_client().indices.stats(index=self.index_name)
        index_stats = response.get("indices", {}).get(self.index_name, {})
        primaries = index_stats.get("primaries", response.get("_all", {}).get("primaries", {}))
        return {
            "backend": "opensearch",
            "ready": self._ready,
            "index_alias": self.index_alias,
            "index_name": self.index_name,
            "index_version": self.index_version,
            "documents": int(primaries.get("docs", {}).get("count", 0)),
            "size_bytes": int(primaries.get("store", {}).get("size_in_bytes", 0)),
            "vector_dimension": VECTOR_DIMENSION,
        }

    def search_community_reports(
        self,
        kb_id: str,
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        entity_names: Optional[Sequence[str]] = None,
        top_k: int = 100,
    ) -> List[SearchHit]:
        filters: List[Dict[str, Any]] = [
            {"term": {"kb_id": kb_id}},
            {"term": {"knowledge_graph_kwd": "community_report"}},
        ]
        if doc_id is not None:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"doc_id": doc_id}},
                            {"term": {"doc_ids": doc_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        should: List[Dict[str, Any]] = []
        search_text = " ".join(
            part
            for part in [query or "", " ".join(entity_names or ())]
            if part.strip()
        )
        if search_text:
            should.append(
                {
                    "multi_match": {
                        "query": search_text,
                        "fields": [
                            "report.title^3",
                            "report.summary^2",
                            "report.findings.summary",
                            "entities",
                        ],
                        "analyzer": "icp_cjk_search",
                    }
                }
            )
        bool_query: Dict[str, Any] = {
            "filter": filters,
            "must_not": [{"term": {"stale_kwd": "topology"}}],
        }
        if should:
            bool_query.update({"should": should, "minimum_should_match": 0})
        response = self._ready_client().search(
            index=self.index_alias,
            body={
                "query": {"bool": bool_query},
                "size": top_k,
                "_source": self._source_filter(),
            },
        )
        results: List[SearchHit] = []
        for hit in self._hits(response):
            score = hit.score * 0.7 if hit.document.get("stale_kwd") == "content" else hit.score
            results.append(SearchHit(hit.id, score, hit.document))
        return results


__all__ = ["OpenSearchClient", "filters_to_query"]
