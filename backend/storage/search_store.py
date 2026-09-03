"""Unified search storage facade.

The application talks to :class:`SearchStore`, not to a vendor SDK.  The
production backend is OpenSearch; ``InMemorySearchBackend`` exists only for
deterministic unit tests and local algorithm checks.  Importing this module is
side-effect free: the global ``search_store`` does not import an OpenSearch SDK
or open a socket until ``ensure_ready`` (or another operation) is called.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence


VECTOR_DIMENSION = 1024
VECTOR_FIELD = "content_vector"


class SearchStoreError(RuntimeError):
    """Base error for search storage failures."""


class SearchStoreConnectionError(SearchStoreError):
    """The configured search backend could not be reached or initialized."""


class SearchStoreSchemaError(SearchStoreError):
    """The configured index is incompatible with the expected schema."""


@dataclass(frozen=True)
class SearchHit:
    """Backend-independent search result."""

    id: str
    score: float
    document: Dict[str, Any]


class SearchBackend(Protocol):
    """Operations required by the application search facade."""

    def ensure_ready(self) -> None: ...

    def upsert(
        self,
        id: str,
        document: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> None: ...

    def delete_by_id(self, id: str) -> bool: ...

    def delete_by_filter(self, filters: Optional[Mapping[str, Any]]) -> int: ...

    def update_by_filter(
        self,
        filters: Optional[Mapping[str, Any]],
        updates: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> int: ...

    def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SearchHit]: ...

    def count(self, filters: Optional[Mapping[str, Any]] = None) -> int: ...

    def keyword_search(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]: ...

    def vector_search(
        self,
        vector: Sequence[float],
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]: ...

    def get_many(self, ids: Sequence[str]) -> List[SearchHit]: ...

    def stats(self) -> Dict[str, Any]: ...

    def search_community_reports(
        self,
        kb_id: str,
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        entity_names: Optional[Sequence[str]] = None,
        top_k: int = 100,
    ) -> List[SearchHit]: ...


def validate_vector(vector: Sequence[float]) -> List[float]:
    """Return a validated float vector with the project's fixed dimension."""

    if isinstance(vector, (str, bytes, bytearray)):
        raise ValueError("vector must be a numeric sequence")
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError("vector must be a numeric sequence") from exc
    if len(values) != VECTOR_DIMENSION:
        raise ValueError(
            f"vector dimension must be {VECTOR_DIMENSION}, got {len(values)}"
        )

    normalized: List[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"vector value at position {index} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"vector value at position {index} is not finite")
        normalized.append(number)
    return normalized


def _resolve_top_k(top_k: int, limit: Optional[int]) -> int:
    value = limit if limit is not None else top_k
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("top_k/limit must be a positive integer")
    return value


def _copy_hit(hit: SearchHit, *, score: Optional[float] = None) -> SearchHit:
    return SearchHit(
        id=str(hit.id),
        score=float(hit.score if score is None else score),
        document=copy.deepcopy(hit.document),
    )


def weighted_rrf(
    rankings: Sequence[tuple[Sequence[SearchHit], float]],
    top_k: int,
    rrf_k: int = 60,
) -> List[SearchHit]:
    """Fuse independently ranked lists using weighted reciprocal rank fusion."""

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scores: Dict[str, float] = {}
    documents: Dict[str, Dict[str, Any]] = {}
    best_rank: Dict[str, int] = {}
    for hits, weight in rankings:
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError("RRF weights must be finite and non-negative")
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            record_id = str(hit.id)
            if record_id in seen:
                continue
            seen.add(record_id)
            scores[record_id] = scores.get(record_id, 0.0) + float(weight) / (
                rrf_k + rank
            )
            documents.setdefault(record_id, copy.deepcopy(hit.document))
            best_rank[record_id] = min(best_rank.get(record_id, rank), rank)

    ordered_ids = sorted(
        scores,
        key=lambda record_id: (-scores[record_id], best_rank[record_id], record_id),
    )
    return [
        SearchHit(record_id, scores[record_id], documents[record_id])
        for record_id in ordered_ids[:top_k]
    ]


class SearchStore:
    """Lazy facade over the configured OpenSearch backend."""

    def __init__(
        self,
        backend: Optional[SearchBackend] = None,
        *,
        backend_factory: Optional[Callable[[], SearchBackend]] = None,
    ) -> None:
        if backend is not None and backend_factory is not None:
            raise ValueError("provide backend or backend_factory, not both")
        self._backend = backend
        self._backend_factory = backend_factory
        self._ready = False
        self._lock = threading.RLock()

    def _get_backend(self) -> SearchBackend:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    if self._backend_factory is not None:
                        self._backend = self._backend_factory()
                    else:
                        # Deliberately local: importing this module must not import
                        # opensearch-py or initiate a production connection.
                        from backend.storage.opensearch_client import OpenSearchClient

                        self._backend = OpenSearchClient()
        return self._backend

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            backend = self._get_backend()
            # A failure is intentionally propagated.  Falling back to an
            # in-memory index would make successful writes disappear on restart.
            backend.ensure_ready()
            self._ready = True

    @property
    def backend(self) -> SearchBackend:
        self.ensure_ready()
        return self._get_backend()

    def upsert(
        self,
        id: str,
        document: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        payload = copy.deepcopy(dict(document))
        embedded_vector = payload.pop(VECTOR_FIELD, None)
        if vector is None:
            vector = embedded_vector
        normalized_vector = validate_vector(vector) if vector is not None else None
        self.backend.upsert(str(id), payload, normalized_vector)

    def delete_by_id(self, id: str) -> bool:
        return bool(self.backend.delete_by_id(str(id)))

    def delete_by_filter(self, filters: Optional[Mapping[str, Any]]) -> int:
        return int(self.backend.delete_by_filter(filters))

    def update_by_filter(
        self,
        filters: Optional[Mapping[str, Any]],
        updates: Mapping[str, Any],
    ) -> int:
        payload = copy.deepcopy(dict(updates))
        vector = payload.pop(VECTOR_FIELD, None)
        normalized_vector = validate_vector(vector) if vector is not None else None
        return int(self.backend.update_by_filter(filters, payload, normalized_vector))

    def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SearchHit]:
        return self.backend.query(filters, limit, offset)

    def count(self, filters: Optional[Mapping[str, Any]] = None) -> int:
        return int(self.backend.count(filters))

    def keyword_search(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
        *,
        limit: Optional[int] = None,
    ) -> List[SearchHit]:
        resolved = _resolve_top_k(top_k, limit)
        return self.backend.keyword_search(str(query), filters, resolved)

    def vector_search(
        self,
        vector: Sequence[float],
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
        *,
        limit: Optional[int] = None,
    ) -> List[SearchHit]:
        resolved = _resolve_top_k(top_k, limit)
        return self.backend.vector_search(validate_vector(vector), filters, resolved)

    def hybrid_search(
        self,
        query: str,
        vector: Optional[Sequence[float]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
        *,
        limit: Optional[int] = None,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
        query_vector: Optional[Sequence[float]] = None,
    ) -> List[SearchHit]:
        """Fuse independent BM25 and k-NN rankings.

        ``query_vector`` is accepted as a compatibility alias during migration
        from the old Elasticsearch client.
        """

        if vector is not None and query_vector is not None:
            raise ValueError("provide vector or query_vector, not both")
        vector = vector if vector is not None else query_vector
        resolved = _resolve_top_k(top_k, limit)
        candidate_k = min(max(resolved * 4, resolved), 10_000)
        rankings: List[tuple[Sequence[SearchHit], float]] = []
        if str(query).strip():
            rankings.append(
                (
                    self.backend.keyword_search(str(query), filters, candidate_k),
                    float(keyword_weight),
                )
            )
        if vector is not None:
            rankings.append(
                (
                    self.backend.vector_search(
                        validate_vector(vector), filters, candidate_k
                    ),
                    float(vector_weight),
                )
            )
        if not rankings:
            return []
        return weighted_rrf(rankings, resolved, rrf_k)

    def get_many(self, ids: Sequence[str]) -> List[SearchHit]:
        return self.backend.get_many([str(record_id) for record_id in ids])

    def stats(self) -> Dict[str, Any]:
        return self.backend.stats()

    def search_community_reports(
        self,
        kb_id: str,
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        entity_names: Optional[Sequence[str]] = None,
        top_k: int = 100,
        *,
        limit: Optional[int] = None,
    ) -> List[SearchHit]:
        resolved = _resolve_top_k(top_k, limit)
        return self.backend.search_community_reports(
            str(kb_id),
            str(doc_id) if doc_id is not None else None,
            query,
            entity_names,
            resolved,
        )


def _nested_value(document: Mapping[str, Any], field: str) -> Any:
    value: Any = document
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _matches_filters(
    document: Mapping[str, Any], filters: Optional[Mapping[str, Any]]
) -> bool:
    if not filters:
        return True
    for field, expected in filters.items():
        actual = _nested_value(document, str(field))
        if isinstance(expected, (list, tuple, set, frozenset)):
            candidates = list(expected)
            if not candidates:
                return False
            if isinstance(actual, (list, tuple, set, frozenset)):
                if not any(item in candidates for item in actual):
                    return False
            elif actual not in candidates:
                return False
        elif isinstance(actual, (list, tuple, set, frozenset)):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemorySearchBackend:
    """Small, explicit test double implementing the production contract."""

    def __init__(self, *, fail_on_ready: Optional[BaseException] = None) -> None:
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._fail_on_ready = fail_on_ready
        self.ensure_ready_calls = 0
        self.ready = False

    def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1
        if self._fail_on_ready is not None:
            raise self._fail_on_ready
        self.ready = True

    def upsert(
        self,
        id: str,
        document: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> None:
        payload = copy.deepcopy(dict(document))
        embedded = payload.pop(VECTOR_FIELD, None)
        vector = vector if vector is not None else embedded
        record_id = str(id)
        self._documents[record_id] = payload
        if vector is None:
            self._vectors.pop(record_id, None)
        else:
            self._vectors[record_id] = validate_vector(vector)

    def delete_by_id(self, id: str) -> bool:
        record_id = str(id)
        existed = record_id in self._documents
        self._documents.pop(record_id, None)
        self._vectors.pop(record_id, None)
        return existed

    def delete_by_filter(self, filters: Optional[Mapping[str, Any]]) -> int:
        record_ids = [
            record_id
            for record_id, document in self._documents.items()
            if _matches_filters(document, filters)
        ]
        for record_id in record_ids:
            self.delete_by_id(record_id)
        return len(record_ids)

    def update_by_filter(
        self,
        filters: Optional[Mapping[str, Any]],
        updates: Mapping[str, Any],
        vector: Optional[Sequence[float]] = None,
    ) -> int:
        record_ids = [
            record_id
            for record_id, document in self._documents.items()
            if _matches_filters(document, filters)
        ]
        normalized = validate_vector(vector) if vector is not None else None
        for record_id in record_ids:
            self._documents[record_id].update(copy.deepcopy(dict(updates)))
            self._documents[record_id].pop(VECTOR_FIELD, None)
            if normalized is not None:
                self._vectors[record_id] = list(normalized)
        return len(record_ids)

    def query(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SearchHit]:
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset non-negative")
        record_ids = sorted(
            record_id
            for record_id, document in self._documents.items()
            if _matches_filters(document, filters)
        )
        return [
            SearchHit(record_id, 1.0, copy.deepcopy(self._documents[record_id]))
            for record_id in record_ids[offset : offset + limit]
        ]

    def count(self, filters: Optional[Mapping[str, Any]] = None) -> int:
        return sum(
            1
            for document in self._documents.values()
            if _matches_filters(document, filters)
        )

    @staticmethod
    def _searchable_text(document: Mapping[str, Any]) -> str:
        fields = (
            "content",
            "title",
            "section",
            "summary",
            "doc_name",
            "entities",
            "report",
        )
        values = [_nested_value(document, field) for field in fields]
        return " ".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value or "")
            for value in values
        ).lower()

    def keyword_search(
        self,
        query: str,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]:
        needle = str(query).strip().lower()
        if not needle:
            return self.query(filters, top_k)
        tokens = [token for token in needle.split() if token] or [needle]
        hits: List[SearchHit] = []
        for record_id, document in self._documents.items():
            if not _matches_filters(document, filters):
                continue
            haystack = self._searchable_text(document)
            score = float(sum(haystack.count(token) for token in tokens))
            if score > 0:
                hits.append(SearchHit(record_id, score, copy.deepcopy(document)))
        return sorted(hits, key=lambda hit: (-hit.score, hit.id))[:top_k]

    def vector_search(
        self,
        vector: Sequence[float],
        filters: Optional[Mapping[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchHit]:
        normalized = validate_vector(vector)
        hits = [
            SearchHit(
                record_id,
                _cosine(normalized, stored_vector),
                copy.deepcopy(self._documents[record_id]),
            )
            for record_id, stored_vector in self._vectors.items()
            if record_id in self._documents
            and _matches_filters(self._documents[record_id], filters)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.id))[:top_k]

    def get_many(self, ids: Sequence[str]) -> List[SearchHit]:
        return [
            SearchHit(record_id, 1.0, copy.deepcopy(self._documents[record_id]))
            for record_id in map(str, ids)
            if record_id in self._documents
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "ready": self.ready,
            "documents": len(self._documents),
            "vectors": len(self._vectors),
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
        filters: Dict[str, Any] = {
            "kb_id": kb_id,
            "knowledge_graph_kwd": "community_report",
        }
        if doc_id is not None:
            filters["doc_ids"] = [doc_id]
        search_text = " ".join(
            part
            for part in [query or "", " ".join(entity_names or ())]
            if part.strip()
        )
        hits = (
            self.keyword_search(search_text, filters, top_k * 2)
            if search_text
            else self.query(filters, top_k * 2)
        )
        results: List[SearchHit] = []
        for hit in hits:
            stale = hit.document.get("stale_kwd", "N")
            if stale == "topology":
                continue
            score = hit.score * 0.7 if stale == "content" else hit.score
            results.append(_copy_hit(hit, score=score))
            if len(results) >= top_k:
                break
        return results


# Construction is deliberately cheap and side-effect free.
search_store = SearchStore()


__all__ = [
    "InMemorySearchBackend",
    "SearchBackend",
    "SearchHit",
    "SearchStore",
    "SearchStoreConnectionError",
    "SearchStoreError",
    "SearchStoreSchemaError",
    "VECTOR_DIMENSION",
    "VECTOR_FIELD",
    "search_store",
    "validate_vector",
    "weighted_rrf",
]
