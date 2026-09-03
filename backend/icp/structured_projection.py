"""Rebuildable HugeGraph projection for explicit PostgreSQL relationships.

Every parsed ICP document receives a source ``Document`` vertex.  The five
table/text document types project deterministic PostgreSQL rows.  FTA/RBD also
project a formal cross-document view, but only from their active, published,
formally validated PostgreSQL revision; provisional topology stays isolated in
the dedicated diagram namespace.

PostgreSQL remains authoritative.  This module deletes and rebuilds only one
document's HugeGraph contribution and verifies the write by reading it back.
"""

from __future__ import annotations

import copy
import json
import threading
import urllib.parse
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

import psycopg2

from backend.config.config import config
from backend.icp.diagram_models import content_hash
from backend.icp.diagram_repository import diagram_repository
from backend.icp.entity_resolution import entity_resolution_repository
from backend.icp.structured_graph import (
    SUPPORTED_DOC_TYPES,
    build_formal_diagram_graph,
    build_structured_graph,
    business_node_id,
)
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.postgres_client import postgres_client


STRUCTURED_KB_ID = "icp_structured"
PROJECTOR_VERSION = "structured-projector-v5.1-schema-safe-identity"
PROJECTABLE_DOC_TYPES = frozenset((*SUPPORTED_DOC_TYPES, "fta", "rbd"))

_TABLES_BY_TYPE = {
    "hmc": "fault_hmc",
    "fmeca": "fmea",
    "mfl": "mfl",
}
_LOCAL_PROJECTION_LOCK = threading.RLock()


@contextmanager
def _projection_lock() -> Iterator[None]:
    """Serialize shared-node replacement locally and across backend workers."""

    with _LOCAL_PROJECTION_LOCK:
        # Unit/developer mock mode has no PostgreSQL session on which to hold an
        # advisory lock.  The in-process lock still makes tests and a one-worker
        # developer server deterministic.
        if not getattr(getattr(postgres_client, "conn", None), "dsn", ""):
            yield
            return
        lock_conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB,
        )
        try:
            with lock_conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_lock(hashtext(%s))",
                    ("icp-rag:structured-projection",),
                )
            lock_conn.commit()
            yield
        finally:
            try:
                with lock_conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(hashtext(%s))",
                        ("icp-rag:structured-projection",),
                    )
                lock_conn.commit()
            except Exception:
                # Closing the dedicated session also releases session-level
                # advisory locks; never mask the projection's original error.
                pass
            finally:
                lock_conn.close()


def _rows(cur: Any, values: Iterable[Any]) -> List[Dict[str, Any]]:
    columns = [getattr(item, "name", item[0]) for item in (cur.description or [])]
    return [
        dict(zip(columns, row)) if not isinstance(row, Mapping) else dict(row)
        for row in values
    ]


def _query_all(table: str, doc_id: str) -> List[Dict[str, Any]]:
    if table not in {
        "fault_hmc",
        "fmea",
        "mfl",
        "icp_components",
        "icp_functions",
        "icp_interfaces",
        "fault_history_cases",
    }:
        raise ValueError(f"unsupported structured graph table: {table}")
    with postgres_client.conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table} WHERE doc_id=%s ORDER BY id", (doc_id,))
        return _rows(cur, cur.fetchall())


def _history_rows(doc_id: str) -> List[Dict[str, Any]]:
    rows = _query_all("fault_history_cases", doc_id)
    with postgres_client.conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                SELECT step_no,content,source_ref,raw_data
                FROM fault_history_steps WHERE case_pk=%s ORDER BY step_no,id
                """,
                (row["id"],),
            )
            row["steps"] = _rows(cur, cur.fetchall())
    return rows


def load_structured_rows(doc_id: str, doc_type: str) -> Any:
    """Load one document snapshot using closed, parameterized table choices."""

    dtype = str(doc_type or "").strip().casefold()
    if dtype == "function_locator":
        return {
            "components": _query_all("icp_components", doc_id),
            "functions": _query_all("icp_functions", doc_id),
            "interfaces": _query_all("icp_interfaces", doc_id),
        }
    if dtype == "fault_history":
        return _history_rows(doc_id)
    table = _TABLES_BY_TYPE.get(dtype)
    if table:
        return _query_all(table, doc_id)
    if dtype in {"fta", "rbd"}:
        return []
    raise ValueError(f"unsupported ICP document type: {doc_type}")


def load_formal_diagram_revisions(doc_id: str, doc_type: str) -> List[Dict[str, Any]]:
    """Load active published diagram revisions; the builder filters formality.

    ``list_active_revisions`` deliberately includes provisional revisions for
    the visual browsing product.  ``build_formal_diagram_graph`` applies the
    stricter formal-claims gate before anything reaches this shared graph.
    """

    dtype = str(doc_type or "").strip().casefold()
    if dtype not in {"fta", "rbd"}:
        raise ValueError(f"unsupported diagram document type: {doc_type}")
    return diagram_repository.list_active_revisions(
        doc_id=str(doc_id or "").strip(), diagram_type=dtype
    )


def _document_vertex(document: Mapping[str, Any]) -> Dict[str, Any]:
    doc_id = str(document.get("doc_id") or "").strip()
    doc_type = str(document.get("doc_type") or "").strip()
    metadata = document.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            metadata = {}
    return {
        "id": business_node_id("document", doc_id),
        "label": "Document",
        "properties": {
            "business_id": doc_id,
            "name": str(document.get("file_name") or doc_id),
            "doc_type": doc_type,
            "storage_key": str(
                document.get("storage_key") or document.get("minio_path") or ""
            ),
            "metadata": metadata if isinstance(metadata, Mapping) else {},
            "identity_scope": "document",
            "doc_ids": [doc_id],
        },
    }


def build_document_graph(doc_id: str, doc_type: str) -> Dict[str, Any]:
    """Build the explicit graph plus one common source document vertex."""

    doc_id = str(doc_id or "").strip()
    doc_type = str(doc_type or "").strip().casefold()
    if not doc_id:
        raise ValueError("doc_id must not be empty")
    if doc_type not in PROJECTABLE_DOC_TYPES:
        raise ValueError(f"unsupported ICP document type: {doc_type}")
    document = postgres_client.get_icp_document(doc_id)
    if not document:
        raise ValueError(f"ICP document not found: {doc_id}")
    if str(document.get("doc_type") or "").strip().casefold() != doc_type:
        raise ValueError("document type mismatch during structured graph projection")

    if doc_type in SUPPORTED_DOC_TYPES:
        graph = build_structured_graph(
            doc_type, doc_id, load_structured_rows(doc_id, doc_type)
        )
    else:
        graph = build_formal_diagram_graph(
            doc_type,
            doc_id,
            load_formal_diagram_revisions(doc_id, doc_type),
        )
    document_vertex = _document_vertex(document)
    vertices = [*graph.get("vertices", []), document_vertex]
    document_id = document_vertex["id"]
    edges = list(graph.get("edges", []))
    for vertex in graph.get("vertices", []):
        source_id = str(vertex.get("id") or "")
        if not source_id:
            continue
        edge_id = "sg:e:extracted_from:" + content_hash(
            {"doc_id": doc_id, "source": source_id, "target": document_id}
        )[:32]
        vertex_props = vertex.get("properties") or {}
        vertex_props = dict(vertex_props) if isinstance(vertex_props, Mapping) else {}
        provenance = {
            key: vertex_props.get(key)
            for key in (
                "diagram_id",
                "diagram_key",
                "revision_id",
                "revision_no",
                "canonical_hash",
                "review_status",
                "validation_status",
                "validation_tier",
                "projection_status",
                "formal_claims_allowed",
                "source_refs",
            )
            if vertex_props.get(key) not in (None, "", [], {})
        }
        edges.append(
            {
                "id": edge_id,
                "label": "extracted_from",
                "source": source_id,
                "target": document_id,
                "properties": {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "explicit": True,
                    **provenance,
                },
            }
        )
    return {**graph, "vertices": vertices, "edges": edges}


def _display_name(vertex: Mapping[str, Any]) -> str:
    props = vertex.get("properties") or {}
    return str(
        props.get("name")
        or props.get("text")
        or props.get("business_id")
        or vertex.get("id")
        or "结构化实体"
    )


def _physical_graph(
    graph: Mapping[str, Any],
    *,
    doc_id: str,
    doc_type: str,
    source_hash: str,
    logical_entity_ids: Mapping[str, str],
    identity_scope: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    created_at = datetime.now(timezone.utc).isoformat()
    id_map: Dict[str, str] = {}
    vertices_by_id: Dict[str, Dict[str, Any]] = {}
    for vertex in graph.get("vertices", []) or []:
        logical_id = str(vertex.get("id") or "")
        if not logical_id:
            continue
        canonical_entity_id = str(logical_entity_ids.get(logical_id) or "").strip()
        if not canonical_entity_id:
            raise ValueError(
                f"structured vertex has no PostgreSQL canonical identity: {logical_id}"
            )
        # Logical business IDs intentionally omit applicability so retrieval can
        # use them as seeds.  Physical IDs must be scope-aware; the canonical PG
        # entity ID includes system/configuration scope and prevents CONFIG-A/B
        # paths from meeting at one HugeGraph vertex.
        physical_id = f"icp:ce:{canonical_entity_id}"
        id_map[logical_id] = physical_id
        raw_props = vertex.get("properties") or {}
        raw_props = dict(raw_props) if isinstance(raw_props, Mapping) else {}
        vertex_scope = (
            {} if str(vertex.get("label") or "") == "Document" else dict(identity_scope)
        )
        serialized_props = {
            **raw_props,
            "canonical_entity_id": canonical_entity_id,
            "identity_scope": vertex_scope,
        }
        serialized = json.dumps(
            serialized_props, ensure_ascii=False, sort_keys=True, default=str
        )
        name = _display_name(vertex)
        entity_type = str(vertex.get("label") or "StructuredEntity")
        contribution = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_hash": source_hash,
            "logical_id": logical_id,
            "name": name,
            "entity_type": entity_type,
            "canonical_entity_id": canonical_entity_id,
            "identity_scope": vertex_scope,
            "text": name,
            "source_refs": _string_list(raw_props.get("source_refs")),
            "original_properties": raw_props,
            "summary": serialized[:32000],
        }
        incoming = {
            "id": physical_id,
            "label": "Entity",
            "properties": {
                "name": name,
                "entity_type": entity_type,
                "description": name,
                "summary": serialized[:32000],
                "canonical_entity_id": canonical_entity_id,
                # HugeGraph 1.7 property values must match the TEXT key.  The
                # contribution record above retains the structured object.
                "identity_scope": json.dumps(
                    vertex_scope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ),
                "doc_id": [doc_id],
                "kb_id": STRUCTURED_KB_ID,
                "project_id": STRUCTURED_KB_ID,
                "tenant_id": STRUCTURED_KB_ID,
                "source_id": [
                    f"structured-doc:{doc_id}",
                    f"logical:{logical_id}",
                    f"canonical-entity:{canonical_entity_id}",
                    f"projection:{doc_id}:{source_hash}",
                ],
                "created_at": created_at,
                "description_parts": json.dumps(
                    [contribution], ensure_ascii=False, default=str
                ),
            },
        }
        existing = vertices_by_id.get(physical_id)
        if existing is None:
            vertices_by_id[physical_id] = incoming
        else:
            vertices_by_id[physical_id] = _merge_physical_vertices(
                existing, incoming
            )

    edges_by_identity: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for edge in graph.get("edges", []) or []:
        source_logical_id = str(edge.get("source") or "")
        target_logical_id = str(edge.get("target") or "")
        source = id_map.get(source_logical_id)
        target = id_map.get(target_logical_id)
        if not source or not target:
            raise ValueError(f"structured relation has unknown endpoint: {edge.get('id')}")
        relation = str(edge.get("label") or "relation").strip() or "relation"
        logical_id = str(edge.get("id") or content_hash(edge)[:32])
        raw_props = edge.get("properties") or {}
        raw_props = dict(raw_props) if isinstance(raw_props, Mapping) else {}
        serialized = json.dumps(raw_props, ensure_ascii=False, sort_keys=True, default=str)
        contribution = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_hash": source_hash,
            "logical_id": logical_id,
            "source_logical_id": source_logical_id,
            "target_logical_id": target_logical_id,
            "source_physical_id": source,
            "target_physical_id": target,
            "source_relation": relation,
            "source_refs": _string_list(raw_props.get("source_refs")),
            "occurrence_qualifiers": (
                list(raw_props.get("occurrence_qualifiers") or [])
                if isinstance(
                    raw_props.get("occurrence_qualifiers"),
                    (list, tuple, set, frozenset),
                )
                else (
                    [raw_props.get("occurrence_qualifiers")]
                    if raw_props.get("occurrence_qualifiers") not in (None, "")
                    else []
                )
            ),
            "revision_id": raw_props.get("revision_id") or "",
            "validation_status": raw_props.get("validation_status") or "",
            "original_properties": raw_props,
            "text": relation,
            "summary": serialized[:32000],
        }
        identity = (doc_id, relation, source, target)
        physical_key = content_hash(
            {
                "doc_id": doc_id,
                "relation": relation,
                "outV": source,
                "inV": target,
            }
        )[:32]
        incoming = {
            "id": f"icp:sg:{physical_key}",
            "label": relation,
            "outV": source,
            "inV": target,
            "outVLabel": "Entity",
            "inVLabel": "Entity",
            "properties": {
                "description": relation,
                "weight": 1,
                "keywords": [doc_type, relation],
                "doc_id": [doc_id],
                "kb_id": STRUCTURED_KB_ID,
                "project_id": STRUCTURED_KB_ID,
                "tenant_id": STRUCTURED_KB_ID,
                "source_id": [
                    f"structured-doc:{doc_id}",
                    f"logical:{logical_id}",
                    f"projection:{doc_id}:{source_hash}",
                ],
                "created_at": created_at,
                "description_parts": json.dumps(
                    [contribution], ensure_ascii=False, default=str
                ),
                "keywords_parts": json.dumps(
                    [
                        {
                            "doc_id": doc_id,
                            "logical_id": logical_id,
                            "keywords": [doc_type, relation],
                        }
                    ],
                    ensure_ascii=False,
                ),
                "weight_parts": json.dumps(
                    [{"doc_id": doc_id, "logical_id": logical_id, "weight": 1}],
                    ensure_ascii=False,
                ),
            },
        }
        existing = edges_by_identity.get(identity)
        if existing is None:
            edges_by_identity[identity] = incoming
        else:
            edges_by_identity[identity] = _merge_physical_edges(existing, incoming)
    return (
        [vertices_by_id[key] for key in sorted(vertices_by_id)],
        [edges_by_identity[key] for key in sorted(edges_by_identity)],
    )


def _string_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [item for item in raw.split(",") if item]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return [str(item) for item in raw if item]
    return []


def _parts(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _stable_parts(*values: Any) -> List[Dict[str, Any]]:
    """Return deterministic, lossless projection contribution records."""

    unique: Dict[str, Dict[str, Any]] = {}
    for raw in values:
        for part in _parts(raw):
            marker = json.dumps(
                part, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            )
            unique[marker] = part
    return [
        unique[key]
        for key in sorted(
            unique,
            key=lambda marker: (
                str(unique[marker].get("doc_id") or ""),
                str(unique[marker].get("logical_id") or ""),
                marker,
            ),
        )
    ]


def _source_ids_for_parts(
    source_ids: Iterable[str], contributions: Iterable[Mapping[str, Any]]
) -> List[str]:
    """Rebuild logical markers when every retained contribution is attributable."""

    parts = [dict(item) for item in contributions]
    logical_ids = {
        str(item.get("logical_id") or "").strip()
        for item in parts
        if str(item.get("logical_id") or "").strip()
    }
    sources = {str(item) for item in source_ids if str(item)}
    # v5 contributions always carry logical_id.  Exact reconstruction prevents
    # a removed document from leaving stale logical markers on a shared item.
    # For an old contribution without attribution, retain its marker set until
    # that other document is rebuilt rather than silently discarding evidence.
    if not parts or all(str(item.get("logical_id") or "").strip() for item in parts):
        sources = {item for item in sources if not item.startswith("logical:")}
    sources.update(f"logical:{logical_id}" for logical_id in logical_ids)
    return sorted(sources)


def _merged_vertex_properties(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, Any]:
    left_props = copy.deepcopy(dict(left or {}))
    right_props = copy.deepcopy(dict(right or {}))
    canonical_ids = {
        str(value or "").strip()
        for value in (
            left_props.get("canonical_entity_id"),
            right_props.get("canonical_entity_id"),
        )
        if str(value or "").strip()
    }
    if len(canonical_ids) > 1:
        raise ValueError("canonical physical vertex merged different entity identities")
    def serialized_scope(value: Any) -> str:
        parsed = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = value
        return json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    scopes = {
        serialized_scope(value)
        for value in (
            left_props.get("identity_scope"),
            right_props.get("identity_scope"),
        )
        if value not in (None, "", {})
    }
    if len(scopes) > 1:
        raise ValueError("canonical physical vertex merged different applicability scopes")

    contributions = _stable_parts(
        left_props.get("description_parts"), right_props.get("description_parts")
    )
    entity_types = sorted(
        {
            str(part.get("entity_type") or "").strip()
            for part in contributions
            if str(part.get("entity_type") or "").strip()
        }
    )
    if len(entity_types) > 1:
        raise ValueError("canonical physical vertex merged different entity types")
    names = sorted(
        {
            str(part.get("name") or part.get("text") or "").strip()
            for part in contributions
            if str(part.get("name") or part.get("text") or "").strip()
        }
    )
    summaries = [
        str(part.get("summary") or "")
        for part in contributions
        if str(part.get("summary") or "")
    ]
    merged = {**left_props, **right_props}
    if scopes:
        merged["identity_scope"] = sorted(scopes)[0]
    merged["doc_id"] = sorted(
        set(_string_list(left_props.get("doc_id")))
        | set(_string_list(right_props.get("doc_id")))
    )
    merged["source_id"] = _source_ids_for_parts(
        [
            *_string_list(left_props.get("source_id")),
            *_string_list(right_props.get("source_id")),
        ],
        contributions,
    )
    merged["description_parts"] = json.dumps(
        contributions, ensure_ascii=False, default=str
    )
    if names:
        merged["name"] = names[0]
        merged["description"] = names[0]
    if entity_types:
        merged["entity_type"] = entity_types[0]
    merged["summary"] = "<SEP>".join(summaries)[:32000]
    merged["kb_id"] = STRUCTURED_KB_ID
    merged["project_id"] = STRUCTURED_KB_ID
    merged["tenant_id"] = STRUCTURED_KB_ID
    created = sorted(
        value
        for value in (
            str(left_props.get("created_at") or ""),
            str(right_props.get("created_at") or ""),
        )
        if value
    )
    if created:
        merged["created_at"] = created[0]
    return merged


def _merge_physical_vertices(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> Dict[str, Any]:
    if (
        str(existing.get("id") or "") != str(incoming.get("id") or "")
        or str(existing.get("label") or "") != str(incoming.get("label") or "")
    ):
        raise ValueError("cannot merge different physical vertices")
    return {
        **copy.deepcopy(dict(existing)),
        "properties": _merged_vertex_properties(
            existing.get("properties") or {}, incoming.get("properties") or {}
        ),
    }


def _endpoint(value: Any) -> str:
    text = str(value or "")
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


def _edge_identity(item: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(item.get("label") or ""),
        _endpoint(item.get("outV") or item.get("source")),
        _endpoint(item.get("inV") or item.get("target")),
    )


def _merged_edge_properties(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, Any]:
    left_props = copy.deepcopy(dict(left or {}))
    right_props = copy.deepcopy(dict(right or {}))
    contributions = _stable_parts(
        left_props.get("description_parts"), right_props.get("description_parts")
    )
    keyword_parts = _stable_parts(
        left_props.get("keywords_parts"), right_props.get("keywords_parts")
    )
    weight_parts = _stable_parts(
        left_props.get("weight_parts"), right_props.get("weight_parts")
    )
    merged = {**left_props, **right_props}
    merged["doc_id"] = sorted(
        set(_string_list(left_props.get("doc_id")))
        | set(_string_list(right_props.get("doc_id")))
    )
    merged["source_id"] = _source_ids_for_parts(
        [
            *_string_list(left_props.get("source_id")),
            *_string_list(right_props.get("source_id")),
        ],
        contributions,
    )
    merged["description_parts"] = json.dumps(
        contributions, ensure_ascii=False, default=str
    )
    merged["keywords_parts"] = json.dumps(
        keyword_parts, ensure_ascii=False, default=str
    )
    merged["weight_parts"] = json.dumps(
        weight_parts, ensure_ascii=False, default=str
    )
    merged["keywords"] = sorted(
        {
            str(keyword)
            for part in keyword_parts
            for keyword in (part.get("keywords") or [])
            if keyword
        }
    )
    merged["weight"] = sum(int(part.get("weight") or 0) for part in weight_parts)
    merged["kb_id"] = STRUCTURED_KB_ID
    merged["project_id"] = STRUCTURED_KB_ID
    merged["tenant_id"] = STRUCTURED_KB_ID
    created = sorted(
        value
        for value in (
            str(left_props.get("created_at") or ""),
            str(right_props.get("created_at") or ""),
        )
        if value
    )
    if created:
        merged["created_at"] = created[0]
    return merged


def _merge_physical_edges(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> Dict[str, Any]:
    if _edge_identity(existing) != _edge_identity(incoming):
        raise ValueError("cannot merge different physical relations")
    return {
        **copy.deepcopy(dict(existing)),
        "properties": _merged_edge_properties(
            existing.get("properties") or {}, incoming.get("properties") or {}
        ),
    }


def _doc_ids(item: Mapping[str, Any]) -> List[str]:
    return _string_list((item.get("properties") or {}).get("doc_id"))


def _has_doc(item: Mapping[str, Any], doc_id: str) -> bool:
    return doc_id in _doc_ids(item)


def _remove_doc_contribution(
    item: Mapping[str, Any], doc_id: str
) -> Dict[str, Any]:
    """Return properties with only this structured document contribution removed."""

    props = copy.deepcopy(dict(item.get("properties") or {}))
    props["doc_id"] = sorted(set(_doc_ids(item)) - {doc_id})
    sources = _string_list(props.get("source_id"))
    props["source_id"] = sorted(
        source
        for source in set(sources)
        if source != doc_id
        and source != f"structured-doc:{doc_id}"
        and not source.startswith(f"projection:{doc_id}:")
    )

    description_parts = [
        part
        for part in _parts(props.get("description_parts"))
        if str(part.get("doc_id") or "") != doc_id
    ]
    if "description_parts" in props:
        description_parts = _stable_parts(description_parts)
        props["description_parts"] = json.dumps(description_parts, ensure_ascii=False)
        props["source_id"] = _source_ids_for_parts(
            props.get("source_id") or [], description_parts
        )
        names = sorted(
            {
                str(part.get("name") or part.get("text") or "").strip()
                for part in description_parts
                if str(part.get("name") or part.get("text") or "").strip()
            }
        )
        summaries = [
            str(part.get("summary") or "")
            for part in description_parts
            if str(part.get("summary") or "")
        ]
        if names:
            props["name"] = names[0]
            props["description"] = names[0]
        else:
            props.pop("name", None)
            props.pop("description", None)
        if summaries:
            props["summary"] = "<SEP>".join(summaries)[:32000]
        else:
            props.pop("summary", None)

    keyword_parts = [
        part
        for part in _parts(props.get("keywords_parts"))
        if str(part.get("doc_id") or "") != doc_id
    ]
    if "keywords_parts" in props:
        keyword_parts = _stable_parts(keyword_parts)
        props["keywords_parts"] = json.dumps(keyword_parts, ensure_ascii=False)
        props["keywords"] = sorted(
            {
                str(keyword)
                for part in keyword_parts
                for keyword in (part.get("keywords") or [])
                if keyword
            }
        )

    weight_parts = [
        part
        for part in _parts(props.get("weight_parts"))
        if str(part.get("doc_id") or "") != doc_id
    ]
    if "weight_parts" in props:
        weight_parts = _stable_parts(weight_parts)
        props["weight_parts"] = json.dumps(weight_parts, ensure_ascii=False)
        props["weight"] = sum(
            int(part.get("weight") or 0) for part in weight_parts
        )
    return props


def _merge_vertex_properties(
    existing: Mapping[str, Any], incoming: Mapping[str, Any], doc_id: str
) -> Dict[str, Any]:
    """Replace one doc contribution while retaining all other business-node sources."""

    base = _remove_doc_contribution(existing, doc_id)
    return _merged_vertex_properties(base, incoming.get("properties") or {})


def _merge_edge_properties(
    existing: Mapping[str, Any], incoming: Mapping[str, Any], doc_id: str
) -> Dict[str, Any]:
    """Replace this document's canonical-edge contribution, retaining others."""

    if _edge_identity(existing) != _edge_identity(incoming):
        raise ValueError("cannot merge different physical relations")
    base = _remove_doc_contribution(existing, doc_id)
    return _merged_edge_properties(base, incoming.get("properties") or {})


def _replace_item_properties(
    item: Mapping[str, Any], properties: Mapping[str, Any], *, edge: bool
) -> bool:
    """Exactly replace properties, bypassing the client's additive update API."""

    method_name = "replace_edge_properties" if edge else "replace_vertex_properties"
    replacement = getattr(hugegraph_client, method_name, None)
    if callable(replacement):
        return bool(replacement(str(item.get("id") or ""), dict(properties)))

    # The legacy client only exposes additive update methods.  Its own update
    # implementation uses this full-resource PUT, so keep the exact replacement
    # local until the storage abstraction grows a public replace operation.
    session = getattr(hugegraph_client, "session", None)
    graph_url = getattr(hugegraph_client, "_get_graph_url", None)
    if session is None or not callable(graph_url):
        raise RuntimeError(f"HugeGraph 客户端缺少精确替换能力: {method_name}")
    item_id = str(item.get("id") or "")
    if not item_id:
        raise RuntimeError("HugeGraph 读回项缺少 id")
    if edge:
        encoded_id = urllib.parse.quote(item_id, safe="")
        path = f"/edges/{encoded_id}"
        payload = {
            "id": item_id,
            "label": item.get("label") or "relation",
            "outV": item.get("outV"),
            "inV": item.get("inV"),
            "outVLabel": item.get("outVLabel") or "Entity",
            "inVLabel": item.get("inVLabel") or "Entity",
            "properties": dict(properties),
        }
    else:
        quote_vertex = getattr(hugegraph_client, "_quote_vid", None)
        raw_id = quote_vertex(item_id) if callable(quote_vertex) else item_id
        encoded_id = urllib.parse.quote(raw_id, safe="")
        path = f"/vertices/{encoded_id}"
        payload = {
            "id": item_id,
            "label": item.get("label") or "Entity",
            "properties": dict(properties),
        }
    response = session.put(graph_url(path), json=payload)
    return int(getattr(response, "status_code", 0)) in {200, 201}


def _delete_edge(edge_id: str) -> bool:
    """Delete a server-generated HugeGraph edge ID with URL-safe encoding."""

    session = getattr(hugegraph_client, "session", None)
    graph_url = getattr(hugegraph_client, "_get_graph_url", None)
    if session is not None and callable(graph_url):
        encoded_id = urllib.parse.quote(str(edge_id or ""), safe="")
        response = session.delete(graph_url(f"/edges/{encoded_id}"))
        return int(getattr(response, "status_code", 0)) == 204
    return bool(hugegraph_client.delete_edge(str(edge_id or "")))


def _read_structured_items() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    vertices = list(
        hugegraph_client.get_vertices_by_kb_with_details(STRUCTURED_KB_ID) or []
    )
    edges = list(
        hugegraph_client.get_edges_by_kb_with_details(STRUCTURED_KB_ID) or []
    )
    return (
        [
            item
            for item in vertices
            if (item.get("properties") or {}).get("kb_id") == STRUCTURED_KB_ID
        ],
        [
            item
            for item in edges
            if (item.get("properties") or {}).get("kb_id") == STRUCTURED_KB_ID
        ],
    )


def _readback_matches(
    doc_id: str,
    vertices: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
) -> Tuple[bool, int, int]:
    expected_vertices = list(vertices)
    expected_edges = list(edges)
    stored_vertices, stored_edges = _read_structured_items()
    expected_vertex_ids = {
        str(item.get("id")) for item in expected_vertices if item.get("id")
    }
    stored_vertices_by_id = {
        str(item.get("id")): item for item in stored_vertices if item.get("id")
    }
    # A secondary-index/offset scan can omit a vertex immediately after its
    # LIST properties are updated, even though a direct CUSTOMIZE_STRING lookup
    # already returns the committed value.  Recover only missing expected IDs;
    # all contribution/source-hash checks below still apply unchanged.
    direct_lookup = getattr(hugegraph_client, "get_vertex_by_id", None)
    if callable(direct_lookup):
        for item_id in sorted(expected_vertex_ids):
            current = stored_vertices_by_id.get(item_id)
            if current is not None and _has_doc(current, doc_id):
                continue
            recovered = direct_lookup(item_id)
            if (
                isinstance(recovered, Mapping)
                and (recovered.get("properties") or {}).get("kb_id")
                == STRUCTURED_KB_ID
            ):
                stored_vertices_by_id[item_id] = dict(recovered)
        stored_vertices = list(stored_vertices_by_id.values())
    actual_vertices_by_id = {
        str(item.get("id")): item
        for item in stored_vertices
        if _has_doc(item, doc_id)
    }
    actual_doc_edges = [item for item in stored_edges if _has_doc(item, doc_id)]
    expected_vertices_by_id = {
        str(item.get("id")): item for item in expected_vertices
    }
    actual_edges_by_identity = {
        _edge_identity(item): item for item in actual_doc_edges
    }
    expected_edges_by_identity = {
        _edge_identity(item): item for item in expected_edges
    }

    def contribution_matches(
        actual: Mapping[str, Any], expected: Mapping[str, Any], *, edge: bool
    ) -> bool:
        actual_props = actual.get("properties") or {}
        expected_props = expected.get("properties") or {}
        expected_markers = {
            value
            for value in _string_list(expected_props.get("source_id"))
            if value.startswith(f"projection:{doc_id}:")
        }
        if not expected_markers.issubset(
            set(_string_list(actual_props.get("source_id")))
        ):
            return False
        expected_parts = [
            part
            for part in _parts(expected_props.get("description_parts"))
            if str(part.get("doc_id") or "") == doc_id
        ]
        actual_parts = [
            part
            for part in _parts(actual_props.get("description_parts"))
            if str(part.get("doc_id") or "") == doc_id
        ]
        expected_parts = _stable_parts(expected_parts)
        actual_parts = _stable_parts(actual_parts)
        if not expected_parts or expected_parts != actual_parts:
            return False
        expected_logical_ids = {
            str(part.get("logical_id") or "").strip()
            for part in expected_parts
            if str(part.get("logical_id") or "").strip()
        }
        actual_logical_ids = {
            str(part.get("logical_id") or "").strip()
            for part in actual_parts
            if str(part.get("logical_id") or "").strip()
        }
        if not expected_logical_ids or expected_logical_ids != actual_logical_ids:
            return False
        actual_logical_markers = {
            value[len("logical:") :]
            for value in _string_list(actual_props.get("source_id"))
            if value.startswith("logical:")
        }
        if not expected_logical_ids.issubset(actual_logical_markers):
            return False
        if edge:
            return _edge_identity(actual) == _edge_identity(expected)
        return str(actual.get("label") or "") == str(expected.get("label") or "")

    identifiers_match = (
        set(actual_vertices_by_id) == set(expected_vertices_by_id)
        and len(actual_edges_by_identity) == len(actual_doc_edges)
        and len(expected_edges_by_identity) == len(expected_edges)
        and set(actual_edges_by_identity) == set(expected_edges_by_identity)
    )
    contributions_match = (
        identifiers_match
        and all(
            contribution_matches(actual_vertices_by_id[item_id], item, edge=False)
            for item_id, item in expected_vertices_by_id.items()
        )
        and all(
            contribution_matches(actual_edges_by_identity[identity], item, edge=True)
            for identity, item in expected_edges_by_identity.items()
        )
    )
    return (
        contributions_match,
        len(actual_vertices_by_id),
        len(actual_doc_edges),
    )


def _remove_existing_contribution(
    doc_id: str,
    stored_vertices: Iterable[Mapping[str, Any]],
    stored_edges: Iterable[Mapping[str, Any]],
) -> None:
    """Remove only ``STRUCTURED_KB_ID`` items for one document (edges first)."""

    for edge in sorted(
        (item for item in stored_edges if _has_doc(item, doc_id)),
        key=lambda item: str(item.get("id") or ""),
    ):
        remaining = sorted(set(_doc_ids(edge)) - {doc_id})
        if remaining:
            props = _remove_doc_contribution(edge, doc_id)
            if not _replace_item_properties(edge, props, edge=True):
                raise RuntimeError(f"裁剪共享结构化关系失败: {edge.get('id')}")
        elif not _delete_edge(str(edge.get("id") or "")):
            raise RuntimeError(f"清理上一版结构化关系失败: {edge.get('id')}")

    for vertex in sorted(
        (item for item in stored_vertices if _has_doc(item, doc_id)),
        key=lambda item: str(item.get("id") or ""),
    ):
        remaining = sorted(set(_doc_ids(vertex)) - {doc_id})
        if remaining:
            props = _remove_doc_contribution(vertex, doc_id)
            if not _replace_item_properties(vertex, props, edge=False):
                raise RuntimeError(f"裁剪共享结构化实体失败: {vertex.get('id')}")
        elif not hugegraph_client.delete_vertex(str(vertex.get("id") or "")):
            raise RuntimeError(f"清理上一版结构化实体失败: {vertex.get('id')}")


def _restore_document_snapshot(
    doc_id: str,
    previous_vertices: Iterable[Mapping[str, Any]],
    previous_edges: Iterable[Mapping[str, Any]],
) -> None:
    """Best-effort compensation after a partial delete/write failure."""

    snapshot_vertices = [copy.deepcopy(dict(item)) for item in previous_vertices]
    snapshot_edges = [copy.deepcopy(dict(item)) for item in previous_edges]
    current_vertices, current_edges = _read_structured_items()
    _remove_existing_contribution(doc_id, current_vertices, current_edges)
    remaining_vertices, remaining_edges = _read_structured_items()
    existing = {str(item.get("id")): item for item in remaining_vertices}
    inserts: List[Dict[str, Any]] = []
    for vertex in snapshot_vertices:
        current = existing.get(str(vertex.get("id") or ""))
        if current:
            if not _replace_item_properties(
                current, vertex.get("properties") or {}, edge=False
            ):
                raise RuntimeError(f"恢复上一版共享实体失败: {vertex.get('id')}")
        else:
            inserts.append(vertex)
    existing_edges = {
        _edge_identity(item): item for item in remaining_edges
    }
    edge_inserts: List[Dict[str, Any]] = []
    for edge in snapshot_edges:
        current = existing_edges.get(_edge_identity(edge))
        if current:
            if not _replace_item_properties(
                current, edge.get("properties") or {}, edge=True
            ):
                raise RuntimeError(
                    "恢复上一版共享关系失败: " + repr(_edge_identity(edge))
                )
        else:
            edge_inserts.append(edge)
    if (inserts or edge_inserts) and not hugegraph_client.batch_insert(
        inserts, edge_inserts
    ):
        raise RuntimeError("恢复上一版结构化图失败")


def _init_projection_table() -> None:
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS icp_structured_graph_projections (
                    doc_id VARCHAR(36) PRIMARY KEY REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
                    doc_type VARCHAR(32) NOT NULL,
                    source_hash CHAR(64) NOT NULL,
                    projector_version VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL CHECK (status IN ('running','published','failed')),
                    vertex_count INTEGER NOT NULL DEFAULT 0,
                    edge_count INTEGER NOT NULL DEFAULT 0,
                    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    error TEXT,
                    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        postgres_client.conn.commit()
    except Exception:
        postgres_client.conn.rollback()
        raise


def get_structured_projection(doc_id: str) -> Dict[str, Any] | None:
    """Read the PostgreSQL projection state for one authoritative document."""

    _init_projection_table()
    with postgres_client.conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id,doc_type,source_hash,projector_version,status,
                   vertex_count,edge_count,warnings,error,
                   started_at,completed_at,updated_at
            FROM icp_structured_graph_projections WHERE doc_id=%s
            """,
            (doc_id,),
        )
        row = cur.fetchone()
        return _rows(cur, [row])[0] if row else None


def _mark_projection(
    doc_id: str,
    doc_type: str,
    source_hash: str,
    status: str,
    *,
    vertex_count: int = 0,
    edge_count: int = 0,
    warnings: Iterable[Mapping[str, Any]] = (),
    error: str = "",
) -> None:
    if status not in {"running", "published", "failed"}:
        raise ValueError(f"unsupported structured projection status: {status}")
    _init_projection_table()
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icp_structured_graph_projections
                    (doc_id,doc_type,source_hash,projector_version,status,
                     vertex_count,edge_count,warnings,error,completed_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                        CASE WHEN %s='running' THEN NULL ELSE CURRENT_TIMESTAMP END)
                ON CONFLICT (doc_id) DO UPDATE SET
                    doc_type=EXCLUDED.doc_type, source_hash=EXCLUDED.source_hash,
                    projector_version=EXCLUDED.projector_version, status=EXCLUDED.status,
                    vertex_count=EXCLUDED.vertex_count, edge_count=EXCLUDED.edge_count,
                    warnings=EXCLUDED.warnings, error=EXCLUDED.error,
                    started_at=CASE WHEN EXCLUDED.status='running' THEN CURRENT_TIMESTAMP
                                    ELSE icp_structured_graph_projections.started_at END,
                    completed_at=EXCLUDED.completed_at, updated_at=CURRENT_TIMESTAMP
                """,
                (
                    doc_id,
                    doc_type,
                    source_hash,
                    PROJECTOR_VERSION,
                    status,
                    vertex_count,
                    edge_count,
                    json.dumps(list(warnings), ensure_ascii=False, default=str),
                    error[:4000] or None,
                    status,
                ),
            )
        postgres_client.conn.commit()
    except Exception:
        postgres_client.conn.rollback()
        raise


def _mark_failed_safely(
    doc_id: str,
    doc_type: str,
    source_hash: str,
    warnings: Iterable[Mapping[str, Any]],
    error: Exception,
) -> None:
    try:
        _mark_projection(
            doc_id,
            doc_type,
            source_hash,
            "failed",
            warnings=warnings,
            error=str(error),
        )
    except Exception:
        # Preserve the original failure; a PostgreSQL outage can make both the
        # authoritative read and the best-effort failure audit unavailable.
        pass


def project_structured_document(doc_id: str, doc_type: str) -> Dict[str, Any]:
    """Rebuild and verify one document's explicit HugeGraph contribution."""

    doc_id = str(doc_id or "").strip()
    doc_type = str(doc_type or "").strip().casefold()
    if not doc_id:
        raise ValueError("doc_id must not be empty")
    if doc_type not in PROJECTABLE_DOC_TYPES:
        raise ValueError(f"unsupported ICP document type: {doc_type}")
    if getattr(hugegraph_client, "_use_mock", False):
        raise RuntimeError("HugeGraph 服务不可用；拒绝将内存 Mock 视为真实投影成功")
    with _projection_lock():
        return _project_structured_document_locked(doc_id, doc_type)


def _project_structured_document_locked(doc_id: str, doc_type: str) -> Dict[str, Any]:
    source_hash = content_hash(
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "projector_version": PROJECTOR_VERSION,
            "source": "unbuilt",
        }
    )
    warnings: List[Mapping[str, Any]] = []
    auditable_attempt = False
    mutation_started = False
    previous_vertices: List[Dict[str, Any]] = []
    previous_edges: List[Dict[str, Any]] = []
    try:
        graph = build_document_graph(doc_id, doc_type)
        source_hash = content_hash(graph)
        warnings = list(graph.get("warnings") or [])
        auditable_attempt = True
        document = postgres_client.get_icp_document(doc_id) or {}
        document_metadata = document.get("metadata") or {}
        if isinstance(document_metadata, str):
            try:
                document_metadata = json.loads(document_metadata)
            except (TypeError, ValueError):
                document_metadata = {}
        entity_snapshot = entity_resolution_repository.sync_structured_graph(
            doc_id=doc_id,
            doc_type=doc_type,
            graph=graph,
            document_metadata=(
                document_metadata if isinstance(document_metadata, Mapping) else {}
            ),
        )
        skipped_edges = list(entity_snapshot.get("skipped_edges") or [])
        if skipped_edges:
            raise RuntimeError(
                "结构化关系存在未注册本体类型，拒绝发布不可再授权的边: "
                + ", ".join(
                    sorted(
                        {
                            str(item.get("relation") or "unknown")
                            for item in skipped_edges
                            if isinstance(item, Mapping)
                        }
                    )
                )
            )
        vertices, edges = _physical_graph(
            graph,
            doc_id=doc_id,
            doc_type=doc_type,
            source_hash=source_hash,
            logical_entity_ids=entity_snapshot.get("logical_entity_ids") or {},
            identity_scope=entity_snapshot.get("identity_scope") or {},
        )
        state = get_structured_projection(doc_id)
        if (
            state
            and str(state.get("status") or "") == "published"
            and str(state.get("source_hash") or "") == source_hash
            and str(state.get("projector_version") or "") == PROJECTOR_VERSION
        ):
            matches, actual_vertices, actual_edges = _readback_matches(
                doc_id, vertices, edges
            )
            if matches:
                return {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "source_hash": source_hash,
                    "projector_version": PROJECTOR_VERSION,
                    "status": "published",
                    "vertices": actual_vertices,
                    "edges": actual_edges,
                    "warnings": warnings,
                    "entity_resolution": entity_snapshot,
                    "reused": True,
                }

        _mark_projection(doc_id, doc_type, source_hash, "running", warnings=warnings)
        stored_vertices, stored_edges = _read_structured_items()
        previous_vertices = [
            copy.deepcopy(item) for item in stored_vertices if _has_doc(item, doc_id)
        ]
        previous_edges = [
            copy.deepcopy(item) for item in stored_edges if _has_doc(item, doc_id)
        ]
        mutation_started = True
        _remove_existing_contribution(
            doc_id, stored_vertices=stored_vertices, stored_edges=stored_edges
        )
        remaining_vertices, remaining_edges = _read_structured_items()
        existing = {
            str(item.get("id")): item
            for item in remaining_vertices
        }
        inserts: List[Dict[str, Any]] = []
        for vertex in vertices:
            existing_vertex = existing.get(vertex["id"])
            if existing_vertex:
                merged = _merge_vertex_properties(existing_vertex, vertex, doc_id)
                if not _replace_item_properties(existing_vertex, merged, edge=False):
                    raise RuntimeError(f"更新共享实体失败: {vertex['id']}")
            else:
                inserts.append(vertex)
        existing_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for stored_edge in remaining_edges:
            identity = _edge_identity(stored_edge)
            if identity in existing_edges:
                raise RuntimeError(
                    "HugeGraph contains duplicate SINGLE relations for "
                    + repr(identity)
                )
            existing_edges[identity] = stored_edge
        edge_inserts: List[Dict[str, Any]] = []
        for edge in edges:
            existing_edge = existing_edges.get(_edge_identity(edge))
            if existing_edge:
                merged = _merge_edge_properties(existing_edge, edge, doc_id)
                if not _replace_item_properties(existing_edge, merged, edge=True):
                    raise RuntimeError(
                        "更新共享结构化关系失败: "
                        + repr(_edge_identity(edge))
                    )
            else:
                edge_inserts.append(edge)
        if not hugegraph_client.batch_insert(inserts, edge_inserts):
            raise RuntimeError("HugeGraph 结构化图批量写入失败")

        matches, actual_vertices, actual_edges = _readback_matches(doc_id, vertices, edges)
        if not matches:
            raise RuntimeError(
                "HugeGraph 结构化投影写后读回不一致: "
                f"{actual_vertices}/{len(vertices)} 节点, {actual_edges}/{len(edges)} 边"
            )
        _mark_projection(
            doc_id,
            doc_type,
            source_hash,
            "published",
            vertex_count=actual_vertices,
            edge_count=actual_edges,
            warnings=warnings,
        )
        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_hash": source_hash,
            "projector_version": PROJECTOR_VERSION,
            "status": "published",
            "vertices": actual_vertices,
            "edges": actual_edges,
            "warnings": warnings,
            "entity_resolution": entity_snapshot,
            "reused": False,
        }
    except Exception as exc:
        failure = exc
        if mutation_started:
            try:
                _restore_document_snapshot(
                    doc_id,
                    previous_vertices=previous_vertices,
                    previous_edges=previous_edges,
                )
            except Exception as rollback_exc:
                failure = RuntimeError(
                    f"{exc}; HugeGraph 补偿失败: {rollback_exc}"
                )
        if auditable_attempt:
            _mark_failed_safely(doc_id, doc_type, source_hash, warnings, failure)
        raise


__all__ = [
    "PROJECTOR_VERSION",
    "PROJECTABLE_DOC_TYPES",
    "STRUCTURED_KB_ID",
    "build_document_graph",
    "get_structured_projection",
    "load_formal_diagram_revisions",
    "load_structured_rows",
    "project_structured_document",
]
