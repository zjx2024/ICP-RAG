"""Evidence-preserving, fail-closed traversal of ICP HugeGraph projections.

HugeGraph is a rebuildable projection, not an authority.  This module therefore
does not expose the client's legacy, unfiltered neighbour helpers.  It reads the
current physical projection, accepts only deterministic structured relations or
formally validated engineering-diagram relations, and returns every path with
its document/revision provenance.

The public entry point is :func:`run_graph_traversal`.  A PostgreSQL-backed
``reauthorize_relations`` callback can be supplied at integration time.  The
callback receives stable relation-assertion descriptors and returns decisions
keyed by ``authorization_key`` (or, for convenience, physical/logical edge ID).
Any callback/backend exception closes the whole query and returns no graph data.
"""

from __future__ import annotations

import hashlib
import json
import asyncio
import time
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Sequence

from backend.agent.contracts import make_evidence, make_tool_result
from backend.icp.diagram_validation import FORMAL_VALIDATION_STATUSES


TraversalMode = Literal["neighbors", "path", "subgraph"]
TraversalDirection = Literal["out", "in", "both"]
RelationDecision = bool | Mapping[str, Any]
RelationReauthorizer = Callable[
    [Sequence[Mapping[str, Any]]], Mapping[str, RelationDecision]
]

STRUCTURED_KB_ID = "icp_structured"

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_NODES = 40
DEFAULT_MAX_EDGES = 80
DEFAULT_MAX_PATHS = 12
DEFAULT_INFERENCE_EXCLUDED_RELATIONS = frozenset({"extracted_from"})

HARD_MAX_DEPTH = 5
HARD_MAX_NODES = 200
HARD_MAX_EDGES = 400
HARD_MAX_PATHS = 50
HARD_MAX_SEEDS = 20
_MAX_PATH_STATES = 5000


class GraphTraversalValidationError(ValueError):
    """Raised when a traversal request exceeds the closed safety contract."""


def _text(value: Any) -> str:
    if isinstance(value, Mapping) and set(value).issubset({"value", "id"}):
        value = value.get("value", value.get("id"))
    return str(value or "").strip()


def _list(value: Any) -> List[str]:
    if isinstance(value, Mapping) and "value" in value:
        value = value.get("value")
    if isinstance(value, str):
        # Current properties are LIST values.  Comma splitting preserves
        # compatibility with older HugeGraph readbacks.
        values: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    elif value in (None, ""):
        values = ()
    else:
        values = (value,)
    return sorted({_text(item) for item in values if _text(item)})


def _properties(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("properties") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _collect_named_values(
    value: Any,
    names: frozenset[str],
    *,
    depth: int = 0,
) -> List[str]:
    """Collect bounded provenance fields from serialized projection payloads."""

    if depth > 5:
        return []
    result: List[str] = []
    if isinstance(value, Mapping):
        for key, child in list(value.items())[:100]:
            if str(key) in names:
                result.extend(_list(child))
            elif isinstance(child, (Mapping, list, tuple)):
                result.extend(
                    _collect_named_values(child, names, depth=depth + 1)
                )
    elif isinstance(value, (list, tuple)):
        for child in list(value)[:100]:
            result.extend(_collect_named_values(child, names, depth=depth + 1))
    return sorted(set(result))


def _serialized_projection_values(
    properties: Mapping[str, Any], names: frozenset[str]
) -> List[str]:
    values: List[str] = []
    for field in ("summary", "description_parts"):
        raw = properties.get(field)
        # Shared structured vertices concatenate summaries using the legacy
        # separator.  Each contribution remains independently parseable.
        parts = str(raw or "").split("<SEP>") if field == "summary" else [raw]
        for part in parts:
            parsed = _json(part)
            if parsed is None:
                continue
            values.extend(_collect_named_values(parsed, names))
            if field == "description_parts" and isinstance(parsed, list):
                for contribution in parsed[:100]:
                    if not isinstance(contribution, Mapping):
                        continue
                    nested = _json(contribution.get("summary"))
                    if nested is not None:
                        values.extend(_collect_named_values(nested, names))
    return sorted(set(values))


def _marker_values(source_ids: Sequence[str], prefix: str) -> List[str]:
    return sorted(
        {
            value[len(prefix) :]
            for value in source_ids
            if value.startswith(prefix) and value[len(prefix) :]
        }
    )


def _logical_ids(item_id: str, source_ids: Sequence[str], kind: str) -> List[str]:
    markers = _marker_values(source_ids, "logical:")
    if markers:
        return markers
    if kind == "vertex" and ":v:" in item_id:
        return [item_id.split(":v:", 1)[1]]
    if kind == "edge" and ":e:" in item_id:
        return [item_id.split(":e:", 1)[1]]
    return []


def _logical_ids_by_doc(
    properties: Mapping[str, Any],
    *,
    logical_ids: Sequence[str],
    doc_ids: Sequence[str],
) -> Dict[str, List[str]]:
    """Recover the doc-owned logical markers from v5 contribution records."""

    result: Dict[str, set[str]] = {}
    contributions = _json(properties.get("description_parts"))
    if isinstance(contributions, list):
        for raw in contributions[:10000]:
            if not isinstance(raw, Mapping):
                continue
            doc_id = _text(raw.get("doc_id"))
            logical_id = _text(raw.get("logical_id") or raw.get("logical_edge_id"))
            if doc_id and logical_id:
                result.setdefault(doc_id, set()).add(logical_id)
    # v4 single-document projections remain attributable.  A multi-document
    # item without doc-owned contribution markers is ambiguous and is rejected
    # by the structured-edge checks below until every owner is rebuilt.
    if len(doc_ids) == 1 and not result.get(doc_ids[0]) and logical_ids:
        result[doc_ids[0]] = set(logical_ids)
    return {
        doc_id: sorted(values)
        for doc_id, values in sorted(result.items())
        if values
    }


def _contribution_values_by_doc(
    properties: Mapping[str, Any], key: str
) -> Dict[str, List[str]]:
    result: Dict[str, set[str]] = {}
    contributions = _json(properties.get("description_parts"))
    if not isinstance(contributions, list):
        return {}
    for raw in contributions[:10000]:
        if not isinstance(raw, Mapping):
            continue
        doc_id = _text(raw.get("doc_id"))
        values = _list(raw.get(key))
        if doc_id and values:
            result.setdefault(doc_id, set()).update(values)
    return {
        doc_id: sorted(values)
        for doc_id, values in sorted(result.items())
        if values
    }


def _projection_candidate(
    raw: Mapping[str, Any], *, kind: Literal["vertex", "edge"], kb_id: str
) -> Dict[str, Any] | None:
    item_id = _text(raw.get("id"))
    props = _properties(raw)
    if not item_id or _text(props.get("kb_id")) != kb_id:
        return None

    source_ids = _list(props.get("source_id"))
    doc_ids = sorted(
        set(_list(props.get("doc_id")))
        | set(_serialized_projection_values(props, frozenset({"doc_id", "doc_ids"})))
    )
    if not source_ids or not doc_ids:
        return None

    logical_ids = _logical_ids(item_id, source_ids, kind)
    if not logical_ids:
        return None
    logical_ids_by_doc = _logical_ids_by_doc(
        props, logical_ids=logical_ids, doc_ids=doc_ids
    )
    revision_ids_by_doc = _contribution_values_by_doc(props, "revision_id")

    validation_markers = _marker_values(source_ids, "validation:")
    revision_markers = _marker_values(source_ids, "revision:")
    canonical_markers = _marker_values(source_ids, "canonical:")
    revision_id = _text(props.get("tenant_id"))

    structured = kb_id == STRUCTURED_KB_ID
    if structured:
        if kind == "vertex" and not item_id.startswith("icp:ce:"):
            return None
        structured_docs = set(_marker_values(source_ids, "structured-doc:"))
        projection_docs = {
            marker.split(":", 1)[0]
            for marker in _marker_values(source_ids, "projection:")
            if marker
        }
        if not set(doc_ids).issubset(structured_docs):
            return None
        if not set(doc_ids).issubset(projection_docs):
            return None
        if kind == "edge":
            contribution_logical_ids = {
                logical_id
                for values in logical_ids_by_doc.values()
                for logical_id in values
            }
            if (
                set(logical_ids_by_doc) != set(doc_ids)
                or contribution_logical_ids != set(logical_ids)
                or any(not value.startswith("sg:e:") for value in logical_ids)
            ):
                return None
        validation_status = "auto_validated"
        formal_basis = "explicit_postgres_projection"
        requires_reauthorization = False
        revision_id = ""
    else:
        # A validation marker alone is insufficient to prove that a diagram is
        # still active/published.  Its relation must be authorized by the PG
        # callback below before it can enter a traversal.
        if (
            len(validation_markers) != 1
            or validation_markers[0] not in FORMAL_VALIDATION_STATUSES
            or len(revision_markers) != 1
            or revision_markers[0] != revision_id
            or not canonical_markers
            or not item_id.startswith(f"icp:{revision_id}:")
        ):
            return None
        validation_status = validation_markers[0]
        formal_basis = "validated_engineering_projection"
        requires_reauthorization = True

    source_refs = sorted(
        set(
            _serialized_projection_values(
                props, frozenset({"source_ref", "source_refs"})
            )
        )
    )
    doc_types = sorted(
        set(
            _serialized_projection_values(
                props, frozenset({"doc_type", "source_doc_types"})
            )
        )
    )
    candidate: Dict[str, Any] = {
        "authorization_key": f"{kind}:{item_id}",
        "id": item_id,
        "logical_id": logical_ids[0] if len(logical_ids) == 1 else "",
        "logical_ids": logical_ids,
        "logical_ids_by_doc": logical_ids_by_doc,
        "kind": kind,
        "kb_id": kb_id,
        "label": _text(raw.get("label")),
        "name": _text(props.get("name")),
        "description": _text(props.get("description")),
        "entity_type": _text(props.get("entity_type")),
        "business_id": "",
        "identity_scope": "",
        "doc_ids": doc_ids,
        "doc_types": doc_types,
        "source_ids": source_ids,
        "source_refs": source_refs,
        # These values came from the rebuildable HugeGraph projection.  Keep
        # them for locating the projected item, but never silently promote
        # them to a PostgreSQL-authorized relationship citation below.
        "projection_doc_ids": list(doc_ids),
        "projection_source_refs": list(source_refs),
        "authorized_doc_ids": [],
        "authorized_source_refs": [],
        "revision_id": revision_id,
        "revision_ids": sorted(
            {
                value
                for values in revision_ids_by_doc.values()
                for value in values
            }
        ),
        "revision_ids_by_doc": revision_ids_by_doc,
        "validation_status": validation_status,
        "projection_status": "published" if structured else "",
        "is_active": True if structured else False,
        "formal_basis": formal_basis,
        "reauthorized": False,
        "requires_reauthorization": requires_reauthorization,
    }

    summary_parts = str(props.get("summary") or "").split("<SEP>")
    for part in summary_parts[:100]:
        summary = _json(part)
        if not isinstance(summary, Mapping):
            continue
        candidate["business_id"] = candidate["business_id"] or _text(
            summary.get("business_id")
        )
        candidate["identity_scope"] = candidate["identity_scope"] or _text(
            summary.get("identity_scope")
        )

    if kind == "edge":
        source = _text(raw.get("outV") or raw.get("source"))
        target = _text(raw.get("inV") or raw.get("target"))
        relation = _text(raw.get("label") or props.get("relation_type"))
        if not source or not target or not relation:
            return None
        if structured and (
            not source.startswith("icp:ce:") or not target.startswith("icp:ce:")
        ):
            return None
        candidate.update(
            {
                "source": source,
                "target": target,
                "relation": relation,
            }
        )
    return candidate


def _authorization_request(
    edge: Mapping[str, Any], vertices: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    source = vertices.get(_text(edge.get("source")), {})
    target = vertices.get(_text(edge.get("target")), {})
    return {
        "authorization_key": edge["authorization_key"],
        "physical_edge_id": edge["id"],
        "logical_edge_id": edge["logical_id"],
        "logical_edge_ids": list(edge.get("logical_ids") or []),
        "logical_edge_ids_by_doc": dict(edge.get("logical_ids_by_doc") or {}),
        "relation": edge["relation"],
        "source_physical_id": edge["source"],
        "source_logical_id": _text(source.get("logical_id")),
        "source_logical_ids": list(source.get("logical_ids") or []),
        "source_logical_ids_by_doc": dict(
            source.get("logical_ids_by_doc") or {}
        ),
        "target_physical_id": edge["target"],
        "target_logical_id": _text(target.get("logical_id")),
        "target_logical_ids": list(target.get("logical_ids") or []),
        "target_logical_ids_by_doc": dict(
            target.get("logical_ids_by_doc") or {}
        ),
        "kb_id": edge["kb_id"],
        "doc_ids": list(
            edge.get("projection_doc_ids") or edge.get("doc_ids") or []
        ),
        "source_refs": list(
            edge.get("projection_source_refs") or edge.get("source_refs") or []
        ),
        "revision_id": edge.get("revision_id") or "",
        "revision_ids": list(edge.get("revision_ids") or []),
        "revision_ids_by_doc": dict(edge.get("revision_ids_by_doc") or {}),
        "validation_status": edge.get("validation_status") or "",
        "formal_basis": edge.get("formal_basis") or "",
    }


def _decision_for(
    decisions: Mapping[str, RelationDecision], edge: Mapping[str, Any]
) -> RelationDecision | None:
    keys = [edge.get("authorization_key"), edge.get("id")]
    # A decision keyed by one logical occurrence must never authorize an
    # aggregate physical edge carrying additional occurrences.
    if len(edge.get("logical_ids") or []) == 1:
        keys.append(edge.get("logical_id"))
    for key in keys:
        if key in decisions:
            return decisions[key]
    return None


def _apply_decision(
    edge: Dict[str, Any], decision: RelationDecision | None, *, required: bool
) -> bool:
    if decision is None:
        if required:
            return False
        # The edge may still be returned for an explicitly non-authoritative
        # caller, but its projection provenance is not an authorized citation.
        edge["doc_ids"] = []
        edge["source_refs"] = []
        edge["authorized_doc_ids"] = []
        edge["authorized_source_refs"] = []
        return True
    if isinstance(decision, bool):
        if not decision:
            return False
        # Engineering projections need explicit current-revision state, not a
        # bare boolean acknowledgement.
        if edge.get("requires_reauthorization"):
            return False
        edge["reauthorized"] = True
        # A bare boolean decision supplies no authoritative locator.
        edge["doc_ids"] = []
        edge["source_refs"] = []
        edge["authorized_doc_ids"] = []
        edge["authorized_source_refs"] = []
        return True
    if not isinstance(decision, Mapping) or decision.get("authorized") is not True:
        return False

    validation_status = _text(
        decision.get("validation_status") or edge.get("validation_status")
    )
    projection_status = _text(
        decision.get("projection_status") or edge.get("projection_status")
    )
    is_active = decision.get("is_active", edge.get("is_active")) is True
    if edge.get("requires_reauthorization") and (
        validation_status not in FORMAL_VALIDATION_STATUSES
        or projection_status != "published"
        or not is_active
    ):
        return False

    edge["validation_status"] = validation_status or "auto_validated"
    edge["projection_status"] = projection_status or "published"
    edge["is_active"] = is_active
    edge["reauthorized"] = True
    authorized_doc_ids = sorted(set(_list(decision.get("doc_ids"))))
    authorized_source_refs = sorted(
        set(_list(decision.get("source_refs") or decision.get("source_ref")))
    )
    # Replace, rather than union with, projection claims.  The callback's
    # PostgreSQL decision is the only authority for a formal path citation.
    edge["authorized_doc_ids"] = authorized_doc_ids
    edge["authorized_source_refs"] = authorized_source_refs
    edge["doc_ids"] = list(authorized_doc_ids)
    edge["source_refs"] = list(authorized_source_refs)
    return True


def _bounded_int(name: str, value: Any, default: int, hard_max: int) -> int:
    if value in (None, ""):
        result = default
    elif isinstance(value, bool):
        raise GraphTraversalValidationError(f"{name} 必须为正整数")
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise GraphTraversalValidationError(f"{name} 必须为正整数") from exc
    if result < 1 or result > hard_max:
        raise GraphTraversalValidationError(
            f"{name} 必须在 1..{hard_max} 范围内"
        )
    return result


def _validated_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    mode = _text(request.get("mode") or "neighbors").casefold()
    if mode not in {"neighbors", "path", "subgraph"}:
        raise GraphTraversalValidationError("mode 仅支持 neighbors/path/subgraph")
    direction = _text(request.get("direction") or "both").casefold()
    if direction not in {"out", "in", "both"}:
        raise GraphTraversalValidationError("direction 仅支持 out/in/both")

    kb_ids = _list(request.get("kb_ids") or request.get("kb_id") or STRUCTURED_KB_ID)
    seeds = _list(
        request.get("seed_ids")
        or request.get("source_ids")
        or request.get("start_ids")
    )
    targets = _list(request.get("target_ids"))
    if not kb_ids:
        raise GraphTraversalValidationError("kb_ids 不能为空")
    if not seeds:
        raise GraphTraversalValidationError("seed_ids 不能为空")
    if len(seeds) > HARD_MAX_SEEDS or len(targets) > HARD_MAX_SEEDS:
        raise GraphTraversalValidationError(
            f"seed_ids/target_ids 各不能超过 {HARD_MAX_SEEDS} 个"
        )
    if mode == "path" and not targets:
        raise GraphTraversalValidationError("path 模式必须提供 target_ids")

    relations = _list(request.get("relations"))
    excluded_relations = set(_list(request.get("excluded_relations")))
    # ``extracted_from`` is provenance, not a domain relationship.  Following
    # it by default creates artificial Entity -> Document -> unrelated Entity
    # paths.  A dedicated provenance query may opt in by explicitly selecting
    # that relation in ``relations``.
    if not relations:
        excluded_relations.update(DEFAULT_INFERENCE_EXCLUDED_RELATIONS)

    return {
        "mode": mode,
        "direction": direction,
        "kb_ids": kb_ids,
        "seed_ids": seeds,
        "target_ids": targets,
        "relations": relations,
        "excluded_relations": sorted(excluded_relations),
        "max_depth": _bounded_int(
            "max_depth", request.get("max_depth"), DEFAULT_MAX_DEPTH, HARD_MAX_DEPTH
        ),
        "max_nodes": _bounded_int(
            "max_nodes", request.get("max_nodes"), DEFAULT_MAX_NODES, HARD_MAX_NODES
        ),
        "max_edges": _bounded_int(
            "max_edges", request.get("max_edges"), DEFAULT_MAX_EDGES, HARD_MAX_EDGES
        ),
        "max_paths": _bounded_int(
            "max_paths", request.get("max_paths"), DEFAULT_MAX_PATHS, HARD_MAX_PATHS
        ),
        "require_relation_reauthorization": bool(
            request.get("require_relation_reauthorization", False)
        ),
    }


def _empty_data() -> Dict[str, Any]:
    return {"nodes": [], "edges": [], "paths": [], "evidence": []}


def _result(
    *,
    status: str,
    request: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    rejected: int = 0,
    truncated: bool = False,
) -> Dict[str, Any]:
    return {
        "status": status,
        "mode": (request or {}).get("mode", ""),
        "request": dict(request or {}),
        **dict(data or _empty_data()),
        "truncated": bool(truncated),
        "diagnostics": {
            "backend": "hugegraph+postgres_reauthorization",
            "warnings": list(dict.fromkeys(_text(item) for item in warnings if _text(item))),
            "errors": list(dict.fromkeys(_text(item) for item in errors if _text(item))),
            "rejected_items": int(rejected),
        },
    }


def _public_node(node: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: node.get(key)
        for key in (
            "id",
            "logical_id",
            "logical_ids",
            "label",
            "name",
            "description",
            "entity_type",
            "business_id",
            "identity_scope",
            "kb_id",
            "doc_ids",
            "doc_types",
            "source_ids",
            "source_refs",
            "authorized_doc_ids",
            "authorized_source_refs",
            "projection_doc_ids",
            "projection_source_refs",
            "revision_id",
            "revision_ids",
            "validation_status",
            "projection_status",
            "is_active",
            "formal_basis",
        )
    }


def _public_edge(edge: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: edge.get(key)
        for key in (
            "id",
            "logical_id",
            "logical_ids",
            "relation",
            "source",
            "target",
            "kb_id",
            "doc_ids",
            "doc_types",
            "source_ids",
            "source_refs",
            "revision_id",
            "revision_ids",
            "validation_status",
            "projection_status",
            "is_active",
            "formal_basis",
            "reauthorized",
        )
    }


def _adjacency(
    vertices: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]],
    *,
    direction: str,
    relations: Sequence[str],
    excluded_relations: Sequence[str] = (),
) -> Dict[str, List[Dict[str, Any]]]:
    allowed = {value.casefold() for value in relations}
    excluded = {value.casefold() for value in excluded_relations}
    result: Dict[str, List[Dict[str, Any]]] = {node_id: [] for node_id in vertices}
    for edge in edges.values():
        relation = _text(edge.get("relation")).casefold()
        if relation in excluded or (allowed and relation not in allowed):
            continue
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        if source not in vertices or target not in vertices:
            continue
        if direction in {"out", "both"}:
            result[source].append(
                {"neighbor": target, "edge_id": edge["id"], "orientation": "forward"}
            )
        if direction in {"in", "both"}:
            result[target].append(
                {"neighbor": source, "edge_id": edge["id"], "orientation": "reverse"}
            )
    for node_id in result:
        result[node_id].sort(
            key=lambda hop: (
                _text(edges[hop["edge_id"]].get("relation")).casefold(),
                hop["neighbor"],
                hop["edge_id"],
                hop["orientation"],
            )
        )
    return result


def _path_record(
    node_ids: Sequence[str],
    hops: Sequence[Mapping[str, Any]],
    vertices: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    doc_ids: set[str] = set()
    source_refs: set[str] = set()
    projection_doc_ids: set[str] = set()
    projection_source_refs: set[str] = set()
    endpoint_doc_ids: set[str] = set()
    endpoint_source_refs: set[str] = set()
    statuses: set[str] = set()
    for index, hop in enumerate(hops):
        edge = edges[hop["edge_id"]]
        current = node_ids[index]
        neighbor = node_ids[index + 1]
        reauthorized = edge.get("reauthorized") is True
        authorized_doc_ids = (
            list(edge.get("authorized_doc_ids") or edge.get("doc_ids") or [])
            if reauthorized
            else []
        )
        authorized_source_refs = (
            list(
                edge.get("authorized_source_refs")
                or edge.get("source_refs")
                or []
            )
            if reauthorized
            else []
        )
        edge_projection_doc_ids = list(edge.get("projection_doc_ids") or [])
        edge_projection_source_refs = list(
            edge.get("projection_source_refs") or []
        )
        step = {
            "source": current,
            "source_logical_id": vertices[current].get("logical_id") or "",
            "source_logical_ids": list(vertices[current].get("logical_ids") or []),
            "relation": edge["relation"],
            "target": neighbor,
            "target_logical_id": vertices[neighbor].get("logical_id") or "",
            "target_logical_ids": list(vertices[neighbor].get("logical_ids") or []),
            "edge_id": edge["id"],
            "logical_edge_id": edge.get("logical_id") or "",
            "logical_edge_ids": list(edge.get("logical_ids") or []),
            "edge_source": edge["source"],
            "edge_target": edge["target"],
            "traversal_direction": hop["orientation"],
            "doc_ids": authorized_doc_ids,
            "source_refs": authorized_source_refs,
            "projection_doc_ids": edge_projection_doc_ids,
            "projection_source_refs": edge_projection_source_refs,
            "reauthorized": reauthorized,
            "revision_id": edge.get("revision_id") or "",
            "revision_ids": list(edge.get("revision_ids") or []),
            "validation_status": edge.get("validation_status") or "",
        }
        steps.append(step)
        doc_ids.update(step["doc_ids"])
        source_refs.update(step["source_refs"])
        projection_doc_ids.update(step["projection_doc_ids"])
        projection_source_refs.update(step["projection_source_refs"])
        if step["validation_status"]:
            statuses.add(step["validation_status"])
    for node_id in node_ids:
        endpoint_doc_ids.update(vertices[node_id].get("doc_ids") or [])
        endpoint_source_refs.update(vertices[node_id].get("source_refs") or [])

    identity = json.dumps(
        {
            "nodes": list(node_ids),
            "steps": [
                [item["edge_id"], item["traversal_direction"]] for item in steps
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "path_id": "gp_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        "depth": len(steps),
        "start_id": node_ids[0],
        "end_id": node_ids[-1],
        "node_ids": list(node_ids),
        "edge_ids": [item["edge_id"] for item in steps],
        "steps": steps,
        "doc_ids": sorted(doc_ids),
        "source_refs": sorted(source_refs),
        "projection_doc_ids": sorted(projection_doc_ids),
        "projection_source_refs": sorted(projection_source_refs),
        "endpoint_doc_ids": sorted(endpoint_doc_ids),
        "endpoint_source_refs": sorted(endpoint_source_refs),
        "reauthorized": bool(steps)
        and all(step.get("reauthorized") is True for step in steps),
        "validation_statuses": sorted(statuses),
    }


def _neighbor_paths(
    seeds: Sequence[str],
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
    vertices: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]],
    *,
    max_depth: int,
    max_nodes: int,
    max_edges: int,
    max_paths: int,
) -> tuple[List[Dict[str, Any]], set[str], set[str], bool]:
    accepted_seeds = [seed for seed in seeds if seed in vertices][:max_nodes]
    visited = set(accepted_seeds)
    tree_edges: set[str] = set()
    best: Dict[str, tuple[List[str], List[Mapping[str, Any]]]] = {
        seed: ([seed], []) for seed in accepted_seeds
    }
    queue = deque((seed, 0) for seed in accepted_seeds)
    truncated = len(accepted_seeds) < len([seed for seed in seeds if seed in vertices])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        node_path, hop_path = best[current]
        for hop in adjacency.get(current, ()):
            neighbor = hop["neighbor"]
            if neighbor in visited:
                continue
            if len(visited) >= max_nodes:
                truncated = True
                continue
            if hop["edge_id"] not in tree_edges and len(tree_edges) >= max_edges:
                truncated = True
                continue
            visited.add(neighbor)
            tree_edges.add(hop["edge_id"])
            best[neighbor] = ([*node_path, neighbor], [*hop_path, hop])
            queue.append((neighbor, depth + 1))
    paths = [
        _path_record(nodes, hops, vertices, edges)
        for node_id, (nodes, hops) in sorted(best.items())
        if node_id not in accepted_seeds and hops
    ]
    paths.sort(key=lambda item: (item["depth"], item["node_ids"], item["edge_ids"]))
    if len(paths) > max_paths:
        paths = paths[:max_paths]
        truncated = True
    return paths, visited, tree_edges, truncated


def _path_search(
    seeds: Sequence[str],
    targets: Sequence[str],
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
    vertices: Mapping[str, Mapping[str, Any]],
    edges: Mapping[str, Mapping[str, Any]],
    *,
    max_depth: int,
    max_nodes: int,
    max_edges: int,
    max_paths: int,
) -> tuple[List[Dict[str, Any]], set[str], set[str], bool]:
    target_set = {value for value in targets if value in vertices}
    queue = deque(
        ([seed], []) for seed in seeds if seed in vertices and seed not in target_set
    )
    explored_nodes = {seed for seed in seeds if seed in vertices}
    explored_edges: set[str] = set()
    records: Dict[str, Dict[str, Any]] = {}
    states = 0
    truncated = False
    while queue and len(records) < max_paths:
        node_path, hop_path = queue.popleft()
        states += 1
        if states > _MAX_PATH_STATES:
            truncated = True
            break
        if len(hop_path) >= max_depth:
            continue
        current = node_path[-1]
        for hop in adjacency.get(current, ()):
            neighbor = hop["neighbor"]
            if neighbor in node_path:
                continue
            if neighbor not in explored_nodes and len(explored_nodes) >= max_nodes:
                truncated = True
                continue
            if hop["edge_id"] not in explored_edges and len(explored_edges) >= max_edges:
                truncated = True
                continue
            explored_nodes.add(neighbor)
            explored_edges.add(hop["edge_id"])
            next_nodes = [*node_path, neighbor]
            next_hops = [*hop_path, hop]
            if neighbor in target_set:
                record = _path_record(next_nodes, next_hops, vertices, edges)
                records[record["path_id"]] = record
                if len(records) >= max_paths:
                    if queue or len(adjacency.get(current, ())) > 1:
                        truncated = True
                    break
            else:
                queue.append((next_nodes, next_hops))
    paths = sorted(
        records.values(),
        key=lambda item: (item["depth"], item["node_ids"], item["edge_ids"]),
    )
    used_nodes = {node for path in paths for node in path["node_ids"]}
    used_edges = {edge for path in paths for edge in path["edge_ids"]}
    return paths, used_nodes, used_edges, truncated


def _path_display(
    path: Mapping[str, Any], vertices: Mapping[str, Mapping[str, Any]]
) -> str:
    steps = path.get("steps") or []
    if not steps:
        return ""
    start_id = _text(path.get("start_id"))
    start = _text(vertices.get(start_id, {}).get("name")) or start_id
    pieces = [start]
    for step in steps:
        target_id = _text(step.get("target"))
        target = _text(vertices.get(target_id, {}).get("name")) or target_id
        relation = _text(step.get("relation")) or "relation"
        arrow = f" —[{relation}]→ " if step.get("traversal_direction") == "forward" else f" ←[{relation}]— "
        pieces.extend([arrow, target])
    return "".join(pieces)


def _path_evidence(
    paths: Sequence[Mapping[str, Any]], vertices: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for path in paths:
        statuses = set(path.get("validation_statuses") or [])
        review_status = (
            "manual_confirmed"
            if statuses == {"manual_confirmed"}
            else "auto_validated"
        )
        doc_ids = list(path.get("doc_ids") or [])
        source_refs = list(path.get("source_refs") or [])
        projection_doc_ids = list(path.get("projection_doc_ids") or [])
        projection_source_refs = list(path.get("projection_source_refs") or [])
        endpoint_doc_ids = list(path.get("endpoint_doc_ids") or [])
        endpoint_source_refs = list(path.get("endpoint_source_refs") or [])
        display = _path_display(path, vertices)
        content = json.dumps(path, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        start_id = _text(path.get("start_id"))
        end_id = _text(path.get("end_id"))
        start_vertex = vertices.get(start_id, {})
        end_vertex = vertices.get(end_id, {})
        path_node_ids = list(path.get("node_ids") or [])
        path_steps = list(path.get("steps") or [])
        answer_fields = {
            "source_entity": _text(start_vertex.get("logical_id")) or start_id,
            "source_name": _text(start_vertex.get("name")) or start_id,
            "target_entity": _text(end_vertex.get("logical_id")) or end_id,
            "target_name": _text(end_vertex.get("name")) or end_id,
            "path_summary": display,
            "path_nodes": [
                _text(vertices.get(node_id, {}).get("name")) or _text(node_id)
                for node_id in path_node_ids
            ],
            "path_relations": [
                {
                    "relation": _text(step.get("relation")),
                    "direction": _text(step.get("traversal_direction")),
                }
                for step in path_steps
            ],
        }
        if len(path_steps) == 1:
            answer_fields["relation"] = _text(path_steps[0].get("relation"))
        result.append(
            make_evidence(
                source_kind="hugegraph",
                authority=(
                    "confirmed_relation"
                    if review_status == "manual_confirmed"
                    else "validated_relation"
                ),
                evidence_level="L2_CORROBORATED",
                claim_type="relation_path",
                doc_id=doc_ids[0] if len(doc_ids) == 1 else "",
                record_id=_text(path.get("path_id")),
                source_ref="；".join(source_refs),
                title=f"关系路径（{path.get('depth', 0)} 跳）",
                content=content,
                display_summary=display,
                match_type="formal_multihop_path",
                review_status=review_status,
                limitations=[
                    "HugeGraph 为 PostgreSQL 权威关系的可重建投影",
                    "端点与投影来源仅供定位，不作为路径关系授权依据",
                ],
                metadata={
                    "validation_status": review_status,
                    "projection_status": "published",
                    "is_active": True,
                    "path_id": path.get("path_id"),
                    "doc_ids": doc_ids,
                    "source_refs": source_refs,
                    "projection_doc_ids": projection_doc_ids,
                    "projection_source_refs": projection_source_refs,
                    "endpoint_doc_ids": endpoint_doc_ids,
                    "endpoint_source_refs": endpoint_source_refs,
                    "reauthorized": path.get("reauthorized") is True,
                    "depth": path.get("depth", 0),
                    "answer_fields": answer_fields,
                },
            )
        )
    return result


def _load_formal_graph(
    request: Mapping[str, Any],
    *,
    client: Any,
    reauthorize_relations: RelationReauthorizer | None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], int, List[str]]:
    if getattr(client, "_use_mock", False):
        raise RuntimeError("HugeGraph 服务不可用；拒绝使用内存 Mock 执行正式关系查询")

    vertex_candidates: Dict[str, Dict[str, Any]] = {}
    edge_candidates: Dict[str, Dict[str, Any]] = {}
    rejected = 0
    for kb_id in request["kb_ids"]:
        raw_vertices = client.get_vertices_by_kb_with_details(kb_id)
        raw_edges = client.get_edges_by_kb_with_details(kb_id)
        if raw_vertices is None or raw_edges is None:
            raise RuntimeError(f"HugeGraph 读取返回无效结果: {kb_id}")
        for raw in raw_vertices:
            if not isinstance(raw, Mapping):
                rejected += 1
                continue
            candidate = _projection_candidate(raw, kind="vertex", kb_id=kb_id)
            if candidate is None:
                rejected += 1
                continue
            existing = vertex_candidates.get(candidate["id"])
            if existing and (
                existing.get("logical_id") != candidate.get("logical_id")
                or existing.get("kb_id") != candidate.get("kb_id")
            ):
                raise RuntimeError(f"HugeGraph 顶点身份冲突: {candidate['id']}")
            vertex_candidates[candidate["id"]] = candidate
        for raw in raw_edges:
            if not isinstance(raw, Mapping):
                rejected += 1
                continue
            candidate = _projection_candidate(raw, kind="edge", kb_id=kb_id)
            if candidate is None:
                rejected += 1
                continue
            existing = edge_candidates.get(candidate["id"])
            if existing and (
                existing.get("logical_id") != candidate.get("logical_id")
                or existing.get("source") != candidate.get("source")
                or existing.get("target") != candidate.get("target")
            ):
                raise RuntimeError(f"HugeGraph 边身份冲突: {candidate['id']}")
            edge_candidates[candidate["id"]] = candidate

    allowed_relations = {
        value.casefold() for value in request.get("relations", ()) if value
    }
    excluded_relations = {
        value.casefold() for value in request.get("excluded_relations", ()) if value
    }
    edge_candidates = {
        edge_id: edge
        for edge_id, edge in edge_candidates.items()
        if _text(edge.get("relation")).casefold() not in excluded_relations
        and (
            not allowed_relations
            or _text(edge.get("relation")).casefold() in allowed_relations
        )
    }

    require_all = bool(request.get("require_relation_reauthorization"))
    if require_all and reauthorize_relations is None:
        raise RuntimeError("关系查询要求 PostgreSQL 再授权，但未配置再授权回调")

    decisions: Mapping[str, RelationDecision] = {}
    if reauthorize_relations is not None and edge_candidates:
        authorization_requests = [
            _authorization_request(edge, vertex_candidates)
            for edge in sorted(edge_candidates.values(), key=lambda item: item["id"])
        ]
        raw_decisions = reauthorize_relations(tuple(authorization_requests))
        if not isinstance(raw_decisions, Mapping):
            raise RuntimeError("关系再授权回调必须返回映射")
        decisions = raw_decisions

    formal_edges: Dict[str, Dict[str, Any]] = {}
    excluded_engineering = 0
    for edge in sorted(edge_candidates.values(), key=lambda item: item["id"]):
        decision = _decision_for(decisions, edge)
        required = require_all or bool(edge.get("requires_reauthorization"))
        if not _apply_decision(edge, decision, required=required):
            rejected += 1
            if edge.get("requires_reauthorization"):
                excluded_engineering += 1
            continue
        source_node = vertex_candidates.get(edge["source"])
        target_node = vertex_candidates.get(edge["target"])
        if source_node is None or target_node is None:
            rejected += 1
            continue
        if edge.get("revision_id") and (
            source_node.get("revision_id") != edge.get("revision_id")
            or target_node.get("revision_id") != edge.get("revision_id")
        ):
            rejected += 1
            continue
        for node in (source_node, target_node):
            if edge.get("requires_reauthorization"):
                node["validation_status"] = edge["validation_status"]
                node["projection_status"] = edge["projection_status"]
                node["is_active"] = edge["is_active"]
                node["reauthorized"] = edge["reauthorized"]
        formal_edges[edge["id"]] = edge

    attached_ids = {
        endpoint
        for edge in formal_edges.values()
        for endpoint in (edge["source"], edge["target"])
    }
    # Keep formal seeds even when they have no accepted edge so an empty result
    # is distinguishable from an unknown physical vertex.  Other isolated nodes
    # cannot contribute to a relation traversal and are not returned.
    attached_ids.update(
        seed for seed in request["seed_ids"] if seed in vertex_candidates
    )
    attached_ids.update(
        target for target in request["target_ids"] if target in vertex_candidates
    )
    formal_vertices = {
        node_id: vertex_candidates[node_id]
        for node_id in sorted(attached_ids)
        if node_id in vertex_candidates
    }
    warnings: List[str] = []
    if excluded_engineering:
        warnings.append(
            f"{excluded_engineering} 条工程图关系未通过当前有效修订再授权，已排除"
        )
    return formal_vertices, formal_edges, rejected, warnings


def _resolve_vertex_ids(
    requested: Sequence[str], vertices: Mapping[str, Mapping[str, Any]]
) -> tuple[List[str], List[str]]:
    """Resolve a physical ID first, then an unambiguous logical projection ID."""

    logical: Dict[str, List[str]] = {}
    for physical_id, vertex in vertices.items():
        for logical_id in vertex.get("logical_ids") or [vertex.get("logical_id")]:
            value = _text(logical_id)
            if value:
                logical.setdefault(value, []).append(physical_id)
    resolved: List[str] = []
    unresolved: List[str] = []
    for value in requested:
        if value in vertices:
            resolved.append(value)
            continue
        candidates = sorted(logical.get(value, []))
        if len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            unresolved.append(value)
    return sorted(set(resolved)), sorted(set(unresolved))


def run_graph_traversal(
    request: Mapping[str, Any],
    *,
    client: Any | None = None,
    reauthorize_relations: RelationReauthorizer | None = None,
) -> Dict[str, Any]:
    """Run a bounded formal-relationship traversal.

    ``seed_ids`` and ``target_ids`` are physical HugeGraph IDs.  Entity linking
    (name/vector to canonical ID) intentionally remains a separate retrieval
    step.  This separation prevents similarity from silently becoming a graph
    identity assertion.
    """

    normalized: Dict[str, Any] | None = None
    try:
        normalized = _validated_request(request)
        if client is None:
            # Lazy import prevents module import from probing the network in
            # focused unit tests and pure planning code.
            from backend.storage.hugegraph_client import hugegraph_client

            client = hugegraph_client
        vertices, edges, rejected, warnings = _load_formal_graph(
            normalized,
            client=client,
            reauthorize_relations=reauthorize_relations,
        )
        resolved_seeds, unresolved_seeds = _resolve_vertex_ids(
            normalized["seed_ids"], vertices
        )
        resolved_targets, unresolved_targets = _resolve_vertex_ids(
            normalized["target_ids"], vertices
        )
        normalized = {
            **normalized,
            "seed_ids": resolved_seeds,
            "target_ids": resolved_targets,
        }
        if unresolved_seeds:
            warnings.append("以下起点不存在或 logical ID 不唯一：" + ", ".join(unresolved_seeds))
        if unresolved_targets:
            warnings.append("以下终点不存在或 logical ID 不唯一：" + ", ".join(unresolved_targets))
        adjacency = _adjacency(
            vertices,
            edges,
            direction=normalized["direction"],
            relations=normalized["relations"],
            excluded_relations=normalized["excluded_relations"],
        )

        if normalized["mode"] == "path":
            paths, used_nodes, used_edges, truncated = _path_search(
                normalized["seed_ids"],
                normalized["target_ids"],
                adjacency,
                vertices,
                edges,
                max_depth=normalized["max_depth"],
                max_nodes=normalized["max_nodes"],
                max_edges=normalized["max_edges"],
                max_paths=normalized["max_paths"],
            )
        else:
            paths, used_nodes, tree_edges, truncated = _neighbor_paths(
                normalized["seed_ids"],
                adjacency,
                vertices,
                edges,
                max_depth=normalized["max_depth"],
                max_nodes=normalized["max_nodes"],
                max_edges=normalized["max_edges"],
                max_paths=normalized["max_paths"],
            )
            if normalized["mode"] == "subgraph":
                eligible_edges = [
                    edge_id
                    for edge_id, edge in sorted(edges.items())
                    if edge["source"] in used_nodes
                    and edge["target"] in used_nodes
                    and (
                        not normalized["relations"]
                        or edge["relation"].casefold()
                        in {value.casefold() for value in normalized["relations"]}
                    )
                    and edge["relation"].casefold()
                    not in {
                        value.casefold()
                        for value in normalized["excluded_relations"]
                    }
                ]
                used_edges = set(eligible_edges[: normalized["max_edges"]])
                truncated = truncated or len(eligible_edges) > len(used_edges)
            else:
                used_edges = tree_edges

        public_vertices = {
            node_id: _public_node(vertices[node_id])
            for node_id in sorted(used_nodes)
            if node_id in vertices
        }
        public_edges = {
            edge_id: _public_edge(edges[edge_id])
            for edge_id in sorted(used_edges)
            if edge_id in edges
        }
        evidence = _path_evidence(paths, public_vertices)
        data = {
            "nodes": list(public_vertices.values()),
            "edges": list(public_edges.values()),
            "paths": paths,
            "evidence": evidence,
        }
        status = "ok" if paths else "empty"
        if truncated and status == "ok":
            status = "partial"
        if not paths:
            warnings.append("在正式关系和查询上限内未找到可用多跳路径")
        return _result(
            status=status,
            request=normalized,
            data=data,
            warnings=warnings,
            rejected=rejected,
            truncated=truncated,
        )
    except GraphTraversalValidationError as exc:
        return _result(status="blocked", request=normalized, errors=[str(exc)])
    except Exception as exc:
        # Never return a partially authorized subgraph after a backend or PG
        # reauthorization failure.
        return _result(status="failed", request=normalized, errors=[str(exc)])


async def graph_traversal_node(
    state: Mapping[str, Any],
    *,
    client: Any | None = None,
    reauthorize_relations: RelationReauthorizer | None = None,
) -> Dict[str, Any]:
    """LangGraph-compatible adapter around :func:`run_graph_traversal`.

    The planner/integration layer places a validated request in
    ``state['graph_traversal_request']``.  This adapter only emits reducer-backed
    tool/evidence fields, so it can be inserted without coupling the traversal
    core to the rest of the diagnosis orchestrator.
    """

    request = state.get("graph_traversal_request") or {}
    allowed_tools = set(state.get("allowed_tools") or ())
    policy_disallows = "allowed_tools" in state and "graph_traversal" not in allowed_tools
    if policy_disallows or not request.get("seed_ids"):
        reason = (
            "关系查询不在本轮工具白名单中"
            if policy_disallows
            else "直接查询未返回可绑定的业务实体ID，未执行关系扩展"
        )
        tool_result = make_tool_result(
            tool="graph_traversal",
            status="empty",
            params=dict(request),
            data={"node_count": 0, "edge_count": 0, "path_count": 0},
            backend="hugegraph+postgres_reauthorization",
            warnings=[reason],
        )
        return {
            "tool_calls": [
                {
                    "tool": "graph_traversal",
                    "count": 0,
                    "duration": 0.0,
                    "params": dict(request),
                    "status": "empty",
                    "evidence_count": 0,
                    "evidence_ids": [],
                    "results_preview": [],
                    "skip_reason": reason,
                }
            ],
            "tool_results": [tool_result],
            "evidence_registry": [],
            "graph_paths": [],
        }

    started = time.monotonic()
    result = await asyncio.to_thread(
        run_graph_traversal,
        request,
        client=client,
        reauthorize_relations=reauthorize_relations,
    )
    paths = [dict(item) for item in (result.get("paths") or [])]
    nodes_by_physical_id = {
        _text(item.get("id")): item
        for item in (result.get("nodes") or [])
        if isinstance(item, Mapping) and _text(item.get("id"))
    }
    binding_by_logical_id = {
        _text(item.get("logical_id")): item
        for item in (state.get("graph_seed_bindings") or [])
        if isinstance(item, Mapping) and _text(item.get("logical_id"))
    }
    binding_by_path_id: Dict[str, Mapping[str, Any]] = {}
    for path in paths:
        start_node = nodes_by_physical_id.get(_text(path.get("start_id")), {})
        binding = binding_by_logical_id.get(_text(start_node.get("logical_id")))
        if binding:
            path["seed_binding_status"] = binding.get("binding_status", "candidate")
            path["seed_match_type"] = binding.get("match_type", "")
            binding_by_path_id[_text(path.get("path_id"))] = binding

    evidence: List[Dict[str, Any]] = []
    for raw_item in result.get("evidence") or []:
        item = dict(raw_item)
        binding = binding_by_path_id.get(_text(item.get("record_id")))
        if binding and binding.get("binding_status") != "exact":
            # The edges remain formally re-authorized, while the user's words
            # only selected a candidate start entity. Keep those two confidence
            # dimensions separate for the evidence gate and answer model.
            item["evidence_level"] = "L3_INFERRED"
            item["match_type"] = "candidate_seed_multihop_path"
            limitations = list(item.get("limitations") or [])
            limitation = "关系边已正式再授权，但起点由语义召回候选绑定，不代表实体已被用户精确确认"
            if limitation not in limitations:
                limitations.append(limitation)
            item["limitations"] = limitations
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "seed_binding_status": "candidate",
                    "seed_match_type": binding.get("match_type", "semantic_candidate"),
                    "seed_logical_id": binding.get("logical_id", ""),
                }
            )
            item["metadata"] = metadata
        evidence.append(item)
    diagnostics = result.get("diagnostics") or {}
    tool_result = make_tool_result(
        tool="graph_traversal",
        status=result.get("status", "failed"),
        params=dict(result.get("request") or {}),
        data={
            "mode": result.get("mode", ""),
            "node_count": len(result.get("nodes") or []),
            "edge_count": len(result.get("edges") or []),
            "path_count": len(result.get("paths") or []),
            "truncated": result.get("truncated") is True,
        },
        evidence_ids=[item["evidence_id"] for item in evidence],
        backend="hugegraph+postgres_reauthorization",
        warnings=diagnostics.get("warnings") or [],
        errors=diagnostics.get("errors") or [],
    )
    previews = [
        {
            "evidence_id": item["evidence_id"],
            "title": item.get("title", "关系路径"),
            "snippet": item.get("display_summary", "")[:260],
            "source_ref": item.get("source_ref", "")[:320],
        }
        for item in evidence[:8]
    ]
    return {
        "tool_calls": [
            {
                "tool": "graph_traversal",
                "count": len(result.get("paths") or []),
                "duration": round(time.monotonic() - started, 3),
                "params": dict(result.get("request") or {}),
                "evidence_count": len(evidence),
                "evidence_ids": [item["evidence_id"] for item in evidence[:8]],
                "results_preview": previews,
            }
        ],
        "tool_results": [tool_result],
        "evidence_registry": evidence,
        "graph_paths": paths,
    }


def build_graph_traversal_node(
    *,
    client: Any | None = None,
    reauthorize_relations: RelationReauthorizer | None = None,
) -> Callable[[Mapping[str, Any]], Any]:
    """Bind storage/PG dependencies once and return a LangGraph node callable."""

    async def node(state: Mapping[str, Any]) -> Dict[str, Any]:
        return await graph_traversal_node(
            state,
            client=client,
            reauthorize_relations=reauthorize_relations,
        )

    return node


# A concise integration alias; the explicit name remains preferable in tests.
graph_traversal = run_graph_traversal


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_PATHS",
    "GraphTraversalValidationError",
    "HARD_MAX_DEPTH",
    "HARD_MAX_EDGES",
    "HARD_MAX_NODES",
    "HARD_MAX_PATHS",
    "RelationReauthorizer",
    "STRUCTURED_KB_ID",
    "build_graph_traversal_node",
    "graph_traversal",
    "graph_traversal_node",
    "run_graph_traversal",
]
