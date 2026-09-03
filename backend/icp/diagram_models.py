"""Canonical engineering-diagram models used before database projection.

The visual/table extractors are deliberately treated as *candidates*.  This
module only normalizes and reconciles those candidates; it never turns an
automatic extraction into a confirmed engineering model.  Confirmation is a
separate, explicit repository operation with reviewer attribution.

The canonical graph format remains compatible with the existing FTA/RBD
calculators (``vertices`` and ``edges``), while adding stable element keys and
source provenance required for repeatable review and projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DIAGRAM_TYPES = frozenset({"function_locator", "fta", "rbd"})

REVIEW_PENDING = "pending"
REVIEW_CONFLICTED = "conflicted"
REVIEW_CONFIRMED = "confirmed"
REVIEW_REJECTED = "rejected"
REVIEW_STATUSES = frozenset(
    {REVIEW_PENDING, REVIEW_CONFLICTED, REVIEW_CONFIRMED, REVIEW_REJECTED}
)

PROJECTION_NOT_PROJECTED = "not_projected"
PROJECTION_PUBLISHING = "publishing"
PROJECTION_PUBLISHED = "published"
PROJECTION_FAILED = "failed"
PROJECTION_STALE = "stale"
PROJECTION_STATUSES = frozenset(
    {
        PROJECTION_NOT_PROJECTED,
        PROJECTION_PUBLISHING,
        PROJECTION_PUBLISHED,
        PROJECTION_FAILED,
        PROJECTION_STALE,
    }
)

# Only explicit human review may enter REVIEW_CONFIRMED.  Code which builds a
# revision from parser output must choose one of these statuses.
AUTOMATIC_REVIEW_STATUSES = frozenset({REVIEW_PENDING, REVIEW_CONFLICTED})

REVIEW_TRANSITIONS = {
    REVIEW_PENDING: frozenset({REVIEW_CONFLICTED, REVIEW_CONFIRMED, REVIEW_REJECTED}),
    REVIEW_CONFLICTED: frozenset({REVIEW_PENDING, REVIEW_REJECTED}),
    REVIEW_CONFIRMED: frozenset(),
    REVIEW_REJECTED: frozenset(),
}

PROJECTION_TRANSITIONS = {
    PROJECTION_NOT_PROJECTED: frozenset({PROJECTION_PUBLISHING}),
    PROJECTION_FAILED: frozenset({PROJECTION_PUBLISHING}),
    PROJECTION_PUBLISHING: frozenset({PROJECTION_PUBLISHED, PROJECTION_FAILED}),
    PROJECTION_PUBLISHED: frozenset({PROJECTION_STALE, PROJECTION_PUBLISHING}),
    PROJECTION_STALE: frozenset({PROJECTION_PUBLISHING}),
}

SOURCE_KINDS = frozenset(
    {
        "image_vision",
        "table_nodes",
        "table_edges",
        "structured_expression",
        "manual",
        "legacy",
    }
)

_SOURCE_PRIORITY = {
    "manual": 0,
    "structured_expression": 10,
    "table_edges": 20,
    "table_nodes": 21,
    "image_vision": 30,
    "legacy": 40,
}

_GENERIC_LOCAL_ID = re.compile(
    r"^(?:(?:v|n|node|e|edge|item)[-_ ]?\d+|[a-z]|top|root|basic|intermediate|gate|start|end|block)$",
    re.IGNORECASE,
)
_STABLE_NODE_KEY = re.compile(r"^node_[0-9a-f]{32}$")
_SPACE_RE = re.compile(r"\s+")
_IDENTITY_PUNCT_RE = re.compile(r"[\s_\-—–/\\:：;；,，。·()（）\[\]【】]+")
_EXPLICIT_ID_KEYS = (
    "business_id",
    "node_id",
    "event_id",
    "module_id",
    "function_id",
    "diagram_node_id",
    "code",
)
_VOLATILE_PROPERTY_KEYS = frozenset(
    {
        "source_ref",
        "source_refs",
        "source_locator",
        "artifact_ref_id",
        "bbox",
        "page",
        "confidence",
        "extraction_confidence",
        "created_at",
        "updated_at",
        "extractor",
        "extractor_version",
        "model",
        "prompt_version",
        "original_id",
    }
)
_VOLATILE_METADATA_KEYS = _VOLATILE_PROPERTY_KEYS | frozenset(
    {"image_url", "storage_key", "minio_path", "source_block"}
)

_LABEL_ALIASES = {
    "topevent": "TopEvent",
    "top_event": "TopEvent",
    "顶事件": "TopEvent",
    "intermediateevent": "IntermediateEvent",
    "intermediate_event": "IntermediateEvent",
    "中间事件": "IntermediateEvent",
    "basicevent": "BasicEvent",
    "basic_event": "BasicEvent",
    "基本事件": "BasicEvent",
    "gate": "Gate",
    "逻辑门": "Gate",
    "start": "Start",
    "开始": "Start",
    "end": "End",
    "结束": "End",
    "reliabilityblock": "ReliabilityBlock",
    "reliability_block": "ReliabilityBlock",
    "block": "ReliabilityBlock",
    "功能块": "ReliabilityBlock",
    "system": "System",
    "系统": "System",
    "subsystem": "Subsystem",
    "子系统": "Subsystem",
    "module": "Module",
    "模块": "Module",
    "function": "Function",
    "功能": "Function",
    "interface": "Interface",
    "接口": "Interface",
}

_RELATION_ALIASES = {
    "cause": "causes",
    "caused_by": "causes",
    "原因": "causes",
    "导致": "causes",
    "contains": "contains",
    "contain": "contains",
    "包含": "contains",
    "implements": "implements",
    "实现": "implements",
    "connects": "connects_to",
    "connects_to": "connects_to",
    "连接": "connects_to",
    "dataflow": "data_flow",
    "data_flow": "data_flow",
    "数据流": "data_flow",
    "depends": "depends_on",
    "depends_on": "depends_on",
    "依赖": "depends_on",
    "series": "series",
    "parallel": "parallel",
    "redundant": "redundant",
    "bypass": "bypass",
}

_CRITICAL_PROPERTIES = {
    "function_locator": frozenset(
        {"name", "business_id", "module_id", "function_id", "interface_id"}
    ),
    "fta": frozenset(
        {
            "name",
            "business_id",
            "event_id",
            "gate_type",
            "gate_group",
            "probability",
            "failure_probability",
            "failure_rate",
            "lambda",
            "λ",
        }
    ),
    "rbd": frozenset(
        {
            "name",
            "business_id",
            "module_id",
            "reliability",
            "success_probability",
            "failure_rate",
            "lambda",
            "λ",
            "connection_type",
            "branch_id",
        }
    ),
}

_CRITICAL_EDGE_PROPERTIES = {
    "function_locator": frozenset({"interface_id", "direction", "protocol"}),
    "fta": frozenset({"gate_type", "gate_group", "cut_set_no"}),
    "rbd": frozenset(
        {"connection_type", "branch_id", "redundancy", "required_count", "total_count"}
    ),
}


class DiagramModelError(ValueError):
    """Raised when a diagram/candidate cannot be normalized safely."""


class DiagramStateTransitionError(DiagramModelError):
    """Raised when review/projection state would move along an invalid edge."""


@dataclass(frozen=True)
class GraphConflict:
    code: str
    message: str
    element_kind: str = "graph"
    element_key: str = ""
    candidate_ids: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    blocking: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "element_kind": self.element_kind,
            "element_key": self.element_key,
            "candidate_ids": list(self.candidate_ids),
            "details": copy.deepcopy(dict(self.details)),
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class MergeResult:
    graph: Mapping[str, Any]
    canonical_hash: str
    review_status: str
    conflicts: Tuple[GraphConflict, ...]
    candidate_hashes: Mapping[str, str]
    duplicate_candidate_ids: Tuple[str, ...] = ()

    @property
    def conflict_count(self) -> int:
        return sum(1 for item in self.conflicts if item.blocking)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph": copy.deepcopy(dict(self.graph)),
            "canonical_hash": self.canonical_hash,
            "review_status": self.review_status,
            "conflict_count": self.conflict_count,
            "conflicts": [item.as_dict() for item in self.conflicts],
            "candidate_hashes": dict(self.candidate_hashes),
            "duplicate_candidate_ids": list(self.duplicate_candidate_ids),
        }


def ensure_review_transition(current: str, target: str, *, automatic: bool = False) -> None:
    if current not in REVIEW_STATUSES or target not in REVIEW_STATUSES:
        raise DiagramStateTransitionError(f"unknown review state: {current!r} -> {target!r}")
    if automatic and target not in AUTOMATIC_REVIEW_STATUSES:
        raise DiagramStateTransitionError("automatic extraction may not confirm or reject a revision")
    if target not in REVIEW_TRANSITIONS[current]:
        raise DiagramStateTransitionError(f"invalid review transition: {current} -> {target}")


def ensure_projection_transition(current: str, target: str) -> None:
    if current not in PROJECTION_STATUSES or target not in PROJECTION_STATUSES:
        raise DiagramStateTransitionError(
            f"unknown projection state: {current!r} -> {target!r}"
        )
    if target not in PROJECTION_TRANSITIONS[current]:
        raise DiagramStateTransitionError(f"invalid projection transition: {current} -> {target}")


def clean_text(value: Any) -> str:
    """Normalize display text without destroying codes or leading zeroes."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _SPACE_RE.sub(" ", text.replace("\r", " ").replace("\n", " ")).strip()


def identity_text(value: Any) -> str:
    """Return a conservative comparison token used only inside one diagram."""

    return _IDENTITY_PUNCT_RE.sub("", clean_text(value).casefold())


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_key(kind: str, namespace: str, *parts: Any) -> str:
    """Create a deterministic, HugeGraph-safe key from semantic identity."""

    kind_token = re.sub(r"[^a-z0-9]+", "_", clean_text(kind).casefold()).strip("_") or "item"
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"icp-rag:diagram:{clean_text(namespace)}")
    name = "\x1f".join(identity_text(part) for part in parts)
    return f"{kind_token}_{uuid.uuid5(namespace_uuid, name).hex}"


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_scalar(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_scalar(item) for item in value]
    return value


def _normalize_label(label: Any) -> str:
    display = clean_text(label) or "Unknown"
    token = identity_text(display)
    return _LABEL_ALIASES.get(token, _LABEL_ALIASES.get(display.casefold(), display))


def _normalize_relation(label: Any) -> str:
    display = clean_text(label) or "relation"
    token = identity_text(display)
    return _RELATION_ALIASES.get(token, _RELATION_ALIASES.get(display.casefold(), display))


def _explicit_node_id(node: Mapping[str, Any], props: Mapping[str, Any]) -> str:
    for key in _EXPLICIT_ID_KEYS:
        value = clean_text(props.get(key))
        if value:
            return value
    local_id = clean_text(node.get("id"))
    if local_id and not _GENERIC_LOCAL_ID.fullmatch(local_id):
        return local_id
    return ""


def _source_ref(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    source_ref = {
        "candidate_id": clean_text(candidate.get("candidate_id") or candidate.get("id")),
        "source_kind": clean_text(candidate.get("source_kind")) or "legacy",
    }
    for key in ("artifact_ref_id", "source_fingerprint", "source_locator"):
        value = candidate.get(key)
        if value not in (None, "", {}, []):
            source_ref[key] = copy.deepcopy(value)
    return source_ref


def _dedupe_source_refs(refs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for ref in refs:
        item = _as_mapping(ref)
        unique[canonical_json(item)] = copy.deepcopy(item)
    return [unique[key] for key in sorted(unique)]


def normalize_graph(
    graph: Mapping[str, Any],
    diagram_id: str,
    *,
    diagram_type: Optional[str] = None,
    source_ref: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one candidate graph and assign stable node/edge identifiers.

    Name-only identities are used only when they are unique inside this graph.
    Ambiguous duplicate names retain separate source-local disambiguators and are
    later reported for human review.
    """

    if not isinstance(graph, Mapping):
        raise DiagramModelError("graph must be an object")
    dtype = clean_text(diagram_type or graph.get("diagram_type")).casefold()
    if dtype and dtype not in DIAGRAM_TYPES:
        raise DiagramModelError(f"unsupported diagram_type: {dtype}")

    raw_vertices = [item for item in graph.get("vertices", []) or [] if isinstance(item, Mapping)]
    raw_edges = [item for item in graph.get("edges", []) or [] if isinstance(item, Mapping)]
    prepared: List[Dict[str, Any]] = []
    identity_counts: Dict[str, int] = {}
    for index, raw in enumerate(raw_vertices):
        props = _as_mapping(raw.get("properties"))
        normalized_props = _normalize_scalar(props)
        name = clean_text(normalized_props.get("name") or raw.get("name") or raw.get("id"))
        if name:
            normalized_props["name"] = name
        label = _normalize_label(raw.get("label"))
        explicit_id = _explicit_node_id(raw, normalized_props)
        stable_id = clean_text(raw.get("id")) if _STABLE_NODE_KEY.fullmatch(clean_text(raw.get("id"))) else ""
        identity = f"id:{identity_text(explicit_id)}" if explicit_id else (
            f"name:{identity_text(label)}:{identity_text(name)}"
        )
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
        prepared.append(
            {
                "raw": raw,
                "index": index,
                "label": label,
                "properties": normalized_props,
                "name": name,
                "explicit_id": explicit_id,
                "stable_id": stable_id,
                "identity": identity,
            }
        )

    id_map: Dict[str, str] = {}
    vertices: List[Dict[str, Any]] = []
    for item in prepared:
        raw = item["raw"]
        local_id = clean_text(raw.get("id")) or f"node-{item['index'] + 1}"
        disambiguator = ""
        if identity_counts[item["identity"]] > 1:
            disambiguator = local_id or str(item["index"])
        node_key = item["stable_id"] or stable_key(
            "node", diagram_id, item["identity"], disambiguator
        )
        id_map[local_id] = node_key
        props = copy.deepcopy(item["properties"])
        props.setdefault("original_id", local_id)
        refs = list(props.get("source_refs") or [])
        if source_ref:
            refs.append(source_ref)
        if refs:
            props["source_refs"] = _dedupe_source_refs(refs)
        vertices.append({"id": node_key, "label": item["label"], "properties": props})

    edges: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_edges):
        raw_source = clean_text(raw.get("source") or raw.get("outV"))
        raw_target = clean_text(raw.get("target") or raw.get("inV"))
        source = id_map.get(raw_source, raw_source)
        target = id_map.get(raw_target, raw_target)
        relation = _normalize_relation(raw.get("label") or _as_mapping(raw.get("properties")).get("relation_type"))
        props = _normalize_scalar(_as_mapping(raw.get("properties")))
        if "gate_type" in props:
            props["gate_type"] = clean_text(props["gate_type"]).upper()
        if "connection_type" in props:
            props["connection_type"] = clean_text(props["connection_type"]).casefold()
        refs = list(props.get("source_refs") or [])
        if source_ref:
            refs.append(source_ref)
        if refs:
            props["source_refs"] = _dedupe_source_refs(refs)
        qualifier = props.get("gate_group") or props.get("branch_id") or props.get("connection_type") or ""
        edge_key = stable_key("edge", diagram_id, source, relation, target, qualifier)
        props.setdefault("original_id", clean_text(raw.get("id")) or f"edge-{index + 1}")
        edges.append(
            {
                "id": edge_key,
                "label": relation,
                "source": source,
                "target": target,
                "properties": props,
            }
        )

    root = clean_text(graph.get("root_fault_id") or graph.get("root_id"))
    normalized = {
        "diagram_type": dtype,
        "root_fault_id": id_map.get(root, root),
        "metadata": _normalize_scalar(_as_mapping(graph.get("metadata"))),
        "vertices": sorted(vertices, key=lambda value: str(value["id"])),
        "edges": sorted(edges, key=lambda value: str(value["id"])),
    }
    return normalized


def _semantic_value(value: Any, *, metadata: bool = False) -> Any:
    ignored = _VOLATILE_METADATA_KEYS if metadata else _VOLATILE_PROPERTY_KEYS
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item, metadata=metadata)
            for key, item in sorted(value.items())
            if str(key) not in ignored
        }
    if isinstance(value, list):
        return [_semantic_value(item, metadata=metadata) for item in value]
    return value


def semantic_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip provenance/volatile extraction fields before revision hashing."""

    vertices = []
    for vertex in graph.get("vertices", []) or []:
        if not isinstance(vertex, Mapping):
            continue
        vertices.append(
            {
                "id": clean_text(vertex.get("id")),
                "label": _normalize_label(vertex.get("label")),
                "properties": _semantic_value(_as_mapping(vertex.get("properties"))),
            }
        )
    edges = []
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        edges.append(
            {
                "id": clean_text(edge.get("id")),
                "label": _normalize_relation(edge.get("label")),
                "source": clean_text(edge.get("source") or edge.get("outV")),
                "target": clean_text(edge.get("target") or edge.get("inV")),
                "properties": _semantic_value(_as_mapping(edge.get("properties"))),
            }
        )
    return {
        "diagram_type": clean_text(graph.get("diagram_type")).casefold(),
        "root_fault_id": clean_text(graph.get("root_fault_id") or graph.get("root_id")),
        "metadata": _semantic_value(_as_mapping(graph.get("metadata")), metadata=True),
        "vertices": sorted(vertices, key=lambda item: item["id"]),
        "edges": sorted(edges, key=lambda item: item["id"]),
    }


def canonical_graph_hash(graph: Mapping[str, Any]) -> str:
    return content_hash(semantic_graph(graph))


def source_fingerprint(value: Any) -> str:
    """Hash raw image bytes, normalized table cells, or another source object."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    return content_hash(_normalize_scalar(value))


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    return clean_text(candidate.get("candidate_id") or candidate.get("id")) or f"candidate-{index + 1}"


def _candidate_graph(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("normalized_graph") or candidate.get("graph") or candidate.get("raw_graph")
    return value if isinstance(value, Mapping) else {}


def _node_explicit_identity(vertex: Mapping[str, Any]) -> str:
    """Return a source-supplied node identity, excluding parser-local IDs."""

    props = _as_mapping(vertex.get("properties"))
    for key in _EXPLICIT_ID_KEYS:
        value = clean_text(props.get(key))
        if value:
            return identity_text(value)
    original_id = clean_text(props.get("original_id"))
    if original_id and not _GENERIC_LOCAL_ID.fullmatch(original_id):
        return identity_text(original_id)
    return ""


def _node_name_alias(vertex: Mapping[str, Any]) -> Tuple[str, str]:
    props = _as_mapping(vertex.get("properties"))
    return (
        identity_text(_normalize_label(vertex.get("label"))),
        identity_text(props.get("name")),
    )


def _values_equal(left: Any, right: Any) -> bool:
    return canonical_json(_semantic_value(left)) == canonical_json(_semantic_value(right))


def _merge_noncritical(left: Any, right: Any) -> Any:
    if left in (None, "", [], {}):
        return copy.deepcopy(right)
    if right in (None, "", [], {}):
        return copy.deepcopy(left)
    if _values_equal(left, right):
        return copy.deepcopy(left)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = copy.deepcopy(dict(left))
        for key, value in right.items():
            result[key] = _merge_noncritical(result.get(key), value)
        return result
    if isinstance(left, list) and isinstance(right, list):
        items = {canonical_json(item): copy.deepcopy(item) for item in [*left, *right]}
        return [items[key] for key in sorted(items)]
    if isinstance(left, str) and isinstance(right, str):
        # Descriptions from two sources are evidence, not a reason to overwrite
        # one another.  Preserve the richer deterministic representation.
        return min((left, right), key=lambda value: (-len(value), value))
    return min((left, right), key=canonical_json)


def _conflict(
    code: str,
    message: str,
    *,
    kind: str = "graph",
    key: str = "",
    candidate_ids: Sequence[str] = (),
    details: Optional[Mapping[str, Any]] = None,
) -> GraphConflict:
    return GraphConflict(
        code=code,
        message=message,
        element_kind=kind,
        element_key=key,
        candidate_ids=tuple(sorted(set(candidate_ids))),
        details=copy.deepcopy(dict(details or {})),
    )


def detect_graph_conflicts(
    graph: Mapping[str, Any], *, candidate_id: str = ""
) -> List[GraphConflict]:
    """Return structural problems which must block automatic use/confirmation."""

    conflicts: List[GraphConflict] = []
    vertices = [item for item in graph.get("vertices", []) or [] if isinstance(item, Mapping)]
    edges = [item for item in graph.get("edges", []) or [] if isinstance(item, Mapping)]
    ids = [clean_text(item.get("id")) for item in vertices]
    known = {item for item in ids if item}
    candidate_ids = (candidate_id,) if candidate_id else ()
    if not vertices:
        conflicts.append(
            _conflict("empty_graph", "候选图没有可确认的节点", candidate_ids=candidate_ids)
        )
        return conflicts
    if len(known) != len(ids):
        conflicts.append(
            _conflict("duplicate_node_key", "候选图存在空或重复节点键", candidate_ids=candidate_ids)
        )
    for edge in edges:
        edge_key = clean_text(edge.get("id"))
        source = clean_text(edge.get("source") or edge.get("outV"))
        target = clean_text(edge.get("target") or edge.get("inV"))
        if source not in known or target not in known:
            conflicts.append(
                _conflict(
                    "dangling_edge",
                    "关系引用了不存在的节点",
                    kind="edge",
                    key=edge_key,
                    candidate_ids=candidate_ids,
                    details={"source": source, "target": target},
                )
            )
        if source and source == target:
            conflicts.append(
                _conflict(
                    "self_loop",
                    "工程拓扑关系不能无说明地指向自身",
                    kind="edge",
                    key=edge_key,
                    candidate_ids=candidate_ids,
                )
            )
    root = clean_text(graph.get("root_fault_id") or graph.get("root_id"))
    if root and root not in known:
        conflicts.append(
            _conflict(
                "missing_root",
                "根节点不在节点集合中",
                candidate_ids=candidate_ids,
                details={"root": root},
            )
        )
    return conflicts


def merge_candidates(
    diagram_id: str,
    diagram_type: str,
    candidates: Sequence[Mapping[str, Any]],
) -> MergeResult:
    """Merge table/image/expression candidates into one reviewable draft.

    Exact stable identities are deduplicated and provenance is accumulated.
    Ambiguous or contradictory topology is retained as a conflict.  The return
    status is *always* pending/conflicted; this function has no confirmation
    path by design.
    """

    dtype = clean_text(diagram_type).casefold()
    if dtype not in DIAGRAM_TYPES:
        raise DiagramModelError(f"unsupported diagram_type: {diagram_type}")
    ordered = sorted(
        enumerate(candidates),
        key=lambda pair: (
            _SOURCE_PRIORITY.get(clean_text(pair[1].get("source_kind")), 100),
            _candidate_id(pair[1], pair[0]),
        ),
    )
    node_map: Dict[str, Dict[str, Any]] = {}
    edge_map: Dict[str, Dict[str, Any]] = {}
    node_sources: Dict[str, List[str]] = {}
    edge_sources: Dict[str, List[str]] = {}
    node_aliases: Dict[Tuple[str, str], List[str]] = {}
    candidate_hashes: Dict[str, str] = {}
    first_hash_owner: Dict[str, str] = {}
    duplicates: List[str] = []
    conflicts: List[GraphConflict] = []
    metadata: Dict[str, Any] = {}
    root_ids: Dict[str, List[str]] = {}

    for original_index, candidate in ordered:
        cid = _candidate_id(candidate, original_index)
        source_kind = clean_text(candidate.get("source_kind")) or "legacy"
        if source_kind not in SOURCE_KINDS:
            raise DiagramModelError(f"unsupported source_kind: {source_kind}")
        ref = _source_ref({**dict(candidate), "candidate_id": cid, "source_kind": source_kind})
        raw_graph = _candidate_graph(candidate)
        # Normalize even when the caller supplied normalized_graph.  The operation
        # is idempotent and protects repository callers from malformed IDs.
        normalized = normalize_graph(raw_graph, diagram_id, diagram_type=dtype, source_ref=ref)
        graph_hash = canonical_graph_hash(normalized)
        candidate_hashes[cid] = graph_hash
        if graph_hash in first_hash_owner:
            duplicates.append(cid)
        else:
            first_hash_owner[graph_hash] = cid
        conflicts.extend(detect_graph_conflicts(normalized, candidate_id=cid))
        metadata = _merge_noncritical(metadata, _as_mapping(normalized.get("metadata")))
        candidate_node_keys: Dict[str, str] = {}

        for vertex in normalized.get("vertices", []) or []:
            source_key = clean_text(vertex.get("id"))
            key = source_key
            existing = node_map.get(key)
            alias = _node_name_alias(vertex)
            alias_keys = node_aliases.get(alias, []) if all(alias) else []
            if existing is None and len(alias_keys) == 1:
                alias_key = alias_keys[0]
                alias_vertex = node_map[alias_key]
                alias_sources = node_sources.get(alias_key, [])
                left_identity = _node_explicit_identity(alias_vertex)
                right_identity = _node_explicit_identity(vertex)
                same_candidate_duplicate = cid in alias_sources
                identity_mismatch = (
                    bool(left_identity)
                    and bool(right_identity)
                    and left_identity != right_identity
                )
                if not same_candidate_duplicate and not identity_mismatch:
                    # Tables normally carry engineering IDs while image OCR often
                    # carries only the visible label.  A unique same-type/name
                    # alias is safe to reconcile when at most one side supplies
                    # an explicit ID.  Edges are remapped below to this key.
                    key = alias_key
                    existing = alias_vertex
                elif identity_mismatch:
                    conflicts.append(
                        _conflict(
                            "same_name_distinct_identity",
                            "同类型同名节点具有不同的明确业务标识，未自动合并",
                            kind="node",
                            key="|".join(alias),
                            candidate_ids=[*alias_sources, cid],
                            details={
                                "left_identity": left_identity,
                                "right_identity": right_identity,
                            },
                        )
                    )
            elif existing is None and len(alias_keys) > 1:
                conflicts.append(
                    _conflict(
                        "ambiguous_node_alias",
                        "同类型同名节点对应多个候选标识，无法自动选择合并目标",
                        kind="node",
                        key="|".join(alias),
                        candidate_ids=[
                            *[source for alias_key in alias_keys for source in node_sources.get(alias_key, [])],
                            cid,
                        ],
                        details={"candidate_node_keys": sorted(alias_keys)},
                    )
                )
            candidate_node_keys[source_key] = key
            if existing is None:
                stored = copy.deepcopy(dict(vertex))
                stored["id"] = key
                node_map[key] = stored
                node_sources[key] = [cid]
                if all(alias):
                    node_aliases.setdefault(alias, []).append(key)
                continue
            sources = [*node_sources[key], cid]
            if _normalize_label(existing.get("label")) != _normalize_label(vertex.get("label")):
                conflicts.append(
                    _conflict(
                        "node_type_mismatch",
                        "同一稳定节点在不同来源中的类型不一致",
                        kind="node",
                        key=key,
                        candidate_ids=sources,
                        details={"left": existing.get("label"), "right": vertex.get("label")},
                    )
                )
            left_props = _as_mapping(existing.get("properties"))
            right_props = _as_mapping(vertex.get("properties"))
            critical = _CRITICAL_PROPERTIES[dtype]
            for prop in sorted(set(left_props) | set(right_props)):
                left_value = left_props.get(prop)
                right_value = right_props.get(prop)
                if (
                    prop in critical
                    and left_value not in (None, "", [], {})
                    and right_value not in (None, "", [], {})
                    and not _values_equal(left_value, right_value)
                ):
                    conflicts.append(
                        _conflict(
                            "node_property_mismatch",
                            f"同一节点的关键属性 {prop} 不一致",
                            kind="node",
                            key=key,
                            candidate_ids=sources,
                            details={"property": prop, "left": left_value, "right": right_value},
                        )
                    )
                left_props[prop] = _merge_noncritical(left_value, right_value)
            existing["properties"] = left_props
            node_sources[key].append(cid)

        for edge in normalized.get("edges", []) or []:
            edge = copy.deepcopy(dict(edge))
            edge["source"] = candidate_node_keys.get(
                clean_text(edge.get("source")), clean_text(edge.get("source"))
            )
            edge["target"] = candidate_node_keys.get(
                clean_text(edge.get("target")), clean_text(edge.get("target"))
            )
            edge_props = _as_mapping(edge.get("properties"))
            qualifier = (
                edge_props.get("gate_group")
                or edge_props.get("branch_id")
                or edge_props.get("connection_type")
                or ""
            )
            key = stable_key(
                "edge",
                diagram_id,
                edge["source"],
                _normalize_relation(edge.get("label")),
                edge["target"],
                qualifier,
            )
            edge["id"] = key
            existing = edge_map.get(key)
            if existing is None:
                edge_map[key] = edge
                edge_sources[key] = [cid]
                continue
            left_props = _as_mapping(existing.get("properties"))
            right_props = _as_mapping(edge.get("properties"))
            sources = [*edge_sources[key], cid]
            for prop in sorted(_CRITICAL_EDGE_PROPERTIES[dtype]):
                left_value = left_props.get(prop)
                right_value = right_props.get(prop)
                if (
                    left_value not in (None, "", [], {})
                    and right_value not in (None, "", [], {})
                    and not _values_equal(left_value, right_value)
                ):
                    conflicts.append(
                        _conflict(
                            "edge_property_mismatch",
                            f"同一关系的关键属性 {prop} 不一致",
                            kind="edge",
                            key=key,
                            candidate_ids=sources,
                            details={
                                "property": prop,
                                "left": left_value,
                                "right": right_value,
                            },
                        )
                    )
            existing["properties"] = _merge_noncritical(left_props, right_props)
            edge_sources[key].append(cid)

        root = clean_text(normalized.get("root_fault_id"))
        root = candidate_node_keys.get(root, root)
        if root:
            root_ids.setdefault(root, []).append(cid)

    # Same visible identity but different node types must not be silently folded.
    by_name: Dict[str, List[Tuple[str, str]]] = {}
    for key, vertex in node_map.items():
        name = identity_text(_as_mapping(vertex.get("properties")).get("name"))
        if name:
            by_name.setdefault(name, []).append((key, _normalize_label(vertex.get("label"))))
    for name, values in by_name.items():
        labels = {label for _, label in values}
        if len(labels) > 1:
            conflicts.append(
                _conflict(
                    "ambiguous_name_types",
                    "同名节点在候选来源中具有不同工程类型，未自动合并",
                    kind="node",
                    key=name,
                    candidate_ids=[cid for key, _ in values for cid in node_sources.get(key, [])],
                    details={"nodes": values},
                )
            )
        elif len(values) > 1:
            conflicts.append(
                _conflict(
                    "ambiguous_same_name",
                    "同类型同名节点仍有多个稳定标识，自动流程无法安全合并，必须隔离或修正来源标识",
                    kind="node",
                    key=name,
                    candidate_ids=[cid for key, _ in values for cid in node_sources.get(key, [])],
                    details={"nodes": values},
                )
            )

    # Detect direction/relation disagreements which naturally have different
    # edge keys and therefore would otherwise coexist unnoticed.
    by_unordered_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for edge in edge_map.values():
        source = clean_text(edge.get("source"))
        target = clean_text(edge.get("target"))
        by_unordered_pair.setdefault(tuple(sorted((source, target))), []).append(edge)
    for pair, values in by_unordered_pair.items():
        directions = {(clean_text(item.get("source")), clean_text(item.get("target"))) for item in values}
        semantic = {
            (
                _normalize_relation(item.get("label")),
                clean_text(_as_mapping(item.get("properties")).get("gate_type")).upper(),
                clean_text(_as_mapping(item.get("properties")).get("connection_type")).casefold(),
            )
            for item in values
        }
        involved = [cid for item in values for cid in edge_sources.get(clean_text(item.get("id")), [])]
        if len(directions) > 1:
            conflicts.append(
                _conflict(
                    "edge_direction_mismatch",
                    "相同节点之间的关系方向不一致",
                    kind="edge",
                    key="|".join(pair),
                    candidate_ids=involved,
                    details={"directions": sorted(directions)},
                )
            )
        elif len(semantic) > 1:
            conflicts.append(
                _conflict(
                    "edge_semantic_mismatch",
                    "相同端点之间的关系/逻辑门语义不一致",
                    kind="edge",
                    key="|".join(pair),
                    candidate_ids=involved,
                    details={"semantics": sorted(semantic)},
                )
            )

    if len(root_ids) > 1:
        conflicts.append(
            _conflict(
                "root_mismatch",
                "不同来源识别出的根节点不一致",
                candidate_ids=[cid for ids in root_ids.values() for cid in ids],
                details={"roots": sorted(root_ids)},
            )
        )

    # Ensure provenance reflects every duplicate/corroborating source.
    for key, vertex in node_map.items():
        props = _as_mapping(vertex.get("properties"))
        refs = list(props.get("source_refs") or [])
        props["source_refs"] = _dedupe_source_refs(refs)
        vertex["properties"] = props
    for key, edge in edge_map.items():
        props = _as_mapping(edge.get("properties"))
        refs = list(props.get("source_refs") or [])
        props["source_refs"] = _dedupe_source_refs(refs)
        edge["properties"] = props

    graph = {
        "diagram_type": dtype,
        "root_fault_id": sorted(root_ids)[0] if len(root_ids) == 1 else "",
        "metadata": metadata,
        "vertices": sorted(node_map.values(), key=lambda item: clean_text(item.get("id"))),
        "edges": sorted(edge_map.values(), key=lambda item: clean_text(item.get("id"))),
    }
    # Deduplicate conflicts deterministically so repeated corroborating candidates
    # do not create a different revision solely through ordering.
    unique_conflicts: Dict[str, GraphConflict] = {}
    for item in conflicts:
        unique_conflicts[canonical_json(item.as_dict())] = item
    ordered_conflicts = tuple(unique_conflicts[key] for key in sorted(unique_conflicts))
    status = REVIEW_CONFLICTED if any(item.blocking for item in ordered_conflicts) else REVIEW_PENDING
    if status not in AUTOMATIC_REVIEW_STATUSES:  # defensive invariant
        raise AssertionError("automatic candidate merge produced a non-automatic review status")
    return MergeResult(
        graph=graph,
        canonical_hash=canonical_graph_hash(graph),
        review_status=status,
        conflicts=ordered_conflicts,
        candidate_hashes=dict(sorted(candidate_hashes.items())),
        duplicate_candidate_ids=tuple(sorted(set(duplicates))),
    )


__all__ = [
    "AUTOMATIC_REVIEW_STATUSES",
    "DIAGRAM_TYPES",
    "DiagramModelError",
    "DiagramStateTransitionError",
    "GraphConflict",
    "MergeResult",
    "PROJECTION_FAILED",
    "PROJECTION_NOT_PROJECTED",
    "PROJECTION_PUBLISHED",
    "PROJECTION_PUBLISHING",
    "PROJECTION_STALE",
    "PROJECTION_STATUSES",
    "REVIEW_CONFIRMED",
    "REVIEW_CONFLICTED",
    "REVIEW_PENDING",
    "REVIEW_REJECTED",
    "REVIEW_STATUSES",
    "SOURCE_KINDS",
    "canonical_graph_hash",
    "canonical_json",
    "clean_text",
    "content_hash",
    "detect_graph_conflicts",
    "ensure_projection_transition",
    "ensure_review_transition",
    "identity_text",
    "merge_candidates",
    "normalize_graph",
    "semantic_graph",
    "source_fingerprint",
    "stable_key",
]
