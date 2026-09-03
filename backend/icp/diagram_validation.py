"""Deterministic unattended validation for ICP engineering diagrams.

Automatic extraction and human confirmation are different things.  This
module therefore produces an explicit validation status and tier instead of
silently turning a vision result into a human-reviewed graph.

The policy is deliberately fail-closed:

* explicit table/expression topology may become ``auto_validated`` after
  type-specific structural checks;
* a conflict-free visual result may become ``provisional`` and be projected
  for browsing, but it must not support formal calculations or authoritative
  topology claims;
* conflicts, incomplete topology and low-confidence visual results are
  ``quarantined``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from backend.icp.diagram_models import clean_text, detect_graph_conflicts


VALIDATION_PENDING = "pending"
VALIDATION_AUTO_VALIDATED = "auto_validated"
VALIDATION_PROVISIONAL = "provisional"
VALIDATION_QUARANTINED = "quarantined"
VALIDATION_FAILED = "failed"
VALIDATION_MANUAL_CONFIRMED = "manual_confirmed"

VALIDATION_STATUSES = frozenset(
    {
        VALIDATION_PENDING,
        VALIDATION_AUTO_VALIDATED,
        VALIDATION_PROVISIONAL,
        VALIDATION_QUARANTINED,
        VALIDATION_FAILED,
        VALIDATION_MANUAL_CONFIRMED,
    }
)
PROJECTABLE_VALIDATION_STATUSES = frozenset(
    {VALIDATION_AUTO_VALIDATED, VALIDATION_PROVISIONAL, VALIDATION_MANUAL_CONFIRMED}
)
FORMAL_VALIDATION_STATUSES = frozenset(
    {VALIDATION_AUTO_VALIDATED, VALIDATION_MANUAL_CONFIRMED}
)

POLICY_VERSION = "icp-diagram-auto-v1"
VISION_CONFIDENCE_THRESHOLD = 0.75
_DETERMINISTIC_TOPOLOGY_SOURCES = frozenset({"table_edges", "structured_expression"})


@dataclass(frozen=True)
class AutomaticValidationResult:
    status: str
    tier: str
    method: str
    policy_version: str
    publishable: bool
    formal_claims_allowed: bool
    source_kinds: Tuple[str, ...]
    confidence: float | None
    checks: Tuple[Mapping[str, Any], ...]
    reasons: Tuple[str, ...]

    def as_report(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tier": self.tier,
            "method": self.method,
            "policy_version": self.policy_version,
            "publishable": self.publishable,
            "formal_claims_allowed": self.formal_claims_allowed,
            "source_kinds": list(self.source_kinds),
            "confidence": self.confidence,
            "checks": [dict(item) for item in self.checks],
            "reasons": list(self.reasons),
        }


def _check(code: str, passed: bool, message: str, *, blocking: bool = True) -> Dict[str, Any]:
    return {
        "code": code,
        "passed": bool(passed),
        "message": message,
        "blocking": bool(blocking),
    }


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _graph_parts(graph: Mapping[str, Any]) -> Tuple[Dict[str, Mapping[str, Any]], List[Mapping[str, Any]]]:
    vertices = {
        clean_text(item.get("id")): item
        for item in graph.get("vertices", []) or []
        if isinstance(item, Mapping) and clean_text(item.get("id"))
    }
    edges = [item for item in graph.get("edges", []) or [] if isinstance(item, Mapping)]
    return vertices, edges


def _adjacency(
    vertices: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    forward = {node_id: set() for node_id in vertices}
    reverse = {node_id: set() for node_id in vertices}
    for edge in edges:
        source = clean_text(edge.get("source") or edge.get("outV"))
        target = clean_text(edge.get("target") or edge.get("inV"))
        if source in forward and target in reverse:
            forward[source].add(target)
            reverse[target].add(source)
    return forward, reverse


def _reachable(start: str, adjacency: Mapping[str, Set[str]]) -> Set[str]:
    if start not in adjacency:
        return set()
    visited: Set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(sorted(adjacency.get(current, set()) - visited))
    return visited


def _has_cycle(adjacency: Mapping[str, Set[str]]) -> bool:
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for child in adjacency.get(node_id, set()):
            if visit(child):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in adjacency if node_id not in visited)


def _label(vertex: Mapping[str, Any]) -> str:
    return clean_text(vertex.get("label"))


def _engineering_parameter_checks(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Reject impossible numeric parameters before formal graph use.

    Missing parameters are allowed because topology browsing is still useful;
    the deterministic calculators already report missing inputs.  Values that
    are present, however, must be finite and within their physical domain.
    """

    bounded = {"probability", "reliability", "availability"}
    non_negative = {
        "failure_rate",
        "repair_rate",
        "mission_time",
        "task_time",
        "threshold",
    }
    invalid: List[str] = []
    items = [
        *(graph.get("vertices", []) or []),
        *(graph.get("edges", []) or []),
    ]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_id = clean_text(item.get("id")) or "<unknown>"
        props = _mapping(item.get("properties"))
        for key in sorted(bounded | non_negative):
            value = props.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, Mapping):
                value = value.get("value")
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid.append(f"{item_id}.{key}=非数值")
                continue
            if number != number or number in {float("inf"), float("-inf")}:
                invalid.append(f"{item_id}.{key}=非有限值")
            elif key in bounded and not 0.0 <= number <= 1.0:
                invalid.append(f"{item_id}.{key} 不在 [0,1]")
            elif key in non_negative and number < 0.0:
                invalid.append(f"{item_id}.{key} 小于 0")
    return [
        _check(
            "engineering_parameter_domains",
            not invalid,
            "工程参数必须满足概率/可靠度区间和时间/失效率非负约束"
            + ("：" + "；".join(invalid[:10]) if invalid else ""),
        )
    ]


def _gate_type(vertex: Mapping[str, Any], outgoing: Sequence[Mapping[str, Any]]) -> str:
    props = _mapping(vertex.get("properties"))
    values: List[Any] = [props.get("gate_type"), props.get("type")]
    values.extend(_mapping(edge.get("properties")).get("gate_type") for edge in outgoing)
    for value in values:
        token = clean_text(value).upper()
        if token in {"AND", "OR", "XOR"}:
            return token
    return ""


def _fta_checks(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    vertices, edges = _graph_parts(graph)
    forward, _ = _adjacency(vertices, edges)
    root = clean_text(graph.get("root_fault_id") or graph.get("root_id"))
    top_nodes = [node_id for node_id, vertex in vertices.items() if _label(vertex) == "TopEvent"]
    checks = [
        _check("fta_minimum_size", len(vertices) >= 2 and len(edges) >= 1, "FTA 至少需要两个事件节点和一条关系"),
        _check("fta_root_present", bool(root) and root in vertices, "FTA 必须有明确且存在的顶事件"),
        _check("fta_single_top_event", len(top_nodes) == 1, "FTA 必须且只能有一个 TopEvent"),
        _check("fta_acyclic", not _has_cycle(forward), "FTA 原因链不能形成有向环"),
    ]
    if root in vertices:
        checks.append(
            _check(
                "fta_root_is_top_event",
                _label(vertices[root]) == "TopEvent",
                "FTA 根节点必须是 TopEvent",
            )
        )
        checks.append(
            _check(
                "fta_all_nodes_reachable",
                _reachable(root, forward) == set(vertices),
                "FTA 所有节点必须从顶事件沿原因链可达",
            )
        )
    allowed_labels = {"TopEvent", "IntermediateEvent", "BasicEvent", "Gate"}
    checks.append(
        _check(
            "fta_node_types",
            all(_label(vertex) in allowed_labels for vertex in vertices.values()),
            "FTA 节点类型必须属于顶事件、中间事件、基本事件或逻辑门",
        )
    )
    edge_by_source: Dict[str, List[Mapping[str, Any]]] = {}
    for edge in edges:
        edge_by_source.setdefault(clean_text(edge.get("source") or edge.get("outV")), []).append(edge)
    gate_nodes = [vertex for vertex in vertices.values() if _label(vertex) == "Gate"]
    checks.append(
        _check(
            "fta_gate_semantics",
            all(
                _gate_type(vertex, edge_by_source.get(clean_text(vertex.get("id")), []))
                for vertex in gate_nodes
            ),
            "每个 FTA 逻辑门都必须明确为 AND、OR 或 XOR",
        )
    )
    return checks


def _rbd_checks(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    vertices, edges = _graph_parts(graph)
    forward, reverse = _adjacency(vertices, edges)
    starts = [node_id for node_id, vertex in vertices.items() if _label(vertex) == "Start"]
    ends = [node_id for node_id, vertex in vertices.items() if _label(vertex) == "End"]
    checks = [
        _check("rbd_minimum_size", len(vertices) >= 3 and len(edges) >= 2, "RBD 至少需要 START、一个功能块、END 和两条关系"),
        _check("rbd_single_start", len(starts) == 1, "RBD 必须且只能有一个 START"),
        _check("rbd_single_end", len(ends) == 1, "RBD 必须且只能有一个 END"),
        _check("rbd_acyclic", not _has_cycle(forward), "RBD 自动验证当前只接受无环可靠性结构"),
    ]
    allowed_labels = {"Start", "End", "ReliabilityBlock"}
    checks.append(
        _check(
            "rbd_node_types",
            all(_label(vertex) in allowed_labels for vertex in vertices.values()),
            "RBD 节点类型必须属于 START、END 或 ReliabilityBlock",
        )
    )
    if len(starts) == 1 and len(ends) == 1:
        from_start = _reachable(starts[0], forward)
        to_end = _reachable(ends[0], reverse)
        checks.extend(
            [
                _check("rbd_success_path", ends[0] in from_start, "RBD 必须存在 START 到 END 的成功路径"),
                _check(
                    "rbd_all_nodes_on_path",
                    from_start & to_end == set(vertices),
                    "RBD 所有节点都必须位于某条 START 到 END 的路径上",
                ),
            ]
        )
    return checks


def _function_locator_checks(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    vertices, edges = _graph_parts(graph)
    allowed_labels = {"System", "Subsystem", "Module", "Function", "Interface"}
    relation_labels = {
        "contains",
        "implements",
        "connects_to",
        "data_flow",
        "depends_on",
    }
    checks = [
        _check(
            "architecture_nonempty",
            bool(vertices),
            "系统架构图至少需要一个明确实体节点",
        ),
        _check(
            "architecture_node_types",
            all(_label(vertex) in allowed_labels for vertex in vertices.values()),
            "系统架构节点必须属于系统、子系统、模块、功能或接口",
        ),
        _check(
            "architecture_relation_types",
            all(clean_text(edge.get("label")) in relation_labels for edge in edges),
            "系统架构关系必须属于包含、实现、连接、数据流或依赖",
        ),
    ]
    # A one-node system declaration is still valid source structure.  Larger
    # graphs must not contain completely detached nodes.
    if len(vertices) > 1:
        forward, reverse = _adjacency(vertices, edges)
        attached = {
            node_id
            for node_id in vertices
            if forward.get(node_id) or reverse.get(node_id)
        }
        checks.append(
            _check(
                "architecture_no_detached_nodes",
                attached == set(vertices),
                "多节点系统架构不能包含完全游离的实体",
            )
        )
    return checks


def _candidate_source_kinds(candidates: Iterable[Mapping[str, Any]]) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                clean_text(item.get("source_kind"))
                for item in candidates
                if clean_text(item.get("source_kind"))
                and clean_text(item.get("merge_role") or "included") not in {"isolated", "rejected"}
                and clean_text(item.get("extraction_status") or "succeeded") == "succeeded"
            }
        )
    )


def _vision_confidence(candidates: Iterable[Mapping[str, Any]]) -> float | None:
    values: List[float] = []
    for candidate in candidates:
        if clean_text(candidate.get("source_kind")) != "image_vision":
            continue
        if clean_text(candidate.get("merge_role") or "included") in {"isolated", "rejected"}:
            continue
        graph = candidate.get("normalized_graph") or candidate.get("raw_graph") or {}
        metadata = _mapping(graph.get("metadata") if isinstance(graph, Mapping) else {})
        try:
            value = float(metadata.get("confidence"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            values.append(value)
    return min(values) if values else None


def _blocking_extraction_warnings(candidates: Iterable[Mapping[str, Any]]) -> List[str]:
    blocking_tokens = ("无法唯一", "歧义", "缺少显式拓扑", "不得确认", "拓扑不完整", "未对应")
    warnings: List[str] = []
    for candidate in candidates:
        if clean_text(candidate.get("merge_role") or "included") in {"isolated", "rejected"}:
            continue
        graph = candidate.get("normalized_graph") or candidate.get("raw_graph") or {}
        metadata = _mapping(graph.get("metadata") if isinstance(graph, Mapping) else {})
        raw_warnings = metadata.get("extraction_warnings") or []
        if isinstance(raw_warnings, str):
            raw_warnings = [raw_warnings]
        for warning in raw_warnings if isinstance(raw_warnings, list) else []:
            text = clean_text(warning)
            if text and any(token in text for token in blocking_tokens):
                warnings.append(text)
    return list(dict.fromkeys(warnings))


def evaluate_automatic_validation(
    *,
    diagram_type: str,
    graph: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    merge_conflicts: Sequence[Mapping[str, Any]] = (),
) -> AutomaticValidationResult:
    """Return a deterministic, auditable decision for one canonical revision."""

    dtype = clean_text(diagram_type).casefold()
    source_kinds = _candidate_source_kinds(candidates)
    confidence = _vision_confidence(candidates)
    extraction_warnings = _blocking_extraction_warnings(candidates)
    checks: List[Dict[str, Any]] = []

    structural_conflicts = detect_graph_conflicts(graph)
    blocking_merge_conflicts = [item for item in merge_conflicts if item.get("blocking", True)]
    checks.extend(
        [
            _check(
                "candidate_merge_conflicts",
                not blocking_merge_conflicts,
                "候选来源之间不能存在未裁决的拓扑或关键属性冲突",
            ),
            _check(
                "canonical_structure",
                not any(item.blocking for item in structural_conflicts),
                "规范图不能包含空图、重复节点、悬空边、自环或缺失根节点",
            ),
            _check(
                "extraction_completeness",
                not extraction_warnings,
                "显式结构抽取不能包含未解析的歧义、缺失拓扑或成员映射",
            ),
        ]
    )
    if dtype == "fta":
        checks.extend(_fta_checks(graph))
        checks.extend(_engineering_parameter_checks(graph))
    elif dtype == "rbd":
        checks.extend(_rbd_checks(graph))
        checks.extend(_engineering_parameter_checks(graph))
    elif dtype == "function_locator":
        checks.extend(_function_locator_checks(graph))
    else:
        checks.append(_check("supported_type", False, f"暂不支持自动投影的图类型: {dtype}"))

    failed_messages = [item["message"] for item in checks if item["blocking"] and not item["passed"]]
    if failed_messages:
        status = VALIDATION_QUARANTINED
        tier = "none"
    else:
        deterministic = bool(set(source_kinds) & _DETERMINISTIC_TOPOLOGY_SOURCES)
        has_vision = "image_vision" in source_kinds
        if deterministic:
            status = VALIDATION_AUTO_VALIDATED
            tier = "corroborated" if has_vision else "deterministic_source"
        elif has_vision and confidence is not None and confidence >= VISION_CONFIDENCE_THRESHOLD:
            status = VALIDATION_PROVISIONAL
            tier = "visual_structural"
        elif has_vision:
            status = VALIDATION_QUARANTINED
            tier = "none"
            failed_messages.append(
                "视觉图缺少可用置信度或置信度低于 "
                f"{VISION_CONFIDENCE_THRESHOLD:.2f}，不能自动入图"
            )
        else:
            status = VALIDATION_QUARANTINED
            tier = "none"
            failed_messages.append("没有可授权自动入图的显式结构或视觉候选")

    return AutomaticValidationResult(
        status=status,
        tier=tier,
        method="automatic",
        policy_version=POLICY_VERSION,
        publishable=status in PROJECTABLE_VALIDATION_STATUSES,
        formal_claims_allowed=status in FORMAL_VALIDATION_STATUSES,
        source_kinds=source_kinds,
        confidence=confidence,
        checks=tuple(checks),
        reasons=tuple(dict.fromkeys(failed_messages)),
    )


def deterministic_candidate_allows_formal_topology(
    *,
    diagram_type: str,
    source_kind: str,
    graph: Mapping[str, Any],
) -> bool:
    """Return whether one deterministic candidate can anchor canonical topology.

    A source label alone is not sufficient: an empty table candidate or a
    malformed expression must not suppress an otherwise inspectable visual
    candidate.  Reusing the full automatic validator keeps this decision in
    sync with the constraints applied after revision creation.
    """

    kind = clean_text(source_kind)
    if kind not in _DETERMINISTIC_TOPOLOGY_SOURCES or not isinstance(graph, Mapping):
        return False
    decision = evaluate_automatic_validation(
        diagram_type=diagram_type,
        graph=graph,
        candidates=(
            {
                "candidate_id": "deterministic-topology-probe",
                "source_kind": kind,
                "merge_role": "included",
                "extraction_status": "succeeded",
                "normalized_graph": graph,
            },
        ),
    )
    return (
        decision.status == VALIDATION_AUTO_VALIDATED
        and decision.formal_claims_allowed
    )


def revision_is_projectable(revision: Mapping[str, Any]) -> bool:
    return (
        clean_text(revision.get("review_status")) == "confirmed"
        or clean_text(revision.get("validation_status")) in PROJECTABLE_VALIDATION_STATUSES
    )


def revision_allows_formal_claims(revision: Mapping[str, Any]) -> bool:
    return (
        clean_text(revision.get("review_status")) == "confirmed"
        or clean_text(revision.get("validation_status")) in FORMAL_VALIDATION_STATUSES
    )


__all__ = [
    "AutomaticValidationResult",
    "FORMAL_VALIDATION_STATUSES",
    "POLICY_VERSION",
    "PROJECTABLE_VALIDATION_STATUSES",
    "VALIDATION_AUTO_VALIDATED",
    "VALIDATION_FAILED",
    "VALIDATION_MANUAL_CONFIRMED",
    "VALIDATION_PENDING",
    "VALIDATION_PROVISIONAL",
    "VALIDATION_QUARANTINED",
    "VALIDATION_STATUSES",
    "VISION_CONFIDENCE_THRESHOLD",
    "deterministic_candidate_allows_formal_topology",
    "evaluate_automatic_validation",
    "revision_allows_formal_claims",
    "revision_is_projectable",
]
