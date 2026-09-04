"""Answer 节点·目标(架构/RBD/单记录聚焦)域策略模块（C1 批 5 从 answer.py 抽出）。

包含：RBD 结构与成功路径解析、目标面/集合核验、架构角色判定、
聚焦目标（closed collection / 单记录）完整性与降级策略等确定性函数。

依赖约定与其他域模块相同：共享 helper 留在 answer.py 内核，本模块顶部
对内核的引用在 answer.py 完整加载后执行（answer.py 底部再导入本模块）。
"""

import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.agent.answer_contract import focused_answer_contract, required_answer_fields
from backend.agent.query_understanding import extract_business_codes
from backend.agent.nodes.answer_presence import (
    approved_id_set,
    normalize_presence_text,
    semantic_tokens as _semantic_tokens,
)
from backend.agent.nodes.answer import (
    _COLLECTION_MEMBER_DETAIL_KEYS,
    _LEGACY_UNCITED_ASSERTION_PREFIX,
    _NO_SUPPORTED_FACTS_MESSAGE,
    _SCALAR_TARGET_FACET_LABELS,
    _TARGET_COMPLETENESS_FIELD_KEYS,
    _TARGET_COMPLETENESS_FIELD_LABELS,
    _TARGET_COMPLETENESS_LABELS,
    _UNIQUE_DIAGNOSIS_BOUNDARY,
    _approved_citation_numbers,
    _citation_ledger,
    _cleanup_generated_markdown_structure,
    _closed_collection_members_present,
    _collection_facet_view,
    _compact_semantic_text,
    _deterministic_ungrounded_lines,
    _diagram_identity_check_key,
    _evidence_answer_fields,
    _explicit_single_failure_clauses,
    _format_target_completion_manifest,
    _json_object,
    _line_has_uncited_assertion,
    _normalize_business_id,
    _parse_small_count,
    _strip_unbound_history_assertions,
    _target_completion_manifest,
    _target_facet_is_present,
    _target_scalar_values,
    _target_value_is_present,
    _topology_dependency,
)


def _rbd_explicit_business_ids(item: Mapping[str, Any]) -> set[str]:
    """Return only durable IDs explicitly carried by an RBD graph's blocks.

    Graph-local vertex IDs and block labels are deliberately excluded.  A
    label such as ``MSM`` may help retrieval, but cannot establish that an RBD
    structure is about a Function Locator component in another document.
    """

    if str(item.get("doc_type") or "") != "rbd":
        return set()
    graph = _json_object(item.get("content"))
    values: List[Any] = []
    for collection_name in ("vertices", "nodes"):
        collection = graph.get(collection_name) or []
        if not isinstance(collection, (list, tuple)):
            continue
        for vertex in collection:
            if not isinstance(vertex, Mapping):
                continue
            properties = (
                vertex.get("properties")
                if isinstance(vertex.get("properties"), Mapping)
                else {}
            )
            values.extend(
                (
                    vertex.get("business_id"),
                    vertex.get("module_id"),
                    vertex.get("component_id"),
                    vertex.get("function_id"),
                    properties.get("business_id"),
                    properties.get("module_id"),
                    properties.get("component_id"),
                    properties.get("function_id"),
                )
            )
    return {
        normalized
        for value in extract_business_codes(values, limit=96)
        if (normalized := _normalize_business_id(value))
    }


_UNCITED_EVIDENCE_BOUNDARY = (
    "证据边界：部分说明未附可直接核验的来源，仅供参考，不作为已确认结论。"
)


def _ensure_explicit_rbd_scope_boundary(
    answer: str,
    state: Dict[str, Any],
    *,
    citations: Sequence[Mapping[str, Any]] = (),
) -> str:
    """State the role boundary of a successfully requested RBD source.

    A successful RBD lookup should not be mistaken for root-cause proof merely
    because the same answer also contains FMECA causes.  This is a stable
    document-role boundary, appended only when a gate-approved RBD structure
    is actually available and the answer did not already express it.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    requested = {
        str(value or "").strip()
        for value in (
            state.get("requested_source_tools")
            or frame.get("requested_source_tools")
            or []
        )
        if str(value or "").strip()
    }
    if "rbd_query" not in requested:
        return str(answer or "").strip()

    current = str(answer or "").strip()
    normalized = _compact_semantic_text(current)
    if "rbd" in normalized and re.search(r"rbd.{0,80}(?:不能|无法|不用于).{0,100}(?:原因|根因|维修|处置)", normalized):
        return current

    if citations:
        citation = next(
            (
                int(item.get("id") or 0)
                for item in citations
                if str(item.get("doc_type") or "") == "rbd"
                and str(item.get("type") or item.get("claim_type") or "")
                == "rbd_structure"
            ),
            0,
        )
    else:
        citation = next(
            (
                number
                for number, item in _citation_ledger(state).items()
                if str(item.get("doc_type") or "") == "rbd"
                and str(item.get("claim_type") or "") == "rbd_structure"
            ),
            0,
        )
    if not citation:
        return current
    boundary = (
        "- **RBD 的结论边界**：该资料用于说明成功路径和冗余依赖，"
        "不能单独确认所询对象的根因或维修动作；这些仍需以 FMECA、"
        "HMC/MFL 或故障历史的对应记录为准。"
        f"[^{citation}]"
    )
    return f"{current}\n\n{boundary}".strip()


def _ensure_approved_rbd_structure_cited(
    answer: str,
    state: Dict[str, Any],
    *,
    citations: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Guarantee at least one citation binds an approved RBD structure row.

    When rbd_query was authorized and gate-approved RBD structure evidence
    exists, the answer must keep the redundancy/success-criterion view bound
    to that row.  A local model answering an impact question from FMECA only
    may skip the approved RBD record entirely; this deterministic closure
    re-attaches the RBD source before public citation compaction so the
    reference survives.  Runs after the model/citation editors so their
    removal of an unused candidate cannot silently drop the row again.
    """

    current = str(answer or "").strip()
    if not current:
        return current
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    requested = {
        str(value or "").strip()
        for value in (
            state.get("requested_source_tools")
            or frame.get("requested_source_tools")
            or []
        )
        if str(value or "").strip()
    }
    rbd_executed = "rbd_query" in requested or any(
        str(result.get("tool") or "").strip() == "rbd_query"
        for result in (state.get("tool_results") or [])
        if isinstance(result, Mapping)
    )
    if not rbd_executed:
        return current
    # 已引用任一 RBD 结构记录 → 无需补引
    prompt_markers = {
        int(item.get("id") or 0)
        for item in (citations or [])
        if str(item.get("doc_type") or "") == "rbd"
        and str(item.get("type") or item.get("claim_type") or "") == "rbd_structure"
    } if citations else {
        number
        for number, item in _citation_ledger(state).items()
        if str(item.get("doc_type") or "") == "rbd"
        and str(item.get("claim_type") or "") == "rbd_structure"
    }
    if any(f"[^{number}]" in current for number in prompt_markers):
        return current
    # 找一条已批准 RBD 结构证据追加
    approved_numbers = _approved_citation_numbers(state)
    ledger = _citation_ledger(state)
    selected = next(
        (
            (number, ledger[number])
            for number in prompt_markers
            if number in approved_numbers and number in ledger
        ),
        None,
    )
    if not selected:
        return current
    number, item = selected
    summary = str(item.get("display_summary") or item.get("content") or "")
    criterion = ""
    if summary:
        match = re.search(r"成功判据\s*[:：]\s*([^。；]*)", summary)
        if match:
            criterion = match.group(1).strip()
    appended = [
        "- **RBD 可靠性结构**："
        + (
            f"成功判据：{criterion}。"
            if criterion
            else "该资料说明目标系统的冗余依赖与成功路径。"
        )
        + f"[^{number}]"
    ]
    return f"{current}\n\n" + "\n".join(appended)


def _downgrade_unapproved_assertions(answer: str, state: Dict[str, Any]) -> str:
    """Remove factual prose that citation repair could not bind safely.

    A global warning forces users to guess which sentence is unsupported.  The
    answer model now gets one citation-editing opportunity; this final guard
    fails closed by removing only the remaining unbound factual lines.
    """
    if not (state.get("evidence_registry") or []) and (
        state.get("domain_results") or state.get("vector_results")
    ):
        # Generic-RAG compatibility: these rows predate the unified evidence
        # gate, so their valid prompt markers cannot be assessed through
        # approved_evidence_ids.  Final compaction still removes unreferenced
        # rows from the public citations list.
        return _cleanup_generated_markdown_structure(answer)

    approved_numbers = _approved_citation_numbers(state)
    exempt_lines = _deterministic_ungrounded_lines(state)

    lines: List[str] = []
    has_evidence_boundary = False
    for line in (answer or "").splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        content = line.lstrip()
        if content.startswith(_LEGACY_UNCITED_ASSERTION_PREFIX):
            content = content[len(_LEGACY_UNCITED_ASSERTION_PREFIX):].lstrip()
            line = f"{indent}{content}"
        stripped = line.strip()
        if stripped == _UNCITED_EVIDENCE_BOUNDARY:
            continue
        if stripped.startswith("证据边界："):
            if stripped != _UNIQUE_DIAGNOSIS_BOUNDARY or has_evidence_boundary:
                continue
            has_evidence_boundary = True
            lines.append(line)
            continue
        if not stripped:
            lines.append(line)
            continue
        if _line_has_uncited_assertion(line, approved_numbers, exempt_lines):
            continue
        lines.append(line)

    result = _cleanup_generated_markdown_structure("\n".join(lines))
    result = _strip_unbound_history_assertions(result, state)
    substantive = [
        line.strip()
        for line in result.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return result if substantive else _NO_SUPPORTED_FACTS_MESSAGE


_TARGET_COMPLETENESS_DOC_TYPES = {
    "record_name": {"function_locator"},
    "record_type": {"function_locator"},
    "record_location": {"function_locator"},
    "responsibility": {"function_locator"},
    "failure_mode": {"fmeca", "mfl"},
    "fault_cause": {"fmeca", "fta"},
    "structural_cause_chain": {"fta"},
    "fault_location": {"function_locator", "mfl"},
    "local_impact": {"fmeca"},
    "final_impact": {"fmeca"},
    # Relation paths may be PostgreSQL-re-authorized HugeGraph projections
    # (which intentionally have no document type), or formal FTA structure.
    # The field manifest still prevents arbitrary prose from satisfying it.
    "impact_propagation": set(),
    "severity": {"fmeca"},
    "failure_probability": {"fmeca"},
    "failure_probability_level": {"fmeca"},
    "detection_method": {"fmeca", "mfl"},
    "mission_phase": {"fmeca", "fault_history"},
    "design_improvement": {"fmeca"},
    "compensation_action": {"fmeca"},
    "platform_test_status": {"mfl"},
    "platform_test_result": {"mfl"},
    "history_time": {"fault_history"},
    "history_root": {"fault_history"},
    "history_steps": {"fault_history"},
    "history_solution": {"fault_history"},
    "history_result": {"fault_history"},
    "history_personnel": {"fault_history"},
    "history_preventive_action": {"fault_history"},
}


_RBD_FORMAL_AUTHORITIES = {"confirmed_relation", "validated_relation"}


_RBD_NON_FORMAL_REVIEW_STATUSES = {
    "pending",
    "provisional",
    "quarantined",
    "failed",
    "rejected",
}


_RBD_CONNECTION_LABELS = {
    "series": "串联",
    "parallel": "并联",
    "redundant": "冗余",
    "bypass": "旁路",
}


_RBD_REDUNDANT_CONNECTION_TYPES = {"parallel", "redundant", "bypass"}


def _rbd_structure_is_formal(item: Mapping[str, Any]) -> bool:
    """Reject a non-formal RBD revision even if its authority is inconsistent.

    Evidence gate normally keeps these states consistent.  The answer layer
    still fails closed so a stale or synthetic ``validated_relation`` cannot
    promote a provisional visual extraction into an operational statement.
    ``not_required`` remains accepted for legacy, already-authorized evidence.
    """

    if (
        str(item.get("doc_type") or "") != "rbd"
        or str(item.get("claim_type") or "") != "rbd_structure"
        or str(item.get("authority") or "") not in _RBD_FORMAL_AUTHORITIES
    ):
        return False
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    graph = _json_object(item.get("content"))
    graph_metadata = (
        graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
    )
    statuses = {
        str(value or "").strip().casefold()
        for value in (
            item.get("review_status"),
            metadata.get("validation_status"),
            graph_metadata.get("validation_status"),
        )
        if str(value or "").strip()
    }
    return not bool(statuses & _RBD_NON_FORMAL_REVIEW_STATUSES)


def _rbd_group_key(branch_id: Any) -> str:
    value = str(branch_id or "").strip()
    if not value:
        return ""
    if ":branch:" in value:
        return value.rsplit(":branch:", 1)[0]
    return value


def _rbd_parallel_groups(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Collect explicit redundancy groups and separate IDs from local labels.

    Deterministic expression graphs persist branch metadata on vertices, while
    some validated visual/legacy graphs persist it only on edges.  Both forms
    describe the same explicit branch group.  Edge groups inherit values only
    from their resolved endpoint vertices.  ``business_ids`` is deliberately
    restricted to fields which the graph schema declares as durable business
    identity; local vertex IDs and display labels remain useful for matching a
    criterion clause inside this graph, but can never authorize a cross-document
    entity handoff.
    """

    grouped: Dict[str, Dict[str, Any]] = {}
    raw_vertices: List[Mapping[str, Any]] = []
    for collection_name in ("vertices", "nodes"):
        collection = graph.get(collection_name) or []
        if isinstance(collection, (list, tuple)):
            raw_vertices.extend(
                value for value in collection if isinstance(value, Mapping)
            )

    vertices_by_id = {
        str(vertex.get("id") or "").strip(): vertex
        for vertex in raw_vertices
        if str(vertex.get("id") or "").strip()
    }

    def add_unique(group: Dict[str, Any], field: str, value: Any) -> None:
        rendered = str(value or "").strip()
        if rendered and rendered not in group[field]:
            group[field].append(rendered)

    def add_vertex_values(
        group: Dict[str, Any], vertex: Mapping[str, Any]
    ) -> None:
        properties = vertex.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}

        # Only explicit schema identity fields may cross the document boundary.
        # In particular, ``vertex.id`` is commonly a graph-local locator and a
        # human-readable name may happen to look like an engineering code.
        for value in (
            vertex.get("business_id"),
            vertex.get("module_id"),
            vertex.get("function_id"),
            properties.get("business_id"),
            properties.get("module_id"),
            properties.get("function_id"),
        ):
            add_unique(group, "business_ids", value)

        # These values are scoped to the current RBD only.  Include the durable
        # IDs as well so an explicit criterion such as ``FUNC-120 至少一路`` can
        # still be located without weakening cross-document authorization.
        for value in (
            *(group.get("business_ids") or []),
            vertex.get("id"),
            vertex.get("name"),
            vertex.get("module_name"),
            vertex.get("abbreviation"),
            vertex.get("short_name"),
            vertex.get("original_id"),
            properties.get("name"),
            properties.get("module_name"),
            properties.get("abbreviation"),
            properties.get("short_name"),
            properties.get("original_id"),
        ):
            add_unique(group, "labels", value)

    for vertex in raw_vertices:
        properties = vertex.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        connection_type = str(
            properties.get("connection_type") or ""
        ).strip().casefold()
        group_key = _rbd_group_key(properties.get("branch_id"))
        if connection_type not in _RBD_REDUNDANT_CONNECTION_TYPES or not group_key:
            continue
        group = grouped.setdefault(
            group_key,
            {
                "key": group_key,
                "connection_type": connection_type,
                "business_ids": [],
                "labels": [],
            },
        )
        add_vertex_values(group, vertex)

    raw_edges = graph.get("edges") or []
    if isinstance(raw_edges, (list, tuple)):
        for edge in raw_edges:
            if not isinstance(edge, Mapping):
                continue
            properties = edge.get("properties")
            properties = properties if isinstance(properties, Mapping) else {}
            connection_type = str(
                properties.get("connection_type") or ""
            ).strip().casefold()
            group_key = _rbd_group_key(properties.get("branch_id"))
            if (
                connection_type not in _RBD_REDUNDANT_CONNECTION_TYPES
                or not group_key
            ):
                continue
            group = grouped.setdefault(
                group_key,
                {
                    "key": group_key,
                    "connection_type": connection_type,
                    "business_ids": [],
                    "labels": [],
                },
            )
            for endpoint_name in ("source", "target"):
                endpoint_id = str(edge.get(endpoint_name) or "").strip()
                endpoint = vertices_by_id.get(endpoint_id)
                if endpoint is not None:
                    add_vertex_values(group, endpoint)
    return [grouped[key] for key in sorted(grouped)]


def _longest_common_contiguous_text(left: str, right: str) -> int:
    """Length of deterministic lexical overlap; this is not fuzzy matching."""

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            size = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(size)
            if size > best:
                best = size
        previous = current
    return best


def _rbd_match_text(value: Any) -> str:
    """Normalize only established engineering wording within one RBD graph."""

    text = _compact_semantic_text(value)
    # Test documents and real tables alternate between “电源” and “供电” for
    # the same explicitly identified PSU branch.  This controlled vocabulary
    # normalization is local to criterion/group binding; it does not merge
    # entities or create cross-document identities.
    text = text.replace("稳压供电", "电源").replace("供电", "电源")
    return text


def _rbd_group_match_score(text: Any, group: Mapping[str, Any]) -> Tuple[int, int]:
    normalized = _rbd_match_text(text)
    if not normalized:
        return (0, 0)
    scores: List[int] = []
    local_values = [
        *(group.get("business_ids") or []),
        *(group.get("labels") or []),
    ]
    for value in dict.fromkeys(str(item or "") for item in local_values):
        identity = _rbd_match_text(value)
        if not identity:
            continue
        if normalized == identity:
            score = len(identity) + 100
        elif normalized in identity or identity in normalized:
            score = min(len(normalized), len(identity)) + 20
        else:
            score = _longest_common_contiguous_text(normalized, identity)
        scores.append(score)
    scores.sort(reverse=True)
    return (scores[0] if scores else 0, scores[1] if len(scores) > 1 else 0)


def _rbd_group_matches_subject_kind(
    group: Mapping[str, Any], subject_kind: Any
) -> bool:
    kind = str(subject_kind or "").strip().casefold()
    if kind not in {"supply_path", "power_path", "data_path", "signal_path"}:
        return True
    identities = _compact_semantic_text(
        " ".join(
            [
                *(group.get("business_ids") or []),
                *(group.get("labels") or []),
            ]
        )
    )
    patterns = {
        "supply_path": r"电源|供电|psu|pwr",
        "power_path": r"电源|供电|psu|pwr",
        "data_path": r"数据|总线|通信|网络|链路|bca|bcb",
        "signal_path": r"信号|通信|链路|通道|接口",
    }
    return bool(re.search(patterns[kind], identities, re.IGNORECASE))


def _unique_rbd_group_match(
    groups: Sequence[Mapping[str, Any]],
    text: Any,
    *,
    subject_kind: Any = "",
) -> str:
    """Bind wording to one explicit branch group or fail closed on ambiguity."""

    ranked = [
        (_rbd_group_match_score(text, group), str(group.get("key") or ""))
        for group in groups
        if _rbd_group_matches_subject_kind(group, subject_kind)
    ]
    ranked = [value for value in ranked if value[0][0] >= 2 and value[1]]
    if not ranked:
        return ""
    ranked.sort(key=lambda value: (value[0][0], value[0][1], value[1]), reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return ""
    return ranked[0][1]


def _unique_rbd_group_match_by_business_ids(
    groups: Sequence[Mapping[str, Any]], business_ids: Any
) -> str:
    """Bind a cross-document target to exactly one persisted RBD group.

    Names and vector similarity are useful for retrieval, but they cannot
    authorize a cross-document topology conclusion. Only exact normalized
    equality against a durable module/function/code identity is accepted; no
    match or more than one matching group fails closed.
    """

    raw_ids = business_ids if isinstance(business_ids, (list, tuple, set)) else []
    normalized_ids = {
        _normalize_business_id(value)
        for value in raw_ids
        if _normalize_business_id(value)
    }
    if not normalized_ids:
        return ""
    matching_groups: List[str] = []
    for group in groups:
        group_ids = {
            _normalize_business_id(value)
            for value in group.get("business_ids") or []
            if _normalize_business_id(value)
        }
        key = str(group.get("key") or "")
        if key and normalized_ids & group_ids:
            matching_groups.append(key)
    unique_groups = list(dict.fromkeys(matching_groups))
    return unique_groups[0] if len(unique_groups) == 1 else ""


def _rbd_target_boundary_clauses(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> List[str]:
    """Return boundary clauses whose redundancy group is the queried object."""

    dependency = _topology_dependency(state)
    fields = _rbd_structure_answer_fields(item)
    clauses = _flatten_target_values(fields.get("single_failure_boundary"))
    if not dependency:
        return clauses
    graph = _json_object(item.get("content"))
    groups = _rbd_parallel_groups(graph)
    subject_group = _unique_rbd_group_match_by_business_ids(
        groups,
        dependency.get("canonical_business_ids"),
    )
    if not subject_group:
        return []
    return [
        clause
        for clause in clauses
        if _unique_rbd_group_match(groups, clause) == subject_group
    ]


def _rbd_matches_topology_dependency(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    if not _topology_dependency(state):
        return True
    if str(item.get("doc_type") or "") != "rbd":
        return True
    # The target-group boundary applies only to source RBD structure.  A
    # deterministic calculator result has already been authorized against its
    # selected formal graph and must not be reinterpreted as a graph payload;
    # it does not carry branch vertices or single-failure clauses to match.
    if str(item.get("claim_type") or "") != "rbd_structure":
        return True
    return bool(_rbd_target_boundary_clauses(item, state))


def _rbd_structure_answer_fields_for_state(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> Mapping[str, Any]:
    fields = dict(_rbd_structure_answer_fields(item))
    if str(item.get("doc_type") or "") != "rbd":
        return fields
    if not _topology_dependency(state):
        # A success criterion may be quoted as source information without a
        # target binding. A single-path failure boundary, however, is an
        # operational conclusion about one particular redundancy group and
        # remains hidden until that group is bound by exact business IDs.
        fields.pop("single_failure_boundary", None)
        return fields
    clauses = _rbd_target_boundary_clauses(item, state)
    if not clauses:
        return {}
    fields["single_failure_boundary"] = clauses[0] if len(clauses) == 1 else clauses
    return fields


def _first_structured_value(
    containers: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Any:
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _rbd_structure_answer_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project only explicit, approved RBD JSON fields into the answer contract.

    RBD query evidence predates ``metadata.answer_fields`` and therefore keeps
    the versioned graph in ``content``.  This adapter deliberately recognizes a
    small whitelist instead of asking the LLM to interpret the raw graph.  It
    does not enumerate paths, calculate reliability, or infer fault tolerance
    from topology.
    """

    fields = dict(_evidence_answer_fields(item))
    if (
        str(item.get("doc_type") or "") != "rbd"
        or str(item.get("claim_type") or "") != "rbd_structure"
    ):
        return fields
    if not _rbd_structure_is_formal(item):
        return {}
    graph = _json_object(item.get("content"))
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
    containers = (fields, metadata, graph)

    aliases = {
        "success_criterion": ("success_criterion", "success_criteria"),
        "structure_description": (
            "structure",
            "structure_description",
            "topology_description",
        ),
        "structure_expression": ("structure_expression", "topology_expression"),
        "path_structure": ("paths", "success_paths", "series_paths"),
        "redundancy_structure": (
            "redundancy_structure",
            "parallel_branches",
            "redundant_paths",
            "redundancy",
        ),
        "single_failure_boundary": (
            "single_failure_boundary",
            "single_path_failure_boundary",
            "single_path_failure_capability",
            "single_failure_tolerance",
            "failure_tolerance",
            "capability_boundary",
        ),
        "missing_parameters": (
            "missing_parameters",
            "missing_reliability_blocks",
            "parameter_gaps",
            "missing_inputs",
        ),
    }
    for target, keys in aliases.items():
        value = _first_structured_value(containers, keys)
        if value not in (None, "", [], {}):
            fields[target] = value

    connection_types: List[str] = []
    for raw_edge in graph.get("edges") or []:
        if not isinstance(raw_edge, Mapping):
            continue
        properties = raw_edge.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        raw_type = str(properties.get("connection_type") or "").strip().casefold()
        label = _RBD_CONNECTION_LABELS.get(raw_type)
        if label and label not in connection_types:
            connection_types.append(label)
    if connection_types:
        fields.setdefault("connection_types", connection_types)
        redundant_types = [
            value for value in connection_types if value in {"并联", "冗余", "旁路"}
        ]
        if redundant_types:
            fields.setdefault("redundancy_structure", redundant_types)

    # Some source tables explicitly mark which columns are missing.  Preserve
    # those declarations, but do not infer a gap merely because a property is
    # absent from a graph vertex.
    declared_missing: List[str] = []
    parameters = metadata.get("parameters")
    if isinstance(parameters, (list, tuple)):
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                continue
            name = str(
                parameter.get("module_name")
                or parameter.get("name")
                or parameter.get("module_id")
                or "参数项"
            ).strip()
            raw_missing = parameter.get("missing_fields") or parameter.get("missing_parameters")
            values = raw_missing if isinstance(raw_missing, (list, tuple, set)) else [raw_missing]
            for value in values:
                label = str(value or "").strip()
                rendered = f"{name}：{label}" if name and label else label
                if rendered and rendered not in declared_missing:
                    declared_missing.append(rendered)
    if declared_missing:
        existing = _flatten_target_values(fields.get("missing_parameters"))
        fields["missing_parameters"] = list(dict.fromkeys([*existing, *declared_missing]))

    # This is not a topology inference: only wording that explicitly states a
    # one-path-failure condition can satisfy the capability-boundary facet.
    boundary_clauses = _explicit_single_failure_clauses(
        fields.get("success_criterion")
    )
    if boundary_clauses and not fields.get("single_failure_boundary"):
        fields["single_failure_boundary"] = (
            boundary_clauses[0]
            if len(boundary_clauses) == 1
            else boundary_clauses
        )
    return fields


def _target_facets_for_evidence(
    item: Mapping[str, Any],
    intents: set[str],
    *,
    fields: Mapping[str, Any] | None = None,
    query_text: str = "",
    required_fields: Sequence[str] = (),
) -> List[str]:
    """Map approved structured fields to explicit, non-computational goals."""

    fields = fields if fields is not None else _evidence_answer_fields(item)
    if not fields:
        return []
    keys = {str(key) for key, value in fields.items() if value not in (None, "", [], {})}
    doc_type = str(item.get("doc_type") or "")
    claim_type = str(item.get("claim_type") or "")
    facets: List[str] = []

    def add(name: str) -> None:
        allowed_doc_types = _TARGET_COMPLETENESS_DOC_TYPES.get(name)
        if allowed_doc_types and doc_type not in allowed_doc_types:
            return
        if keys & _TARGET_COMPLETENESS_FIELD_KEYS[name] and name not in facets:
            facets.append(name)

    # The planning model has already translated the user's current question
    # into a validated field contract.  Map only those declared fields to
    # approved structured evidence; never infer extra answer goals from raw
    # query words at this stage.
    for name in required_fields:
        if name in _TARGET_COMPLETENESS_FIELD_KEYS:
            add(name)
    if required_fields:
        return facets

    if (
        fields.get("collection_closed") is True
        and str(fields.get("collection_role") or "") == "member"
        and fields.get("collection_parent_id")
        and fields.get("collection_member_count")
    ):
        add("closed_collection_members")

    normalized_query = _compact_semantic_text(query_text)
    identity_pairs = (
        ("function_id", "function_name"),
        ("component_id", "component_name"),
        ("interface_id", "signal_name"),
        ("record_code", "failure_mode"),
        ("record_code", "module_failure"),
        ("record_code", "sur_failure"),
        ("record_code", "subcircuit_failure"),
        ("failure_mode_id", "failure_mode"),
        ("case_id", "title"),
        ("case_id", "fault_symptom"),
    )
    for identifier_key, name_key in identity_pairs:
        identifier = _compact_semantic_text(fields.get(identifier_key))
        if (
            len(identifier) >= 4
            and identifier in normalized_query
            and fields.get(name_key)
        ):
            add("record_identity")
            break

    if "exact_code" in intents and doc_type == "hmc":
        add("hmc_identity")
        add("hmc_symptom")
        # The exact-code answer contract explicitly asks what the record says
        # and how the same record says to handle it.  This is not a new
        # maintenance intent; it is a field of the exact HMC result itself.
        if claim_type in {"maintenance_action", "hmc_fault", "hmc_maintenance"}:
            add("maintenance_action")

    if "maintenance_query" in intents:
        if doc_type == "mfl":
            add("maintenance_locator")
            add("maintenance_api")
        if doc_type in {"hmc", "mfl", "fmeca"}:
            add("maintenance_action")

    if intents & {"architecture_query", "entity_profile", "function_location"}:
        if doc_type == "function_locator":
            add("architecture_hierarchy")
            add("architecture_function")
            add("architecture_interface")
        if claim_type == "relation_path" or str(item.get("source_kind") or "") == "hugegraph":
            add("relation_path")
    if (
        "reliability_query" in intents
        and doc_type == "rbd"
        and claim_type == "rbd_structure"
        and _rbd_structure_is_formal(item)
    ):
        add("rbd_success_criterion")
        add("rbd_path_structure")
        add("rbd_redundancy_structure")
        add("rbd_single_failure_boundary")
        add("rbd_missing_parameters")
    return facets


def _flatten_target_values(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        result: List[str] = []
        for nested in value.values():
            result.extend(_flatten_target_values(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for nested in value:
            result.extend(_flatten_target_values(nested))
        return result
    if isinstance(value, bool):
        return ["是" if value else "否"]
    text = str(value or "").strip()
    return [text] if text else []


_ARCHITECTURE_ROLE_QUERY_PATTERNS = {
    "receivers": re.compile(
        r"(?:哪些|什么|谁|哪个|哪几个).{0,18}(?:接收|接入|收发)"
        r"|(?:接收|接入|收发).{0,18}(?:哪些|什么|谁|哪个|哪几个)"
    ),
    "responsibility": re.compile(
        r"(?:谁|哪个|哪一|什么).{0,18}(?:负责|承担)"
        r"|(?:负责|承担).{0,18}(?:谁|哪个|哪一|什么)"
    ),
    "downstream": re.compile(
        r"(?:之后|随后|然后|后续).{0,18}(?:交给|送入|送往|流向|传给)"
        r"|(?:交给|送入|送往|流向|传给).{0,18}(?:谁|哪里|哪个|哪一|什么)"
    ),
}


# ---- target 域专属常量（随本模块迁移）----
_ARCHITECTURE_ROLE_IGNORED_BIGRAMS = {
    "哪些", "什么", "哪个", "几个", "单元", "模块", "组件", "接收", "接入",
    "收发", "负责", "承担", "之后", "随后", "然后", "后续", "交给", "送入",
    "送往", "流向", "传给", "信息", "数据", "子系", "系统", "功能",
}


def _closed_collection_group_details(
    items: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return complete, explicitly closed direct-member group metadata.

    The repository stamps these fields only after it has read every direct
    child of one PostgreSQL parent.  A group is ignored if its declared count
    disagrees with the unique member IDs available in the current evidence
    set; this prevents a bounded/truncated query from being presented as a
    complete architecture.
    """

    grouped: Dict[Tuple[str, str, str, int], Dict[str, int]] = {}
    for index, item in enumerate(items):
        fields = _architecture_role_fields(item)
        if (
            fields.get("collection_closed") is not True
            or str(fields.get("collection_role") or "") != "member"
        ):
            continue
        parent_id = str(fields.get("collection_parent_id") or "").strip()
        member_type = str(fields.get("collection_member_kind") or "").strip()
        member_id = str(
            fields.get("collection_member_id") or fields.get("component_id") or ""
        ).strip()
        try:
            expected_count = int(fields.get("collection_member_count") or 0)
        except (TypeError, ValueError):
            expected_count = 0
        if not parent_id or not member_type or not member_id or expected_count <= 0:
            continue
        key = (
            str(item.get("doc_id") or ""),
            parent_id,
            member_type,
            expected_count,
        )
        grouped.setdefault(key, {}).setdefault(member_id, index)

    complete: List[Dict[str, Any]] = []
    for (doc_id, parent_id, member_type, expected_count), members in grouped.items():
        if len(members) != expected_count:
            continue
        indexes = [members[member_id] for member_id in sorted(members)]
        first_fields = _architecture_role_fields(items[indexes[0]])
        complete.append(
            {
                "doc_id": doc_id,
                "parent_id": parent_id,
                "parent_name": str(
                    first_fields.get("collection_parent_name") or ""
                ).strip(),
                "member_kind": member_type,
                "expected_count": expected_count,
                "indexes": indexes,
            }
        )
    return complete


def _closed_collection_groups(
    items: Sequence[Mapping[str, Any]],
) -> List[List[int]]:
    """Compatibility view containing only indexes for each closed group."""

    return [
        list(group["indexes"])
        for group in _closed_collection_group_details(items)
    ]


def _requested_closed_collection_group(
    items: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Select the one closed group that answers this turn's collection slot.

    An unrelated closed architecture inventory must never unlock exhaustive
    FMECA wording.  The tool-resolved parent business ID is preferred; legacy
    states without it may use an exact parent name, but only when that leaves
    one group.  A request for causes is represented as a complete
    failure-mode→source-cause mapping and is closed only if every mode has a
    non-empty cause field.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") != "complete_collection":
        return None
    requested_kind = str(frame.get("collection_member_kind") or "").strip()
    stored_kind = "failure_mode" if requested_kind == "cause" else requested_kind
    groups = _closed_collection_group_details(items)
    if stored_kind:
        groups = [group for group in groups if group["member_kind"] == stored_kind]
    if not groups:
        return None

    resolution = (
        state.get("collection_resolution")
        if isinstance(state.get("collection_resolution"), Mapping)
        else {}
    )
    resolved_parent_id = str(resolution.get("parent_id") or "").strip()
    if resolved_parent_id:
        groups = [group for group in groups if group["parent_id"] == resolved_parent_id]
    else:
        subject = _compact_semantic_text(frame.get("collection_subject"))
        if subject:
            def scope_core(value: Any) -> str:
                return re.sub(
                    r"(?:全部|所有|完整|故障模式|故障原因|成因|原因|清单|列表|记录|"
                    r"子系统|系统|模块|组件|的)",
                    "",
                    _compact_semantic_text(value),
                )

            subject_core = scope_core(subject)
            matched: List[Dict[str, Any]] = []
            for group in groups:
                parent_id = _compact_semantic_text(group.get("parent_id"))
                parent_name = _compact_semantic_text(group.get("parent_name"))
                parent_core = scope_core(parent_name)
                if (
                    subject == parent_id
                    or subject == parent_name
                    or (len(parent_id) >= 4 and parent_id in subject)
                    or (len(parent_name) >= 4 and parent_name in subject)
                    or (
                        len(subject_core) >= 4
                        and subject_core == parent_core
                    )
                ):
                    matched.append(group)
            groups = matched
    if len(groups) != 1:
        return None

    selected = groups[0]
    if requested_kind == "cause":
        for index in selected["indexes"]:
            if str(items[index].get("doc_type") or "") != "fmeca":
                return None
            fields = _architecture_role_fields(items[index])
            if not str(fields.get("fault_cause") or "").strip():
                return None
    return selected


def _architecture_role_bigrams(value: Any) -> set[str]:
    text = _compact_semantic_text(value)
    return {
        text[index : index + 2]
        for index in range(max(len(text) - 1, 0))
        if text[index : index + 2] not in _ARCHITECTURE_ROLE_IGNORED_BIGRAMS
    }


def _architecture_role_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = item.get("answer_fields")
    if isinstance(fields, Mapping):
        return fields
    return _evidence_answer_fields(item)


def _architecture_role_text(item: Mapping[str, Any]) -> str:
    fields = _architecture_role_fields(item)
    return "\n".join(
        value
        for raw in fields.values()
        for value in _flatten_target_values(raw)
    )


def _architecture_role_overlap(query: str, item: Mapping[str, Any]) -> int:
    return len(
        _architecture_role_bigrams(query)
        & _architecture_role_bigrams(_architecture_role_text(item))
    )


def _architecture_role_hint(query: str, verbs: str) -> str:
    matches = list(re.finditer(
        rf"(?:{verbs})(?:由|为|是|给|到|入|往)?(.{{0,24}}?)(?:[，。；？！?]|之后|随后|然后|后续|$)",
        str(query or ""),
    ))
    if not matches:
        return str(query or "")
    # A flow question often mentions an incoming ``送入`` before asking
    # where data is ``交给`` next.  The last role verb describes the requested
    # target; using the first one accidentally reuses the upstream clause.
    match = matches[-1]
    return re.sub(r"^(?:谁|哪个|哪一|什么|哪些|哪几个)", "", match.group(1)).strip()


def _architecture_role_indices(
    query: str,
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, List[int]]:
    """Resolve user-requested architecture roles from approved structured fields.

    The selection is ID-agnostic.  It uses the question's grammatical role,
    explicit component/function types and overlap with source fields.  This
    makes plural questions require every matching receiver rather than letting
    one unrelated hierarchy mention satisfy the entire architecture target.
    """

    query_text = str(query or "")
    roles: Dict[str, List[int]] = {}
    indexed = [
        (index, item, _architecture_role_fields(item))
        for index, item in enumerate(items)
        if str(item.get("doc_type") or "") == "function_locator"
        and _architecture_role_fields(item)
    ]
    if not indexed:
        return roles

    if _ARCHITECTURE_ROLE_QUERY_PATTERNS["receivers"].search(query_text):
        receiver_candidates: List[Tuple[int, int, int]] = []
        for index, item, fields in indexed:
            description = " ".join(
                _flatten_target_values(fields.get("description"))
                + _flatten_target_values(fields.get("responsibility"))
                + _flatten_target_values(fields.get("function_name"))
            )
            if not re.search(r"接收|接入|收发|输入", description):
                continue
            overlap = _architecture_role_overlap(query_text, item)
            if overlap <= 0:
                continue
            is_module = str(fields.get("component_type") or "").lower() == "module"
            receiver_candidates.append((index, overlap, 1 if is_module else 0))
        if receiver_candidates:
            if any(value[2] for value in receiver_candidates):
                receiver_candidates = [value for value in receiver_candidates if value[2]]
            best = max(value[1] for value in receiver_candidates)
            plural = bool(re.search(r"哪些|哪几个|各(?:个|路|通道)|所有|双通道|两路", query_text))
            selected = [
                index
                for index, score, _is_module in receiver_candidates
                if score >= (max(1, best - 1) if plural else best)
            ][:6 if plural else 1]
            roles["architecture_receivers"] = selected

            # ``selected`` contains indexes into ``items``, not ``indexed``;
            # resolve parents directly to avoid depending on result order.
            parent_ids = {
                str(_architecture_role_fields(items[index]).get("parent_component_id") or "").strip()
                for index in selected
                if str(_architecture_role_fields(items[index]).get("parent_component_id") or "").strip()
            }
            parent_indexes = [
                index
                for index, _item, fields in indexed
                if str(fields.get("component_id") or "").strip() in parent_ids
            ]
            if parent_indexes:
                roles["architecture_receiver_context"] = parent_indexes[:3]

    if _ARCHITECTURE_ROLE_QUERY_PATTERNS["responsibility"].search(query_text):
        hint = _architecture_role_hint(query_text, r"负责|承担") or query_text
        component_candidates: List[Tuple[int, int]] = []
        function_candidates: List[Tuple[int, int]] = []
        for index, item, fields in indexed:
            score = _architecture_role_overlap(hint, item)
            if score <= 0:
                continue
            if fields.get("component_id") and str(fields.get("component_type") or "").lower() == "module":
                component_candidates.append((index, score))
            if fields.get("function_id"):
                function_candidates.append((index, score))
        if component_candidates:
            roles["architecture_responsible_component"] = [
                max(component_candidates, key=lambda value: value[1])[0]
            ]
        if function_candidates:
            roles["architecture_responsibility_function"] = [
                max(function_candidates, key=lambda value: value[1])[0]
            ]

    if _ARCHITECTURE_ROLE_QUERY_PATTERNS["downstream"].search(query_text):
        hint = _architecture_role_hint(
            query_text, r"交给|送入|送往|流向|传给"
        ) or query_text
        downstream_candidates: List[Tuple[int, int, int]] = []
        for index, item, fields in indexed:
            is_relation = int(
                bool(
                    fields.get("target_component_id")
                    or fields.get("target_entity")
                    or fields.get("path_summary")
                )
            )
            is_component = bool(fields.get("component_id"))
            if not is_relation and not is_component:
                continue
            score = _architecture_role_overlap(hint, item)
            if score <= 0:
                continue
            subsystem_bonus = int(
                str(fields.get("component_type") or "").lower() == "subsystem"
            )
            downstream_candidates.append((index, score, is_relation + subsystem_bonus))
        if downstream_candidates:
            roles["architecture_downstream"] = [
                max(downstream_candidates, key=lambda value: (value[1], value[2]))[0]
            ]
    return roles


def _architecture_declared_member_counts(answer: str, member_type: str) -> List[int]:
    label = {
        "subsystem": "子系统",
        "module": "模块",
        "system": "系统",
        "failure_mode": "故障模式",
    }.get(str(member_type or "").lower(), str(member_type or "").strip())
    if not label:
        return []
    numeral = r"\d{1,3}|[零一二两三四五六七八九十]{1,3}"
    patterns = (
        rf"(?:共|共有|包含|包括|覆盖|由|有|设有|直接包含|直接包括)"
        rf"\s*(?:记录|列出)?\s*({numeral})\s*(?:个|项|条)?\s*{re.escape(label)}",
        rf"{re.escape(label)}\s*(?:记录)?\s*"
        rf"(?:共|共有|包含|包括|覆盖|有|为|数量为|总数为)\s*"
        rf"({numeral})\s*(?:个|项|条)?",
        rf"以上\s*({numeral})\s*(?:个|项|条)?\s*{re.escape(label)}",
    )
    counts: List[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, str(answer or "")):
            parsed = _parse_small_count(match.group(1))
            if parsed >= 0 and parsed not in counts:
                counts.append(parsed)
    return counts


_ARCHITECTURE_ROLE_FACETS = {
    "closed_collection_members",
    "architecture_receivers",
    "architecture_receiver_context",
    "architecture_responsible_component",
    "architecture_responsibility_function",
    "architecture_downstream",
}


_NARRATIVE_GUARANTEE_FACETS = frozenset(
    {"structural_cause_chain", "impact_propagation", "fault_cause"}
)


def _append_missing_rbd_chain_identities(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Restate chain names when several RBD diagrams share one answer.

    A multi-chain enumeration must let the user tell which bullet belongs to
    which diagram, otherwise a chain discussed only through its blocks cannot
    be audited against the source it came from.  Only the titles of RBD
    diagrams already cited in the answer are copied, each bound to its existing
    ledger number; nothing is composed and single-diagram answers stay as-is.
    """

    current = str(answer or "").strip()
    if not current:
        return current
    ledger = _citation_ledger(state)
    cited_rbd: List[Tuple[int, str]] = []
    for citation, item in ledger.items():
        if str(item.get("doc_type") or "").strip() != "rbd":
            continue
        if f"[^{citation}]" not in current:
            continue
        identity = str(item.get("title") or "").strip()
        if not identity:
            identity = str(item.get("record_id") or "").strip()
        if not identity:
            continue
        cited_rbd.append((citation, identity))
    if len(cited_rbd) < 2:
        return current
    ordered = sorted(cited_rbd, key=lambda pair: pair[0])
    missing = [
        (citation, identity)
        for citation, identity in ordered
        if not _target_value_is_present(current, _diagram_identity_check_key(identity))
    ]
    if not missing:
        return current
    names = "、".join(
        f"{identity}[^{citation}]" for citation, identity in missing
    )
    appended = f"\n\n链路名称（用于区分以上各条已引用框图）：{names}。"
    return f"{current}{appended}"


def _append_missing_narrative_facts(
    answer: str,
    missing: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    """Restate approved FTA/propagation/FMECA-cause facts the editors dropped.

    Only exact gate-approved field values are copied, bound to their existing
    citations; nothing is composed, and facets already present stay untouched.
    """

    narrative = [facet for facet in missing if facet in _NARRATIVE_GUARANTEE_FACETS]
    if not narrative:
        return str(answer or "").strip()
    current = str(answer or "").strip()
    role_items = [
        item
        for item in manifest
        if set(item.get("supports") or []) & set(narrative)
    ]
    still = [
        facet
        for facet in narrative
        if not _target_facet_is_present(current, facet, role_items)
    ]
    if not still:
        return current
    ordered = sorted(
        {
            int(item.get("citation_id") or 0)
            for item in role_items
            if int(item.get("citation_id") or 0) > 0
        }
    )
    completion = _render_focused_target_manifest(role_items, still, ordered)
    return _merge_focused_target_completion(current, completion)


def _strip_existing_collection_member_blocks(
    answer: str,
    member_names: Sequence[str],
) -> str:
    """Remove stale partial enumerations before inserting a canonical N-of-N list.

    The completeness editor can fail after the model already listed one or a
    few collection members.  Keeping that prose below the deterministic full
    list repeats modes and their causes.  Markdown paragraphs/list blocks that
    mention an exact member name are therefore replaced as a unit; unrelated
    analysis paragraphs remain untouched.
    """

    names = [str(value or "").strip() for value in member_names if str(value or "").strip()]
    if not names:
        return str(answer or "").strip()
    blocks = re.split(r"\n\s*\n", str(answer or "").strip())
    retained = [block for block in blocks if not any(name in block for name in names)]
    return "\n\n".join(block.strip() for block in retained if block.strip()).strip()


def _append_missing_architecture_roles(
    answer: str,
    missing: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any] | None = None,
) -> str:
    """Deterministically preserve approved role facts if the editor omits them.

    LLM editing remains responsible for natural organization.  This fallback
    only restates exact structured fields with their existing citation, so a
    plural receiver question cannot regress merely because the editor selected
    one of several approved modules on a particular generation.
    """

    role_missing = [facet for facet in missing if facet in _ARCHITECTURE_ROLE_FACETS]
    if not role_missing:
        return answer
    lines: List[str] = []
    working = str(answer or "")

    inventory_block = ""
    if "closed_collection_members" in role_missing:
        inventory_items = [
            item
            for item in manifest
            if "closed_collection_members" in set(item.get("supports") or [])
        ]
        groups = _closed_collection_groups(inventory_items)
        needs_inventory_rebuild = bool(
            groups
            and not _closed_collection_members_present(
                working, inventory_items, manifest
            )
        )
        if groups and isinstance(state, Mapping):
            frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
            if str(frame.get("coverage_scope") or "") == "complete_collection":
                first_group_fields = (
                    inventory_items[groups[0][0]].get("answer_fields") or {}
                )
                expected_total = len(groups[0])
                declared_totals = _architecture_declared_member_counts(
                    working,
                    str(first_group_fields.get("collection_member_kind") or ""),
                )
                needs_inventory_rebuild = needs_inventory_rebuild or (
                    declared_totals != [expected_total]
                )
        if groups and needs_inventory_rebuild:
            group = groups[0]
            first_fields = inventory_items[group[0]].get("answer_fields") or {}
            member_type = str(first_fields.get("collection_member_kind") or "")
            frame = (
                state.get("query_frame")
                if isinstance(state, Mapping)
                and isinstance(state.get("query_frame"), Mapping)
                else {}
            )
            requested_member_kind = str(
                frame.get("collection_member_kind") or member_type
            ).strip()
            member_label = {
                "subsystem": "子系统",
                "module": "模块",
                "system": "系统",
                "failure_mode": "故障模式",
            }.get(member_type, "直接成员")
            expected_count = len(group)
            if member_type == "failure_mode":
                member_names = [
                    str(
                        (inventory_items[index].get("answer_fields") or {}).get(
                            "collection_member_name"
                        )
                        or ""
                    ).strip()
                    for index in group
                ]
                working = _strip_existing_collection_member_blocks(
                    working,
                    member_names,
                )
                inventory_citations = {
                    int(inventory_items[index].get("citation_id") or 0)
                    for index in group
                    if int(inventory_items[index].get("citation_id") or 0) > 0
                }
                retained_blocks: List[str] = []
                for block in re.split(r"\n\s*\n", working):
                    cited = {
                        int(value)
                        for value in re.findall(r"\[\^(\d+)\]", block)
                    }
                    # The canonical inventory below supersedes every partial
                    # model paragraph supported only by those same member
                    # rows.  This removes leftovers such as “目前只看到前三
                    # 条”，while preserving genuinely cross-document context.
                    if cited and cited <= inventory_citations:
                        continue
                    if block.strip():
                        retained_blocks.append(block.strip())
                working = "\n\n".join(retained_blocks)
            # Drop only the exact lines that assert a contradictory closed-list
            # count.  Other architecture details remain available below the
            # canonical complete summary.
            retained_lines = []
            for line in working.splitlines():
                counts = _architecture_declared_member_counts(line, member_type)
                if counts and any(value != expected_count for value in counts):
                    continue
                retained_lines.append(line)
            working = "\n".join(retained_lines).strip()

            parent_name = str(first_fields.get("collection_parent_name") or "").strip()
            first_citation = int(inventory_items[group[0]].get("citation_id") or 0)
            collection_marker = f"[^{first_citation}]" if first_citation > 0 else ""
            if member_type == "failure_mode" and requested_member_kind == "cause":
                count_label = f"覆盖 {expected_count} 条故障模式记录"
                heading_label = "故障模式与原文原因映射"
            elif member_type == "failure_mode":
                count_label = f"共 {expected_count} 条故障模式"
                heading_label = "完整 FMECA 故障模式记录"
            else:
                count_label = f"共 {expected_count} 个{member_label}"
                heading_label = f"完整{member_label}"
            heading = (
                f"### {parent_name}的{heading_label}（{count_label}）"
                if parent_name
                else f"### {heading_label}（{count_label}）"
            )
            heading += collection_marker
            inventory_lines = [heading]
            if member_type == "failure_mode":
                if requested_member_kind == "cause":
                    inventory_lines.append(
                        "以下按故障模式逐项保留 FMECA 原文原因；这里的数量是模式记录数，"
                        "不把自由文本原因自动合并或计为若干个独立根因。"
                        + collection_marker
                    )
                elif expected_count > 1:
                    inventory_lines.append(
                        f"**结论：不是只有一条记录。** 当前适用 FMECA 在该范围内共记录 "
                        f"{expected_count} 条故障模式记录：{collection_marker}"
                    )
                else:
                    inventory_lines.append(
                        "当前适用 FMECA 在该范围内记录了 1 条故障模式记录。"
                        + collection_marker
                    )
            elif parent_name:
                inventory_lines.append(
                    "权威结构化资料中的完整直接成员如下：" + collection_marker
                )
            for number, index in enumerate(group, start=1):
                item = inventory_items[index]
                fields, view_citation = _collection_facet_view(item, manifest)
                name = str(
                    fields.get("collection_member_name")
                    or fields.get("component_name")
                    or fields.get("component_id")
                    or item.get("record_id")
                    or ""
                ).strip()
                abbreviation = str(fields.get("abbreviation") or "").strip()
                record_code = str(
                    fields.get("record_code")
                    or fields.get("collection_member_id")
                    or item.get("record_id")
                    or ""
                ).strip()
                description = str(fields.get("description") or "").strip().rstrip("。；;")
                if not description:
                    description = str(
                        fields.get("collection_member_description") or ""
                    ).strip().rstrip("。；;")
                label = f"{name}（{abbreviation}）" if abbreviation and abbreviation != name else name
                if member_type == "failure_mode" and record_code and record_code != label:
                    label += f"（{record_code}）"
                rendered = f"{number}. **{label}**"
                # 详情来自同身份富记录时，引用绑定到实际承载字段的记录。
                citation = view_citation or int(item.get("citation_id") or 0)
                rendered += f"。[^{citation}]" if citation > 0 else "。"
                inventory_lines.append(rendered)
                if member_type == "failure_mode":
                    details = (
                        ("原文原因", fields.get("fault_cause") or description),
                        ("局部影响", fields.get("local_impact")),
                        ("最终影响", fields.get("final_impact")),
                        ("严酷度", fields.get("severity")),
                        ("检测/判别", fields.get("detection_method")),
                        ("设计改进", fields.get("design_improvement")),
                        ("使用补偿", fields.get("compensation_action")),
                    )
                    for detail_label, raw_value in details:
                        value = str(raw_value or "").strip().rstrip("。；;")
                        if value:
                            inventory_lines.append(
                                f"   - {detail_label}：{value}。"
                                + (f"[^{citation}]" if citation > 0 else "")
                            )
                elif description:
                    inventory_lines.append(
                        f"   - 说明：{description}。"
                        + (f"[^{citation}]" if citation > 0 else "")
                    )
            inventory_block = "\n\n".join(
                [inventory_lines[0], "\n".join(inventory_lines[1:])]
            )
            # The model draft may have been reduced to this fail-closed
            # placeholder before the authoritative inventory is restored.
            # Once an N-of-N block exists, retaining the placeholder would be
            # both redundant and directly contradictory to the visible facts.
            working = "\n".join(
                line
                for line in working.splitlines()
                if line.strip() != _NO_SUPPORTED_FACTS_MESSAGE
            ).strip()
        role_missing = [
            facet for facet in role_missing if facet != "closed_collection_members"
        ]
    for facet in role_missing:
        for item in manifest:
            if facet not in set(item.get("supports") or []):
                continue
            if _target_facet_is_present(working, facet, [item]):
                continue
            citation = int(item.get("citation_id") or 0)
            fields = (
                item.get("answer_fields")
                if isinstance(item.get("answer_fields"), Mapping)
                else {}
            )
            component_name = str(fields.get("component_name") or "").strip()
            abbreviation = str(fields.get("abbreviation") or "").strip()
            component_id = str(fields.get("component_id") or "").strip()
            function_name = str(fields.get("function_name") or fields.get("function") or "").strip()
            function_id = str(fields.get("function_id") or "").strip()
            module_name = str(fields.get("module_name") or "").strip()
            description = str(
                fields.get("description") or fields.get("responsibility") or ""
            ).strip().rstrip("。；;")

            def component_label() -> str:
                name = component_name or component_id or str(item.get("record_id") or "")
                details = [value for value in (abbreviation, component_id) if value and value != name]
                return f"{name}（{' / '.join(details)}）" if details else name

            if facet == "architecture_receivers":
                fact = f"接收单元：**{component_label()}**"
                if description:
                    fact += f"——{description}"
            elif facet == "architecture_receiver_context":
                fact = f"所属层级：**{component_label()}**"
                if description:
                    fact += f"——{description}"
            elif facet == "architecture_responsible_component":
                fact = f"职责组件：**{component_label()}**"
                if description:
                    fact += f"——{description}"
            elif facet == "architecture_responsibility_function":
                name = function_name or function_id or str(item.get("record_id") or "")
                details = [value for value in (function_id, module_name) if value and value != name]
                label = f"{name}（{' / '.join(details)}）" if details else name
                fact = f"对应功能：**{label}**"
                if description:
                    fact += f"——{description}"
            else:
                source = str(
                    fields.get("source_name")
                    or fields.get("source_entity")
                    or fields.get("source_component_id")
                    or ""
                ).strip()
                target = str(
                    fields.get("target_name")
                    or fields.get("target_entity")
                    or fields.get("target_component_id")
                    or ""
                ).strip()
                relation = str(
                    fields.get("relation") or fields.get("relation_type") or ""
                ).strip()
                if source and target:
                    fact = f"下游关系：**{source} → {target}**"
                    if relation:
                        fact += f"（{relation}）"
                else:
                    fact = f"下游处理单元：**{component_label()}**"
                    if description:
                        fact += f"——{description}"
            if citation > 0:
                fact += f"。[^{citation}]"
            if fact not in lines:
                rendered = f"- {fact}"
                lines.append(rendered)
                working = f"{working.rstrip()}\n{rendered}".strip()
    additions = [value for value in (inventory_block, "\n".join(lines)) if value]
    if not additions:
        return working
    if inventory_block:
        return "\n\n".join([inventory_block, working, *additions[1:]]).strip()
    return f"{working.rstrip()}\n\n" + "\n".join(lines)


def _closed_collection_identity_only(state: Mapping[str, Any]) -> bool:
    """Whether the user requested an exhaustive member list and nothing else."""

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    fields = set(required_answer_fields(state))
    if str(frame.get("collection_member_kind") or "") in {"failure_mode", "cause"}:
        # FMECA 模式记录的身份与内容不可分：模式名称一行只是标签，用户
        # 要“完整列出失效表现”时需要的是该条记录的原因、影响、严酷度与
        # 检测字段。组件/子系统层级列表的“身份”本身就是答案，保持原语义。
        return False
    return (
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and "closed_collection_members" in fields
        # ``failure_mode`` is a harmless redundant label for a collection
        # whose members are themselves failure modes.  Every analytical
        # dimension remains an explicit opt-in field.
        and not fields - {"closed_collection_members", "failure_mode"}
    )


def _resolved_architecture_collection(state: Mapping[str, Any]) -> bool:
    """Whether an architecture collection was closed by PostgreSQL.

    The planner's wording can use ``功能域`` or ``模块`` while the stored
    hierarchy proves a system's direct ``subsystem`` members.  The diagnosis
    tool records that repaired contract in ``collection_resolution``.  Treat
    it as authoritative at the terminal rendering boundary even when an old
    planner added another architecture facet to the frame; otherwise the LLM
    draft can reintroduce a mixed, partial count.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    resolution = (
        state.get("collection_resolution")
        if isinstance(state.get("collection_resolution"), Mapping)
        else {}
    )
    return bool(
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and str(resolution.get("member_kind") or "") in {
            "subsystem",
            "module",
            "component",
        }
        and resolution.get("closed") is True
    )


def _render_closed_collection_identity_only(
    state: Mapping[str, Any],
) -> str:
    """Render an exact N-of-N inventory from gate-approved PostgreSQL rows.

    The LLM still understands the request, chooses the collection contract and
    writes the normal draft.  Exhaustive membership, however, is a structured
    data result: the last renderer prevents that draft from silently adding
    unrequested causes, impacts, severity, repair or history fields.
    """

    _missing, manifest = _target_completion_manifest("", dict(state))
    if not manifest:
        return ""
    return _append_missing_architecture_roles(
        "",
        ["closed_collection_members"],
        manifest,
        state,
    ).strip()


def _render_focused_target_manifest(
    manifest: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    ordered_citation_ids: Sequence[int],
) -> str:
    by_citation = {
        int(item.get("citation_id") or 0): item
        for item in manifest
        if int(item.get("citation_id") or 0) > 0
    }
    items = [by_citation[value] for value in ordered_citation_ids]
    lines: List[str] = []

    # Scalars retain their all-record rendering contract: equal values may be
    # grouped, but every approved business record and citation remains bound.
    for facet in missing:
        if facet not in _SCALAR_TARGET_FACET_LABELS:
            continue
        facet_items = [
            item for item in items if facet in set(item.get("supports") or [])
        ]
        if not facet_items:
            continue
        label = _TARGET_COMPLETENESS_LABELS.get(facet, facet)
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for item in facet_items:
            values = _target_scalar_values(
                facet, item.get("answer_fields") or {}
            )
            if values:
                grouped.setdefault(values[0], []).append(item)
        for value, group in grouped.items():
            names = [
                str(item.get("record_name") or item.get("record_id") or "记录")
                for item in group
            ]
            markers = "".join(
                f"[^{int(item.get('citation_id') or 0)}]" for item in group
            )
            subject = "、".join(names)
            predicate = "均为" if len(group) > 1 else "为"
            lines.append(
                f"- **{subject}**的{label}{predicate} **{value}**。{markers}"
            )

    # Narrative fields are rendered record-first.  A FMECA row therefore
    # keeps local and final impact in one evidence unit, while an explicit RBD
    # criterion reused as its single-failure boundary is shown only once.
    narrative_facets = [
        facet for facet in missing if facet not in _SCALAR_TARGET_FACET_LABELS
    ]
    for item in items:
        supported = set(item.get("supports") or [])
        fields = (
            item.get("answer_fields")
            if isinstance(item.get("answer_fields"), Mapping)
            else {}
        )
        facts: List[str] = []
        seen_values: set[str] = set()
        for facet in narrative_facets:
            if facet not in supported:
                continue
            label = _TARGET_COMPLETENESS_LABELS.get(facet, facet)
            field_keys = _TARGET_COMPLETENESS_FIELD_KEYS.get(facet, set())
            for key, raw_value in fields.items():
                if key not in field_keys:
                    continue
                values: List[str] = []
                for value in _flatten_target_values(raw_value):
                    cleaned = re.sub(r"[\s。；;，,]+$", "", value).strip()
                    signature = _compact_semantic_text(cleaned)
                    if not cleaned or not signature or signature in seen_values:
                        continue
                    seen_values.add(signature)
                    values.append(cleaned)
                if not values:
                    continue
                field_label = _TARGET_COMPLETENESS_FIELD_LABELS.get(key, label)
                facts.append(f"{field_label}：{'；'.join(values)}")
        if not facts:
            continue
        identity = str(
            item.get("record_name") or item.get("record_id") or "记录"
        )
        citation = int(item.get("citation_id") or 0)
        lines.append(
            f"- **{identity}**：" + "；".join(facts) + f"。[^{citation}]"
        )
    return "\n".join(lines).strip()


def _focused_target_selection_is_complete(
    ordered_citation_ids: Sequence[int],
    missing: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
) -> bool:
    """Validate a focused fallback selection against each facet's semantics.

    Narrative fields are existential: the selected approved records only need
    to cover every missing facet.  FMECA scalar fields are intentionally
    exhaustive, because omitting one business record would bind another
    record's severity or probability to the wrong candidate.
    """

    selected = set(ordered_citation_ids)
    relevant_items = [
        item
        for item in manifest
        if int(item.get("citation_id") or 0) > 0
        and set(item.get("supports") or []) & set(missing)
    ]
    allowed = {
        int(item.get("citation_id") or 0)
        for item in relevant_items
    }
    if not selected or not selected <= allowed:
        return False
    for facet in missing:
        facet_ids = {
            int(item.get("citation_id") or 0)
            for item in relevant_items
            if facet in set(item.get("supports") or [])
        }
        if not facet_ids:
            return False
        if facet in _SCALAR_TARGET_FACET_LABELS:
            if not facet_ids <= selected:
                return False
        elif not facet_ids & selected:
            return False
    return True


def _single_record_focused_target_fallback(
    missing: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Return the only safe deterministic selection, if one exists."""

    support_by_citation: Dict[int, set[str]] = {}
    missing_set = set(missing)
    for item in manifest:
        citation = int(item.get("citation_id") or 0)
        supports = set(item.get("supports") or []) & missing_set
        if citation > 0 and supports:
            support_by_citation.setdefault(citation, set()).update(supports)
    if len(support_by_citation) != 1:
        return []
    citation, supports = next(iter(support_by_citation.items()))
    selection = [citation]
    if missing_set <= supports and _focused_target_selection_is_complete(
        selection, missing, manifest
    ):
        return selection
    return []


def _merge_focused_target_completion(answer: str, completion: str) -> str:
    """Append exact missing facts without replacing already valid evidence."""

    current = str(answer or "").strip()
    addition = str(completion or "").strip()
    if not addition or addition in current:
        return current
    if not current:
        return addition
    return f"{current}\n\n{addition}"


def _degrade_partial_target_answer(
    answer: str,
    missing: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    """Best-effort deterministic completion of still-missing target fields.

    The normal bounded editors run earlier; this is only reached when approved
    evidence exists but the answer still omitted a required field.  It restates
    exact structured field values with their existing citation and never
    invents content, so a missing scalar can no longer turn the whole turn into
    a blank system error.
    """

    relevant = [
        item
        for item in manifest
        if int(item.get("citation_id") or 0) > 0
        and set(item.get("supports") or []) & set(missing)
    ]
    if not relevant:
        return answer
    ordered = sorted({int(item.get("citation_id") or 0) for item in relevant})
    completion = _render_focused_target_manifest(manifest, missing, ordered)
    return _merge_focused_target_completion(answer, completion)


def _append_unconfirmed_target_notice(answer: str, labels: str) -> str:
    """Attach an explicit, citation-free “unconfirmed” note to a partial answer.

    The note distinguishes a best-effort answer from a system error and makes
    clear the missing fields cannot be stated as approved facts, which keeps
    the product constraint “no evidence, no unique diagnosis” intact.
    """

    notice = (
        "### 尚未从已批准证据确认\n"
        f"以下必答字段尚未从已批准证据确认：{labels}。"
        "以上内容不构成唯一诊断结论；"
        "请补充模块编号、原始告警或适用构型后继续查询。"
    )
    current = str(answer or "").rstrip()
    if not current:
        return notice
    return f"{current}\n\n{notice}"


_FOCUSED_IDENTITY_CONTEXT_KEYS = {
    "function_id",
    "function_name",
    "component_id",
    "component_name",
    "component_type",
    "interface_id",
    "signal_name",
    "record_code",
    "failure_mode_id",
    "failure_mode",
    "hmc_code",
    "lcn",
    "hamc",
    "hsc",
    "case_id",
    "title",
    "module_failure",
    "sur_failure",
    "subcircuit_failure",
}


_FOCUSED_DETAIL_LABEL_KEYS = {
    "fault_cause",
    "root_cause",
    "structural_cause_chain",
    "cause_chain",
    "top_event",
    "basic_events",
    "intermediate_events",
    "logic_gates",
    "local_impact",
    "final_impact",
    "severity",
    "severity_raw",
    "severity_normalized",
    "failure_probability",
    "failure_probability_normalized",
    "failure_probability_level",
    "detection_method",
    "detect_method",
    "maintenance_action",
    "repair_method",
    "compensation_action",
    "occurred_at",
    "mission_phase",
    "diagnosis_method",
    "steps",
    "diagnosis_steps",
    "final_solution",
    "verification_result",
    "preventive_action",
    "api_reportable",
    "api_name",
    "api_method",
    "api_usage",
    "platform_test",
    "platform_test_raw",
    "platform_test_status",
    "operator",
    "personnel",
    "maintenance_personnel",
    "troubleshooter",
}


_FOCUSED_DETAIL_FIELD_LABELS = {
    "mission_phase": "任务阶段/工作模式",
    "diagnosis_method": "排故方法",
    "preventive_action": "预防措施",
    "structural_cause_chain": "FTA结构原因链",
    "cause_chain": "原因链",
    "top_event": "顶事件",
    "basic_events": "基本事件",
    "intermediate_events": "中间事件",
    "logic_gates": "逻辑门",
    "platform_test_status": "平台级实验状态",
    "platform_test": "平台级实验结果",
    "operator": "排故人员",
    "personnel": "排故人员",
}


def _focused_answer_has_unrequested_fields(
    answer: str,
    state: Mapping[str, Any],
) -> bool:
    """Detect source fields that a focused answer contract did not request."""

    contract_fields = required_answer_fields(state)
    if not focused_answer_contract(state) or not contract_fields:
        return False
    requested_keys: set[str] = set()
    for facet in contract_fields:
        requested_keys.update(_TARGET_COMPLETENESS_FIELD_KEYS.get(facet, set()))
    member_frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    if "closed_collection_members" in contract_fields and str(
        member_frame.get("collection_member_kind") or ""
    ) in {"failure_mode", "cause"}:
        # 故障模式集合的必答内容包含该条 FMECA 字段，不是“未要求信息”。
        requested_keys.update(_COLLECTION_MEMBER_DETAIL_KEYS)
    allowed_keys = requested_keys | _FOCUSED_IDENTITY_CONTEXT_KEYS
    requested_values = {
        _compact_semantic_text(value)
        for item in _citation_ledger(dict(state)).values()
        for key, raw in _evidence_answer_fields(item).items()
        if key in allowed_keys
        for value in _flatten_target_values(raw)
        if _compact_semantic_text(value)
    }
    answer_text = str(answer or "")
    for item in _citation_ledger(dict(state)).values():
        fields = _evidence_answer_fields(item)
        for key, raw in fields.items():
            if key in allowed_keys or raw in (None, "", [], {}):
                continue
            for value in _flatten_target_values(raw):
                normalized = _compact_semantic_text(value)
                if (
                    len(normalized) >= 4
                    and normalized not in requested_values
                    and _target_value_is_present(answer_text, value)
                ):
                    return True
            if key not in _FOCUSED_DETAIL_LABEL_KEYS:
                continue
            label = _FOCUSED_DETAIL_FIELD_LABELS.get(
                key, _TARGET_COMPLETENESS_FIELD_LABELS.get(key, "")
            )
            if label and re.search(
                rf"(?:^|[\n*\s-])\**{re.escape(label)}\**\s*[:：]",
                answer_text,
                re.MULTILINE,
            ):
                return True
    return False


def _constrain_focused_answer_fields(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Fail closed to the approved field manifest when an LLM over-answers."""

    if not _focused_answer_has_unrequested_fields(answer, state):
        return answer
    fields = required_answer_fields(state)
    _missing, manifest = _target_completion_manifest("", dict(state))
    ordered = [
        int(item.get("citation_id") or 0)
        for item in manifest
        if int(item.get("citation_id") or 0) > 0
    ]
    rendered = _render_focused_target_manifest(manifest, fields, ordered)
    return rendered or answer


def _target_preflight_instruction(state: Dict[str, Any]) -> str:
    missing, manifest = _target_completion_manifest("", state)
    if not missing or not manifest:
        return ""
    inventory_manifest = ""
    if "closed_collection_members" in missing:
        inventory_manifest = (
            "\n以下是 PostgreSQL 已闭合的直接成员清单，必须先逐项完整覆盖，"
            "再展开下一级细节：\n"
            + _format_target_completion_manifest(
                manifest, ["closed_collection_members"]
            )
        )
    focused_instruction = ""
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    contract_fields = required_answer_fields(state)
    if (
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and contract_fields == ["closed_collection_members"]
    ):
        if str(frame.get("collection_member_kind") or "") in {"failure_mode", "cause"}:
            # FMECA 模式的“完整列出”= 每条记录的字段内容，而非只有标题。
            focused_instruction = (
                "本轮要求逐项完整列出故障模式记录：每条给出模式名称、业务代码，"
                "以及该条 FMECA 记录的原文原因、局部影响、最终影响、严酷度与"
                "检测判别方法；没有字段的内容不得补造，也不得用纯标题清单代替。"
            )
        else:
            focused_instruction = (
                "本轮只要求完整成员清单：回答集合总数并逐项列出成员名称/业务代码；"
                "不要展开原因、影响、严酷度、检测、改进、处置或历史。"
            )
    elif focused_answer_contract(state):
        focused_instruction = (
            "本轮是聚焦字段查询：除区分业务记录所需的名称外，只回答上述必答字段；"
            "即使证据还含原因、影响、检测、维修或历史，也不得顺带展开。"
        )
    return (
        "【本轮明确目标闭环检查（内部指令，不要原样输出）】\n"
        "用户明确询问且已批准字段可支持："
        + "、".join(_TARGET_COMPLETENESS_LABELS[facet] for facet in missing)
        + "。最终回答应逐项覆盖并绑定相应证据；没有字段的内容不得补造。"
        + focused_instruction
        + inventory_manifest
    )


