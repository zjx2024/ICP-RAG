"""ICP 回答前的确定性证据门禁。

该节点不调用 LLM，也不修改数据库。它只对工具已产生的 Evidence
进行去重、审核状态、派生链和适用范围检查，为 Answer 节点生成机器可读的
``safety_decision``。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from backend.agent.contracts import (
    answer_mode_for_intent,
    deduplicate_evidence,
    make_tool_result,
)
from backend.agent.tool_policy import detect_calculation_requests, extract_hmc_components
from backend.icp.diagram_validation import (
    FORMAL_VALIDATION_STATUSES,
    VALIDATION_FAILED,
    VALIDATION_PROVISIONAL,
    VALIDATION_QUARANTINED,
)


_DIRECT_AUTHORITIES = {
    "authoritative_record",
    "source_artifact",
    "confirmed_relation",
    "validated_relation",
    "deterministic_derivation",
}
_GRAPH_CLAIM_TYPES = {
    "topology",
    "relation",
    "diagram_structure",
    "architecture_edge",
    "fta_structure",
    "rbd_structure",
    "relation_path",
}
_CALCULATION_TOOLS = {"fta_analyzer", "rbd_calculator"}
_STRUCTURED_HMC_MATCH_TYPES = {"exact_code", "exact_components"}
_POSTGRES_TOOL_DOC_TYPES = {
    "function_locator": "function_locator",
    "hmc_query": "hmc",
    "fmeca_query": "fmeca",
    "fta_query": "fta",
    "rbd_query": "rbd",
    "mfl_query": "mfl",
    "fault_history_query": "fault_history",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _evidence_business_tokens(item: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """Return explicit business identifiers usable for cross-store de-duplication."""
    doc_type = _clean(item.get("doc_type")).casefold()
    return {
        (doc_type, value.casefold())
        for value in (_clean(item.get("record_id")), _clean(item.get("title")))
        if value
    }


def _focused_direct_answer_context(
    items: Sequence[Dict[str, Any]],
    answer_mode: str,
) -> List[Dict[str, Any]]:
    """Narrow direct-answer context while retaining the complete audit registry.

    When reranking has one clearly dominant hit, very low-score neighbours such
    as generic occurrences of “老化” must not be shown as co-equal evidence. A
    PostgreSQL record also supersedes its OpenSearch projection when their
    explicit business identifiers match.
    """
    if answer_mode not in {"knowledge_explanation", "maintenance_guidance"}:
        return list(items)

    direct_tokens: Set[Tuple[str, str]] = set()
    # Structured SQL phrase scores are integer coverage weights, while
    # OpenSearch-to-PostgreSQL hydration carries a 0..1 semantic score.  They
    # are deliberately not comparable.  Threshold direct rows only inside one
    # document type *and* one score family; a score of 34 from an HMC keyword
    # match must never suppress a 0.94 exact history hydration.
    direct_scores: Dict[Tuple[str, str], List[float]] = {}
    search_items: List[Dict[str, Any]] = []
    for item in items:
        if _clean(item.get("authority")) == "search_projection":
            search_items.append(item)
        elif _clean(item.get("authority")) in _DIRECT_AUTHORITIES:
            direct_tokens.update(_evidence_business_tokens(item))
            score = item.get("retrieval_score")
            if isinstance(score, (int, float)):
                match_type = _clean(item.get("match_type"))
                score_family = (
                    "semantic"
                    if match_type
                    in {
                        "semantic_id_hydration",
                        "semantic_locator_hydration",
                        "exact_history_reference",
                    }
                    else "structured"
                )
                key = (_clean(item.get("doc_type")), score_family)
                direct_scores.setdefault(key, []).append(float(score))

    numeric_scores = [
        float(score)
        for item in search_items
        for score in [item.get("rerank_score", item.get("retrieval_score"))]
        if isinstance(score, (int, float))
    ]
    strongest_score = max(numeric_scores, default=0.0)
    strong_threshold = strongest_score * 0.15 if strongest_score >= 0.1 else None
    direct_thresholds = {
        key: strongest * 0.15
        for key, values in direct_scores.items()
        for strongest in [max(values, default=0.0)]
        if strongest >= 9.0
    }
    weak_search_budget = 4
    kept_weak_search = 0
    focused: List[Dict[str, Any]] = []

    for item in items:
        if _clean(item.get("authority")) != "search_projection":
            raw_direct_score = item.get("retrieval_score")
            match_type = _clean(item.get("match_type"))
            score_family = (
                "semantic"
                if match_type
                in {
                    "semantic_id_hydration",
                    "semantic_locator_hydration",
                    "exact_history_reference",
                }
                else "structured"
            )
            direct_threshold = direct_thresholds.get(
                (_clean(item.get("doc_type")), score_family)
            )
            if (
                direct_threshold is not None
                and isinstance(raw_direct_score, (int, float))
                and float(raw_direct_score) < direct_threshold
            ):
                continue
            focused.append(item)
            continue
        if direct_tokens & _evidence_business_tokens(item):
            continue
        raw_score = item.get("rerank_score", item.get("retrieval_score"))
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        if strong_threshold is not None:
            if score is not None and score >= strong_threshold:
                focused.append(item)
            continue
        if kept_weak_search < weak_search_budget:
            focused.append(item)
            kept_weak_search += 1
    return focused


def _restore_requested_closed_collection_members(
    original_items: Sequence[Dict[str, Any]],
    focused_items: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Never score-prune members of an explicitly requested closed set.

    The direct-answer focus pass is useful for ordinary Top-K explanation, but
    its relative-score threshold cannot be applied to a PostgreSQL collection
    that the current query asks to enumerate completely.  Otherwise a valid
    15-row FMECA set can be reduced to its three highest lexical matches before
    the answer node ever has a chance to enforce N-of-N coverage.

    This guard does not infer an entity or declare a collection complete.  It
    only preserves rows already stamped ``collection_closed`` by the
    authoritative repository.  The answer layer still binds the requested
    parent and verifies the declared member count.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), dict) else {}
    if _clean(frame.get("coverage_scope")) != "complete_collection":
        return list(focused_items)

    requested_kind = _clean(frame.get("collection_member_kind"))
    stored_kind = "failure_mode" if requested_kind == "cause" else requested_kind
    resolution = (
        state.get("collection_resolution")
        if isinstance(state.get("collection_resolution"), dict)
        else {}
    )
    resolved_parent_id = _clean(resolution.get("parent_id"))
    focused_ids = {
        _clean(item.get("evidence_id"))
        for item in focused_items
        if _clean(item.get("evidence_id"))
    }

    restored: List[Dict[str, Any]] = []
    for item in original_items:
        evidence_id = _clean(item.get("evidence_id"))
        metadata = item.get("metadata") or {}
        fields = metadata.get("answer_fields") if isinstance(metadata, dict) else {}
        is_requested_member = (
            isinstance(fields, dict)
            and fields.get("collection_closed") is True
            and _clean(fields.get("collection_role")) == "member"
            and (not stored_kind or _clean(fields.get("collection_member_kind")) == stored_kind)
            and (
                not resolved_parent_id
                or _clean(fields.get("collection_parent_id")) == resolved_parent_id
            )
        )
        if evidence_id in focused_ids or is_requested_member:
            restored.append(item)
    return restored


def _requested_scope(state: Dict[str, Any]) -> Dict[str, str]:
    raw = state.get("retrieval_scope") or {}
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, str] = {}
    for key in ("system", "configuration", "document_number", "document_version"):
        value = _clean(raw.get(key))
        if value:
            result[key] = value
    return result


def _scope_conflicts(
    evidence: Sequence[Dict[str, Any]],
    requested: Dict[str, str],
) -> Tuple[str, List[str], List[str]]:
    """检查明确适用范围冲突，不将不同文档各自的版本号误判为冲突。"""
    conflicting_ids: List[str] = []
    warnings: List[str] = []
    scoped_count = 0
    configurations: Dict[str, List[str]] = {}
    versions_by_document: Dict[str, Dict[str, List[str]]] = {}

    for item in evidence:
        scope = item.get("scope") or {}
        if not isinstance(scope, dict):
            continue
        evidence_id = _clean(item.get("evidence_id"))
        if any(_clean(scope.get(key)) for key in ("system", "configuration", "document_number", "document_version")):
            scoped_count += 1

        for key in ("system", "configuration", "document_number", "document_version"):
            expected = _clean(requested.get(key))
            actual = _clean(scope.get(key))
            if expected and actual and expected.casefold() != actual.casefold():
                conflicting_ids.append(evidence_id)
                warnings.append(f"证据 {evidence_id} 的 {key}={actual} 与请求范围 {expected} 不一致")

        configuration = _clean(scope.get("configuration"))
        if configuration:
            configurations.setdefault(configuration.casefold(), []).append(evidence_id)

        document_number = _clean(scope.get("document_number"))
        document_version = _clean(scope.get("document_version"))
        if document_number and document_version:
            versions_by_document.setdefault(document_number.casefold(), {}).setdefault(
                document_version.casefold(), []
            ).append(evidence_id)

    # 用户未指定构型时，同一回答混入多个明确构型仍需要追问。
    if not requested.get("configuration") and len(configurations) > 1:
        for ids in configurations.values():
            conflicting_ids.extend(ids)
        warnings.append("检索证据涉及多个适用构型，不能直接合并")

    # 只比较同一文档编号的版本，不比较不同手册各自的 V1/V2。
    if not requested.get("document_version"):
        for document_number, versions in versions_by_document.items():
            if len(versions) <= 1:
                continue
            for ids in versions.values():
                conflicting_ids.extend(ids)
            warnings.append(f"文档 {document_number} 同时命中多个版本，需要确认有效版本")

    conflicting_ids = _unique(conflicting_ids)
    if conflicting_ids:
        return "conflict", conflicting_ids, _unique(warnings)
    if requested or scoped_count:
        return "compatible", [], []
    return "unknown", [], []


def _is_graph_evidence(item: Dict[str, Any]) -> bool:
    return (
        _clean(item.get("source_kind")) == "hugegraph"
        or _clean(item.get("authority"))
        in {"confirmed_relation", "validated_relation", "unconfirmed_extraction"}
        or _clean(item.get("claim_type")) in _GRAPH_CLAIM_TYPES
    )


def _graph_validation_status(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    validation_status = (
        _clean(metadata.get("validation_status"))
        if isinstance(metadata, dict)
        else ""
    )
    if validation_status in {VALIDATION_PROVISIONAL, VALIDATION_QUARANTINED, VALIDATION_FAILED}:
        return validation_status
    if validation_status in FORMAL_VALIDATION_STATUSES:
        return validation_status
    review_status = _clean(item.get("review_status"))
    if review_status == "confirmed":
        return "manual_confirmed"
    return validation_status or review_status or "pending"


def _graph_evidence_is_excluded(item: Dict[str, Any]) -> bool:
    return _is_graph_evidence(item) and _graph_validation_status(item) in {
        VALIDATION_QUARANTINED,
        VALIDATION_FAILED,
        "rejected",
    }


def _graph_revision_is_usable(item: Dict[str, Any]) -> bool:
    """Only active, formally validated, published revisions support graph claims."""
    if not _is_graph_evidence(item):
        return True
    metadata = item.get("metadata") or {}
    return (
        _graph_validation_status(item) in FORMAL_VALIDATION_STATUSES
        and isinstance(metadata, dict)
        and metadata.get("is_active") is True
        and _clean(metadata.get("projection_status")) == "published"
    )


def _is_directly_approvable(item: Dict[str, Any], evidence_ids: Set[str]) -> Tuple[bool, str]:
    authority = _clean(item.get("authority"))
    review_status = _clean(item.get("review_status")) or "not_required"
    metadata = item.get("metadata") or {}
    if (
        isinstance(metadata, dict)
        and _clean(metadata.get("seed_binding_status")) == "candidate"
        and metadata.get("query_binding_approved") is not True
    ):
        # PostgreSQL re-authorization proves that every returned edge exists;
        # it does not prove that a semantically retrieved start node is relevant
        # to this user's wording. Keep the path in the audit/context catalog as
        # a clue, but do not let it become a directly cited conclusion.
        return False, "多跳关系的起点仅由语义候选绑定，尚未证明与本问题直接相关"
    if review_status in {"rejected", VALIDATION_QUARANTINED, VALIDATION_FAILED}:
        return False, f"图验证状态为 {review_status}"
    if _is_graph_evidence(item):
        validation_status = _graph_validation_status(item)
        if validation_status not in FORMAL_VALIDATION_STATUSES:
            if validation_status == VALIDATION_PROVISIONAL:
                return False, "工程图仅为 provisional，不能支撑正式拓扑或计算"
            return False, "工程图结构尚未通过正式验证"
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict) or metadata.get("is_active") is not True:
            return False, "工程图修订不是当前有效版本"
        if _clean(metadata.get("projection_status")) != "published":
            return False, "工程图修订尚未发布"
    if authority not in _DIRECT_AUTHORITIES:
        return False, "仅为检索投影或未解决推断"
    if authority == "deterministic_derivation":
        parents = {_clean(value) for value in item.get("derived_from", []) if _clean(value)}
        if not parents:
            return False, "确定性计算缺少派生证据"
        missing = parents - evidence_ids
        if missing:
            return False, f"确定性计算的派生证据不完整: {sorted(missing)}"
    return True, ""


def _postgres_failure_without_authority(
    tool_results: Sequence[Dict[str, Any]],
    approved: Sequence[Dict[str, Any]],
) -> bool:
    has_authoritative_record = any(
        _clean(item.get("authority"))
        in {"authoritative_record", "source_artifact", "confirmed_relation", "validated_relation"}
        for item in approved
    )
    if has_authoritative_record:
        return False
    for result in tool_results:
        diagnostics = result.get("diagnostics") or {}
        if _clean(result.get("status")) == "failed" and _clean(diagnostics.get("backend")) == "postgres":
            return True
    return False


def _postgres_tools_without_own_authority(
    tool_results: Sequence[Dict[str, Any]],
    approved: Sequence[Dict[str, Any]],
) -> List[str]:
    """Find degraded PG tools that have no approved fact of their own type."""
    approved_doc_types = {
        _clean(item.get("doc_type"))
        for item in approved
        if _clean(item.get("authority"))
        in {"authoritative_record", "source_artifact", "confirmed_relation", "validated_relation"}
    }
    missing: List[str] = []
    for result in tool_results:
        tool = _clean(result.get("tool"))
        doc_type = _POSTGRES_TOOL_DOC_TYPES.get(tool)
        diagnostics = result.get("diagnostics") or {}
        backend = _clean(diagnostics.get("backend"))
        status = _clean(result.get("status"))
        errors = diagnostics.get("errors") or []
        data = result.get("data") or {}
        postgres_errors = (
            data.get("postgres_errors", [])
            if isinstance(data, dict) and "+" in backend
            else errors
        )
        if (
            doc_type
            and "postgres" in backend
            and status in {"failed", "partial"}
            and postgres_errors
            and doc_type not in approved_doc_types
        ):
            missing.append(tool)
    return _unique(missing)


def _hmc_request_match_type(state: Dict[str, Any]) -> str:
    """Return ``exact_code``/``exact_components`` for supplied HMC fields.

    ``intent=exact_code`` is only a routing hint: the planner intentionally also
    uses it for *near* HMC input. Treating that hint as proof of a complete code
    turns an omitted HAMC into a false "PostgreSQL unavailable" diagnosis. The
    original user text is authoritative here; a model-rewritten query must not
    invent the missing component.
    """
    original_query = _clean(state.get("query"))
    if original_query:
        components = extract_hmc_components(original_query)
        if components:
            return "exact_code" if components.get("hamc") else "exact_components"
        return ""

    # Older callers and focused contract tests may not carry the original query.
    # Prefer an explicit execution parameter, then retain the legacy intent as a
    # final compatibility signal.
    for result in state.get("tool_results", []) or []:
        if _clean(result.get("tool")) != "hmc_query":
            continue
        params = result.get("params") or {}
        match_type = _clean(params.get("match_type")) if isinstance(params, dict) else ""
        if match_type in _STRUCTURED_HMC_MATCH_TYPES:
            return match_type
        if isinstance(params, dict) and isinstance(params.get("exact"), bool):
            return "exact_code" if params["exact"] else ""
    rewritten_query = _clean(state.get("rewritten_query"))
    if rewritten_query:
        components = extract_hmc_components(rewritten_query)
        if components:
            return "exact_code" if components.get("hamc") else "exact_components"
        return ""
    return "exact_code" if _clean(state.get("intent")) == "exact_code" else ""


def _complete_hmc_was_requested(state: Dict[str, Any]) -> bool:
    return _hmc_request_match_type(state) == "exact_code"


async def evidence_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """生成证据可用性与回答安全决策。

    新查询节点尚未接入 Evidence 时返回 ``unassessed`` 兼容态，不改写
    旧的候选、置信度、上下文或追问字段。
    """
    evidence = deduplicate_evidence(state.get("evidence_registry", []) or [])
    existing_clarification = _clean(state.get("clarification_question"))
    query_frame = state.get("query_frame") if isinstance(state.get("query_frame"), dict) else {}
    diagnostic_closure = query_frame.get("diagnostic_closure")
    # The planner has already made the semantic distinction between an
    # abstract explanation and an observed current incident.  Keep the latter
    # out of the Top-K explanation pruning path, otherwise the direct MFL/HMC
    # repair record can be discarded while a generic detection row survives.
    answer_mode = (
        "fault_diagnosis"
        if isinstance(diagnostic_closure, dict) and diagnostic_closure.get("required") is True
        else answer_mode_for_intent(state.get("intent"))
    )
    diagnosis_intent = answer_mode in {"fault_diagnosis", "exact_code_lookup"}
    architecture_intent = answer_mode == "architecture_knowledge"
    # 只有真正的旧链路（既没有统一证据，也没有新 ToolResult）才进入兼容态。
    # 新 ICP 链路的“合法空结果/工具失败”本身就是需要审查的结果，不能被
    # unassessed 掩盖。
    if not evidence and not (state.get("tool_results") or state.get("allowed_tools")):
        decision = {
            "status": "unassessed",
            "scope_status": "unknown",
            "can_claim_unique": False,
            "requires_clarification": bool(existing_clarification),
            "context_evidence_ids": [],
            "approved_evidence_ids": [],
            "blocked_evidence_ids": [],
            "blocked_claims": ["unique_diagnosis"] if diagnosis_intent else [],
            "warnings": ["当前节点尚未产生统一 Evidence，保留旧链路兼容行为"],
            "questions": [existing_clarification] if existing_clarification else [],
        }
        return {
            "context_evidence_ids": [],
            "approved_evidence_ids": [],
            "safety_decision": decision,
            "tool_results": [
                make_tool_result(
                    tool="evidence_gate",
                    status="empty",
                    data=decision,
                    warnings=decision["warnings"],
                )
            ],
        }

    all_ids = {_clean(item.get("evidence_id")) for item in evidence if _clean(item.get("evidence_id"))}
    excluded_items = [
        item
        for item in evidence
        if _clean(item.get("review_status")) == "rejected"
        or _graph_evidence_is_excluded(item)
    ]
    excluded_ids = {
        _clean(item.get("evidence_id"))
        for item in excluded_items
        if _clean(item.get("evidence_id"))
    }
    # Quarantined/failed graph material is retained in the audit registry but is
    # deliberately absent from the LLM reasoning context.  Provisional evidence
    # remains visible as a clearly blocked clue.
    context_items = [
        item for item in evidence if _clean(item.get("evidence_id")) not in excluded_ids
    ]
    focused_context_items = _focused_direct_answer_context(context_items, answer_mode)
    context_items = _restore_requested_closed_collection_members(
        context_items,
        focused_context_items,
        state,
    )
    context_ids = [_clean(item.get("evidence_id")) for item in context_items]
    approved_items: List[Dict[str, Any]] = []
    blocked_ids: List[str] = sorted(excluded_ids)
    warnings: List[str] = [
        f"证据 {_clean(item.get('evidence_id'))}: 图验证状态为 {_graph_validation_status(item)}，已从推理上下文排除"
        for item in excluded_items
        if _is_graph_evidence(item)
    ]
    unusable_graph = any(_is_graph_evidence(item) for item in excluded_items)

    # 先审批非计算证据，计算证据的 derived_from 再依赖这一集合。
    base_items = [item for item in context_items if _clean(item.get("authority")) != "deterministic_derivation"]
    calculation_items = [item for item in context_items if _clean(item.get("authority")) == "deterministic_derivation"]
    for item in [*base_items, *calculation_items]:
        current_approved_ids = {
            _clean(value.get("evidence_id")) for value in approved_items if _clean(value.get("evidence_id"))
        }
        approvable, reason = _is_directly_approvable(
            item,
            current_approved_ids if calculation_items else all_ids,
        )
        evidence_id = _clean(item.get("evidence_id"))
        if approvable:
            approved_items.append(item)
        else:
            blocked_ids.append(evidence_id)
            if reason:
                warnings.append(f"证据 {evidence_id}: {reason}")
            if _is_graph_evidence(item) and not _graph_revision_is_usable(item):
                unusable_graph = True

    scope_status, scope_conflict_ids, scope_warnings = _scope_conflicts(
        context_items,
        _requested_scope(state),
    )
    blocked_ids.extend(scope_conflict_ids)
    warnings.extend(scope_warnings)
    if scope_conflict_ids:
        approved_items = [
            item for item in approved_items if _clean(item.get("evidence_id")) not in set(scope_conflict_ids)
        ]

    approved_ids = [_clean(item.get("evidence_id")) for item in approved_items]
    has_approved_graph = any(_is_graph_evidence(item) for item in approved_items)
    selected_tools = {_clean(item) for item in (state.get("allowed_tools") or []) if _clean(item)}
    calculation_requested = bool(selected_tools & _CALCULATION_TOOLS)
    blocked_claims = ["unique_diagnosis"] if diagnosis_intent else []
    if unusable_graph and not has_approved_graph:
        blocked_claims.append("formal_topology")
    has_approved_calculation = any(
        _clean(item.get("authority")) == "deterministic_derivation" for item in approved_items
    )
    incomplete_numeric_calculations = [
        item
        for item in approved_items
        if _clean(item.get("authority")) == "deterministic_derivation"
        and isinstance(item.get("metadata"), dict)
        and item["metadata"].get("numeric_complete") is False
    ]
    raw_calculation_requests = state.get("calculation_requests")
    if raw_calculation_requests is None:
        # Compatibility for saved/running states created before the structured
        # field existed. Authorization still reads only the original user text.
        raw_calculation_requests = detect_calculation_requests(state.get("query"))
    calculation_requests = {
        _clean(item) for item in (raw_calculation_requests or []) if _clean(item)
    }
    numeric_result_requested = bool(
        calculation_requests & {"fta_probability", "rbd_reliability"}
    )
    if calculation_requested and not has_approved_calculation:
        blocked_claims.append("formal_calculation")
    if numeric_result_requested and incomplete_numeric_calculations:
        blocked_claims.append("formal_numeric_result")

    pg_unavailable = _postgres_failure_without_authority(
        state.get("tool_results", []) or [],
        approved_items,
    )
    missing_pg_tools = _postgres_tools_without_own_authority(
        state.get("tool_results", []) or [],
        approved_items,
    )
    intent = _clean(state.get("intent"))
    hmc_query_selected = "hmc_query" in selected_tools or any(
        _clean(item.get("tool")) == "hmc_query"
        for item in (state.get("tool_results", []) or [])
    )
    hmc_request_match_type = _hmc_request_match_type(state) if hmc_query_selected else ""
    complete_hmc_requested = hmc_request_match_type == "exact_code"
    component_hmc_requested = hmc_request_match_type == "exact_components"
    structured_hmc_requested = complete_hmc_requested or component_hmc_requested
    incomplete_hmc_request = (
        intent == "exact_code" and hmc_query_selected and not structured_hmc_requested
    )
    approved_structured_hmc = [
        item
        for item in approved_items
        if _clean(item.get("doc_type")) == "hmc"
        and _clean(item.get("source_kind")) == "postgres"
        and _clean(item.get("authority")) == "authoritative_record"
        and _clean(item.get("claim_type")) == "hmc_fault"
        and _clean(item.get("match_type")) in _STRUCTURED_HMC_MATCH_TYPES
    ]
    has_approved_exact_hmc = any(
        _clean(item.get("match_type")) == "exact_code" for item in approved_structured_hmc
    )
    has_approved_component_hmc = any(
        _clean(item.get("match_type")) == "exact_components"
        for item in approved_structured_hmc
    )
    has_requested_hmc_match = (
        has_approved_exact_hmc
        if complete_hmc_requested
        else (bool(approved_structured_hmc) if component_hmc_requested else False)
    )
    component_record_ids = {
        _clean(item.get("record_id")) or _clean(item.get("evidence_id"))
        for item in approved_structured_hmc
        if _clean(item.get("match_type")) == "exact_components"
    }
    ambiguous_component_hmc = component_hmc_requested and len(component_record_ids) > 1

    # 结构化代码查询时，“PG 正常返回空”和“PG 查询失败”是两种不同的
    # 证据状态。两者都不能被 FMECA/OpenSearch 替代，但只有后者能说
    # “权威事实查询不可用”。
    hmc_postgres_failed = "hmc_query" in missing_pg_tools
    structured_hmc_backend_unavailable = (
        structured_hmc_requested and not has_requested_hmc_match and hmc_postgres_failed
    )
    structured_hmc_no_match = (
        structured_hmc_requested and not has_requested_hmc_match and not hmc_postgres_failed
    )
    if structured_hmc_no_match:
        # 用户提供了结构化 HMC 字段但权威精确查询未命中时，其他 HMC 关键词/
        # 语义近邻记录（例如 semantic_id_hydration、keyword、hybrid_search）
        # 不得被当作该代码的含义逐条列举（静默替代）。把它们从已批准集合
        # 移到已阻止集合；审计登记仍保留在 evidence registry 中。
        hmc_projection_ids = [
            _clean(item.get("evidence_id"))
            for item in approved_items
            if _clean(item.get("doc_type")) == "hmc"
            and _clean(item.get("match_type")) not in _STRUCTURED_HMC_MATCH_TYPES
        ]
        if hmc_projection_ids:
            hmc_projection_set = set(hmc_projection_ids)
            approved_items = [
                item
                for item in approved_items
                if _clean(item.get("evidence_id")) not in hmc_projection_set
            ]
            approved_ids = [_clean(item.get("evidence_id")) for item in approved_items]
            blocked_ids = _unique([*blocked_ids, *hmc_projection_ids])
            warnings.append(
                "用户输入的 HMC 结构化代码未命中权威记录，其他 HMC 关键词/语义近邻"
                "记录不得作为该代码的含义（已阻止作为该代码的直接证据）"
            )
    elif has_requested_hmc_match:
        # 精确命中的反面对称：上一轮或同类问题中其它相似报码的语义近邻记录
        # 不是当前代码的维修对象，混入会让“这个报码怎么处理”变成别的报码的
        # 处置（用户也常明确要求“别混相似报码”）。从已批准集合移到已阻止
        # 集合，审计登记仍保留。
        hmc_neighbor_ids = [
            _clean(item.get("evidence_id"))
            for item in approved_items
            if _clean(item.get("doc_type")) == "hmc"
            and _clean(item.get("match_type")) not in _STRUCTURED_HMC_MATCH_TYPES
        ]
        if hmc_neighbor_ids:
            hmc_neighbor_set = set(hmc_neighbor_ids)
            approved_items = [
                item
                for item in approved_items
                if _clean(item.get("evidence_id")) not in hmc_neighbor_set
            ]
            approved_ids = [_clean(item.get("evidence_id")) for item in approved_items]
            blocked_ids = _unique([*blocked_ids, *hmc_neighbor_ids])
            warnings.append(
                "已精确命中权威 HMC 记录，其他相似报码的语义近邻记录不得混入"
                "该代码的维修/处置说明（已阻止作为直接证据）"
            )
    for tool in missing_pg_tools:
        blocked_claims.append(f"{tool}_authoritative_facts")
        warnings.append(f"{tool} 的 PostgreSQL 权威事实查询失败，其他来源不能替代该类精确事实")
    if missing_pg_tools or pg_unavailable:
        blocked_claims.append("authoritative_source_unavailable")
    if pg_unavailable:
        blocked_claims.append("authoritative_diagnosis")
        warnings.append("PostgreSQL 权威事实不可用，只能返回检索线索")
    if structured_hmc_backend_unavailable or structured_hmc_no_match:
        blocked_claims.extend(["exact_hmc_fact", "authoritative_diagnosis"])
        if "hmc_query" not in missing_pg_tools:
            warnings.append(
                "HMC PostgreSQL 权威结构化查询已完成但未找到匹配记录，"
                "其他文档或检索投影不能替代"
            )
    if component_hmc_requested and has_approved_component_hmc and not ambiguous_component_hmc:
        warnings.append("HMC 按 LCN+HSC 字段命中；缺省的 HAMC 值来自权威记录，未由系统猜测")
    if incomplete_hmc_request:
        blocked_claims.append("incomplete_hmc_code")
        warnings.append("当前输入不包含完整 LCN/HAMC/HSC，未执行权威精确代码判定")

    questions: List[str] = []
    if existing_clarification:
        questions.append(existing_clarification)
    if scope_status == "conflict":
        questions.append("请确认本次问题适用的系统构型和有效文档版本。")
    if calculation_requested and unusable_graph and not has_approved_graph:
        questions.append(
            "相关工程图未达到 auto_validated/manual_confirmed 且当前有效的正式等级，"
            "本次不会使用 provisional 或 quarantined 图进行确定性计算。"
        )
    elif calculation_requested and not has_approved_calculation:
        questions.append("尚未得到基于正式验证图修订和完整参数的确定性计算结果。")
    if numeric_result_requested and incomplete_numeric_calculations:
        questions.append(
            "确定性算法仅返回了可安全使用的拓扑结果；关键参数缺失或模型触发计算限制，"
            "消除限制后才能给出正式数值。"
        )
    candidates = state.get("candidate_faults") or []
    support = float(state.get("diagnostic_support", state.get("diagnostic_confidence", 0.0)) or 0.0)
    weak_single_candidate = (
        diagnosis_intent
        and len(candidates) == 1
        and not bool(candidates[0].get("exact_hmc_match"))
        and not bool(candidates[0].get("structured_hmc_match"))
        and support < 0.8
    )
    if weak_single_candidate and not existing_clarification:
        questions.append(
            "当前只有一个非精确关键词候选，仍需补充原始告警/HMC、故障阶段或可复现条件后再收敛。"
        )
    if ambiguous_component_hmc and not existing_clarification:
        questions.append(
            "LCN 与 HSC 命中多条 HMC 权威记录，请补充 HAMC 或原始告警文字以区分候选。"
        )
    if not approved_ids and not (structured_hmc_requested or incomplete_hmc_request):
        if diagnosis_intent:
            questions.append(
                "当前没有可支撑权威结论的直接证据，"
                "请补充故障码、构型或有效文档范围。"
            )
        elif architecture_intent:
            questions.append("当前没有匹配的架构/功能权威记录，请确认系统、模块名称及有效文档范围。")
        else:
            # A complete knowledge question does not become incomplete merely
            # because retrieval returned no approvable record. The answer model
            # should state the evidence gap directly instead of asking a generic
            # diagnostic questionnaire that cannot improve the requested fact.
            warnings.append("当前没有可批准的直接证据，只能简洁说明资料缺口")
    if incomplete_hmc_request:
        questions.append(
            "当前输入未包含完整的 LCN/HAMC/HSC，请补充缺失分量（例如 HAMC）"
            "后再进行权威精确查询。"
        )
    if structured_hmc_backend_unavailable:
        questions.append(
            "HMC PostgreSQL 权威结构化查询失败，请在服务恢复后重试。"
        )
    elif structured_hmc_no_match:
        if component_hmc_requested:
            questions.append(
                "HMC PostgreSQL 权威查询已完成，但 LCN+HSC 未找到匹配记录；"
                "请核对两个字段，并补充 HAMC、适用构型或有效文档版本。"
            )
        else:
            questions.append(
                "HMC PostgreSQL 权威精确查询已完成但未找到匹配记录，"
                "请核对完整 LCN/HAMC/HSC、适用构型和有效文档版本。"
            )

    requires_clarification = bool(questions)
    if pg_unavailable or structured_hmc_backend_unavailable:
        status = "blocked"
    elif requires_clarification:
        status = "clarify"
    else:
        status = "answer"

    decision = {
        "status": status,
        "scope_status": scope_status,
        # 第一阶段不开放唯一故障结论；精确 HMC 也只证明记录命中。
        "can_claim_unique": False,
        "requires_clarification": requires_clarification,
        "context_evidence_ids": context_ids,
        "approved_evidence_ids": approved_ids,
        "blocked_evidence_ids": _unique(blocked_ids),
        "blocked_claims": _unique(blocked_claims),
        "unavailable_authoritative_tools": missing_pg_tools,
        "warnings": _unique(warnings),
        "questions": _unique(questions),
    }
    gate_status = "blocked" if status == "blocked" else ("partial" if status == "clarify" else "ok")
    return {
        "context_evidence_ids": context_ids,
        "approved_evidence_ids": approved_ids,
        "safety_decision": decision,
        "tool_results": [
            make_tool_result(
                tool="evidence_gate",
                status=gate_status,
                data=decision,
                evidence_ids=approved_ids,
                warnings=decision["warnings"],
            )
        ],
    }
