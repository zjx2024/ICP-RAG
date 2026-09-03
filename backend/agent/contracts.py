"""ICP 智能体的工具与证据契约。

契约故意使用 ``TypedDict`` 和普通字典，而不引入新的运行时依赖。
旧 Agent 仍可以继续传递原有的 ``domain_results`` / ``vector_results``；
新 ICP 链路逐步将每个结果规范化为 Evidence，再由证据门禁决定用途。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict


EvidenceSourceKind = Literal[
    "postgres",
    "opensearch",
    "hugegraph",
    "artifact",
    "calculation",
]
EvidenceAuthority = Literal[
    "authoritative_record",
    "source_artifact",
    "search_projection",
    "confirmed_relation",
    "validated_relation",
    "unconfirmed_extraction",
    "deterministic_derivation",
]
EvidenceLevel = Literal[
    "L1_DIRECT",
    "L2_CORROBORATED",
    "L3_INFERRED",
    "L4_UNRESOLVED",
]
ReviewStatus = Literal[
    "confirmed",
    "manual_confirmed",
    "auto_validated",
    "provisional",
    "pending",
    "quarantined",
    "failed",
    "rejected",
    "not_required",
]
ToolStatus = Literal["ok", "partial", "empty", "failed", "blocked"]
AnswerMode = Literal[
    "architecture_knowledge",
    "knowledge_explanation",
    "fault_diagnosis",
    "maintenance_guidance",
    "exact_code_lookup",
]


EVIDENCE_SOURCE_KINDS = {
    "postgres",
    "opensearch",
    "hugegraph",
    "artifact",
    "calculation",
}
EVIDENCE_AUTHORITIES = {
    "authoritative_record",
    "source_artifact",
    "search_projection",
    "confirmed_relation",
    "validated_relation",
    "unconfirmed_extraction",
    "deterministic_derivation",
}
EVIDENCE_LEVELS = {
    "L1_DIRECT",
    "L2_CORROBORATED",
    "L3_INFERRED",
    "L4_UNRESOLVED",
}
REVIEW_STATUSES = {
    "confirmed",
    "manual_confirmed",
    "auto_validated",
    "provisional",
    "pending",
    "quarantined",
    "failed",
    "rejected",
    "not_required",
}
TOOL_STATUSES = {"ok", "partial", "empty", "failed", "blocked"}
ANSWER_MODES = {
    "architecture_knowledge",
    "knowledge_explanation",
    "fault_diagnosis",
    "maintenance_guidance",
    "exact_code_lookup",
}


def answer_mode_for_intent(intent: Any) -> AnswerMode:
    """Map retrieval intent to the stable user-facing answer contract.

    Evidence gathering can still produce diagnostic candidates for every intent,
    but a knowledge question must not inherit the full diagnostic report shape.
    Keeping this mapping deterministic also prevents a planner fallback from
    exposing chain-of-thought-like evidence tiers to users.
    """
    normalized = _clean_string(intent)
    if normalized in {"architecture_query", "entity_profile", "function_location"}:
        return "architecture_knowledge"
    if normalized == "exact_code":
        return "exact_code_lookup"
    if normalized == "maintenance_query":
        return "maintenance_guidance"
    if normalized in {
        "cause_analysis",
        "impact_analysis",
        "reliability_query",
        "history_query",
    }:
        return "knowledge_explanation"
    return "fault_diagnosis"


class EvidenceScope(TypedDict, total=False):
    system: str
    configuration: str
    document_number: str
    document_version: str


class Evidence(TypedDict):
    evidence_id: str
    source_kind: EvidenceSourceKind
    authority: EvidenceAuthority
    evidence_level: EvidenceLevel
    claim_type: str
    doc_type: str
    doc_id: str
    doc_name: str
    record_id: str
    chunk_id: str
    source_ref: str
    title: str
    content: str
    # Human-readable, deterministic projection of structured evidence.  The
    # internal ``content`` field may retain canonical JSON for model/audit use;
    # browser payloads must prefer this summary when it is present.
    display_summary: str
    match_type: str
    review_status: ReviewStatus
    scope: EvidenceScope
    retrieval_score: Optional[float]
    rerank_score: Optional[float]
    derived_from: List[str]
    limitations: List[str]
    metadata: Dict[str, Any]


class ToolDiagnostics(TypedDict):
    backend: str
    fallbacks: List[str]
    warnings: List[str]
    errors: List[str]


class ToolResult(TypedDict):
    tool: str
    status: ToolStatus
    params: Dict[str, Any]
    data: Any
    evidence_ids: List[str]
    diagnostics: ToolDiagnostics


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evidence_identity(
    *,
    source_kind: str,
    doc_type: str = "",
    doc_id: str = "",
    doc_name: str = "",
    record_id: str = "",
    chunk_id: str = "",
    source_ref: str = "",
    claim_type: str = "",
    content: str = "",
) -> str:
    """生成稳定、不依赖返回顺序的证据 ID。

    PostgreSQL 行通常使用 ``doc_id + record_id``，OpenSearch 命中通常
    使用 ``doc_id + chunk_id``。只有当来源没有稳定业务键时才使用内容。
    ``claim_type`` 使同一 HMC 行的“故障现象”与“维修方法”可成为独立证据。
    """
    identity = {
        "source_kind": _clean_string(source_kind),
        "doc_type": _clean_string(doc_type),
        "doc_id": _clean_string(doc_id),
        "record_id": _clean_string(record_id),
        "chunk_id": _clean_string(chunk_id),
        "source_ref": _clean_string(source_ref),
        "claim_type": _clean_string(claim_type),
    }
    if not any(identity[key] for key in ("doc_id", "record_id", "chunk_id", "source_ref")):
        identity["content"] = _clean_string(content)
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"ev_{digest}"


def make_evidence(
    *,
    source_kind: EvidenceSourceKind,
    authority: EvidenceAuthority,
    evidence_level: EvidenceLevel,
    claim_type: str,
    doc_type: str = "",
    doc_id: str = "",
    doc_name: str = "",
    record_id: str = "",
    chunk_id: str = "",
    source_ref: str = "",
    title: str = "",
    content: str = "",
    display_summary: str = "",
    match_type: str = "",
    review_status: ReviewStatus = "not_required",
    scope: Optional[Dict[str, Any]] = None,
    retrieval_score: Any = None,
    rerank_score: Any = None,
    derived_from: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    evidence_id: str = "",
) -> Evidence:
    """构造字段完整的 Evidence，对枚举字段做快速失败校验。"""
    if source_kind not in EVIDENCE_SOURCE_KINDS:
        raise ValueError(f"不支持的证据来源: {source_kind}")
    if authority not in EVIDENCE_AUTHORITIES:
        raise ValueError(f"不支持的证据权威类型: {authority}")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"不支持的证据级别: {evidence_level}")
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"不支持的审核状态: {review_status}")

    normalized_scope: EvidenceScope = {}
    for key in ("system", "configuration", "document_number", "document_version"):
        value = _clean_string((scope or {}).get(key))
        if value:
            normalized_scope[key] = value

    normalized_derived = list(
        dict.fromkeys(_clean_string(item) for item in (derived_from or []) if _clean_string(item))
    )
    normalized_limitations = list(
        dict.fromkeys(_clean_string(item) for item in (limitations or []) if _clean_string(item))
    )
    normalized_id = _clean_string(evidence_id) or evidence_identity(
        source_kind=source_kind,
        doc_type=doc_type,
        doc_id=doc_id,
        record_id=record_id,
        chunk_id=chunk_id,
        source_ref=source_ref,
        claim_type=claim_type,
        content=content,
    )

    return {
        "evidence_id": normalized_id,
        "source_kind": source_kind,
        "authority": authority,
        "evidence_level": evidence_level,
        "claim_type": _clean_string(claim_type),
        "doc_type": _clean_string(doc_type),
        "doc_id": _clean_string(doc_id),
        "doc_name": _clean_string(doc_name),
        "record_id": _clean_string(record_id),
        "chunk_id": _clean_string(chunk_id),
        "source_ref": _clean_string(source_ref),
        "title": _clean_string(title),
        "content": _clean_string(content),
        "display_summary": _clean_string(display_summary),
        "match_type": _clean_string(match_type),
        "review_status": review_status,
        "scope": normalized_scope,
        "retrieval_score": _optional_float(retrieval_score),
        "rerank_score": _optional_float(rerank_score),
        "derived_from": normalized_derived,
        "limitations": normalized_limitations,
        "metadata": dict(metadata or {}),
    }


def make_tool_result(
    *,
    tool: str,
    status: ToolStatus,
    params: Optional[Dict[str, Any]] = None,
    data: Any = None,
    evidence_ids: Optional[List[str]] = None,
    backend: str = "",
    fallbacks: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> ToolResult:
    """构造可区分空结果、失败、部分降级和安全阻断的工具结果。"""
    if status not in TOOL_STATUSES:
        raise ValueError(f"不支持的工具状态: {status}")
    return {
        "tool": _clean_string(tool),
        "status": status,
        "params": dict(params or {}),
        "data": data,
        "evidence_ids": list(
            dict.fromkeys(_clean_string(item) for item in (evidence_ids or []) if _clean_string(item))
        ),
        "diagnostics": {
            "backend": _clean_string(backend),
            "fallbacks": [_clean_string(item) for item in (fallbacks or []) if _clean_string(item)],
            "warnings": [_clean_string(item) for item in (warnings or []) if _clean_string(item)],
            "errors": [_clean_string(item) for item in (errors or []) if _clean_string(item)],
        },
    }


def deduplicate_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 evidence_id 去重；保留首次出现顺序和最后一份完整内容。"""
    ordered_ids: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        evidence_id = _clean_string(item.get("evidence_id"))
        if not evidence_id:
            continue
        if evidence_id not in by_id:
            ordered_ids.append(evidence_id)
        by_id[evidence_id] = dict(item)
    return [by_id[evidence_id] for evidence_id in ordered_ids]
