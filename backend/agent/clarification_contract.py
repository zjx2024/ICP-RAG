"""Machine-readable clarification contracts for the ICP agent.

Retrieval and safety nodes may know *why* a claim cannot be made and may know
which already-defined state-machine slot is unresolved.  They do not know how
the user should be asked about that slot.  Keeping those two concerns separate
prevents deterministic nodes from turning a small list of Chinese examples
into an accidental intent classifier or a canonical user-facing question.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional


MISSING_INPUT_KINDS = frozenset(
    {
        "identity_slot",
        "calculation_parameter",
        "applicability_scope",
    }
)
CLARIFICATION_REASON_KINDS = frozenset(
    {
        "insufficient_evidence",
        "ambiguous_identity",
        "unresolved_identity",
        "unresolved_relation",
        "scope_conflict",
        "authoritative_source_unavailable",
        "authoritative_record_not_found",
        "unvalidated_graph",
        "incomplete_calculation",
    }
)
CLARIFICATION_SOURCE_KINDS = frozenset(
    {
        "state",
        "query_frame",
        "binding",
        "tool_result",
        "evidence",
        "evidence_synthesis",
        "evidence_gate",
    }
)
CLARIFICATION_FALLBACK_KINDS = frozenset({"generic_clarification"})
ICP_DOC_TYPES = frozenset(
    {
        "function_locator",
        "hmc",
        "fmeca",
        "fta",
        "rbd",
        "mfl",
        "fault_history",
    }
)
CLARIFICATION_IDENTITY_FIELDS = frozenset(
    {
        "lcn",
        "hamc",
        "hsc",
        "binding_basis",
        "candidate_count",
        "diagnostic_support",
        "parameter",
        "slot",
        "system",
        "configuration",
        "document_number",
        "document_version",
        "block_ids",
        "event_ids",
        "excluded_mfl_codes",
    }
)


def _clean(value: Any, *, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def _unique_strings(values: Iterable[Any], *, limit: int = 32) -> List[str]:
    result: List[str] = []
    for raw in values:
        value = _clean(raw)
        if value and value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def make_clarification_source(
    *,
    kind: str,
    name: str,
    evidence_ids: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Build a closed source locator; ``name`` identifies the concrete node."""

    if kind not in CLARIFICATION_SOURCE_KINDS:
        raise ValueError(f"unsupported clarification source kind: {kind}")
    normalized_name = _clean(name, limit=160)
    if not normalized_name:
        raise ValueError("clarification source name is required")
    return {
        "kind": kind,
        "name": normalized_name,
        "evidence_ids": _unique_strings(evidence_ids),
    }


def make_clarification_identity(
    *,
    doc_type: str = "",
    record_ids: Iterable[Any] = (),
    fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an auditable identity without accepting free-form prose."""

    normalized_doc_type = _clean(doc_type, limit=40)
    if normalized_doc_type and normalized_doc_type not in ICP_DOC_TYPES:
        raise ValueError(f"unsupported clarification identity doc_type: {doc_type}")
    if isinstance(record_ids, (str, bytes, Mapping)):
        raise ValueError("clarification record_ids must be a collection")
    if fields is not None and not isinstance(fields, Mapping):
        raise ValueError("clarification identity fields must be a mapping")
    normalized_fields: Dict[str, str] = {}
    for raw_key, raw_value in list((fields or {}).items())[:24]:
        key = _clean(raw_key, limit=80)
        value = _clean(raw_value)
        if key and key not in CLARIFICATION_IDENTITY_FIELDS:
            raise ValueError(f"unsupported clarification identity field: {key}")
        if key and value:
            normalized_fields[key] = value
    return {
        "doc_type": normalized_doc_type,
        "record_ids": _unique_strings(record_ids),
        "fields": normalized_fields,
    }


def normalize_clarification_identity(value: Any) -> Optional[Dict[str, Any]]:
    """Validate identity objects crossing nodes; malformed objects are denied."""

    if not isinstance(value, Mapping):
        return None
    record_ids = value.get("record_ids") or []
    fields = value.get("fields") or {}
    if not isinstance(record_ids, (list, tuple, set)) or not isinstance(fields, Mapping):
        return None
    try:
        return make_clarification_identity(
            doc_type=value.get("doc_type") or "",
            record_ids=record_ids,
            fields=fields,
        )
    except ValueError:
        return None


def _contract_identity(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return make_clarification_identity()
    normalized = normalize_clarification_identity(value)
    if normalized is None:
        raise ValueError("invalid clarification identity")
    return normalized


def make_missing_input_requirement(
    *,
    kind: str,
    slot: str,
    source_kind: str,
    source_name: str,
    identity: Optional[Mapping[str, Any]] = None,
    basis: Any = "",
    evidence_ids: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Describe a real state-machine gap, never a suggested question."""

    if kind not in MISSING_INPUT_KINDS:
        raise ValueError(f"unsupported missing-input kind: {kind}")
    normalized_slot = _clean(slot, limit=160)
    if not normalized_slot:
        raise ValueError("missing-input slot is required")
    result: Dict[str, Any] = {
        "kind": kind,
        "slot": normalized_slot,
        "source": make_clarification_source(
            kind=source_kind,
            name=source_name,
            evidence_ids=evidence_ids,
        ),
        "identity": _contract_identity(identity),
    }
    normalized_basis = _clean(basis, limit=500)
    if normalized_basis:
        result["basis"] = normalized_basis
    return result


def make_clarification_reason(
    *,
    kind: str,
    source_kind: str,
    source_name: str,
    identity: Optional[Mapping[str, Any]] = None,
    blocked_claims: Iterable[Any] = (),
    evidence_ids: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Describe a safety condition without prescribing user-facing wording."""

    if kind not in CLARIFICATION_REASON_KINDS:
        raise ValueError(f"unsupported clarification reason kind: {kind}")
    return {
        "kind": kind,
        "source": make_clarification_source(
            kind=source_kind,
            name=source_name,
            evidence_ids=evidence_ids,
        ),
        "identity": _contract_identity(identity),
        "blocked_claims": _unique_strings(blocked_claims),
    }


def make_clarification_fallback(*, required: bool) -> Dict[str, Any]:
    """Return a marker only; the answer model owns the actual question."""

    return {"kind": "generic_clarification", "required": bool(required)}


def normalize_missing_input_requirements(value: Any) -> List[Dict[str, Any]]:
    """Validate contracts crossing a node boundary; unknown kinds fail closed."""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for raw in value[:32]:
        if not isinstance(raw, Mapping):
            continue
        source = raw.get("source")
        if not isinstance(source, Mapping):
            continue
        kind = _clean(raw.get("kind"), limit=80)
        source_kind = _clean(source.get("kind"), limit=80)
        if kind not in MISSING_INPUT_KINDS or source_kind not in CLARIFICATION_SOURCE_KINDS:
            continue
        try:
            result.append(
                make_missing_input_requirement(
                    kind=kind,
                    slot=raw.get("slot"),
                    source_kind=source_kind,
                    source_name=source.get("name"),
                    identity=(
                        raw.get("identity")
                        if isinstance(raw.get("identity"), Mapping)
                        else {}
                    ),
                    basis=raw.get("basis"),
                    evidence_ids=source.get("evidence_ids") or [],
                )
            )
        except ValueError:
            continue
    return unique_contract_items(result)


def normalize_clarification_reasons(value: Any) -> List[Dict[str, Any]]:
    """Validate reason records without interpreting their natural language."""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for raw in value[:32]:
        if not isinstance(raw, Mapping):
            continue
        source = raw.get("source")
        if not isinstance(source, Mapping):
            continue
        kind = _clean(raw.get("kind"), limit=80)
        source_kind = _clean(source.get("kind"), limit=80)
        if (
            kind not in CLARIFICATION_REASON_KINDS
            or source_kind not in CLARIFICATION_SOURCE_KINDS
        ):
            continue
        try:
            result.append(
                make_clarification_reason(
                    kind=kind,
                    source_kind=source_kind,
                    source_name=source.get("name"),
                    identity=(
                        raw.get("identity")
                        if isinstance(raw.get("identity"), Mapping)
                        else {}
                    ),
                    blocked_claims=raw.get("blocked_claims") or [],
                    evidence_ids=source.get("evidence_ids") or [],
                )
            )
        except ValueError:
            continue
    return unique_contract_items(result)


def unique_contract_items(*groups: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Stable de-duplication for reducer-free state fields."""

    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group or ():
            item = dict(raw)
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result
