"""Bounded evidence payloads and source-owned identity accessors.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.evidence_display import summarize_diagram_locator
from backend.agent.evidence_display import summarize_fta_calculation
from backend.agent.evidence_display import summarize_graph_structure
from backend.agent.evidence_display import summarize_rbd_calculation
from backend.agent.nodes.answer_presence import normalize_presence_text
from backend.agent.query_understanding import extract_business_codes
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple
import json
import math
import re
import unicodedata


def _first_structured_value(
    containers: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Any:
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


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


EVIDENCE_METADATA_MAX_DEPTH = 4


EVIDENCE_METADATA_MAX_ITEMS = 32


EVIDENCE_METADATA_MAX_LIST_ITEMS = 24


EVIDENCE_METADATA_MAX_STRING = 600


EVIDENCE_METADATA_MAX_NODES = 64


def _evidence_answer_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    if not metadata and isinstance(item.get("document_metadata"), Mapping):
        # Public citation payloads use ``document_metadata`` while the
        # internal Evidence contract uses ``metadata``.  Completion checks
        # should behave identically for either shape.
        metadata = item.get("document_metadata")
    fields = metadata.get("answer_fields") if isinstance(metadata.get("answer_fields"), Mapping) else {}
    return fields


def _explicit_function_ids(item: Mapping[str, Any]) -> set[str]:
    fields = _evidence_answer_fields(item)
    text = "\n".join(
        str(value or "")
        for value in (
            fields.get("product_flag"),
            fields.get("function"),
            item.get("title"),
            item.get("display_summary") or item.get("content"),
        )
    )
    return {
        value.upper()
        for value in re.findall(r"(?<![A-Za-z0-9])FUNC-[A-Za-z0-9._-]+", text, re.IGNORECASE)
    }


def _evidence_object_business_ids(item: Mapping[str, Any]) -> set[str]:
    """Return only explicit object IDs stored in structured answer fields.

    Record IDs such as ``FM-SYN-032`` identify the evidence row, not the
    module/function described by that row, so they are deliberately excluded.
    Free-text summaries are also excluded: a code mentioned in a cause or
    effect paragraph cannot establish cross-document entity identity.
    """

    fields = _evidence_answer_fields(item)
    values = [
        fields.get("product_flag"),
        fields.get("component_id"),
        fields.get("module_id"),
        fields.get("function_id"),
        fields.get("interface_id"),
        fields.get("collection_parent_id"),
        fields.get("collection_member_id"),
    ]
    return {
        normalized
        for value in extract_business_codes(values, limit=24)
        if (normalized := _normalize_business_id(value))
    }


_IDENTITY_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:[A-Z][A-Z0-9]*(?:[-_.][A-Z0-9*]+)+)|(?:[A-Z]{2,4}))"
    r"(?![A-Za-z0-9])"
)


_SOURCE_FAMILY_CODES = {
    "API", "END", "FMECA", "FTA", "HMC", "MFL", "RBD", "START",
}


def _exact_structured_identity_codes(values: Sequence[Any]) -> set[str]:
    """Return stored code identities without promoting ordinary prose.

    Hyphenated/dotted business IDs may be long; bare values are limited to
    short, explicitly uppercase abbreviations.  In particular this must not
    turn English labels such as ``input conditioning module`` into the fake
    identities INPUT/CONDITIONING/MODULE.
    """

    result: set[str] = set()
    for raw in values:
        for match in _IDENTITY_CODE_RE.finditer(str(raw or "")):
            value = match.group(1).upper()
            if value not in _SOURCE_FAMILY_CODES:
                result.add(value)
    return result


def _binding_identity_codes(binding: Mapping[str, Any]) -> set[str]:
    if binding.get("bound") is not True:
        return set()
    return _exact_structured_identity_codes(
        [*(binding.get("business_ids") or []), *(binding.get("terms") or [])]
    )


def _evidence_object_identity_codes(item: Mapping[str, Any]) -> set[str]:
    """Read exact object codes from identity columns, excluding narrative."""

    fields = _evidence_answer_fields(item)
    return _exact_structured_identity_codes(
        [
            fields.get("product_flag"),
            fields.get("abbreviation"),
            fields.get("component_id"),
            fields.get("module_id"),
            fields.get("function_id"),
            fields.get("interface_id"),
            fields.get("collection_member_id"),
        ]
    )


def _selected_anchor_record_ids(
    state: Mapping[str, Any], doc_type: str
) -> set[str]:
    """Exact current-turn record identities selected from prior citations."""

    return {
        normalize_presence_text(anchor.get("record_id"))
        for anchor in (state.get("selected_conversation_anchors") or [])
        if isinstance(anchor, Mapping)
        and str(anchor.get("doc_type") or "").strip() == doc_type
        and normalize_presence_text(anchor.get("record_id"))
    }


def _answer_relevance_score(item: Mapping[str, Any]) -> Tuple[float, float]:
    answer_score = item.get("answer_rerank_score")
    retrieval_score = item.get("rerank_score")
    if not isinstance(retrieval_score, (int, float)):
        retrieval_score = item.get("retrieval_score")
    return (
        float(answer_score) if isinstance(answer_score, (int, float)) else -1.0,
        float(retrieval_score) if isinstance(retrieval_score, (int, float)) else -1.0,
    )


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit]


def _bounded_json_value(
    value: Any,
    *,
    depth: int = 0,
    _budget: List[int] | None = None,
) -> Any:
    """Convert arbitrary evidence metadata into bounded JSON-safe values."""
    if _budget is None:
        _budget = [EVIDENCE_METADATA_MAX_NODES]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _bounded_text(value, EVIDENCE_METADATA_MAX_STRING)
    if isinstance(value, bytes):
        return _bounded_text(value.decode("utf-8", errors="replace"), EVIDENCE_METADATA_MAX_STRING)
    if depth >= EVIDENCE_METADATA_MAX_DEPTH:
        return _bounded_text(value, EVIDENCE_METADATA_MAX_STRING)
    if isinstance(value, dict):
        bounded: Dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:EVIDENCE_METADATA_MAX_ITEMS]:
            if _budget[0] <= 0:
                break
            _budget[0] -= 1
            key = _bounded_text(raw_key, 160)
            bounded[key] = _bounded_json_value(
                raw_value,
                depth=depth + 1,
                _budget=_budget,
            )
        return bounded
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        if isinstance(value, (set, frozenset)):
            values.sort(key=lambda item: str(item))
        bounded_values = []
        for item in values[:EVIDENCE_METADATA_MAX_LIST_ITEMS]:
            if _budget[0] <= 0:
                break
            _budget[0] -= 1
            bounded_values.append(
                _bounded_json_value(item, depth=depth + 1, _budget=_budget)
            )
        return bounded_values
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return _bounded_text(isoformat(), EVIDENCE_METADATA_MAX_STRING)
        except Exception:
            pass
    return _bounded_text(value, EVIDENCE_METADATA_MAX_STRING)


def _bounded_string_list(value: Any, *, item_limit: int) -> List[str]:
    if value is None:
        values: List[Any] = []
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        values = [value]
    return [
        _bounded_text(item, item_limit)
        for item in values[:EVIDENCE_METADATA_MAX_LIST_ITEMS]
    ]


def _public_display_summary(item: Dict[str, Any]) -> str:
    """Return a readable summary without exposing structured JSON to clients.

    New producers set ``display_summary`` explicitly.  The JSON fallback keeps
    manually-created/legacy in-memory Evidence safe as well; malformed or
    truncated machine data is represented by its title instead of being sent
    verbatim as an apparent source excerpt.
    """

    explicit = str(item.get("display_summary") or "").strip()
    if explicit:
        return explicit
    claim_type = str(item.get("claim_type") or "").strip()
    if claim_type == "diagram_structure":
        return summarize_diagram_locator(
            doc_type=str(item.get("doc_type") or ""),
            title=str(item.get("title") or ""),
        )
    if claim_type not in {
        "fta_structure",
        "rbd_structure",
        "fta_calculation",
        "rbd_calculation",
    }:
        return ""
    content = str(item.get("content") or "").strip()
    if not content.startswith(("{", "[")):
        return ""
    try:
        structured = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        structured = None
    if isinstance(structured, dict):
        if claim_type == "fta_structure":
            return summarize_graph_structure(
                structured,
                diagram_type="fta",
                title=str(item.get("title") or ""),
            )
        if claim_type == "rbd_structure":
            return summarize_graph_structure(
                structured,
                diagram_type="rbd",
                title=str(item.get("title") or ""),
            )
        if claim_type == "fta_calculation":
            return summarize_fta_calculation(structured)
        return summarize_rbd_calculation(structured)
    label = {
        "fta_structure": "FTA 结构",
        "rbd_structure": "RBD 结构",
        "fta_calculation": "FTA 确定性计算",
        "rbd_calculation": "RBD 确定性计算",
    }[claim_type]
    title = str(item.get("title") or "未命名证据").strip()
    return f"{label}：{title}。详细机器结构请按来源定位查看。"


def _evidence_payload(item: Dict[str, Any], *, content_limit: int = 3500) -> Dict[str, Any]:
    evidence_id = str(item.get("evidence_id") or "")
    raw_scope = item.get("scope") or {}
    metadata = dict(raw_scope) if isinstance(raw_scope, dict) else {}
    raw_metadata = item.get("metadata") or {}
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    bounded_metadata = _bounded_json_value(metadata)
    display_summary = _bounded_text(_public_display_summary(item), content_limit)
    public_content = display_summary or _bounded_text(item.get("content"), content_limit)
    return {
        "evidence_id": evidence_id,
        "doc_id": _bounded_text(item.get("doc_id"), 240),
        "doc_name": _bounded_text(item.get("doc_name"), 500),
        "title": _bounded_text(item.get("title"), 500),
        "chunk_id": _bounded_text(item.get("chunk_id"), 240),
        "record_id": _bounded_text(item.get("record_id"), 240),
        "content": public_content,
        "display_summary": display_summary,
        "type": _bounded_text(item.get("claim_type") or "evidence", 120),
        "source_ref": (
            _bounded_text(
                item.get("source_ref") or item.get("record_id")
                or item.get("chunk_id") or evidence_id,
                800,
            )
        ),
        "document_metadata": bounded_metadata if isinstance(bounded_metadata, dict) else {},
        "source_kind": _bounded_text(item.get("source_kind"), 80),
        "doc_type": _bounded_text(item.get("doc_type"), 80),
        "authority": _bounded_text(item.get("authority"), 120),
        "evidence_level": _bounded_text(item.get("evidence_level"), 80),
        "review_status": _bounded_text(item.get("review_status"), 80),
        "match_type": _bounded_text(item.get("match_type"), 120),
        "retrieval_score": _bounded_json_value(item.get("retrieval_score")),
        "rerank_score": _bounded_json_value(item.get("rerank_score")),
        # Kept in the audit payload so evaluation can distinguish source
        # authority from answer-query relevance.  The UI may ignore it.
        "answer_rerank_score": _bounded_json_value(item.get("answer_rerank_score")),
        "derived_from": _bounded_string_list(item.get("derived_from"), item_limit=240),
        "limitations": _bounded_string_list(item.get("limitations"), item_limit=600),
    }


def _json_object(value: Any) -> Mapping[str, Any]:
    """Parse a JSON object without treating arbitrary prose as structured data."""

    if isinstance(value, Mapping):
        return value
    text = str(value or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _normalize_business_id(value: Any) -> str:
    """Normalize a durable engineering ID without erasing its separators."""

    # Search-term normalization intentionally removes punctuation, but
    # cross-document identity equality cannot: AB-C and A-BC are different
    # business IDs even though both collapse to ``abc`` for lexical recall.
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


_DIAGRAM_IDENTITY_SUFFIX = re.compile(r"(?:可靠性)?模型$")


def _diagram_identity_check_key(identity: str) -> str:
    """Return the distinct chain name used to detect an identity in the answer."""

    stripped = _DIAGRAM_IDENTITY_SUFFIX.sub("", str(identity or "").strip())
    if len(stripped) >= 4:
        return stripped
    return str(identity or "").strip()
