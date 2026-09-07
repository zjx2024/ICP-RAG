"""Shared authority contract for selecting a formal RBD model.

Natural-language names, aliases and retrieval similarity are useful locators,
but they are not engineering-object identities.  This module is the single
boundary used by diagnosis, adaptive state merging, the evidence gate and the
deterministic calculator.  A model becomes authoritative only when an exact
stable RBD identity, a selected RBD citation, or an exact persisted block
business ID resolves to one active formal graph revision.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from backend.icp.diagram_validation import FORMAL_VALIDATION_STATUSES
from backend.agent.current_model_selection import CONTRACT_VERSION as SELECTION_CONTRACT_VERSION
from backend.agent.current_model_selection import selection_matches_model


RBD_AUTHORITY_CONTRACT_VERSION = "rbd-authority-v1"
RBD_SINGLE_FAILURE_RELATION_VERSION = "rbd-single-failure-relation-v1"
RBD_SELECTOR_BASES = frozenset(
    {
        "exact_rbd_id",
        "selected_rbd_identity",
        "exact_block_business_id",
        "current_model_selection",
    }
)
# 语义模型选择服务的 ``not_represented`` 终态：用户点名的部件/失效路径在
# 本轮候选图中未被独立建模。它与“未解析”不同——这是模型对候选集合的明确
# 判定，允许发布确定性的否定回答（未找到该部件的直接成功路径），而不是
# 落回“无法确认+要求图号”的澄清式空发布。
RBD_NOT_REPRESENTED_BASE = "current_model_selection_not_represented"
RBD_BINDING_BASES = frozenset(
    {*RBD_SELECTOR_BASES, "explicit_model_set", RBD_NOT_REPRESENTED_BASE}
)
RBD_STABLE_ID_FIELDS = (
    "revision_id",
    "diagram_id",
    "diagram_key",
    "record_id",
)
# ``block_id`` and graph vertex ``id`` are deliberately absent.  They are
# revision-local topology keys, not durable cross-document business identity.
RBD_BLOCK_BUSINESS_ID_FIELDS = (
    "business_id",
    "module_id",
    "component_id",
    "function_id",
)

_RBD_EXPLICIT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9._-])RBD-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.IGNORECASE,
)
_LABELLED_STABLE_ID_RE = re.compile(
    r"\b(?:revision_id|diagram_id|diagram_key|record_id)\s*[:=]\s*"
    r"([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff._:/-]*)",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _identity(value: Any) -> str:
    return unicodedata.normalize("NFKC", _text(value)).casefold()


def _unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        rendered = _text(value)
        normalized = _identity(rendered)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(rendered)
    return result


def _layers(value: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    result: List[Mapping[str, Any]] = [value]
    for key in ("metadata", "document_metadata", "answer_fields", "scope"):
        nested = value.get(key)
        if not isinstance(nested, Mapping):
            continue
        result.append(nested)
        answer_fields = nested.get("answer_fields")
        if isinstance(answer_fields, Mapping):
            result.append(answer_fields)
        scope = nested.get("scope")
        if isinstance(scope, Mapping):
            result.append(scope)
    return result


def _first(value: Mapping[str, Any], key: str) -> str:
    for layer in _layers(value):
        rendered = _text(layer.get(key))
        if rendered:
            return rendered
    return ""


def _values(value: Mapping[str, Any], key: str) -> set[str]:
    return {
        normalized
        for layer in _layers(value)
        if (normalized := _identity(layer.get(key)))
    }


def rbd_validation_status(value: Mapping[str, Any]) -> str:
    explicit = _values(value, "validation_status")
    if len(explicit) > 1:
        return "conflicting_validation_status"
    status = next(iter(explicit), "")
    if status in FORMAL_VALIDATION_STATUSES:
        return status
    review_status = _first(value, "review_status")
    if review_status in {"confirmed", "manual_confirmed"}:
        return "manual_confirmed"
    return status or review_status


def rbd_stable_identity_values(value: Mapping[str, Any]) -> set[str]:
    return {
        normalized
        for layer in _layers(value)
        for key in RBD_STABLE_ID_FIELDS
        if (normalized := _identity(layer.get(key)))
    }


def rbd_scope_values(value: Mapping[str, Any], scope_name: str) -> set[str]:
    aliases = {
        "doc_id": ("doc_id",),
        "configuration": (
            "configuration",
            "applicable_configuration",
            "configuration_id",
            "config_id",
            "config",
        ),
    }[scope_name]
    return {
        normalized
        for layer in _layers(value)
        for key in aliases
        if (normalized := _identity(layer.get(key)))
    }


def _canonical_graph(value: Mapping[str, Any]) -> Mapping[str, Any]:
    graph = value.get("canonical_graph")
    if not isinstance(graph, Mapping):
        graph = value.get("graph")
    if not isinstance(graph, Mapping):
        raw_content = value.get("content")
        if isinstance(raw_content, str) and raw_content.strip().startswith("{"):
            try:
                parsed = json.loads(raw_content)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = {}
            graph = parsed if isinstance(parsed, Mapping) else {}
    return graph if isinstance(graph, Mapping) else {}


def rbd_block_business_ids(value: Mapping[str, Any]) -> set[str]:
    """Return durable business IDs explicitly stored on formal RBD blocks."""

    result: set[str] = set()
    graph = _canonical_graph(value)
    for collection_name in ("vertices", "nodes", "blocks"):
        raw_collection = graph.get(collection_name) or []
        if isinstance(raw_collection, Mapping):
            collection = raw_collection.values()
        elif isinstance(raw_collection, Sequence) and not isinstance(
            raw_collection, (str, bytes)
        ):
            collection = raw_collection
        else:
            continue
        for raw_block in collection:
            if not isinstance(raw_block, Mapping):
                continue
            properties = raw_block.get("properties")
            properties = properties if isinstance(properties, Mapping) else {}
            for field in RBD_BLOCK_BUSINESS_ID_FIELDS:
                for owner in (raw_block, properties):
                    normalized = _identity(owner.get(field))
                    if normalized:
                        result.add(normalized)

    # Evidence does not duplicate the full canonical graph in metadata.  The
    # diagnosis adapter stores only the explicit durable IDs for downstream
    # revalidation; arbitrary presentation fields are never inspected.
    for layer in _layers(value):
        declared = layer.get("rbd_block_business_ids")
        if isinstance(declared, Sequence) and not isinstance(
            declared, (str, bytes)
        ):
            result.update(
                normalized
                for raw in declared
                if (normalized := _identity(raw))
            )
    return result


def rbd_model_descriptor(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return revision-scoped identity and lifecycle facts for one RBD row."""

    declared_type = _identity(
        value.get("diagram_type") or value.get("doc_type") or _first(value, "doc_type")
    )
    if declared_type and declared_type != "rbd":
        return {}
    revision_values = _values(value, "revision_id")
    projection_values = _values(value, "projection_status")
    active_values = {
        layer.get("is_active")
        for layer in _layers(value)
        if isinstance(layer.get("is_active"), bool)
    }
    if (
        len(revision_values) != 1
        or len(projection_values) > 1
        or len(active_values) > 1
        or any(len(_values(value, key)) > 1 for key in ("diagram_id", "diagram_key"))
    ):
        return {}
    revision_id = _first(value, "revision_id")
    if not revision_id:
        # A diagram key and a revision ID are different identities.  Treating
        # record_id as both would make an old/stale revision indistinguishable
        # from the active one that reused the same business key.
        return {}
    validation_status = rbd_validation_status(value)
    projection_status = next(iter(projection_values), "")
    is_active = active_values == {True}
    return {
        "revision_id": revision_id,
        "diagram_id": _first(value, "diagram_id"),
        "diagram_key": _first(value, "diagram_key"),
        "record_id": _text(value.get("record_id")),
        "stable_ids": sorted(rbd_stable_identity_values(value)),
        "doc_ids": sorted(rbd_scope_values(value, "doc_id")),
        "configurations": sorted(rbd_scope_values(value, "configuration")),
        "validation_status": validation_status,
        "projection_status": projection_status,
        "is_active": is_active,
        "rbd_block_business_ids": sorted(rbd_block_business_ids(value)),
    }


def rbd_model_is_formal(value: Mapping[str, Any]) -> bool:
    descriptor = rbd_model_descriptor(value)
    return bool(
        descriptor
        and descriptor["validation_status"] in FORMAL_VALIDATION_STATUSES
        and descriptor["projection_status"] == "published"
        and descriptor["is_active"] is True
    )


def extract_explicit_rbd_ids(raw_user_query: Any) -> List[str]:
    """Extract syntactically explicit RBD/revision identities from the raw turn."""

    query = str(raw_user_query or "")
    return _unique(
        [
            *(match.group(0) for match in _RBD_EXPLICIT_ID_RE.finditer(query)),
            *(match.group(1) for match in _LABELLED_STABLE_ID_RE.finditer(query)),
        ]
    )


def literal_identity_in_query(raw_user_query: Any, value: Any) -> bool:
    """Check exact contiguous provenance without interpreting its meaning."""

    query = unicodedata.normalize("NFKC", str(raw_user_query or "")).casefold()
    token = unicodedata.normalize("NFKC", _text(value)).casefold()
    if not query or not token:
        return False
    start = 0
    while True:
        index = query.find(token, start)
        if index < 0:
            return False
        before = query[index - 1] if index else ""
        end = index + len(token)
        after = query[end] if end < len(query) else ""
        if not (before and before.isascii() and before.isalnum()) and not (
            after and after.isascii() and after.isalnum()
        ):
            return True
        start = index + 1


def selected_rbd_anchor_selectors(
    selected_anchors: Sequence[Any],
) -> List[Dict[str, set[str]]]:
    """Read exact RBD citation identities with document/configuration scope."""

    selectors: List[Dict[str, set[str]]] = []
    for raw_anchor in selected_anchors or []:
        if not isinstance(raw_anchor, Mapping):
            continue
        if _identity(raw_anchor.get("doc_type")) != "rbd":
            continue
        identities = rbd_stable_identity_values(raw_anchor)
        if not identities:
            continue
        selectors.append(
            {
                "identities": identities,
                "doc_ids": rbd_scope_values(raw_anchor, "doc_id"),
                "configurations": rbd_scope_values(raw_anchor, "configuration"),
            }
        )
    return selectors


def _scope_compatible(
    descriptor: Mapping[str, Any], selector: Mapping[str, Iterable[str]]
) -> bool:
    for key in ("doc_ids", "configurations"):
        requested = set(selector.get(key) or set())
        actual = set(descriptor.get(key) or [])
        if requested and not (requested & actual):
            return False
    return True


def _candidate_identity(descriptor: Mapping[str, Any]) -> str:
    revision_id = _identity(descriptor.get("revision_id"))
    return f"revision_id:{revision_id}" if revision_id else ""


def _model_binding(
    descriptor: Mapping[str, Any],
    *,
    selector_basis: str,
    selector_values: Sequence[Any],
) -> Dict[str, Any]:
    values = _unique(selector_values)
    normalized_values = {_identity(value) for value in values}
    business_ids = set(descriptor.get("rbd_block_business_ids") or [])
    result = {
        **dict(descriptor),
        "selector_basis": selector_basis,
        "selector_values": values,
        "matched_business_ids": (
            sorted(business_ids & normalized_values)
            if selector_basis == "exact_block_business_id"
            else []
        ),
    }
    return result


def bind_rbd_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    required: bool,
    raw_user_query: Any,
    explicit_rbd_ids: Sequence[Any] = (),
    explicit_block_business_ids: Sequence[Any] = (),
    anchor_selectors: Sequence[Mapping[str, Iterable[str]]] = (),
    allow_multiple: bool = False,
    current_selection: Mapping[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select formal RBD revisions and emit the canonical authority contract."""

    copied = [dict(row) for row in rows if isinstance(row, Mapping)]
    described = [
        (row, descriptor)
        for row in copied
        if (descriptor := rbd_model_descriptor(row))
    ]
    candidate_ids = _unique(
        _candidate_identity(descriptor) for _, descriptor in described
    )
    base: Dict[str, Any] = {
        "contract_version": RBD_AUTHORITY_CONTRACT_VERSION,
        "required": bool(required),
        "bound": False,
        "ambiguous": False,
        "basis": "",
        "record_ids": [],
        "candidate_record_ids": candidate_ids,
        "model_bindings": [],
    }
    if not required:
        return copied, base

    syntactic_rbd_ids = {
        _identity(value) for value in extract_explicit_rbd_ids(raw_user_query)
    }
    explicit_ids = [
        value
        for value in _unique(explicit_rbd_ids)
        if _identity(value) in syntactic_rbd_ids
    ]
    block_ids = [
        value
        for value in _unique(explicit_block_business_ids)
        if literal_identity_in_query(raw_user_query, value)
    ]
    selectors = [
        {
            "identities": set(item.get("identities") or set()),
            "doc_ids": set(item.get("doc_ids") or set()),
            "configurations": set(item.get("configurations") or set()),
        }
        for item in anchor_selectors
        if isinstance(item, Mapping) and set(item.get("identities") or set())
    ]

    selector_basis = ""
    selector_specs: List[Dict[str, Any]] = []
    if explicit_ids:
        selector_basis = "exact_rbd_id"
        selector_specs = [
            {"values": {_identity(value)}, "raw_values": [value]}
            for value in explicit_ids
        ]
    elif block_ids:
        selector_basis = "exact_block_business_id"
        selector_specs = [
            {"values": {_identity(value)}, "raw_values": [value]}
            for value in block_ids
        ]
    elif current_selection:
        selector_basis = "current_model_selection"
        if (
            current_selection.get("contract_version") == SELECTION_CONTRACT_VERSION
            and current_selection.get("status") == "not_represented"
        ):
            # 语义选择服务明确判定：用户点名的部件/失效路径在本轮候选图中
            # 未被独立建模。这与“未解析”不同——这是可发布的确定性否定
            # （不能把上级模块的框图当成该部件的直接模型），不是模糊匹配
            # 失败。绑定记录 not_represented 终态，消费方据此输出否定回答
            # 并保留候选图的审计目录，而不是要求用户提供图号。
            base["basis"] = RBD_NOT_REPRESENTED_BASE
            base["current_selection"] = dict(current_selection)
            base["not_represented_subjects"] = sorted(
                {
                    _text(value)
                    for value in (current_selection.get("selections") or [])
                    if isinstance(value, Mapping) and _text(value.get("matched_title"))
                }
            )
            return [], base
        selected_descriptors = [
            descriptor for _, descriptor in described
            if selection_matches_model(current_selection, descriptor, query=str(raw_user_query))
        ]
        expected_count = len(current_selection.get("selections") or [])
        if not selected_descriptors or len(selected_descriptors) != expected_count:
            base["basis"] = "current_model_selection_unresolved"
            base["ambiguous"] = current_selection.get("status") == "ambiguous"
            return [], base
        selector_specs = [
            {"values": {_identity(descriptor["revision_id"])},
             "raw_values": [descriptor["revision_id"]],
             "doc_ids": set(descriptor["doc_ids"]),
             "configurations": set(descriptor["configurations"])}
            for descriptor in selected_descriptors
        ]
        base["current_selection"] = dict(current_selection)
        # Multiple choices were explicitly made against the current question;
        # no retrieval top-k or title keyword implicitly authorizes a set.
        allow_multiple = expected_count > 1 or allow_multiple
    elif selectors:
        selector_basis = "selected_rbd_identity"
        selector_specs = [
            {
                "values": set(selector["identities"]),
                "raw_values": sorted(selector["identities"]),
                "doc_ids": selector["doc_ids"],
                "configurations": selector["configurations"],
            }
            for selector in selectors
        ]
    else:
        base["basis"] = (
            "single_unbound_candidate" if len(described) == 1 else "ambiguous_unresolved"
        )
        base["ambiguous"] = len(described) > 1
        return ([described[0][0]] if len(described) == 1 else []), base

    matched_by_spec: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = []
    for spec in selector_specs:
        matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for row, descriptor in described:
            if selector_basis == "exact_block_business_id":
                identity_match = bool(
                    set(descriptor["rbd_block_business_ids"]) & set(spec["values"])
                )
            else:
                identity_match = bool(
                    set(descriptor["stable_ids"]) & set(spec["values"])
                )
            if not identity_match or not _scope_compatible(descriptor, spec):
                continue
            matches.append((row, descriptor))
        matched_by_spec.append(matches)

    contextual_matches: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    formal_matches_by_spec: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = []
    for matches in matched_by_spec:
        for row, descriptor in matches:
            contextual_matches.setdefault(_candidate_identity(descriptor), (row, descriptor))
        formal_matches_by_spec.append(
            [(row, descriptor) for row, descriptor in matches if rbd_model_is_formal(row)]
        )

    if any(len(matches) != 1 for matches in formal_matches_by_spec):
        base["ambiguous"] = any(len(matches) > 1 for matches in formal_matches_by_spec)
        base["basis"] = (
            f"ambiguous_{selector_basis}"
            if base["ambiguous"]
            else f"{selector_basis}_not_formal_or_not_found"
        )
        contextual_rows = [row for row, _ in contextual_matches.values()]
        return (contextual_rows if len(contextual_rows) == 1 else []), base

    selected_by_revision: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], List[Any]]] = {}
    for spec, matches in zip(selector_specs, formal_matches_by_spec):
        row, descriptor = matches[0]
        key = _candidate_identity(descriptor)
        existing = selected_by_revision.get(key)
        if existing:
            existing[2].extend(spec["raw_values"])
        else:
            selected_by_revision[key] = (row, descriptor, list(spec["raw_values"]))
    if len(selected_by_revision) > 1 and not allow_multiple:
        base["ambiguous"] = True
        base["basis"] = f"ambiguous_{selector_basis}"
        return [], base

    model_bindings = [
        _model_binding(
            descriptor,
            selector_basis=selector_basis,
            selector_values=selector_values,
        )
        for _, descriptor, selector_values in selected_by_revision.values()
    ]
    record_ids = [
        _candidate_identity(descriptor)
        for _, descriptor, _ in selected_by_revision.values()
    ]
    base.update(
        {
            "bound": bool(model_bindings),
            "basis": (
                "explicit_model_set" if len(model_bindings) > 1 else selector_basis
            ),
            "record_ids": record_ids,
            "model_bindings": model_bindings,
        }
    )
    return (
        [row for row, _, _ in selected_by_revision.values()],
        base,
    )


def validate_rbd_binding_contract(binding: Any) -> Dict[str, Any]:
    """Validate the closed binding shape without trusting its provenance."""

    if not isinstance(binding, Mapping):
        return {"valid": False, "reason": "missing_binding", "revision_ids": []}
    if binding.get("contract_version") != RBD_AUTHORITY_CONTRACT_VERSION:
        return {"valid": False, "reason": "unknown_contract_version", "revision_ids": []}
    if binding.get("required") is not True:
        return {"valid": False, "reason": "binding_not_required", "revision_ids": []}
    if _text(binding.get("basis")) == RBD_NOT_REPRESENTED_BASE:
        # ``not_represented`` 是无模型的确定性终态：语义选择服务对“该部件
        # 未被独立建模”的明确判定。它没有可绑定的 revision，验证只检查
        # 终态字段自洽；消费方据此发布否定回答而非拓扑结论。
        if binding.get("ambiguous") is True:
            return {"valid": False, "reason": "not_represented_cannot_be_ambiguous", "revision_ids": []}
        selection = binding.get("current_selection")
        if not isinstance(selection, Mapping) or selection.get("status") != "not_represented":
            return {"valid": False, "reason": "not_represented_selection_missing", "revision_ids": []}
        return {
            "valid": True,
            "reason": "",
            "revision_ids": [],
            "not_represented": True,
            "model_bindings": [],
        }
    if binding.get("bound") is not True or binding.get("ambiguous") is True:
        return {"valid": False, "reason": "binding_not_unique", "revision_ids": []}
    basis = _text(binding.get("basis"))
    if basis not in RBD_BINDING_BASES:
        return {"valid": False, "reason": "unknown_binding_basis", "revision_ids": []}
    raw_models = binding.get("model_bindings")
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
        return {"valid": False, "reason": "missing_model_bindings", "revision_ids": []}
    models = [dict(item) for item in raw_models if isinstance(item, Mapping)]
    if not models or len(models) != len(raw_models):
        return {"valid": False, "reason": "invalid_model_binding", "revision_ids": []}
    if len(models) > 1 and basis != "explicit_model_set":
        return {"valid": False, "reason": "multi_model_basis_required", "revision_ids": []}
    if len(models) == 1 and basis == "explicit_model_set":
        return {"valid": False, "reason": "single_model_basis_mismatch", "revision_ids": []}

    revision_ids: List[str] = []
    seen: set[str] = set()
    for model in models:
        revision_id = _identity(model.get("revision_id"))
        selector_basis = _text(model.get("selector_basis"))
        selector_values = _unique(model.get("selector_values") or [])
        if not revision_id or revision_id in seen:
            return {"valid": False, "reason": "invalid_revision_identity", "revision_ids": []}
        if selector_basis not in RBD_SELECTOR_BASES or not selector_values:
            return {"valid": False, "reason": "invalid_selector_contract", "revision_ids": []}
        if len(models) == 1 and basis != selector_basis:
            return {"valid": False, "reason": "selector_basis_mismatch", "revision_ids": []}
        if (
            _text(model.get("validation_status")) not in FORMAL_VALIDATION_STATUSES
            or _text(model.get("projection_status")) != "published"
            or model.get("is_active") is not True
        ):
            return {"valid": False, "reason": "non_formal_model", "revision_ids": []}
        if selector_basis == "exact_block_business_id":
            matched = {_identity(value) for value in model.get("matched_business_ids") or []}
            if not matched or not matched <= {_identity(value) for value in selector_values}:
                return {"valid": False, "reason": "missing_business_id_intersection", "revision_ids": []}
        seen.add(revision_id)
        revision_ids.append(revision_id)

    declared_records = {_identity(value) for value in binding.get("record_ids") or []}
    expected_records = {f"revision_id:{revision_id}" for revision_id in revision_ids}
    if declared_records != expected_records:
        return {"valid": False, "reason": "record_revision_mismatch", "revision_ids": []}
    return {
        "valid": True,
        "reason": "",
        "revision_ids": revision_ids,
        "model_bindings": models,
    }


def validate_rbd_authority(
    binding: Any,
    *,
    models: Sequence[Mapping[str, Any]],
    raw_user_query: Any,
    selected_anchors: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Revalidate binding provenance, lifecycle and revision at a consumer."""

    structural = validate_rbd_binding_contract(binding)
    if not structural["valid"]:
        return structural
    actual: List[Tuple[int, Dict[str, Any]]] = []
    for index, value in enumerate(models or []):
        if not isinstance(value, Mapping) or not rbd_model_is_formal(value):
            continue
        descriptor = rbd_model_descriptor(value)
        if descriptor:
            actual.append((index, descriptor))
    anchor_selectors = selected_rbd_anchor_selectors(selected_anchors)
    matched_indexes: List[int] = []

    for expected in structural["model_bindings"]:
        expected_revision = _identity(expected.get("revision_id"))
        candidates = [
            (index, descriptor)
            for index, descriptor in actual
            if _identity(descriptor.get("revision_id")) == expected_revision
        ]
        if len(candidates) != 1:
            return {**structural, "valid": False, "reason": "revision_not_unique_at_consumer"}
        index, descriptor = candidates[0]
        for field in (
            "diagram_id",
            "diagram_key",
            "validation_status",
            "projection_status",
            "is_active",
        ):
            expected_value = expected.get(field)
            actual_value = descriptor.get(field)
            if field == "is_active":
                matches = expected_value is True and actual_value is True
            else:
                matches = _identity(expected_value) == _identity(actual_value)
            if not matches:
                return {**structural, "valid": False, "reason": f"consumer_{field}_mismatch"}
        for field in ("doc_ids", "configurations"):
            expected_values = {_identity(value) for value in expected.get(field) or []}
            actual_values = set(descriptor.get(field) or [])
            if expected_values != actual_values:
                return {**structural, "valid": False, "reason": f"consumer_{field}_mismatch"}

        selector_basis = _text(expected.get("selector_basis"))
        selector_values = {_identity(value) for value in expected.get("selector_values") or []}
        if selector_basis == "exact_rbd_id":
            explicit_query_ids = {
                _identity(value) for value in extract_explicit_rbd_ids(raw_user_query)
            }
            proven = bool(selector_values) and all(
                _identity(value) in explicit_query_ids
                and _identity(value) in set(descriptor["stable_ids"])
                for value in expected.get("selector_values") or []
            )
        elif selector_basis == "exact_block_business_id":
            actual_business_ids = set(descriptor["rbd_block_business_ids"])
            proven = bool(selector_values) and selector_values <= actual_business_ids and all(
                literal_identity_in_query(raw_user_query, value)
                for value in expected.get("selector_values") or []
            )
        elif selector_basis == "current_model_selection":
            proven = selection_matches_model(
                binding.get("current_selection"), descriptor, query=str(raw_user_query)
            ) and selector_values == {_identity(descriptor["revision_id"])}
        else:
            proven = any(
                bool(selector_values)
                and selector_values <= set(anchor.get("identities") or set())
                and bool(selector_values & set(descriptor["stable_ids"]))
                and _scope_compatible(descriptor, anchor)
                for anchor in anchor_selectors
            )
        if not proven:
            return {**structural, "valid": False, "reason": "selector_provenance_failed"}
        matched_indexes.append(index)

    if len(set(matched_indexes)) != len(matched_indexes):
        return {**structural, "valid": False, "reason": "revision_reused_by_multiple_bindings"}
    return {
        **structural,
        "valid": True,
        "reason": "",
        "model_indexes": matched_indexes,
    }


def _rbd_group_business_ids(value: Mapping[str, Any]) -> Dict[str, set[str]]:
    """Index explicit RBD branch groups by durable block business identity."""

    graph = _canonical_graph(value)
    groups: Dict[str, set[str]] = defaultdict(set)
    blocks_by_local_id: Dict[str, Mapping[str, Any]] = {}

    def business_ids(block: Mapping[str, Any]) -> set[str]:
        properties = block.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        return {
            identity
            for field in RBD_BLOCK_BUSINESS_ID_FIELDS
            for owner in (block, properties)
            if (identity := _identity(owner.get(field)))
        }

    for collection_name in ("vertices", "nodes", "blocks"):
        raw_collection = graph.get(collection_name) or []
        if isinstance(raw_collection, Mapping):
            collection = raw_collection.values()
        elif isinstance(raw_collection, Sequence) and not isinstance(
            raw_collection, (str, bytes)
        ):
            collection = raw_collection
        else:
            continue
        for raw_block in collection:
            if not isinstance(raw_block, Mapping):
                continue
            properties = raw_block.get("properties")
            properties = properties if isinstance(properties, Mapping) else {}
            local_id = _text(raw_block.get("id"))
            if local_id:
                blocks_by_local_id[local_id] = raw_block
            branch_id = _text(
                raw_block.get("branch_id") or properties.get("branch_id")
            )
            if not branch_id:
                continue
            group_id = branch_id.rsplit(":branch:", 1)[0]
            groups[group_id].update(business_ids(raw_block))
    raw_edges = graph.get("edges") or []
    if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, (str, bytes)):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping):
                continue
            properties = raw_edge.get("properties")
            properties = properties if isinstance(properties, Mapping) else {}
            branch_id = _text(
                raw_edge.get("branch_id") or properties.get("branch_id")
            )
            if not branch_id:
                continue
            group_id = branch_id.rsplit(":branch:", 1)[0]
            for endpoint in (raw_edge.get("source"), raw_edge.get("target")):
                block = blocks_by_local_id.get(_text(endpoint))
                if block is not None:
                    groups[group_id].update(business_ids(block))
    return groups


def _declared_single_failure_relations(value: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Read only the dedicated structured relation field, never prose aliases."""

    result: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    graph = _canonical_graph(value)
    graph_metadata = graph.get("metadata")
    extra_layers = [graph]
    if isinstance(graph_metadata, Mapping):
        extra_layers.append(graph_metadata)
    for layer in [*_layers(value), *extra_layers]:
        raw = layer.get("single_failure_relations")
        if isinstance(raw, Mapping):
            items: Sequence[Any] = [raw]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = raw
        else:
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            marker = repr(sorted((str(key), repr(nested)) for key, nested in item.items()))
            if marker in seen:
                continue
            seen.add(marker)
            result.append(item)
    return result


def validated_rbd_single_failure_relations(
    value: Mapping[str, Any],
    *,
    binding: Any,
    raw_user_query: Any,
    selected_anchors: Sequence[Any] = (),
    target_business_ids: Sequence[Any] = (),
    models: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Return formally asserted single-failure relations for one bound model.

    Human-readable success criteria and topology labels are deliberately not
    interpreted.  A relation must carry a closed machine contract, match the
    bound active revision, and identify failed/surviving members by durable
    business IDs that occur in one explicit branch group.
    """

    authority = validate_rbd_authority(
        binding,
        models=list(models) or [value],
        raw_user_query=raw_user_query,
        selected_anchors=selected_anchors,
    )
    if not authority.get("valid"):
        return []
    descriptor = rbd_model_descriptor(value)
    revision_id = _identity(descriptor.get("revision_id"))
    if revision_id not in set(authority.get("revision_ids") or []):
        return []
    validation_status = _identity(descriptor.get("validation_status"))
    groups = _rbd_group_business_ids(value)
    targets = {
        identity for raw in target_business_ids if (identity := _identity(raw))
    }
    accepted: List[Dict[str, Any]] = []
    for raw in _declared_single_failure_relations(value):
        if raw.get("contract_version") != RBD_SINGLE_FAILURE_RELATION_VERSION:
            continue
        if _text(raw.get("relation_type")) != "single_failure_survivability":
            continue
        if _identity(raw.get("revision_id")) != revision_id:
            continue
        if _identity(raw.get("validation_status")) != validation_status:
            continue
        derivation = _text(raw.get("derivation"))
        if derivation not in {
            "deterministic_topology_validation",
            "manual_confirmation",
        }:
            continue
        if (
            validation_status == "auto_validated"
            and derivation != "deterministic_topology_validation"
        ):
            continue
        if (
            validation_status == "manual_confirmed"
            and derivation != "manual_confirmation"
        ):
            continue
        if _text(raw.get("outcome")) != "success_criterion_satisfied":
            continue
        group_id = _text(raw.get("group_id"))
        group_ids = groups.get(group_id) or set()
        failed = {
            identity
            for item in (raw.get("failed_business_ids") or [])
            if (identity := _identity(item))
        }
        surviving = {
            identity
            for item in (raw.get("surviving_business_ids") or [])
            if (identity := _identity(item))
        }
        statement = _text(raw.get("statement"))
        if (
            not group_id
            or not group_ids
            or len(failed) != 1
            or not surviving
            or bool(failed & surviving)
            or not (failed | surviving) <= group_ids
            or not statement
        ):
            continue
        if targets and not (targets & failed):
            continue
        accepted.append(
            {
                "group_id": group_id,
                "failed_business_ids": sorted(failed),
                "surviving_business_ids": sorted(surviving),
                "statement": statement,
                "revision_id": descriptor["revision_id"],
            }
        )
    # More than one matching assertion is not a unique target relation.  The
    # consumer must request/bind a narrower exact object instead of choosing.
    return accepted if len(accepted) == 1 else []


def _binding_candidate_ids(binding: Mapping[str, Any]) -> set[str]:
    return {
        _identity(value)
        for key in ("candidate_record_ids", "record_ids")
        for value in (binding.get(key) or [])
        if _identity(value)
    }


def _binding_candidate_values(*bindings: Mapping[str, Any]) -> List[str]:
    values: Dict[str, str] = {}
    for binding in bindings:
        for key in ("candidate_record_ids", "record_ids"):
            for raw in binding.get(key) or []:
                rendered = _text(raw)
                normalized = _identity(rendered)
                if normalized:
                    values.setdefault(normalized, rendered)
    return [values[key] for key in sorted(values)]


def _model_fingerprint(model: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        _identity(model.get("revision_id")),
        _identity(model.get("diagram_id")),
        _identity(model.get("diagram_key")),
        tuple(sorted(_identity(value) for value in model.get("doc_ids") or [])),
        tuple(sorted(_identity(value) for value in model.get("configurations") or [])),
        _text(model.get("validation_status")),
        _text(model.get("projection_status")),
        model.get("is_active") is True,
    )


def merge_rbd_query_bindings(first: Any, second: Any) -> Dict[str, Any]:
    """Merge adaptive batches without allowing a weak batch to widen identity."""

    previous = dict(first) if isinstance(first, Mapping) else {}
    current = dict(second) if isinstance(second, Mapping) else {}
    previous_required = previous.get("required") is True
    current_required = current.get("required") is True
    if not previous_required and not current_required:
        return current
    if previous_required != current_required:
        survivor = previous if previous_required else current
        validation = validate_rbd_binding_contract(survivor)
        if validation["valid"]:
            return survivor
        candidates = _binding_candidate_values(survivor)
        return {
            "contract_version": RBD_AUTHORITY_CONTRACT_VERSION,
            "required": True,
            "bound": False,
            "ambiguous": bool(
                survivor.get("ambiguous") is True or len(candidates) > 1
            ),
            "basis": _text(survivor.get("basis") or "unresolved"),
            "record_ids": [],
            "candidate_record_ids": candidates,
            "model_bindings": [],
        }

    previous_validation = validate_rbd_binding_contract(previous)
    current_validation = validate_rbd_binding_contract(current)
    candidates = _binding_candidate_values(previous, current)

    def conflict(reason: str) -> Dict[str, Any]:
        return {
            "contract_version": RBD_AUTHORITY_CONTRACT_VERSION,
            "required": True,
            "bound": False,
            "ambiguous": True,
            "basis": reason,
            "record_ids": [],
            "candidate_record_ids": candidates,
            "model_bindings": [],
        }

    if previous_validation["valid"] and current_validation["valid"]:
        previous_models = {
            _identity(model.get("revision_id")): model
            for model in previous_validation["model_bindings"]
        }
        current_models = {
            _identity(model.get("revision_id")): model
            for model in current_validation["model_bindings"]
        }
        if set(previous_models) != set(current_models):
            return conflict("ambiguous_adaptive_rbd_model_conflict")
        if any(
            _model_fingerprint(previous_models[key])
            != _model_fingerprint(current_models[key])
            for key in previous_models
        ):
            return conflict("ambiguous_adaptive_rbd_revision_conflict")
        merged = dict(previous)
        merged["candidate_record_ids"] = candidates
        return merged

    if previous_validation["valid"] or current_validation["valid"]:
        strong = previous if previous_validation["valid"] else current
        weak = current if previous_validation["valid"] else previous
        strong_ids = {_identity(value) for value in strong.get("record_ids") or []}
        weak_ids = _binding_candidate_ids(weak)
        # A participating batch that asserts ``bound`` through an unknown/weak
        # basis is not neutral, even when it happens to repeat the same text ID.
        # Only an actually empty unresolved batch may leave the prior strong
        # decision untouched.
        if (
            weak.get("bound") is True
            or weak.get("ambiguous") is True
            or (weak_ids and not weak_ids <= strong_ids)
        ):
            return conflict("ambiguous_adaptive_rbd_model_conflict")
        merged = dict(strong)
        merged["candidate_record_ids"] = candidates
        return merged

    return {
        "contract_version": RBD_AUTHORITY_CONTRACT_VERSION,
        "required": True,
        "bound": False,
        "ambiguous": bool(
            previous.get("ambiguous") is True
            or current.get("ambiguous") is True
            or len(candidates) > 1
        ),
        "basis": (
            "ambiguous_adaptive_rbd_candidates"
            if len(candidates) > 1
            else _text(previous.get("basis") or current.get("basis") or "unresolved")
        ),
        "record_ids": [],
        "candidate_record_ids": candidates,
        "model_bindings": [],
    }
