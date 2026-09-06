"""Deterministic presence checks over already-scoped fact manifests.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.nodes.answer_collections import _architecture_declared_member_counts
from backend.agent.nodes.answer_collections import _closed_collection_groups
from backend.agent.nodes.answer_collections import _collection_facet_view
from backend.agent.nodes.answer_field_catalog import _SCALAR_TARGET_FACET_FIELD_PRIORITY
from backend.agent.nodes.answer_field_catalog import _SCALAR_TARGET_FACET_LABELS
from backend.agent.nodes.answer_field_catalog import _TARGET_COMPLETENESS_FIELD_KEYS
from backend.agent.nodes.answer_payload import _flatten_target_values
from backend.agent.nodes.answer_text import _compact_semantic_text
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple
import re


def _target_value_is_present(answer_text: str, value: str) -> bool:
    normalized_answer = _compact_semantic_text(answer_text)
    normalized_value = _compact_semantic_text(value)
    if not normalized_value:
        return False
    if normalized_value in normalized_answer:
        return True
    # A model may make a small grammatical edit to a long Chinese field.  A
    # high character-bigram overlap recognizes that paraphrase without letting
    # a generic shared word such as “模块” satisfy the field.
    if len(normalized_value) < 8 or len(normalized_answer) < 8:
        return False
    value_pairs = {
        normalized_value[index : index + 2]
        for index in range(len(normalized_value) - 1)
    }
    answer_pairs = {
        normalized_answer[index : index + 2]
        for index in range(len(normalized_answer) - 1)
    }
    return bool(value_pairs) and len(value_pairs & answer_pairs) / len(value_pairs) >= 0.65


def _closed_collection_members_present(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    # ``manifest`` here is the filtered group view; ``pool`` is the complete
    # evidence manifest that may hold the rich FMECA sibling of a sparse
    # architecture-side collection row.
    bridge_pool = pool if pool is not None else manifest
    groups = _closed_collection_groups(manifest)
    if not groups:
        return False
    # The evidence window reserves one current closed group.  Do not merge
    # separate documents/versions into a synthetic inventory.
    group = groups[0]
    answer_text = str(answer or "")
    expected_count = len(group)
    member_type = ""
    for index in group:
        item = manifest[index]
        fields, _view_citation = _collection_facet_view(item, bridge_pool)
        member_type = member_type or str(fields.get("collection_member_kind") or "")
        citation = int(item.get("citation_id") or 0)
        cited_lines = [
            line
            for line in answer_text.splitlines()
            if citation > 0 and f"[^{citation}]" in line
        ]
        identities = [
            value
            for key in (
                "collection_member_name",
                "collection_member_id",
                "failure_mode",
                "record_code",
                "component_name",
                "component_id",
                "abbreviation",
            )
            for value in _flatten_target_values(fields.get(key))
        ]
        if not cited_lines or not identities:
            return False
        cited_text = "\n".join(cited_lines)
        if not any(
            _target_value_is_present(cited_text, value) for value in identities
        ):
            return False
        if member_type == "failure_mode":
            # FMECA 故障模式记录的内容是字段（原因/影响/严酷度/检测），
            # 不只是标题。仅凭名字出现就认定“已列全”，会让纯标题清单压制
            # 带原文原因与影响的权威清单块，用户拿到的正是“内容没给全”。
            cause = str(fields.get("fault_cause") or "").strip()
            if cause and not _target_value_is_present(cited_text, cause):
                return False
    declared_counts = _architecture_declared_member_counts(answer_text, member_type)
    return not declared_counts or all(value == expected_count for value in declared_counts)


def _record_identity_present(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
) -> bool:
    answer_text = str(answer or "")
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
        ("hmc_code", "fault_symptom"),
    )
    for item in manifest:
        fields = item.get("answer_fields") or {}
        citation = int(item.get("citation_id") or 0)
        cited_lines = [
            line
            for line in answer_text.splitlines()
            if citation > 0 and f"[^{citation}]" in line
        ]
        if not cited_lines:
            continue
        cited_text = "\n".join(cited_lines)
        for identifier_key, name_key in identity_pairs:
            identifier = str(fields.get(identifier_key) or "").strip()
            name = str(fields.get(name_key) or "").strip()
            if not identifier or not name:
                continue
            if _compact_semantic_text(identifier) == _compact_semantic_text(name):
                continue
            if not (
                _target_value_is_present(cited_text, identifier)
                and _target_value_is_present(cited_text, name)
            ):
                continue
            module_name = str(fields.get("module_name") or "").strip()
            if module_name and not _target_value_is_present(cited_text, module_name):
                continue
            return True
    return False


def _target_scalar_values(
    facet: str, fields: Mapping[str, Any]
) -> List[str]:
    """Return one source field's public scalar representations.

    A FMECA row can preserve both an original Roman numeral and a normalized
    value.  Either representation may appear in the model-organized answer,
    so the completeness check accepts any explicit representation from that
    same approved row.
    """

    values: List[str] = []
    for key in _SCALAR_TARGET_FACET_FIELD_PRIORITY.get(facet, ()):
        for value in _flatten_target_values(fields.get(key)):
            if value not in values:
                values.append(value)
    return values


def _target_scalar_item_present(
    answer: str,
    facet: str,
    item: Mapping[str, Any],
    *,
    require_record_identity: bool,
) -> bool:
    citation = int(item.get("citation_id") or 0)
    if citation <= 0:
        return False
    fields = (
        item.get("answer_fields")
        if isinstance(item.get("answer_fields"), Mapping)
        else {}
    )
    values = _target_scalar_values(facet, fields)
    if not values:
        return False
    marker = f"[^{citation}]"
    identities = [
        str(item.get("record_id") or "").strip(),
        str(item.get("record_name") or "").strip(),
        *(
            value
            for key in ("failure_mode", "record_code", "failure_mode_id")
            for value in _flatten_target_values(fields.get(key))
        ),
    ]
    identities = [value for value in identities if value]
    for line in str(answer or "").splitlines():
        if marker not in line:
            continue
        if not any(_target_value_is_present(line, value) for value in values):
            continue
        if not require_record_identity:
            return True
        citation_markers = re.findall(r"\[\^(\d+)\]", line)
        if len(set(citation_markers)) == 1:
            return True
        if any(_target_value_is_present(line, value) for value in identities):
            return True
    return False


def _target_facet_is_present(
    answer: str,
    facet: str,
    manifest: Sequence[Mapping[str, Any]],
) -> bool:
    answer_text = str(answer or "")
    role_items = [
        item for item in manifest if facet in set(item.get("supports") or [])
    ]
    if facet in _SCALAR_TARGET_FACET_LABELS:
        # Every distinct approved business record in the current answer
        # window must expose the requested scalar.  This prevents one mode's
        # ``III`` from hiding an omitted second candidate.  When records have
        # different values, a shared sentence must also name each record so
        # ``II / III [^1][^2]`` cannot bind ambiguously.
        grouped: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for item in role_items:
            key = (
                str(item.get("doc_type") or ""),
                str(item.get("record_id") or item.get("citation_id") or ""),
            )
            grouped.setdefault(key, item)
        scalar_items = list(grouped.values())
        signatures = {
            tuple(_target_scalar_values(facet, item.get("answer_fields") or {}))
            for item in scalar_items
        }
        require_identity = len(signatures) > 1
        return bool(scalar_items) and all(
            _target_scalar_item_present(
                answer_text,
                facet,
                item,
                require_record_identity=require_identity,
            )
            for item in scalar_items
        )
    if facet == "record_identity":
        return _record_identity_present(answer_text, role_items)
    if facet == "closed_collection_members":
        return _closed_collection_members_present(answer_text, role_items, manifest)
    if facet in {
        "architecture_receivers",
        "architecture_receiver_context",
        "architecture_responsible_component",
        "architecture_responsibility_function",
        "architecture_downstream",
    }:
        if not role_items:
            return False

        def role_item_present(item: Mapping[str, Any]) -> bool:
            citation = int(item.get("citation_id") or 0)
            fields = (
                item.get("answer_fields")
                if isinstance(item.get("answer_fields"), Mapping)
                else {}
            )
            cited_lines = [
                line
                for line in answer_text.splitlines()
                if citation > 0 and f"[^{citation}]" in line
            ]
            identity_values = [
                value
                for key in (
                    "component_name",
                    "abbreviation",
                    "component_id",
                    "module_name",
                    "function_name",
                    "function_id",
                    "target_name",
                    "target_entity",
                    "target_component_id",
                )
                for value in _flatten_target_values(fields.get(key))
            ]
            if not cited_lines or not identity_values:
                return False
            cited_text = "\n".join(cited_lines)
            if not any(
                _target_value_is_present(cited_text, value)
                for value in identity_values
            ):
                return False
            # Whether natural prose actually expresses “receives”, “owns” or
            # “is downstream of” is an open semantic judgement.  The
            # deterministic layer verifies only that the approved role record
            # is visibly identified on its own citation.  Response obligations
            # retain the semantic-review status instead of requiring one of a
            # handful of preferred Chinese verbs.
            return True

        present = [role_item_present(item) for item in role_items]
        if facet in {"architecture_receivers", "architecture_receiver_context"}:
            return all(present)
        return any(present)

    for item in manifest:
        if facet not in set(item.get("supports") or []):
            continue
        citation = int(item.get("citation_id") or 0)
        fields = (
            item.get("answer_fields")
            if isinstance(item.get("answer_fields"), Mapping)
            else {}
        )
        cited_lines = [
            line
            for line in answer_text.splitlines()
            if citation > 0 and f"[^{citation}]" in line
        ]
        if facet == "hmc_identity":
            codes = _flatten_target_values(fields.get("hmc_code"))
            if codes and any(
                _target_value_is_present(answer_text, value) for value in codes
            ):
                return True
            if codes:
                continue
            components = [
                value
                for key in ("lcn", "hamc", "hsc")
                for value in _flatten_target_values(fields.get(key))
                if value not in {"*", "**"}
            ]
            if components and all(
                _target_value_is_present(answer_text, value) for value in components
            ):
                return True
            continue
        if not cited_lines:
            continue
        cited_text = "\n".join(cited_lines)
        if facet == "maintenance_api":
            required_values = [
                value
                for key in ("api_name", "api_method", "api_usage")
                for value in _flatten_target_values(fields.get(key))
            ]
            if required_values and all(
                _target_value_is_present(cited_text, value)
                for value in required_values
            ):
                return True
            if required_values:
                continue
            report_values = [
                value
                for key in ("api_reportable", "api_reportable_raw")
                for value in _flatten_target_values(fields.get(key))
            ]
            if any(
                _target_value_is_present(cited_text, value)
                for value in report_values
            ):
                return True
            continue
        if facet == "relation_path":
            summaries = _flatten_target_values(fields.get("path_summary"))
            if summaries and any(
                _target_value_is_present(cited_text, value) for value in summaries
            ):
                return True
            if summaries:
                continue
            sources = [
                value
                for key in ("source_entity", "source_name")
                for value in _flatten_target_values(fields.get(key))
            ]
            targets = [
                value
                for key in ("target_entity", "target_name")
                for value in _flatten_target_values(fields.get(key))
            ]
            relations = [
                value
                for key in ("relation", "relation_type")
                for value in _flatten_target_values(fields.get(key))
            ]
            if (
                sources
                and targets
                and any(_target_value_is_present(cited_text, value) for value in sources)
                and any(_target_value_is_present(cited_text, value) for value in targets)
                and (
                    not relations
                    or any(
                        _target_value_is_present(cited_text, value)
                        for value in relations
                    )
                )
            ):
                return True
            continue
        if facet in {"rbd_redundancy_structure", "rbd_missing_parameters"}:
            values = [
                value
                for key, raw in fields.items()
                if key in _TARGET_COMPLETENESS_FIELD_KEYS[facet]
                for value in _flatten_target_values(raw)
            ]
            if values and all(
                _target_value_is_present(cited_text, value) for value in values
            ):
                return True
            continue
        values = [
            value
            for key, raw in fields.items()
            if key in _TARGET_COMPLETENESS_FIELD_KEYS[facet]
            for value in _flatten_target_values(raw)
        ]
        if facet == "maintenance_locator":
            normalized_cited = _compact_semantic_text(cited_text)
            value_present = any(
                _compact_semantic_text(value) in normalized_cited
                for value in values
                if _compact_semantic_text(value)
            )
        else:
            value_present = any(
                _target_value_is_present(cited_text, value) for value in values
            )
        if value_present:
            return True
    return False
