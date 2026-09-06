"""Source-owned collection groups and architectural role views.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_payload import _flatten_target_values
from backend.agent.nodes.answer_text import _compact_semantic_text
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple
import re


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


def _architecture_role_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = item.get("answer_fields")
    if isinstance(fields, Mapping):
        return fields
    return _evidence_answer_fields(item)


def _architecture_role_indices(
    state: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, List[int]]:
    """Bind planner-declared architecture roles to exact structured subjects.

    The target model has already interpreted grammatical role and plurality.
    Answer-time code consumes only its validated ``architecture_roles``
    objects and exact identity columns.  Question verbs, word overlap and
    descriptions are never used to create or widen a role selection.
    """

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    declared_roles = [
        role
        for role in (frame.get("architecture_roles") or [])
        if isinstance(role, Mapping)
        and str(role.get("role") or "").strip()
        and str(role.get("subject") or "").strip()
    ]
    if not declared_roles:
        return {}
    roles: Dict[str, List[int]] = {}
    indexed = [
        (index, item, _architecture_role_fields(item))
        for index, item in enumerate(items)
        if str(item.get("doc_type") or "") == "function_locator"
        and _architecture_role_fields(item)
    ]
    if not indexed:
        return roles

    def identity_values(fields: Mapping[str, Any], *, include_targets: bool) -> set[str]:
        keys = [
            "component_id",
            "component_name",
            "module_id",
            "module_name",
            "function_id",
            "function_name",
            "abbreviation",
        ]
        if include_targets:
            keys.extend(("target_component_id", "target_entity", "target_name"))
        return {
            _compact_semantic_text(value).casefold()
            for key in keys
            for value in _flatten_target_values(fields.get(key))
            if _compact_semantic_text(value)
        }

    def exact_matches(subject: str, *, include_targets: bool = False) -> List[int]:
        key = _compact_semantic_text(subject).casefold()
        if not key:
            return []
        return [
            index
            for index, _item, fields in indexed
            if key in identity_values(fields, include_targets=include_targets)
        ]

    def extend(facet: str, indexes: Sequence[int]) -> None:
        bucket = roles.setdefault(facet, [])
        for index in indexes:
            if index not in bucket:
                bucket.append(index)

    for declared in declared_roles:
        role = str(declared.get("role") or "").strip()
        subject = str(declared.get("subject") or "").strip()
        matched = exact_matches(
            subject,
            include_targets=role in {"downstream", "destination"},
        )
        if not matched:
            continue
        if role == "receiver":
            extend("architecture_receivers", matched)
            parent_ids = {
                str(_architecture_role_fields(items[index]).get("parent_component_id") or "").strip()
                for index in matched
                if str(_architecture_role_fields(items[index]).get("parent_component_id") or "").strip()
            }
            extend(
                "architecture_receiver_context",
                [
                    index
                    for index, _item, fields in indexed
                    if str(fields.get("component_id") or "").strip() in parent_ids
                ],
            )
        elif role in {"responsible_component", "converter", "processor", "validator"}:
            extend(
                "architecture_responsible_component",
                [index for index in matched if _architecture_role_fields(items[index]).get("component_id")],
            )
            extend(
                "architecture_responsibility_function",
                [index for index in matched if _architecture_role_fields(items[index]).get("function_id")],
            )
        elif role in {"downstream", "destination"}:
            extend("architecture_downstream", matched)
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
        rf"(?:总数|数量)\s*(?:是|为|=|：|:)\s*"
        rf"({numeral})\s*(?:个|项|条)?\s*{re.escape(label)}",
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


_CHINESE_COUNT_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _parse_small_count(value: Any) -> int:
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _CHINESE_COUNT_DIGITS.get(left, 1 if not left else -1)
        units = _CHINESE_COUNT_DIGITS.get(right, 0 if not right else -1)
        return tens * 10 + units if tens >= 0 and units >= 0 else -1
    return _CHINESE_COUNT_DIGITS.get(text, -1)


def _collection_facet_view(
    item: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], int]:
    """Bridge a sparse collection-member row to the same record's rich row.

    The closed-collection inventory may be seeded from the architecture side
    (collection_* fields only) while the authoritative FMECA projection of the
    same business code carries causes and impacts.  Only an exact record-ID or
    record-code match authorizes this field bridge; name or vector similarity
    never does.  Returns the fields to render and the citation to bind.
    """

    def own_fields(source: Mapping[str, Any]) -> Mapping[str, Any]:
        fields = source.get("answer_fields")
        return fields if isinstance(fields, Mapping) else {}

    fields = own_fields(item)
    citation = int(item.get("citation_id") or 0)
    if str(fields.get("collection_member_kind") or "") != "failure_mode":
        return fields, citation
    if str(fields.get("fault_cause") or "").strip():
        return fields, citation
    member_code = str(
        fields.get("collection_member_id")
        or fields.get("record_code")
        or item.get("record_id")
        or ""
    ).strip()
    own_record_id = str(item.get("record_id") or "").strip()
    if not member_code:
        return fields, citation
    for other in manifest:
        if other is item:
            continue
        other_fields = own_fields(other)
        other_code = str(other.get("record_id") or other_fields.get("record_code") or "").strip()
        if not other_code or other_code not in {member_code, own_record_id}:
            continue
        if not str(other_fields.get("fault_cause") or "").strip():
            continue
        merged = dict(other_fields)
        merged.update(
            {
                key: value
                for key, value in fields.items()
                if value not in (None, "", [], {})
            }
        )
        other_citation = int(other.get("citation_id") or 0)
        return merged, (other_citation or citation)
    return fields, citation
