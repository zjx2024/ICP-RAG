"""Deterministic Function Locator extraction over :mod:`document_ir`.

Only explicit tables, labelled fields and arrow expressions become structured
drafts. Embedded images are returned as unconfirmed evidence descriptors; this
module never calls a vision model and never promotes visual output to a
PostgreSQL-authoritative function or relationship.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.core.parsers.document_ir import (
    DocumentIR,
    ParagraphIR,
    TableIR,
    clean_text,
    load_document_ir,
)
from backend.core.parsers.records import NormalizationIssue, RecordDraft, SourceRef
from backend.core.parsers.table_engine import (
    TableRecord,
    TableSchema,
    TableSource,
    extract_records,
    normalize_header,
)


NULL_MARKERS = {"", "-", "—", "–", "N/A", "NA", "不适用", "无"}


SYSTEM_SCHEMA = TableSchema.from_mapping(
    "function_system",
    {
        "component_id": ("系统ID", "系统编号", "系统代码", "SYSTEMID"),
        "abbreviation": ("系统缩写", "系统简称", "缩写", "简称"),
        "name": ("系统名称", "系统全称", "SYSTEMNAME"),
        "description": ("系统说明", "系统描述", "主要职责", "职责"),
    },
    required_fields=("name",),
    identity_groups=(("component_id",), ("name",)),
    min_mapped_fields=2,
    continuation_fields=("description",),
)

SUBSYSTEM_SCHEMA = TableSchema.from_mapping(
    "function_subsystem",
    {
        "component_id": ("子系统ID", "子系统编号", "子系统代码", "SUBSYSTEMID"),
        "parent_system_ref": ("所属系统ID", "所属系统", "系统ID", "系统名称"),
        "abbreviation": ("子系统缩写", "子系统简称", "缩写", "简称"),
        "name": ("子系统名称", "名称", "子系统"),
        "description": ("主要职责", "子系统职责", "职责", "说明", "描述"),
        "contained_modules": ("包含模块", "模块组成", "组成模块", "下属模块"),
    },
    required_fields=("name",),
    identity_groups=(("component_id",), ("abbreviation", "name"), ("name",)),
    min_mapped_fields=3,
    continuation_fields=("description", "contained_modules"),
)

MODULE_SCHEMA = TableSchema.from_mapping(
    "function_module",
    {
        "subsystem_ref": ("所属子系统ID", "所属子系统", "子系统ID", "子系统名称", "子系统"),
        "component_id": ("模块ID", "模块编号", "模块代码", "部件ID", "MODULEID"),
        "abbreviation": ("模块缩写", "模块简称", "缩写", "简称"),
        "name": ("模块名称", "部件名称", "模块", "部件"),
        "description": ("主要职责", "模块职责", "功能说明", "说明", "描述"),
    },
    required_fields=("name",),
    identity_groups=(("component_id",), ("abbreviation", "name"), ("name",)),
    min_mapped_fields=3,
    continuation_fields=("description",),
)

FUNCTION_SCHEMA = TableSchema.from_mapping(
    "function_record",
    {
        "function_id": ("功能ID", "功能编号", "功能代码", "功能标识", "FUNCID", "FUNCTIONID"),
        "parent_function_id": ("父功能ID", "上级功能ID", "父级功能", "上级功能"),
        "function_name": ("功能名称", "功能名", "叶级功能", "FUNCTIONNAME"),
        "description": ("功能描述", "功能说明", "主要职责", "说明", "描述"),
        "system_ref": ("所属系统ID", "所属系统", "系统名称", "系统"),
        "subsystem_ref": ("所属子系统ID", "所属子系统", "子系统名称", "子系统"),
        "module_id": ("模块ID", "实现模块ID", "物理模块ID", "MODULEID"),
        "module_ref": ("主要实现模块", "实现模块", "所属模块", "模块名称", "物理模块", "模块"),
        "module_abbreviation": ("模块缩写", "模块简称"),
        "aliases": ("功能别名", "别名", "简称", "同义词", "ALIASES"),
        "inputs": ("输入", "功能输入", "输入信号", "INPUTS"),
        "outputs": ("输出", "功能输出", "输出信号", "OUTPUTS"),
        "failure_symptoms": (
            "典型用户表达",
            "功能失效表现",
            "典型失效表现",
            "典型现象",
            "异常现象",
            "故障现象",
        ),
    },
    required_fields=("function_name",),
    identity_groups=(("function_id",), ("function_name",)),
    min_mapped_fields=3,
    continuation_fields=("description", "failure_symptoms", "inputs", "outputs"),
)

INTERFACE_SCHEMA = TableSchema.from_mapping(
    "function_interface",
    {
        "interface_id": ("接口ID", "接口编号", "关系ID", "数据流ID", "INTERFACEID"),
        "source_ref": (
            "源组件ID",
            "源模块ID",
            "发送方ID",
            "源组件",
            "源模块",
            "发送方",
            "起点",
            "FROM",
        ),
        "target_ref": (
            "目标组件ID",
            "目标模块ID",
            "接收方ID",
            "目标组件",
            "目标模块",
            "接收方",
            "终点",
            "TO",
        ),
        "relation_type": ("关系类型", "接口类型", "数据流类型", "连接类型"),
        "signal_name": ("信号名称", "数据名称", "接口名称", "信息流", "信号", "数据"),
        "direction": ("方向", "数据方向", "信号方向"),
        "protocol": ("协议", "接口协议", "通信协议"),
        "description": ("接口说明", "关系说明", "说明", "描述"),
    },
    required_fields=("source_ref", "target_ref"),
    identity_groups=(("interface_id",), ("source_ref", "target_ref")),
    min_mapped_fields=3,
    continuation_fields=("description",),
)


FUNCTION_SCHEMAS: Mapping[str, TableSchema] = {
    "system": SYSTEM_SCHEMA,
    "subsystem": SUBSYSTEM_SCHEMA,
    "module": MODULE_SCHEMA,
    "function": FUNCTION_SCHEMA,
    "interface": INTERFACE_SCHEMA,
}

_COMPONENT_ROLE_PRIORITY = {
    "system_table": 40,
    "subsystem_table": 40,
    "module_table": 40,
    "document_scope": 30,
    "function_detail_module_reference": 20,
    "subsystem_module_declaration": 10,
}


@dataclass
class FunctionExtractionResult:
    components: List[RecordDraft] = field(default_factory=list)
    functions: List[RecordDraft] = field(default_factory=list)
    interfaces: List[RecordDraft] = field(default_factory=list)
    image_evidence: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[NormalizationIssue] = field(default_factory=list)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "systems": sum(
                item.values.get("component_type") == "system" for item in self.components
            ),
            "subsystems": sum(
                item.values.get("component_type") == "subsystem" for item in self.components
            ),
            "modules": sum(
                item.values.get("component_type") == "module" for item in self.components
            ),
            "functions": len(self.functions),
            "functions_with_failure_symptoms": sum(
                bool(_value(item.values.get("failure_symptoms"))) for item in self.functions
            ),
            "interfaces": len(self.interfaces),
            "image_evidence": len(self.image_evidence),
            "invalid_records": sum(
                not item.valid
                for item in [*self.components, *self.functions, *self.interfaces]
            ),
        }

    def to_rows(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "components": [item.to_row() for item in self.components],
            "functions": [item.to_row() for item in self.functions],
            "interfaces": [item.to_row() for item in self.interfaces],
            "image_evidence": list(self.image_evidence),
        }


def _value(value: object) -> str:
    text = clean_text(value)
    return "" if text.upper() in NULL_MARKERS else text


def _canonical_id(value: object) -> str:
    text = _value(value).replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", "", text).upper()


def _semantic_id(prefix: str, *parts: object) -> str:
    identity = "|".join(normalize_header(_value(part)) for part in parts if _value(part))
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-SRC-{digest}"


def _split_values(value: object) -> List[str]:
    result: List[str] = []
    seen = set()
    for part in re.split(r"[、,，;；|\n]+", _value(value)):
        item = clean_text(part)
        key = normalize_header(item)
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _source_record(record: TableRecord, role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "raw_values": dict(record.raw_data.get("columns") or {}),
        # Field-oriented raw values retain line breaks that ``CellIR.text``
        # intentionally collapses for table matching.  ``包含模块`` needs the
        # original boundaries so several declarations in one cell do not turn
        # into a single synthetic module name.
        "field_raw_values": dict(record.raw_data.get("field_raw_values") or {}),
        "header_labels": list(record.raw_data.get("header_labels") or []),
        "physical_row": record.raw_data.get("physical_row"),
        "source_refs": [str(source) for source in record.source_refs],
        "continuation_rows": list(record.raw_data.get("_continuation_rows") or []),
    }


def _table_draft(record: TableRecord, record_type: str, role: str) -> RecordDraft:
    source = _source_record(record, role)
    return RecordDraft(
        record_type=record_type,
        values={key: _value(value) for key, value in record.values.items()},
        raw_data={
            "raw_values": dict(source["raw_values"]),
            "source_records": [source],
            "table_role": role,
        },
        source_refs=list(record.source_refs),
        issues=list(record.issues),
    )


def _append_source_records(target: RecordDraft, incoming: RecordDraft) -> None:
    target.extend_sources(incoming.source_refs)
    target.issues.extend(incoming.issues)
    records = target.raw_data.setdefault("source_records", [])
    records.extend(incoming.raw_data.get("source_records") or [])


def _merge_simple(drafts: Iterable[RecordDraft], key_field: str) -> List[RecordDraft]:
    merged: Dict[str, RecordDraft] = {}
    for draft in drafts:
        key = _canonical_id(draft.values.get(key_field))
        if not key:
            continue
        if key not in merged:
            merged[key] = draft
            continue
        target = merged[key]
        for field_name, incoming in draft.values.items():
            if incoming in (None, "", []):
                continue
            current = target.values.get(field_name)
            if current in (None, "", []):
                target.values[field_name] = incoming
                continue
            if isinstance(current, list) or isinstance(incoming, list):
                values = list(current if isinstance(current, list) else [current])
                for item in incoming if isinstance(incoming, list) else [incoming]:
                    if normalize_header(item) not in {normalize_header(value) for value in values}:
                        values.append(item)
                target.values[field_name] = values
                continue
            if normalize_header(current) != normalize_header(incoming):
                conflicts = target.raw_data.setdefault("conflicting_values", {})
                conflicts.setdefault(field_name, [current])
                if incoming not in conflicts[field_name]:
                    conflicts[field_name].append(incoming)
                target.add_issue(
                    "conflicting_explicit_values",
                    f"同一业务ID的字段 {field_name} 存在不同原值，保留首个结构化值并保存全部证据",
                    field=field_name,
                )
        _append_source_records(target, draft)
    return list(merged.values())


def _component_merge_order(drafts: Iterable[RecordDraft]) -> List[RecordDraft]:
    return sorted(
        drafts,
        key=lambda item: -_COMPONENT_ROLE_PRIORITY.get(
            str(item.raw_data.get("table_role") or ""), 0
        ),
    )


def _normalize_component(draft: RecordDraft, component_type: str) -> RecordDraft:
    component_id = _canonical_id(draft.values.get("component_id"))
    name = _value(draft.values.get("name"))
    abbreviation = _value(draft.values.get("abbreviation"))
    if not component_id and name:
        component_id = _semantic_id(
            {"system": "SYS", "subsystem": "SUB", "module": "MOD"}[component_type],
            abbreviation,
            name,
        )
        draft.add_issue(
            "generated_source_local_component_id",
            "原文未提供组件业务ID，已生成文档内稳定ID；不得据此跨文档自动合并",
            field="component_id",
        )
    draft.values.update(
        {
            "component_id": component_id,
            "component_type": component_type,
            "name": name,
            "abbreviation": abbreviation,
            "description": _value(draft.values.get("description")),
            "parent_component_id": _canonical_id(
                draft.values.get("parent_component_id")
            ),
        }
    )
    draft.record_key = component_id
    if not component_id:
        draft.add_issue(
            "missing_component_id", "组件缺少可用标识", field="component_id", severity="error"
        )
    if not name:
        draft.add_issue(
            "missing_component_name", "组件缺少名称", field="name", severity="error"
        )
    return draft


_SYSTEM_META_FIELDS = {
    normalize_header(label): field_name
    for field_name, labels in {
        "component_id": ("系统ID", "系统编号", "系统代码"),
        "name": ("系统名称", "系统全称"),
        "applicable_system": ("适用系统", "适用产品系统"),
        "description": ("系统说明", "系统描述"),
    }.items()
    for label in labels
}
_SYSTEM_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]{2,80})(?:\s+|[：:])?(.*)$")


def _system_from_value(value: str) -> Tuple[str, str]:
    cleaned = _value(value)
    match = _SYSTEM_VALUE_RE.match(cleaned)
    if not match:
        return "", cleaned
    component_id = _canonical_id(match.group(1))
    name = _value(match.group(2)) or component_id
    return component_id, name


def _metadata_system_drafts(document: DocumentIR) -> List[RecordDraft]:
    drafts: List[RecordDraft] = []
    for table in document.tables:
        pending: Dict[str, str] = {}
        pending_sources: List[SourceRef] = []
        pending_raw: Dict[str, str] = {}
        for row in table.rows:
            values = [cell.text for cell in row]
            nonempty = [(index, _value(value)) for index, value in enumerate(values) if _value(value)]
            if len(nonempty) != 2:
                continue
            label_index, label = nonempty[0]
            _value_index, raw_value = nonempty[1]
            field_name = _SYSTEM_META_FIELDS.get(normalize_header(label))
            if not field_name:
                continue
            source = row[label_index].source
            if field_name == "applicable_system":
                component_id, name = _system_from_value(raw_value)
                draft = RecordDraft(
                    record_type="function_component",
                    values={"component_id": component_id, "name": name},
                    raw_data={
                        "raw_values": {label: raw_value},
                        "source_records": [
                            {
                                "role": "document_scope",
                                "raw_values": {label: raw_value},
                                "source_refs": [str(source)],
                            }
                        ],
                        "table_role": "document_scope",
                    },
                    source_refs=[source],
                )
                drafts.append(_normalize_component(draft, "system"))
                continue
            pending[field_name] = raw_value
            pending_sources.append(source)
            pending_raw[label] = raw_value
        if pending:
            draft = RecordDraft(
                record_type="function_component",
                values=pending,
                raw_data={
                    "raw_values": pending_raw,
                    "source_records": [
                        {
                            "role": "document_scope",
                            "raw_values": pending_raw,
                            "source_refs": [str(source) for source in pending_sources],
                        }
                    ],
                    "table_role": "document_scope",
                },
                source_refs=pending_sources,
            )
            drafts.append(_normalize_component(draft, "system"))
    return drafts


def _alias_index(drafts: Sequence[RecordDraft], *, function: bool = False) -> Dict[str, set[str]]:
    result: Dict[str, set[str]] = {}
    for draft in drafts:
        id_field = "function_id" if function else "component_id"
        name_field = "function_name" if function else "name"
        entity_id = _canonical_id(draft.values.get(id_field))
        identity_parts = [
            _value(entity_id),
            _value(draft.values.get(name_field)),
        ]
        if function:
            identity_parts.extend(
                _value(alias) for alias in draft.values.get("aliases") or []
            )
        else:
            identity_parts.append(_value(draft.values.get("abbreviation")))

        # References in engineering tables commonly repeat two explicit
        # identity fields, for example ``MOD-PWR-ICU / 输入调理模块`` or
        # ``输入调理模块（ICU）``.  Register those complete combinations rather
        # than recovering them later with substring matching.  A normalized
        # combination is accepted only when the entire reference matches it;
        # collisions remain ambiguous because the index stores a set of IDs.
        parts = list(dict.fromkeys(part for part in identity_parts if part))
        aliases: List[object] = list(parts)
        for left in parts:
            for right in parts:
                if left != right:
                    aliases.append(f"{left} {right}")
        if len(parts) == 3:
            for first in parts:
                for second in parts:
                    for third in parts:
                        if len({first, second, third}) == 3:
                            aliases.append(f"{first} {second} {third}")
        for alias in aliases:
            normalized = normalize_header(_value(alias))
            if normalized:
                result.setdefault(normalized, set()).add(entity_id)
    return result


def _resolve_reference(reference: object, aliases: Mapping[str, set[str]]) -> Tuple[str, str]:
    normalized = normalize_header(_value(reference))
    if not normalized:
        return "", "missing"
    direct = set(aliases.get(normalized) or ())
    if len(direct) == 1:
        return next(iter(direct)), "exact"
    # This resolver persists parent/implementation/interface relationships.
    # A short alias merely appearing inside descriptive prose is not evidence
    # for such an edge.  Search recall belongs in the retrieval layer; here we
    # accept only an exact ID/name/abbreviation or an explicitly registered
    # complete identity combination.
    return "", "ambiguous" if direct else "unresolved"


_MODULE_ABBREVIATION_PATTERN = r"[A-Za-z][A-Za-z0-9_.-]{1,30}"
_DECLARED_MODULE_ABBREVIATION_FIRST_RE = re.compile(
    rf"^\s*(?P<abbreviation>{_MODULE_ABBREVIATION_PATTERN})"
    r"\s*[（(]\s*(?P<name>[^()（）]{1,120}?)\s*[）)]\s*$"
)
_DECLARED_MODULE_NAME_FIRST_RE = re.compile(
    rf"^\s*(?P<name>[^()（）]{{1,120}}?)\s*[（(]\s*"
    rf"(?P<abbreviation>{_MODULE_ABBREVIATION_PATTERN})\s*[）)]\s*$"
)
_MODULE_ABBREVIATION_RE = re.compile(rf"^{_MODULE_ABBREVIATION_PATTERN}$")


def _is_likely_module_abbreviation(value: str) -> bool:
    """Return whether a whitespace token can deterministically delimit a pair.

    Parentheses make the role explicit and accept the full controlled token
    grammar.  With whitespace alone, requiring upper-case ASCII avoids
    treating every word in an English module name as a new abbreviation.
    """

    text = value.strip()
    return bool(_MODULE_ABBREVIATION_RE.fullmatch(text) and text == text.upper())


def _space_declared_modules(item: str) -> List[Tuple[str, str, str]]:
    """Parse one or more unparenthesized ``abbreviation/name`` pairs.

    Repeated pairs are accepted only when their direction is unambiguous: all
    abbreviations lead their names or all abbreviations trail their names.
    Ambiguous prose remains one source-local name rather than creating edges.
    """

    words = item.split()
    abbreviation_positions = [
        index
        for index, word in enumerate(words)
        if _is_likely_module_abbreviation(word)
    ]
    if not abbreviation_positions:
        return []

    if abbreviation_positions[0] == 0:
        parsed: List[Tuple[str, str, str]] = []
        for offset, position in enumerate(abbreviation_positions):
            next_position = (
                abbreviation_positions[offset + 1]
                if offset + 1 < len(abbreviation_positions)
                else len(words)
            )
            name = _value(" ".join(words[position + 1 : next_position]))
            if not name:
                return []
            raw_item = _value(" ".join(words[position:next_position]))
            parsed.append((words[position], name, raw_item))
        return parsed

    if abbreviation_positions[-1] == len(words) - 1:
        parsed = []
        start = 0
        for position in abbreviation_positions:
            name = _value(" ".join(words[start:position]))
            if not name:
                return []
            raw_item = _value(" ".join(words[start : position + 1]))
            parsed.append((words[position], name, raw_item))
            start = position + 1
        return parsed if start == len(words) else []

    return []


def _declared_modules(value: object) -> List[Tuple[str, str, str]]:
    result: List[Tuple[str, str, str]] = []
    raw_value = "" if value is None else str(value)
    raw_value = raw_value.replace("\r\n", "\n").replace("\r", "\n")
    # Parenthesized declarations are self-delimiting.  Split adjacent pairs
    # even when a source cell used only whitespace (or no punctuation) between
    # them; raw newlines are retained by ``field_raw_values`` above.
    raw_value = re.sub(r"(?<=[)）])(?=\s*\S)", "\n", raw_value)
    for raw_item in re.split(r"[、,，;；|\n]+", raw_value):
        item = _value(raw_item)
        if not item:
            continue
        match = _DECLARED_MODULE_ABBREVIATION_FIRST_RE.match(item)
        if match:
            result.append(
                (
                    _value(match.group("abbreviation")),
                    _value(match.group("name")),
                    item,
                )
            )
            continue
        match = _DECLARED_MODULE_NAME_FIRST_RE.match(item)
        if match:
            result.append(
                (
                    _value(match.group("abbreviation")),
                    _value(match.group("name")),
                    item,
                )
            )
            continue
        parsed = _space_declared_modules(item)
        result.extend(parsed or [("", item, item)])
    return result


def _add_declared_modules(
    subsystems: Sequence[RecordDraft], modules: List[RecordDraft]
) -> List[RecordDraft]:
    """Merge explicit ``包含模块`` declarations without inventing hierarchy.

    A subsystem table can be the only source that states a module's parent.
    When its declared abbreviation/name resolves to exactly one already parsed
    module, that declaration is therefore allowed to fill an otherwise empty
    ``parent_component_id``.  It never overwrites a different parent supplied
    by a module table: both sources remain traceable and the conflict is kept
    on the module record for later review.
    """

    module_aliases = _alias_index(modules)
    by_id = {_canonical_id(item.values.get("component_id")): item for item in modules}
    generated: List[RecordDraft] = []
    for subsystem in subsystems:
        subsystem_id = _canonical_id(subsystem.values.get("component_id"))
        declaration = subsystem.values.get("contained_modules")
        raw_declarations = [
            (source.get("field_raw_values") or {}).get("contained_modules")
            for source in subsystem.raw_data.get("source_records") or []
        ]
        # Prefer the raw cell whose normalized content is the selected
        # structured value.  This preserves line boundaries without reviving a
        # conflicting declaration that lost the deterministic merge priority.
        selected_declaration = next(
            (
                raw
                for raw in raw_declarations
                if raw not in (None, "")
                and normalize_header(raw) == normalize_header(declaration)
            ),
            declaration,
        )
        for abbreviation, name, raw_item in _declared_modules(selected_declaration):
            # A structured parent edge may only come from an exact stored ID,
            # abbreviation, full name, or their exact combined form.  The
            # generic resolver also supports substring recall for search-like
            # references; using that here could turn “输入滤波模块” into the
            # unrelated short alias “输入” and persist a false hierarchy.
            exact_ids: set[str] = set()
            for candidate in (
                abbreviation,
                name,
                f"{abbreviation} {name}".strip(),
                raw_item,
            ):
                normalized = normalize_header(_value(candidate))
                if normalized:
                    exact_ids.update(module_aliases.get(normalized) or ())
            module_id = next(iter(exact_ids)) if len(exact_ids) == 1 else ""
            if module_id and module_id in by_id:
                target = by_id[module_id]
                target.extend_sources(subsystem.source_refs)
                target.raw_data.setdefault("source_records", []).append(
                    {
                        "role": "subsystem_module_declaration",
                        "raw_values": {
                            "包含模块": raw_item,
                            "声明父组件ID": subsystem_id,
                        },
                        "source_refs": [str(source) for source in subsystem.source_refs],
                    }
                )
                existing_parent_id = _canonical_id(
                    target.values.get("parent_component_id")
                )
                if subsystem_id and not existing_parent_id:
                    # This is a direct structured declaration, not an inferred
                    # relationship.  It is the authoritative fallback when a
                    # module row omitted its 所属子系统 column/value.
                    target.values["parent_component_id"] = subsystem_id
                elif (
                    subsystem_id
                    and existing_parent_id
                    and existing_parent_id != subsystem_id
                ):
                    conflicts = target.raw_data.setdefault("conflicting_values", {})
                    parents = conflicts.setdefault(
                        "parent_component_id", [existing_parent_id]
                    )
                    if subsystem_id not in parents:
                        parents.append(subsystem_id)
                    target.add_issue(
                        "conflicting_declared_module_parent",
                        (
                            f"模块 {module_id} 已显式归属 {existing_parent_id}，"
                            f"但子系统 {subsystem_id} 的包含模块声明指向该模块；"
                            "保留原父组件并记录冲突"
                        ),
                        field="parent_component_id",
                    )
                continue
            source_refs = list(subsystem.source_refs)
            draft = RecordDraft(
                record_type="function_component",
                values={
                    "component_id": "",
                    "component_type": "module",
                    "parent_component_id": subsystem.values.get("component_id"),
                    "abbreviation": abbreviation,
                    "name": name,
                    "description": "",
                },
                raw_data={
                    "raw_values": {"包含模块": raw_item},
                    "source_records": [
                        {
                            "role": "subsystem_module_declaration",
                            "raw_values": {"包含模块": raw_item},
                            "source_refs": [str(source) for source in source_refs],
                        }
                    ],
                    "table_role": "subsystem_module_declaration",
                },
                source_refs=source_refs,
            )
            generated.append(_normalize_component(draft, "module"))
    return _merge_simple([*modules, *generated], "component_id")


def _function_role(record: TableRecord) -> str:
    fields = set(record.header.mapping)
    if "failure_symptoms" in fields or "module_id" in fields:
        return "function_detail"
    return "function_catalog"


def _header_tokens(record: TableRecord) -> set[str]:
    return {normalize_header(label) for label in record.header.labels if _value(label)}


def _is_subsystem_record(record: TableRecord) -> bool:
    """Reject generic 名称/缩写 matches from module and function tables."""

    fields = set(record.header.mapping)
    labels = _header_tokens(record)
    if fields & {"component_id", "contained_modules", "parent_system_ref"}:
        return True
    if labels & {"子系统名称", "子系统全称"}:
        return True
    if labels & {"模块ID", "模块编号", "模块名称", "功能ID", "功能名称"}:
        return False
    section = normalize_header(" ".join(record.source_refs[0].section_path))
    return "子系统" in section and "模块" not in "".join(labels)


def _is_module_record(record: TableRecord) -> bool:
    """Require a module-specific anchor instead of the generic 模块 substring."""

    fields = set(record.header.mapping)
    labels = _header_tokens(record)
    if "component_id" in fields:
        return True
    if labels & {"模块名称", "模块全称", "部件名称"}:
        return True
    has_parent = bool(labels & {"所属子系统", "所属子系统ID", "子系统名称"})
    return has_parent and bool(fields & {"name", "abbreviation"}) and "包含模块" not in labels


def _normalize_function(draft: RecordDraft) -> RecordDraft:
    function_name = _value(draft.values.get("function_name"))
    function_id = _canonical_id(draft.values.get("function_id"))
    module_ref = _value(draft.values.get("module_ref"))
    if not function_id and function_name:
        function_id = _semantic_id("FUNC", function_name, module_ref)
        draft.add_issue(
            "generated_source_local_function_id",
            "原文未提供功能ID，已生成文档内稳定ID；不得据此跨文档自动合并",
            field="function_id",
        )
    symptoms = _split_values(draft.values.get("failure_symptoms"))
    draft.values.update(
        {
            "function_id": function_id,
            "parent_function_id": _canonical_id(draft.values.get("parent_function_id")),
            "function_name": function_name,
            "description": _value(draft.values.get("description")),
            "module_id": _canonical_id(draft.values.get("module_id")),
            "module_ref": module_ref,
            "module_name": _value(draft.values.get("module_name")),
            "aliases": _split_values(draft.values.get("aliases")),
            "inputs": _split_values(draft.values.get("inputs")),
            "outputs": _split_values(draft.values.get("outputs")),
            "failure_symptoms": "；".join(symptoms),
            "failure_symptom_list": symptoms,
            "system_name": _value(draft.values.get("system_name") or draft.values.get("system_ref")),
            "subsystem_name": _value(
                draft.values.get("subsystem_name") or draft.values.get("subsystem_ref")
            ),
        }
    )
    draft.record_key = function_id
    if not function_id:
        draft.add_issue(
            "missing_function_id", "功能缺少可用标识", field="function_id", severity="error"
        )
    if not function_name:
        draft.add_issue(
            "missing_function_name", "功能缺少名称", field="function_name", severity="error"
        )
    return draft


_FUNCTION_FIELD_PRIORITY = {
    "function_id": {"function_catalog": 40, "function_detail": 40, "labelled_paragraph": 30},
    "parent_function_id": {"function_catalog": 40, "function_detail": 20, "labelled_paragraph": 30},
    "function_name": {"function_catalog": 40, "function_detail": 30, "labelled_paragraph": 30},
    "description": {"function_catalog": 40, "function_detail": 30, "labelled_paragraph": 30},
    "module_id": {"function_catalog": 20, "function_detail": 40, "labelled_paragraph": 30},
    "module_ref": {"function_catalog": 30, "function_detail": 40, "labelled_paragraph": 30},
    "module_name": {"function_catalog": 20, "function_detail": 40, "labelled_paragraph": 30},
    "failure_symptoms": {"function_catalog": 20, "function_detail": 40, "labelled_paragraph": 30},
}


def _merge_functions(drafts: Iterable[RecordDraft]) -> List[RecordDraft]:
    groups: Dict[str, List[RecordDraft]] = {}
    for draft in drafts:
        key = _canonical_id(draft.values.get("function_id"))
        if key:
            groups.setdefault(key, []).append(draft)

    result: List[RecordDraft] = []
    list_fields = {"aliases", "inputs", "outputs", "failure_symptom_list"}
    for function_id, group in groups.items():
        fields = {field_name for draft in group for field_name in draft.values}
        values: Dict[str, Any] = {}
        conflicts: Dict[str, List[Any]] = {}
        for field_name in fields:
            present = [
                (index, draft, draft.values.get(field_name))
                for index, draft in enumerate(group)
                if draft.values.get(field_name) not in (None, "", [])
            ]
            if not present:
                continue
            if field_name in list_fields:
                combined: List[Any] = []
                seen = set()
                for _index, _draft, field_value in present:
                    for item in field_value if isinstance(field_value, list) else [field_value]:
                        normalized = normalize_header(item)
                        if normalized not in seen:
                            combined.append(item)
                            seen.add(normalized)
                values[field_name] = combined
                continue
            if field_name == "failure_symptoms":
                combined = []
                seen = set()
                for _index, _draft, field_value in present:
                    for item in _split_values(field_value):
                        normalized = normalize_header(item)
                        if normalized not in seen:
                            combined.append(item)
                            seen.add(normalized)
                values[field_name] = "；".join(combined)
                continue

            def choice_key(item: Tuple[int, RecordDraft, Any]) -> Tuple[int, int, int]:
                index, draft, field_value = item
                role = str(draft.raw_data.get("table_role") or "")
                priority = _FUNCTION_FIELD_PRIORITY.get(field_name, {}).get(role, 20)
                detail_length = len(_value(field_value)) if field_name == "description" else 0
                return priority, detail_length, -index

            chosen = max(present, key=choice_key)[2]
            values[field_name] = chosen
            distinct: List[Any] = []
            seen: set[str] = set()
            for _index, _draft, field_value in present:
                normalized = normalize_header(field_value)
                if field_name == "module_ref" and any(
                    normalized in existing or existing in normalized for existing in seen
                ):
                    continue
                if normalized not in seen:
                    distinct.append(field_value)
                    seen.add(normalized)
            if len(distinct) > 1:
                conflicts[field_name] = distinct

        source_refs: List[SourceRef] = []
        source_records: List[Dict[str, Any]] = []
        issues: List[NormalizationIssue] = []
        for draft in group:
            known = {str(source) for source in source_refs}
            source_refs.extend(source for source in draft.source_refs if str(source) not in known)
            source_records.extend(draft.raw_data.get("source_records") or [])
            issues.extend(draft.issues)
        merged = RecordDraft(
            record_type="function_record",
            values=values,
            raw_data={
                "raw_values": dict((source_records[0] if source_records else {}).get("raw_values") or {}),
                "source_records": source_records,
            },
            source_refs=source_refs,
            issues=issues,
            record_key=function_id,
        )
        if conflicts:
            merged.raw_data["conflicting_values"] = conflicts
            for field_name in conflicts:
                merged.add_issue(
                    "conflicting_explicit_values",
                    f"同一功能ID的字段 {field_name} 存在不同原值，已按表角色选取权威候选并保留全部证据",
                    field=field_name,
                )
        result.append(merged)
    return result


_PARAGRAPH_FUNCTION_FIELDS = {
    normalize_header(alias): field_name
    for field_name, aliases in {
        "function_id": ("功能ID", "功能编号", "功能代码"),
        "parent_function_id": ("父功能ID", "上级功能ID"),
        "function_name": ("功能名称", "功能名"),
        "description": ("功能描述", "功能说明"),
        "module_id": ("模块ID", "实现模块ID"),
        "module_ref": ("主要实现模块", "实现模块", "所属模块"),
        "aliases": ("功能别名", "别名"),
        "inputs": ("输入", "功能输入"),
        "outputs": ("输出", "功能输出"),
        "failure_symptoms": ("功能失效表现", "典型失效表现", "典型用户表达"),
    }.items()
    for alias in aliases
}
_LABELLED_FIELD_RE = re.compile(r"^([^：:]{1,40})\s*[:：]\s*(.*)$")


def _paragraph_function_drafts(document: DocumentIR) -> List[RecordDraft]:
    drafts: List[RecordDraft] = []
    values: Dict[str, str] = {}
    raw_values: Dict[str, str] = {}
    sources: List[SourceRef] = []

    def flush() -> None:
        nonlocal values, raw_values, sources
        if values.get("function_name") or values.get("function_id"):
            draft = RecordDraft(
                record_type="function_record",
                values=dict(values),
                raw_data={
                    "raw_values": dict(raw_values),
                    "source_records": [
                        {
                            "role": "labelled_paragraph",
                            "raw_values": dict(raw_values),
                            "source_refs": [str(source) for source in sources],
                        }
                    ],
                    "table_role": "labelled_paragraph",
                },
                source_refs=list(sources),
            )
            drafts.append(_normalize_function(draft))
        values, raw_values, sources = {}, {}, []

    for element in document.elements:
        if isinstance(element, TableIR) or (isinstance(element, ParagraphIR) and element.heading_level):
            flush()
            continue
        paragraph: ParagraphIR = element
        matched_any = False
        for part in re.split(r"[\n；;]+", paragraph.raw_text or paragraph.text):
            match = _LABELLED_FIELD_RE.match(clean_text(part))
            if not match:
                continue
            label, field_value = clean_text(match.group(1)), _value(match.group(2))
            field_name = _PARAGRAPH_FUNCTION_FIELDS.get(normalize_header(label))
            if not field_name:
                continue
            if field_name == "function_id" and values.get("function_id"):
                if _canonical_id(values["function_id"]) != _canonical_id(field_value):
                    flush()
            values[field_name] = field_value
            raw_values[label] = field_value
            matched_any = True
        if matched_any:
            sources.append(paragraph.source)
    flush()
    return drafts


_ARROW_SPLIT_RE = re.compile(r"\s*(?:--?>|→|⇒)\s*")
_ARROW_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_.-]{2,80}$")


def _explicit_arrow_interfaces(document: DocumentIR) -> List[RecordDraft]:
    drafts: List[RecordDraft] = []
    for paragraph in document.paragraphs:
        text = _value(paragraph.text)
        if not text or len(text) > 300 or not _ARROW_SPLIT_RE.search(text):
            continue
        expression, separator, description = text.partition("：")
        if not separator:
            expression, separator, description = text.partition(":")
        endpoints = [_value(item) for item in _ARROW_SPLIT_RE.split(expression)]
        if not (2 <= len(endpoints) <= 8) or not all(
            _ARROW_ENDPOINT_RE.fullmatch(item) for item in endpoints
        ):
            continue
        for index, (source_ref, target_ref) in enumerate(zip(endpoints, endpoints[1:]), start=1):
            interface_id = _semantic_id(
                "IF", source_ref, target_ref, description, str(paragraph.source)
            )
            draft = RecordDraft(
                record_type="function_interface",
                values={
                    "interface_id": interface_id,
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "relation_type": "explicit_flow",
                    "signal_name": _value(description),
                    "direction": "source_to_target",
                    "description": _value(description),
                },
                raw_data={
                    "raw_values": {"expression": text},
                    "source_records": [
                        {
                            "role": "explicit_text_expression",
                            "raw_values": {"expression": text, "edge_index": index},
                            "source_refs": [str(paragraph.source)],
                        }
                    ],
                    "table_role": "explicit_text_expression",
                },
                source_refs=[paragraph.source],
                record_key=interface_id,
            )
            draft.add_issue(
                "generated_interface_id",
                "显式箭头关系未提供接口ID，已按端点和来源生成稳定ID",
                field="interface_id",
            )
            drafts.append(draft)
    return drafts


_RELATION_TYPES = {
    "数据流": "data_flow",
    "信息流": "data_flow",
    "信号": "signal_flow",
    "供电": "power_supply",
    "电源": "power_supply",
    "时钟": "clock_supply",
    "状态": "status_flow",
    "控制": "control_flow",
    "连接": "connects_to",
}


def _normalize_interface(draft: RecordDraft) -> RecordDraft:
    source_ref = _value(draft.values.get("source_ref"))
    target_ref = _value(draft.values.get("target_ref"))
    signal_name = _value(draft.values.get("signal_name"))
    relation_raw = _value(draft.values.get("relation_type"))
    relation_type = _RELATION_TYPES.get(normalize_header(relation_raw), relation_raw)
    if not relation_type:
        relation_type = "connects_to"
        draft.add_issue(
            "default_relation_type",
            "接口表明确给出源和目标但未给关系类型，按中性 connects_to 保存",
            field="relation_type",
        )
    interface_id = _canonical_id(draft.values.get("interface_id"))
    if not interface_id and source_ref and target_ref:
        interface_id = _semantic_id("IF", source_ref, target_ref, relation_type, signal_name)
        draft.add_issue(
            "generated_interface_id",
            "接口表未提供接口ID，已按显式端点和信号生成稳定ID",
            field="interface_id",
        )
    draft.values.update(
        {
            "interface_id": interface_id,
            "source_ref": source_ref,
            "target_ref": target_ref,
            "source_component_id": "",
            "target_component_id": "",
            "relation_type": relation_type,
            "signal_name": signal_name,
            "direction": _value(draft.values.get("direction")) or "source_to_target",
            "protocol": _value(draft.values.get("protocol")),
            "description": _value(draft.values.get("description")),
        }
    )
    draft.record_key = interface_id
    if not source_ref or not target_ref:
        draft.add_issue(
            "missing_interface_endpoint",
            "接口关系缺少明确源或目标",
            field="source_ref" if not source_ref else "target_ref",
            severity="error",
        )
    return draft


def _resolve_component_hierarchy(
    systems: List[RecordDraft],
    subsystems: List[RecordDraft],
    modules: List[RecordDraft],
) -> Tuple[List[RecordDraft], List[RecordDraft], List[RecordDraft]]:
    system_aliases = _alias_index(systems)
    default_system_id = systems[0].values.get("component_id") if len(systems) == 1 else ""
    for subsystem in subsystems:
        parent_ref = subsystem.values.get("parent_system_ref")
        parent_id, status = _resolve_reference(parent_ref, system_aliases)
        if not parent_id and not _value(parent_ref):
            parent_id = _canonical_id(default_system_id)
        subsystem.values["parent_component_id"] = parent_id
        if _value(parent_ref) and not parent_id:
            subsystem.add_issue(
                "unresolved_parent_system",
                f"无法唯一解析所属系统: {parent_ref}",
                field="parent_system_ref",
            )
        elif status == "contained":
            subsystem.add_issue(
                "parent_system_alias_match",
                f"所属系统通过规范化别名匹配为 {parent_id}",
                field="parent_system_ref",
            )

    subsystem_aliases = _alias_index(subsystems)
    for module in modules:
        if _canonical_id(module.values.get("parent_component_id")):
            continue
        parent_ref = module.values.get("subsystem_ref")
        parent_id, status = _resolve_reference(parent_ref, subsystem_aliases)
        module.values["parent_component_id"] = parent_id
        if _value(parent_ref) and not parent_id:
            module.add_issue(
                "unresolved_parent_subsystem",
                f"无法唯一解析所属子系统: {parent_ref}",
                field="subsystem_ref",
            )
        elif status == "contained":
            module.add_issue(
                "parent_subsystem_alias_match",
                f"所属子系统通过规范化别名匹配为 {parent_id}",
                field="subsystem_ref",
            )
    return systems, subsystems, modules


def _enrich_functions(
    functions: List[RecordDraft],
    systems: Sequence[RecordDraft],
    subsystems: Sequence[RecordDraft],
    modules: Sequence[RecordDraft],
) -> None:
    module_aliases = _alias_index(modules)
    modules_by_id = {_canonical_id(item.values.get("component_id")): item for item in modules}
    subsystems_by_id = {
        _canonical_id(item.values.get("component_id")): item for item in subsystems
    }
    systems_by_id = {_canonical_id(item.values.get("component_id")): item for item in systems}
    default_system = systems[0] if len(systems) == 1 else None
    function_ids = {_canonical_id(item.values.get("function_id")) for item in functions}

    for function in functions:
        explicit_module_id = _canonical_id(function.values.get("module_id"))
        module_id = explicit_module_id if explicit_module_id in modules_by_id else ""
        reference = (
            explicit_module_id
            or function.values.get("module_ref")
            or function.values.get("module_name")
            or function.values.get("module_abbreviation")
        )
        status = "exact" if module_id else "missing"
        if not module_id:
            module_id, status = _resolve_reference(reference, module_aliases)
        if explicit_module_id and not module_id:
            module_id = explicit_module_id
            function.add_issue(
                "module_id_not_declared",
                f"功能引用的模块ID {explicit_module_id} 未在本文件组件表声明",
                field="module_id",
            )
        elif not module_id and _BUSINESS_ID_RE.fullmatch(_value(reference)):
            module_id = _canonical_id(reference)
            function.add_issue(
                "module_id_not_declared",
                f"功能引用的模块ID {module_id} 未在本文件组件表声明",
                field="module_ref",
            )
        elif _value(reference) and not module_id:
            function.add_issue(
                "unresolved_function_module",
                f"无法唯一解析实现模块: {reference}",
                field="module_ref",
            )
        elif status == "contained":
            function.add_issue(
                "function_module_alias_match",
                f"实现模块通过规范化别名匹配为 {module_id}",
                field="module_ref",
            )
        function.values["module_id"] = module_id

        module = modules_by_id.get(module_id)
        subsystem = None
        system = default_system
        if module is not None:
            function.values["module_name"] = module.values.get("name") or function.values.get(
                "module_name"
            )
            subsystem = subsystems_by_id.get(
                _canonical_id(module.values.get("parent_component_id"))
            )
        if subsystem is not None:
            if not _value(function.values.get("subsystem_name")):
                function.values["subsystem_name"] = subsystem.values.get("name")
            system = systems_by_id.get(
                _canonical_id(subsystem.values.get("parent_component_id"))
            ) or system
        if system is not None and not _value(function.values.get("system_name")):
            function.values["system_name"] = system.values.get("name")

        parent_id = _canonical_id(function.values.get("parent_function_id"))
        if parent_id and parent_id not in function_ids:
            function.add_issue(
                "parent_function_not_declared",
                f"父功能 {parent_id} 未在本文件功能表声明",
                field="parent_function_id",
            )


_BUSINESS_ID_RE = re.compile(
    r"^(?:SYS|SYSTEM|SUB|MOD|MODULE|FUNC|FUNCTION)[-_][A-Z0-9][A-Z0-9_.-]*$",
    re.I,
)


def _resolve_interfaces(
    interfaces: List[RecordDraft],
    components: Sequence[RecordDraft],
    functions: Sequence[RecordDraft],
) -> None:
    aliases = _alias_index(components)
    for alias, entity_ids in _alias_index(functions, function=True).items():
        aliases.setdefault(alias, set()).update(entity_ids)
    for interface in interfaces:
        for side in ("source", "target"):
            reference = interface.values.get(f"{side}_ref")
            entity_id, status = _resolve_reference(reference, aliases)
            if not entity_id and _BUSINESS_ID_RE.fullmatch(_value(reference)):
                entity_id = _canonical_id(reference)
                interface.add_issue(
                    "interface_endpoint_not_declared",
                    f"接口{side}端点 {entity_id} 有显式ID，但未在本文件组件/功能表声明",
                    field=f"{side}_ref",
                )
            elif not entity_id:
                interface.add_issue(
                    "unresolved_interface_endpoint",
                    f"接口{side}端点无法唯一解析: {reference}",
                    field=f"{side}_ref",
                    severity="error",
                )
            elif status == "contained":
                interface.add_issue(
                    "interface_endpoint_alias_match",
                    f"接口{side}端点通过规范化别名匹配为 {entity_id}",
                    field=f"{side}_ref",
                )
            interface.values[f"{side}_component_id"] = entity_id


def _image_evidence(document: DocumentIR) -> List[Dict[str, Any]]:
    return [
        {
            "original_name": image.original_name,
            "content_type": image.content_type,
            "sha256": image.sha256,
            "source_ref": str(image.source),
            "authority": "unconfirmed_extraction",
            "eligible_for_authoritative_records": False,
        }
        for image in document.images
    ]


def extract_function_drafts(
    file_path: str, *, source_name: Optional[str] = None
) -> FunctionExtractionResult:
    """Extract traceable Function Locator drafts without external writes/models."""

    document = load_document_ir(file_path, source_name=source_name)
    system_drafts: List[RecordDraft] = _metadata_system_drafts(document)
    subsystem_drafts: List[RecordDraft] = []
    module_drafts: List[RecordDraft] = []
    function_drafts: List[RecordDraft] = []
    interface_drafts: List[RecordDraft] = []

    for table in document.tables:
        source = TableSource.from_ir(table)
        for record in extract_records(source, SYSTEM_SCHEMA):
            system_drafts.append(
                _normalize_component(
                    _table_draft(record, "function_component", "system_table"), "system"
                )
            )
        for record in extract_records(source, SUBSYSTEM_SCHEMA):
            if not _is_subsystem_record(record):
                continue
            subsystem_drafts.append(
                _normalize_component(
                    _table_draft(record, "function_component", "subsystem_table"),
                    "subsystem",
                )
            )
        for record in extract_records(source, MODULE_SCHEMA):
            if not _is_module_record(record):
                continue
            labels = _header_tokens(record)
            module_role = "module_table"
            module_draft = _table_draft(record, "function_component", module_role)
            if ("功能ID" in labels or "功能编号" in labels) and not (
                labels & {"所属子系统", "所属子系统ID"}
            ):
                # A function/symptom row explicitly identifies the module and
                # is useful for entity reconciliation, but its “功能说明” is not
                # automatically the module's authoritative responsibility.
                module_role = "function_detail_module_reference"
                module_draft.raw_data["table_role"] = module_role
                module_draft.raw_data["source_records"][0]["role"] = module_role
                module_draft.values["description"] = ""
            module_drafts.append(_normalize_component(module_draft, "module"))
        for record in extract_records(source, FUNCTION_SCHEMA):
            function_drafts.append(
                _normalize_function(
                    _table_draft(record, "function_record", _function_role(record))
                )
            )
        for record in extract_records(source, INTERFACE_SCHEMA):
            interface_drafts.append(
                _normalize_interface(
                    _table_draft(record, "function_interface", "interface_table")
                )
            )

    systems = _merge_simple(_component_merge_order(system_drafts), "component_id")
    subsystems = _merge_simple(_component_merge_order(subsystem_drafts), "component_id")
    modules = _merge_simple(_component_merge_order(module_drafts), "component_id")
    systems, subsystems, modules = _resolve_component_hierarchy(
        systems, subsystems, modules
    )
    modules = _add_declared_modules(subsystems, modules)

    function_drafts.extend(_paragraph_function_drafts(document))
    functions = _merge_functions(function_drafts)
    _enrich_functions(functions, systems, subsystems, modules)

    interface_drafts.extend(_explicit_arrow_interfaces(document))
    interfaces = _merge_simple(interface_drafts, "interface_id")
    _resolve_interfaces(interfaces, [*systems, *subsystems, *modules], functions)

    return FunctionExtractionResult(
        components=[*systems, *subsystems, *modules],
        functions=functions,
        interfaces=interfaces,
        image_evidence=_image_evidence(document),
    )


__all__ = [
    "FUNCTION_SCHEMAS",
    "FunctionExtractionResult",
    "extract_function_drafts",
]
