"""Deterministic extraction of explicit FTA/RBD structures from document IR.

The engineering picture remains an independent visual candidate.  This module
only consumes relations which the source states explicitly: FTA event/cut-set
tables and RBD structure expressions.  Its output is therefore suitable for a
review revision, but is never marked as confirmed automatically.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.core.parsers.document_ir import DocumentIR, ParagraphIR, TableIR, clean_text
from backend.core.parsers.rbd_expression import RbdExpressionError, rbd_expression_to_graph
from backend.core.parsers.table_engine import TableSchema, TableSource, extract_records


_IDENTITY_RE = re.compile(r"[\s_\-—–/\\:：;；,，。·()（）\[\]【】]+")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:[.．]\d+)*\s*")
_PLAIN_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_TIMES_TEN_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))[×xX*]10\^([+-]?\d+)$"
)
_POWER_TEN_RE = re.compile(r"^10\^([+-]?\d+)$")
_UNIT_SUFFIX_RE = re.compile(
    r"(?:1?/h|h\^-?1|/小时|每小时|小时\^-?1|h|小时)$", re.IGNORECASE
)
_AMBIGUOUS_NUMBER_RE = re.compile(r"[<>≤≥≈~～]|\d\s*(?:至|—|–)\s*\d")


def _identity(value: Any) -> str:
    return _IDENTITY_RE.sub("", clean_text(value).casefold())


def _float(value: Any) -> Optional[float]:
    """Parse an explicitly scalar engineering number, otherwise fail closed."""

    text = clean_text(value)
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("−", "-").replace("﹣", "-")
    normalized = normalized.replace(",", "").replace(" ", "")
    if _AMBIGUOUS_NUMBER_RE.search(normalized):
        return None
    percent = normalized.endswith("%")
    if percent:
        normalized = normalized[:-1]
    normalized = _UNIT_SUFFIX_RE.sub("", normalized)
    try:
        multiplied = _TIMES_TEN_RE.fullmatch(normalized)
        if multiplied:
            result = float(multiplied.group(1)) * (10.0 ** int(multiplied.group(2)))
        elif _POWER_TEN_RE.fullmatch(normalized):
            result = 10.0 ** int(_POWER_TEN_RE.fullmatch(normalized).group(1))
        elif _PLAIN_NUMBER_RE.fullmatch(normalized):
            result = float(normalized)
        else:
            return None
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if percent:
        result /= 100.0
    return result


def _number_warning(raw: Any, parsed: Optional[float], field_name: str) -> str:
    text = clean_text(raw)
    if text and parsed is None:
        return f"{field_name}“{text}”不是可无歧义解析的标量，已仅保留原值"
    return ""


def _source(table: TableIR, row_index: Optional[int] = None) -> str:
    ref = table.source.child(element_type="row", row_index=row_index) if row_index else table.source
    return str(ref)


def _top_section(path: Sequence[str]) -> str:
    return clean_text(path[0]) if path else ""


def _diagram_title(section: str, fallback: str) -> str:
    return _LEADING_NUMBER_RE.sub("", clean_text(section)).strip() or fallback


def _candidate_id(prefix: str, business_key: str, source_ref: str) -> str:
    digest = hashlib.sha256(f"{prefix}\x1f{business_key}\x1f{source_ref}".encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


@dataclass
class DiagramExtraction:
    diagram_type: str
    business_key: str
    title: str
    candidate: Dict[str, Any]
    facts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


_FTA_EVENT_HEADERS = {
    "event_id": ("事件ID", "事件编号", "基本事件ID", "节点ID"),
    "failure_mode_id": ("故障模式ID", "故障模式编号", "FMECA编号"),
    "module": ("模块", "所属模块", "单元", "设备"),
    "event_name": ("基本事件", "事件名称", "故障事件", "节点名称"),
    "cause": ("主要原因说明", "原因说明", "故障原因", "原因"),
    "probability": ("合成概率", "事件概率", "概率", "失效概率"),
}
_FTA_CUT_HEADERS = {
    "sequence": ("序号", "编号"),
    "order": ("阶数", "阶"),
    "cut_set": ("最小割集", "割集", "基本事件组合"),
}

_FTA_EVENT_SCHEMA = TableSchema.from_mapping(
    "fta_events",
    _FTA_EVENT_HEADERS,
    required_fields=("event_id", "event_name"),
    identity_groups=(("event_id",),),
    min_mapped_fields=2,
    max_header_depth=6,
    continuation_fields=("cause",),
)
_FTA_CUT_SCHEMA = TableSchema.from_mapping(
    "fta_cut_sets",
    _FTA_CUT_HEADERS,
    required_fields=("cut_set",),
    identity_groups=(("cut_set",),),
    min_mapped_fields=1,
    max_header_depth=6,
)


def _key_value_table(table: TableIR) -> Dict[str, str]:
    rows = table.matrix()
    result: Dict[str, str] = {}
    if not rows:
        return result
    start = 1 if _identity(rows[0][0] if rows[0] else "") in {"字段", "项目", "名称"} else 0
    for row in rows[start:]:
        if len(row) >= 2 and clean_text(row[0]) and clean_text(row[1]):
            result[clean_text(row[0])] = clean_text(row[1])
    return result


def _event_lookup_keys(module: str, name: str) -> Iterable[str]:
    yield _identity(f"{module}:{name}")
    yield _identity(name)


def _split_cut_set(value: str) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    # AND symbols are structural here because the column is explicitly a
    # minimal-cut-set column; commas inside prose are deliberately not split.
    parts = re.split(r"\s*(?:∧|＆|&&|\bAND\b|\band\b|与)\s*", text)
    return [clean_text(item) for item in parts if clean_text(item)]


def extract_fta_from_ir(ir: DocumentIR) -> List[DiagramExtraction]:
    """Extract one review candidate per explicit FTA chapter."""

    groups: Dict[str, Dict[str, Any]] = {}
    for table in ir.tables:
        section = _top_section(table.section_path)
        if not section:
            continue
        group = groups.setdefault(
            section,
            {"metadata": {}, "events": [], "cuts": [], "sources": []},
        )
        rows = table.matrix()
        if not rows:
            continue
        event_records = extract_records(TableSource.from_ir(table), _FTA_EVENT_SCHEMA)
        cut_records = extract_records(TableSource.from_ir(table), _FTA_CUT_SCHEMA)
        if event_records:
            for record in event_records:
                event_id = clean_text(record.values.get("event_id"))
                event_name = clean_text(record.values.get("event_name"))
                if not event_id or not event_name:
                    continue
                group["events"].append(
                    {
                        "event_id": event_id,
                        "failure_mode_id": clean_text(record.values.get("failure_mode_id")),
                        "module": clean_text(record.values.get("module")),
                        "name": event_name,
                        "description": clean_text(record.values.get("cause")),
                        "probability_raw": clean_text(record.values.get("probability")),
                        "probability": _float(record.values.get("probability")),
                        "source_ref": str(record.source_refs[0]),
                        "raw_data": record.raw_data,
                        "normalization_warnings": [
                            *[issue.as_dict() for issue in record.issues],
                            *(
                                [
                                    {
                                        "code": "ambiguous_numeric_preserved",
                                        "message": _number_warning(
                                            record.values.get("probability"),
                                            _float(record.values.get("probability")),
                                            "事件概率",
                                        ),
                                        "field": "probability",
                                        "severity": "warning",
                                    }
                                ]
                                if _number_warning(
                                    record.values.get("probability"),
                                    _float(record.values.get("probability")),
                                    "事件概率",
                                )
                                else []
                            ),
                        ],
                    }
                )
            group["sources"].append(_source(table))
        elif cut_records:
            for record in cut_records:
                expression = clean_text(record.values.get("cut_set"))
                members = _split_cut_set(expression)
                if not members:
                    continue
                group["cuts"].append(
                    {
                        "sequence": clean_text(record.values.get("sequence")) or str(len(group["cuts"]) + 1),
                        "order_raw": clean_text(record.values.get("order")),
                        "expression": expression,
                        "members": members,
                        "source_ref": str(record.source_refs[0]),
                        "raw_data": record.raw_data,
                        "normalization_warnings": [issue.as_dict() for issue in record.issues],
                    }
                )
            group["sources"].append(_source(table))
        else:
            values = _key_value_table(table)
            if any(_identity(key) in {"故障树编号", "顶事件", "关联功能", "建模假设"} for key in values):
                group["metadata"].update(values)
                group["sources"].append(_source(table))

    results: List[DiagramExtraction] = []
    for section, group in groups.items():
        metadata = group["metadata"]
        tree_id = next(
            (value for key, value in metadata.items() if _identity(key) in {"故障树编号", "fta编号", "模型编号"}),
            "",
        )
        top_event = next(
            (value for key, value in metadata.items() if _identity(key) in {"顶事件", "顶层事件"}),
            "",
        )
        if not tree_id and not top_event and not group["events"]:
            continue
        business_key = tree_id or f"FTA:{_identity(section)}"
        title = top_event or _diagram_title(section, business_key)
        root_id = f"top:{business_key}"
        vertices: List[Dict[str, Any]] = [
            {
                "id": root_id,
                "label": "TopEvent",
                "properties": {
                    "name": title,
                    "business_id": business_key,
                    "event_id": business_key,
                },
            }
        ]
        edges: List[Dict[str, Any]] = []
        event_by_key: Dict[str, str] = {}
        warnings: List[str] = []
        for event in group["events"]:
            event_id = event["event_id"]
            props = {
                "name": event["name"],
                "event_id": event_id,
                "business_id": event_id,
                "failure_mode_id": event["failure_mode_id"],
                "module": event["module"],
                "description": event["description"],
                "probability_raw": event["probability_raw"],
                "source_ref": event["source_ref"],
            }
            if event["probability"] is not None:
                props["probability"] = event["probability"]
            vertices.append({"id": event_id, "label": "BasicEvent", "properties": props})
            for lookup in _event_lookup_keys(event["module"], event["name"]):
                if lookup in event_by_key and event_by_key[lookup] != event_id:
                    warnings.append(f"基本事件名称存在歧义，未按名称自动合并: {event['name']}")
                    event_by_key[lookup] = ""
                else:
                    event_by_key[lookup] = event_id

        for cut_index, cut in enumerate(group["cuts"], start=1):
            member_ids: List[str] = []
            for member in cut["members"]:
                event_id = event_by_key.get(_identity(member), "")
                if not event_id and ":" in member:
                    event_id = event_by_key.get(_identity(member.split(":", 1)[1]), "")
                if not event_id:
                    warnings.append(f"最小割集成员无法唯一对应基本事件: {member}")
                    continue
                if event_id not in member_ids:
                    member_ids.append(event_id)
            if len(member_ids) != len(cut["members"]):
                continue
            if len(member_ids) == 1:
                edges.append(
                    {
                        "id": f"cut:{business_key}:{cut_index}:root",
                        "label": "causes",
                        "source": root_id,
                        "target": member_ids[0],
                        "properties": {
                            "gate_type": "OR",
                            "cut_set_no": cut["sequence"],
                            "source_ref": cut["source_ref"],
                        },
                    }
                )
                continue
            gate_id = f"cut:{business_key}:{cut_index}:and"
            vertices.append(
                {
                    "id": gate_id,
                    "label": "Gate",
                    "properties": {
                        "name": f"最小割集 {cut['sequence']}",
                        "gate_type": "AND",
                        "gate_group": cut["sequence"],
                        "source_ref": cut["source_ref"],
                    },
                }
            )
            edges.append(
                {
                    "id": f"cut:{business_key}:{cut_index}:root",
                    "label": "causes",
                    "source": root_id,
                    "target": gate_id,
                    "properties": {"gate_type": "OR", "cut_set_no": cut["sequence"]},
                }
            )
            for member_index, event_id in enumerate(member_ids, start=1):
                edges.append(
                    {
                        "id": f"cut:{business_key}:{cut_index}:{member_index}",
                        "label": "causes",
                        "source": gate_id,
                        "target": event_id,
                        "properties": {"gate_type": "AND", "cut_set_no": cut["sequence"]},
                    }
                )

        source_ref = group["sources"][0] if group["sources"] else ir.file_name
        linked_function = next(
            (value for key, value in metadata.items() if _identity(key) in {"关联功能", "相关功能"}),
            "",
        )
        assumption = next(
            (value for key, value in metadata.items() if _identity(key) in {"建模假设", "分析假设"}),
            "",
        )
        graph = {
            "diagram_type": "fta",
            "root_fault_id": root_id,
            "metadata": {
                "business_key": business_key,
                "title": title,
                "linked_function": linked_function,
                "modeling_assumption": assumption,
                "topology_source": "explicit_minimal_cut_sets",
                "source_refs": list(dict.fromkeys(group["sources"])),
            },
            "vertices": vertices,
            "edges": edges,
        }
        candidate = {
            "candidate_id": _candidate_id("fta_table", business_key, source_ref),
            "source_kind": "table_edges",
            "source_locator": source_ref,
            "source_fingerprint": hashlib.sha256(
                repr((metadata, group["events"], group["cuts"])).encode("utf-8")
            ).hexdigest(),
            "graph": graph,
            "confidence": 1.0,
        }
        results.append(
            DiagramExtraction(
                diagram_type="fta",
                business_key=business_key,
                title=title,
                candidate=candidate,
                facts={
                    "source_section": section,
                    "metadata": metadata,
                    "basic_events": group["events"],
                    "minimal_cut_sets": group["cuts"],
                },
                warnings=list(dict.fromkeys(warnings)),
            )
        )
    return results


_RBD_PARAM_HEADERS = {
    "module_id": ("模块ID", "模块编号", "单元ID", "设备ID"),
    "abbreviation": ("缩写", "简称", "模块缩写", "代号"),
    "name": ("模块名称", "单元名称", "设备名称", "名称"),
    "failure_rate": ("失效率λ（1/h）", "失效率λ", "失效率", "故障率", "lambda"),
    "reliability": ("可靠度", "任务可靠度", "R(t)"),
    "repair_rate": ("修复率μ（1/h）", "修复率μ", "修复率", "维修率", "mu"),
    "model": ("可靠度模型", "分布模型", "模型"),
}
_RBD_RESULT_HEADERS = {
    "mission_time": ("任务时间/h", "任务时间", "时间/h", "时间"),
    "reliability": ("系统可靠度", "可靠度", "R(t)"),
    "unreliability": ("不可靠度", "失效概率", "Q(t)"),
}

_RBD_PARAMETER_SCHEMA = TableSchema.from_mapping(
    "rbd_parameters",
    _RBD_PARAM_HEADERS,
    required_fields=(),
    identity_groups=(("module_id",), ("abbreviation",), ("name",)),
    min_mapped_fields=2,
    max_header_depth=6,
    continuation_fields=("name", "model"),
)
_RBD_RESULT_SCHEMA = TableSchema.from_mapping(
    "rbd_results",
    _RBD_RESULT_HEADERS,
    required_fields=("mission_time", "reliability"),
    identity_groups=(("mission_time",),),
    min_mapped_fields=2,
    max_header_depth=6,
)


def _rbd_evidence_only_candidate(
    ir: DocumentIR,
    section: str,
    group: Dict[str, Any],
    *,
    topology_source: str,
    warnings: Sequence[str],
) -> DiagramExtraction:
    """Keep parameter/result evidence without pretending it is a topology."""

    title = _diagram_title(section, "RBD 模型")
    business_key = f"RBD:{_identity(title)}"
    source_ref = next(
        (
            clean_text(item.get("source_ref"))
            for item in [*group["parameters"], *group["results"]]
            if clean_text(item.get("source_ref"))
        ),
        ir.file_name,
    )
    graph = {
        "diagram_type": "rbd",
        "root_fault_id": "",
        "metadata": {
            "business_key": business_key,
            "title": title,
            "success_criterion": group["criterion"],
            "parameters": group["parameters"],
            "declared_results": group["results"],
            "topology_source": topology_source,
            "topology_complete": False,
        },
        "vertices": [],
        "edges": [],
    }
    candidate = {
        "candidate_id": _candidate_id("rbd_parameters", business_key, source_ref),
        "source_kind": "table_nodes",
        "source_locator": source_ref,
        "source_fingerprint": hashlib.sha256(
            repr((group["parameters"], group["results"], topology_source)).encode("utf-8")
        ).hexdigest(),
        "graph": graph,
        "confidence": 1.0,
    }
    return DiagramExtraction(
        diagram_type="rbd",
        business_key=business_key,
        title=title,
        candidate=candidate,
        facts={
            "source_section": section,
            "success_criterion": group["criterion"],
            "expression": "",
            "parameters": group["parameters"],
            "declared_results": group["results"],
        },
        warnings=list(dict.fromkeys(warnings)),
    )


def extract_rbd_from_ir(ir: DocumentIR) -> List[DiagramExtraction]:
    """Extract explicit RBD expressions, parameters, and declared results."""

    groups: Dict[str, Dict[str, Any]] = {}
    for paragraph in ir.paragraphs:
        section = _top_section(paragraph.section_path)
        if not section:
            continue
        group = groups.setdefault(section, {"expressions": [], "criterion": "", "parameters": [], "results": []})
        if re.search(r"(?:结构|拓扑)\s*表达式\s*[：:]", paragraph.text, flags=re.IGNORECASE):
            group["expressions"].append((paragraph.text, str(paragraph.source)))
        elif re.match(r"\s*(?:系统)?成功判据\s*[：:]", paragraph.text):
            group["criterion"] = re.split(r"[：:]", paragraph.text, maxsplit=1)[-1].strip()

    for table in ir.tables:
        section = _top_section(table.section_path)
        if not section:
            continue
        group = groups.setdefault(section, {"expressions": [], "criterion": "", "parameters": [], "results": []})
        rows = table.matrix()
        if not rows:
            continue
        parameter_records = extract_records(TableSource.from_ir(table), _RBD_PARAMETER_SCHEMA)
        result_records = extract_records(TableSource.from_ir(table), _RBD_RESULT_SCHEMA)
        if parameter_records:
            for record in parameter_records:
                abbreviation = clean_text(record.values.get("abbreviation"))
                module_id = clean_text(record.values.get("module_id"))
                name = clean_text(record.values.get("name"))
                if not abbreviation and not module_id and not name:
                    continue
                failure_rate_raw = clean_text(record.values.get("failure_rate"))
                reliability_raw = clean_text(record.values.get("reliability"))
                repair_rate_raw = clean_text(record.values.get("repair_rate"))
                failure_rate = _float(failure_rate_raw)
                reliability = _float(reliability_raw)
                repair_rate = _float(repair_rate_raw)
                numeric_warnings = [
                    warning
                    for warning in (
                        _number_warning(failure_rate_raw, failure_rate, "失效率"),
                        _number_warning(reliability_raw, reliability, "可靠度"),
                        _number_warning(repair_rate_raw, repair_rate, "修复率"),
                    )
                    if warning
                ]
                group["parameters"].append(
                    {
                        "module_id": module_id,
                        "abbreviation": abbreviation,
                        "name": name,
                        "failure_rate_raw": failure_rate_raw,
                        "failure_rate": failure_rate,
                        "reliability_raw": reliability_raw,
                        "reliability": reliability,
                        "repair_rate_raw": repair_rate_raw,
                        "repair_rate": repair_rate,
                        "model": clean_text(record.values.get("model")),
                        "source_ref": str(record.source_refs[0]),
                        "raw_data": record.raw_data,
                        "normalization_warnings": [
                            *[issue.as_dict() for issue in record.issues],
                            *[
                                {
                                    "code": "ambiguous_numeric_preserved",
                                    "message": warning,
                                    "field": "rbd_parameter",
                                    "severity": "warning",
                                }
                                for warning in numeric_warnings
                            ],
                        ],
                    }
                )
        elif result_records:
            for record in result_records:
                mission_time_raw = clean_text(record.values.get("mission_time"))
                reliability_raw = clean_text(record.values.get("reliability"))
                unreliability_raw = clean_text(record.values.get("unreliability"))
                mission_time = _float(mission_time_raw)
                reliability = _float(reliability_raw)
                unreliability = _float(unreliability_raw)
                numeric_warnings = [
                    warning
                    for warning in (
                        _number_warning(mission_time_raw, mission_time, "任务时间"),
                        _number_warning(reliability_raw, reliability, "系统可靠度"),
                        _number_warning(unreliability_raw, unreliability, "不可靠度"),
                    )
                    if warning
                ]
                group["results"].append(
                    {
                        "mission_time": mission_time,
                        "mission_time_raw": mission_time_raw,
                        "reliability": reliability,
                        "reliability_raw": reliability_raw,
                        "unreliability": unreliability,
                        "unreliability_raw": unreliability_raw,
                        "source_ref": str(record.source_refs[0]),
                        "raw_data": record.raw_data,
                        "normalization_warnings": [
                            *[issue.as_dict() for issue in record.issues],
                            *[
                                {
                                    "code": "ambiguous_numeric_preserved",
                                    "message": warning,
                                    "field": "rbd_result",
                                    "severity": "warning",
                                }
                                for warning in numeric_warnings
                            ],
                        ],
                    }
                )

    extracted: List[DiagramExtraction] = []
    for section, group in groups.items():
        title = _diagram_title(section, "RBD 模型")
        business_key = f"RBD:{_identity(title)}"
        if not group["expressions"]:
            # Parameters/results are still first-class evidence, but an empty
            # topology must remain conflicted and impossible to publish.  Keep
            # the facts in the candidate metadata instead of inventing edges.
            if not group["parameters"] and not group["results"]:
                continue
            extracted.append(
                _rbd_evidence_only_candidate(
                    ir,
                    section,
                    group,
                    topology_source="parameters_only",
                    warnings=["仅识别到 RBD 参数/结果，缺少显式拓扑；候选不得确认或计算"],
                )
            )
            continue
        successful_expressions = 0
        expression_warnings: List[str] = []
        for expression_index, (expression, source_ref) in enumerate(group["expressions"], start=1):
            warnings: List[str] = []
            try:
                graph = rbd_expression_to_graph(
                    expression,
                    group["parameters"],
                    source_ref=source_ref,
                )
            except RbdExpressionError as exc:
                # A malformed explicit expression must be visible to review and
                # must never silently fall back to an inferred topology.
                warnings.append(str(exc))
                expression_warnings.extend(warnings)
                continue
            successful_expressions += 1
            graph["metadata"].update(
                {
                    "business_key": business_key,
                    "title": title,
                    "success_criterion": group["criterion"],
                    "declared_results": group["results"],
                    "topology_source": "structured_expression",
                }
            )
            candidate = {
                "candidate_id": _candidate_id("rbd_expression", business_key, source_ref),
                "source_kind": "structured_expression",
                "source_locator": source_ref,
                "source_fingerprint": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                "graph": graph,
                "confidence": 1.0,
            }
            extracted.append(
                DiagramExtraction(
                    diagram_type="rbd",
                    business_key=business_key,
                    title=title,
                    candidate=candidate,
                    facts={
                        "source_section": section,
                        "success_criterion": group["criterion"],
                        "expression": expression,
                        "parameters": group["parameters"],
                        "declared_results": group["results"],
                    },
                    warnings=warnings,
                )
            )
        if (
            successful_expressions == 0
            and (group["parameters"] or group["results"])
        ):
            extracted.append(
                _rbd_evidence_only_candidate(
                    ir,
                    section,
                    group,
                    topology_source="invalid_expression_parameters",
                    warnings=[
                        *expression_warnings,
                        "显式 RBD 拓扑表达式无效；仅保留参数/结果证据，不得确认或计算",
                    ],
                )
            )
    return extracted


__all__ = [
    "DiagramExtraction",
    "extract_fta_from_ir",
    "extract_rbd_from_ir",
]
