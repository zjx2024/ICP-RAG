"""Deterministic, user-readable projections of structured ICP evidence.

Canonical FTA/RBD graphs and calculation results remain machine data.  This
module creates bounded Chinese summaries for citations and execution previews;
it never asks an LLM to reinterpret the structure and never changes the raw
payload used by the diagnosis workflow.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_SUMMARY_ITEM_LIMIT = 6
_SUMMARY_TEXT_LIMIT = 180
_BASIC_EVENT_ITEM_LIMIT = 5
_BASIC_EVENT_DESCRIPTION_LIMIT = 72
_EXPLICIT_GRAPH_NOTE_LIMIT = 220
# 用户经常直接询问“有哪些分支/环节”，小规模集合的节选不是答案而是缺失。
# 权威结构本身有界：总量不超过该上限时一律完整呈现，超界才回到节选。
_FULL_ENUMERATION_LIMIT = 12


def _text(value: Any, limit: int = _SUMMARY_TEXT_LIMIT) -> str:
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return ""
    result = re.sub(r"\s+", " ", str(value or "")).strip()
    return result if len(result) <= limit else f"{result[: max(limit - 1, 0)]}…"


def _unique(values: Iterable[Any], *, limit: int = _SUMMARY_ITEM_LIMIT) -> tuple[List[str], int]:
    unique_values = list(
        dict.fromkeys(
            normalized
            for normalized in (_text(value) for value in values)
            if normalized
        )
    )
    total = len(unique_values)
    # 小集合完整呈现；超过有界上限时才回到调用方的节选 limit。
    cap = total if total <= _FULL_ENUMERATION_LIMIT else max(int(limit), 0)
    return unique_values[:cap], total


def _vertices(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = graph.get("vertices") or graph.get("nodes") or []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _edges(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in (graph.get("edges") or [])
        if isinstance(item, Mapping)
    ]


def _properties(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("properties") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _metadata(graph: Mapping[str, Any]) -> Dict[str, Any]:
    value = graph.get("metadata") or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _node_name(item: Mapping[str, Any]) -> str:
    properties = _properties(item)
    return _text(
        properties.get("name")
        or properties.get("title")
        or properties.get("event_name")
        or properties.get("module_name")
        or item.get("name")
        or item.get("title")
        or item.get("id")
    )


def _node_label(item: Mapping[str, Any]) -> str:
    return _text(item.get("label") or _properties(item).get("type"), 80).casefold()


def _with_total(items: Sequence[str], total: int) -> str:
    if not items:
        return ""
    suffix = f"（共 {total} 项）" if total > len(items) else ""
    return "、".join(items) + suffix


def _number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value, 80)
    if not math.isfinite(number):
        return ""
    return f"{number:.8g}"


def _count(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _explicit_note(value: Any, *, limit: int = _EXPLICIT_GRAPH_NOTE_LIMIT) -> str:
    """Render only explicitly stored scalar/list metadata, never mappings."""

    if isinstance(value, (list, tuple)):
        values, _ = _unique(value, limit=4)
        return _text("；".join(values), limit)
    return _text(value, limit)


def _basic_event_excerpts(nodes: Sequence[Mapping[str, Any]]) -> tuple[List[str], int]:
    entries: List[tuple[str, str]] = []
    seen_names: set[str] = set()
    for node in nodes:
        name = _node_name(node)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        properties = _properties(node)
        description = _text(
            properties.get("description")
            or properties.get("failure_cause")
            or properties.get("cause")
            or properties.get("reason"),
            _BASIC_EVENT_DESCRIPTION_LIMIT,
        )
        entries.append((name, description))
    total = len(entries)
    # 与 _unique 相同的原则：有界小集合不再“节选”，直接完整呈现。
    cap = total if total <= _FULL_ENUMERATION_LIMIT else _BASIC_EVENT_ITEM_LIMIT
    excerpts = [
        f"{name}（{description}）" if description else name
        for name, description in entries[:cap]
    ]
    return excerpts, total


def deterministic_graph_source_ref(graph: Mapping[str, Any]) -> str:
    """Return a source locator only when the graph carries an explicit one.

    Callers still decide whether the graph's validation tier is authoritative
    enough to expose this fallback.  Generated node IDs and revision UUIDs are
    intentionally not treated as source locations.
    """

    metadata = _metadata(graph)
    direct = _text(metadata.get("source_ref"), 800)
    if direct:
        return direct
    refs = metadata.get("source_refs") or []
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    for raw in refs:
        if isinstance(raw, str):
            value = _text(raw, 800)
        elif isinstance(raw, Mapping):
            locator = raw.get("source_locator") or {}
            value = _text(
                raw.get("source_ref")
                or (locator.get("ref") if isinstance(locator, Mapping) else ""),
                800,
            )
        else:
            value = ""
        if value:
            return value
    return ""


def summarize_graph_structure(
    graph: Mapping[str, Any],
    *,
    diagram_type: str,
    title: str = "",
) -> str:
    """Create a bounded FTA/RBD structure summary without inferring new facts."""

    graph = graph if isinstance(graph, Mapping) else {}
    vertices = _vertices(graph)
    edges = _edges(graph)
    title_text = _text(title) or _text(_metadata(graph).get("title"))
    kind = str(diagram_type or "").strip().casefold()

    if kind == "fta":
        by_id = {_text(item.get("id"), 240): item for item in vertices}
        root_id = _text(graph.get("root_fault_id"), 240)
        root = by_id.get(root_id)
        if root is None:
            root = next((item for item in vertices if _node_label(item) == "topevent"), None)
        top_event = _node_name(root or {}) or title_text
        gate_types, gate_total = _unique(
            [
                *(_properties(item).get("gate_type") for item in vertices),
                *(_properties(item).get("gate_type") for item in edges),
            ]
        )
        basic_nodes = [
            item
            for item in vertices
            if _node_label(item) in {"basicevent", "basic_event", "basic event"}
        ]
        event_excerpts, event_total = _basic_event_excerpts(basic_nodes)
        modeling_assumption = _explicit_note(
            _metadata(graph).get("modeling_assumption")
        )
        heading = title_text or top_event or "未命名故障树"
        parts = [f"FTA 结构：{heading}"]
        if top_event and top_event != heading:
            parts.append(f"顶事件：{top_event}")
        parts.append(f"结构包含 {len(vertices)} 个节点、{len(edges)} 条关系")
        if gate_types:
            parts.append(f"逻辑门：{_with_total(gate_types, gate_total)}")
        if event_excerpts:
            # 摘要只报基本事件数量，不把 name（description）明细列给回答模型：
            # 明细交由 _ensure_fta_structure_only_completeness 从图确定性补全，
            # 否则模型会把同一列举复述成重复段落（或把 {‘name’: …} 抄成字面量）。
            parts.append(f"基本事件共 {event_total} 项")
        if modeling_assumption:
            parts.append(f"建模说明：{modeling_assumption}")
        return "；".join(parts) + "。"

    start = next(
        (
            item
            for item in vertices
            if _node_label(item) == "start" or _node_name(item).casefold() in {"start", "起点", "开始"}
        ),
        None,
    )
    end = next(
        (
            item
            for item in vertices
            if _node_label(item) == "end" or _node_name(item).casefold() in {"end", "终点", "结束"}
        ),
        None,
    )
    excluded = {
        _text((start or {}).get("id"), 240),
        _text((end or {}).get("id"), 240),
    }
    blocks = [item for item in vertices if _text(item.get("id"), 240) not in excluded]
    block_names, block_total = _unique(_node_name(item) for item in blocks)
    metadata = _metadata(graph)
    success_criterion = _explicit_note(
        metadata.get("success_criterion") or metadata.get("success_criteria")
    )
    structure_note = _explicit_note(
        metadata.get("structure")
        or metadata.get("structure_description")
        or metadata.get("topology_description")
    )
    parts = [f"RBD 结构：{title_text or '未命名可靠性框图'}"]
    if start or end:
        parts.append(f"起止：{_node_name(start or {}) or 'START'} → {_node_name(end or {}) or 'END'}")
    parts.append(f"结构包含 {len(vertices)} 个节点、{len(edges)} 条关系")
    if block_names:
        parts.append(f"功能块：{_with_total(block_names, block_total)}")
    if success_criterion:
        parts.append(f"成功判据：{success_criterion}")
    if structure_note:
        parts.append(f"结构说明：{structure_note}")
    return "；".join(parts) + "。"


def summarize_diagram_locator(*, doc_type: str, title: str = "") -> str:
    """Summarize an OpenSearch diagram locator without promoting it to fact."""

    kind = str(doc_type or "").strip().casefold()
    label = {
        "fta": "FTA 故障树",
        "rbd": "RBD 可靠性框图",
        "function_locator": "系统架构图",
    }.get(kind, "工程图")
    return f"{label}检索线索：{_text(title) or '未命名图结构'}。详细结构需按来源定位回查当前图修订。"


def summarize_fta_calculation(
    result: Mapping[str, Any],
    *,
    graph: Mapping[str, Any] | None = None,
) -> str:
    graph = graph if isinstance(graph, Mapping) else {}
    names_by_id = {
        _text(item.get("id"), 240): _node_name(item)
        for item in _vertices(graph)
    }
    cut_summaries: List[str] = []
    raw_cuts = result.get("minimal_cut_sets") or []
    if isinstance(raw_cuts, (list, tuple)):
        for raw_cut in raw_cuts[:12]:
            values = raw_cut if isinstance(raw_cut, (list, tuple, set, frozenset)) else [raw_cut]
            labels, _ = _unique(names_by_id.get(_text(value, 240)) or value for value in values)
            if labels:
                cut_summaries.append(" + ".join(labels))
    title = _text(result.get("title")) or _text(result.get("top_event")) or "未命名故障树"
    parts = [f"FTA 确定性计算：{title}"]
    top_event = _text(result.get("top_event"))
    if top_event:
        parts.append(f"顶事件：{top_event}")
    count = _count(result.get("minimal_cut_set_count"))
    cut_text = f"最小割集 {count} 个"
    if cut_summaries:
        enumerated = "；".join(
            f"{index}. {summary}"
            for index, summary in enumerate(cut_summaries, start=1)
        )
        label = "完整列表" if count <= len(cut_summaries) else "前 12 个"
        cut_text += f"（{label}：{enumerated}{'；…' if count > len(cut_summaries) else ''}）"
    parts.append(cut_text)
    probability = _number(result.get("top_event_probability"))
    if probability:
        parts.append(f"顶事件概率：{probability}")
    else:
        missing, missing_total = _unique(result.get("missing_probability_nodes") or [])
        reason = _text(result.get("probability_block_reason"), 260)
        if missing:
            parts.append(f"未形成概率结论，缺少参数：{_with_total(missing, missing_total)}")
        elif reason:
            parts.append(f"未形成概率结论：{reason}")
        else:
            parts.append("未形成顶事件概率结论")
    return "；".join(parts) + "。"


def summarize_rbd_calculation(result: Mapping[str, Any]) -> str:
    title = _text(result.get("title")) or "未命名可靠性框图"
    parts = [f"RBD 确定性计算：{title}"]
    count = _count(result.get("success_path_count"))
    count_label = "已枚举成功路径" if result.get("path_limit_reached") else "成功路径"
    path_summaries: List[str] = []
    raw_paths = result.get("success_paths") or []
    if isinstance(raw_paths, (list, tuple)):
        for raw_path in raw_paths[:3]:
            values = raw_path if isinstance(raw_path, (list, tuple)) else [raw_path]
            labels, _ = _unique(values)
            if labels:
                path_summaries.append(" → ".join(labels))
    path_text = f"{count_label} {count} 条"
    if path_summaries:
        path_text += f"（示例：{'；'.join(path_summaries)}{'；…' if count > len(path_summaries) else ''}）"
    parts.append(path_text)
    reliability = _number(result.get("system_reliability"))
    if reliability:
        parts.append(f"系统可靠度：{reliability}")
    else:
        missing, missing_total = _unique(result.get("missing_reliability_blocks") or [])
        reason = _text(result.get("calculation_limit_reason"), 260)
        if missing:
            parts.append(f"未形成可靠度结论，缺少参数：{_with_total(missing, missing_total)}")
        elif reason:
            parts.append(f"未形成可靠度结论：{reason}")
        else:
            parts.append("未形成系统可靠度结论")
    return "；".join(parts) + "。"


__all__ = [
    "deterministic_graph_source_ref",
    "summarize_diagram_locator",
    "summarize_fta_calculation",
    "summarize_graph_structure",
    "summarize_rbd_calculation",
]
