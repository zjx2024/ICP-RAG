"""ICP FTA/RBD 确定性计算节点。"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.evidence_display import (
    deterministic_graph_source_ref,
    summarize_fta_calculation,
    summarize_rbd_calculation,
)
from backend.agent.query_understanding import build_tool_terms
from backend.agent.tool_policy import detect_calculation_requests
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import (
    FORMAL_VALIDATION_STATUSES,
    VALIDATION_FAILED,
    VALIDATION_PROVISIONAL,
    VALIDATION_QUARANTINED,
)
from backend.icp.reliability import ReliabilityModelError, calculate_fta, calculate_rbd

logger = logging.getLogger(__name__)

_TOOL_CALL_EVIDENCE_LIMIT = 8
_TOOL_CALL_PREVIEW_SNIPPET_LIMIT = 260
_TOOL_CALL_PREVIEW_SOURCE_LIMIT = 320


_CHINESE_DURATION_DIGITS = {
    "零": 0,
    "〇": 0,
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


def _chinese_duration_number(value: str) -> Optional[float]:
    """Parse a bounded Chinese integer used directly before a time unit."""

    text = str(value or "").strip()
    if not text:
        return None
    if "百" in text:
        left, right = text.split("百", 1)
        hundreds = _CHINESE_DURATION_DIGITS.get(left, 1 if not left else -1)
        if hundreds < 0:
            return None
        tail = _chinese_duration_number(right) if right else 0
        return float(hundreds * 100 + (tail or 0))
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _CHINESE_DURATION_DIGITS.get(left, 1 if not left else -1)
        ones = _CHINESE_DURATION_DIGITS.get(right, 0 if not right else -1)
        if tens < 0 or ones < 0:
            return None
        return float(tens * 10 + ones)
    digits = [_CHINESE_DURATION_DIGITS.get(char, -1) for char in text]
    if not digits or any(value < 0 for value in digits):
        return None
    return float("".join(str(value) for value in digits))


def _mission_time(query: str) -> Optional[float]:
    value = str(query or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:小时|h|hr|hour)", value, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    chinese_match = re.search(
        r"([零〇一二两三四五六七八九十百]+)\s*(?:小时|钟头)",
        value,
    )
    return _chinese_duration_number(chinese_match.group(1)) if chinese_match else None


def _graph_validation_status(row: Dict[str, Any]) -> str:
    validation_status = str(row.get("validation_status") or "").strip()
    if validation_status in {VALIDATION_PROVISIONAL, VALIDATION_QUARANTINED, VALIDATION_FAILED}:
        return validation_status
    if validation_status in FORMAL_VALIDATION_STATUSES:
        return validation_status
    # Backward compatibility for human-confirmed rows created before the
    # validation columns existed.  Explicit automatic statuses above win.
    if row.get("review_status") == "confirmed":
        return "manual_confirmed"
    return validation_status or str(row.get("review_status") or "pending").strip()


def _is_usable_graph_row(row: Dict[str, Any]) -> bool:
    return (
        _graph_validation_status(row) in FORMAL_VALIDATION_STATUSES
        and row.get("projection_status") == "published"
        and row.get("is_active") is True
    )


def _is_usable_graph_evidence(item: Dict[str, Any]) -> bool:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False
    validation_status = str(metadata.get("validation_status") or item.get("review_status") or "").strip()
    if validation_status == "confirmed":
        validation_status = "manual_confirmed"
    return (
        validation_status in FORMAL_VALIDATION_STATUSES
        and item.get("authority") in {"confirmed_relation", "validated_relation"}
        and metadata.get("is_active") is True
        and metadata.get("projection_status") == "published"
    )


def _graph_identity_values(row: Mapping[str, Any]) -> set[str]:
    """Return stable identifiers exposed by a graph revision or source row.

    The graph repository commonly returns a revision UUID, while FTA/RBD
    source evidence uses the business key (for example ``FTA-SYN-005``).
    Both identify the same model and must bind to one another for deterministic
    calculation evidence to pass the provenance gate.
    """

    values: set[str] = set()
    for key in (
        "revision_id", "id", "diagram_key", "diagram_id", "record_id",
        "title", "image_title", "name",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            values.add(value.casefold())
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("revision_id", "diagram_key", "diagram_id", "record_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                values.add(value.casefold())
    return values


def _requested_graph_keys(state: Mapping[str, Any], kind: str) -> set[str]:
    """Extract explicit or cited FTA/RBD business keys for this turn."""

    prefix = kind.upper()
    texts: List[str] = [str(state.get("query") or "")]
    frame = state.get("query_frame")
    if isinstance(frame, Mapping):
        texts.extend(str(value or "") for value in (frame.get("entities") or []))
        texts.extend(str(value or "") for value in (frame.get("normalized_terms") or []))
    for anchor in state.get("selected_conversation_anchors") or []:
        if not isinstance(anchor, Mapping):
            continue
        texts.extend(str(anchor.get(key) or "") for key in ("record_id", "title"))
        fields = anchor.get("answer_fields")
        if isinstance(fields, Mapping):
            texts.extend(
                str(fields.get(key) or "")
                for key in ("diagram_key", "diagram_id", "revision_id", "record_id")
            )
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(prefix)}[-_][A-Za-z0-9][A-Za-z0-9._-]*",
        re.IGNORECASE,
    )
    return {
        match.group(0).casefold()
        for text in texts
        for match in pattern.finditer(text)
    }


def _calculation_parent_evidence_ids(
    row: Mapping[str, Any],
    kind: str,
    evidence_registry: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Find the active, validated structural source for a calculation row."""

    row_keys = _graph_identity_values(row)
    result: List[str] = []
    for raw_item in evidence_registry or []:
        if not isinstance(raw_item, Mapping) or str(raw_item.get("doc_type") or "") != kind:
            continue
        item = dict(raw_item)
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = item.get("document_metadata")
        if isinstance(metadata, Mapping):
            item["metadata"] = dict(metadata)
        if not (row_keys & _graph_identity_values(item)):
            continue
        if not _is_usable_graph_evidence(item):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id and evidence_id not in result:
            result.append(evidence_id)
    return result


def _query_graphs(
    doc_type: str,
    terms: Sequence[str],
    limit: int = 3,
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Read only active, formally validated and published PG revisions."""
    if not terms:
        return []
    try:
        tokens = [str(value).casefold() for value in terms[:8] if str(value).strip()]
        rows: List[Dict[str, Any]] = []
        for revision in diagram_repository.list_active_revisions(diagram_type=doc_type):
            # 仓储层已经筛选，这里仍保留边界校验，防止测试替身或未来查询
            # 防止旧修订或仅 provisional 的修订进入正式计算。
            if not _is_usable_graph_row(revision):
                continue
            haystack = (
                f"{revision.get('title', '')} {revision.get('diagram_key', '')} "
                f"{json.dumps(revision.get('canonical_graph') or {}, ensure_ascii=False, default=str)}"
            ).casefold()
            if not any(token in haystack for token in tokens):
                continue
            item = dict(revision)
            item["graph"] = item.get("canonical_graph") or {}
            item["image_title"] = item.get("title") or item.get("diagram_key") or ""
            rows.append(item)
            if len(rows) >= limit:
                break
        return rows
    except Exception as exc:
        logger.warning("Graph calculation query failed: %s", exc)
        if errors is not None:
            errors.append(f"{doc_type.upper()} 图修订查询失败: {exc}")
        return []


def _format_calculations(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["【FTA/RBD确定性计算】"]
    for item in results:
        kind = item.get("model")
        title = item.get("title", "")
        if kind == "FTA":
            lines.append(
                f"- FTA {title}：顶事件={item.get('top_event', '')}；"
                f"最小割集数={item.get('minimal_cut_set_count', 0)}；"
                f"最小割集={json.dumps(item.get('minimal_cut_sets', []), ensure_ascii=False)}；"
                f"顶事件概率={item.get('top_event_probability')}；"
                f"缺少概率参数={item.get('missing_probability_nodes', [])}；"
                f"概率阻断原因={item.get('probability_block_reason', '')}"
            )
        elif kind == "RBD":
            path_count_label = (
                "已枚举成功路径数"
                if item.get("path_limit_reached")
                else "成功路径数"
            )
            lines.append(
                f"- RBD {title}：{path_count_label}={item.get('success_path_count', 0)}；"
                f"成功路径={json.dumps(item.get('success_paths', []), ensure_ascii=False)}；"
                f"系统可靠度={item.get('system_reliability')}；"
                f"缺少可靠度参数={item.get('missing_reliability_blocks', [])}；"
                f"路径枚举截断={item.get('path_limit_reached', False)}；"
                f"计算限制={item.get('calculation_limit_reason', '')}"
            )
        else:
            lines.append(f"- {title}：{item.get('error', '')}")
    lines.append("说明：数值由确定性算法计算；缺失参数时仅返回拓扑/割集，不补造参数。")
    return "\n".join(lines)


def _attach_calculation_call_evidence(
    tool_calls: Sequence[Dict[str, Any]],
    tool_results: Sequence[Dict[str, Any]],
    evidence_registry: Sequence[Dict[str, Any]],
) -> None:
    """Expose only bounded, normalized calculation evidence in tool events."""

    result_ids_by_tool: Dict[str, List[str]] = {}
    for result in tool_results or []:
        tool = str(result.get("tool") or "").strip()
        if not tool:
            continue
        target = result_ids_by_tool.setdefault(tool, [])
        for value in result.get("evidence_ids") or []:
            evidence_id = str(value or "").strip()
            if evidence_id and evidence_id not in target:
                target.append(evidence_id)
    evidence_by_id = {
        str(item.get("evidence_id") or "").strip(): item
        for item in evidence_registry or []
        if str(item.get("evidence_id") or "").strip()
    }

    for call in tool_calls or []:
        all_ids = result_ids_by_tool.get(str(call.get("tool") or "").strip(), [])
        selected_ids = all_ids[:_TOOL_CALL_EVIDENCE_LIMIT]
        call["evidence_ids"] = selected_ids
        call["evidence_count"] = len(all_ids)
        previews: List[Dict[str, Any]] = []
        existing_previews = list(call.get("results_preview") or [])[:8]
        for index, evidence_id in enumerate(selected_ids):
            item = evidence_by_id.get(evidence_id)
            if not item:
                continue
            existing = existing_previews[index] if index < len(existing_previews) else {}
            display_summary = " ".join(
                str(item.get("display_summary") or "").split()
            )[:_TOOL_CALL_PREVIEW_SNIPPET_LIMIT]
            previews.append(
                {
                    "title": str(
                        existing.get("title")
                        or item.get("title")
                        or item.get("doc_name")
                        or "计算证据"
                    ).strip()[:100],
                    "snippet": display_summary or " ".join(
                        str(existing.get("snippet") or item.get("content") or "").split()
                    )[:_TOOL_CALL_PREVIEW_SNIPPET_LIMIT],
                    "display_summary": display_summary,
                    "evidence_id": evidence_id,
                    "source_ref": str(item.get("source_ref") or "").strip()[
                        :_TOOL_CALL_PREVIEW_SOURCE_LIMIT
                    ],
                }
            )
        if previews:
            call["results_preview"] = previews


async def icp_calculation_node(state) -> Dict[str, Any]:
    # A rewritten query may improve retrieval, but it must not invent a task
    # duration. Numeric parameters are read from the original user request.
    raw_user_query = str(state.get("query") or "")
    mission_time = _mission_time(raw_user_query)
    terms_by_type = {
        "fta": build_tool_terms(state, "fta_query", limit=16),
        "rbd": build_tool_terms(state, "rbd_query", limit=16),
    }

    allowed_tools = set(state.get("allowed_tools") or [])
    raw_requests = state.get("calculation_requests")
    if raw_requests is None:
        # 兼容旧会话状态，但仍只从用户原始问题授权。
        raw_requests = detect_calculation_requests(raw_user_query)
    calculation_requests = {
        str(value or "").strip() for value in (raw_requests or []) if str(value or "").strip()
    }
    want_fta = (
        "fta_analyzer" in allowed_tools
        and bool(calculation_requests & {"fta_cut_sets", "fta_probability"})
    )
    want_rbd = (
        "rbd_calculator" in allowed_tools
        and "rbd_reliability" in calculation_requests
    )

    results: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    formal_graphs = [
        dict(item) for item in (state.get("confirmed_graphs") or [])
        if _is_usable_graph_row(item)
    ]

    def graph_rows(kind: str, errors: List[str]) -> List[Dict[str, Any]]:
        selected = [
            item for item in formal_graphs
            if str(item.get("diagram_type") or item.get("doc_type") or "") == kind
        ]
        # A semantic retrieval may have hydrated only part of a requested model
        # set (for example two of four RBDs). Always merge the repository query
        # instead of treating any non-empty selection as complete.
        queried = _query_graphs(kind, terms_by_type[kind], errors=errors)
        requested_keys = _requested_graph_keys(state, kind)
        if requested_keys:
            # A follow-up such as “按这份 FTA 计算” inherits the selected
            # business key from its cited anchor. Do not add sibling diagrams
            # merely because a broad lexical query also matches them.
            selected = [
                item for item in selected
                if _graph_identity_values(item) & requested_keys
            ]
            queried = [
                item for item in queried
                if _graph_identity_values(item) & requested_keys
            ]
        result: List[Dict[str, Any]] = []
        seen = set()
        for item in [*selected, *queried]:
            identity = str(item.get("revision_id") or item.get("id") or "").strip()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            result.append(item)
        return result

    def parent_evidence_ids(row: Dict[str, Any], kind: str) -> List[str]:
        return _calculation_parent_evidence_ids(
            row,
            kind,
            state.get("evidence_registry") or [],
        )

    if want_fta:
        started = time.time()
        previews = []
        errors: List[str] = []
        warnings: List[str] = []
        rows = graph_rows("fta", errors)
        for row in rows:
            try:
                result = calculate_fta(row.get("graph") or {}, mission_time=mission_time)
                result["title"] = row.get("image_title", "")
                result["doc_id"] = row.get("doc_id", "")
                result["revision_id"] = row.get("revision_id", "")
                results.append(result)
                parents = parent_evidence_ids(row, "fta")
                missing_probabilities = result.get("missing_probability_nodes") or []
                probability_block_reason = str(result.get("probability_block_reason") or "").strip()
                numeric_complete = (
                    result.get("top_event_probability") is not None
                    and not missing_probabilities
                    and not probability_block_reason
                )
                if not parents:
                    warnings.append(
                        f"FTA 修订 {row.get('revision_id') or row.get('id') or '未知'} 缺少可引用的有效图证据"
                    )
                graph = row.get("graph") or {}
                source_ref = str(row.get("source_ref") or row.get("image_url") or "")
                if not source_ref:
                    source_ref = deterministic_graph_source_ref(graph)
                evidence.append(
                    make_evidence(
                        source_kind="calculation",
                        authority="deterministic_derivation",
                        evidence_level="L2_CORROBORATED",
                        claim_type="fta_calculation",
                        doc_type="fta",
                        doc_id=str(row.get("doc_id") or ""),
                        doc_name=str(row.get("doc_name") or ""),
                        record_id=str(row.get("revision_id") or ""),
                        source_ref=source_ref,
                        title=f"FTA 确定性计算：{result['title']}",
                        content=json.dumps(result, ensure_ascii=False, default=str),
                        display_summary=summarize_fta_calculation(result, graph=graph),
                        derived_from=parents,
                        limitations=[
                            result.get("assumption", ""),
                            "" if parents else "缺少可引用的当前有效图修订证据，门禁将阻止正式计算结论",
                            (
                                ""
                                if numeric_complete
                                else probability_block_reason
                                or "缺少基本事件概率/故障率参数，不得形成顶事件概率结论"
                            ),
                        ],
                        metadata={
                            "algorithm": "backend.icp.reliability.calculate_fta",
                            "graph_revision_id": str(row.get("revision_id") or ""),
                            "numeric_complete": numeric_complete,
                        },
                    )
                )
                previews.append({
                    "title": row.get("image_title", ""),
                    "snippet": f"最小割集 {result.get('minimal_cut_set_count', 0)} 个；顶事件概率 {result.get('top_event_probability')}",
                })
                if not numeric_complete:
                    reason = probability_block_reason or "缺少概率参数"
                    warnings.append(
                        f"FTA {row.get('revision_id') or row.get('id') or '未知'} {reason}，"
                        "仅保留拓扑/最小割集结果"
                    )
            except ReliabilityModelError as exc:
                results.append({"model": "FTA", "title": row.get("image_title", ""), "error": str(exc)})
                errors.append(str(exc))
            except Exception as exc:
                logger.exception("FTA deterministic calculation failed")
                results.append({"model": "FTA", "title": row.get("image_title", ""), "error": str(exc)})
                errors.append(f"未预期计算错误: {exc}")
        tool_calls.append({
            "tool": "fta_analyzer",
            "count": len(rows),
            "duration": round(time.time() - started, 3),
            "params": {"mission_time": mission_time},
            "results_preview": previews,
        })
        fta_evidence = [item for item in evidence if item.get("claim_type") == "fta_calculation"]
        tool_results.append(
            make_tool_result(
                tool="fta_analyzer",
                status=(
                    "failed" if errors and not previews
                    else "partial" if errors or warnings
                    else "ok" if previews
                    else "empty"
                ),
                params={"mission_time": mission_time},
                data={"model_count": len(rows), "result_count": len(previews)},
                evidence_ids=[item["evidence_id"] for item in fta_evidence],
                backend="postgres+local_deterministic",
                warnings=warnings,
                errors=errors,
            )
        )

    if want_rbd:
        started = time.time()
        previews = []
        errors: List[str] = []
        warnings: List[str] = []
        rows = graph_rows("rbd", errors)
        for row in rows:
            try:
                result = calculate_rbd(row.get("graph") or {}, mission_time=mission_time)
                result["title"] = row.get("image_title", "")
                result["doc_id"] = row.get("doc_id", "")
                result["revision_id"] = row.get("revision_id", "")
                results.append(result)
                parents = parent_evidence_ids(row, "rbd")
                missing_reliability = result.get("missing_reliability_blocks") or []
                path_limit_reached = result.get("path_limit_reached") is True
                calculation_limit_reason = str(result.get("calculation_limit_reason") or "").strip()
                numeric_complete = (
                    result.get("system_reliability") is not None
                    and not missing_reliability
                    and not path_limit_reached
                    and not calculation_limit_reason
                )
                if not parents:
                    warnings.append(
                        f"RBD 修订 {row.get('revision_id') or row.get('id') or '未知'} 缺少可引用的有效图证据"
                    )
                graph = row.get("graph") or {}
                source_ref = str(row.get("source_ref") or row.get("image_url") or "")
                if not source_ref:
                    source_ref = deterministic_graph_source_ref(graph)
                evidence.append(
                    make_evidence(
                        source_kind="calculation",
                        authority="deterministic_derivation",
                        evidence_level="L2_CORROBORATED",
                        claim_type="rbd_calculation",
                        doc_type="rbd",
                        doc_id=str(row.get("doc_id") or ""),
                        doc_name=str(row.get("doc_name") or ""),
                        record_id=str(row.get("revision_id") or ""),
                        source_ref=source_ref,
                        title=f"RBD 确定性计算：{result['title']}",
                        content=json.dumps(result, ensure_ascii=False, default=str),
                        display_summary=summarize_rbd_calculation(result),
                        derived_from=parents,
                        limitations=[
                            result.get("assumption", ""),
                            "" if parents else "缺少可引用的当前有效图修订证据，门禁将阻止正式计算结论",
                            (
                                ""
                                if numeric_complete
                                else calculation_limit_reason
                                or "缺少模块可靠度/故障率参数，不得形成系统可靠度数值结论"
                            ),
                        ],
                        metadata={
                            "algorithm": "backend.icp.reliability.calculate_rbd",
                            "graph_revision_id": str(row.get("revision_id") or ""),
                            "numeric_complete": numeric_complete,
                        },
                    )
                )
                previews.append({
                    "title": row.get("image_title", ""),
                    "snippet": (
                        f"{'已枚举' if result.get('path_limit_reached') else ''}成功路径 "
                        f"{result.get('success_path_count', 0)} 条；系统可靠度 "
                        f"{result.get('system_reliability')}"
                    ),
                })
                if not numeric_complete:
                    reason = calculation_limit_reason or "缺少可靠度参数"
                    warnings.append(
                        f"RBD {row.get('revision_id') or row.get('id') or '未知'} {reason}，"
                        "仅保留成功路径结果"
                    )
            except ReliabilityModelError as exc:
                results.append({"model": "RBD", "title": row.get("image_title", ""), "error": str(exc)})
                errors.append(str(exc))
            except Exception as exc:
                logger.exception("RBD deterministic calculation failed")
                results.append({"model": "RBD", "title": row.get("image_title", ""), "error": str(exc)})
                errors.append(f"未预期计算错误: {exc}")
        tool_calls.append({
            "tool": "rbd_calculator",
            "count": len(rows),
            "duration": round(time.time() - started, 3),
            "params": {"mission_time": mission_time},
            "results_preview": previews,
        })
        rbd_evidence = [item for item in evidence if item.get("claim_type") == "rbd_calculation"]
        tool_results.append(
            make_tool_result(
                tool="rbd_calculator",
                status=(
                    "failed" if errors and not previews
                    else "partial" if errors or warnings
                    else "ok" if previews
                    else "empty"
                ),
                params={"mission_time": mission_time},
                data={"model_count": len(rows), "result_count": len(previews)},
                evidence_ids=[item["evidence_id"] for item in rbd_evidence],
                backend="postgres+local_deterministic",
                warnings=warnings,
                errors=errors,
            )
        )

    calculation_text = _format_calculations(results)
    existing = state.get("diagnostic_context", "") or ""
    context = f"{existing}\n\n{calculation_text}".strip() if calculation_text else existing
    _attach_calculation_call_evidence(tool_calls, tool_results, evidence)
    return {
        "calculation_results": results,
        "diagnostic_context": context,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "evidence_registry": evidence,
    }
