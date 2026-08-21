"""ICP FTA/RBD 确定性计算节点。"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from backend.icp.reliability import ReliabilityModelError, calculate_fta, calculate_rbd
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)


def _terms(state: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for value in state.get("entity_keywords", []) or []:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    query = str(state.get("rewritten_query") or state.get("query") or "").strip()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-/]{1,30}|[\u4e00-\u9fff]{2,12}", query):
        if token not in result and token not in {"故障", "原因", "可靠度", "可靠性", "系统", "查询"}:
            result.append(token)
    return result[:10]


def _mission_time(query: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:小时|h|hr|hour)", query, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _query_graphs(doc_type: str, terms: Sequence[str], limit: int = 3) -> List[Dict[str, Any]]:
    if not terms:
        return []
    clauses: List[str] = []
    params: List[Any] = [doc_type]
    for term in terms[:8]:
        clauses.append("(image_title ILIKE %s OR graph::text ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])
    params.append(limit)
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, doc_id, image_title, graph, created_at
                FROM document_fault_trees
                WHERE kb_id=%s AND ({' OR '.join(clauses)})
                ORDER BY created_at DESC LIMIT %s
                """,
                tuple(params),
            )
            columns = [desc[0] for desc in cur.description]
            rows = []
            for row in cur.fetchall():
                item = {name: value for name, value in zip(columns, row)}
                if isinstance(item.get("graph"), str):
                    try:
                        item["graph"] = json.loads(item["graph"])
                    except Exception:
                        item["graph"] = {}
                rows.append(item)
            return rows
    except Exception as exc:
        logger.warning("Graph calculation query failed: %s", exc)
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
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
                f"缺少概率参数={item.get('missing_probability_nodes', [])}"
            )
        elif kind == "RBD":
            lines.append(
                f"- RBD {title}：成功路径数={item.get('success_path_count', 0)}；"
                f"成功路径={json.dumps(item.get('success_paths', []), ensure_ascii=False)}；"
                f"系统可靠度={item.get('system_reliability')}；"
                f"缺少可靠度参数={item.get('missing_reliability_blocks', [])}"
            )
        else:
            lines.append(f"- {title}：{item.get('error', '')}")
    lines.append("说明：数值由确定性算法计算；缺失参数时仅返回拓扑/割集，不补造参数。")
    return "\n".join(lines)


async def icp_calculation_node(state) -> Dict[str, Any]:
    query = str(state.get("rewritten_query") or state.get("query") or "")
    intent = str(state.get("intent") or "")
    combined = query.lower()
    terms = _terms(state)
    mission_time = _mission_time(query)

    want_fta = intent in {"cause_analysis", "hybrid"} or any(
        keyword in combined for keyword in ("最小割集", "顶事件概率", "故障树", "原因链")
    )
    want_rbd = intent == "reliability_query" or any(
        keyword in combined for keyword in ("可靠度", "可靠性", "rbd", "串联", "并联", "冗余", "成功判据")
    )

    results: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []

    if want_fta:
        started = time.time()
        rows = _query_graphs("fta", terms)
        previews = []
        for row in rows:
            try:
                result = calculate_fta(row.get("graph") or {}, mission_time=mission_time)
                result["title"] = row.get("image_title", "")
                result["doc_id"] = row.get("doc_id", "")
                results.append(result)
                previews.append({
                    "title": row.get("image_title", ""),
                    "snippet": f"最小割集 {result.get('minimal_cut_set_count', 0)} 个；顶事件概率 {result.get('top_event_probability')}",
                })
            except ReliabilityModelError as exc:
                results.append({"model": "FTA", "title": row.get("image_title", ""), "error": str(exc)})
        tool_calls.append({
            "tool": "fta_analyzer",
            "count": len(rows),
            "duration": round(time.time() - started, 3),
            "params": {"mission_time": mission_time},
            "results_preview": previews,
        })

    if want_rbd:
        started = time.time()
        rows = _query_graphs("rbd", terms)
        previews = []
        for row in rows:
            try:
                result = calculate_rbd(row.get("graph") or {}, mission_time=mission_time)
                result["title"] = row.get("image_title", "")
                result["doc_id"] = row.get("doc_id", "")
                results.append(result)
                previews.append({
                    "title": row.get("image_title", ""),
                    "snippet": f"成功路径 {result.get('success_path_count', 0)} 条；系统可靠度 {result.get('system_reliability')}",
                })
            except ReliabilityModelError as exc:
                results.append({"model": "RBD", "title": row.get("image_title", ""), "error": str(exc)})
        tool_calls.append({
            "tool": "rbd_calculator",
            "count": len(rows),
            "duration": round(time.time() - started, 3),
            "params": {"mission_time": mission_time},
            "results_preview": previews,
        })

    calculation_text = _format_calculations(results)
    existing = state.get("diagnostic_context", "") or ""
    context = f"{existing}\n\n{calculation_text}".strip() if calculation_text else existing
    return {
        "calculation_results": results,
        "diagnostic_context": context,
        "tool_calls": tool_calls,
    }
