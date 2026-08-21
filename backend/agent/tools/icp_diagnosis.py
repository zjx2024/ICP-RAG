"""ICP 专用故障诊断编排工具。

该节点不是让大模型自由猜测，而是根据问题类型调用七类文档的专用检索逻辑：
Function Locator、HMC、FMECA、FTA、RBD、MFL、故障历史，并输出可追溯候选。
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from backend.core.retrieval import knowledge_retriever
from backend.icp.repository import icp_repository
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

ICP_DOC_TYPES = (
    "function_locator",
    "hmc",
    "fmeca",
    "fta",
    "rbd",
    "mfl",
    "fault_history",
)

_HMC_PATTERN = re.compile(
    r"(?P<lcn>\d{6,10})\s*[-_/]?\s*(?P<hamc>\*\*|[A-Za-z0-9]{1,4})\s*[-_/]?\s*(?P<hsc>\d{3,6})"
)


def _unique_terms(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for value in state.get("entity_keywords", []) or []:
        if value:
            values.append(str(value).strip())
    rewritten = str(state.get("rewritten_query") or "").strip()
    original = str(state.get("query") or "").strip()
    if rewritten:
        values.append(rewritten)
    if original and original != rewritten:
        values.append(original)

    # 拆出技术词，过滤低信息词。
    stop = {
        "故障", "异常", "问题", "怎么办", "怎么回事", "是什么", "可能", "出现",
        "系统", "设备", "模块", "请问", "一下", "相关", "查询",
    }
    for text in (rewritten, original):
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-/]{1,30}|[\u4e00-\u9fff]{2,12}", text):
            if token not in stop:
                values.append(token)

    result: List[str] = []
    seen = set()
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result[:16]


def _extract_hmc(query: str) -> Dict[str, str]:
    match = _HMC_PATTERN.search(query.replace("—", "-").replace("–", "-"))
    if not match:
        return {}
    result = {k: v.upper() for k, v in match.groupdict().items()}
    result["hmc_code"] = f"{result['lcn']}{result['hamc']}{result['hsc']}"
    return result


def _row_dicts(cur, rows) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cur.description]
    return [{name: value for name, value in zip(columns, row)} for row in rows]


def _query_hmc(query: str, terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    exact = _extract_hmc(query)
    try:
        with postgres_client.conn.cursor() as cur:
            if exact:
                cur.execute(
                    """
                    SELECT * FROM fault_hmc
                    WHERE lcn=%s AND hamc=%s AND hsc=%s
                    ORDER BY updated_at DESC LIMIT %s
                    """,
                    (exact["lcn"], exact["hamc"], exact["hsc"], limit),
                )
                rows = _row_dicts(cur, cur.fetchall())
                for row in rows:
                    row["match_type"] = "exact_code"
                return rows

            clauses: List[str] = []
            params: List[Any] = []
            for term in terms[:8]:
                pattern = f"%{term}%"
                clauses.append(
                    "(hmc_code ILIKE %s OR lcn ILIKE %s OR hamc ILIKE %s OR hsc ILIKE %s "
                    "OR fault_desc ILIKE %s OR repair_method ILIKE %s)"
                )
                params.extend([pattern] * 6)
            if not clauses:
                return []
            params.append(limit)
            cur.execute(
                f"SELECT * FROM fault_hmc WHERE {' OR '.join(clauses)} ORDER BY updated_at DESC LIMIT %s",
                tuple(params),
            )
            rows = _row_dicts(cur, cur.fetchall())
            for row in rows:
                row["match_type"] = "keyword"
            return rows
    except Exception as exc:
        logger.warning("HMC query failed: %s", exc)
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _query_table(table: str, columns: Sequence[str], terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    if not terms:
        return []
    clauses: List[str] = []
    params: List[Any] = []
    for term in terms[:8]:
        pattern = f"%{term}%"
        clauses.append("(" + " OR ".join(f"CAST({col} AS TEXT) ILIKE %s" for col in columns) + ")")
        params.extend([pattern] * len(columns))
    params.append(limit)
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {table} WHERE {' OR '.join(clauses)} ORDER BY updated_at DESC LIMIT %s",
                tuple(params),
            )
            return _row_dicts(cur, cur.fetchall())
    except Exception as exc:
        logger.warning("ICP table query failed (%s): %s", table, exc)
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _query_fmeca(terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    return _query_table(
        "fmea",
        (
            "code", "product_flag", "func_management", "fault_mode", "fault_cause",
            "mission_phase", "local_impact", "final_impact", "severity", "detect_method",
            "design_improve", "compensate_action",
        ),
        terms,
        limit,
    )


def _query_mfl(terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    return _query_table(
        "mfl",
        (
            "mfl_code", "module_fault", "sur_fault", "subcircuit_fault",
            "api_name", "api_method", "api_usage",
        ),
        terms,
        limit,
    )


def _query_diagrams(doc_type: str, terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    if doc_type not in ("fta", "rbd") or not terms:
        return []
    clauses = []
    params: List[Any] = [doc_type]
    for term in terms[:8]:
        clauses.append("(image_title ILIKE %s OR graph::text ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])
    params.append(limit)
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id AS diagram_id, doc_id, kb_id AS doc_type, image_title, graph, created_at
                FROM document_fault_trees
                WHERE kb_id=%s AND ({' OR '.join(clauses)})
                ORDER BY created_at DESC LIMIT %s
                """,
                tuple(params),
            )
            rows = _row_dicts(cur, cur.fetchall())
            for row in rows:
                graph = row.get("graph")
                if isinstance(graph, str):
                    try:
                        row["graph"] = json.loads(graph)
                    except Exception:
                        pass
            return rows
    except Exception as exc:
        logger.warning("Diagram query failed (%s): %s", doc_type, exc)
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _hybrid_search(state: Dict[str, Any], doc_types: Iterable[str], top_k: int) -> List[Dict[str, Any]]:
    query = state.get("rewritten_query") or state.get("query", "")
    if not query:
        return []
    embedding_model_id = state.get("embedding_model_id") or None
    try:
        results = knowledge_retriever.search_es_hybrid(
            kb_id="",
            kb_ids=list(doc_types),
            query=query,
            top_k=top_k,
            keyword_weight=0.45,
            vector_weight=0.55,
            embedding_model_id=embedding_model_id,
        )
    except Exception as exc:
        logger.warning("ICP hybrid search failed: %s", exc)
        return []

    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in results:
        key = (
            item.get("chunk_id"),
            (item.get("content") or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:top_k]


def _tool_call(name: str, count: int, started: float, preview: List[Dict[str, Any]], **params) -> Dict[str, Any]:
    return {
        "tool": name,
        "count": count,
        "duration": round(time.time() - started, 3),
        "params": params,
        "results_preview": preview[:8],
    }


def _preview_rows(rows: Sequence[Dict[str, Any]], title_keys: Sequence[str], body_keys: Sequence[str]) -> List[Dict[str, Any]]:
    preview = []
    for row in rows[:8]:
        title = next((str(row.get(key) or "").strip() for key in title_keys if row.get(key)), "")
        body = "；".join(str(row.get(key) or "").strip() for key in body_keys if row.get(key))
        preview.append({"title": title[:100], "snippet": body[:260]})
    return preview


def _format_domain_context(
    query: str,
    function_rows: Sequence[Dict[str, Any]],
    hmc_rows: Sequence[Dict[str, Any]],
    fmeca_rows: Sequence[Dict[str, Any]],
    fta_rows: Sequence[Dict[str, Any]],
    rbd_rows: Sequence[Dict[str, Any]],
    mfl_rows: Sequence[Dict[str, Any]],
    history_rows: Sequence[Dict[str, Any]],
    hybrid_rows: Sequence[Dict[str, Any]],
    confidence: float,
    clarification: str,
) -> str:
    sections = [f"【用户故障描述】\n{query}", f"【当前诊断置信度】\n{confidence:.2f}"]

    if function_rows:
        lines = []
        for row in function_rows[:6]:
            lines.append(
                f"- {row.get('function_id', '')} {row.get('function_name', '')}；"
                f"系统/模块：{row.get('system_name', '')}/{row.get('module_name', '')}；"
                f"说明：{row.get('description', '') or row.get('failure_symptoms', '')}"
            )
        sections.append("【Function Locator 候选】\n" + "\n".join(lines))

    if hmc_rows:
        lines = []
        for row in hmc_rows[:8]:
            code = row.get("hmc_code") or f"{row.get('lcn', '')}{row.get('hamc', '')}{row.get('hsc', '')}"
            lines.append(
                f"- HMC {code}：{row.get('fault_desc', '')}；维修：{row.get('repair_method', '')}；"
                f"匹配方式：{row.get('match_type', '结构化查询')}"
            )
        sections.append("【HMC】\n" + "\n".join(lines))

    if fmeca_rows:
        lines = []
        for row in fmeca_rows[:6]:
            lines.append(
                f"- {row.get('code', '')} {row.get('fault_mode', '')}；原因：{row.get('fault_cause', '')}；"
                f"局部影响：{row.get('local_impact', '')}；最终影响：{row.get('final_impact', '')}；"
                f"严酷度：{row.get('severity', '')}；检测：{row.get('detect_method', '')}"
            )
        sections.append("【FMECA】\n" + "\n".join(lines))

    for label, rows in (("FTA", fta_rows), ("RBD", rbd_rows)):
        if not rows:
            continue
        lines = []
        for row in rows[:4]:
            graph = row.get("graph") or {}
            if isinstance(graph, dict):
                vertices = graph.get("vertices", []) or []
                edges = graph.get("edges", []) or []
                names = []
                for vertex in vertices[:10]:
                    props = vertex.get("properties", {}) if isinstance(vertex, dict) else {}
                    names.append(str(props.get("name") or vertex.get("id") or ""))
                summary = f"节点 {len(vertices)}、关系 {len(edges)}；关键节点：{', '.join(names[:6])}"
            else:
                summary = str(graph)[:400]
            lines.append(f"- {row.get('image_title', '')}：{summary}")
        sections.append(f"【{label}】\n" + "\n".join(lines))

    if mfl_rows:
        lines = []
        for row in mfl_rows[:6]:
            lines.append(
                f"- {row.get('mfl_code', '')}：模块故障 {row.get('module_fault', '')}；"
                f"子电路 {row.get('subcircuit_fault', '')}；API {row.get('api_name', '')} {row.get('api_method', '')}；"
                f"用法：{row.get('api_usage', '')}"
            )
        sections.append("【MFL】\n" + "\n".join(lines))

    if history_rows:
        lines = []
        for row in history_rows[:5]:
            lines.append(
                f"- {row.get('case_id', '')}：现象 {row.get('fault_symptom', '')}；"
                f"根因 {row.get('root_cause', '')}；处置 {row.get('final_solution', '')}；"
                f"关联 HMC {row.get('hmc_code', '')}"
            )
        sections.append("【故障历史】\n" + "\n".join(lines))

    if hybrid_rows:
        lines = []
        for row in hybrid_rows[:10]:
            lines.append(
                f"- [{row.get('kb_id') or row.get('doc_type_kwd') or row.get('section', '')}] "
                f"{row.get('title', '')}：{(row.get('content') or '')[:320]}"
            )
        sections.append("【跨七类文档混合召回】\n" + "\n".join(lines))

    if clarification:
        sections.append("【仍需确认】\n" + clarification)
    return "\n\n".join(sections)


def _candidate_faults(
    hmc_rows: Sequence[Dict[str, Any]],
    fmeca_rows: Sequence[Dict[str, Any]],
    hybrid_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "evidence": []})
    for row in hmc_rows:
        name = str(row.get("fault_desc") or row.get("hmc_code") or "").strip()
        if not name:
            continue
        candidates[name]["score"] += 0.55 if row.get("match_type") == "exact_code" else 0.3
        candidates[name]["evidence"].append({"source": "HMC", "id": row.get("hmc_code")})
    for row in fmeca_rows:
        name = str(row.get("fault_mode") or row.get("code") or "").strip()
        if not name:
            continue
        candidates[name]["score"] += 0.25
        candidates[name]["evidence"].append({"source": "FMECA", "id": row.get("code")})
    for row in hybrid_rows[:12]:
        name = str(row.get("title") or "").strip()
        if not name:
            continue
        raw_score = row.get("score", 0)
        try:
            score = min(float(raw_score), 1.0)
        except Exception:
            score = 0.0
        candidates[name]["score"] += 0.1 + 0.15 * max(score, 0)
        candidates[name]["evidence"].append(
            {"source": row.get("kb_id") or row.get("section") or "RAG", "id": row.get("chunk_id")}
        )

    ranked = []
    for name, value in candidates.items():
        ranked.append(
            {
                "fault": name,
                "score": round(min(value["score"], 0.99), 3),
                "evidence": value["evidence"][:6],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:8]


async def icp_diagnosis_node(state) -> Dict[str, Any]:
    """一次完成 ICP 专用多源检索与候选整理。"""
    query = str(state.get("rewritten_query") or state.get("query") or "").strip()
    terms = _unique_terms(state)
    intent = state.get("intent", "symptom_diagnosis")
    limit = max(4, int(state.get("sql_limit", 8) or 8))
    retrieval_top_k = max(8, int(state.get("retrieval_top_k", 12) or 12))

    tool_calls: List[Dict[str, Any]] = []

    started = time.time()
    function_rows = icp_repository.search_functions(terms, limit=limit)
    function_hybrid = _hybrid_search(state, ("function_locator",), top_k=min(limit, 8))
    tool_calls.append(
        _tool_call(
            "function_locator",
            len(function_rows) + len(function_hybrid),
            started,
            _preview_rows(function_rows, ("function_id", "function_name"), ("function_name", "module_name", "description"))
            + _preview_rows(function_hybrid, ("title",), ("content",)),
            intent=intent,
        )
    )

    started = time.time()
    hmc_rows = _query_hmc(query, terms, limit)
    tool_calls.append(
        _tool_call(
            "hmc_query",
            len(hmc_rows),
            started,
            _preview_rows(hmc_rows, ("hmc_code", "hsc"), ("fault_desc", "repair_method")),
            exact=bool(_extract_hmc(query)),
        )
    )

    started = time.time()
    fmeca_rows = _query_fmeca(terms, limit)
    tool_calls.append(
        _tool_call(
            "fmeca_query",
            len(fmeca_rows),
            started,
            _preview_rows(fmeca_rows, ("code", "fault_mode"), ("fault_mode", "fault_cause", "final_impact")),
        )
    )

    started = time.time()
    fta_rows = _query_diagrams("fta", terms, max(4, limit // 2))
    tool_calls.append(
        _tool_call(
            "fta_query",
            len(fta_rows),
            started,
            _preview_rows(fta_rows, ("image_title", "diagram_id"), ("image_title",)),
        )
    )

    started = time.time()
    rbd_rows = _query_diagrams("rbd", terms, max(4, limit // 2))
    tool_calls.append(
        _tool_call(
            "rbd_query",
            len(rbd_rows),
            started,
            _preview_rows(rbd_rows, ("image_title", "diagram_id"), ("image_title",)),
        )
    )

    started = time.time()
    mfl_rows = _query_mfl(terms, limit)
    tool_calls.append(
        _tool_call(
            "mfl_query",
            len(mfl_rows),
            started,
            _preview_rows(mfl_rows, ("mfl_code", "api_name"), ("module_fault", "subcircuit_fault", "api_usage")),
        )
    )

    started = time.time()
    history_rows = icp_repository.search_history(terms, limit=limit)
    history_hybrid = _hybrid_search(state, ("fault_history",), top_k=min(limit, 8))
    tool_calls.append(
        _tool_call(
            "fault_history_query",
            len(history_rows) + len(history_hybrid),
            started,
            _preview_rows(history_rows, ("case_id",), ("fault_symptom", "root_cause", "final_solution"))
            + _preview_rows(history_hybrid, ("title",), ("content",)),
        )
    )

    started = time.time()
    hybrid_rows = _hybrid_search(state, ICP_DOC_TYPES, top_k=retrieval_top_k)
    tool_calls.append(
        _tool_call(
            "hybrid_search",
            len(hybrid_rows),
            started,
            _preview_rows(hybrid_rows, ("title",), ("content",)),
            doc_types=list(ICP_DOC_TYPES),
            top_k=retrieval_top_k,
        )
    )

    candidates = _candidate_faults(hmc_rows, fmeca_rows, hybrid_rows)
    exact_hmc = any(row.get("match_type") == "exact_code" for row in hmc_rows)
    source_count = sum(bool(rows) for rows in (function_rows or function_hybrid, hmc_rows, fmeca_rows, fta_rows, mfl_rows, history_rows or history_hybrid))
    if exact_hmc:
        confidence = 0.98
    elif candidates and source_count >= 4:
        confidence = 0.82
    elif candidates and source_count >= 2:
        confidence = 0.68
    elif hybrid_rows:
        confidence = 0.48
    else:
        confidence = 0.2

    clarification = ""
    if confidence < 0.65:
        clarification = (
            "当前信息不足以唯一定位故障。请补充：①故障发生阶段（上电/初始化/运行中）；"
            "②原始告警文字或 HMC；③是否伴随重启、过温、通信中断或显示异常；"
            "④重新上电后是否恢复。"
        )
    elif len(candidates) >= 2 and candidates[0]["score"] - candidates[1]["score"] < 0.12:
        clarification = (
            "前两项候选接近，请补充故障发生阶段、可复现条件以及是否存在明确告警码，"
            "以区分相邻故障模式。"
        )

    context = _format_domain_context(
        query=query,
        function_rows=function_rows,
        hmc_rows=hmc_rows,
        fmeca_rows=fmeca_rows,
        fta_rows=fta_rows,
        rbd_rows=rbd_rows,
        mfl_rows=mfl_rows,
        history_rows=history_rows,
        hybrid_rows=hybrid_rows,
        confidence=confidence,
        clarification=clarification,
    )

    domain_results: List[Dict[str, Any]] = []
    for row in hybrid_rows[:20]:
        domain_results.append(
            {
                "source_type": row.get("kb_id") or row.get("section") or "document",
                "doc_id": row.get("doc_id", ""),
                "doc_name": row.get("doc_name", ""),
                "chunk_id": row.get("chunk_id", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "score": row.get("score", 0),
            }
        )
    for row in hmc_rows[:8]:
        domain_results.append(
            {
                "source_type": "hmc",
                "doc_id": row.get("doc_id", ""),
                "title": row.get("hmc_code", "HMC"),
                "content": f"{row.get('fault_desc', '')}\n维修方法：{row.get('repair_method', '')}",
                "score": 1.0 if row.get("match_type") == "exact_code" else 0.75,
            }
        )

    tool_calls.append(
        _tool_call(
            "fault_reasoning",
            len(candidates),
            time.time(),
            [{"title": c["fault"], "snippet": f"综合得分 {c['score']}"} for c in candidates],
            confidence=round(confidence, 3),
            clarification_required=bool(clarification),
        )
    )

    return {
        "diagnostic_context": context,
        "candidate_faults": candidates,
        "diagnostic_confidence": round(confidence, 3),
        "clarification_question": clarification,
        "domain_results": domain_results,
        # 兼容旧 Answer/引用逻辑：把跨源结果同时写入 vector_results。
        "vector_results": domain_results,
        "tool_calls": tool_calls,
    }
