"""sql_query 工具节点。

设计原则：只读复用现有 postgres_client 连接，仅 SELECT 现有 images/tables 表。
"""
import time
import logging
from typing import Dict

from backend.storage.postgres_client import postgres_client  # 只读引用，不修改

logger = logging.getLogger(__name__)

# 仅允许查询的现有表（白名单，只读）
ALLOWED_TABLES = {"document_images", "document_tables", "documents", "document_chunks",
                  "fault_hmc", "fmea", "mfl", "icp_documents"}


def _is_safe_sql(sql: str) -> bool:
    """安全检查：仅允许 SELECT，仅允许白名单表"""
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    # 禁止写操作
    for kw in ("insert", "update", "delete", "drop", "alter", "truncate", "create"):
        if kw in s:
            return False
    # 必须包含至少一个白名单表
    return any(t in s for t in ALLOWED_TABLES)


def _inject_is_deleted(sql: str) -> str:
    """为 documents 查询自动注入 is_deleted = FALSE（复用现有教训）"""
    s = sql.lower()
    if "documents" in s and "is_deleted" not in s:
        # 简单注入 WHERE 条件
        if "where" in s:
            sql = sql.replace("WHERE", "WHERE is_deleted = FALSE AND", 1)
            sql = sql.replace("where", "WHERE is_deleted = FALSE AND", 1)
        else:
            # 在 ORDER BY 或末尾前插入
            for kw in (" order by", " limit", " group by"):
                if kw in s:
                    sql = sql.replace(kw, f" WHERE is_deleted = FALSE{kw}", 1)
                    break
            else:
                sql = sql.rstrip(";") + " WHERE is_deleted = FALSE"
    return sql


async def sql_query_node(state) -> Dict:
    """查询 PG 图片/表格元数据，只读复用现有 postgres_client 连接"""
    query = state.get("rewritten_query") or state.get("query", "")
    # 关键词检测同时检查原始 query 和 rewritten_query（rewritten 可能丢失关键词）
    original_query = state.get("query", "")
    combined_query = f"{original_query} {query}".lower()
    kb_id = state.get("kb_id", "")
    # 多库关联：kb_ids 优先（可能多个知识库），无则回退 kb_id 单库
    kb_ids = state.get("kb_ids") or ([kb_id] if kb_id else [])
    # 数据库查询返回条数：读取 agent 配置的 sql_limit
    limit = int(state.get("sql_limit", 5) or 5)

    start = time.time()
    results = []
    error = None

    try:
        # 判断是否是图片相关请求（检查原始+改写后的 query）
        image_keywords = ("图片", "image", "放映", "照片", "截图", "看图")
        want_images = any(kw in combined_query for kw in image_keywords)

        with postgres_client.conn.cursor() as cur:
            image_fallback = False
            if want_images:
                # 用户想看图片：优先按查询内容（剔除"图片"等噪音词）匹配标题/描述，
                # 避免直接返回知识库任意图片导致"图片全是错的"
                search_term = query
                for kw in ("图片", "image", "放映", "照片", "截图", "看图"):
                    search_term = search_term.replace(kw, "")
                search_term = search_term.strip()
                img_rows = []
                if search_term:
                    cur.execute("""
                        SELECT di.image_id, di.doc_id, di.title, di.description, di.image_type, di.image_url,
                               d.file_name as doc_name
                        FROM document_images di
                        JOIN documents d ON di.doc_id = d.doc_id
                        WHERE d.kb_id = ANY(%s) AND d.is_deleted = FALSE
                          AND (di.title ILIKE %s OR di.description ILIKE %s)
                        LIMIT %s
                    """, (kb_ids, f"%{search_term}%", f"%{search_term}%", limit))
                    img_rows = cur.fetchall()
                if not img_rows:
                    # 回退：仅"看图"意图或按查询词未匹配到图片时，展示该知识库图片供参考（标注兜底）
                    image_fallback = True
                    cur.execute("""
                        SELECT di.image_id, di.doc_id, di.title, di.description, di.image_type, di.image_url,
                               d.file_name as doc_name
                        FROM document_images di
                        JOIN documents d ON di.doc_id = d.doc_id
                        WHERE d.kb_id = ANY(%s) AND d.is_deleted = FALSE
                        LIMIT %s
                    """, (kb_ids, limit))
                    img_rows = cur.fetchall()
            else:
                # 精确匹配：基于查询词查文档图片和表格的标题/描述
                cur.execute("""
                    SELECT di.image_id, di.doc_id, di.title, di.description, di.image_type, di.image_url,
                           d.file_name as doc_name
                    FROM document_images di
                    JOIN documents d ON di.doc_id = d.doc_id
                    WHERE d.kb_id = ANY(%s) AND d.is_deleted = FALSE
                      AND (di.title ILIKE %s OR di.description ILIKE %s)
                    LIMIT %s
                """, (kb_ids, f"%{query}%", f"%{query}%", limit))
            img_rows = cur.fetchall()
            for row in img_rows:
                results.append({
                    "type": "image",
                    "image_id": row[0], "doc_id": row[1], "title": row[2],
                    "description": row[3], "image_type": row[4], "image_url": row[5],
                    "doc_name": row[6],
                })

            # 查表格（只读）：搜索 title/description/meta_data
            table_keywords = ("表格", "数据表", "参数表", "标准表", "原始数据", "table", "查看表", "看看表")
            want_tables = any(kw in combined_query for kw in table_keywords)

            if want_tables:
                # 用户想看表格：直接返回知识库中的表格
                cur.execute("""
                    SELECT dt.table_id, dt.doc_id, dt.title, dt.description, dt.meta_data,
                           d.file_name as doc_name
                    FROM document_tables dt
                    JOIN documents d ON dt.doc_id = d.doc_id
                    WHERE d.kb_id = ANY(%s) AND d.is_deleted = FALSE
                    LIMIT %s
                """, (kb_ids, limit))
            else:
                # 精确匹配：搜索 title/description/meta_data（meta_data 转文本搜索）
                cur.execute("""
                    SELECT dt.table_id, dt.doc_id, dt.title, dt.description, dt.meta_data,
                           d.file_name as doc_name
                    FROM document_tables dt
                    JOIN documents d ON dt.doc_id = d.doc_id
                    WHERE d.kb_id = ANY(%s) AND d.is_deleted = FALSE
                      AND (dt.title ILIKE %s OR dt.description ILIKE %s
                           OR dt.meta_data::text ILIKE %s)
                    LIMIT %s
                """, (kb_ids, f"%{query}%", f"%{query}%", f"%{query}%", limit))
            tbl_rows = cur.fetchall()
            for row in tbl_rows:
                results.append({
                    "type": "table",
                    "table_id": row[0], "doc_id": row[1], "title": row[2],
                    "description": row[3], "meta_data": row[4], "doc_name": row[5],
                })

            # ICP 真源表（hmc/fmeca/mfl）：按关键词路由到对应表做全列检索
            icp_routes = [
                ("hmc", ("hmc", "故障码", "维修码")),
                ("fmeca", ("fmeca", "fmea", "故障影响", "严酷度", "故障模式")),
                ("mfl", ("mfl", "清单", "api上报")),
            ]
            for icp_table, kws in icp_routes:
                if not any(kw in combined_query for kw in kws):
                    continue
                search_cols = {
                    "hmc": ["hmc_code", "lcn", "hamc", "hsc", "fault_desc", "repair_method"],
                    "fmeca": ["code", "product_flag", "func_management", "fault_mode", "fault_cause",
                              "mission_phase", "local_impact", "final_impact", "detect_method",
                              "design_improve", "compensate_action"],
                    "mfl": ["mfl_code", "module_fault", "sur_fault", "subcircuit_fault",
                            "api_name", "api_method", "api_usage"],
                }[icp_table]
                like_clause = " OR ".join(
                    f"CAST({c} AS TEXT) ILIKE %s" for c in search_cols)
                cur.execute(f"""
                    SELECT * FROM {icp_table}
                    WHERE {like_clause}
                    LIMIT %s
                """, tuple(f"%{query}%" for _ in search_cols) + (limit,))
                cols = [d[0] for d in cur.description]
                icp_rows = cur.fetchall()
                for row in icp_rows:
                    results.append({
                        "type": "icp_" + icp_table,
                        "table": icp_table,
                        "data": {cols[i]: row[i] for i in range(len(cols))},
                    })
    except Exception as e:
        logger.error(f"sql_query 调用失败: {e}")
        error = str(e)

    duration = round(time.time() - start, 2)
    # 构建结果预览：让用户看到实际查到了哪些图片/表格
    results_preview = []
    for r in results[:5]:
        if r.get("type") == "image":
            results_preview.append({
                "type": "image",
                "title": (r.get("title") or "").strip()[:80],
                "doc_name": (r.get("doc_name") or "").strip()[:60],
                "has_url": bool(r.get("image_url")),
            })
        elif r.get("type") == "table":
            # 表格数据预览：取前 3 行
            meta = r.get("meta_data")
            table_rows = []
            if isinstance(meta, str):
                try:
                    import json as _json
                    meta = _json.loads(meta)
                except Exception:
                    meta = None
            if isinstance(meta, dict):
                data = meta.get("data", [])
                if isinstance(data, list):
                    for row in data[:3]:
                        if isinstance(row, list):
                            table_rows.append([str(c)[:30] for c in row])
            results_preview.append({
                "type": "table",
                "title": (r.get("title") or "").strip()[:80],
                "doc_name": (r.get("doc_name") or "").strip()[:60],
                "rows_total": len(meta.get("data", [])) if isinstance(meta, dict) and isinstance(meta.get("data"), list) else 0,
                "rows_preview": table_rows,
            })
    tool_call = {
        "tool": "sql_query",
        "query": query,
        "count": len(results),
        "duration": duration,
        "want_images": want_images,
        "want_tables": want_tables,
        "image_fallback": image_fallback,
        "results_preview": results_preview,
    }
    if error:
        tool_call["error"] = error
    return {
        "sql_results": results,
        "tool_calls": [tool_call],
    }
