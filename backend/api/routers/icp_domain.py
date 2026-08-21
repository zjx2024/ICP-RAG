"""ICP 专用结构化数据与可靠性计算接口。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException, Query

from backend.icp.repository import icp_repository
from backend.icp.reliability import ReliabilityModelError, calculate_fta, calculate_rbd
from backend.storage.postgres_client import postgres_client

router = APIRouter(prefix="/icp-domain", tags=["ICP 故障诊断领域"])


@router.get("/functions/{doc_id}")
async def list_functions(
    doc_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    keyword: str = "",
):
    return icp_repository.get_functions_by_doc(doc_id, page, size, keyword)


@router.get("/fault-history/{doc_id}")
async def list_fault_history(
    doc_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    keyword: str = "",
):
    return icp_repository.get_history_by_doc(doc_id, page, size, keyword)


def _record_count(cur, doc_type: str) -> int:
    """返回各类型真正的结构化记录数，而不是通用 document_chunks 近似值。"""
    if doc_type == "hmc":
        cur.execute("SELECT COUNT(*) FROM fault_hmc")
    elif doc_type == "fmeca":
        cur.execute("SELECT COUNT(*) FROM fmea")
    elif doc_type == "mfl":
        cur.execute("SELECT COUNT(*) FROM mfl")
    elif doc_type in ("fta", "rbd"):
        cur.execute("SELECT COUNT(*) FROM document_fault_trees WHERE kb_id=%s", (doc_type,))
    elif doc_type == "function_locator":
        cur.execute("SELECT COUNT(*) FROM icp_functions")
    elif doc_type == "fault_history":
        cur.execute("SELECT COUNT(*) FROM fault_history_cases")
    else:
        return 0
    return int(cur.fetchone()[0])


@router.get("/readiness")
async def diagnosis_readiness():
    """返回七类文档的上传、可用状态和结构化记录数。"""
    rows = []
    with postgres_client.conn.cursor() as cur:
        cur.execute(
            """
            SELECT dt.doc_type, dt.name,
                   COUNT(d.doc_id) AS doc_count,
                   COUNT(d.doc_id) FILTER (WHERE d.status IN ('indexed','kg_pending','kg_done')) AS ready_count,
                   COUNT(d.doc_id) FILTER (WHERE d.status='failed') AS failed_count
            FROM document_types dt
            LEFT JOIN icp_documents d ON d.doc_type=dt.doc_type
            WHERE dt.is_active=TRUE
            GROUP BY dt.doc_type, dt.name, dt.sort_order
            ORDER BY dt.sort_order
            """
        )
        base_rows = cur.fetchall()
        for row in base_rows:
            rows.append({
                "doc_type": row[0],
                "name": row[1],
                "doc_count": row[2],
                "ready_count": row[3],
                "failed_count": row[4],
                "record_count": _record_count(cur, row[0]),
            })
    return {
        "types": rows,
        "all_types_present": len(rows) == 7 and all(item["ready_count"] > 0 for item in rows),
        "ready_type_count": sum(1 for item in rows if item["ready_count"] > 0),
        "total_records": sum(item["record_count"] for item in rows),
    }


def _load_diagram(diagram_id: str, expected_type: str):
    with postgres_client.conn.cursor() as cur:
        cur.execute(
            "SELECT id, doc_id, kb_id, image_title, graph FROM document_fault_trees WHERE id=%s",
            (diagram_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="图模型不存在")
    if row[2] != expected_type:
        raise HTTPException(status_code=400, detail=f"图模型不是 {expected_type.upper()} 类型")
    graph = row[4]
    if isinstance(graph, str):
        try:
            graph = json.loads(graph)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"图模型 JSON 无法解析: {exc}") from exc
    return {"id": row[0], "doc_id": row[1], "title": row[3], "graph": graph or {}}


@router.post("/fta/{diagram_id}/calculate")
async def calculate_fta_model(diagram_id: str, request: dict = Body(default={})):
    item = _load_diagram(diagram_id, "fta")
    try:
        result = calculate_fta(item["graph"], mission_time=request.get("mission_time"))
    except ReliabilityModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "diagram_id": item["id"], "doc_id": item["doc_id"], "title": item["title"]}


@router.post("/rbd/{diagram_id}/calculate")
async def calculate_rbd_model(diagram_id: str, request: dict = Body(default={})):
    item = _load_diagram(diagram_id, "rbd")
    try:
        result = calculate_rbd(
            item["graph"],
            mission_time=request.get("mission_time"),
            max_paths=int(request.get("max_paths", 12)),
        )
    except ReliabilityModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "diagram_id": item["id"], "doc_id": item["doc_id"], "title": item["title"]}
