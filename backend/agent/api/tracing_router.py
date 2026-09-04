"""B1 全链路 tracing 查询 API（JSON-lines 落盘回放与聚合统计）。"""
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from backend.agent.tracing import aggregate_stats, get_trace, list_traces

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/traces")
async def api_list_traces(limit: int = 20) -> List[dict]:
    """S 最近 trace 摘要列表（按时间倒序）。"""
    return list_traces(limit=limit)


@router.get("/traces/stats")
async def api_trace_aggregate() -> dict:
    """进程级聚合：平均模型调用次数、LLM 失败率、三类降级率。"""
    return aggregate_stats()


@router.get("/traces/{trace_id}")
async def api_get_trace(trace_id: str) -> dict:
    """按 trace_id 回放完整事件（summary 首行 + JSON-lines 事件）。"""
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(404, "trace 不存在")
    return trace