"""任务状态与解析/图谱日志查询路由。

通用知识库 CRUD、上传、解析、KG 生成、社区报告与删除级联端点已随旧链路退役；
本文件只保留 ICP 前端仍在复用的任务监控与日志留痕端点（kb_id 槽位实际传 doc_type）。
"""
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core.task_manager import task_manager
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


class TaskStatusResponse(BaseModel):
    task_id: str
    doc_id: str
    file_name: str
    task_type: str
    status: str
    progress: int
    logs: List[str]
    result: Optional[Dict] = None
    error: Optional[str] = None


@router.get("/{kb_id}/documents/{doc_id}/kg-logs")
async def get_kg_logs_by_doc(kb_id: str, doc_id: str):
    """获取文档的知识图谱生成日志。文档/知识库已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_kg_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    doc_name = document["file_name"] if document else (logs[0].get("doc_name") if logs else None)
    kb_name = kb["name"] if kb else (logs[0].get("kb_name") if logs else None)
    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "kb_exists": bool(kb),
        "doc_exists": bool(document),
        "total_logs": len(logs),
        "logs": logs
    }

@router.get("/{kb_id}/documents/{doc_id}/kg-logs/tasks")
async def get_kg_log_tasks_by_doc(kb_id: str, doc_id: str):
    logs = postgres_client.get_kg_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)

    task_summaries = {}
    for log in logs:
        task_id = log["task_id"]
        if task_id not in task_summaries:
            task_summaries[task_id] = {
                "task_id": task_id,
                "start_time": None,
                "end_time": None,
                "status": "running",
                "total_entities": 0,
                "total_relations": 0,
                "total_chunks": 0,
                "log_count": 0,
                "error_count": 0
            }

        task_summary = task_summaries[task_id]
        task_summary["log_count"] += 1

        if log["log_level"] == "ERROR":
            task_summary["error_count"] += 1

        if log["entity_count"]:
            task_summary["total_entities"] += log["entity_count"]
        if log["relation_count"]:
            task_summary["total_relations"] += log["relation_count"]

        if log["log_type"] == "chunk_process":
            task_summary["total_chunks"] += 1

        if log["log_type"] == "finish":
            task_summary["status"] = "completed"
        elif log["log_type"] == "cancel":
            task_summary["status"] = "cancelled"
        elif log["log_type"] == "error":
            task_summary["status"] = "failed"

        if not task_summary["start_time"] or (log["created_at"] and log["created_at"] < task_summary["start_time"]):
            task_summary["start_time"] = log["created_at"]

        if not task_summary["end_time"] or (log["created_at"] and log["created_at"] > task_summary["end_time"]):
            task_summary["end_time"] = log["created_at"]

    sorted_tasks = sorted(task_summaries.values(), key=lambda x: x["start_time"], reverse=True)

    return {
        "doc_id": doc_id,
        "doc_name": document["file_name"] if document else (logs[0].get("doc_name") if logs else None),
        "kb_id": kb_id,
        "total_tasks": len(sorted_tasks),
        "tasks": sorted_tasks
    }

@router.get("/{kb_id}/documents/{doc_id}/kg-logs/{task_id}")
async def get_kg_logs_by_task(kb_id: str, doc_id: str, task_id: str):
    """获取某次知识图谱生成任务的日志。文档/知识库已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_kg_logs_by_task(task_id)
    summary = postgres_client.get_kg_log_summary_by_task(task_id)
    document = postgres_client.get_document(doc_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    doc_name = document["file_name"] if document else (logs[0].get("doc_name") if logs else None)
    kb_name = kb["name"] if kb else (logs[0].get("kb_name") if logs else None)
    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "kb_exists": bool(kb),
        "doc_exists": bool(document),
        "summary": summary,
        "total_logs": len(logs),
        "logs": logs
    }

@router.get("/{kb_id}/documents/{doc_id}/parse-logs")
async def get_parse_logs_by_doc(kb_id: str, doc_id: str):
    """获取文档的解析日志。文档/知识库已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_parse_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    doc_name = document["file_name"] if document else (logs[0].get("doc_name") if logs else None)
    kb_name = kb["name"] if kb else (logs[0].get("kb_name") if logs else None)
    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "kb_exists": bool(kb),
        "doc_exists": bool(document),
        "total_logs": len(logs),
        "logs": logs
    }

@router.get("/{kb_id}/documents/{doc_id}/parse-logs/tasks")
async def get_parse_log_tasks_by_doc(kb_id: str, doc_id: str):
    logs = postgres_client.get_parse_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)

    task_summaries = {}
    for log in logs:
        task_id = log["task_id"]
        if task_id not in task_summaries:
            task_summaries[task_id] = {
                "task_id": task_id,
                "start_time": None,
                "end_time": None,
                "status": "running",
                "total_blocks": 0,
                "total_images": 0,
                "total_tables": 0,
                "total_chunks": 0,
                "log_count": 0,
                "error_count": 0
            }

        task_summary = task_summaries[task_id]
        task_summary["log_count"] += 1

        if log["log_level"] == "ERROR":
            task_summary["error_count"] += 1

        if log["block_count"]:
            task_summary["total_blocks"] = max(task_summary["total_blocks"], log["block_count"])
        if log["image_count"]:
            task_summary["total_images"] = max(task_summary["total_images"], log["image_count"])
        if log["table_count"]:
            task_summary["total_tables"] = max(task_summary["total_tables"], log["table_count"])
        if log["chunk_count"]:
            task_summary["total_chunks"] = max(task_summary["total_chunks"], log["chunk_count"])

        if log["log_type"] == "finish":
            task_summary["status"] = "completed"
        elif log["log_type"] == "cancel":
            task_summary["status"] = "cancelled"
        elif log["log_type"] == "error":
            task_summary["status"] = "failed"

        if not task_summary["start_time"] or (log["created_at"] and log["created_at"] < task_summary["start_time"]):
            task_summary["start_time"] = log["created_at"]

        if not task_summary["end_time"] or (log["created_at"] and log["created_at"] > task_summary["end_time"]):
            task_summary["end_time"] = log["created_at"]

    sorted_tasks = sorted(task_summaries.values(), key=lambda x: x["start_time"], reverse=True)

    return {
        "doc_id": doc_id,
        "doc_name": document["file_name"] if document else (logs[0].get("doc_name") if logs else None),
        "kb_id": kb_id,
        "total_tasks": len(sorted_tasks),
        "tasks": sorted_tasks
    }

@router.get("/{kb_id}/documents/{doc_id}/parse-logs/{task_id}")
async def get_parse_logs_by_task(kb_id: str, doc_id: str, task_id: str):
    """获取某次解析任务的日志。文档/知识库已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_parse_logs_by_task(task_id)
    summary = postgres_client.get_parse_log_summary_by_task(task_id)
    document = postgres_client.get_document(doc_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    doc_name = document["file_name"] if document else (logs[0].get("doc_name") if logs else None)
    kb_name = kb["name"] if kb else (logs[0].get("kb_name") if logs else None)
    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "kb_exists": bool(kb),
        "doc_exists": bool(document),
        "summary": summary,
        "total_logs": len(logs),
        "logs": logs
    }

@router.get("/logs/tasks")
async def get_all_task_summaries_global():
    task_summaries = postgres_client.get_all_task_summaries(None)
    # 名字优先用日志快照（文档已删除时仍可显示），并补充存在性标记
    for summary in task_summaries:
        doc = postgres_client.get_document(summary["doc_id"]) if summary.get("doc_id") else None
        kb = postgres_client.get_knowledge_base(summary["kb_id"]) if summary.get("kb_id") else None
        summary["doc_name"] = doc["file_name"] if doc else (summary.get("doc_name") or "未知文档")
        summary["kb_name"] = kb["name"] if kb else (summary.get("kb_name") or "未知知识库")
        summary["doc_exists"] = bool(doc)
        summary["kb_exists"] = bool(kb)
    return {
        "kb_id": None,
        "total_tasks": len(task_summaries),
        "tasks": task_summaries
    }

@router.get("/{kb_id}/logs/tasks")
async def get_all_task_summaries(kb_id: str):
    task_summaries = postgres_client.get_all_task_summaries(kb_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    for summary in task_summaries:
        doc = postgres_client.get_document(summary["doc_id"]) if summary.get("doc_id") else None
        summary["doc_name"] = doc["file_name"] if doc else (summary.get("doc_name") or "未知文档")
        summary["kb_name"] = kb["name"] if kb else (summary.get("kb_name") or "未知知识库")
        summary["doc_exists"] = bool(doc)
        summary["kb_exists"] = bool(kb)
    return {
        "kb_id": kb_id,
        "kb_name": kb["name"] if kb else None,
        "kb_exists": bool(kb),
        "total_tasks": len(task_summaries),
        "tasks": task_summaries
    }

@router.get("/{kb_id}/documents/{doc_id}/logs/tasks")
async def get_doc_task_summaries(kb_id: str, doc_id: str):
    parse_logs = postgres_client.get_parse_logs_by_doc(doc_id)
    kg_logs = postgres_client.get_kg_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)

    task_summaries = {}

    for log in parse_logs:
        task_id = log["task_id"]
        if task_id not in task_summaries:
            task_summaries[task_id] = {
                "task_type": "parse",
                "task_id": task_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "progress": 0,
                "has_error": False,
                "is_finished": False,
                "is_cancelled": False,
                "start_time": None,
                "end_time": None,
                "status": "running"
            }

        task = task_summaries[task_id]
        task["progress"] = max(task["progress"], log.get("progress", 0))
        if log["log_level"] == "ERROR":
            task["has_error"] = True
        if log["log_type"] in ("finish", "cancel", "error"):
            task["is_finished"] = True
        if log["log_type"] == "cancel":
            task["is_cancelled"] = True

        if not task["start_time"] or (log["created_at"] and log["created_at"] < task["start_time"]):
            task["start_time"] = log["created_at"]
        if not task["end_time"] or (log["created_at"] and log["created_at"] > task["end_time"]):
            task["end_time"] = log["created_at"]

    for log in kg_logs:
        task_id = log["task_id"]
        if task_id not in task_summaries:
            task_summaries[task_id] = {
                "task_type": "kg",
                "task_id": task_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "progress": 0,
                "has_error": False,
                "is_finished": False,
                "is_cancelled": False,
                "start_time": None,
                "end_time": None,
                "status": "running"
            }

        task = task_summaries[task_id]
        task["progress"] = max(task["progress"], log.get("progress", 0))
        if log["log_level"] == "ERROR":
            task["has_error"] = True
        if log["log_type"] in ("finish", "cancel", "error"):
            task["is_finished"] = True
        if log["log_type"] == "cancel":
            task["is_cancelled"] = True

        if not task["start_time"] or (log["created_at"] and log["created_at"] < task["start_time"]):
            task["start_time"] = log["created_at"]
        if not task["end_time"] or (log["created_at"] and log["created_at"] > task["end_time"]):
            task["end_time"] = log["created_at"]

    for task in task_summaries.values():
        if task["is_finished"]:
            if task.get("is_cancelled"):
                task["status"] = "cancelled"
            elif task["has_error"]:
                task["status"] = "failed"
            else:
                task["status"] = "completed"
        elif task["progress"] >= 100:
            task["status"] = "completed"

    sorted_tasks = sorted(task_summaries.values(), key=lambda x: x["start_time"], reverse=True)

    return {
        "doc_id": doc_id,
        "doc_name": document["file_name"] if document else None,
        "kb_id": kb_id,
        "total_tasks": len(sorted_tasks),
        "tasks": sorted_tasks
    }


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "doc_id": task.doc_id,
        "file_name": task.file_name,
        "task_type": task.task_type,
        "status": task.status,
        "progress": task.progress,
        "logs": task.logs,
        "result": task.result,
        "error": task.error
    }

@router.get("/tasks/{task_id}/stream")
async def stream_task_events(task_id: str):
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        import json
        async for event in task_manager.stream_events(task_id):
            data = json.dumps(event)
            yield f"data: {data}\n\n"

        final_task = await task_manager.get_task(task_id)
        if final_task:
            final_data = json.dumps({
                'type': 'final',
                'status': final_task.status,
                'progress': final_task.progress,
                'result': final_task.result,
                'error': final_task.error
            })
            yield f"data: {final_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Task is already {task.status}")

    success = await task_manager.cancel_task(task_id)
    if success:
        return {"success": True, "message": "Task cancellation requested", "task_id": task_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel task")
