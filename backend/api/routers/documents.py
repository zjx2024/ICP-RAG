import os
import uuid
import json
import asyncio
import logging
import concurrent.futures
import networkx as nx
from datetime import datetime
from functools import partial
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.llm_service import llm_service
from backend.core.task_manager import task_manager
from backend.graphrag.graph_extractor import GraphExtractor, doc_id_to_str
from backend.storage.search_store import search_store
from backend.storage.artifact_store import (
    artifact_store,
    safe_original_name,
    source_artifact_key,
    storage_key_from,
)
from backend.storage.postgres_client import postgres_client
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.deletion_logger import DeletionLogger
from backend.api.routers._deletion_helpers import (
    step_index, parse_json_parts, handle_community_reports,
)
from backend.storage.registration import confirm_registration_or_compensate

from backend.services.model_service import (
    resolve_llm_model_config, require_embedding_model_configured,
)
from backend.services.document_service import (
    llm_semaphore, convert_fault_tree_to_hugegraph, parse_document_background,
)
from backend.services.kg_service import generate_kg_background

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


def _hit_document(hit):
    """兼容 SearchHit 对象与测试 double 的投影文档取值。"""
    if hasattr(hit, "document"):
        return hit.document
    if isinstance(hit, dict):
        return hit.get("document") or hit.get("_source") or hit
    return {}


def _hit_id(hit):
    if hasattr(hit, "id"):
        return hit.id
    if isinstance(hit, dict):
        return hit.get("id") or hit.get("_id") or (_hit_document(hit).get("chunk_id"))
    return None


def _all_projection_hits(filters, batch_size=1000):
    """分页读取检索投影，避免路由层依赖 OpenSearch 原始 DSL。"""
    hits = []
    offset = 0
    while True:
        batch = search_store.query(filters=filters, limit=batch_size, offset=offset)
        hits.extend(batch)
        if len(batch) < batch_size:
            return hits
        offset += len(batch)


def _delete_non_report_projection(doc_id, kb_id=None):
    filters = {"doc_id": doc_id}
    if kb_id:
        filters["kb_id"] = kb_id
    projection_ids = []
    for hit in _all_projection_hits(filters):
        source = _hit_document(hit)
        if source.get("knowledge_graph_kwd") == "community_report":
            continue
        projection_id = _hit_id(hit)
        if projection_id:
            projection_ids.append(projection_id)
    return sum(1 for projection_id in projection_ids if search_store.delete_by_id(projection_id))



class KnowledgeBaseResponse(BaseModel):
    kb_id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

class CreateKBRequest(BaseModel):
    name: str
    description: Optional[str] = None

class UploadResponse(BaseModel):
    doc_id: str
    file_name: str
    status: str
    storage_key: Optional[str] = None

class DocumentStatusResponse(BaseModel):
    doc_id: str
    file_name: str
    kb_id: str
    status: str
    storage_key: Optional[str] = None
    minio_path: Optional[str] = None  # Deprecated compatibility field.
    total_chunks: int
    created_at: datetime

class ParseResponse(BaseModel):
    task_id: str
    doc_id: str
    file_name: str
    status: str

class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    title: str
    content: str
    content_type: str
    chunk_index: int
    selected_for_kg: bool
    images: List[dict]
    tables: List[dict]

class KGResponse(BaseModel):
    task_id: str
    doc_id: str
    status: str

class ChunkSelectionRequest(BaseModel):
    chunk_ids: List[str]
    selected: bool

class GenerateKGRequest(BaseModel):
    selected_chunk_ids: List[str]
    model_id: Optional[str] = None
    mode: Optional[str] = "enhanced"
    entity_resolution: Optional[bool] = True
    with_community: Optional[bool] = True

class ParseRequest(BaseModel):
    model_id: Optional[str] = None

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




@router.post("/", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(request: CreateKBRequest):
    kb_id = str(uuid.uuid4())
    postgres_client.create_knowledge_base(kb_id, request.name, request.description)
    
    kb = postgres_client.get_knowledge_base(kb_id)
    if kb:
        return kb
    raise HTTPException(status_code=500, detail="Failed to create knowledge base")

@router.get("/", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases():
    return postgres_client.list_knowledge_bases()

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb

@router.put("/{kb_id}/", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(kb_id: str, request: CreateKBRequest):
    """更新知识库名称/描述（改），支持部分更新（只传需修改的字段）。"""
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    updated = postgres_client.update_knowledge_base(
        kb_id,
        name=request.name or None,
        description=request.description,
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update knowledge base")
    return postgres_client.get_knowledge_base(kb_id)

@router.delete("/{kb_id}/")
async def delete_knowledge_base(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    documents = postgres_client.get_documents_by_kb(kb_id) or []
    try:
        search_store.delete_by_filter({"kb_id": kb_id})

        hugegraph_client.delete_by_kb_id(kb_id)

        if not postgres_client.delete_knowledge_base_with_documents(kb_id):
            raise RuntimeError("Knowledge base no longer exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # The PG truth is gone before file cleanup.  Failures here produce
    # auditable orphans instead of visible rows pointing at missing originals.
    cleanup_errors = []
    for document in documents:
        try:
            if document.get("storage_key") or document.get("minio_path"):
                artifact_store.delete_file(storage_key_from(document))
        except Exception as exc:
            cleanup_errors.append(f"{document.get('doc_id')}: 原件回收失败: {exc}")
        try:
            artifact_store.delete_prefix(f"images/{document['doc_id']}")
        except Exception as exc:
            cleanup_errors.append(f"{document.get('doc_id')}: 图片回收失败: {exc}")
    try:
        artifact_store.delete_prefix(f"documents/{kb_id}")
    except Exception as exc:
        cleanup_errors.append(f"知识库原件目录回收失败: {exc}")

    return {
        "status": "success_with_warnings" if cleanup_errors else "success",
        "message": f"Knowledge base {kb_id} and all related data deleted",
        "cleanup_errors": cleanup_errors,
    }

@router.post("/{kb_id}/documents/dedup-index")
async def dedup_index(kb_id: str):
    """清除检索投影中重复解析产生的冗余分块。

    以 PG documents 表为有效依据，删除 OpenSearch 中该知识库下不属于任何有效文档的
    doc_id 的全部分块（历史重复解析产生的多份索引）。
    社区报告（knowledge_graph_kwd=community_report）按字段排除，不会误删。
    """
    try:
        # PG 有效 doc_id（数据唯一权威）
        docs = postgres_client.get_documents_by_kb(kb_id) or []
        valid_doc_ids = {d["doc_id"] for d in docs if d.get("doc_id")}

        # 通过门面分页扫描，不在 API 层拼接 OpenSearch DSL。
        orphan_ids = {}
        for hit in _all_projection_hits({"kb_id": kb_id}):
            source = _hit_document(hit)
            if source.get("knowledge_graph_kwd") == "community_report":
                continue
            doc_id = source.get("doc_id")
            projection_id = _hit_id(hit)
            if doc_id and doc_id not in valid_doc_ids and projection_id:
                orphan_ids.setdefault(doc_id, []).append(projection_id)

        total = 0
        details = []
        for orphan_doc_id, projection_ids in orphan_ids.items():
            cnt = sum(
                1 for projection_id in projection_ids if search_store.delete_by_id(projection_id)
            )
            total += cnt
            details.append({"doc_id": orphan_doc_id, "deleted": cnt})

        if total:
            return {"total_deleted": total, "details": details,
                    "message": f"已清除 {total} 条重复索引块"}
        return {"total_deleted": 0, "details": [],
                "message": "没有发现重复索引块（检索投影正常）"}
    except Exception as e:
        logger.error(f"清除重复索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除重复索引失败: {str(e)}")


@router.post("/{kb_id}/documents/upload", response_model=list[UploadResponse])
async def upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = []
    
    for file in files:
        doc_id = str(uuid.uuid4())
        file_name = safe_original_name(file.filename)
        storage_key = source_artifact_key(f"documents/{kb_id}", doc_id, file_name)
        try:
            metadata = await asyncio.to_thread(
                artifact_store.upload_file,
                storage_key,
                file.file,
                content_type=file.content_type,
                original_name=file_name,
            )
            try:
                # The legacy minio_path column is retained as a storage_key
                # compatibility column during the LocalFS transition.
                postgres_client.insert_document(doc_id, file_name, kb_id, storage_key)
            except Exception as insert_error:
                committed = confirm_registration_or_compensate(
                    doc_id=doc_id,
                    storage_key=storage_key,
                    lookup=postgres_client.get_document,
                    rollback=postgres_client.conn.rollback,
                )
                if not committed:
                    raise insert_error
        except Exception as exc:
            logger.exception("文档写入 ArtifactStore 失败: %s", file_name)
            raise HTTPException(status_code=500, detail=f"文件 {file_name} 保存失败: {exc}") from exc

        results.append({
            "doc_id": doc_id,
            "file_name": file_name,
            "status": "uploaded",
            "storage_key": metadata.storage_key,
        })
    
    return results

@router.get("/{kb_id}/documents", response_model=list[DocumentStatusResponse])
async def list_documents(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    documents = postgres_client.get_documents_by_kb(kb_id)
    return documents

@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentStatusResponse)
async def get_document(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    return document

@router.post("/{kb_id}/documents/{doc_id}/parse", response_model=ParseResponse)
async def parse_document(kb_id: str, doc_id: str, request: ParseRequest = Body(default=ParseRequest())):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    if document["status"] not in ["uploaded", "failed"]:
        raise HTTPException(status_code=400, detail="Document already parsed or processing")

    # 预检查：解析同时需要 Chat(LLM) 模型（图片/表格分析）与 Embedding 模型（向量入库）
    # 统一走 llm_models 表，未配置或类型不匹配时返回明确错误
    model_config = resolve_llm_model_config(request.model_id)
    require_embedding_model_configured()

    postgres_client.update_document_status(doc_id, "parsing")
    
    task_id = await task_manager.create_task(
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=document["file_name"],
        task_type="parse"
    )
    
    asyncio.create_task(parse_document_background(task_id, kb_id, doc_id, model_config))
    
    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "file_name": document["file_name"],
        "status": "parsing"
    }

@router.post("/{kb_id}/documents/batch-parse", response_model=list[ParseResponse])
async def batch_parse_documents(kb_id: str, doc_ids: List[str] = Body(...), model_id: Optional[str] = None):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 预检查：批量解析同样需要 Chat(LLM) + Embedding 模型
    model_config = resolve_llm_model_config(model_id)
    require_embedding_model_configured()

    results = []
    
    for doc_id in doc_ids:
        document = postgres_client.get_document(doc_id)
        if not document:
            continue
        
        if document["kb_id"] != kb_id:
            continue
        
        if document["status"] not in ["uploaded", "failed"]:
            continue
        
        postgres_client.update_document_status(doc_id, "parsing")
        
        task_id = await task_manager.create_task(
            doc_id=doc_id,
            kb_id=kb_id,
            file_name=document["file_name"],
            task_type="parse"
        )
        
        asyncio.create_task(parse_document_background(task_id, kb_id, doc_id, model_config))
        
        results.append({
            "task_id": task_id,
            "doc_id": doc_id,
            "file_name": document["file_name"],
            "status": "parsing"
        })
    
    return results

@router.get("/{kb_id}/documents/{doc_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    if document["status"] not in ["indexed", "completed"]:
        raise HTTPException(status_code=400, detail="Document must be indexed first")
    
    chunks = postgres_client.get_chunks_by_doc(doc_id)
    return chunks

@router.post("/{kb_id}/documents/{doc_id}/chunks/selection")
async def update_chunk_selection(kb_id: str, doc_id: str, request: ChunkSelectionRequest):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    for chunk_id in request.chunk_ids:
        postgres_client.update_chunk_selection(chunk_id, request.selected)
    
    return {
        "status": "success",
        "message": f"已{'选中' if request.selected else '取消选中'} {len(request.chunk_ids)} 个分块"
    }

@router.post("/{kb_id}/documents/{doc_id}/generate_kg", response_model=KGResponse)
async def generate_knowledge_graph(kb_id: str, doc_id: str, request: GenerateKGRequest = Body(...)):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    if document["status"] not in ["indexed", "completed"]:
        raise HTTPException(status_code=400, detail="Document must be indexed first")
    
    if not request.selected_chunk_ids:
        raise HTTPException(status_code=400, detail="Please select at least one chunk")

    # 预检查：图谱生成需要 Chat(LLM) 模型（实体/关系抽取）
    # 统一走 llm_models 表，未配置或类型不匹配时返回明确错误
    model_config = resolve_llm_model_config(request.model_id)

    postgres_client.update_document_status(doc_id, "generating_kg")
    print(f"DEBUG: Document status updated to generating_kg, doc_id={doc_id}")
    
    task_id = await task_manager.create_task(
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=document["file_name"],
        task_type="generate_kg"
    )
    print(f"DEBUG: Task created, task_id={task_id}")
    
    print(f"DEBUG: Creating background task with selected_chunk_ids={request.selected_chunk_ids}, mode={request.mode}, entity_resolution={request.entity_resolution}, with_community={request.with_community}")
    task = asyncio.create_task(generate_kg_background(task_id, kb_id, doc_id, request.selected_chunk_ids, model_config, request.mode, request.entity_resolution, request.with_community))
    print(f"DEBUG: Background task created successfully, task={task}")
    
    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "status": "generating_kg"
    }

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
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    logs = postgres_client.get_kg_logs_by_doc(doc_id)
    
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
        "doc_name": document["file_name"],
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

@router.get("/{kb_id}/kg-logs")
async def get_kg_logs_by_kb(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    logs = postgres_client.get_kg_logs_by_kb(kb_id)
    
    return {
        "kb_id": kb_id,
        "kb_name": kb["name"],
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
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    logs = postgres_client.get_parse_logs_by_doc(doc_id)
    
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
        "doc_name": document["file_name"],
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
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    parse_logs = postgres_client.get_parse_logs_by_doc(doc_id)
    kg_logs = postgres_client.get_kg_logs_by_doc(doc_id)
    
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
        "doc_name": document["file_name"],
        "kb_id": kb_id,
        "total_tasks": len(sorted_tasks),
        "tasks": sorted_tasks
    }

@router.get("/{kb_id}/documents/{doc_id}/original-chunks")
async def get_original_chunks(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    chunks = postgres_client.get_chunks_by_doc(doc_id)
    
    original_chunks = []
    for chunk in chunks:
        original_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "selected_for_kg": chunk.get("selected_for_kg", False)
        })
    
    return {"doc_id": doc_id, "doc_name": document["file_name"], "chunks": original_chunks}

@router.get("/{kb_id}/documents/{doc_id}/vector-store-content")
async def get_vector_store_content(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    try:
        hits = search_store.query(filters={"kb_id": kb_id, "doc_id": doc_id}, limit=1000)
        if not hits:
            # 历史重复解析场景：投影中该文件的 doc_id 与当前 PG 记录不一致，
            # 回退按 kb_id + 文档名精确匹配。
            hits = [
                hit
                for hit in search_store.keyword_search(
                    document["file_name"], filters={"kb_id": kb_id}, top_k=1000
                )
                if _hit_document(hit).get("doc_name") == document["file_name"]
            ]
        
        contents = []
        for hit in hits:
            source = _hit_document(hit)
            contents.append({
                "chunk_id": source.get("chunk_id"),
                "content_type": source.get("content_type", "text"),
                "content": source.get("content", ""),
                "doc_name": source.get("doc_name", "")
            })
        
        return {"doc_id": doc_id, "doc_name": document["file_name"], "contents": contents}
    except Exception as e:
        return {"doc_id": doc_id, "doc_name": document["file_name"], "contents": [], "error": str(e)}

@router.get("/{kb_id}/documents/{doc_id}/postgres-content")
async def get_postgres_content(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    images = postgres_client.get_images_by_doc(doc_id)
    tables = postgres_client.get_tables_by_doc(doc_id)
    
    formatted_images = []
    for img in images:
        formatted_images.append({
            "image_id": img["image_id"],
            "title": img.get("title", ""),
            "description": img.get("description", ""),
            "image_type": img.get("image_type", "normal"),
            "storage_key": img.get("storage_key") or img.get("minio_path", ""),
            "minio_path": img.get("minio_path", ""),
            "image_url": img.get("image_url", ""),
            "chunk_id": img.get("chunk_id", "")
        })
    
    formatted_tables = []
    for tbl in tables:
        md = tbl.get("meta_data")
        if md:
            if isinstance(md, (dict, list)):
                parsed_md = md
            elif isinstance(md, str):
                try:
                    parsed_md = json.loads(md)
                except Exception:
                    parsed_md = {}
            else:
                parsed_md = {}
        else:
            parsed_md = {}
        formatted_tables.append({
            "table_id": tbl["table_id"],
            "title": tbl.get("title", ""),
            "description": tbl.get("description", ""),
            "meta_data": parsed_md,
            "chunk_id": tbl.get("chunk_id", "")
        })
    
    return {
        "doc_id": doc_id,
        "doc_name": document["file_name"],
        "images": formatted_images,
        "tables": formatted_tables
    }

@router.get("/{kb_id}/documents/{doc_id}/graph-data")
async def get_graph_data(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")
    
    try:
        vertices = hugegraph_client.get_vertices_by_kb_with_details(kb_id)
        edges = hugegraph_client.get_edges_by_kb_with_details(kb_id)
        
        filtered_vertices = []
        filtered_edges = []
        
        def contains_doc_id(props_doc_id):
            if isinstance(props_doc_id, list):
                if len(props_doc_id) > 0 and isinstance(props_doc_id[0], str) and ',' in props_doc_id[0]:
                    return doc_id in props_doc_id[0].split(",")
                return doc_id in props_doc_id
            elif isinstance(props_doc_id, str):
                if props_doc_id:
                    return doc_id in props_doc_id.split(",")
                return False
            return False
        
        for vertex in vertices:
            props = vertex.get("properties", {})
            if contains_doc_id(props.get("doc_id")):
                filtered_vertices.append(vertex)
        
        vertex_ids = {v.get("id") for v in filtered_vertices}
        
        for edge in edges:
            edge_props = edge.get("properties", {})
            if (edge.get("outV") in vertex_ids and 
                edge.get("inV") in vertex_ids and 
                contains_doc_id(edge_props.get("doc_id"))):
                filtered_edges.append(edge)
        
        return {
            "doc_id": doc_id,
            "vertices": filtered_vertices,
            "edges": filtered_edges,
            "total_vertices": len(filtered_vertices),
            "total_edges": len(filtered_edges)
        }
    except Exception as e:
        return {
            "doc_id": doc_id,
            "vertices": [],
            "edges": [],
            "total_vertices": 0,
            "total_edges": 0,
            "error": str(e)
        }

@router.delete("/{kb_id}/documents/{doc_id}/graph-data")
async def delete_document_graph_data(kb_id: str, doc_id: str):
    """删除文档关联的图谱数据（实体/关系/社区报告），保留文档本身。

    用于图谱可视化页面：删除该文档关联的 HugeGraph 实体与关系、
    PostgreSQL 社区报告真源及 OpenSearch 投影。不删除 PG 文档记录、chunks、媒体文件。
    多文档共享的实体/关系仅裁剪 doc_id 归属；社区报告按单/多文档分级处理。
    """
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")

    errors = []
    fully_removed_entities = []

    # Step 1: 删除 HugeGraph 关系边
    try:
        edge_result = hugegraph_client.delete_edges_by_doc_id(doc_id)
        edge_ok = edge_result is True or (isinstance(edge_result, dict) and edge_result.get("success", True))
        if not edge_ok:
            errors.append("Edge processing returned error")
    except Exception as e:
        errors.append(f"Failed to delete KG edges: {str(e)}")

    # Step 2: 删除 HugeGraph 实体顶点（跨文档共享实体仅裁剪 doc_id）
    try:
        vertex_result = hugegraph_client.delete_vertices_by_doc_id(doc_id)
        if isinstance(vertex_result, dict):
            fully_removed_entities = vertex_result.get("removed_names", [])
            if not vertex_result.get("success", True):
                errors.append("Vertex processing returned error")
    except Exception as e:
        errors.append(f"Failed to delete KG vertices: {str(e)}")

    # Step 3: 处理社区报告（单文档删除 / 多文档标记 stale）
    report_deleted = 0
    report_updated = 0
    try:
        report_result = handle_community_reports(
            search_store, postgres_client, doc_id, kb_id, fully_removed_entities
        )
        report_deleted = report_result["deleted"]
        report_updated = report_result["updated"]
    except Exception as e:
        errors.append(f"Failed to handle community reports: {str(e)}")

    return {
        "success": len(errors) == 0,
        "doc_id": doc_id,
        "removed_entities": fully_removed_entities,
        "community_reports_deleted": report_deleted,
        "community_reports_updated": report_updated,
        "errors": errors,
    }

@router.get("/{kb_id}/graph-data")
async def get_kb_graph_data(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        vertices = hugegraph_client.get_vertices_by_kb_with_details(kb_id)
        edges = hugegraph_client.get_edges_by_kb_with_details(kb_id)
        
        return {
            "kb_id": kb_id,
            "kb_name": kb["name"],
            "vertices": vertices,
            "edges": edges,
            "total_vertices": len(vertices),
            "total_edges": len(edges)
        }
    except Exception as e:
        return {
            "kb_id": kb_id,
            "vertices": [],
            "edges": [],
            "total_vertices": 0,
            "total_edges": 0,
            "error": str(e)
        }

@router.get("/{kb_id}/hugegraph-stats")
async def get_hugegraph_stats(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        stats = hugegraph_client.get_graph_stats()
        return {"kb_id": kb_id, "stats": stats}
    except Exception as e:
        return {"kb_id": kb_id, "stats": {"vertices": 0, "edges": 0}, "error": str(e)}

@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(kb_id: str, doc_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    document = postgres_client.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document["kb_id"] != kb_id:
        raise HTTPException(status_code=403, detail="Document does not belong to this knowledge base")

    # 初始化删除日志记录器，传入来源名字快照（删除流程后期文档记录会被删，名字需提前捕获留痕）
    deletion_logger = DeletionLogger(
        postgres_client, kb_id, doc_id,
        kb_name=kb.get("name"),
        doc_name=document.get("file_name"),
    )
    deletion_logger.log_step_start("init", f"开始删除文档 '{document.get('file_name', '')}'（doc_id={doc_id}）")
    logger.info(f"Starting document deletion: kb_id={kb_id}, doc_id={doc_id}, task_id={deletion_logger.task_id}")

    # 重置 delete_step 以便从头开始删除
    postgres_client.update_document_status(doc_id, "deleting")
    postgres_client.set_delete_step(doc_id, None)
    current_step = None
    errors = []
    fully_removed_entities = []

    # Step 1: HugeGraph 边裁剪（含 description_parts）
    if step_index(current_step) < step_index("edges_done"):
        deletion_logger.log_step_start("edges", "删除 HugeGraph 边（edges）...")
        logger.info(f"Step 1: Deleting HugeGraph edges for doc_id={doc_id}")
        try:
            result = hugegraph_client.delete_edges_by_doc_id(doc_id)
            if result is True or (isinstance(result, dict) and result.get("success", True)):
                deletion_logger.log_step_complete("edges", count=0, errors=0,
                                         message="边（edges）处理成功")
                logger.info(f"Step 1: Edges deleted successfully for doc_id={doc_id}")
            else:
                deletion_logger.log_step_complete("edges", count=0, errors=1,
                                         message="边（edges）处理出错")
                errors.append("边（edges）处理出错")
                logger.warning(f"Step 1: Edge processing returned error for doc_id={doc_id}")
        except Exception as e:
            errors.append(f"删除 HugeGraph 边失败: {str(e)}")
            deletion_logger.log_step_complete("edges", count=0, errors=1,
                                     message=f"删除 HugeGraph 边失败: {str(e)}")
            logger.error(f"Step 1: Failed to delete KG edges for doc_id={doc_id}, error={str(e)}")
        postgres_client.set_delete_step(doc_id, "edges_done")
        current_step = "edges_done"

    # Step 2: HugeGraph 顶点裁剪（含 description_parts），收集被彻底删除的实体名
    if step_index(current_step) < step_index("vertices_done"):
        deletion_logger.log_step_start("vertices", "删除 HugeGraph 顶点（vertices）...")
        try:
            result = hugegraph_client.delete_vertices_by_doc_id(doc_id)
            if isinstance(result, dict):
                fully_removed_entities = result.get("removed_names", [])
                deleted = result.get("deleted", 0)
                updated = result.get("updated", 0)
                matched = result.get("matched", 0)
                vert_errors = result.get("errors", 0)
                error_details = result.get("error_details", [])
                success = result.get("success", True)
                deletion_logger.log_step_complete("vertices", count=matched, errors=vert_errors,
                                         message=f"顶点处理: 匹配 {matched} 个，删除 {deleted} 个，更新 {updated} 个，{vert_errors} 个错误，彻底删除 {len(fully_removed_entities)} 个实体")
                # 记录被删除/更新的顶点详情
                deletion_logger.log_item("vertices", "result", "summary",
                               status="success" if success else "warning",
                               details={"matched": matched, "deleted": deleted,
                                        "updated": updated, "errors": vert_errors,
                                        "removed_names": fully_removed_entities})
                # 记录错误详情（前 10 条），便于排查顶点删除失败原因
                if error_details:
                    deletion_logger.log_item("vertices", "errors", "details",
                                   status="error",
                                   details={"error_count": vert_errors,
                                            "error_details": error_details})
                    errors.append(f"顶点删除出现 {vert_errors} 个错误（详情见日志）")
            else:
                deletion_logger.log_step_complete("vertices", count=0, errors=1,
                                         message="顶点处理返回异常结果")
                errors.append("顶点处理返回异常结果")
        except Exception as e:
            errors.append(f"删除 HugeGraph 顶点失败: {str(e)}")
            deletion_logger.log_step_complete("vertices", count=0, errors=1,
                                     message=f"删除 HugeGraph 顶点失败: {str(e)}")
        postgres_client.set_delete_step(doc_id, "vertices_done")
        current_step = "vertices_done"

    # Step 3: 媒体文件（ArtifactStore 图片/表格）
    if step_index(current_step) < step_index("media_done"):
        deletion_logger.log_step_start("media", "删除媒体文件（图片/表格）...")
        image_count = 0
        table_count = 0
        media_errors = 0
        try:
            images = postgres_client.get_images_by_doc(doc_id)
            for image in images:
                storage_key = image.get("storage_key") or image.get("minio_path")
                if storage_key:
                    try:
                        artifact_store.delete_file(storage_key)
                        image_count += 1
                        deletion_logger.log_item("media", "image", image.get("image_id", ""),
                                       status="deleted",
                                       details={"storage_key": storage_key})
                    except Exception as e:
                        media_errors += 1
                        errors.append(f"删除图片文件失败 {storage_key}: {str(e)}")
                        deletion_logger.log_item("media", "image", image.get("image_id", ""),
                                       status="error",
                                       message=f"失败: {str(e)}",
                                       details={"storage_key": storage_key})
        except Exception as e:
            media_errors += 1
            errors.append(f"获取图片列表失败: {str(e)}")
        try:
            tables = postgres_client.get_tables_by_doc(doc_id)
            for table in tables:
                storage_key = table.get("storage_key") or table.get("minio_path")
                if storage_key:
                    try:
                        artifact_store.delete_file(storage_key)
                        table_count += 1
                        deletion_logger.log_item("media", "table", table.get("table_id", ""),
                                       status="deleted",
                                       details={"storage_key": storage_key})
                    except Exception as e:
                        media_errors += 1
                        errors.append(f"删除表格文件失败 {storage_key}: {str(e)}")
                        deletion_logger.log_item("media", "table", table.get("table_id", ""),
                                       status="error",
                                       message=f"失败: {str(e)}",
                                       details={"storage_key": storage_key})
        except Exception as e:
            media_errors += 1
            errors.append(f"获取表格列表失败: {str(e)}")
        try:
            # Also remove derived files that predate DB image rows or whose
            # rows were never inserted successfully.
            image_count += artifact_store.delete_prefix(f"images/{doc_id}")
        except Exception as e:
            media_errors += 1
            errors.append(f"删除派生图片目录失败: {str(e)}")
        deletion_logger.log_step_complete("media", count=image_count + table_count, errors=media_errors,
                                 message=f"媒体删除完成: {image_count} 张图片，{table_count} 个表格")
        if media_errors:
            postgres_client.update_document_status(doc_id, "delete_failed")
            return {
                "status": "partial_success",
                "message": "媒体文件回收失败，文档记录已保留，可重试删除",
                "errors": errors,
                "delete_step": current_step,
                "task_id": deletion_logger.task_id,
            }
        postgres_client.set_delete_step(doc_id, "media_done")
        current_step = "media_done"

    # Step 4: PG chunks
    if step_index(current_step) < step_index("chunks_done"):
        deletion_logger.log_step_start("chunks", "删除文档分块（chunks）...")
        try:
            chunks = postgres_client.get_chunks_by_doc(doc_id)
            chunk_count = len(chunks) if chunks else 0
            postgres_client.delete_chunks_by_doc(doc_id)
            deletion_logger.log_step_complete("chunks", count=chunk_count, errors=0,
                                     message=f"已删除 {chunk_count} 个文档分块")
            deletion_logger.log_item("chunks", "result", "summary",
                           status="deleted",
                           details={"chunks_deleted": chunk_count})
        except Exception as e:
            errors.append(f"删除分块失败: {str(e)}")
            deletion_logger.log_step_complete("chunks", count=0, errors=1,
                                     message=f"删除分块失败: {str(e)}")
        postgres_client.set_delete_step(doc_id, "chunks_done")
        current_step = "chunks_done"

    # Step 5: 跳过日志删除（外键约束已移除，日志自动保留）
    if step_index(current_step) < step_index("logs_done"):
        deletion_logger.log_step_start("logs", "检查处理日志...")
        kg_logs = postgres_client.get_kg_logs_by_doc(doc_id)
        kg_log_count = len(kg_logs) if kg_logs else 0
        parse_logs = postgres_client.get_parse_logs_by_doc(doc_id)
        parse_log_count = len(parse_logs) if parse_logs else 0
        deletion_logger.log_step_complete("logs", count=kg_log_count + parse_log_count, errors=0,
                                 message=f"日志已保留（外键约束已移除）: {kg_log_count} 条图谱日志，{parse_log_count} 条解析日志")
        deletion_logger.log_item("logs", "result", "summary",
                       status="preserved",
                       details={"kg_logs_preserved": kg_log_count, "parse_logs_preserved": parse_log_count})
        logger.info(f"Step 5: Logs will be preserved automatically (FK removed): {kg_log_count} KG, {parse_log_count} parse")
        postgres_client.set_delete_step(doc_id, "logs_done")
        current_step = "logs_done"

    # Step 6: 检索投影分块 + 社区报告分级。数据库的步骤名
    # 仍保留 es_done，以兼容已在执行的旧删除任务。
    if step_index(current_step) < step_index("es_done"):
        deletion_logger.log_step_start("search", "删除 OpenSearch 检索投影...")
        search_errors = 0
        chunk_deleted = 0
        report_deleted = 0
        report_updated = 0
        try:
            # 先逐条删除非社区报告投影，避免将多文档共享报告误删。
            chunk_deleted = _delete_non_report_projection(doc_id, kb_id)
            deletion_logger.log_item(
                "search", "chunks", doc_id, status="deleted",
                details={"chunks_deleted": chunk_deleted},
            )

            report_result = handle_community_reports(
                search_store, postgres_client, doc_id, kb_id, fully_removed_entities
            )
            report_deleted = report_result["deleted"]
            report_updated = report_result["updated"]
        except Exception as e:
            search_errors += 1
            errors.append(f"删除检索投影失败: {str(e)}")
            deletion_logger.log_item("search", "chunks", doc_id,
                           status="error",
                           message=f"失败: {str(e)}")
        deletion_logger.log_step_complete(
            "search", count=chunk_deleted + report_deleted, errors=search_errors,
            message=(
                f"检索投影清理完成: 删除 {chunk_deleted} 个分块，"
                f"{report_deleted} 个社区报告，更新 {report_updated} 个社区报告"
            ),
        )
        if search_errors:
            postgres_client.update_document_status(doc_id, "delete_failed")
            return {
                "status": "partial_success",
                "message": "检索投影清理失败，原始文件和文档记录已保留，可重试删除",
                "errors": errors,
                "delete_step": current_step,
                "task_id": deletion_logger.task_id,
            }
        postgres_client.set_delete_step(doc_id, "es_done")
        current_step = "es_done"

    # Step 7: ArtifactStore 原始文档
    if step_index(current_step) < step_index("file_done"):
        deletion_logger.log_step_start("file", "回收 ArtifactStore 中的原始文档...")
        storage_key = document.get("storage_key") or document.get("minio_path")
        if storage_key:
            try:
                artifact_store.delete_file(storage_key)
                deletion_logger.log_item("file", "document", doc_id,
                               status="deleted",
                               details={"storage_key": storage_key})
                deletion_logger.log_step_complete("file", count=1, errors=0,
                                         message=f"已回收文档文件: {storage_key}")
            except Exception as e:
                errors.append(f"删除文档文件失败: {str(e)}")
                deletion_logger.log_step_complete("file", count=0, errors=1,
                                         message=f"删除文档文件失败: {str(e)}")
                postgres_client.update_document_status(doc_id, "delete_failed")
                return {
                    "status": "partial_success",
                    "message": "原始文件回收失败，文档记录已保留，可重试删除",
                    "errors": errors,
                    "delete_step": current_step,
                    "task_id": deletion_logger.task_id,
                }
        else:
            deletion_logger.log_step_complete("file", count=0, errors=0,
                                     message="文档无 ArtifactStore storage_key")
        postgres_client.set_delete_step(doc_id, "file_done")
        current_step = "file_done"

    # Step 8: PG 记录硬删除（日志已保留，可安全删除文档记录）
    if step_index(current_step) < step_index("pg_record_done"):
        deletion_logger.log_step_start("pg_record", "删除文档记录...")
        try:
            postgres_client.delete_document(doc_id)  # 硬删除，日志因 FK 约束移除而保留
            deletion_logger.log_step_complete("pg_record", count=1, errors=0,
                                     message="文档记录已从 PostgreSQL 删除")
            logger.info(f"Step 8: Document {doc_id} deleted from PostgreSQL, logs preserved")
        except Exception as e:
            errors.append(f"删除文档记录失败: {str(e)}")
            deletion_logger.log_step_complete("pg_record", count=0, errors=1,
                                     message=f"失败: {str(e)}")
            postgres_client.update_document_status(doc_id, "delete_failed")
            deletion_logger.log_summary({
                "status": "partial_success",
                "message": f"文档 {doc_id} 部分删除",
                "errors": len(errors),
                "error_details": errors,
            })
            return {
                "status": "partial_success",
                "message": f"文档 {doc_id} 删除完成但存在警告",
                "errors": errors,
                "delete_step": current_step,
                "task_id": deletion_logger.task_id,
            }

    # 汇总并记录
    final_status = "success" if not errors else "warning"
    deletion_logger.log_summary({
        "status": final_status,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "file_name": document.get("file_name", ""),
        "errors": len(errors),
        "error_details": errors,
        "fully_removed_entities": fully_removed_entities,
        "final_step": current_step,
    })

    # TODO(P4): 异步触发 stale 社区报告重建（content/topology），刷新正文后清空 stale_kwd

    if errors:
        logger.warning(f"Document deletion completed with warnings: doc_id={doc_id}, errors={len(errors)}, task_id={deletion_logger.task_id}")
        return {
            "status": "partial_success",
            "message": f"文档 {doc_id} 删除完成但存在警告",
            "errors": errors,
            "delete_step": "done",
            "task_id": deletion_logger.task_id,
        }

    logger.info(f"Document deletion completed successfully: doc_id={doc_id}, task_id={deletion_logger.task_id}")
    return {"status": "success", "message": f"文档 {doc_id} 及其关联数据已全部删除",
            "task_id": deletion_logger.task_id}

# ==================== Deletion Log Endpoints ====================

@router.get("/deletion-logs/tasks")
async def get_deletion_tasks_all(limit: int = 20):
    """获取最近的文档删除任务列表（所有知识库）。"""
    tasks = postgres_client.get_latest_deletion_tasks(kb_id=None, limit=limit)
    return {"tasks": tasks}

@router.get("/{kb_id}/deletion-logs/tasks")
async def get_deletion_tasks(kb_id: str = None, limit: int = 20):
    """获取指定知识库最近的文档删除任务列表。"""
    tasks = postgres_client.get_latest_deletion_tasks(kb_id=kb_id, limit=limit)
    return {"tasks": tasks}

@router.get("/{kb_id}/deletion-logs/{task_id}")
async def get_deletion_logs_by_task(kb_id: str, task_id: str):
    """获取某次删除任务的完整日志详情。即使文档/知识库已删除也返回日志（留痕）。"""
    logs = postgres_client.get_deletion_logs_by_task(task_id)
    if not logs:
        raise HTTPException(status_code=404, detail="Deletion logs not found for this task")

    first = logs[0]
    doc_id = first.get("doc_id")
    # 来源名字优先用日志里的快照，存在性实时查询
    kb_name = first.get("kb_name")
    doc_name = first.get("doc_name")
    return {
        "task_id": task_id,
        "kb_id": first.get("kb_id") or kb_id,
        "doc_id": doc_id,
        "kb_name": kb_name,
        "doc_name": doc_name,
        "kb_exists": postgres_client.kb_exists(first.get("kb_id") or kb_id),
        "doc_exists": postgres_client.doc_exists(doc_id) if doc_id else False,
        "logs": logs,
        "total": len(logs),
    }

@router.get("/{kb_id}/deletion-logs")
async def get_deletion_logs_by_kb(kb_id: str, limit: int = 500):
    """获取知识库下所有文档删除日志。知识库已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_deletion_logs_by_kb(kb_id, limit=limit)
    kb = postgres_client.get_knowledge_base(kb_id)
    return {
        "kb_id": kb_id,
        "kb_name": kb["name"] if kb else (logs[0].get("kb_name") if logs else None),
        "kb_exists": bool(kb),
        "logs": logs,
        "total": len(logs),
    }

@router.get("/{kb_id}/documents/{doc_id}/deletion-logs")
async def get_deletion_logs_by_doc(kb_id: str, doc_id: str):
    """获取某个文档的所有删除日志。文档已删除时仍返回历史日志（留痕）。"""
    logs = postgres_client.get_deletion_logs_by_doc(doc_id)
    document = postgres_client.get_document(doc_id)
    kb = postgres_client.get_knowledge_base(kb_id)
    doc_name = document["file_name"] if document else (logs[0].get("doc_name") if logs else None)
    kb_name = kb["name"] if kb else (logs[0].get("kb_name") if logs else None)
    return {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "kb_name": kb_name,
        "doc_name": doc_name,
        "kb_exists": bool(kb),
        "doc_exists": bool(document),
        "logs": logs,
        "total": len(logs),
    }

@router.delete("/{kb_id}/deletion-logs/{task_id}")
async def delete_deletion_logs(kb_id: str, task_id: str):
    """清理某次删除任务的日志。"""
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    deleted = postgres_client.delete_deletion_logs_by_task(task_id)
    return {"task_id": task_id, "logs_deleted": deleted}

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

@router.get("/{kb_id}/community-reports")
async def get_community_reports(kb_id: str, doc_id: str = None):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        # 用户可见的报告详情始终从 PostgreSQL 权威源读取。
        reports = postgres_client.get_community_reports(kb_id, doc_id)
        
        formatted_reports = []
        for report in reports:
            report_data = report.get("report", {})
            formatted_reports.append({
                "report_id": report.get("report_id", ""),
                "community_id": report.get("community_id", ""),
                "entities": report.get("entities", []),
                "title": report_data.get("title", ""),
                "summary": report_data.get("summary", ""),
                "rating": report_data.get("rating", 0.0),
                "rating_explanation": report_data.get("rating_explanation", ""),
                "findings": report_data.get("findings", []),
                "doc_id": report.get("doc_id", ""),
                "doc_ids": report.get("doc_ids", []),
                "kb_id": report.get("kb_id", ""),
                "created_at": report.get("created_at", "")
            })
        
        return {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "total_reports": len(formatted_reports),
            "reports": formatted_reports
        }
    except Exception as e:
        return {
            "kb_id": kb_id,
            "doc_id": doc_id,
            "total_reports": 0,
            "reports": [],
            "error": str(e)
        }
