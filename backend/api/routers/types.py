"""ICP 类型驱动文档路由（doc_type 路径，替代知识库概念）。

首页 7 类文档卡片 → /document-types/{docType}/documents 文档管理 → 上传/解析/查看。
查看页四视图：原文 / 向量化 / PG 结构 / 图谱（按 data_form 组合）。
"""
import asyncio
import json
import mimetypes
import os
import uuid
import logging
from typing import List, Optional
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Body
from fastapi.responses import Response, StreamingResponse

from backend.core.parsers import ParserRegistry
from backend.core.original_content import (
    build_evidence_preview,
    build_original_content,
    read_original_docx_image,
)
from backend.agent.evidence_display import (
    deterministic_graph_source_ref,
    summarize_graph_structure,
)
from backend.core.task_manager import task_manager
from backend.services.document_service import parse_icp_document_background
from backend.services.kg_service import (
    generate_icp_graph_background,
    projection_item_revision_id,
    select_latest_confirmed_revisions,  # legacy patch/import compatibility
)
from backend.services.model_service import (
    resolve_optional_llm_model_config,
)
from backend.storage.search_store import search_store
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.artifact_store import (
    artifact_store,
    safe_original_name,
    source_artifact_key,
    storage_key_from,
)
from backend.storage.postgres_client import postgres_client
from backend.storage.registration import confirm_registration_or_compensate
from backend.icp.diagram_repository import (
    DiagramConcurrencyError,
    DiagramNotFoundError,
    DiagramStateError,
    diagram_repository,
)
from backend.icp.structured_projection import STRUCTURED_KB_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-types", tags=["ICP 文档类型"])

_TABLE_TYPES = ("hmc", "fmeca", "mfl")
_IMAGE_TYPES = ("fta", "rbd")
_TEXT_TYPES = ("fault_history", "function_locator")
_DIAGRAM_TYPES = ("function_locator", "fta", "rbd")
_GRAPH_TYPES = (
    "function_locator",
    "hmc",
    "fta",
    "rbd",
    "fmeca",
    "mfl",
    "fault_history",
)


def _require_diagram_type(doc_type: str) -> None:
    if doc_type not in _DIAGRAM_TYPES:
        raise HTTPException(
            status_code=400,
            detail="仅系统架构、FTA、RBD 文档包含工程图修订",
        )


def _require_reviewable_diagram_type(doc_type: str) -> None:
    if doc_type not in _IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="仅 FTA/RBD 保留可选的高级修订编辑接口",
        )


def _require_graph_type(doc_type: str) -> None:
    if doc_type not in _GRAPH_TYPES:
        raise HTTPException(status_code=400, detail="该类型不属于七类 ICP 文档")


def _required_lock_version(request: dict) -> int:
    value = request.get("lock_version")
    if value is None or isinstance(value, bool):
        raise HTTPException(status_code=422, detail="lock_version 为必填正整数")
    if isinstance(value, int):
        lock_version = value
    elif isinstance(value, str) and value.strip().isdigit():
        lock_version = int(value.strip())
    else:
        raise HTTPException(status_code=422, detail="lock_version 为必填正整数")
    if lock_version <= 0:
        raise HTTPException(status_code=422, detail="lock_version 为必填正整数")
    return lock_version


def _safe_evidence_image_url(value) -> str:
    """Accept a display URL without deriving one from a storage/source key."""

    url = str(value or "").strip()
    if not url or "\\" in url or any(ord(char) < 32 for char in url):
        return ""
    parsed = urlsplit(url)
    if parsed.scheme:
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
        path = parsed.path
    else:
        if not url.startswith("/") or url.startswith("//"):
            return ""
        path = parsed.path
    if any(part == ".." for part in unquote(path).split("/")):
        return ""
    return url


def _candidate_evidence_image_urls(candidates) -> set[str]:
    """Extract only explicit image URLs from revision-candidate locators."""

    urls: set[str] = set()
    for candidate in candidates or []:
        locator = candidate.get("source_locator") if isinstance(candidate, dict) else None
        if isinstance(locator, str):
            try:
                locator = json.loads(locator)
            except (TypeError, ValueError):
                locator = {}
        if not isinstance(locator, dict):
            continue
        values = [locator.get("image_url")]
        image_locator = locator.get("image")
        if isinstance(image_locator, dict):
            values.extend([image_locator.get("image_url"), image_locator.get("url")])
        for value in values:
            url = _safe_evidence_image_url(value)
            if url:
                urls.add(url)
    return urls


def _revision_candidate_evidence_urls(doc_id: str) -> set[str]:
    """Collect current candidates' evidence URLs without exposing storage_key."""

    urls: set[str] = set()
    for diagram in diagram_repository.list_diagrams_by_document(doc_id):
        revisions = diagram_repository.list_revisions(str(diagram["diagram_id"]))
        if not revisions:
            continue
        candidates = diagram_repository.get_revision_candidates(
            str(revisions[0]["revision_id"])
        )
        urls.update(_candidate_evidence_image_urls(candidates))
    return urls


# ===================== 类型注册表 =====================

@router.get("")
async def list_document_types():
    return postgres_client.list_document_types()


@router.get("/{doc_type}")
async def get_document_type(doc_type: str):
    t = postgres_client.get_document_type(doc_type)
    if not t:
        raise HTTPException(status_code=404, detail=f"文档类型不存在: {doc_type}")
    return t


# ===================== 文档管理 =====================

@router.post("/{doc_type}/documents/upload")
async def upload_icp_documents(doc_type: str, files: List[UploadFile] = File(...)):
    t = postgres_client.get_document_type(doc_type)
    if not t:
        raise HTTPException(status_code=404, detail=f"文档类型不存在: {doc_type}")

    allowed = [fmt.strip().lower() for fmt in (t.get("input_formats") or "").split(",") if fmt.strip()]
    results = []
    for file in files:
        file_name = safe_original_name(file.filename)
        ext = os.path.splitext(file_name)[1].lower()
        if allowed and ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file_name} 格式不支持（{t['name']} 支持 {t['input_formats']}）")

        doc_id = str(uuid.uuid4())
        storage_key = source_artifact_key(f"icp/{doc_type}", doc_id, file_name)
        try:
            metadata = await asyncio.to_thread(
                artifact_store.upload_file,
                storage_key,
                file.file,
                content_type=file.content_type,
                original_name=file_name,
            )
            try:
                # Phase 1 keeps the physical minio_path column as a compatibility
                # column; its value is now an ArtifactStore storage_key.
                postgres_client.insert_icp_document(doc_id, doc_type, file_name, storage_key)
            except Exception as insert_error:
                committed = confirm_registration_or_compensate(
                    doc_id=doc_id,
                    storage_key=storage_key,
                    lookup=postgres_client.get_icp_document,
                    rollback=postgres_client.conn.rollback,
                )
                if not committed:
                    raise insert_error
        except Exception as exc:
            logger.exception("ICP 文档写入 ArtifactStore 失败: %s", file_name)
            raise HTTPException(status_code=500, detail=f"文件 {file_name} 保存失败: {exc}") from exc

        results.append({
            "doc_id": doc_id,
            "file_name": file_name,
            "status": "uploaded",
            "storage_key": metadata.storage_key,
        })

    return results


@router.get("/{doc_type}/documents")
async def list_icp_documents(doc_type: str, status: Optional[str] = None):
    if not postgres_client.get_document_type(doc_type):
        raise HTTPException(status_code=404, detail=f"文档类型不存在: {doc_type}")
    return postgres_client.list_icp_documents(doc_type, status)


def _get_icp_document(doc_type: str, doc_id: str):
    doc = postgres_client.get_icp_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if doc["doc_type"] != doc_type:
        raise HTTPException(status_code=403, detail="文档不属于该类型")
    return doc


@router.post("/{doc_type}/documents/{doc_id}/parse")
async def parse_icp_document(doc_type: str, doc_id: str, request: dict = Body(default={})):
    doc = _get_icp_document(doc_type, doc_id)
    allowed_statuses = ("uploaded", "failed", "indexed", "kg_pending", "kg_done")
    if doc["status"] not in allowed_statuses:
        raise HTTPException(status_code=400, detail="文档正在处理中，请勿重复触发")

    # 七类专用解析以 PostgreSQL 确定性事实入库为主。未显式选模型且系统
    # 没有默认 LLM 时允许 model_config=None；各解析器会把视觉/检索投影
    # 的降级写入结果。显式 model_id 仍由 resolver 严格校验。
    model_config = resolve_optional_llm_model_config(request.get("model_id"))

    if not postgres_client.transition_icp_document_status(doc_id, allowed_statuses, "parsing"):
        raise HTTPException(status_code=409, detail="文档状态已变化，请刷新后重试")
    try:
        task_id = await task_manager.create_task(
            doc_id=doc_id, kb_id=doc_type, file_name=doc["file_name"], task_type="parse")
    except Exception:
        postgres_client.update_icp_document_status(doc_id, doc["status"])
        raise
    asyncio.create_task(parse_icp_document_background(task_id, doc_id, model_config))

    return {"task_id": task_id, "doc_id": doc_id, "doc_type": doc_type, "status": "parsing"}


@router.get("/{doc_type}/documents/{doc_id}/parse-result")
async def get_parse_result(doc_type: str, doc_id: str):
    doc = _get_icp_document(doc_type, doc_id)
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "status": doc["status"],
        "parse_count": doc.get("parse_count", 0),
        "last_result": doc.get("last_result"),
    }


@router.get("/{doc_type}/documents/{doc_id}/original")
async def get_original(
    doc_type: str,
    doc_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(100, ge=1, le=10_000),
):
    """原文内容：docx 标题树/表格/证据图、txt、excel 或直传图片。"""
    doc = _get_icp_document(doc_type, doc_id)
    ext = os.path.splitext(doc["file_name"])[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        block = {
            "id": "source-image",
            "kind": "image",
            "sequence": 1,
            "title": doc["file_name"],
            "level": 0,
            "text": "",
            "image_index": 1,
            "image_url": f"/api/v1/document-types/{doc_type}/documents/{doc_id}/source-image",
        }
        return {
            "doc_type": doc_type,
            "file_name": doc["file_name"],
            "total": 1,
            "page": page,
            "size": size,
            "counts": {"paragraphs": 0, "tables": 0, "images": 1},
            "blocks": [block] if page == 1 else [],
        }
    try:
        with artifact_store.materialize(storage_key_from(doc), suffix=ext) as local_path:
            result = build_original_content(
                local_path,
                doc_type=doc_type,
                doc_id=doc_id,
                file_name=doc["file_name"],
                page=page,
                size=size,
            )
        return result
    except Exception as exc:
        logger.exception("读取 ICP 原始文档失败: %s", doc_id)
        raise HTTPException(status_code=500, detail=f"原始文档读取失败: {exc}") from exc


@router.get("/{doc_type}/documents/{doc_id}/evidence-preview")
async def get_evidence_preview(
    doc_type: str,
    doc_id: str,
    source_ref: str = Query("", max_length=2048),
    record_id: str = Query("", max_length=240),
    title: str = Query("", max_length=500),
):
    """Return a compact source preview for one diagnosis evidence card.

    Historical messages may only retain a diagram revision UUID and truncated
    graph JSON.  The UUID is used solely to recover the active revision's
    explicit source locator; the preview itself is always read from the
    original artifact in source order.
    """

    doc = _get_icp_document(doc_type, doc_id)
    resolved_source_ref = str(source_ref or "").strip()
    resolved_title = str(title or "").strip()
    structure_summary = ""

    revision_key = str(record_id or "").strip()
    if not revision_key and resolved_source_ref and ";block=" not in resolved_source_ref:
        revision_key = resolved_source_ref
    if doc_type in _IMAGE_TYPES and revision_key:
        revision = diagram_repository.get_revision(revision_key)
        if (
            revision
            and str(revision.get("doc_id") or "") == doc_id
            and str(revision.get("diagram_type") or "") == doc_type
        ):
            graph = revision.get("canonical_graph") or {}
            resolved_title = resolved_title or str(revision.get("title") or "")
            structure_summary = summarize_graph_structure(
                graph,
                diagram_type=doc_type,
                title=resolved_title,
            )
            explicit_ref = deterministic_graph_source_ref(graph)
            if explicit_ref and (
                not resolved_source_ref
                or resolved_source_ref == revision_key
                or ";block=" not in resolved_source_ref
            ):
                resolved_source_ref = explicit_ref

    ext = os.path.splitext(doc["file_name"])[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        return {
            "found": True,
            "resolved_source_ref": resolved_source_ref,
            "structure_summary": structure_summary,
            "excerpt": "",
            "table": None,
            "image": {
                "title": doc["file_name"],
                "image_url": (
                    f"/api/v1/document-types/{doc_type}/documents/{doc_id}/source-image"
                ),
                "source_ref": resolved_source_ref,
            },
        }
    try:
        with artifact_store.materialize(storage_key_from(doc), suffix=ext) as local_path:
            original = build_original_content(
                local_path,
                doc_type=doc_type,
                doc_id=doc_id,
                file_name=doc["file_name"],
                page=1,
                size=10_000,
            )
        return build_evidence_preview(
            original.get("blocks") or [],
            source_ref=resolved_source_ref,
            title=resolved_title,
            structure_summary=structure_summary,
        )
    except Exception as exc:
        logger.exception("读取诊断证据原文失败: doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="诊断证据原文读取失败") from exc


@router.get("/{doc_type}/documents/{doc_id}/embedded-images/{image_index}")
async def get_embedded_image(doc_type: str, doc_id: str, image_index: int):
    """Read one source DOCX picture without creating another stored copy."""

    doc = _get_icp_document(doc_type, doc_id)
    ext = os.path.splitext(doc["file_name"])[1].lower()
    if ext != ".docx":
        raise HTTPException(status_code=400, detail="该文档没有可读取的 DOCX 内嵌图片")
    try:
        with artifact_store.materialize(storage_key_from(doc), suffix=ext) as local_path:
            blob, content_type, _name = read_original_docx_image(
                local_path,
                image_index,
                source_name=doc["file_name"],
            )
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("读取 DOCX 内嵌图片失败: doc_id=%s image_index=%s", doc_id, image_index)
        raise HTTPException(status_code=500, detail="图片读取失败") from exc
    return Response(
        content=blob,
        media_type=content_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{doc_type}/documents/{doc_id}/source-image")
async def get_source_image(doc_type: str, doc_id: str):
    """Stream one direct-upload image without exposing the original namespace."""
    doc = _get_icp_document(doc_type, doc_id)
    ext = os.path.splitext(doc["file_name"])[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        raise HTTPException(status_code=400, detail="该文档不是直传图片")
    storage_key = storage_key_from(doc)
    metadata = artifact_store.stat(storage_key)

    def body():
        with artifact_store.open_file(storage_key, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        body(),
        media_type=mimetypes.guess_type(doc["file_name"])[0] or metadata.content_type,
        headers={"Content-Length": str(metadata.size_bytes), "Cache-Control": "private, max-age=60"},
    )


@router.get("/{doc_type}/documents/{doc_id}/vectors")
async def get_vectors(doc_type: str, doc_id: str, page: int = 1, size: int = 50):
    _get_icp_document(doc_type, doc_id)
    filters = {"doc_id": doc_id, "removed_kwd": "N"}
    hits = search_store.query(filters=filters, limit=size, offset=max(page - 1, 0) * size)
    vectors = []
    for hit in hits:
        source = hit.document if hasattr(hit, "document") else hit.get("document", hit)
        vectors.append(source)
    return {
        "total": search_store.count(filters=filters),
        "page": page,
        "size": size,
        "vectors": vectors,
    }


@router.get("/{doc_type}/documents/{doc_id}/rows")
async def get_rows(doc_type: str, doc_id: str, page: int = 1, size: int = 50,
                   keyword: str = None, severity: str = None,
                   api_reportable: str = None, platform_test: str = None):
    """PG 真源表行数据（表格类）。支持 keyword 全列检索与列过滤。"""
    doc = _get_icp_document(doc_type, doc_id)
    filters = {}
    if keyword:
        filters["keyword"] = keyword
    if severity:
        filters["severity"] = severity
    if api_reportable in ("true", "false"):
        filters["api_reportable"] = api_reportable
    if platform_test in ("true", "false"):
        filters["platform_test"] = platform_test

    if doc_type == "hmc":
        return postgres_client.get_hmc_rows(doc_id, page, size, filters)
    if doc_type == "fmeca":
        return postgres_client.get_fmea_rows(doc_id, page, size, filters)
    if doc_type == "mfl":
        return postgres_client.get_mfl_rows(doc_id, page, size, filters)
    raise HTTPException(status_code=400, detail="该类型不是表格类文档，无 PG 行数据")


@router.get("/{doc_type}/documents/{doc_id}/fault-trees")
async def get_fault_trees(doc_type: str, doc_id: str):
    """兼容路径：返回每张工程图最新的可审核修订，而非旧随机图行。"""
    _require_diagram_type(doc_type)
    _get_icp_document(doc_type, doc_id)
    results = []
    for diagram in diagram_repository.list_diagrams_by_document(doc_id):
        if str(diagram.get("diagram_type") or "") != doc_type:
            continue
        revisions = diagram_repository.list_revisions(str(diagram["diagram_id"]))
        if not revisions:
            continue
        revision = revisions[0]
        results.append(
            {
                "fault_tree_id": revision["revision_id"],
                "diagram_id": diagram["diagram_id"],
                "revision_id": revision["revision_id"],
                "revision_no": revision["revision_no"],
                "doc_id": doc_id,
                "kb_id": doc_type,
                "chunk_id": diagram.get("source_section") or "",
                "image_id": "",
                "image_title": diagram.get("title") or diagram.get("diagram_key") or "",
                "graph": revision.get("canonical_graph") or {},
                "review_status": revision.get("review_status"),
                "validation_status": revision.get("validation_status"),
                "validation_tier": revision.get("validation_tier"),
                "validation_method": revision.get("validation_method"),
                "validator_version": revision.get("validator_version"),
                "validated_at": revision.get("validated_at"),
                "validation_report": revision.get("validation_report") or {},
                "quarantine_reason": revision.get("quarantine_reason") or "",
                "projection_status": revision.get("projection_status"),
                "conflict_count": revision.get("conflict_count", 0),
                "conflicts": revision.get("conflicts") or [],
                "lock_version": revision.get("lock_version"),
                "reviewed_by": revision.get("reviewed_by"),
                "review_note": revision.get("review_note"),
                "is_active": bool(revision.get("is_active")),
                "candidate_count": len(
                    diagram_repository.get_revision_candidates(str(revision["revision_id"]))
                ),
            }
        )
    return results


@router.get("/{doc_type}/documents/{doc_id}/diagrams")
async def list_reviewable_diagrams(doc_type: str, doc_id: str):
    """List diagram masters with their latest review revision."""
    return await get_fault_trees(doc_type, doc_id)


def _diagram_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DiagramNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DiagramConcurrencyError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, DiagramStateError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.patch("/{doc_type}/documents/{doc_id}/diagram-revisions/{revision_id}")
async def update_diagram_revision(
    doc_type: str,
    doc_id: str,
    revision_id: str,
    request: dict = Body(...),
):
    _require_reviewable_diagram_type(doc_type)
    _get_icp_document(doc_type, doc_id)
    revision = diagram_repository.get_revision(revision_id)
    if not revision or revision.get("doc_id") != doc_id or revision.get("diagram_type") != doc_type:
        raise HTTPException(status_code=404, detail="图修订不存在或不属于该文档")
    expected_lock_version = _required_lock_version(request)
    try:
        return diagram_repository.update_draft_revision(
            revision_id,
            request.get("canonical_graph") or {},
            expected_lock_version=expected_lock_version,
            validation_report=request.get("validation_report") or {},
        )
    except Exception as exc:
        raise _diagram_error(exc) from exc


@router.post("/{doc_type}/documents/{doc_id}/diagram-revisions/{revision_id}/confirm")
async def confirm_diagram_revision(
    doc_type: str,
    doc_id: str,
    revision_id: str,
    request: dict = Body(...),
):
    _require_reviewable_diagram_type(doc_type)
    _get_icp_document(doc_type, doc_id)
    revision = diagram_repository.get_revision(revision_id)
    if not revision or revision.get("doc_id") != doc_id or revision.get("diagram_type") != doc_type:
        raise HTTPException(status_code=404, detail="图修订不存在或不属于该文档")
    expected_lock_version = _required_lock_version(request)
    try:
        result = diagram_repository.confirm_revision(
            revision_id,
            reviewed_by=str(request.get("reviewed_by") or ""),
            review_note=str(request.get("review_note") or ""),
            expected_lock_version=expected_lock_version,
        )
        return result
    except Exception as exc:
        raise _diagram_error(exc) from exc


@router.post("/{doc_type}/documents/{doc_id}/diagram-revisions/{revision_id}/reject")
async def reject_diagram_revision(
    doc_type: str,
    doc_id: str,
    revision_id: str,
    request: dict = Body(...),
):
    _require_reviewable_diagram_type(doc_type)
    _get_icp_document(doc_type, doc_id)
    revision = diagram_repository.get_revision(revision_id)
    if not revision or revision.get("doc_id") != doc_id or revision.get("diagram_type") != doc_type:
        raise HTTPException(status_code=404, detail="图修订不存在或不属于该文档")
    expected_lock_version = _required_lock_version(request)
    try:
        return diagram_repository.reject_revision(
            revision_id,
            reviewed_by=str(request.get("reviewed_by") or ""),
            review_note=str(request.get("review_note") or ""),
            expected_lock_version=expected_lock_version,
        )
    except Exception as exc:
        raise _diagram_error(exc) from exc


@router.get("/{doc_type}/documents/{doc_id}/graph-data")
async def get_icp_graph_data(doc_type: str, doc_id: str):
    """Return this document's verified structured and engineering projections."""
    _require_graph_type(doc_type)
    try:
        _get_icp_document(doc_type, doc_id)
        active_revisions = (
            diagram_repository.list_active_revisions(
                doc_id=doc_id,
                diagram_type=doc_type,
            )
            if doc_type in _DIAGRAM_TYPES
            else []
        )
        active_revision_ids = {
            str(item["revision_id"]) for item in active_revisions
        }

        def contains_doc_id(props_doc_id):
            if isinstance(props_doc_id, list):
                return doc_id in props_doc_id
            if isinstance(props_doc_id, str):
                return doc_id in props_doc_id.split(",")
            return False

        structured_vertices = hugegraph_client.get_vertices_by_kb_with_details(
            STRUCTURED_KB_ID
        )
        structured_edges = hugegraph_client.get_edges_by_kb_with_details(
            STRUCTURED_KB_ID
        )
        fv = [
            vertex
            for vertex in structured_vertices
            if vertex.get("properties", {}).get("kb_id") == STRUCTURED_KB_ID
            and contains_doc_id(vertex.get("properties", {}).get("doc_id"))
        ]
        fe = [
            edge
            for edge in structured_edges
            if edge.get("properties", {}).get("kb_id") == STRUCTURED_KB_ID
            and contains_doc_id(edge.get("properties", {}).get("doc_id"))
        ]

        diagram_vertices = (
            hugegraph_client.get_vertices_by_kb_with_details(doc_type)
            if active_revision_ids
            else []
        )
        diagram_edges = (
            hugegraph_client.get_edges_by_kb_with_details(doc_type)
            if active_revision_ids
            else []
        )
        fv.extend(
            vertex
            for vertex in diagram_vertices
            if contains_doc_id(vertex.get("properties", {}).get("doc_id"))
            and projection_item_revision_id(vertex) in active_revision_ids
        )
        fe.extend(
            edge
            for edge in diagram_edges
            if contains_doc_id(edge.get("properties", {}).get("doc_id"))
            and projection_item_revision_id(edge) in active_revision_ids
        )
        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "active_revision_ids": sorted(active_revision_ids),
            "vertices": fv,
            "edges": fe,
            "total_vertices": len(fv),
            "total_edges": len(fe),
            "structured_vertices": len(
                [
                    item
                    for item in structured_vertices
                    if item.get("properties", {}).get("kb_id") == STRUCTURED_KB_ID
                    and contains_doc_id(item.get("properties", {}).get("doc_id"))
                ]
            ),
            "structured_edges": len(
                [
                    item
                    for item in structured_edges
                    if item.get("properties", {}).get("kb_id") == STRUCTURED_KB_ID
                    and contains_doc_id(item.get("properties", {}).get("doc_id"))
                ]
            ),
            "diagram_validation": [
                {
                    "revision_id": item.get("revision_id"),
                    "validation_status": item.get("validation_status"),
                    "validation_tier": item.get("validation_tier"),
                    "validation_method": item.get("validation_method"),
                }
                for item in active_revisions
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("读取 ICP HugeGraph 投影失败: doc_type=%s doc_id=%s", doc_type, doc_id)
        raise HTTPException(status_code=503, detail=f"图谱服务读取失败: {exc}") from exc


@router.post("/{doc_type}/documents/{doc_id}/generate_kg")
async def generate_kg(doc_type: str, doc_id: str, request: dict = Body(default={})):
    """自动校验并重建七类文档的显式 HugeGraph 投影。"""
    _require_graph_type(doc_type)
    doc = _get_icp_document(doc_type, doc_id)
    allowed_statuses = ("indexed", "kg_pending", "kg_done")
    if doc["status"] not in allowed_statuses:
        raise HTTPException(status_code=400, detail="文档正在处理中或尚未成功解析")

    raw_requested_ids = request.get("revision_ids")
    if raw_requested_ids is not None and not isinstance(raw_requested_ids, list):
        raise HTTPException(status_code=400, detail="revision_ids 必须是数组")
    if raw_requested_ids is not None and not any(
        str(value).strip() for value in raw_requested_ids
    ):
        raise HTTPException(
            status_code=422,
            detail="显式 revision_ids 不能为空；省略该字段才表示发布全部合格修订",
        )
    if doc_type not in _DIAGRAM_TYPES and raw_requested_ids:
        raise HTTPException(status_code=422, detail="该类型没有工程图 revision_ids")
    revision_ids = (
        [str(value).strip() for value in raw_requested_ids if str(value).strip()]
        if raw_requested_ids is not None
        else None
    )

    if not postgres_client.transition_icp_document_status(doc_id, allowed_statuses, "generating_kg"):
        raise HTTPException(status_code=409, detail="文档状态已变化，请刷新后重试")
    try:
        task_id = await task_manager.create_task(
            doc_id=doc_id, kb_id=doc_type, file_name=doc["file_name"], task_type="generate_kg")
    except Exception:
        postgres_client.update_icp_document_status(doc_id, doc["status"])
        raise
    asyncio.create_task(
        generate_icp_graph_background(
            task_id,
            doc_id,
            doc_type,
            doc["file_name"],
            requested_revision_ids=revision_ids,
        )
    )

    return {
        "task_id": task_id,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "revision_ids": revision_ids,
        "status": "generating_kg",
    }


@router.delete("/{doc_type}/documents/{doc_id}")
async def delete_icp_document(doc_type: str, doc_id: str):
    """删除文档：ArtifactStore + 真源表/从属数据 + OpenSearch + HugeGraph + 主表。"""
    doc = _get_icp_document(doc_type, doc_id)
    previous_status = doc["status"]
    deletable_statuses = ("uploaded", "failed", "indexed", "kg_pending", "kg_done")
    if previous_status not in deletable_statuses:
        raise HTTPException(status_code=409, detail="文档正在解析、生成图谱或删除中，请稍后重试")
    if not postgres_client.transition_icp_document_status(doc_id, deletable_statuses, "deleting"):
        raise HTTPException(status_code=409, detail="文档状态已变化，请刷新后重试")

    storage_key = None
    try:
        if doc.get("storage_key") or doc.get("minio_path"):
            storage_key = storage_key_from(doc)
    except Exception as exc:
        logger.warning("文档 %s 的 storage_key 非法，将在删除后由审计清理: %s", doc_id, exc)

    try:
        # 先清理可重建投影；失败时原件和 PG 真源仍完整，可安全重试。
        search_store.delete_by_filter({"doc_id": doc_id})
        hugegraph_client.delete_edges_by_doc_id(doc_id)
        hugegraph_client.delete_vertices_by_doc_id(doc_id)
    except Exception as exc:
        postgres_client.update_icp_document_status(doc_id, previous_status)
        logger.exception("ICP 检索投影删除失败，保留文档: %s", doc_id)
        raise HTTPException(status_code=500, detail=f"检索投影删除失败，文档和原件已保留: {exc}") from exc

    try:
        if not postgres_client.delete_icp_document_with_sources(doc_id):
            raise RuntimeError("文档记录已不存在")
    except Exception as exc:
        try:
            postgres_client.update_icp_document_status(doc_id, previous_status)
        except Exception:
            logger.exception("恢复 ICP 文档状态失败: %s", doc_id)
        raise HTTPException(status_code=500, detail=f"数据库删除失败，原件已保留: {exc}") from exc

    # PG 权威记录删除成功后再回收文件。失败最多留下可由审计定位的孤儿，
    # 不会留下仍可见但原件已经不可读的文档。
    cleanup_errors = []
    if storage_key:
        try:
            artifact_store.delete_file(storage_key)
        except Exception as exc:
            logger.exception("ICP 原件回收失败: %s", doc_id)
            cleanup_errors.append(f"原件回收失败: {exc}")
    try:
        artifact_store.delete_prefix(f"images/{doc_id}")
    except Exception as exc:
        logger.exception("ICP 派生图片回收失败: %s", doc_id)
        cleanup_errors.append(f"派生图片回收失败: {exc}")

    return {
        "status": "success_with_warnings" if cleanup_errors else "success",
        "doc_id": doc_id,
        "message": "文档已删除" if not cleanup_errors else "文档已删除，文件清理需审计重试",
        "cleanup_errors": cleanup_errors,
    }


# ===================== 检索 =====================

@router.post("/search")
async def structured_search(request: dict = Body(...)):
    """doc_type 过滤的混合检索（替代 kb_id）。filters 支持三码/severity/api_reportable/platform_test 等。"""
    doc_type = request.get("doc_type", "")
    query = request.get("query", "")
    k = int(request.get("k", 10))
    filters = request.get("filters") or {}
    if not doc_type:
        raise HTTPException(status_code=400, detail="doc_type 必填")
    if not postgres_client.get_document_type(doc_type):
        raise HTTPException(status_code=404, detail=f"文档类型不存在: {doc_type}")

    query_vector = None
    if query:
        from backend.core.embedding import embedding_service
        query_vector = embedding_service.embed(query)

    projection_filters = {
        "doc_type_kwd": doc_type,
        "removed_kwd": "N",
        **{key: value for key, value in filters.items() if value not in (None, "")},
    }
    if query and query_vector:
        raw_hits = search_store.hybrid_search(
            query, query_vector, filters=projection_filters, top_k=k
        )
    elif query:
        raw_hits = search_store.keyword_search(query, filters=projection_filters, top_k=k)
    else:
        raw_hits = search_store.query(filters=projection_filters, limit=k)
    hits = [
        hit.document if hasattr(hit, "document") else hit.get("document", hit)
        for hit in raw_hits
    ]
    return {"doc_type": doc_type, "query": query, "total": len(hits), "hits": hits}
