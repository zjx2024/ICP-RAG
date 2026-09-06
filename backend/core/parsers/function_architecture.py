"""鲁棒的系统架构与功能说明解析器。

权威组件、功能和接口只来自确定性表格、带标签字段及严格箭头表达式。
视觉模型结果只保存为 provisional/quarantined 图候选，绝不反向生成 PostgreSQL 功能事实。
"""
from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Mapping, Tuple

from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.parsers.base import ParseContext, ParseResult, Parser
from backend.core.parsers.document_ir import load_document_ir
from backend.core.parsers.function_extraction import extract_function_drafts
from backend.core.parsers.registry import ParserRegistry
from backend.icp.diagram_repository import diagram_repository
from backend.icp.image_analysis import analyze_icp_diagram
from backend.icp.repository import icp_repository
from backend.storage.artifact_store import artifact_store
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store

_SUPPORTED = (".docx", ".xlsx", ".xlsm", ".xls", ".txt", ".md")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _stable_projection_id(doc_id: str, kind: str, business_id: str) -> str:
    digest = hashlib.sha256(
        f"function_locator\x1f{kind}\x1f{business_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{doc_id}_function_locator_{digest}"


def _scope(ctx: ParseContext) -> Dict[str, str]:
    metadata = (ctx.common_tags or {}).get("document_metadata") or {}
    if not isinstance(metadata, Mapping):
        return {}
    mapping = {
        "system": "applicable_system",
        "configuration": "applicable_configuration",
        "document_number": "document_number",
        "document_version": "document_version",
    }
    return {
        target: _clean(metadata.get(source))
        for target, source in mapping.items()
        if _clean(metadata.get(source))
    }


def _materialize_image(image: Dict[str, Any]):
    storage_key = image.get("storage_key") or image.get("minio_path")
    if storage_key:
        return artifact_store.materialize(
            storage_key, suffix=os.path.splitext(str(storage_key))[1]
        )
    return nullcontext(image.get("local_path") or "")


def _image_content_type(image: Mapping[str, Any]) -> str:
    declared = _clean(image.get("content_type")).split(";", 1)[0].lower()
    if declared:
        return declared
    storage_key = _clean(image.get("storage_key") or image.get("minio_path"))
    if storage_key:
        try:
            return _clean(artifact_store.stat(storage_key).content_type).lower()
        except Exception:
            pass
    return "application/octet-stream"


@ParserRegistry.register("function_locator")
class FunctionArchitectureParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in _SUPPORTED:
            return ["系统架构与功能文档支持 .docx/.xlsx/.xlsm/.xls/.txt/.md"]
        return []

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        extraction = extract_function_drafts(
            file_path, source_name=ctx.file_name
        )
        all_drafts = [
            *extraction.components,
            *extraction.functions,
            *extraction.interfaces,
        ]
        if not all_drafts and not extraction.image_evidence:
            raise ValueError("未识别到系统、子系统、模块、功能、明确接口或架构图证据")
        if all_drafts and not any(draft.valid for draft in all_drafts):
            first_errors = "; ".join(
                issue.message for issue in all_drafts[0].errors
            ) or "全部结构化记录均无效"
            raise ValueError(
                "Function Locator 未产生有效权威记录，保留上一版快照: "
                f"{first_errors}"
            )

        # Extraction is complete before the previous authority snapshot is
        # replaced. Invalid rows remain in diagnostics but never become facts.
        icp_repository.clear_function_snapshot(ctx.doc_id)
        database_failures: List[Dict[str, str]] = []
        invalid_records: List[Dict[str, Any]] = []
        projection_failures: List[Dict[str, str]] = []
        component_rows: List[Tuple[int, Dict[str, Any]]] = []
        function_rows: List[Tuple[int, Dict[str, Any]]] = []
        interface_rows: List[Tuple[int, Dict[str, Any]]] = []

        for kind, drafts, writer, target in (
            ("component", extraction.components, icp_repository.upsert_component, component_rows),
            ("function", extraction.functions, icp_repository.upsert_function, function_rows),
            ("interface", extraction.interfaces, icp_repository.upsert_interface, interface_rows),
        ):
            for draft in drafts:
                row = draft.to_row()
                if not draft.valid:
                    invalid_records.append(
                        {
                            "kind": kind,
                            "business_id": draft.record_key,
                            "source_ref": draft.source_ref,
                            "errors": [issue.as_dict() for issue in draft.errors],
                        }
                    )
                    continue
                try:
                    row_id = writer(ctx.doc_id, row)
                    target.append((row_id, row))
                except Exception as exc:
                    database_failures.append(
                        {
                            "kind": kind,
                            "business_id": draft.record_key,
                            "source_ref": draft.source_ref,
                            "reason": str(exc)[:500],
                        }
                    )

        if database_failures:
            raise RuntimeError(
                "Function Locator PostgreSQL 真源写入失败 "
                f"{len(database_failures)} 条；文档不会进入可诊断状态。"
                f"首个错误: {database_failures[0]['reason']}"
            )

        vectors = 0
        for row_id, row in component_rows:
            content = (
                f"组件ID: {row.get('component_id', '')}\n"
                f"类型: {row.get('component_type', '')}\n名称: {row.get('name', '')}\n"
                f"简称: {row.get('abbreviation', '')}\n"
                f"父组件ID: {row.get('parent_component_id', '')}\n"
                f"说明: {row.get('description', '')}"
            )
            vectors += self._safe_index(
                ctx,
                _stable_projection_id(ctx.doc_id, "component", str(row.get("component_id") or row_id)),
                str(row.get("name") or row.get("component_id") or "组件"),
                content,
                "function_component",
                {
                    "business_id_kwd": row.get("component_id", ""),
                    "node_type_kwd": row.get("component_type", ""),
                    "parent_kwd": row.get("parent_component_id", ""),
                    "pg_id_kwd": str(row_id),
                    "source_ref": row.get("source_ref", ""),
                },
                projection_failures,
            )

        for row_id, row in function_rows:
            content = (
                f"功能ID: {row.get('function_id', '')}\n功能名称: {row.get('function_name', '')}\n"
                f"系统/子系统/模块: {row.get('system_name', '')} / "
                f"{row.get('subsystem_name', '')} / {row.get('module_name', '')}\n"
                f"功能说明: {row.get('description', '')}\n"
                f"输入: {'；'.join(row.get('inputs') or [])}\n"
                f"输出: {'；'.join(row.get('outputs') or [])}\n"
                f"典型失效表现: {row.get('failure_symptoms', '')}"
            )
            vectors += self._safe_index(
                ctx,
                _stable_projection_id(ctx.doc_id, "function", str(row.get("function_id") or row_id)),
                str(row.get("function_name") or row.get("function_id") or "功能"),
                content,
                "function_record",
                {
                    "business_id_kwd": row.get("function_id", ""),
                    "parent_kwd": row.get("parent_function_id", ""),
                    "module_id_kwd": row.get("module_id", ""),
                    "pg_id_kwd": str(row_id),
                    "source_ref": row.get("source_ref", ""),
                },
                projection_failures,
            )

        for row_id, row in interface_rows:
            content = (
                f"接口ID: {row.get('interface_id', '')}\n"
                f"源: {row.get('source_component_id', '')}\n"
                f"目标: {row.get('target_component_id', '')}\n"
                f"关系: {row.get('relation_type', '')}\n信号: {row.get('signal_name', '')}\n"
                f"方向: {row.get('direction', '')}\n协议: {row.get('protocol', '')}\n"
                f"说明: {row.get('description', '')}"
            )
            vectors += self._safe_index(
                ctx,
                _stable_projection_id(ctx.doc_id, "interface", str(row.get("interface_id") or row_id)),
                str(row.get("signal_name") or row.get("interface_id") or "接口"),
                content,
                "function_interface",
                {
                    "business_id_kwd": row.get("interface_id", ""),
                    "pg_id_kwd": str(row_id),
                    "source_ref": row.get("source_ref", ""),
                },
                projection_failures,
            )

        text_chunks, text_vectors = self._index_text_sections(file_path, ctx, projection_failures)
        vectors += text_vectors
        image_stats = self._process_images(file_path, ctx, projection_failures)
        vectors += image_stats["vectors"]

        stats = {
            **extraction.stats,
            "components_written": len(component_rows),
            "functions_written": len(function_rows),
            "interfaces_written": len(interface_rows),
            "invalid_record_details": invalid_records[:100],
            "normalization_warnings": sum(len(item.issues) for item in all_drafts),
            "text_chunks": text_chunks,
            "images_saved": image_stats["images_saved"],
            "vision_candidates": image_stats["vision_candidates"],
            "vision_failed": image_stats["vision_failed"][:100],
            "review_status_counts": image_stats["review_status_counts"],
            "projection_failed": projection_failures[:100],
            "vectors": vectors,
            # This phase publishes only reviewed FTA/RBD topology to HugeGraph.
            "kg_pending": False,
        }
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=(
                f"解析完成：系统/子系统/模块 {len(component_rows)} 条，功能 "
                f"{len(function_rows)} 条，明确接口 {len(interface_rows)} 条，"
                f"架构图证据 {image_stats['images_saved']} 张，向量化 {vectors} 条"
            ),
        )

    def _index_text_sections(
        self,
        file_path: str,
        ctx: ParseContext,
        failures: List[Dict[str, str]],
    ) -> Tuple[int, int]:
        document = load_document_ir(file_path, source_name=ctx.file_name)
        source_name = document.file_name
        grouped: Dict[str, List[str]] = {}
        for paragraph in document.paragraphs:
            text = _clean(paragraph.text)
            if not text:
                continue
            section = " / ".join(value for value in paragraph.section_path if value) or "文档说明"
            grouped.setdefault(section, []).append(text)
        vectors = 0
        for section, paragraphs in grouped.items():
            content = "\n".join(paragraphs)
            vectors += self._safe_index(
                ctx,
                _stable_projection_id(ctx.doc_id, "section", section),
                section,
                content,
                "function_text",
                {"source_ref": f"{source_name}#{section}"},
                failures,
            )
        return len(grouped), vectors

    def _process_images(
        self,
        file_path: str,
        ctx: ParseContext,
        projection_failures: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "images_saved": 0,
            "vision_candidates": 0,
            "vision_failed": [],
            "vectors": 0,
            "review_status_counts": {},
        }
        if os.path.splitext(file_path)[1].lower() != ".docx":
            return result
        parsed = DocumentParser(
            file_path, ctx.doc_id, source_name=ctx.file_name
        ).process()
        if parsed.get("error"):
            result["vision_failed"].append(
                {"kind": "image_evidence", "reason": str(parsed["error"])[:500]}
            )
            return result

        candidate_ids: Dict[str, List[str]] = {}
        present_diagram_ids: List[str] = []
        for block_key, block in (parsed.get("blocks") or {}).items():
            title = _clean(block.get("title") or block_key)
            table_text = "\n".join(
                _clean(table.get("markdown")) for table in block.get("tables", [])
            )
            context_text = "\n".join(
                value for value in (title, _clean(block.get("text")), table_text) if value
            )
            for image in block.get("images", []):
                storage_key = str(image.get("storage_key") or image.get("minio_path") or "")
                image_title = _clean(image.get("title") or f"{title}-架构图")
                try:
                    fingerprint = artifact_store.stat(storage_key).sha256
                except Exception:
                    fingerprint = hashlib.sha256(
                        f"{ctx.doc_id}\x1f{block_key}\x1f{image.get('index')}".encode("utf-8")
                    ).hexdigest()
                image_id = f"{ctx.doc_id}_architecture_image_{fingerprint[:24]}"
                mime_type = _image_content_type(image)
                image_source_ref = _clean(image.get("source_ref") or image_id)
                try:
                    postgres_client.insert_image(
                        image_id,
                        ctx.doc_id,
                        None,
                        image_title,
                        "待视觉解析的系统架构工程图证据",
                        storage_key,
                        "architecture_evidence",
                        image.get("image_url", ""),
                        mime_type,
                        image_source_ref,
                    )
                    result["images_saved"] += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Function Locator PostgreSQL 图片证据写入失败（{image_source_ref}）: {exc}"
                    ) from exc

                try:
                    diagram = diagram_repository.upsert_diagram(
                        doc_id=ctx.doc_id,
                        diagram_type="function_locator",
                        diagram_key=f"ARCH-IMAGE:{fingerprint}",
                        title=image_title,
                        source_section=title,
                        scope=_scope(ctx),
                    )
                    diagram_id = str(diagram["diagram_id"])
                    present_diagram_ids.append(diagram_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"Function Locator PostgreSQL 架构主记录写入失败（{image_source_ref}）: {exc}"
                    ) from exc

                try:
                    with _materialize_image(image) as local_path:
                        if not local_path or not os.path.exists(local_path):
                            raise FileNotFoundError("架构图证据无法物化")
                        analysis = analyze_icp_diagram(
                            local_path,
                            "function_locator",
                            image_title,
                            context_text,
                            ctx.model_config,
                        )
                except Exception as exc:
                    result["vision_failed"].append(
                        {"kind": "architecture_vision", "source_ref": image_id, "reason": str(exc)[:500]}
                    )
                    failed_candidate = diagram_repository.upsert_candidate(
                        diagram_id=diagram_id,
                        source_kind="image_vision",
                        source_fingerprint=fingerprint,
                        raw_graph={
                            "diagram_type": "function_locator",
                            "root_fault_id": "",
                            "metadata": {
                                "image_url": image.get("image_url", ""),
                                "storage_key": storage_key,
                                "source_block": block_key,
                                "mime_type": mime_type,
                                "source_ref": image_source_ref,
                            },
                            "vertices": [],
                            "edges": [],
                        },
                        source_locator={
                            "block": block_key,
                            "image_id": image_id,
                            "image_url": image.get("image_url", ""),
                            "storage_key": storage_key,
                            "mime_type": mime_type,
                            "source_ref": image_source_ref,
                            "extraction_error": str(exc)[:500],
                        },
                        extractor_name="analyze_icp_diagram",
                        extractor_version="1",
                        extraction_status="failed",
                    )
                    candidate_ids.setdefault(diagram_id, []).append(
                        str(failed_candidate["candidate_id"])
                    )
                    continue

                vertices = list(analysis.get("vertices") or [])
                edges = list(analysis.get("edges") or [])
                if not vertices and not edges:
                    result["vision_failed"].append(
                        {
                            "kind": "architecture_vision",
                            "source_ref": image_id,
                            "reason": "视觉解析未产生可审核节点/关系"
                            + (
                                f": {_clean(analysis.get('description'))[:300]}"
                                if _clean(analysis.get("description"))
                                else ""
                            ),
                        }
                    )
                graph = {
                    "diagram_type": "function_locator",
                    "root_fault_id": analysis.get("root_id") or "",
                    "metadata": {
                        **(analysis.get("metadata") or {}),
                        "description": analysis.get("description", ""),
                        "confidence": analysis.get("confidence", 0.0),
                        "image_url": image.get("image_url", ""),
                        "storage_key": storage_key,
                        "source_block": block_key,
                        "mime_type": mime_type,
                        "source_ref": image_source_ref,
                    },
                    "vertices": vertices,
                    "edges": edges,
                }
                try:
                    postgres_client.insert_image(
                        image_id,
                        ctx.doc_id,
                        None,
                        image_title,
                        _clean(analysis.get("description")),
                        storage_key,
                        "architecture_candidate",
                        image.get("image_url", ""),
                        mime_type,
                        image_source_ref,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Function Locator PostgreSQL 图片候选说明写入失败（{image_source_ref}）: {exc}"
                    ) from exc

                summary = (
                    f"系统架构工程图视觉候选（自动分级）：{image_title}\n"
                    f"图意：{analysis.get('description', '')}\n"
                    f"候选节点数：{len(vertices)}；候选关系数：{len(edges)}\n"
                    "用途限制：仅用于定位原图；视觉候选不得作为正式架构关系。"
                )
                result["vectors"] += self._safe_index(
                    ctx,
                    _stable_projection_id(ctx.doc_id, "architecture_image", fingerprint),
                    image_title,
                    summary,
                    "function_diagram_candidate_summary",
                    {
                        "image_url": image.get("image_url", ""),
                        "diagram_id_kwd": diagram_id,
                        "projection_role_kwd": "diagram_locator",
                        "review_status_kwd": "pending",
                        "source_ref": image_id,
                    },
                    projection_failures,
                )
                try:
                    model_name = ""
                    if isinstance(ctx.model_config, Mapping):
                        model_name = _clean(
                            ctx.model_config.get("model_name")
                            or ctx.model_config.get("name")
                            or ctx.model_config.get("model")
                        )
                    candidate = diagram_repository.upsert_candidate(
                        diagram_id=diagram_id,
                        source_kind="image_vision",
                        source_fingerprint=fingerprint,
                        raw_graph=graph,
                        source_locator={
                            "block": block_key,
                            "image_id": image_id,
                            "image_url": image.get("image_url", ""),
                            "storage_key": storage_key,
                            "mime_type": mime_type,
                            "source_ref": image_source_ref,
                        },
                        extractor_name="analyze_icp_diagram",
                        extractor_version="1",
                        model_name=model_name or None,
                        prompt_version="function-architecture-v1",
                        extraction_status=(
                            "succeeded" if vertices or edges else "skipped"
                        ),
                    )
                    candidate_ids.setdefault(diagram_id, []).append(
                        str(candidate["candidate_id"])
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Function Locator PostgreSQL 架构候选写入失败（{image_source_ref}）: {exc}"
                    ) from exc

        for diagram_id, ids in candidate_ids.items():
            revision = diagram_repository.create_or_reuse_revision(
                diagram_id=diagram_id,
                candidate_ids=list(dict.fromkeys(ids)),
            )
            status = str(revision.get("review_status") or "pending")
            counts = result["review_status_counts"]
            counts[status] = counts.get(status, 0) + 1
            result["vision_candidates"] += 1

        try:
            diagram_repository.archive_missing_diagrams(
                doc_id=ctx.doc_id,
                present_diagram_ids=list(dict.fromkeys(present_diagram_ids)),
                allow_empty=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Function Locator PostgreSQL 架构快照收敛失败: {exc}"
            ) from exc
        return result

    @staticmethod
    def _safe_index(
        ctx: ParseContext,
        chunk_id: str,
        title: str,
        content: str,
        content_type: str,
        extra: Dict[str, Any],
        failures: List[Dict[str, str]],
    ) -> int:
        if not _clean(content):
            return 0
        try:
            vector = embedding_service.embed(content)
            body = ctx.enrich(
                {
                    "chunk_id": chunk_id,
                    "doc_id": ctx.doc_id,
                    "doc_name": ctx.file_name,
                    "title": title,
                    "section": "function_locator",
                    "content": content,
                    "kb_id": ctx.doc_type,
                    "content_type": content_type,
                    "doc_type_kwd": ctx.doc_type,
                    "removed_kwd": "N",
                    "available_int": 1,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    **extra,
                }
            )
            search_store.upsert(chunk_id, body, vector=vector)
            return 1
        except Exception as exc:
            failures.append(
                {
                    "kind": content_type,
                    "source_ref": str(extra.get("source_ref") or chunk_id),
                    "reason": str(exc)[:500],
                }
            )
            return 0


__all__ = ["FunctionArchitectureParser"]
