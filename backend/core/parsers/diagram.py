"""FTA / RBD 专用图文解析器。

Word 文档按章节保留邻近段落、表格和图题；图片分别按 FTA 或 RBD 语义解析。
结构化图保存到 PostgreSQL，图摘要、节点、关系和说明段落写入检索投影。
"""
from __future__ import annotations

import json
import hashlib
import logging
import mimetypes
import os
import re
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.diagram_extraction import (
    DiagramExtraction,
    extract_fta_from_ir,
    extract_rbd_from_ir,
)
from backend.core.parsers.document_ir import load_document_ir
from backend.core.parsers.registry import ParserRegistry
from backend.core.parsers.records import source_file_name
from backend.icp.image_analysis import analyze_icp_diagram
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import (
    deterministic_candidate_allows_formal_topology,
)
from backend.config.config import config
from backend.storage.artifact_store import artifact_store
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _vertex_name(vertex: Dict[str, Any]) -> str:
    props = vertex.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    return _safe_text(props.get("name") or vertex.get("name") or vertex.get("id"))


def _materialize_image(image: Dict[str, Any]):
    storage_key = image.get("storage_key") or image.get("minio_path")
    if storage_key:
        return artifact_store.materialize(storage_key, suffix=os.path.splitext(storage_key)[1])
    return nullcontext(image.get("local_path") or "")


def _image_content_type(image: Dict[str, Any]) -> str:
    declared = _safe_text(image.get("content_type")).split(";", 1)[0].lower()
    if declared:
        return declared
    storage_key = _safe_text(image.get("storage_key") or image.get("minio_path"))
    if storage_key:
        try:
            return _safe_text(artifact_store.stat(storage_key).content_type).lower()
        except Exception:
            pass
    return "application/octet-stream"


def _sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_content_sha(image: Dict[str, Any], local_path: str) -> str:
    """Return the evidence bytes' SHA-256, preferring ArtifactStore metadata."""

    storage_key = str(image.get("storage_key") or image.get("minio_path") or "").strip()
    if storage_key:
        try:
            value = str(artifact_store.stat(storage_key).sha256 or "").strip().casefold()
            if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
                return value
        except Exception:
            # ``materialize`` already proved that the bytes are readable; hash
            # the local copy if metadata lookup is temporarily unavailable.
            pass
    return _sha256_file(local_path)


def _stable_image_id(doc_id: str, content_sha: str) -> str:
    """Deduplicate repeat parses while keeping identical bytes in other docs separate."""

    return f"{doc_id}_img_{content_sha}"


def _identity(value: Any) -> str:
    text = _safe_text(value).casefold()
    text = re.sub(r"^\d+(?:[.．]\d+)*\s*", "", text)
    return re.sub(r"[\s_\-—–/\\:：;；,，。·()（）\[\]【】]", "", text)


def _scope(ctx: ParseContext) -> Dict[str, str]:
    metadata = (ctx.common_tags or {}).get("document_metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    mapping = {
        "system": "applicable_system",
        "configuration": "applicable_configuration",
        "document_number": "document_number",
        "document_version": "document_version",
    }
    return {
        target: str(metadata.get(source) or "").strip()
        for target, source in mapping.items()
        if str(metadata.get(source) or "").strip()
    }


def _best_extraction(
    block_title: str,
    image_title: str,
    extractions: List[DiagramExtraction],
) -> Optional[DiagramExtraction]:
    query_tokens = [_identity(block_title), _identity(image_title)]
    best: Optional[DiagramExtraction] = None
    best_score = 0
    for item in extractions:
        values = [
            _identity(item.title),
            _identity(item.facts.get("source_section")),
            _identity(item.business_key),
        ]
        score = 0
        for query in query_tokens:
            for value in values:
                if not query or not value:
                    continue
                if query == value:
                    score = max(score, 1000 + len(value))
                elif query in value or value in query:
                    score = max(score, min(len(query), len(value)))
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 4 else None


def _revision_candidate_ids(
    all_candidate_ids: Iterable[str],
    deterministic_candidate_ids: Iterable[str],
    formal_deterministic_candidate_ids: Iterable[str],
    visual_candidate_ids: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Select canonical and evidence-only candidates without losing provenance.

    Once at least one explicit topology independently passes the formal
    validator, image vision remains attached to the revision as an isolated
    candidate instead of contributing a second START/END topology.  If no
    deterministic candidate passes, all candidates remain in the draft so an
    invalid explicit source cannot be hidden by preferential selection.
    """

    all_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in all_candidate_ids
            if value is not None and str(value).strip()
        )
    )
    deterministic_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in deterministic_candidate_ids
            if value is not None and str(value).strip()
        )
    )
    formal_ids = {
        str(value).strip()
        for value in formal_deterministic_candidate_ids
        if value is not None and str(value).strip()
    }
    visual_ids = list(
        dict.fromkeys(
            str(value).strip()
            for value in visual_candidate_ids
            if value is not None and str(value).strip()
        )
    )
    if formal_ids.intersection(deterministic_ids):
        return deterministic_ids, visual_ids
    return all_ids, []


@ParserRegistry.register("fta")
@ParserRegistry.register("rbd")
class DiagramParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".docx",) + _IMAGE_EXTS:
            return [f"{self.doc_type.upper()} 仅支持 .docx 或图片文件"]
        return []

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        blocks = self._load_blocks(file_path, ctx)
        if not blocks:
            raise ValueError("未解析到图文内容")

        graph_count = 0
        nodes_total = 0
        edges_total = 0
        vectors = 0
        text_blocks = 0
        failed: List[Dict[str, str]] = []
        projection_failed: List[Dict[str, str]] = []
        image_count = sum(len(block.get("images", [])) for _, block in blocks)
        extractions: List[DiagramExtraction] = []
        diagram_by_business_key: Dict[str, Dict[str, Any]] = {}
        candidate_ids: Dict[str, List[str]] = {}
        deterministic_candidate_ids: Dict[str, List[str]] = {}
        formal_deterministic_candidate_ids: Dict[str, List[str]] = {}
        vision_candidate_ids: Dict[str, List[str]] = {}
        present_diagram_ids: set[str] = set()

        # Explicit table/expression structures are parsed deterministically and
        # persisted as candidates before any vision call.  A model outage must
        # not make an explicit RBD expression or FTA cut-set table disappear.
        if os.path.splitext(file_path)[1].lower() == ".docx":
            document_ir = load_document_ir(file_path, source_name=ctx.file_name)
            extractions = (
                extract_fta_from_ir(document_ir)
                if ctx.doc_type == "fta"
                else extract_rbd_from_ir(document_ir)
            )
            for extraction in extractions:
                diagram = diagram_repository.upsert_diagram(
                    doc_id=ctx.doc_id,
                    diagram_type=ctx.doc_type,
                    diagram_key=extraction.business_key,
                    title=extraction.title,
                    source_section=str(extraction.facts.get("source_section") or ""),
                    scope=_scope(ctx),
                )
                raw_candidate = extraction.candidate
                raw_graph = dict(raw_candidate["graph"])
                raw_metadata = dict(raw_graph.get("metadata") or {})
                raw_metadata["extraction_warnings"] = list(extraction.warnings)
                raw_metadata["confidence"] = raw_candidate.get("confidence")
                raw_graph["metadata"] = raw_metadata
                candidate = diagram_repository.upsert_candidate(
                    diagram_id=str(diagram["diagram_id"]),
                    source_kind=str(raw_candidate["source_kind"]),
                    source_fingerprint=str(raw_candidate["source_fingerprint"]),
                    raw_graph=raw_graph,
                    source_locator={"ref": raw_candidate.get("source_locator", "")},
                    extractor_name="icp_deterministic_structure",
                    extractor_version="1",
                )
                diagram_by_business_key[extraction.business_key] = diagram
                present_diagram_ids.add(str(diagram["diagram_id"]))
                candidate_ids.setdefault(str(diagram["diagram_id"]), []).append(
                    str(candidate["candidate_id"])
                )
                deterministic_candidate_ids.setdefault(str(diagram["diagram_id"]), []).append(
                    str(candidate["candidate_id"])
                )
                if deterministic_candidate_allows_formal_topology(
                    diagram_type=ctx.doc_type,
                    source_kind=str(raw_candidate["source_kind"]),
                    graph=raw_graph,
                ):
                    formal_deterministic_candidate_ids.setdefault(
                        str(diagram["diagram_id"]), []
                    ).append(str(candidate["candidate_id"]))
                try:
                    facts_content = json.dumps(extraction.facts, ensure_ascii=False, default=str)
                    facts_key = hashlib.sha256(
                        f"{ctx.doc_type}\x1f{extraction.business_key}\x1ffacts".encode("utf-8")
                    ).hexdigest()[:24]
                    vectors += self._index(
                        ctx,
                        f"{ctx.doc_id}_{ctx.doc_type}_facts_{facts_key}",
                        extraction.title,
                        f"{ctx.doc_type.upper()} 显式结构事实（待图修订确认）：\n{facts_content}",
                        f"{ctx.doc_type}_structured_facts",
                        {
                            "business_id_kwd": extraction.business_key,
                            "diagram_id_kwd": str(diagram["diagram_id"]),
                            "projection_role_kwd": "diagram_locator",
                            "review_status_kwd": "pending",
                            "source_ref": raw_candidate.get("source_locator", ""),
                        },
                    )
                except Exception as exc:
                    failure = {
                        "diagram": extraction.title,
                        "reason": f"显式结构事实索引失败: {exc}",
                    }
                    failed.append(failure)
                    projection_failed.append(failure)
                for warning in extraction.warnings:
                    failed.append({"diagram": extraction.title, "reason": warning})

        for block_index, (block_key, block) in enumerate(blocks, start=1):
            title = _safe_text(block.get("title") or f"章节{block_index}")
            text = _safe_text(block.get("text"))
            table_texts = [
                _safe_text(table.get("markdown"))
                for table in block.get("tables", [])
                if table.get("markdown")
            ]
            context_text = "\n".join(part for part in [title, text, *table_texts] if part)

            if context_text:
                try:
                    vectors += self._index(
                        ctx=ctx,
                        chunk_id=f"{ctx.doc_id}_{ctx.doc_type}_text_{block_index}",
                        title=title,
                        content=(
                            f"{ctx.doc_type.upper()}文档章节：{title}\n"
                            f"说明文字：{text}\n"
                            f"相关表格：{'；'.join(table_texts)}"
                        ),
                        content_type=f"{ctx.doc_type}_text",
                        extra={"chapter_kwd": block_key},
                    )
                    text_blocks += 1
                except Exception as exc:
                    failure = {"block": block_key, "reason": f"说明文字索引失败: {exc}"}
                    failed.append(failure)
                    projection_failed.append(failure)

            for image_index, image in enumerate(block.get("images", []), start=1):
                image_title = _safe_text(image.get("title") or f"{title}-图{image_index}")
                ctx.progress(
                    f"[{ctx.doc_type.upper()} 图 {image_index}/{max(image_count, 1)}] 解析 {image_title}"
                )
                storage_key = str(
                    image.get("storage_key") or image.get("minio_path") or ""
                ).strip()
                image_url = str(image.get("image_url") or "").strip()
                mime_type = _image_content_type(image)
                image_source_ref = _safe_text(image.get("source_ref") or image_title)
                with _materialize_image(image) as local_path:
                    if not local_path or not os.path.exists(local_path):
                        raise FileNotFoundError("本地图片不存在")
                    fingerprint = _image_content_sha(image, local_path)
                    image_id = _stable_image_id(ctx.doc_id, fingerprint)

                    # Persist the source evidence before any model call.  The
                    # stable content identity makes this an idempotent upsert,
                    # and a vision outage cannot erase the original picture.
                    postgres_client.insert_image(
                        image_id,
                        ctx.doc_id,
                        None,
                        image_title,
                        f"{ctx.doc_type.upper()} 原始工程图证据；来源章节：{title}",
                        storage_key,
                        f"{ctx.doc_type}_diagram",
                        image_url,
                        mime_type,
                        image_source_ref,
                    )

                    matched = _best_extraction(title, image_title, extractions)
                    if matched:
                        diagram = diagram_by_business_key[matched.business_key]
                    else:
                        diagram = diagram_repository.upsert_diagram(
                            doc_id=ctx.doc_id,
                            diagram_type=ctx.doc_type,
                            diagram_key=f"IMAGE:{fingerprint}",
                            title=image_title,
                            source_section=title,
                            scope=_scope(ctx),
                        )
                    diagram_id = str(diagram["diagram_id"])
                    present_diagram_ids.add(diagram_id)

                    try:
                        analysis = analyze_icp_diagram(
                            image_path=local_path,
                            doc_type=ctx.doc_type,
                            title=image_title,
                            context_text=context_text,
                            model_config=ctx.model_config,
                        )
                        if not isinstance(analysis, dict):
                            raise ValueError("视觉模型未返回图对象")
                        vertices = analysis.get("vertices") or []
                        edges = analysis.get("edges") or []
                        if not isinstance(vertices, list) or not isinstance(edges, list):
                            raise ValueError("视觉模型返回的节点或关系不是数组")
                        if not vertices and not edges:
                            detail = _safe_text(analysis.get("description"))[:300]
                            failed.append(
                                {
                                    "image": image_title,
                                    "reason": "视觉解析未产生可审核节点/关系"
                                    + (f": {detail}" if detail else ""),
                                }
                            )
                    except Exception as exc:
                        reason = f"视觉解析失败: {exc}"
                        failed.append({"image": image_title, "reason": reason})
                        # A discovered source image must have a durable terminal
                        # extraction outcome.  Otherwise the diagram master has
                        # no revision and the document remains ``kg_pending``
                        # forever even though the safe result is simply an
                        # empty, quarantined visual candidate.  Persisting an
                        # empty failed candidate records the failure without
                        # fabricating any topology; automatic validation will
                        # quarantine the resulting revision.
                        failed_graph = {
                            "diagram_type": ctx.doc_type,
                            "root_fault_id": "",
                            "metadata": {
                                "source_block": block_key,
                                "image_url": image_url,
                                "storage_key": storage_key,
                                "minio_path": storage_key,
                                "mime_type": mime_type,
                                "source_ref": image_source_ref,
                            },
                            "vertices": [],
                            "edges": [],
                        }
                        candidate = diagram_repository.upsert_candidate(
                            diagram_id=diagram_id,
                            source_kind="image_vision",
                            source_fingerprint=fingerprint,
                            raw_graph=failed_graph,
                            source_locator={
                                "block": block_key,
                                "image_id": image_id,
                                "storage_key": storage_key,
                                "image_url": image_url,
                                "mime_type": mime_type,
                                "source_ref": image_source_ref,
                                "extraction_error": str(exc)[:500],
                            },
                            extractor_name="analyze_icp_diagram",
                            extractor_version="1",
                            extraction_status="failed",
                        )
                        candidate_id = str(candidate["candidate_id"])
                        candidate_ids.setdefault(diagram_id, []).append(candidate_id)
                        vision_candidate_ids.setdefault(diagram_id, []).append(candidate_id)
                        continue

                graph = {
                    "diagram_type": ctx.doc_type,
                    "root_fault_id": analysis.get("root_id") or "",
                    "metadata": {
                        **(analysis.get("metadata") or {}),
                        "description": analysis.get("description", ""),
                        "confidence": analysis.get("confidence", 0.0),
                        "source_block": block_key,
                        "image_url": image_url,
                        "storage_key": storage_key,
                        "minio_path": storage_key,
                        "mime_type": mime_type,
                        "source_ref": image_source_ref,
                    },
                    "vertices": vertices,
                    "edges": edges,
                }

                model_name = ""
                if isinstance(ctx.model_config, dict):
                    model_name = str(
                        ctx.model_config.get("model_name")
                        or ctx.model_config.get("name")
                        or ctx.model_config.get("model")
                        or ""
                    )
                candidate = diagram_repository.upsert_candidate(
                    diagram_id=diagram_id,
                    source_kind="image_vision",
                    source_fingerprint=fingerprint,
                    raw_graph=graph,
                    source_locator={
                        "block": block_key,
                        "image_id": image_id,
                        "storage_key": storage_key,
                        "image_url": image_url,
                        "mime_type": mime_type,
                        "source_ref": image_source_ref,
                    },
                    extractor_name="analyze_icp_diagram",
                    extractor_version="1",
                    model_name=model_name or None,
                    extraction_status=("succeeded" if vertices or edges else "skipped"),
                )
                candidate_ids.setdefault(diagram_id, []).append(
                    str(candidate["candidate_id"])
                )
                vision_candidate_ids.setdefault(diagram_id, []).append(
                    str(candidate["candidate_id"])
                )

                try:
                    vectors += self._index_graph(
                        ctx=ctx,
                        image_id=image_id,
                        image_title=image_title,
                        analysis=analysis,
                        vertices=vertices,
                        edges=edges,
                        image_url=image_url,
                        block_key=block_key,
                        diagram_id=diagram_id,
                    )
                except Exception as exc:
                    failure = {"image": image_title, "reason": f"图向量化失败: {exc}"}
                    failed.append(failure)
                    projection_failed.append(failure)

        review_counts: Dict[str, int] = {}
        projection_counts: Dict[str, int] = {}
        for diagram_id, ids in candidate_ids.items():
            canonical_ids, isolated_ids = _revision_candidate_ids(
                ids,
                deterministic_candidate_ids.get(diagram_id, []),
                formal_deterministic_candidate_ids.get(diagram_id, []),
                vision_candidate_ids.get(diagram_id, []),
            )
            revision = diagram_repository.create_or_reuse_revision(
                diagram_id=diagram_id,
                candidate_ids=canonical_ids,
                isolated_candidate_ids=isolated_ids,
            )
            graph_count += 1
            graph = revision.get("canonical_graph") or {}
            nodes_total += len(graph.get("vertices") or [])
            edges_total += len(graph.get("edges") or [])
            review_status = str(revision.get("review_status") or "pending")
            projection_status = str(revision.get("projection_status") or "not_projected")
            review_counts[review_status] = review_counts.get(review_status, 0) + 1
            projection_counts[projection_status] = projection_counts.get(projection_status, 0) + 1

        # This is the commit marker for the parser's complete diagram snapshot.
        # Every authority write above is allowed to raise; OpenSearch and vision
        # failures are recorded separately and never become graph facts.  Only
        # after all candidates/revisions are durable may missing old diagrams be
        # retired.  FTA/RBD input with no explicit structure and no source image
        # is treated as an incomplete/unsupported parse, not as permission to
        # retire every previously reviewed graph.
        if not present_diagram_ids:
            raise ValueError(
                f"未识别到任何 {ctx.doc_type.upper()} 显式结构或原始工程图；保留上一版图快照"
            )
        snapshot = diagram_repository.archive_missing_diagrams(
            doc_id=ctx.doc_id,
            present_diagram_ids=sorted(present_diagram_ids),
            allow_empty=False,
        )

        needs_review_or_projection = any(
            status != "published" for status, count in projection_counts.items() if count
        )

        stats = {
            "images": image_count,
            "graph_count": graph_count,
            "nodes": nodes_total,
            "edges": edges_total,
            "text_blocks": text_blocks,
            "vectors": vectors,
            "failed": failed[:100],
            "projection_failed": projection_failed[:100],
            "review_status_counts": review_counts,
            "projection_status_counts": projection_counts,
            "present_diagrams": len(present_diagram_ids),
            "archived_diagrams": len(snapshot.get("archived_diagram_ids") or []),
            "stale_revisions": len(snapshot.get("stale_revision_ids") or []),
            "review_pending": any(
                review_counts.get(status, 0) for status in ("pending", "conflicted")
            ),
            "kg_pending": graph_count > 0 and needs_review_or_projection,
        }
        message = (
            f"解析完成：{image_count} 张证据图，{graph_count} 个待自动处理 {ctx.doc_type.upper()} 修订，"
            f"{nodes_total} 节点 / {edges_total} 关系候选，{text_blocks} 个说明块，向量化 {vectors} 条"
        )
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=message,
        )

    def _load_blocks(self, file_path: str, ctx: ParseContext) -> List[Tuple[str, Dict[str, Any]]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _IMAGE_EXTS:
            source_name = source_file_name(file_path, ctx.file_name)
            source_title = os.path.splitext(source_name)[0]
            storage_key = f"images/{ctx.doc_id}/source{ext}"
            content_type = mimetypes.guess_type(ctx.file_name or file_path)[0] or "application/octet-stream"
            with open(file_path, "rb") as source:
                artifact_store.upload_file(
                    storage_key,
                    source,
                    content_type=content_type,
                    original_name=source_name,
                )
            return [
                (
                    "image_001",
                    {
                        "title": source_title,
                        "text": "",
                        "tables": [],
                        "images": [
                            {
                                "local_path": "",
                                "title": source_title,
                                "image_url": f"{config.IMAGE_URL_PREFIX}/{ctx.doc_id}/source{ext}",
                                "storage_key": storage_key,
                                "minio_path": storage_key,
                                "content_type": content_type,
                                "source_ref": f"{source_name}#direct-image",
                            }
                        ],
                    },
                )
            ]

        result = DocumentParser(
            file_path, ctx.doc_id, source_name=ctx.file_name
        ).process()
        if result.get("error"):
            raise ValueError(result["error"])
        return list((result.get("blocks") or {}).items())

    def _index(
        self,
        ctx: ParseContext,
        chunk_id: str,
        title: str,
        content: str,
        content_type: str,
        extra: Dict[str, Any] | None = None,
    ) -> int:
        if not content.strip():
            return 0
        vector = embedding_service.embed(content)
        body = ctx.enrich({
            "chunk_id": chunk_id,
            "doc_id": ctx.doc_id,
            "doc_name": ctx.file_name,
            "title": title,
            "section": ctx.doc_type,
            "content": content,
            "kb_id": ctx.doc_type,
            "content_type": content_type,
            "doc_type_kwd": ctx.doc_type,
            "removed_kwd": "N",
            "available_int": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **(extra or {}),
        })
        search_store.upsert(chunk_id, body, vector=vector)
        return 1

    def _index_graph(
        self,
        ctx: ParseContext,
        image_id: str,
        image_title: str,
        analysis: Dict[str, Any],
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        image_url: str,
        block_key: str,
        diagram_id: str,
    ) -> int:
        """Index only a candidate summary, never unsupported node/edge claims."""
        names = {str(vertex.get("id")): _vertex_name(vertex) for vertex in vertices}
        whole_content = (
            f"{ctx.doc_type.upper()}工程图视觉候选（自动分级）：{image_title}\n"
            f"图意：{analysis.get('description', '')}\n"
            f"候选节点数：{len(vertices)}；候选关系数：{len(edges)}\n"
            f"候选关键节点：{', '.join(name for name in names.values() if name)[:1200]}\n"
            "用途限制：仅用于定位原图和自动校验；provisional 不得用于正式拓扑或数值计算。"
        )
        return self._index(
            ctx,
            image_id,
            image_title,
            whole_content,
            f"{ctx.doc_type}_diagram_candidate_summary",
            {
                "image_url": image_url,
                "chapter_kwd": block_key,
                "diagram_id_kwd": diagram_id,
                "projection_role_kwd": "diagram_locator",
                "diagram_confidence": analysis.get("confidence", 0.0),
                "review_status_kwd": "pending",
            },
        )
