"""FTA / RBD 专用图文解析器。

Word 文档按章节保留邻近段落、表格和图题；图片分别按 FTA 或 RBD 语义解析。
结构化图保存到 PostgreSQL，图摘要、节点、关系和说明段落写入 Elasticsearch。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.icp.image_analysis import analyze_icp_diagram
from backend.storage.es_client import es_client
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _vertex_name(vertex: Dict[str, Any]) -> str:
    props = vertex.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    return _safe_text(props.get("name") or vertex.get("name") or vertex.get("id"))


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

        index_name = es_client.get_main_index_name("default")
        es_client.create_index(index_name)

        graph_count = 0
        nodes_total = 0
        edges_total = 0
        vectors = 0
        text_blocks = 0
        failed: List[Dict[str, str]] = []
        image_count = sum(len(block.get("images", [])) for _, block in blocks)

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
                        index_name=index_name,
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
                    failed.append({"block": block_key, "reason": f"说明文字索引失败: {exc}"})

            for image_index, image in enumerate(block.get("images", []), start=1):
                local_path = image.get("local_path") or ""
                image_title = _safe_text(image.get("title") or f"{title}-图{image_index}")
                if not local_path or not os.path.exists(local_path):
                    failed.append({"image": image_title, "reason": "本地图片不存在"})
                    continue

                ctx.progress(
                    f"[{ctx.doc_type.upper()} 图 {image_index}/{max(image_count, 1)}] 解析 {image_title}"
                )
                try:
                    analysis = analyze_icp_diagram(
                        image_path=local_path,
                        doc_type=ctx.doc_type,
                        title=image_title,
                        context_text=context_text,
                        model_config=ctx.model_config,
                    )
                except Exception as exc:
                    failed.append({"image": image_title, "reason": f"视觉解析失败: {exc}"})
                    continue

                vertices = analysis.get("vertices") or []
                edges = analysis.get("edges") or []
                graph = {
                    "diagram_type": ctx.doc_type,
                    "root_fault_id": analysis.get("root_id") or "",
                    "metadata": {
                        **(analysis.get("metadata") or {}),
                        "description": analysis.get("description", ""),
                        "confidence": analysis.get("confidence", 0.0),
                        "source_block": block_key,
                        "image_url": image.get("image_url", ""),
                        "minio_path": image.get("minio_path", ""),
                    },
                    "vertices": vertices,
                    "edges": edges,
                }

                image_id = f"{ctx.doc_id}_{ctx.doc_type}_img_{block_index}_{image_index}"
                try:
                    if vertices or edges:
                        postgres_client.insert_document_fault_tree(
                            str(uuid4()),
                            ctx.doc_id,
                            ctx.doc_type,
                            block_key,
                            image_id,
                            image_title,
                            json.dumps(graph, ensure_ascii=False),
                        )
                        graph_count += 1
                        nodes_total += len(vertices)
                        edges_total += len(edges)
                except Exception as exc:
                    failed.append({"image": image_title, "reason": f"图结构入库失败: {exc}"})

                try:
                    vectors += self._index_graph(
                        index_name=index_name,
                        ctx=ctx,
                        image_id=image_id,
                        image_title=image_title,
                        analysis=analysis,
                        vertices=vertices,
                        edges=edges,
                        image_url=image.get("image_url", ""),
                        block_key=block_key,
                    )
                except Exception as exc:
                    failed.append({"image": image_title, "reason": f"图向量化失败: {exc}"})

        stats = {
            "images": image_count,
            "graph_count": graph_count,
            "nodes": nodes_total,
            "edges": edges_total,
            "text_blocks": text_blocks,
            "vectors": vectors,
            "failed": failed[:100],
            "kg_pending": graph_count > 0,
        }
        message = (
            f"解析完成：{image_count} 张图，{graph_count} 个结构化 {ctx.doc_type.upper()} 模型，"
            f"{nodes_total} 节点 / {edges_total} 关系，{text_blocks} 个说明块，向量化 {vectors} 条"
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
            return [
                (
                    "image_001",
                    {
                        "title": os.path.splitext(os.path.basename(file_path))[0],
                        "text": "",
                        "tables": [],
                        "images": [
                            {
                                "local_path": file_path,
                                "title": os.path.splitext(os.path.basename(file_path))[0],
                                "image_url": "",
                                "minio_path": "",
                            }
                        ],
                    },
                )
            ]

        result = DocumentParser(file_path, ctx.doc_id).process()
        if result.get("error"):
            raise ValueError(result["error"])
        return list((result.get("blocks") or {}).items())

    def _index(
        self,
        index_name: str,
        ctx: ParseContext,
        chunk_id: str,
        title: str,
        content: str,
        content_type: str,
        extra: Dict[str, Any] | None = None,
    ) -> int:
        if not content.strip():
            return 0
        body = {
            "chunk_id": chunk_id,
            "doc_id": ctx.doc_id,
            "doc_name": ctx.file_name,
            "title": title,
            "section": ctx.doc_type,
            "content": content,
            "content_vector": embedding_service.embed(content),
            "kb_id": ctx.doc_type,
            "content_type": content_type,
            "doc_type_kwd": ctx.doc_type,
            "removed_kwd": "N",
            "available_int": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **(extra or {}),
        }
        es_client.client.index(index=index_name, id=chunk_id, body=body)
        return 1

    def _index_graph(
        self,
        index_name: str,
        ctx: ParseContext,
        image_id: str,
        image_title: str,
        analysis: Dict[str, Any],
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        image_url: str,
        block_key: str,
    ) -> int:
        count = 0
        names = {str(vertex.get("id")): _vertex_name(vertex) for vertex in vertices}
        whole_content = (
            f"{ctx.doc_type.upper()}图：{image_title}\n"
            f"图意：{analysis.get('description', '')}\n"
            f"成功/顶事件信息：{json.dumps(analysis.get('metadata') or {}, ensure_ascii=False)}\n"
            f"关键节点：{', '.join(name for name in names.values() if name)[:1200]}"
        )
        count += self._index(
            index_name,
            ctx,
            image_id,
            image_title,
            whole_content,
            f"{ctx.doc_type}_diagram",
            {
                "image_url": image_url,
                "chapter_kwd": block_key,
                "diagram_confidence": analysis.get("confidence", 0.0),
            },
        )

        parent_names: Dict[str, List[str]] = {}
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            parent_names.setdefault(target, []).append(names.get(source, source))

        for index, vertex in enumerate(vertices):
            vertex_id = str(vertex.get("id") or f"v{index}")
            props = vertex.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            name = names.get(vertex_id, vertex_id)
            content = (
                f"{ctx.doc_type.upper()}节点：{name}\n类型：{vertex.get('label', '')}\n"
                f"描述：{props.get('description', '')}\n上游节点：{', '.join(parent_names.get(vertex_id, []))}"
            )
            count += self._index(
                index_name,
                ctx,
                f"{image_id}_node_{index}",
                name,
                content,
                f"{ctx.doc_type}_node",
                {
                    "business_id_kwd": name,
                    "node_type_kwd": vertex.get("label", ""),
                    "parent_kwd": ",".join(parent_names.get(vertex_id, [])),
                    "chapter_kwd": block_key,
                },
            )

        for index, edge in enumerate(edges):
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            props = edge.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            source_name = names.get(source, source)
            target_name = names.get(target, target)
            content = (
                f"{ctx.doc_type.upper()}关系：{source_name} → {target_name}\n"
                f"关系类型：{edge.get('label', '')}\n"
                f"逻辑/连接属性：{json.dumps(props, ensure_ascii=False)}"
            )
            count += self._index(
                index_name,
                ctx,
                f"{image_id}_edge_{index}",
                f"{source_name} → {target_name}",
                content,
                f"{ctx.doc_type}_edge",
                {
                    "business_id_kwd": source_name,
                    "parent_kwd": source_name,
                    "target_kwd": target_name,
                    "chapter_kwd": block_key,
                },
            )
        return count
