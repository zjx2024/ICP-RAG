"""图类解析器（fta / rbd）：提取图片 → 复用 vision 分析（analyze_image）→ 三元组存 document_fault_trees → 整图描述 + 节点/边逐条向量化写检索投影。

参照现有故障树处理链路（document_service 内 insert_document_fault_tree），
升级为：fault_tree_graph 的 vertices/edges 逐条展平向量化，使顶事件/中间事件/底事件均可被语义召回。
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple
from uuid import uuid4

from backend.core.embedding import embedding_service
from backend.core.llm_service import llm_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


@ParserRegistry.register("fta")
@ParserRegistry.register("rbd")
class FaultTreeParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        errors = []
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".docx",) + _IMAGE_EXTS:
            errors.append(f"不支持的格式: {ext}，图类文档仅支持 .docx / 图片文件")
        return errors

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        images = self._extract_images(file_path, ctx)
        if not images:
            raise ValueError("未在文档中找到图片内容")

        graph_count = 0
        nodes_total = 0
        edges_total = 0
        vectors = 0

        for i, (img_path, img_title) in enumerate(images, start=1):
            ctx.progress(f"[图片 {i}/{len(images)}] 分析: {img_title or img_path.split('/')[-1]}")
            analysis = llm_service.analyze_image(
                img_path, img_title, section=ctx.doc_type,
                context_text="", model_config=ctx.model_config)
            description = analysis.get("description", "")
            graph = analysis.get("fault_tree_graph") or {}
            image_id = f"{ctx.doc_id}_img_{i}"

            # 1. 整图描述向量化
            whole_content = f"图片: {img_title}\n{description}"
            try:
                vector = embedding_service.embed(whole_content)
                doc = ctx.enrich({
                    "chunk_id": image_id,
                    "doc_id": ctx.doc_id,
                    "doc_name": ctx.file_name,
                    "title": img_title,
                    "section": ctx.doc_type,
                    "content": whole_content,
                    "kb_id": ctx.kb_id or ctx.doc_type,
                    "content_type": "fault_tree_image",
                    "doc_type_kwd": ctx.doc_type,
                    "removed_kwd": "N",
                    "available_int": 1,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                })
                search_store.upsert(image_id, doc, vector=vector)
                vectors += 1
            except Exception as e:
                ctx.progress(f"[图片 {i}] 整图向量化失败: {str(e)[:100]}")

            # 2. 三元组入库（复用 document_fault_trees，kb_id 语义存 doc_type）
            vertices = graph.get("vertices", []) or []
            edges = graph.get("edges", []) or []
            if vertices or edges:
                fault_tree_id = str(uuid4())
                postgres_client.insert_document_fault_tree(
                    fault_tree_id, ctx.doc_id, ctx.kb_id or ctx.doc_type,
                    image_id, image_id, img_title,
                    json.dumps(graph, ensure_ascii=False))
                graph_count += 1
                nodes_total += len(vertices)
                edges_total += len(edges)

                # 3. 节点/边逐条向量化
                node_ids = [v.get("id", f"v_{idx}") for idx, v in enumerate(vertices)]
                edge_parent = {}
                edge_gate = {}
                for e in edges:
                    edge_parent[e.get("target", "")] = e.get("source", "")
                    gate = e.get("properties", {})
                    edge_gate[e.get("target", "")] = gate.get("gate_type", "") if isinstance(gate, dict) else ""

                paths = self._build_paths(vertices, edges, graph.get("root_fault_id"))
                name_of = {v.get("id", f"v_{idx}"): (v.get("properties", {}) or {}).get("name", "") if isinstance(v.get("properties"), dict) else "" for idx, v in enumerate(vertices)}

                for idx, v in enumerate(vertices):
                    vid = v.get("id", f"v_{idx}")
                    props = v.get("properties", {}) or {}
                    vname = props.get("name", "") if isinstance(props, dict) else ""
                    vdesc = props.get("description", "") if isinstance(props, dict) else ""
                    vlabel = v.get("label", "Fault")
                    gate = edge_gate.get(vid, "")
                    path = paths.get(vid, [vid])
                    chapter = " / ".join(name_of.get(p, p) for p in path)
                    content = (f"故障节点: {vname}（{vlabel}）\n描述: {vdesc}\n"
                               f"父节点: {name_of.get(edge_parent.get(vid, ''), '')}\n"
                               f"逻辑门: {gate or '无'}")
                    try:
                        vector = embedding_service.embed(content)
                        doc = ctx.enrich({
                            "chunk_id": f"{image_id}_node_{idx}",
                            "doc_id": ctx.doc_id,
                            "doc_name": ctx.file_name,
                            "title": name_of.get(graph.get("root_fault_id", ""), vname) or vname,
                            "chapter": chapter,
                            "section": ctx.doc_type,
                            "content": content,
                            "kb_id": ctx.kb_id or ctx.doc_type,
                            "content_type": "fault_tree_node",
                            "doc_type_kwd": ctx.doc_type,
                            "business_id_kwd": vname,
                            "parent_kwd": name_of.get(edge_parent.get(vid, ""), ""),
                            "node_type_kwd": vlabel,
                            "removed_kwd": "N",
                            "available_int": 1,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        })
                        search_store.upsert(doc["chunk_id"], doc, vector=vector)
                        vectors += 1
                    except Exception as e:
                        ctx.progress(f"[节点 {vname}] 向量化失败: {str(e)[:100]}")

                for idx, e in enumerate(edges):
                    src = e.get("source", "")
                    tgt = e.get("target", "")
                    elabel = e.get("label", "causes")
                    gate = (e.get("properties", {}) or {}).get("gate_type", "") if isinstance(e.get("properties"), dict) else ""
                    content = f"故障关系: {name_of.get(src, src)} → {name_of.get(tgt, tgt)}（{elabel}，门类型: {gate or '直接原因'}）"
                    try:
                        vector = embedding_service.embed(content)
                        doc = ctx.enrich({
                            "chunk_id": f"{image_id}_edge_{idx}",
                            "doc_id": ctx.doc_id,
                            "doc_name": ctx.file_name,
                            "title": name_of.get(graph.get("root_fault_id", ""), name_of.get(src, "")) or name_of.get(src, ""),
                            "section": ctx.doc_type,
                            "content": content,
                            "kb_id": ctx.kb_id or ctx.doc_type,
                            "content_type": "fault_tree_edge",
                            "doc_type_kwd": ctx.doc_type,
                            "business_id_kwd": name_of.get(src, ""),
                            "parent_kwd": name_of.get(src, ""),
                            "removed_kwd": "N",
                            "available_int": 1,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        })
                        search_store.upsert(doc["chunk_id"], doc, vector=vector)
                        vectors += 1
                    except Exception as e:
                        ctx.progress(f"[关系 {idx}] 向量化失败: {str(e)[:100]}")
            else:
                ctx.progress(f"[图片 {i}] 未识别到故障树结构（{analysis.get('image_type', 'unknown')}），仅保留整图描述")

        stats = {
            "images": len(images),
            "graph_count": graph_count,
            "nodes": nodes_total,
            "edges": edges_total,
            "vectors": vectors,
            "kg_pending": graph_count > 0,
        }
        message = (f"解析完成：识别 {len(images)} 张图，{graph_count} 张含故障树结构"
                   f"（{nodes_total} 节点 / {edges_total} 关系），向量化 {vectors} 条")
        return ParseResult(doc_id=ctx.doc_id, doc_type=ctx.doc_type, stats=stats,
                           vectors=vectors, message=message)

    # ---------- 内部实现 ----------

    def _extract_images(self, file_path: str, ctx: ParseContext) -> List[Tuple[str, str]]:
        """返回 [(本地图片路径, 标题)]。docx 提取内嵌图片；直接图片文件原样返回。"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _IMAGE_EXTS:
            return [(file_path, os.path.splitext(os.path.basename(file_path))[0])]

        from docx import Document
        doc = Document(file_path)
        images: List[Tuple[str, str]] = []
        tmp_dir = f"/tmp/icp_{ctx.doc_id}"
        os.makedirs(tmp_dir, exist_ok=True)
        for rel in doc.part.rels.values():
            if "image" not in (rel.reltype or ""):
                continue
            part = rel.target_part
            blob = part.blob
            content_type = getattr(part, "content_type", "") or ""
            sub_ext = "png"
            if "jpeg" in content_type or "jpg" in content_type:
                sub_ext = "jpg"
            elif "gif" in content_type:
                sub_ext = "gif"
            local_path = os.path.join(tmp_dir, f"img_{len(images)}_{uuid4().hex[:6]}.{sub_ext}")
            with open(local_path, "wb") as f:
                f.write(blob)
            images.append((local_path, f"图片{len(images) + 1}"))
        return images

    @staticmethod
    def _build_paths(vertices: List[Dict], edges: List[Dict], root_id: str) -> Dict[str, List[str]]:
        """由根节点 BFS 构建每个节点的层级路径（chapter 用）。"""
        children: Dict[str, List[str]] = {}
        for e in edges:
            children.setdefault(e.get("source", ""), []).append(e.get("target", ""))
        paths: Dict[str, List[str]] = {}
        start = root_id or (vertices[0].get("id", "") if vertices else "")
        if not start:
            return paths

        def walk(node_id: str, path: List[str]):
            if len(path) > 64 or node_id in paths:
                return
            paths[node_id] = list(path)
            for c in children.get(node_id, []):
                walk(c, path + [c])

        walk(start, [start])
        return paths
