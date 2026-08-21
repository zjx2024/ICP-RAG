"""系统架构与功能文档解析器（Function Locator 数据源）。

输入通常是“架构图 + 模块/功能说明段落 + 少量表格”。解析器同时保存：
- 可追溯的原始说明分块（ES）
- 标准化系统/模块/功能候选（PostgreSQL + ES）
- 架构图节点与关系（PostgreSQL JSON + ES）
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.icp.image_analysis import analyze_icp_diagram
from backend.icp.repository import icp_repository
from backend.storage.es_client import es_client
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

_FIELD_ALIASES = {
    "function_id": ["功能ID", "功能编号", "功能代码", "功能标识", "FUNCID", "FUNCTIONID"],
    "function_name": ["功能名称", "功能名", "功能", "FUNCTIONNAME"],
    "system_name": ["系统名称", "所属系统", "系统", "SYSTEM"],
    "subsystem_name": ["子系统名称", "所属子系统", "子系统", "SUBSYSTEM"],
    "module_name": ["模块名称", "所属模块", "模块", "部件", "MODULE"],
    "description": ["功能描述", "说明", "描述", "功能说明", "DESCRIPTION"],
    "aliases": ["别名", "简称", "同义词", "ALIASES"],
    "failure_symptoms": ["功能失效表现", "失效表现", "典型现象", "异常现象", "故障现象"],
}

_PARAGRAPH_FIELDS = {
    "功能ID": "function_id",
    "功能编号": "function_id",
    "功能代码": "function_id",
    "功能名称": "function_name",
    "所属系统": "system_name",
    "系统名称": "system_name",
    "所属子系统": "subsystem_name",
    "子系统名称": "subsystem_name",
    "所属模块": "module_name",
    "模块名称": "module_name",
    "功能描述": "description",
    "功能说明": "description",
    "别名": "aliases",
    "简称": "aliases",
    "功能失效表现": "failure_symptoms",
    "失效表现": "failure_symptoms",
    "典型现象": "failure_symptoms",
}

_FIELD_RE = re.compile(r"^([^：:]{2,30})\s*[:：]\s*(.*)$")
_ID_RE = re.compile(r"\b(?:FUNC|FUNCTION|F)[-_]?[A-Za-z0-9][A-Za-z0-9_.\-/]{1,40}\b", re.IGNORECASE)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _norm(value: Any) -> str:
    text = _clean(value).upper()
    for ch in (" ", "_", "-", "/", "\\", "：", ":", "（", "）", "(", ")"):
        text = text.replace(ch, "")
    return text


def _alias_map() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            result[_norm(alias)] = canonical
    return result


def _match_header(value: str, aliases: Dict[str, str]) -> Optional[str]:
    norm = _norm(value)
    if not norm:
        return None
    if norm in aliases:
        return aliases[norm]
    matches = [(alias, field) for alias, field in aliases.items() if alias and alias in norm]
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches[0][1] if matches else None


def _parse_aliases(value: Any) -> List[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[、,，;；/|]", text) if part.strip()]


def _stable_id(doc_id: str, prefix: str, index: int) -> str:
    return f"{prefix}-{doc_id[:8].upper()}-{index:04d}"


@ParserRegistry.register("function_locator")
class FunctionArchitectureParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".docx", ".txt"):
            return ["系统架构与功能文档仅支持 .docx/.txt"]
        return []

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        index_name = es_client.get_main_index_name("default")
        es_client.create_index(index_name)
        icp_repository.init_tables()

        if os.path.splitext(file_path)[1].lower() == ".txt":
            blocks = self._text_blocks(file_path)
        else:
            result = DocumentParser(file_path, ctx.doc_id).process()
            if result.get("error"):
                raise ValueError(result["error"])
            blocks = list((result.get("blocks") or {}).items())

        if not blocks:
            raise ValueError("未解析到系统架构或功能说明内容")

        function_count = 0
        graph_count = 0
        nodes_total = 0
        edges_total = 0
        vectors = 0
        failed: List[Dict[str, str]] = []
        generated_ids = set()

        for block_index, (block_key, block) in enumerate(blocks, start=1):
            title = _clean(block.get("title") or f"章节{block_index}")
            text = _clean(block.get("text"))
            table_texts = [
                _clean(table.get("markdown"))
                for table in block.get("tables", [])
                if table.get("markdown")
            ]
            context_text = "\n".join(part for part in [title, text, *table_texts] if part)

            if context_text:
                try:
                    vectors += self._index(
                        index_name,
                        ctx,
                        f"{ctx.doc_id}_function_text_{block_index}",
                        title,
                        f"系统架构/功能章节：{title}\n{text}\n相关表格：{'；'.join(table_texts)}",
                        "function_text",
                        {"chapter_kwd": block_key},
                    )
                except Exception as exc:
                    failed.append({"block": block_key, "reason": f"说明文字索引失败: {exc}"})

            # 1. 从 Word 表格提取标准功能记录。
            for table_index, table in enumerate(block.get("tables", []), start=1):
                data = (table.get("meta_data") or {}).get("data", [])
                for item in self._functions_from_table(data, ctx.doc_id, block_index * 1000 + table_index * 100):
                    try:
                        if item["function_id"] in generated_ids:
                            continue
                        item["source_ref"] = f"{ctx.file_name}#{block_key}/table-{table_index}"
                        row_id = icp_repository.upsert_function(ctx.doc_id, item)
                        generated_ids.add(item["function_id"])
                        function_count += 1
                        vectors += self._index_function(index_name, ctx, row_id, item)
                    except Exception as exc:
                        failed.append({"block": block_key, "reason": f"功能表行入库失败: {exc}"})

            # 2. 从结构化段落或标题提取功能记录。
            for item in self._functions_from_paragraph(title, text, ctx.doc_id, block_index):
                try:
                    if item["function_id"] in generated_ids:
                        continue
                    item["source_ref"] = f"{ctx.file_name}#{block_key}"
                    row_id = icp_repository.upsert_function(ctx.doc_id, item)
                    generated_ids.add(item["function_id"])
                    function_count += 1
                    vectors += self._index_function(index_name, ctx, row_id, item)
                except Exception as exc:
                    failed.append({"block": block_key, "reason": f"功能段落入库失败: {exc}"})

            # 3. 解析系统架构/功能图片，并从 Function 节点生成 Function Locator 数据。
            for image_index, image in enumerate(block.get("images", []), start=1):
                local_path = image.get("local_path") or ""
                image_title = _clean(image.get("title") or f"{title}-图{image_index}")
                if not local_path or not os.path.exists(local_path):
                    failed.append({"image": image_title, "reason": "本地图片不存在"})
                    continue
                try:
                    analysis = analyze_icp_diagram(
                        local_path,
                        "function_locator",
                        image_title,
                        context_text,
                        ctx.model_config,
                    )
                    vertices = analysis.get("vertices") or []
                    edges = analysis.get("edges") or []
                    graph = {
                        "diagram_type": "function_locator",
                        "root_fault_id": analysis.get("root_id") or "",
                        "metadata": {
                            **(analysis.get("metadata") or {}),
                            "description": analysis.get("description", ""),
                            "confidence": analysis.get("confidence", 0.0),
                            "image_url": image.get("image_url", ""),
                            "source_block": block_key,
                        },
                        "vertices": vertices,
                        "edges": edges,
                    }
                    image_id = f"{ctx.doc_id}_architecture_{block_index}_{image_index}"
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

                    vectors += self._index_graph(
                        index_name,
                        ctx,
                        image_id,
                        image_title,
                        analysis,
                        vertices,
                        edges,
                        image.get("image_url", ""),
                        block_key,
                    )
                    for item in self._functions_from_graph(
                        vertices,
                        edges,
                        ctx.doc_id,
                        block_index * 1000 + image_index * 100,
                    ):
                        if item["function_id"] in generated_ids:
                            continue
                        item["source_ref"] = f"{ctx.file_name}#{block_key}/{image_title}"
                        row_id = icp_repository.upsert_function(ctx.doc_id, item)
                        generated_ids.add(item["function_id"])
                        function_count += 1
                        vectors += self._index_function(index_name, ctx, row_id, item)
                except Exception as exc:
                    failed.append({"image": image_title, "reason": f"架构图解析失败: {exc}"})

        stats = {
            "functions": function_count,
            "graph_count": graph_count,
            "nodes": nodes_total,
            "edges": edges_total,
            "vectors": vectors,
            "failed": failed[:100],
            "kg_pending": graph_count > 0,
        }
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=(
                f"解析完成：功能记录 {function_count} 条，架构图 {graph_count} 个，"
                f"{nodes_total} 节点 / {edges_total} 关系，向量化 {vectors} 条"
            ),
        )

    @staticmethod
    def _text_blocks(file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        for encoding in ("utf-8", "gb18030", "gbk"):
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    text = handle.read()
                return [("text_001", {"title": "系统架构与功能说明", "text": text, "tables": [], "images": []})]
            except UnicodeDecodeError:
                continue
        raise ValueError("文本文件编码无法识别")

    def _functions_from_table(
        self,
        rows: Sequence[Sequence[Any]],
        doc_id: str,
        offset: int,
    ) -> Iterable[Dict[str, Any]]:
        if not rows:
            return []
        aliases = _alias_map()
        max_scan = min(8, len(rows))
        best: Optional[Tuple[int, Dict[str, int]]] = None
        best_score = -1
        for start in range(max_scan):
            for depth in range(1, min(4, max_scan - start) + 1):
                segment = rows[start:start + depth]
                width = max((len(row) for row in segment), default=0)
                mapping: Dict[str, int] = {}
                for col in range(width):
                    parts = [_clean(row[col]) for row in segment if col < len(row) and _clean(row[col])]
                    canonical = None
                    for part in reversed(parts):
                        canonical = aliases.get(_norm(part))
                        if canonical:
                            break
                    if not canonical:
                        canonical = _match_header(" ".join(parts), aliases)
                    if canonical and canonical not in mapping:
                        mapping[canonical] = col
                if "function_name" in mapping and len(mapping) > best_score:
                    best = (start + depth - 1, mapping)
                    best_score = len(mapping)
        if not best:
            return []

        header_end, mapping = best
        items: List[Dict[str, Any]] = []
        for index, row in enumerate(rows[header_end + 1:], start=1):
            values = {field: _clean(row[col]) if col < len(row) else "" for field, col in mapping.items()}
            if not values.get("function_name"):
                continue
            values["function_id"] = values.get("function_id") or _stable_id(doc_id, "FUNC", offset + index)
            values["aliases"] = _parse_aliases(values.get("aliases"))
            items.append(values)
        return items

    def _functions_from_paragraph(
        self,
        title: str,
        text: str,
        doc_id: str,
        index: int,
    ) -> Iterable[Dict[str, Any]]:
        values: Dict[str, Any] = {}
        for line in re.split(r"[\n。]", text or ""):
            match = _FIELD_RE.match(_clean(line))
            if not match:
                continue
            field = _PARAGRAPH_FIELDS.get(_clean(match.group(1)))
            if field:
                values[field] = _clean(match.group(2))

        id_match = _ID_RE.search(f"{title} {text}")
        if id_match and not values.get("function_id"):
            values["function_id"] = id_match.group(0).upper()

        if not values.get("function_name"):
            if "功能" in title and len(title) <= 120:
                values["function_name"] = re.sub(r"^\d+(?:\.\d+)*\s*", "", title).strip()
            elif any(keyword in text for keyword in ("负责", "实现", "用于", "完成")) and "功能" in text:
                values["function_name"] = title or f"功能说明{index}"

        if not values.get("function_name"):
            return []
        values["function_id"] = values.get("function_id") or _stable_id(doc_id, "FUNC", index)
        values["description"] = values.get("description") or text
        values["aliases"] = _parse_aliases(values.get("aliases"))
        return [values]

    def _functions_from_graph(
        self,
        vertices: Sequence[Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
        doc_id: str,
        offset: int,
    ) -> Iterable[Dict[str, Any]]:
        by_id = {str(vertex.get("id")): vertex for vertex in vertices if vertex.get("id")}
        parents: Dict[str, List[str]] = {}
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source in by_id and target in by_id:
                parents.setdefault(target, []).append(source)

        def ancestor_name(vertex_id: str, labels: Sequence[str]) -> str:
            queue = list(parents.get(vertex_id, []))
            visited = set()
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                vertex = by_id.get(current, {})
                if str(vertex.get("label")) in labels:
                    props = vertex.get("properties") or {}
                    return _clean(props.get("name") if isinstance(props, dict) else current)
                queue.extend(parents.get(current, []))
            return ""

        items = []
        for index, vertex in enumerate(vertices, start=1):
            if str(vertex.get("label")) != "Function":
                continue
            props = vertex.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            function_name = _clean(props.get("name") or vertex.get("id"))
            if not function_name:
                continue
            function_id = _clean(props.get("function_id") or vertex.get("id"))
            if not function_id or function_id.lower().startswith("v"):
                function_id = _stable_id(doc_id, "FUNC", offset + index)
            items.append(
                {
                    "function_id": function_id,
                    "function_name": function_name,
                    "description": _clean(props.get("description")),
                    "aliases": _parse_aliases(props.get("aliases")),
                    "failure_symptoms": _clean(props.get("failure_symptoms")),
                    "module_name": ancestor_name(str(vertex.get("id")), ("Module",)),
                    "subsystem_name": ancestor_name(str(vertex.get("id")), ("Subsystem",)),
                    "system_name": ancestor_name(str(vertex.get("id")), ("System",)),
                }
            )
        return items

    def _index_function(self, index_name: str, ctx: ParseContext, row_id: int, item: Dict[str, Any]) -> int:
        content = (
            f"功能ID: {item.get('function_id', '')}\n功能名称: {item.get('function_name', '')}\n"
            f"系统: {item.get('system_name', '')}\n子系统: {item.get('subsystem_name', '')}\n"
            f"模块: {item.get('module_name', '')}\n功能描述: {item.get('description', '')}\n"
            f"别名: {', '.join(item.get('aliases') or [])}\n功能失效表现: {item.get('failure_symptoms', '')}"
        )
        return self._index(
            index_name,
            ctx,
            f"{ctx.doc_id}_function_{row_id}",
            item.get("function_name", ""),
            content,
            "function_record",
            {
                "business_id_kwd": item.get("function_id", ""),
                "system_kwd": item.get("system_name", ""),
                "module_kwd": item.get("module_name", ""),
            },
        )

    def _index_graph(
        self,
        index_name: str,
        ctx: ParseContext,
        image_id: str,
        title: str,
        analysis: Dict[str, Any],
        vertices: Sequence[Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
        image_url: str,
        block_key: str,
    ) -> int:
        names = {}
        for vertex in vertices:
            props = vertex.get("properties") or {}
            names[str(vertex.get("id"))] = _clean(
                props.get("name") if isinstance(props, dict) else vertex.get("id")
            )
        content = (
            f"ICP系统架构图: {title}\n图意: {analysis.get('description', '')}\n"
            f"节点: {', '.join(name for name in names.values() if name)}\n"
            f"关系数: {len(edges)}"
        )
        count = self._index(
            index_name,
            ctx,
            image_id,
            title,
            content,
            "function_architecture_diagram",
            {"image_url": image_url, "chapter_kwd": block_key},
        )
        for index, edge in enumerate(edges):
            source = names.get(str(edge.get("source")), str(edge.get("source") or ""))
            target = names.get(str(edge.get("target")), str(edge.get("target") or ""))
            relation = _clean(edge.get("label"))
            edge_content = f"系统架构关系: {source} → {target}；关系: {relation}"
            count += self._index(
                index_name,
                ctx,
                f"{image_id}_edge_{index}",
                f"{source} → {target}",
                edge_content,
                "function_architecture_edge",
                {"business_id_kwd": source, "target_kwd": target, "chapter_kwd": block_key},
            )
        return count

    @staticmethod
    def _index(
        index_name: str,
        ctx: ParseContext,
        chunk_id: str,
        title: str,
        content: str,
        content_type: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not content.strip():
            return 0
        es_client.client.index(
            index=index_name,
            id=chunk_id,
            body={
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
            },
        )
        return 1
