"""文本类解析器（fault_history / function_locator）：按标题/编号分块 → 抽取业务标识 → embedding 写 ES。

说明：ICP 文本类文档（icp_documents）不落入 documents 表，document_chunks 存在
documents 外键约束，故文本分块以 ES 为主存储（原文结构由 /original 按文件重建）。
"""
import logging
import os
import re
from datetime import datetime
from typing import Dict, List

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.storage.es_client import es_client

logger = logging.getLogger(__name__)

# 业务标识（fault_code / device_code）抽取规则：编号前缀写法或标准编号形态
_ID_PATTERNS = [
    r'(?:编号|代码|ID|CODE|故障编号|设备编号)\s*[:：]\s*([A-Za-z0-9][A-Za-z0-9\-_/. ]{1,40})',
    r'\b([A-Z]{2,6}[-/]?\d{2,}[A-Za-z0-9\-_]*)\b',
]


def extract_business_id(title: str, content: str) -> str:
    """从标题/正文抽取业务标识，无匹配返回空串。"""
    for text in (title, content[:500]):
        if not text:
            continue
        for pat in _ID_PATTERNS:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip().split(" ")[0][:64]
    return ""


@ParserRegistry.register("fault_history")
@ParserRegistry.register("function_locator")
class TextParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        errors = []
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".docx", ".txt"):
            errors.append(f"不支持的格式: {ext}，文本类文档仅支持 .docx / .txt")
        return errors

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        blocks = self._split_blocks(file_path, ctx)
        if not blocks:
            raise ValueError("未解析到任何文本分块")

        index_name = es_client.get_main_index_name("default")
        chunks = 0
        vectors = 0
        total = len(blocks)
        failed: List[Dict] = []

        for idx, (block_key, block_info) in enumerate(blocks.items(), start=1):
            ctx.progress(f"[分块 {idx}/{total}] {block_info.get('title', block_key)}")
            text_content = block_info.get("text", "") or ""
            if not text_content.strip():
                continue
            title = block_info.get("title", block_key)
            chunk_id = f"{ctx.doc_id}_{block_key}"
            biz_id = extract_business_id(title, text_content)

            try:
                vector = embedding_service.embed(text_content)
                doc = {
                    "chunk_id": chunk_id,
                    "doc_id": ctx.doc_id,
                    "doc_name": ctx.file_name,
                    "title": title,
                    "section": block_key,
                    "content": text_content,
                    "content_vector": vector,
                    "kb_id": ctx.kb_id or ctx.doc_type,
                    "content_type": "text",
                    "doc_type_kwd": ctx.doc_type,
                    "business_id_kwd": biz_id,
                    "removed_kwd": "N",
                    "available_int": 1,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                es_client.client.index(index=index_name, id=chunk_id, body=doc)
                chunks += 1
                vectors += 1
            except Exception as e:
                failed.append({"block": block_key, "reason": f"向量化失败: {str(e)[:100]}"})

        stats = {
            "chunks": chunks,
            "vectors": vectors,
            "failed": failed,
        }
        message = f"解析完成：{chunks} 个分块，向量化 {vectors} 条"
        return ParseResult(doc_id=ctx.doc_id, doc_type=ctx.doc_type, stats=stats,
                           vectors=vectors, message=message)

    # ---------- 内部实现 ----------

    def _split_blocks(self, file_path: str, ctx: ParseContext) -> Dict:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            from backend.core.docx_parser import DocumentParser
            parser = DocumentParser(file_path, ctx.doc_id)
            result = parser.process()
            if "error" in result:
                raise ValueError(result["error"])
            return result["blocks"]
        # .txt
        from backend.core.docx_parser import TextParser as DocxTextParser
        tp = DocxTextParser(file_path)
        if not tp.load():
            raise ValueError("文本文件加载失败")
        return tp.parse_markdown_headers()
