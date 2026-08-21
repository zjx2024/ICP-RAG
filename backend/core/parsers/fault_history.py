"""故障历史文档解析器。

识别“故障编号—现象—原因—排故步骤—最终方案—关联文档”案例结构，
写入 PostgreSQL 结构表，并按案例/步骤写入 Elasticsearch。
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.icp.repository import icp_repository
from backend.storage.es_client import es_client


_FIELD_MAP = {
    "发生时间": "occurred_at",
    "排故人员": "personnel",
    "设备类型": "device_type",
    "故障现象": "fault_symptom",
    "关联系统设计模块": "related_module",
    "关联模块": "related_module",
    "使用的排故方法": "diagnosis_method",
    "排故方法": "diagnosis_method",
    "故障原因分析": "root_cause",
    "根本原因": "root_cause",
    "最终解决方案": "final_solution",
    "最终方案": "final_solution",
    "预防措施": "preventive_action",
    "关联文档索引": "document_links",
    "处理步骤（含 API 调用）": "steps",
    "处理步骤": "steps",
}

_CASE_RE = re.compile(r"^故障编号\s*[:：]\s*([^\s，,；;]+)")
_FIELD_RE = re.compile(r"^([^：:]{2,40})\s*[:：]\s*(.*)$")
_STEP_RE = re.compile(r"^第\s*(\d+)\s*步\s*[:：]?\s*(.*)$")


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _read_lines(file_path: str) -> List[str]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        for encoding in ("utf-8", "gb18030", "gbk"):
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    return [_clean(line) for line in handle if _clean(line)]
            except UnicodeDecodeError:
                continue
        raise ValueError("文本文件编码无法识别")

    from docx import Document

    document = Document(file_path)
    lines = [_clean(paragraph.text) for paragraph in document.paragraphs if _clean(paragraph.text)]
    # 若正文没有案例，尝试读取表格中的历史记录。
    if not any(_CASE_RE.match(line) for line in lines):
        for table in document.tables:
            for row in table.rows:
                values = [_clean(cell.text) for cell in row.cells]
                if any(values):
                    lines.append(" | ".join(values))
    return lines


def _parse_links(text: str) -> Dict[str, str]:
    result = {"hmc_code": "", "fmeca_code": "", "fta_ref": "", "mfl_code": ""}
    patterns = {
        "hmc_code": r"HMC\s*[-=：:>]*\s*([^；;，,\s]+)",
        "fmeca_code": r"FMECA\s*[-=：:>]*\s*([^；;，,\s（(]+)",
        "fta_ref": r"FTA\s*[-=：:>]*\s*([^；;]+)",
        "mfl_code": r"MFL\s*[-=：:>]*\s*([^；;，,\s（(]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[key] = _clean(match.group(1))
    return result


def _parse_cases(lines: Iterable[str]) -> List[Tuple[Dict, List[str]]]:
    cases: List[Tuple[Dict, List[str]]] = []
    current: Dict[str, str] = {}
    steps: List[str] = []
    active_field = ""

    def finish() -> None:
        nonlocal current, steps, active_field
        if current.get("case_id") and current.get("fault_symptom"):
            links = _parse_links(current.pop("document_links", ""))
            for key, value in links.items():
                current.setdefault(key, value)
            cases.append((dict(current), list(steps)))
        current = {}
        steps = []
        active_field = ""

    for raw_line in lines:
        line = _clean(raw_line)
        if not line:
            continue
        case_match = _CASE_RE.match(line)
        if case_match:
            finish()
            current["case_id"] = case_match.group(1)
            active_field = ""
            continue

        if not current:
            continue

        step_match = _STEP_RE.match(line)
        if step_match:
            steps.append(_clean(step_match.group(2)))
            active_field = "steps"
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            label = _clean(field_match.group(1))
            value = _clean(field_match.group(2))
            canonical = _FIELD_MAP.get(label)
            if canonical == "steps":
                active_field = "steps"
                if value:
                    steps.append(value)
            elif canonical:
                current[canonical] = value
                active_field = canonical
            else:
                active_field = ""
            continue

        if active_field == "steps":
            steps.append(line)
        elif active_field and active_field in current:
            current[active_field] = _clean(f"{current[active_field]} {line}")

    finish()
    return cases


@ParserRegistry.register("fault_history")
class FaultHistoryParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        return [] if ext in (".docx", ".txt") else ["故障历史仅支持 .docx/.txt"]

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        lines = _read_lines(file_path)
        cases = _parse_cases(lines)
        if not cases:
            raise ValueError("未识别到包含‘故障编号’和‘故障现象’的历史案例")

        index_name = es_client.get_main_index_name("default")
        es_client.create_index(index_name)
        vectors = 0
        failed = []

        for index, (case, steps) in enumerate(cases, start=1):
            case["source_ref"] = f"{ctx.file_name}#case-{index}"
            try:
                row_id = icp_repository.upsert_history_case(ctx.doc_id, case, steps)
                content = (
                    f"故障历史编号: {case.get('case_id', '')}\n发生时间: {case.get('occurred_at', '')}\n"
                    f"设备类型: {case.get('device_type', '')}\n关联模块: {case.get('related_module', '')}\n"
                    f"故障现象: {case.get('fault_symptom', '')}\n排故方法: {case.get('diagnosis_method', '')}\n"
                    f"根本原因: {case.get('root_cause', '')}\n处理步骤: {'；'.join(steps)}\n"
                    f"最终方案: {case.get('final_solution', '')}\n预防措施: {case.get('preventive_action', '')}\n"
                    f"关联 HMC: {case.get('hmc_code', '')}；FMECA: {case.get('fmeca_code', '')}；"
                    f"FTA: {case.get('fta_ref', '')}；MFL: {case.get('mfl_code', '')}"
                )
                vector = embedding_service.embed(content)
                chunk_id = f"{ctx.doc_id}_history_{row_id}"
                es_client.client.index(
                    index=index_name,
                    id=chunk_id,
                    body={
                        "chunk_id": chunk_id,
                        "doc_id": ctx.doc_id,
                        "doc_name": ctx.file_name,
                        "title": case.get("case_id", ""),
                        "section": "fault_history",
                        "content": content,
                        "content_vector": vector,
                        "kb_id": ctx.kb_id or ctx.doc_type,
                        "content_type": "fault_history_case",
                        "doc_type_kwd": ctx.doc_type,
                        "business_id_kwd": case.get("case_id", ""),
                        "hmc_code_kwd": case.get("hmc_code", ""),
                        "removed_kwd": "N",
                        "available_int": 1,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                vectors += 1
                ctx.progress(f"[案例 {index}/{len(cases)}] {case.get('case_id')} 入库成功")
            except Exception as exc:
                failed.append({"case": case.get("case_id", index), "reason": str(exc)[:300]})

        stats = {
            "cases": len(cases) - len(failed),
            "failed": failed,
            "steps": sum(len(steps) for _, steps in cases),
            "vectors": vectors,
        }
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=f"解析完成：历史案例 {stats['cases']} 条，步骤 {stats['steps']} 条，失败 {len(failed)} 条",
        )
