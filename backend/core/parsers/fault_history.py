"""Fault-history adapter for mixed headings, key/value tables and paragraphs."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.document_ir import ParagraphIR, TableIR, clean_text, load_document_ir
from backend.core.parsers.records import NormalizationIssue, RecordDraft, SourceRef
from backend.core.parsers.registry import ParserRegistry
from backend.core.parsers.table_engine import TableSchema, TableSource, extract_records
from backend.icp.repository import icp_repository
from backend.storage.search_store import search_store


_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "case_id": ("故障编号", "案例编号", "工单编号", "记录编号", "CASEID"),
    "occurred_at": ("发生时间", "故障时间", "日期/阶段", "发生日期", "时间"),
    "mission_phase": ("任务阶段/工作模式", "任务阶段与工作方式", "任务阶段", "工作模式"),
    "personnel": ("排故人员", "维修人员", "处置人员", "责任人"),
    "device_type": ("设备类型", "设备型号", "产品型号", "适用设备"),
    "fault_symptom": ("故障现象", "故障模式与现象概要", "现象描述", "故障描述", "异常现象"),
    "related_module": ("关联系统设计模块", "关联模块", "所属模块", "子系统/模块", "故障模块"),
    "related_function": ("关联功能", "功能ID", "功能"),
    "diagnosis_method": ("使用的排故方法", "排故方法", "诊断方法", "检查方法"),
    "root_cause": ("故障原因分析", "根本原因", "根因", "故障原因"),
    "steps": ("处理步骤（含API调用）", "处理步骤(含API调用)", "处理步骤", "排故步骤", "处置步骤"),
    "final_solution": ("最终解决方案", "最终方案", "主要修复措施", "修复措施", "处置结果"),
    "verification_result": ("验证结果", "复测结果", "确认结果"),
    "preventive_action": ("预防措施", "预防性措施", "后续措施"),
    "document_links": ("关联文档索引", "关联文档", "FTA/RBD", "FMECA/MFL"),
    "initial_hmc": ("初始HMC代码", "HMC代码", "关联HMC"),
}


def _norm_label(value: object) -> str:
    text = clean_text(value).rstrip("：:").upper()
    return re.sub(r"[\s_\-—–/\\:：()（）\[\]【】]+", "", text)


_ALIAS_TO_FIELD = {
    _norm_label(alias): field_name
    for field_name, aliases in _FIELD_ALIASES.items()
    for alias in aliases
}

_FIELD_RE = re.compile(r"^([^：:]{1,50})\s*[:：]\s*(.*)$")
_EXPLICIT_CASE_RE = re.compile(r"^(?:故障|案例|工单|记录)编号\s*[:：]\s*([^\s，,；;]+)", re.I)
_CASE_TOKEN_RE = re.compile(r"\b(?:HIS|CASE|FAULT)[-_][A-Z0-9][A-Z0-9_.-]{2,80}\b", re.I)
_STEP_RE = re.compile(
    r"^\s*(?:第\s*(\d+)\s*步\s*[:：、.)．]?|[（(]\s*(\d+)\s*[）)]\s*|"
    r"(\d+)\s*[.．、)]\s+)(.*)$"
)
_SEPARATOR_RE = re.compile(r"^[\-—–_=─━·•\s]{5,}$")


def _canonical_field(label: object) -> str:
    normalized = _norm_label(label)
    if normalized in _ALIAS_TO_FIELD:
        return _ALIAS_TO_FIELD[normalized]
    matches = [
        (len(alias), field_name)
        for alias, field_name in _ALIAS_TO_FIELD.items()
        if len(alias) >= 2 and alias in normalized
    ]
    return max(matches, default=(0, ""))[1]


def _case_id_from_text(text: str, *, allow_token: bool = True) -> str:
    value = clean_text(text)
    explicit = _EXPLICIT_CASE_RE.search(value)
    if explicit:
        return explicit.group(1).strip()
    if allow_token:
        token = _CASE_TOKEN_RE.search(value)
        if token:
            return token.group(0).upper()
    return ""


def _step_from_text(text: str) -> Optional[Tuple[int, str]]:
    match = _STEP_RE.match(clean_text(text))
    if not match:
        return None
    number = next((int(value) for value in match.groups()[:3] if value), 0)
    return number, clean_text(match.group(4))


def _normalize_hmc_code(code: str) -> str:
    return re.sub(r"[\s\-/—–]", "", clean_text(code))


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        cleaned = clean_text(value).strip("，,；;。.)）")
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def _extract_links(text: str) -> Dict[str, Any]:
    value = clean_text(text)
    hmc_codes = _unique(
        "".join(match.groups())
        for match in re.finditer(
            r"(?<!\d)(\d{4,})\s*[-—–/]\s*(\*\*|[A-Za-z0-9]+)\s*[-—–/]\s*(\d{1,})(?!\d)",
            value,
        )
    )
    fmeca_refs = _unique(
        match.group(0).upper()
        for match in re.finditer(r"\b(?:FMECA|FM)-[A-Z0-9][A-Z0-9-]*\b", value, re.I)
    )
    # Prefer the actual fault-mode business ID (FM-*) over a document label.
    fmeca_refs.sort(key=lambda item: (not item.startswith("FM-"), item))
    fta_refs = _unique(
        match.group(0).upper()
        for match in re.finditer(r"\b(?:FTA|EVT)-[A-Z0-9][A-Z0-9-]*\b", value, re.I)
    )
    rbd_refs = _unique(
        match.group(0).upper()
        for match in re.finditer(r"\bRBD-[A-Z0-9][A-Z0-9-]*\b", value, re.I)
    )
    mfl_refs = _unique(
        match.group(0).upper()
        for match in re.finditer(r"\bMFL-[A-Z0-9][A-Z0-9-]*\b", value, re.I)
    )
    function_refs = _unique(
        match.group(0).upper()
        for match in re.finditer(r"\bFUNC-[A-Z0-9][A-Z0-9-]*\b", value, re.I)
    )
    return {
        "hmc_codes": [_normalize_hmc_code(item) for item in hmc_codes],
        "fmeca_refs": fmeca_refs,
        "fta_refs": fta_refs,
        "rbd_refs": rbd_refs,
        "mfl_refs": mfl_refs,
        "function_refs": function_refs,
    }


@dataclass
class _CaseBuilder:
    values: Dict[str, Any] = field(default_factory=dict)
    raw_fields: Dict[str, str] = field(default_factory=dict)
    sources: List[SourceRef] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    active_field: str = ""

    @property
    def case_id(self) -> str:
        return clean_text(self.values.get("case_id"))

    def add_source(self, source: SourceRef) -> None:
        if str(source) not in {str(item) for item in self.sources}:
            self.sources.append(source)

    def set_field(self, field_name: str, value: str, source: SourceRef, raw_label: str = "") -> None:
        cleaned = clean_text(value)
        if field_name == "steps":
            self.active_field = "steps"
            if cleaned:
                self.add_step(len(self.steps) + 1, cleaned, source)
            return
        if cleaned:
            previous = clean_text(self.values.get(field_name))
            if previous and cleaned != previous:
                # Tables usually carry concise authoritative case attributes;
                # paragraphs may add detail. Keep both without silently replacing.
                self.values[field_name] = f"{previous} {cleaned}"
            else:
                self.values[field_name] = cleaned
        self.active_field = field_name
        self.add_source(source)
        if raw_label:
            self.raw_fields[raw_label] = cleaned

    def add_step(self, step_no: int, content: str, source: SourceRef) -> None:
        cleaned = clean_text(content)
        if not cleaned:
            return
        self.steps.append({"step_no": step_no or len(self.steps) + 1, "content": cleaned, "source_ref": str(source)})
        self.active_field = "steps"
        self.add_source(source)

    def append_continuation(self, text: str, source: SourceRef) -> None:
        cleaned = clean_text(text)
        if not cleaned:
            return
        if self.active_field == "steps" and self.steps:
            self.steps[-1]["content"] = clean_text(f"{self.steps[-1]['content']} {cleaned}")
            return
        if self.active_field and self.active_field in self.values:
            self.values[self.active_field] = clean_text(f"{self.values[self.active_field]} {cleaned}")
            self.add_source(source)

    def to_draft(self) -> RecordDraft:
        values = dict(self.values)
        values["steps"] = [item["content"] for item in self.steps]
        values["step_records"] = list(self.steps)
        link_text = " ".join(
            clean_text(values.get(field_name))
            for field_name in ("document_links", "initial_hmc", "related_function")
        )
        links = _extract_links(link_text)
        values["hmc_code"] = links["hmc_codes"][0] if links["hmc_codes"] else ""
        values["fmeca_code"] = links["fmeca_refs"][0] if links["fmeca_refs"] else ""
        values["fta_ref"] = links["fta_refs"][0] if links["fta_refs"] else ""
        values["rbd_ref"] = links["rbd_refs"][0] if links["rbd_refs"] else ""
        values["mfl_code"] = links["mfl_refs"][0] if links["mfl_refs"] else ""
        # Store normalized references as a JSON object.  The exact source text
        # remains in raw_data.fields, so normalization never destroys evidence.
        values["document_links"] = links
        draft = RecordDraft(
            record_type="fault_history",
            values=values,
            raw_data={
                "fields": dict(self.raw_fields),
                "steps_with_sources": list(self.steps),
                "link_refs": links,
            },
            source_refs=list(self.sources),
            record_key=clean_text(values.get("case_id")),
        )
        if not clean_text(values.get("case_id")):
            draft.add_issue("missing_case_id", "历史案例缺少故障编号", field="case_id", severity="error")
        if not clean_text(values.get("fault_symptom")):
            draft.add_issue(
                "missing_fault_symptom", "历史案例缺少故障现象", field="fault_symptom", severity="error"
            )
        if not self.steps:
            draft.add_issue("missing_steps", "历史案例未识别到处理步骤", field="steps")
        return draft


def _pairs_from_table(table: TableIR) -> List[Tuple[str, str, SourceRef]]:
    """Read narrow key/value tables; wide summary tables are handled separately."""

    if table.width > 6:
        return []
    if table.rows:
        first_row = [cell.text for cell in table.rows[0]]
        nonempty = [value for value in first_row if clean_text(value)]
        recognized = sum(bool(_canonical_field(value)) for value in nonempty)
        # A wide summary table often has only 3--6 columns, too.  Its first
        # row is predominantly field names, whereas a key/value row alternates
        # field names and values.  Route the former to the schema fallback.
        if len(nonempty) >= 3 and recognized >= 3 and recognized / len(nonempty) >= 0.6:
            return []
    pairs: List[Tuple[str, str, SourceRef]] = []
    for row in table.rows:
        values = [cell.text for cell in row]
        nonempty = [value for value in values if clean_text(value)]
        if len(nonempty) < 2:
            continue
        # Support both two-column and key/value/key/value layouts.
        for index in range(0, len(values) - 1, 2):
            label = clean_text(values[index])
            value = clean_text(values[index + 1])
            field_name = _canonical_field(label)
            if field_name and value:
                pairs.append((label, value, row[index].source))
    return pairs


_HISTORY_ROW_SCHEMA = TableSchema.from_mapping(
    "fault_history",
    {
        field_name: aliases
        for field_name, aliases in _FIELD_ALIASES.items()
        if field_name not in {"steps", "initial_hmc", "related_function"}
    },
    required_fields=("case_id", "fault_symptom"),
    identity_groups=(("case_id",),),
    min_mapped_fields=3,
    max_header_depth=3,
    continuation_fields=("fault_symptom", "root_cause", "final_solution", "preventive_action"),
)


def _wide_table_fallback(tables: Sequence[TableIR]) -> List[RecordDraft]:
    drafts: List[RecordDraft] = []
    for table in tables:
        for record in extract_records(TableSource.from_ir(table), _HISTORY_ROW_SCHEMA):
            builder = _CaseBuilder()
            for field_name, value in record.values.items():
                if clean_text(value):
                    builder.set_field(field_name, clean_text(value), record.source_refs[0], field_name)
            builder.raw_fields.update(record.raw_data.get("columns") or {})
            draft = builder.to_draft()
            draft.issues.extend(record.issues)
            drafts.append(draft)
    return drafts


def extract_history_drafts(
    file_path: str, *, source_name: Optional[str] = None
) -> List[RecordDraft]:
    """Pure ordered extraction used by tests and the persistence adapter."""

    document = load_document_ir(file_path, source_name=source_name)
    drafts: List[RecordDraft] = []
    current: Optional[_CaseBuilder] = None

    def finish() -> None:
        nonlocal current
        if current is not None and (current.case_id or current.values):
            drafts.append(current.to_draft())
        current = None

    def start(case_id: str, source: SourceRef, title: str = "") -> _CaseBuilder:
        nonlocal current
        normalized = clean_text(case_id).upper()
        if current is not None and current.case_id and current.case_id != normalized:
            finish()
        if current is None:
            current = _CaseBuilder()
        current.values["case_id"] = normalized
        if title:
            current.raw_fields.setdefault("case_title", clean_text(title))
        current.add_source(source)
        return current

    for element in document.elements:
        if isinstance(element, TableIR):
            pairs = _pairs_from_table(element)
            table_case_id = next(
                (value for label, value, _source in pairs if _canonical_field(label) == "case_id"),
                "",
            )
            if table_case_id:
                start(table_case_id, element.source)
            if current is None:
                continue
            for label, value, source in pairs:
                field_name = _canonical_field(label)
                if not field_name:
                    continue
                if field_name == "case_id":
                    current.raw_fields[label] = clean_text(value)
                    continue
                current.set_field(field_name, value, source, label)
            continue

        paragraph: ParagraphIR = element
        text = clean_text(paragraph.text)
        if not text or _SEPARATOR_RE.fullmatch(text):
            continue
        case_id = _case_id_from_text(text, allow_token=bool(paragraph.heading_level))
        if case_id:
            start(case_id, paragraph.source, text)
            # A line formatted as “故障编号：...” contains no additional field.
            if _EXPLICIT_CASE_RE.match(text):
                continue
        if current is None:
            continue

        step = _step_from_text(text)
        if step and not case_id:
            current.add_step(step[0], step[1], paragraph.source)
            continue

        field_match = _FIELD_RE.match(text)
        if field_match:
            label = clean_text(field_match.group(1))
            value = clean_text(field_match.group(2))
            field_name = _canonical_field(label)
            if field_name:
                current.set_field(field_name, value, paragraph.source, label)
                continue

        # A non-case section heading ends the detailed record region. This also
        # prevents summary/index tables from being appended to the final case.
        if paragraph.heading_level and not case_id:
            finish()
            continue
        current.append_continuation(text, paragraph.source)

    finish()
    detailed = [draft for draft in drafts if draft.values.get("case_id")]
    if detailed:
        # Heading and key/value table evidence belong to one case, not duplicate records.
        merged: Dict[str, RecordDraft] = {}
        for draft in detailed:
            case_id = clean_text(draft.values.get("case_id")).upper()
            if case_id not in merged:
                merged[case_id] = draft
                continue
            target = merged[case_id]
            for field_name, value in draft.values.items():
                if value not in (None, "", []):
                    target.values[field_name] = value
            target.raw_data.update(draft.raw_data)
            target.extend_sources(draft.source_refs)
            target.issues.extend(draft.issues)
        return list(merged.values())
    return _wide_table_fallback(document.tables)


@ParserRegistry.register("fault_history")
class FaultHistoryParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        return [] if ext in (".docx", ".txt", ".md") else ["故障历史仅支持 .docx/.txt/.md"]

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        drafts = extract_history_drafts(file_path, source_name=ctx.file_name)
        if not drafts:
            raise ValueError("未识别到故障历史案例")
        if not any(draft.valid for draft in drafts):
            reason = "; ".join(issue.message for issue in drafts[0].errors)
            raise ValueError(f"故障历史未产生有效结构化案例: {reason}")

        icp_repository.clear_history_snapshot(ctx.doc_id)

        vectors = 0
        failed: List[Dict[str, Any]] = []
        database_failed: List[Dict[str, Any]] = []
        projection_failed: List[Dict[str, Any]] = []
        cases_ok = 0
        steps_ok = 0
        for index, draft in enumerate(drafts, start=1):
            row = draft.to_row()
            steps = list(row.get("steps") or [])
            if draft.errors:
                failed.append({
                    "case": row.get("case_id") or index,
                    "source_ref": draft.source_ref,
                    "reason": "; ".join(issue.message for issue in draft.errors),
                })
                continue
            try:
                row_id = icp_repository.upsert_history_case(
                    ctx.doc_id,
                    row,
                    row.get("step_records") or steps,
                )
                cases_ok += 1
                steps_ok += len(steps)
            except Exception as exc:
                failure = {
                    "case": row.get("case_id") or index,
                    "source_ref": draft.source_ref,
                    "reason": str(exc)[:300],
                }
                failed.append(failure)
                database_failed.append(failure)
                continue

            content = (
                f"故障历史编号: {row.get('case_id', '')}\n发生时间: {row.get('occurred_at', '')}\n"
                f"任务阶段/工作模式: {row.get('mission_phase', '')}\n设备类型: {row.get('device_type', '')}\n"
                f"关联模块: {row.get('related_module', '')}\n关联功能: {row.get('related_function', '')}\n"
                f"故障现象: {row.get('fault_symptom', '')}\n排故方法: {row.get('diagnosis_method', '')}\n"
                f"根本原因: {row.get('root_cause', '')}\n处理步骤: {'；'.join(steps)}\n"
                f"最终方案: {row.get('final_solution', '')}\n验证结果: {row.get('verification_result', '')}\n"
                f"预防措施: {row.get('preventive_action', '')}\n关联 HMC: {row.get('hmc_code', '')}；"
                f"FMECA: {row.get('fmeca_code', '')}；FTA: {row.get('fta_ref', '')}；"
                f"RBD: {row.get('rbd_ref', '')}；MFL: {row.get('mfl_code', '')}"
            )
            try:
                vector = embedding_service.embed(content)
                stable_key = hashlib.sha256(
                    f"fault_history\x1f{row.get('case_id', '')}".encode("utf-8")
                ).hexdigest()[:24]
                chunk_id = f"{ctx.doc_id}_fault_history_{stable_key}"
                search_store.upsert(
                    chunk_id,
                    ctx.enrich({
                        "chunk_id": chunk_id,
                        "doc_id": ctx.doc_id,
                        "doc_name": ctx.file_name,
                        "title": row.get("case_id", ""),
                        "section": "fault_history",
                        "content": content,
                        "kb_id": ctx.kb_id or ctx.doc_type,
                        "content_type": "fault_history_case",
                        "doc_type_kwd": ctx.doc_type,
                        "business_id_kwd": row.get("case_id", ""),
                        "hmc_code_kwd": row.get("hmc_code", ""),
                        "source_ref": draft.source_ref,
                        "pg_id_kwd": str(row_id),
                        "removed_kwd": "N",
                        "available_int": 1,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }),
                    vector=vector,
                )
                vectors += 1
            except Exception as exc:
                projection_failed.append({
                    "case": row.get("case_id") or index,
                    "source_ref": draft.source_ref,
                    "reason": str(exc)[:300],
                })
            ctx.progress(f"[案例 {index}/{len(drafts)}] {row.get('case_id')} 入库成功")

        if database_failed:
            raise RuntimeError(
                "故障历史 PostgreSQL 真源写入失败 "
                f"{len(database_failed)} 条；文档不会进入可诊断状态。"
                f"首个错误: {database_failed[0]['reason']}"
            )

        if cases_ok == 0:
            reason = failed[0]["reason"] if failed else "无有效记录"
            raise ValueError(f"故障历史未产生有效结构化案例: {reason}")

        stats = {
            "cases": cases_ok,
            "failed": failed[:100],
            "steps": steps_ok,
            "vectors": vectors,
            "projection_failed": projection_failed[:100],
            "normalization_warnings": sum(len(draft.issues) for draft in drafts),
        }
        message = (
            f"解析完成：历史案例 {cases_ok} 条，步骤 {steps_ok} 条，失败 {len(failed)} 条"
        )
        if projection_failed:
            message += f"，检索投影失败 {len(projection_failed)} 条"
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=message,
        )


__all__ = ["FaultHistoryParser", "extract_history_drafts"]
