"""HMC / FMECA / MFL deterministic table adapters.

Layout discovery is handled by :mod:`table_engine`; this module only defines
domain aliases, validation, normalization, persistence compatibility and
search projections. Every extracted row retains raw fields and a physical
source reference even though the current PG methods ignore those extra keys.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.records import RecordDraft
from backend.core.parsers.registry import ParserRegistry
from backend.core.parsers.table_engine import (
    TableSchema,
    extract_table_drafts as engine_extract_table_drafts,
)
from backend.core.value_normalization import normalize_optional_identifier
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store

logger = logging.getLogger(__name__)


SCHEMAS: Dict[str, Dict[str, Any]] = {
    "hmc": {
        # A missing maintenance procedure is preserved as an incomplete record;
        # rejecting it would also discard a valid exact code and symptom.
        "required": ["fault_desc"],
        "identity_groups": [("hmc_code",), ("lcn", "hsc")],
        "columns": {
            "hmc_code": ["HMC代码", "HMC码", "HMC_CODE", "完整HMC", "故障码", "故障代码", "HMC"],
            "lcn": ["LCN", "LCN代码", "线路可更换单元代码"],
            "hamc": ["HAMC", "HAMC代码"],
            "hsc": ["HSC", "HSC代码"],
            "fault_desc": ["故障描述现象", "故障描述及现象", "故障描述/现象", "故障现象", "故障描述"],
            "repair_method": [
                "故障分析程序/维修方法", "故障分析程序维修方法", "故障分析程序维修改法",
                "故障分析程序", "维修方法", "修复方法", "维修建议", "处理方法",
            ],
        },
        "continuation_fields": ("fault_desc", "repair_method"),
        # A vertically merged code normally scopes several independently
        # worded symptom/procedure rows.  Inherit only these explicit identity
        # columns; the variant layer keeps each physical row as evidence.
        "merged_cell_inherit_fields": ("hmc_code", "lcn", "hamc", "hsc"),
    },
    "fmeca": {
        "required": ["fault_mode"],
        "identity_groups": [("code",), ("sequence_no",)],
        "columns": {
            # Keep 序号 separate. Treating it as code made the first column win
            # over an explicit FM-SYN-* identifier in the adjacent column.
            "code": ["代码", "编码", "故障模式代码", "故障模式编号", "故障代码", "CODE"],
            "sequence_no": ["序号", "项次", "NO", "NO."],
            "product_flag": ["产品或功能标志", "产品/功能标志", "产品标志", "产品标识", "PRODUCT_FLAG"],
            "func_management": ["功能管理", "功能管理项目", "功能名称", "功能", "FUNC"],
            "fault_mode": ["故障模式", "故障类型", "失效模式", "FAULT_MODE"],
            "fault_cause": ["故障原因", "失效原因", "原因", "FAULT_CAUSE"],
            "mission_phase": ["任务阶段与工作方式", "任务阶段/工作方式", "任务阶段", "工作方式", "阶段", "MISSION_PHASE"],
            "local_impact": ["故障影响局部", "局部影响", "故障局部影响", "LOCAL_IMPACT"],
            "final_impact": ["最终影响", "故障最终影响", "系统影响", "FINAL_IMPACT"],
            "severity": ["严酷度类别", "严酷度等级", "严酷度", "严重度", "SEVERITY", "等级"],
            "probability": ["故障概率等级", "故障概率", "发生概率", "概率", "PROBABILITY"],
            "detect_method": ["故障检测方法", "检测方法", "检测手段", "DETECT_METHOD"],
            "design_improve": ["设计改进措施", "设计改进", "改进措施", "DESIGN_IMPROVE"],
            "compensate_action": ["使用补偿措施", "补偿措施", "补偿动作", "COMPENSATE_ACTION"],
        },
        "continuation_fields": (
            "fault_mode", "fault_cause", "mission_phase", "local_impact", "final_impact",
            "detect_method", "design_improve", "compensate_action",
        ),
    },
    "mfl": {
        "required": ["mfl_code"],
        "identity_groups": [("mfl_code",)],
        "columns": {
            "mfl_code": ["MFL代码", "MFL_CODE", "MFL码", "MFL", "维修故障代码", "故障代码"],
            "module_fault": ["模块故障模式", "模块故障描述", "模块故障"],
            "sur_fault": ["SUR板故障模式", "SUR故障模式", "上层故障", "上级故障"],
            "subcircuit_fault": ["功能子电路级故障模式", "功能子电路故障模式", "子电路故障", "子电路"],
            "api_reportable": ["API是否可直接上报", "API可直接上报", "API上报", "API可上报", "是否上报"],
            "api_name": ["对应功能子电路API名称", "功能子电路API名称", "API名称", "API名"],
            "api_method": ["API调用方法", "API调用方式", "API方法", "请求方法"],
            "api_usage": ["API使用方法", "API用法", "API说明", "用法"],
            "platform_test": ["是否进行平台级摸底实验", "平台级摸底实验状态", "平台级摸底实验", "平台测试", "平台验证状态", "是否平台测试"],
            "repair_action": ["维修处置", "维修措施", "维修建议", "处置措施"],
        },
        "continuation_fields": (
            "module_fault", "sur_fault", "subcircuit_fault", "api_usage", "repair_action",
        ),
    },
}


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _norm(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", _clean_cell(value))
    return re.sub(r"[\s_\-—–/\\:：()（）]+", "", normalized.upper())


def _normalized_code(value: object) -> str:
    """Normalize compatibility-width code glyphs while raw_data keeps source bytes."""

    return unicodedata.normalize("NFKC", _clean_cell(value))


def _schema_for(doc_type: str) -> TableSchema:
    spec = SCHEMAS[doc_type]
    return TableSchema.from_mapping(
        doc_type,
        spec["columns"],
        required_fields=spec["required"],
        identity_groups=spec["identity_groups"],
        min_mapped_fields=3 if doc_type != "mfl" else 2,
        max_header_depth=6,
        continuation_fields=spec.get("continuation_fields", ()),
        merged_cell_inherit_fields=spec.get("merged_cell_inherit_fields", ()),
    )


_HMC_PARTS_RE = re.compile(
    r"^\s*([A-Za-z0-9]+)\s*[-/ ]\s*(\*\*|[A-Za-z0-9]+)\s*[-/ ]\s*([A-Za-z0-9]+)\s*$"
)
_HMC_SEPARATORS_RE = re.compile(r"[\s\-/—–]")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_ROMAN_SEVERITY = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V"})


def _parse_boolean(value: object) -> tuple[Optional[bool], bool]:
    normalized = _norm(value)
    if normalized in {
        "是", "TRUE", "Y", "YES", "1", "支持", "可", "可以", "√",
    }:
        return True, True
    if normalized in {
        "否", "FALSE", "N", "NO", "0", "不支持", "不可", "×", "X",
    }:
        return False, True
    if normalized in {"", "-", "—", "不适用", "NA", "N/A"}:
        return None, True
    return None, False


def _normalize_probability(draft: RecordDraft) -> None:
    raw = _clean_cell(draft.values.get("probability"))
    draft.values["probability_raw"] = raw
    draft.values["probability_value"] = None
    draft.values["probability_level"] = ""
    if not raw:
        draft.values["probability"] = None
        return
    numeric = unicodedata.normalize("NFKC", raw).replace(",", "")
    try:
        if numeric.endswith("%") and _NUMERIC_RE.fullmatch(numeric[:-1].strip()):
            normalized = str(Decimal(numeric[:-1].strip()) / Decimal(100))
            draft.values["probability"] = normalized
            draft.values["probability_value"] = normalized
            return
        if _NUMERIC_RE.fullmatch(numeric):
            Decimal(numeric)
            draft.values["probability"] = numeric
            draft.values["probability_value"] = numeric
            return
    except InvalidOperation:
        pass
    # A-E and similar categories are not numeric probabilities. Never invent
    # a numeric mapping simply to satisfy the legacy NUMERIC column.
    draft.values["probability_class"] = raw
    draft.values["probability_level"] = raw
    draft.values["probability"] = None
    draft.add_issue(
        "probability_category_preserved",
        f"概率值“{raw}”是等级/带单位文本，已保留原值而未伪造数值",
        field="probability",
    )


def _normalize_hmc(draft: RecordDraft) -> None:
    for field_name in ("hmc_code", "lcn", "hamc", "hsc"):
        raw = _clean_cell(draft.values.get(field_name))
        draft.values[f"{field_name}_raw"] = raw
        draft.values[field_name] = _normalized_code(raw)

    code_raw = draft.values.get("hmc_code_raw", "")
    parts = _HMC_PARTS_RE.match(code_raw)
    if parts:
        stated_parts = {
            "lcn": _clean_cell(draft.values.get("lcn")),
            "hamc": _clean_cell(draft.values.get("hamc")),
            "hsc": _clean_cell(draft.values.get("hsc")),
        }
        code_parts = {"lcn": parts.group(1), "hamc": parts.group(2), "hsc": parts.group(3)}
        conflicts: List[str] = []
        for field_name in ("lcn", "hamc", "hsc"):
            stated = stated_parts[field_name]
            from_code = code_parts[field_name]
            if not stated:
                continue
            if field_name == "hsc" and stated.isdigit() and from_code.isdigit():
                equal = stated.zfill(4) == from_code.zfill(4)
            else:
                equal = stated.upper() == from_code.upper()
            if not equal:
                conflicts.append(
                    f"{field_name.upper()}={stated}（完整码为 {from_code}）"
                )
        if conflicts:
            draft.add_issue(
                "inconsistent_hmc_parts",
                "完整 HMC 与拆分字段冲突: " + "；".join(conflicts),
                field="hmc_code",
                severity="error",
            )
        if not _clean_cell(draft.values.get("lcn")):
            draft.values["lcn"] = parts.group(1)
        if not _clean_cell(draft.values.get("hamc")):
            draft.values["hamc"] = parts.group(2)
        if not _clean_cell(draft.values.get("hsc")):
            draft.values["hsc"] = parts.group(3)

    # Compact complete codes have supplier-specific split widths.  When the
    # same row explicitly states all three parts, concatenation is nevertheless
    # an unambiguous consistency check and a conflict must fail closed.
    stated_lcn = _clean_cell(draft.values.get("lcn"))
    stated_hamc = _clean_cell(draft.values.get("hamc"))
    stated_hsc = _clean_cell(draft.values.get("hsc"))
    if code_raw and not parts and stated_lcn and stated_hamc and stated_hsc:
        comparable_hsc = stated_hsc.zfill(4) if stated_hsc.isdigit() else stated_hsc
        stated_code = f"{stated_lcn}{stated_hamc}{comparable_hsc}"
        complete_code = _HMC_SEPARATORS_RE.sub("", code_raw)
        if complete_code.upper() != stated_code.upper():
            draft.add_issue(
                "inconsistent_hmc_parts",
                "完整 HMC 与拆分字段冲突: "
                f"完整码={code_raw}，拆分码={stated_lcn}-{stated_hamc}-{stated_hsc}",
                field="hmc_code",
                severity="error",
            )

    lcn = _clean_cell(draft.values.get("lcn"))
    hamc = _clean_cell(draft.values.get("hamc"))
    hsc = _clean_cell(draft.values.get("hsc"))
    if hsc.isdigit() and len(hsc) < 4:
        padded = hsc.zfill(4)
        draft.add_issue(
            "hsc_display_padding",
            f"HSC按四位显示格式由“{hsc}”补为“{padded}”",
            field="hsc",
        )
        hsc = padded
    draft.values.update({"lcn": lcn, "hamc": hamc, "hsc": hsc})

    if code_raw:
        canonical_code = _HMC_SEPARATORS_RE.sub("", code_raw)
    elif lcn and hamc and hsc:
        canonical_code = f"{lcn}{hamc}{hsc}"
    else:
        canonical_code = ""
    draft.values["hmc_code"] = canonical_code
    draft.values["hmc_display_code"] = (
        f"{lcn}-{hamc}-{hsc}" if lcn and hamc and hsc else code_raw or canonical_code
    )
    variant_seed = f"hmc\x1f{canonical_code}\x1f{draft.source_ref}"
    variant_id = hashlib.sha256(variant_seed.encode("utf-8")).hexdigest()[:32]
    draft.values["variant_id"] = variant_id
    draft.record_key = variant_id
    if not _clean_cell(draft.values.get("repair_method")):
        draft.values["repair_method"] = ""
        draft.add_issue(
            "missing_repair_method",
            "该HMC记录没有维修方法，仍保留精确代码和故障现象",
            field="repair_method",
        )


def _normalize_fmeca(draft: RecordDraft) -> None:
    code = _normalized_code(draft.values.get("code"))
    sequence_no = _normalized_code(draft.values.get("sequence_no"))
    if not code and sequence_no:
        digest = hashlib.sha1(draft.source_ref.encode("utf-8")).hexdigest()[:12].upper()
        code = f"FMECA-SRC-{digest}"
        draft.add_issue(
            "synthetic_source_local_code",
            "原表只有序号而无稳定故障模式代码，已生成来源局部ID；不得跨文档自动合并",
            field="code",
        )
    draft.values["code"] = code
    severity_raw = _clean_cell(draft.values.get("severity"))
    draft.values["severity_raw"] = severity_raw
    severity_normalized = severity_raw.translate(_ROMAN_SEVERITY).replace("类", "").strip()
    # ``severity`` remains as a compatibility alias for existing callers.
    draft.values["severity_normalized"] = severity_normalized
    draft.values["severity"] = severity_normalized
    _normalize_probability(draft)
    draft.record_key = code


def _normalize_mfl(draft: RecordDraft) -> None:
    draft.values["mfl_code"] = _normalized_code(draft.values.get("mfl_code"))
    # A visual dash/N/A in an optional identifier column means “not supplied”,
    # not a globally mergeable API business identity.  ``raw_data`` continues
    # to retain the exact source cell for audit.
    draft.values["api_name"] = normalize_optional_identifier(
        draft.values.get("api_name")
    )
    raw = _clean_cell(draft.values.get("api_reportable"))
    normalized, recognized = _parse_boolean(raw)
    draft.values["api_reportable_raw"] = raw
    draft.values["api_reportable"] = normalized
    if not recognized:
        draft.add_issue(
            "unrecognized_boolean_or_status",
            f"字段值“{raw}”不能无损归为是/否，已保留为未知",
            field="api_reportable",
        )

    platform_raw = _clean_cell(draft.values.get("platform_test"))
    platform_normalized, platform_recognized = _parse_boolean(platform_raw)
    platform_statuses = {
        "已完成", "完成", "部分完成", "待验证", "验证中", "已开展",
        "已进行", "未开展", "未进行", "未完成",
    }
    if _norm(platform_raw) in {_norm(value) for value in platform_statuses}:
        # Completion is a lifecycle enum, not a boolean.  In particular,
        # ``部分完成`` must never become true in PG/OpenSearch filters.
        platform_normalized, platform_recognized = None, True
    draft.values["platform_test_raw"] = platform_raw
    draft.values["platform_test"] = platform_normalized
    if not platform_recognized:
        draft.add_issue(
            "unrecognized_boolean_or_status",
            f"字段值“{platform_raw}”不能无损归为是/否或已知状态，已保留为未知",
            field="platform_test",
        )
    draft.values["platform_test_status"] = draft.values.get("platform_test_raw", "")
    # The parser-facing name follows the source document wording while the PG
    # schema uses ``maintenance_action``.  Keep both so no value disappears at
    # the adapter/repository boundary.
    draft.values["maintenance_action"] = _clean_cell(draft.values.get("repair_action"))
    draft.record_key = _clean_cell(draft.values.get("mfl_code"))


def _validate_draft(draft: RecordDraft, doc_type: str) -> None:
    spec = SCHEMAS[doc_type]
    for field_name in spec["required"]:
        if not _clean_cell(draft.values.get(field_name)):
            draft.add_issue(
                "missing_required_field",
                f"必填字段缺失: {field_name}",
                field=field_name,
                severity="error",
            )
    if doc_type == "hmc" and not draft.values.get("hmc_code"):
        draft.add_issue(
            "missing_hmc_identity",
            "HMC缺少完整代码，且LCN/HAMC/HSC不足以生成规范代码",
            field="hmc_code",
            severity="error",
        )
    elif doc_type == "fmeca" and not draft.values.get("code"):
        draft.add_issue(
            "missing_fmeca_identity", "FMECA缺少记录代码/序号", field="code", severity="error"
        )
    elif doc_type == "mfl" and not draft.values.get("mfl_code"):
        draft.add_issue(
            "missing_mfl_identity", "MFL代码不能为空", field="mfl_code", severity="error"
        )


def _apply_legacy_xls_limits(draft: RecordDraft, doc_type: str) -> None:
    """Expose format loss in the pandas/xlrd legacy BIFF path.

    The reader retains values but cannot provide the display formats, formula
    cache and merge metadata available for xlsx.  A numeric identifier may
    already have lost leading zeroes, so it cannot become an authoritative
    code merely by converting the value back to text.
    """

    draft.add_issue(
        "legacy_xls_format_limit",
        ".xls 仅保留单元格值，无法可靠保留显示格式、公式缓存与合并元数据；"
        "代码前导零必须由文本型源单元格证明",
    )
    identity_fields = {
        "hmc": ("hmc_code", "lcn", "hamc", "hsc"),
        "fmeca": ("code",),
        "mfl": ("mfl_code",),
    }[doc_type]
    metadata = draft.raw_data.get("field_cell_metadata") or {}
    for field_name in identity_fields:
        cell = metadata.get(field_name) or {}
        raw_value = cell.get("raw_value")
        raw_type = str(cell.get("raw_type") or "")
        if raw_value in (None, "") or raw_type in {"str", "NoneType"}:
            continue
        if isinstance(raw_value, (int, float, Decimal)) and not isinstance(raw_value, bool):
            draft.add_issue(
                "legacy_xls_numeric_code_unsafe",
                f".xls 字段 {field_name} 是数值单元格，无法证明前导零；"
                "已保留原值但不晋升为权威记录",
                field=field_name,
                severity="error",
            )


def extract_domain_drafts(
    file_path: str,
    doc_type: str,
    *,
    source_name: Optional[str] = None,
) -> List[RecordDraft]:
    """Pure extraction entry point used by tests and the persistence adapter."""

    if doc_type not in SCHEMAS:
        raise ValueError(f"不支持的表格文档类型: {doc_type}")
    drafts = engine_extract_table_drafts(
        file_path,
        _schema_for(doc_type),
        record_type=doc_type,
        source_name=source_name,
    )
    for draft in drafts:
        draft.values = {key: _clean_cell(value) for key, value in draft.values.items()}
        if os.path.splitext(file_path)[1].lower() == ".xls":
            _apply_legacy_xls_limits(draft, doc_type)
        if doc_type == "hmc":
            _normalize_hmc(draft)
        elif doc_type == "fmeca":
            _normalize_fmeca(draft)
        else:
            _normalize_mfl(draft)
        _validate_draft(draft, doc_type)
    return drafts


@ParserRegistry.register("hmc")
@ParserRegistry.register("fmeca")
@ParserRegistry.register("mfl")
class ExcelRowParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        return [] if ext in (".xlsx", ".xlsm", ".xls", ".docx") else [
            f"不支持的格式: {ext}，表格类文档仅支持 .xlsx/.xlsm/.xls/.docx"
        ]

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        drafts = extract_domain_drafts(
            file_path, ctx.doc_type, source_name=ctx.file_name
        )
        if not drafts:
            raise ValueError(f"未找到可识别的 {ctx.doc_type} 数据表")
        if not any(draft.valid for draft in drafts):
            reason = "; ".join(issue.message for issue in drafts[0].errors)
            raise ValueError(f"{ctx.doc_type} 未产生有效结构化记录: {reason}")

        # A document ID represents the current uploaded snapshot.  Clear only
        # this type's prior authority rows after extraction/validation succeeds;
        # the immutable artifact remains available for retry and rebuild.
        postgres_client.clear_icp_authoritative_rows(ctx.doc_id, ctx.doc_type)

        rows_ok = 0
        rows_failed: List[Dict[str, Any]] = []
        database_failed: List[Dict[str, Any]] = []
        projection_failed: List[Dict[str, Any]] = []
        vectors = 0
        for index, draft in enumerate(drafts, start=1):
            row = draft.to_row()
            if draft.errors:
                rows_failed.append({
                    "row": draft.source_ref or index,
                    "reason": "; ".join(issue.message for issue in draft.errors),
                    "warnings": row["normalization_warnings"],
                })
                continue
            try:
                row_id = self._upsert(ctx.doc_type, ctx.doc_id, row)
                rows_ok += 1
            except Exception as exc:
                failure = {
                    "row": draft.source_ref or index,
                    "reason": str(exc)[:300],
                    "warnings": row["normalization_warnings"],
                }
                rows_failed.append(failure)
                database_failed.append(failure)
                continue

            try:
                vectors += self._vectorize(ctx, row, row_id)
            except Exception as exc:
                projection_failed.append({"row": draft.source_ref or index, "reason": str(exc)[:300]})
            ctx.progress(
                f"[行 {index}/{len(drafts)}] {self._business_code(ctx.doc_type, row)} 入库成功"
            )

        if database_failed:
            raise RuntimeError(
                f"{ctx.doc_type} PostgreSQL 真源写入失败 "
                f"{len(database_failed)} 行；文档不会进入可诊断状态。"
                f"首个错误: {database_failed[0]['reason']}"
            )

        if rows_ok == 0:
            reason = rows_failed[0]["reason"] if rows_failed else "无有效记录"
            raise ValueError(f"{ctx.doc_type} 未产生有效结构化记录: {reason}")

        stats = {
            "rows_ok": rows_ok,
            "rows_failed": len(rows_failed),
            "failed_rows": rows_failed[:100],
            "projection_failed": projection_failed[:100],
            "total": len(drafts),
            "vectors": vectors,
            "normalization_warnings": sum(len(draft.issues) for draft in drafts),
        }
        message = (
            f"解析完成：成功 {rows_ok} 行，失败 {len(rows_failed)} 行，"
            f"向量化 {vectors} 条"
        )
        if projection_failed:
            message += f"，检索投影失败 {len(projection_failed)} 行"
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=message,
        )

    @staticmethod
    def _upsert(doc_type: str, doc_id: str, row: Dict[str, Any]) -> int:
        if doc_type == "hmc":
            return postgres_client.upsert_hmc(doc_id, row)
        if doc_type == "fmeca":
            return postgres_client.upsert_fmea(doc_id, row)
        return postgres_client.upsert_mfl(doc_id, row)

    @staticmethod
    def _business_code(doc_type: str, row: Dict[str, Any]) -> str:
        if doc_type == "hmc":
            return str(row.get("hmc_display_code") or row.get("hmc_code") or "")
        if doc_type == "fmeca":
            return str(row.get("code") or "")
        return str(row.get("mfl_code") or "")

    def _index_vector(
        self,
        ctx: ParseContext,
        row_id: int,
        suffix: str,
        title: str,
        content: str,
        content_type: str,
        tags: Dict[str, object],
        source_ref: str = "",
    ) -> int:
        if not content.strip():
            return 0
        vector = embedding_service.embed(content)
        business_key = str(
            tags.get("variant_id_kwd") or tags.get("business_id_kwd") or title or row_id
        ).strip()
        stable_key = hashlib.sha256(
            f"{ctx.doc_type}\x1f{business_key}\x1f{suffix}".encode("utf-8")
        ).hexdigest()[:24]
        # The projection ID is derived from the document/business identity,
        # not a PostgreSQL sequence value, so a PG rebuild does not duplicate
        # OpenSearch documents. ``pg_id_kwd`` still links to the current row.
        chunk_id = f"{ctx.doc_id}_{ctx.doc_type}_{stable_key}"
        body = ctx.enrich({
            "chunk_id": chunk_id,
            "doc_id": ctx.doc_id,
            "doc_name": ctx.file_name,
            "title": title,
            "section": ctx.doc_type,
            "content": content,
            "kb_id": ctx.kb_id or ctx.doc_type,
            "content_type": content_type,
            "doc_type_kwd": ctx.doc_type,
            "pg_id_kwd": str(row_id),
            "source_ref": source_ref,
            "removed_kwd": "N",
            "available_int": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **tags,
        })
        search_store.upsert(chunk_id, body, vector=vector)
        return 1

    def _vectorize(self, ctx: ParseContext, row: Dict[str, Any], row_id: int) -> int:
        doc_type = ctx.doc_type
        source_ref = str(row.get("source_ref") or "")
        if doc_type == "hmc":
            code = str(row.get("hmc_code") or "")
            display_code = str(row.get("hmc_display_code") or code)
            tags = {
                "business_id_kwd": code,
                "variant_id_kwd": row.get("variant_id", ""),
                "hmc_code_kwd": code,
                "lcn_kwd": row.get("lcn", ""),
                "hamc_kwd": row.get("hamc", ""),
                "hsc_kwd": row.get("hsc", ""),
            }
            diagnostic_text = (
                f"HMC: {display_code}\nLCN: {row.get('lcn', '')}；HAMC: {row.get('hamc', '')}；HSC: {row.get('hsc', '')}\n"
                f"故障现象: {row.get('fault_desc', '')}"
            )
            maintenance_text = (
                f"HMC: {display_code}\n故障现象: {row.get('fault_desc', '')}\n"
                f"故障分析程序与维修方法: {row.get('repair_method', '')}"
            )
            return (
                self._index_vector(ctx, row_id, "diagnostic", display_code, diagnostic_text, "hmc_diagnostic", tags, source_ref)
                + self._index_vector(ctx, row_id, "maintenance", display_code, maintenance_text, "hmc_maintenance", tags, source_ref)
            )

        if doc_type == "fmeca":
            probability = row.get("probability_raw") or row.get("probability") or ""
            content = (
                f"FMECA代码: {row.get('code', '')}\n产品/功能: {row.get('product_flag', '')} / {row.get('func_management', '')}\n"
                f"故障模式: {row.get('fault_mode', '')}\n故障原因: {row.get('fault_cause', '')}\n"
                f"任务阶段与工作方式: {row.get('mission_phase', '')}\n局部影响: {row.get('local_impact', '')}\n"
                f"最终影响: {row.get('final_impact', '')}\n严酷度: {row.get('severity', '')}\n"
                f"故障概率/等级: {probability}\n检测方法: {row.get('detect_method', '')}\n"
                f"设计改进: {row.get('design_improve', '')}\n使用补偿措施: {row.get('compensate_action', '')}"
            )
            tags = {
                "business_id_kwd": row.get("code", ""),
                "severity_kwd": row.get("severity", ""),
                "product_kwd": row.get("product_flag", ""),
            }
            return self._index_vector(
                ctx, row_id, "fmeca", str(row.get("code") or ""), content,
                "fmeca_row", tags, source_ref,
            )

        reportable = row.get("api_reportable")
        reportable_text = "未知" if reportable is None else ("是" if reportable else "否")
        platform_status = row.get("platform_test_status") or (
            "未知" if row.get("platform_test") is None else ("是" if row.get("platform_test") else "否")
        )
        content = (
            f"MFL: {row.get('mfl_code', '')}\n模块故障模式: {row.get('module_fault', '')}\n"
            f"SUR板故障模式: {row.get('sur_fault', '')}\n功能子电路级故障模式: {row.get('subcircuit_fault', '')}\n"
            f"API可直接上报: {reportable_text}\nAPI名称: {row.get('api_name', '')}\n"
            f"API调用方法: {row.get('api_method', '')}\nAPI使用方法: {row.get('api_usage', '')}\n"
            f"平台级摸底实验: {platform_status}\n维修处置: {row.get('repair_action', '')}"
        )
        tags = {
            "business_id_kwd": row.get("mfl_code", ""),
            "api_reportable_kwd": "unknown" if reportable is None else str(reportable).lower(),
            "platform_test_kwd": (
                "unknown" if row.get("platform_test") is None else str(row.get("platform_test")).lower()
            ),
            "platform_test_status_kwd": platform_status,
        }
        return self._index_vector(
            ctx, row_id, "mfl", str(row.get("mfl_code") or ""), content,
            "mfl_row", tags, source_ref,
        )


__all__ = ["ExcelRowParser", "SCHEMAS", "extract_domain_drafts"]
