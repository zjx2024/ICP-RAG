"""HMC / FMECA / MFL 专用表格解析器。

支持 Word/Excel 中常见的多级合并表头、跨页重复表头、前导零和字段别名。
结构化记录写 PostgreSQL，诊断文本与维修文本按用途写 Elasticsearch。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.storage.es_client import es_client
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

SCHEMAS = {
    "hmc": {
        "required": ["fault_desc", "repair_method"],
        "columns": {
            "hmc_code": ["HMC代码", "HMC码", "HMC_CODE", "HMC", "故障码", "故障代码"],
            "lcn": ["LCN", "LCN代码"],
            "hamc": ["HAMC", "HAMC代码"],
            "hsc": ["HSC", "HSC代码"],
            "fault_desc": ["故障描述现象", "故障描述", "故障现象", "故障描述及现象", "故障描述/现象"],
            "repair_method": [
                "故障分析程序维修方法", "故障分析程序/维修方法", "故障分析程序维修改法",
                "维修方法", "修复方法", "维修建议", "故障分析程序", "处理方法",
            ],
        },
    },
    "fmeca": {
        "required": ["code", "fault_mode"],
        "columns": {
            "code": ["代码", "编码", "CODE", "故障代码", "序号"],
            "product_flag": ["产品或功能标志", "产品标志", "产品标识", "PRODUCT_FLAG"],
            "func_management": ["功能管理", "功能管理项目", "功能名称", "功能", "FUNC"],
            "fault_mode": ["故障模式", "故障类型", "FAULT_MODE"],
            "fault_cause": ["故障原因", "原因", "FAULT_CAUSE"],
            "mission_phase": ["任务阶段与工作方式", "任务阶段", "工作方式", "阶段", "MISSION_PHASE"],
            "local_impact": ["故障影响局部", "局部影响", "故障局部影响", "LOCAL_IMPACT"],
            "final_impact": ["最终影响", "故障最终影响", "FINAL_IMPACT"],
            "severity": ["严酷度类别", "严酷度", "严重度", "SEVERITY", "等级"],
            "probability": ["故障概率", "概率", "发生概率", "PROBABILITY"],
            "detect_method": ["故障检测方法", "检测方法", "检测手段", "DETECT_METHOD"],
            "design_improve": ["设计改进措施", "设计改进", "改进措施", "DESIGN_IMPROVE"],
            "compensate_action": ["使用补偿措施", "补偿措施", "补偿动作", "COMPENSATE_ACTION"],
        },
    },
    "mfl": {
        "required": ["mfl_code"],
        "columns": {
            "mfl_code": ["MFL", "MFL代码", "MFL_CODE", "MFL码", "故障代码"],
            "module_fault": ["模块故障模式", "模块故障", "模块故障描述"],
            "sur_fault": ["SUR板故障模式", "SUR故障模式", "上层故障", "上级故障"],
            "subcircuit_fault": ["功能子电路级故障模式", "功能子电路故障模式", "子电路故障", "子电路"],
            "api_reportable": ["API是否可直接上报", "API可直接上报", "API上报", "API可上报", "是否上报"],
            "api_name": ["对应功能子电路API名称", "功能子电路API名称", "API名称", "API名"],
            "api_method": ["API调用方法", "API方法", "请求方法"],
            "api_usage": ["API使用方法", "API用法", "API说明", "用法"],
            "platform_test": ["是否进行平台级摸底实验", "平台级摸底实验", "平台测试", "平台可测", "是否平台测试"],
        },
    },
}


def _norm(value: object) -> str:
    text = "" if value is None else str(value)
    for ch in (" ", "\t", "\r", "\n", "_", "-", "/", "\\", "：", ":", "（", "）", "(", ")"):
        text = text.replace(ch, "")
    return text.upper()


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())


def _is_true(value: object) -> bool:
    return _norm(value) in {"是", "TRUE", "Y", "YES", "1", "支持", "可", "可以", "√"}


def _match_alias(value: str, alias_map: Dict[str, str]) -> Optional[str]:
    norm = _norm(value)
    if not norm:
        return None
    if norm in alias_map:
        return alias_map[norm]
    # 多级表头会形成“故障影响局部影响”等组合，允许别名被组合表头包含。
    matches = [(alias, canonical) for alias, canonical in alias_map.items() if alias and alias in norm]
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches[0][1]


def _looks_like_header(cells: Sequence[str], alias_map: Dict[str, str]) -> bool:
    return sum(1 for cell in cells if _match_alias(cell, alias_map)) >= 2


@ParserRegistry.register("hmc")
@ParserRegistry.register("fmeca")
@ParserRegistry.register("mfl")
class ExcelRowParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        return [] if ext in (".xlsx", ".xls", ".docx") else [
            f"不支持的格式: {ext}，表格类文档仅支持 .xlsx/.xls/.docx"
        ]

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        schema = SCHEMAS[ctx.doc_type]
        alias_map: Dict[str, str] = {}
        for canonical, aliases in schema["columns"].items():
            for alias in aliases:
                alias_map[_norm(alias)] = canonical

        rows_ok = 0
        rows_failed: List[Dict] = []
        vectors = 0
        total = 0
        matched_any = False
        index_name = es_client.get_main_index_name("default")
        es_client.create_index(index_name)

        for table_no, rows in self._iter_tables(file_path):
            detected = self._detect_header(rows, alias_map, schema["required"], ctx.doc_type)
            if detected is None:
                ctx.progress(f"[表 {table_no}] 未识别到 {ctx.doc_type} 表头，跳过")
                continue
            header_end, header_index, header_labels = detected
            matched_any = True
            ctx.progress(
                f"[表 {table_no}] 识别多级表头 {header_end + 1} 行，字段: {sorted(header_index.keys())}"
            )

            for row_idx, cells in enumerate(rows[header_end + 1:], start=header_end + 2):
                cells = [_clean_cell(cell) for cell in cells]
                if not any(cells):
                    continue
                if _looks_like_header(cells, alias_map):
                    # Word 跨页续表常会重复表头。
                    continue
                total += 1
                row = {
                    canonical: (cells[col_idx] if col_idx < len(cells) else "")
                    for canonical, col_idx in header_index.items()
                }
                self._normalize_business_row(ctx.doc_type, row)
                missing = self._missing_required(ctx.doc_type, row, schema["required"])
                if missing:
                    rows_failed.append({
                        "row": f"表{table_no}-行{row_idx}",
                        "reason": f"必填字段缺失: {missing}",
                        "header": header_labels,
                    })
                    continue
                try:
                    row_id = self._upsert(ctx.doc_type, ctx.doc_id, row)
                    vectors += self._vectorize(ctx, row, row_id, index_name)
                    rows_ok += 1
                    ctx.progress(f"[行 {total}] {self._business_code(ctx.doc_type, row)} 入库成功")
                except Exception as exc:
                    rows_failed.append({"row": f"表{table_no}-行{row_idx}", "reason": str(exc)[:300]})

        if not matched_any:
            raise ValueError(f"未找到可识别的 {ctx.doc_type} 数据表")

        stats = {
            "rows_ok": rows_ok,
            "rows_failed": len(rows_failed),
            "failed_rows": rows_failed[:100],
            "total": total,
            "vectors": vectors,
        }
        message = f"解析完成：成功 {rows_ok} 行，失败 {len(rows_failed)} 行，向量化 {vectors} 条"
        return ParseResult(
            doc_id=ctx.doc_id,
            doc_type=ctx.doc_type,
            stats=stats,
            vectors=vectors,
            message=message,
        )

    def _iter_tables(self, file_path: str) -> Iterable[Tuple[int, List[List[str]]]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            from docx import Document

            document = Document(file_path)
            for index, table in enumerate(document.tables, start=1):
                rows = [[_clean_cell(cell.text) for cell in row.cells] for row in table.rows]
                if rows:
                    yield index, rows
            return

        import pandas as pd

        sheets = pd.read_excel(file_path, sheet_name=None, header=None, dtype=str)
        for index, (_sheet_name, frame) in enumerate(sheets.items(), start=1):
            rows: List[List[str]] = []
            for _, series in frame.iterrows():
                cells = ["" if pd.isna(cell) else _clean_cell(cell) for cell in series.tolist()]
                if any(cells):
                    rows.append(cells)
            if rows:
                yield index, rows

    def _detect_header(
        self,
        rows: List[List[str]],
        alias_map: Dict[str, str],
        required: Sequence[str],
        doc_type: str,
    ) -> Optional[Tuple[int, Dict[str, int], List[str]]]:
        """在前 10 行中尝试 1~4 层表头组合。"""
        max_scan = min(10, len(rows))
        best = None
        best_score = -1
        for start in range(max_scan):
            for depth in range(1, min(4, max_scan - start) + 1):
                segment = rows[start:start + depth]
                width = max((len(row) for row in segment), default=0)
                mapping: Dict[str, int] = {}
                labels: List[str] = []
                for col_idx in range(width):
                    parts: List[str] = []
                    for row in segment:
                        value = row[col_idx] if col_idx < len(row) else ""
                        value = _clean_cell(value)
                        if value and (not parts or value != parts[-1]):
                            parts.append(value)
                    combined = " ".join(parts)
                    labels.append(combined)
                    canonical = _match_alias(combined, alias_map)
                    if canonical and canonical not in mapping:
                        mapping[canonical] = col_idx

                score = len(mapping)
                if doc_type == "hmc":
                    code_ok = "hmc_code" in mapping or all(k in mapping for k in ("lcn", "hamc", "hsc"))
                else:
                    code_ok = True
                required_ok = all(key in mapping for key in required)
                if code_ok and required_ok and score > best_score:
                    best_score = score
                    best = (start + depth - 1, mapping, labels)
        return best

    @staticmethod
    def _missing_required(doc_type: str, row: Dict[str, str], required: Sequence[str]) -> List[str]:
        missing = [key for key in required if not row.get(key)]
        if doc_type == "hmc" and not row.get("hmc_code"):
            missing.extend(key for key in ("lcn", "hamc", "hsc") if not row.get(key))
        return sorted(set(missing))

    @staticmethod
    def _normalize_business_row(doc_type: str, row: Dict[str, str]) -> None:
        for key, value in list(row.items()):
            row[key] = _clean_cell(value)

        if doc_type == "hmc":
            lcn = row.get("lcn", "").strip()
            hamc = row.get("hamc", "").strip() or "**"
            hsc = row.get("hsc", "").strip()
            if hsc and hsc.isdigit():
                hsc = hsc.zfill(4)
                row["hsc"] = hsc
            if lcn and lcn.isdigit():
                row["lcn"] = lcn
            row["hamc"] = hamc
            if not row.get("hmc_code") and lcn and hsc:
                row["hmc_code"] = f"{lcn}{hamc}{hsc}"
            elif row.get("hmc_code"):
                row["hmc_code"] = row["hmc_code"].replace("-", "").replace(" ", "")
        elif doc_type == "mfl":
            row["api_reportable"] = _is_true(row.get("api_reportable"))
            row["platform_test"] = _is_true(row.get("platform_test"))
        elif doc_type == "fmeca":
            severity = row.get("severity", "")
            row["severity"] = severity.replace("类", "").strip() if severity else ""

    @staticmethod
    def _upsert(doc_type: str, doc_id: str, row: Dict) -> int:
        if doc_type == "hmc":
            return postgres_client.upsert_hmc(doc_id, row)
        if doc_type == "fmeca":
            return postgres_client.upsert_fmea(doc_id, row)
        return postgres_client.upsert_mfl(doc_id, row)

    @staticmethod
    def _business_code(doc_type: str, row: Dict) -> str:
        if doc_type == "hmc":
            return row.get("hmc_code", "")
        if doc_type == "fmeca":
            return row.get("code", "")
        return row.get("mfl_code", "")

    def _index_vector(
        self,
        ctx: ParseContext,
        index_name: str,
        row_id: int,
        suffix: str,
        title: str,
        content: str,
        content_type: str,
        tags: Dict[str, object],
    ) -> int:
        if not content.strip():
            return 0
        vector = embedding_service.embed(content)
        chunk_id = f"{ctx.doc_id}_row_{row_id}_{suffix}"
        body = {
            "chunk_id": chunk_id,
            "doc_id": ctx.doc_id,
            "doc_name": ctx.file_name,
            "title": title,
            "section": ctx.doc_type,
            "content": content,
            "content_vector": vector,
            "kb_id": ctx.kb_id or ctx.doc_type,
            "content_type": content_type,
            "doc_type_kwd": ctx.doc_type,
            "pg_id_kwd": str(row_id),
            "removed_kwd": "N",
            "available_int": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **tags,
        }
        es_client.client.index(index=index_name, id=chunk_id, body=body)
        return 1

    def _vectorize(self, ctx: ParseContext, row: Dict, row_id: int, index_name: str) -> int:
        doc_type = ctx.doc_type
        if doc_type == "hmc":
            code = row.get("hmc_code", "")
            tags = {
                "business_id_kwd": code,
                "hmc_code_kwd": code,
                "lcn_kwd": row.get("lcn", ""),
                "hamc_kwd": row.get("hamc", ""),
                "hsc_kwd": row.get("hsc", ""),
            }
            diagnostic_text = (
                f"HMC: {code}\nLCN: {row.get('lcn', '')}；HAMC: {row.get('hamc', '')}；HSC: {row.get('hsc', '')}\n"
                f"故障现象: {row.get('fault_desc', '')}"
            )
            maintenance_text = (
                f"HMC: {code}\n故障现象: {row.get('fault_desc', '')}\n"
                f"故障分析程序与维修方法: {row.get('repair_method', '')}"
            )
            return (
                self._index_vector(ctx, index_name, row_id, "diagnostic", code, diagnostic_text, "hmc_diagnostic", tags)
                + self._index_vector(ctx, index_name, row_id, "maintenance", code, maintenance_text, "hmc_maintenance", tags)
            )

        if doc_type == "fmeca":
            content = (
                f"FMECA代码: {row.get('code', '')}\n产品/功能: {row.get('product_flag', '')} / {row.get('func_management', '')}\n"
                f"故障模式: {row.get('fault_mode', '')}\n故障原因: {row.get('fault_cause', '')}\n"
                f"任务阶段与工作方式: {row.get('mission_phase', '')}\n局部影响: {row.get('local_impact', '')}\n"
                f"最终影响: {row.get('final_impact', '')}\n严酷度: {row.get('severity', '')}\n"
                f"故障概率: {row.get('probability', '')}\n检测方法: {row.get('detect_method', '')}\n"
                f"设计改进: {row.get('design_improve', '')}\n使用补偿措施: {row.get('compensate_action', '')}"
            )
            tags = {
                "business_id_kwd": row.get("code", ""),
                "severity_kwd": row.get("severity", ""),
                "product_kwd": row.get("product_flag", ""),
            }
            return self._index_vector(
                ctx, index_name, row_id, "fmeca", row.get("code", ""), content, "fmeca_row", tags
            )

        content = (
            f"MFL: {row.get('mfl_code', '')}\n模块故障模式: {row.get('module_fault', '')}\n"
            f"SUR板故障模式: {row.get('sur_fault', '')}\n功能子电路级故障模式: {row.get('subcircuit_fault', '')}\n"
            f"API可直接上报: {'是' if row.get('api_reportable') else '否'}\n"
            f"API名称: {row.get('api_name', '')}\nAPI调用方法: {row.get('api_method', '')}\n"
            f"API使用方法: {row.get('api_usage', '')}\n平台级摸底实验: {'是' if row.get('platform_test') else '否'}"
        )
        tags = {
            "business_id_kwd": row.get("mfl_code", ""),
            "api_reportable_kwd": "true" if row.get("api_reportable") else "false",
            "platform_test_kwd": "true" if row.get("platform_test") else "false",
        }
        return self._index_vector(
            ctx, index_name, row_id, "mfl", row.get("mfl_code", ""), content, "mfl_row", tags
        )
