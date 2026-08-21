"""表格类解析器（hmc / fmeca / mfl）：读 Excel 或 docx 内嵌表格 → 逐行 upsert PG 真源表 → 重要字段向量化 + 过滤标签写 ES。

一行一条存储（ON CONFLICT upsert 幂等），失败行跳过并记录原因。
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from backend.core.embedding import embedding_service
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry
from backend.storage.es_client import es_client
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

# 每类文档的表头别名表（canonical 列 -> 可接受表头，匹配时忽略大小写与空白/下划线）
SCHEMAS = {
    "hmc": {
        "required": ["hmc_code", "fault_desc", "repair_method"],
        "columns": {
            "hmc_code": ["HMC代码", "HMC码", "HMC_CODE", "HMC", "故障码"],
            "lcn": ["LCN", "LCN代码"],
            "hamc": ["HAMC", "HAMC代码"],
            "hsc": ["HSC", "HSC代码"],
            "fault_desc": ["故障描述现象", "故障描述", "故障现象", "故障描述及现象"],
            "repair_method": ["维修方法", "修复方法", "维修建议"],
        },
    },
    "fmeca": {
        "required": ["code", "fault_mode"],
        "columns": {
            "code": ["代码", "编码", "CODE", "故障代码"],
            "product_flag": ["产品标志", "产品标识", "PRODUCT_FLAG"],
            "func_management": ["功能管理", "功能名称", "FUNC"],
            "fault_mode": ["故障模式", "故障类型", "FAULT_MODE"],
            "fault_cause": ["故障原因", "原因", "FAULT_CAUSE"],
            "mission_phase": ["任务阶段", "阶段", "MISSION_PHASE"],
            "local_impact": ["局部影响", "LOCAL_IMPACT"],
            "final_impact": ["最终影响", "FINAL_IMPACT"],
            "severity": ["严酷度", "严重度", "SEVERITY", "等级"],
            "probability": ["概率", "发生概率", "PROBABILITY"],
            "detect_method": ["检测方法", "检测手段", "DETECT_METHOD"],
            "design_improve": ["设计改进", "改进措施", "DESIGN_IMPROVE"],
            "compensate_action": ["补偿措施", "补偿动作", "COMPENSATE_ACTION"],
        },
    },
    "mfl": {
        "required": ["mfl_code"],
        "columns": {
            "mfl_code": ["MFL代码", "MFL_CODE", "MFL码", "故障代码"],
            "module_fault": ["模块故障", "模块故障描述"],
            "sur_fault": ["上层故障", "上级故障"],
            "subcircuit_fault": ["子电路故障", "子电路"],
            "api_reportable": ["API上报", "API可上报", "是否上报"],
            "api_name": ["API名称", "API名"],
            "api_method": ["API方法", "请求方法"],
            "api_usage": ["API用法", "API说明", "用法"],
            "platform_test": ["平台测试", "平台可测", "是否平台测试"],
        },
    },
}


def _norm(header: str) -> str:
    """表头归一化：去空白/下划线/连字符并转大写。"""
    return "".join(header.split()).replace("_", "").replace("-", "").upper()


@ParserRegistry.register("hmc")
@ParserRegistry.register("fmeca")
@ParserRegistry.register("mfl")
class ExcelRowParser(Parser):
    def validate(self, file_path: str) -> List[str]:
        errors = []
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".xlsx", ".xls", ".docx"):
            errors.append(f"不支持的格式: {ext}，表格类文档仅支持 .xlsx/.xls/.docx")
        return errors

    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        schema = SCHEMAS[ctx.doc_type]
        required = schema["required"]
        aliases = schema["columns"]
        # 归一化别名表
        norm_aliases: Dict[str, str] = {}
        for col, names in aliases.items():
            for n in names:
                norm_aliases[_norm(n)] = col

        rows_ok = 0
        rows_failed: List[Dict] = []
        vectors = 0
        total = 0
        index_name = es_client.get_main_index_name("default")

        tables = self._iter_tables(file_path)
        matched_any = False

        for table_no, header, data_rows in tables:
            header_index: Dict[str, int] = {}
            for idx, cell in enumerate(header):
                col = norm_aliases.get(_norm(cell))
                if col and col not in header_index:
                    header_index[col] = idx
            missing = [c for c in required if c not in header_index]
            if missing:
                ctx.progress(f"[表 {table_no}] 表头缺列: {missing}，跳过该表")
                continue
            matched_any = True

            for row_no, cells in enumerate(data_rows, start=2):
                if not any(cells):
                    continue
                total += 1
                row = {col: (cells[idx] if idx < len(cells) else "").strip() for col, idx in header_index.items()}
                miss_vals = [c for c in required if not row.get(c)]
                if miss_vals:
                    rows_failed.append({"row": f"表{table_no}-行{row_no}", "reason": f"必填列缺失: {miss_vals}"})
                    continue
                try:
                    row_id = self._upsert(ctx.doc_type, ctx.doc_id, row)
                    ok = self._vectorize(ctx, row, row_id, index_name)
                    if ok:
                        vectors += 1
                    rows_ok += 1
                    ctx.progress(f"[行 {total}] {self._business_code(ctx.doc_type, row)} 入库成功")
                except Exception as e:
                    rows_failed.append({"row": f"表{table_no}-行{row_no}", "reason": str(e)[:200]})

        if not matched_any:
            raise ValueError(f"未找到匹配 {ctx.doc_type} 模板的表格，需要表头列: {required}")

        stats = {
            "rows_ok": rows_ok,
            "rows_failed": len(rows_failed),
            "failed_rows": rows_failed,
            "total": total,
            "vectors": vectors,
        }
        message = f"解析完成：成功 {rows_ok} 行，失败 {len(rows_failed)} 行，向量化 {vectors} 条"
        return ParseResult(doc_id=ctx.doc_id, doc_type=ctx.doc_type, stats=stats,
                           vectors=vectors, message=message)

    # ---------- 内部实现 ----------

    def _iter_tables(self, file_path: str):
        """产出 (table_no, header, data_rows)。Excel 取第一个工作表；docx 遍历所有表格。"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            for idx, table in enumerate(doc.tables, start=1):
                rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
                header, data = self._split_header(rows)
                if header:
                    yield idx, header, data
        else:
            rows = self._read_excel_rows(file_path)
            header, data = self._split_header(rows)
            if header:
                yield 1, header, data

    def _read_excel_rows(self, file_path: str) -> List[List[str]]:
        import pandas as pd
        xls = pd.read_excel(file_path, sheet_name=0, header=None, dtype=str)
        rows = []
        for _, r in xls.iterrows():
            cells = ["" if pd.isna(c) else str(c).strip() for c in r.tolist()]
            if any(cells):
                rows.append(cells)
        return rows

    @staticmethod
    def _split_header(rows: List[List[str]]):
        for idx, row in enumerate(rows):
            if any(row):
                return row, rows[idx + 1:]
        return None, []

    def _upsert(self, doc_type: str, doc_id: str, row: Dict) -> int:
        if doc_type == "hmc":
            return postgres_client.upsert_hmc(doc_id, row)
        if doc_type == "fmeca":
            return postgres_client.upsert_fmea(doc_id, row)
        return postgres_client.upsert_mfl(doc_id, row)

    def _business_code(self, doc_type: str, row: Dict) -> str:
        if doc_type == "hmc":
            return row.get("hmc_code", "")
        if doc_type == "fmeca":
            return row.get("code", "")
        return row.get("mfl_code", "")

    def _vectorize(self, ctx: ParseContext, row: Dict, row_id: int, index_name: str) -> bool:
        """重要字段拼接 → embedding → ES（含过滤标签字段）。返回是否成功。"""
        doc_type = ctx.doc_type
        if doc_type == "hmc":
            content = (f"HMC码: {row.get('hmc_code', '')} "
                       f"LCN: {row.get('lcn', '**')} HAMC: {row.get('hamc', '**')} HSC: {row.get('hsc', '**')}\n"
                       f"故障描述: {row.get('fault_desc', '')}\n维修方法: {row.get('repair_method', '')}")
            tags = {
                "business_id_kwd": row.get("hmc_code", ""),
                "hmc_code_kwd": row.get("hmc_code", ""),
                "lcn_kwd": row.get("lcn", "**"),
                "hamc_kwd": row.get("hamc", "**"),
                "hsc_kwd": row.get("hsc", "**"),
            }
        elif doc_type == "fmeca":
            content = (f"故障代码: {row.get('code', '')} 产品: {row.get('product_flag', '')} "
                       f"功能: {row.get('func_management', '')}\n"
                       f"故障模式: {row.get('fault_mode', '')}\n故障原因: {row.get('fault_cause', '')}\n"
                       f"局部影响: {row.get('local_impact', '')}\n最终影响: {row.get('final_impact', '')}\n"
                       f"检测方法: {row.get('detect_method', '')}\n设计改进: {row.get('design_improve', '')}\n"
                       f"补偿措施: {row.get('compensate_action', '')}")
            tags = {
                "business_id_kwd": row.get("code", ""),
                "severity_kwd": row.get("severity", ""),
                "product_kwd": row.get("product_flag", ""),
            }
        else:  # mfl
            content = (f"MFL码: {row.get('mfl_code', '')}\n"
                       f"模块故障: {row.get('module_fault', '')}\n上层故障: {row.get('sur_fault', '')}\n"
                       f"子电路故障: {row.get('subcircuit_fault', '')}\n"
                       f"API: {row.get('api_name', '')} {row.get('api_method', '')}\n用法: {row.get('api_usage', '')}")
            tags = {
                "business_id_kwd": row.get("mfl_code", ""),
                "api_reportable_kwd": "true" if row.get("api_reportable") in (True, "是", "true", "TRUE", "1") else "false",
                "platform_test_kwd": "true" if row.get("platform_test") in (True, "是", "true", "TRUE", "1") else "false",
            }

        vector = embedding_service.embed(content)
        doc = {
            "chunk_id": f"{ctx.doc_id}_row_{row_id}",
            "doc_id": ctx.doc_id,
            "doc_name": ctx.file_name,
            "title": self._business_code(doc_type, row),
            "section": doc_type,
            "content": content,
            "content_vector": vector,
            "kb_id": ctx.kb_id or doc_type,
            "content_type": "table_row",
            "doc_type_kwd": doc_type,
            "pg_id_kwd": str(row_id),
            "removed_kwd": "N",
            "available_int": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **tags,
        }
        es_client.client.index(index=index_name, id=doc["chunk_id"], body=doc)
        return True
