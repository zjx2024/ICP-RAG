"""七类 ICP 原始文档注册表规范化。"""
from __future__ import annotations

import logging

from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

DOCUMENT_TYPES = [
    (
        "function_locator",
        "系统架构与功能说明",
        "系统/子系统/模块/功能/接口及典型失效表现，是 Function Locator 的知识骨架",
        "text",
        ".docx,.xlsx,.xlsm,.xls,.txt,.md",
        "function_architecture",
        "icp_functions",
        "function_locator",
        False,
        1,
    ),
    (
        "hmc",
        "HMC 故障代码手册",
        "LCN/HAMC/HSC、故障描述现象以及故障分析程序/维修方法",
        "table",
        ".docx,.xlsx,.xlsm,.xls",
        "hmc",
        "fault_hmc",
        "hmc",
        False,
        2,
    ),
    (
        "fta",
        "FTA 故障树分析",
        "顶事件、中间事件、基本事件及 AND/OR 逻辑关系",
        "image",
        ".docx,.png,.jpg,.jpeg,.bmp,.gif,.webp",
        "fta",
        "document_fault_trees",
        "fta",
        True,
        3,
    ),
    (
        "rbd",
        "RBD 可靠性框图分析",
        "功能块、串并联/冗余结构、成功判据与可靠性参数",
        "image",
        ".docx,.png,.jpg,.jpeg,.bmp,.gif,.webp",
        "rbd",
        "document_fault_trees",
        "rbd",
        True,
        4,
    ),
    (
        "fmeca",
        "FMECA 故障模式影响及危害性分析",
        "故障模式、原因、局部/最终影响、严酷度、概率、检测与改进措施",
        "table",
        ".docx,.xlsx,.xlsm,.xls",
        "fmeca",
        "fmea",
        "fmeca",
        False,
        5,
    ),
    (
        "mfl",
        "MFL 维修故障清单",
        "模块/SUR板/功能子电路故障模式、API 上报和维修定位信息",
        "table",
        ".docx,.xlsx,.xlsm,.xls",
        "mfl",
        "mfl",
        "mfl",
        False,
        6,
    ),
    (
        "fault_history",
        "故障历史查询文档",
        "历史故障现象、排故方法、根因、处理步骤、最终方案和关联文档",
        "text",
        ".docx,.txt,.md",
        "fault_history",
        "fault_history_cases",
        "fault_history",
        False,
        7,
    ),
]


def normalize_document_types() -> None:
    """以幂等 upsert 更新七类文档名称、顺序和解析元数据。"""
    try:
        with postgres_client.conn.cursor() as cur:
            for row in DOCUMENT_TYPES:
                cur.execute(
                    """
                    INSERT INTO document_types
                        (doc_type, name, description, data_form, input_formats, parser_key,
                         pg_source, es_doc_type, needs_kg, is_active, sort_order)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
                    ON CONFLICT (doc_type) DO UPDATE SET
                        name=EXCLUDED.name,
                        description=EXCLUDED.description,
                        data_form=EXCLUDED.data_form,
                        input_formats=EXCLUDED.input_formats,
                        parser_key=EXCLUDED.parser_key,
                        pg_source=EXCLUDED.pg_source,
                        es_doc_type=EXCLUDED.es_doc_type,
                        needs_kg=EXCLUDED.needs_kg,
                        is_active=TRUE,
                        sort_order=EXCLUDED.sort_order
                    """,
                    row,
                )
        postgres_client.conn.commit()
    except Exception as exc:
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        logger.warning("Failed to normalize ICP document types: %s", exc)
        # The startup layer decides whether this is a warning (developer mode)
        # or a fatal dependency error (strict/container mode). Swallowing it
        # here would let a service report ready with a stale seven-type registry.
        raise
