"""ICP 故障诊断智能体默认配置。

保留现有通用 Agent 数据表和 Dify 接口，仅确保系统首次启动时存在一个
面向七类 ICP 文档的专用智能体。重复启动不会重复创建。
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.agent.db import agent_db
from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)

ICP_AGENT_NAME = "ICP故障诊断智能体"
ICP_DOC_TYPES = [
    "function_locator",
    "hmc",
    "fmeca",
    "fta",
    "rbd",
    "mfl",
    "fault_history",
]

ICP_SYSTEM_PROMPT = """你是 ICP 航空电子系统故障诊断智能体。
你的结论必须来自已上传并解析的七类文档：系统架构与功能、HMC、FMECA、FTA、RBD、MFL、故障历史。

工作原则：
1. 用户通常只能描述模糊故障现象，先定位可能的系统/模块/功能，再检索候选故障。
2. 有完整 HMC/LCN/HAMC/HSC 时优先精确查询，不使用向量近似代替代码匹配。
3. 内部必须区分“资料明确记载”“跨文档关联推断”和“仍需确认”，但不要求把这些标签固定展示为用户可见章节；简单知识问答应直接回答用户所问内容。
4. 只有用户确实在请求故障定位且缺失信息会改变候选判断时才追问；信息不足时不得强行唯一定位。
5. FTA 用于原因链，FMECA 用于影响与严酷度，RBD 用于可靠性结构和计算，MFL 用于维修定位与 API，历史文档用于经验复用。
6. 维修建议必须附带证据来源；不得把合成测试数据表述为真实适航或放行依据。
7. 默认关闭长期记忆，避免旧会话中的非正式判断污染新的故障诊断。
8. 所有最终答案均由 LLM 依据门禁后的证据整理；影响解释、架构和其他知识问答不得套用“初步判断—候选故障—建议检查—下一步确认”的完整诊断模板。
9. 直接查询得到明确业务ID后，可查询经 PostgreSQL 逐边再授权的跨文档关系路径；语义命中的起点始终只是候选，不得把多跳路径描述成已确认的唯一因果链。
"""


def _find_existing_agent() -> Optional[str]:
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                "SELECT agent_id FROM agent_configs WHERE name=%s ORDER BY created_at ASC LIMIT 1",
                (ICP_AGENT_NAME,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def ensure_icp_diagnosis_agent() -> Optional[str]:
    """确保专用 ICP 智能体存在，返回 agent_id。"""
    try:
        agent_db.init_tables()
        existing = _find_existing_agent()
        if existing:
            # 旧版本可能仍绑定通用工具，启动时收敛为专用编排器。
            try:
                agent_db.update_agent(
                    existing,
                    description="基于七类 ICP 文档的专用故障诊断智能体",
                    system_prompt=ICP_SYSTEM_PROMPT,
                    kb_ids=ICP_DOC_TYPES,
                    enabled_tools=["icp_diagnosis", "graph_traversal"],
                    memory_enabled=False,
                    retrieval_top_k=12,
                    kg_top_k=12,
                    sql_limit=12,
                )
            except Exception as exc:
                logger.warning("Failed to refresh ICP agent configuration: %s", exc)
            return existing

        agent = agent_db.create_agent(
            name=ICP_AGENT_NAME,
            description="基于七类 ICP 文档的专用故障诊断智能体",
            system_prompt=ICP_SYSTEM_PROMPT,
            kb_ids=ICP_DOC_TYPES,
            enabled_tools=["icp_diagnosis", "graph_traversal"],
            temperature=0.2,
            top_p=0.8,
            max_tokens=4096,
            max_reflections=1,
            retrieval_top_k=12,
            summary_threshold=20,
            keep_recent_turns=8,
            streaming=True,
            memory_enabled=False,
            kg_top_k=12,
            sql_limit=12,
        )
        agent_id = agent.get("agent_id") if isinstance(agent, dict) else None
        logger.info("Created default ICP diagnosis agent: %s", agent_id)
        return agent_id
    except Exception as exc:
        logger.warning("Default ICP diagnosis agent initialization failed: %s", exc)
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return None
