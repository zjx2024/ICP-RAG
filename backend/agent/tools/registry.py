"""工具注册。

保留旧通用工具用于兼容历史 Agent；默认 ICP 智能体使用 icp_diagnosis 编排器，
编排器内部会按需调用七类文档的专用工具并逐项返回调用记录。
"""
from typing import Dict, Callable, List


ICP_TOOL_DEFS = [
    {"name": "function_locator", "label": "功能定位", "description": "将模糊现象定位到 ICP 系统、模块和功能候选"},
    {"name": "hybrid_search", "label": "混合召回", "description": "在七类文档中执行关键词与向量混合检索"},
    {"name": "hmc_query", "label": "HMC 查询", "description": "精确查询 LCN/HAMC/HSC 与故障现象、维修方法"},
    {"name": "fmeca_query", "label": "FMECA 查询", "description": "查询故障模式、原因、影响、严酷度和检测方法"},
    {"name": "fta_query", "label": "FTA 查询", "description": "查询顶事件、原因链与 AND/OR 逻辑"},
    {"name": "fta_analyzer", "label": "FTA 确定性分析", "description": "计算最小割集和顶事件概率，缺参时明确报告"},
    {"name": "rbd_query", "label": "RBD 查询", "description": "查询可靠性框图、串并联结构和成功判据"},
    {"name": "rbd_calculator", "label": "RBD 可靠度计算", "description": "基于成功路径和参数确定性计算系统可靠度"},
    {"name": "mfl_query", "label": "MFL 查询", "description": "查询维修故障清单、功能电路和 API 信息"},
    {"name": "fault_history_query", "label": "故障历史", "description": "检索历史排故案例、根因、步骤和最终处置"},
    {
        "name": "graph_traversal",
        "label": "跨文档关系查询",
        "description": "从明确业务ID出发查询经 PostgreSQL 逐边再授权的受限多跳关系",
    },
    {"name": "fault_reasoning", "label": "故障推理", "description": "融合多源证据、排序候选并决定是否需要追问"},
]


def get_tool_nodes(enabled_tools: List[str]) -> Dict[str, Callable]:
    nodes: Dict[str, Callable] = {}
    if "icp_diagnosis" in enabled_tools or any(
        item["name"] in enabled_tools for item in ICP_TOOL_DEFS
    ):
        from backend.agent.tools.icp_diagnosis import icp_diagnosis_node
        nodes["icp_diagnosis"] = icp_diagnosis_node

    if "vector_search" in enabled_tools:
        from backend.agent.tools.vector_search import vector_search_node
        nodes["vector_search"] = vector_search_node
    if "kg_search" in enabled_tools:
        from backend.agent.tools.kg_search import kg_search_node
        nodes["kg_search"] = kg_search_node
    if "sql_query" in enabled_tools:
        from backend.agent.tools.sql_query import sql_query_node
        nodes["sql_query"] = sql_query_node
    return nodes


def get_available_tools() -> List[Dict]:
    return [
        {
            "name": "icp_diagnosis",
            "label": "ICP 故障诊断编排器",
            "description": "自动调度 Function Locator、HMC、FMECA、FTA、RBD、MFL、故障历史及确定性计算工具",
        },
        *ICP_TOOL_DEFS,
    ]
