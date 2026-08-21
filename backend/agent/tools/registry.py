"""工具注册：按 Agent 配置动态启用工具节点。"""
from typing import Dict, Callable, List


def get_tool_nodes(enabled_tools: List[str]) -> Dict[str, Callable]:
    """根据 Agent 配置动态返回启用的工具节点"""
    nodes = {}
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
    """列出所有可用工具（供前端展示）"""
    return [
        {"name": "vector_search", "label": "向量检索", "description": "ES 向量+关键词混合检索文档块"},
        {"name": "kg_search", "label": "知识图谱检索", "description": "HugeGraph 实体关系+社区报告检索"},
        {"name": "sql_query", "label": "数据库查询", "description": "查询图片/表格元数据"},
    ]
