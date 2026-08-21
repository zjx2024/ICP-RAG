"""LangGraph 状态图编译。

设计原则：
- 状态图: plan → route → execute(可并行) → reflect → answer
- 复用现有 PG 作为 Checkpoint（PostgresSaver）和 Store（PostgresStore）
- 路由函数根据 intent 分发到对应工具节点
- Reflect 决定是否重试（最多 max_reflections 次）
"""
import logging
from typing import Dict, Literal

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes.plan import plan_node
from backend.agent.nodes.reflect import reflect_node
from backend.agent.nodes.answer import answer_node
from backend.agent.tools.vector_search import vector_search_node
from backend.agent.tools.kg_search import kg_search_node
from backend.agent.tools.sql_query import sql_query_node
from backend.agent.checkpoint import get_checkpointer, get_store

logger = logging.getLogger(__name__)

# 全局编译实例缓存（按 agent_id 缓存，因为不同 Agent 启用的工具不同）
_compiled_graphs: Dict[str, object] = {}


def _route_by_intent(state: AgentState) -> str:
    """根据意图路由（支持并行通过返回列表）"""
    intent = state.get("intent", "hybrid")
    # 同时检查原始 query 和 rewritten_query（rewritten 可能丢失关键词）
    original = (state.get("query") or "").lower()
    rewritten = (state.get("rewritten_query") or "").lower()
    combined = f"{original} {rewritten}"

    # 图片相关请求：始终路由到 sql_query（图片元数据存在 PG 中）
    image_keywords = ("图片", "image", "放映", "照片", "截图", "看图")
    want_images = any(kw in combined for kw in image_keywords)

    # 表格相关请求：始终路由到 sql_query（表格数据存在 PG document_tables 中）
    table_keywords = ("表格", "数据表", "参数表", "标准表", "原始数据", "table", "查看表", "看看表")
    want_tables = any(kw in combined for kw in table_keywords)

    # 诊断类问题（为什么/怎么回事/原因/导致）：因果链在图谱与故障树中，
    # 强制纳入 kg_search，确保"过热→各层原因"的完整因果链被检索
    diag_keywords = ("为什么", "怎么回事", "原因", "导致", "由于", "造成", "故障",
                     "如何解决", "怎么解决", "为什么会", "是什么原因")
    want_diag = any(kw in combined for kw in diag_keywords)

    # 单意图返回单节点名；hybrid 返回多个节点名（LangGraph 会并行执行）
    if intent == "document":
        tools = ["vector_search"]
    elif intent == "graph":
        tools = ["kg_search"]
    elif intent == "table":
        tools = ["sql_query"]
    else:  # hybrid
        tools = ["vector_search", "kg_search"]

    # 诊断类问题追加 kg_search（图谱因果链 / 故障树）
    if want_diag and "kg_search" not in tools:
        tools.append("kg_search")

    # 图片/表格请求追加 sql_query
    if (want_images or want_tables) and "sql_query" not in tools:
        tools.append("sql_query")

    return tools[0] if len(tools) == 1 else tools


def _decide_next(state: AgentState) -> str:
    """Reflect 后决策下一步"""
    next_action = state.get("next_action", "done")
    iteration = state.get("iteration", 0)

    # 超过最大反思次数，强制结束
    if iteration >= 4:
        return "answer"

    if next_action == "retry_document":
        return "vector_search"
    elif next_action == "retry_graph":
        return "kg_search"
    elif next_action == "retry_table":
        return "sql_query"
    else:
        return "answer"


async def build_graph(enabled_tools=None):
    """构建 LangGraph 状态图

    Args:
        enabled_tools: Agent 启用的工具列表，决定哪些工具节点可用
    """
    enabled_tools = enabled_tools or ["vector_search", "kg_search", "sql_query"]

    graph = StateGraph(AgentState)

    # ── 节点 ──
    graph.add_node("plan", plan_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("answer", answer_node)

    if "vector_search" in enabled_tools:
        graph.add_node("vector_search", vector_search_node)
    if "kg_search" in enabled_tools:
        graph.add_node("kg_search", kg_search_node)
    if "sql_query" in enabled_tools:
        graph.add_node("sql_query", sql_query_node)

    # ── 边 ──
    graph.set_entry_point("plan")

    # Plan → Route（根据意图分发，可并行）
    available_tools = []
    if "vector_search" in enabled_tools:
        available_tools.append("vector_search")
    if "kg_search" in enabled_tools:
        available_tools.append("kg_search")
    if "sql_query" in enabled_tools:
        available_tools.append("sql_query")

    if not available_tools:
        # 无工具时直接回答
        graph.add_edge("plan", "answer")
    else:
        # Plan → 工具节点
        graph.add_conditional_edges("plan", _route_by_intent, {
            "vector_search": "vector_search" if "vector_search" in enabled_tools else "reflect",
            "kg_search": "kg_search" if "kg_search" in enabled_tools else "reflect",
            "sql_query": "sql_query" if "sql_query" in enabled_tools else "reflect",
            # hybrid 并行
            # LangGraph 支持返回列表触发并行
        })

        # 工具 → Reflect
        for tool in available_tools:
            graph.add_edge(tool, "reflect")

        # Reflect → 决策
        graph.add_conditional_edges("reflect", _decide_next, {
            "vector_search": "vector_search",
            "kg_search": "kg_search",
            "sql_query": "sql_query",
            "answer": "answer",
        })

    # Answer → 结束
    graph.add_edge("answer", END)

    # ── 编译：复用现有 PG ──
    checkpointer = await get_checkpointer()
    store = await get_store()

    compile_kwargs = {"checkpointer": checkpointer}
    if store:
        compile_kwargs["store"] = store

    return graph.compile(**compile_kwargs)


async def get_compiled_graph(agent_id: str, enabled_tools=None):
    """获取编译后的图（按 agent_id 缓存）"""
    cache_key = f"{agent_id}:{','.join(sorted(enabled_tools or []))}"
    if cache_key not in _compiled_graphs:
        _compiled_graphs[cache_key] = await build_graph(enabled_tools)
    return _compiled_graphs[cache_key]


def clear_graph_cache(agent_id: str = None):
    """清除图缓存（Agent 配置更新后调用）"""
    if agent_id:
        keys_to_remove = [k for k in _compiled_graphs if k.startswith(f"{agent_id}:")]
        for k in keys_to_remove:
            del _compiled_graphs[k]
    else:
        _compiled_graphs.clear()
