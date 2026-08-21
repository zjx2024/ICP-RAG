"""LangGraph 状态图编译。

- ICP 专用 Agent：plan → 多源诊断 → FTA/RBD确定性计算 → answer
- 历史通用 Agent：保留 vector/kg/sql → reflect → answer 兼容路径

ICP 会话历史已经由 PostgreSQL 消息表在每次请求时显式注入，因此专用图不再叠加
LangGraph checkpoint 状态，避免上一轮的工具调用、候选和消息被 reducer 重复累积。
"""
import logging
from typing import Dict

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes.plan import plan_node
from backend.agent.nodes.reflect import reflect_node
from backend.agent.nodes.answer import answer_node
from backend.agent.tools.vector_search import vector_search_node
from backend.agent.tools.kg_search import kg_search_node
from backend.agent.tools.sql_query import sql_query_node
from backend.agent.tools.icp_diagnosis import icp_diagnosis_node
from backend.agent.tools.icp_calculation import icp_calculation_node
from backend.agent.checkpoint import get_checkpointer, get_store

logger = logging.getLogger(__name__)

_compiled_graphs: Dict[str, object] = {}

ICP_TOOL_NAMES = {
    "icp_diagnosis",
    "function_locator",
    "hybrid_search",
    "hmc_query",
    "fmeca_query",
    "fta_query",
    "fta_analyzer",
    "rbd_query",
    "rbd_calculator",
    "mfl_query",
    "fault_history_query",
    "fault_reasoning",
}


def _is_icp_agent(enabled_tools) -> bool:
    return bool(ICP_TOOL_NAMES.intersection(set(enabled_tools or [])))


def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "hybrid")
    original = (state.get("query") or "").lower()
    rewritten = (state.get("rewritten_query") or "").lower()
    combined = f"{original} {rewritten}"

    want_images = any(kw in combined for kw in ("图片", "image", "照片", "截图", "看图"))
    want_tables = any(kw in combined for kw in ("表格", "数据表", "参数表", "原始数据", "table"))
    want_diag = any(kw in combined for kw in (
        "为什么", "怎么回事", "原因", "导致", "由于", "造成", "故障", "如何解决", "怎么解决"
    ))

    if intent == "document":
        tools = ["vector_search"]
    elif intent == "graph":
        tools = ["kg_search"]
    elif intent == "table":
        tools = ["sql_query"]
    else:
        tools = ["vector_search", "kg_search"]

    if want_diag and "kg_search" not in tools:
        tools.append("kg_search")
    if (want_images or want_tables) and "sql_query" not in tools:
        tools.append("sql_query")
    return tools[0] if len(tools) == 1 else tools


def _decide_next(state: AgentState) -> str:
    next_action = state.get("next_action", "done")
    iteration = state.get("iteration", 0)
    if iteration >= 4:
        return "answer"
    return {
        "retry_document": "vector_search",
        "retry_graph": "kg_search",
        "retry_table": "sql_query",
    }.get(next_action, "answer")


async def _compile_legacy(graph: StateGraph):
    checkpointer = await get_checkpointer()
    store = await get_store()
    kwargs = {"checkpointer": checkpointer}
    if store:
        kwargs["store"] = store
    return graph.compile(**kwargs)


async def _build_icp_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("icp_diagnosis", icp_diagnosis_node)
    graph.add_node("icp_calculation", icp_calculation_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "icp_diagnosis")
    graph.add_edge("icp_diagnosis", "icp_calculation")
    graph.add_edge("icp_calculation", "answer")
    graph.add_edge("answer", END)
    # 每轮使用 chat_router 从数据库加载的显式历史，避免 checkpoint 与 reducer 重复叠加。
    return graph.compile()


async def _build_legacy_graph(enabled_tools):
    enabled_tools = enabled_tools or ["vector_search", "kg_search", "sql_query"]
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("answer", answer_node)

    available_tools = []
    if "vector_search" in enabled_tools:
        graph.add_node("vector_search", vector_search_node)
        available_tools.append("vector_search")
    if "kg_search" in enabled_tools:
        graph.add_node("kg_search", kg_search_node)
        available_tools.append("kg_search")
    if "sql_query" in enabled_tools:
        graph.add_node("sql_query", sql_query_node)
        available_tools.append("sql_query")

    graph.set_entry_point("plan")
    if not available_tools:
        graph.add_edge("plan", "answer")
    else:
        route_map = {
            "vector_search": "vector_search" if "vector_search" in available_tools else "reflect",
            "kg_search": "kg_search" if "kg_search" in available_tools else "reflect",
            "sql_query": "sql_query" if "sql_query" in available_tools else "reflect",
        }
        graph.add_conditional_edges("plan", _route_by_intent, route_map)
        for tool in available_tools:
            graph.add_edge(tool, "reflect")
        graph.add_conditional_edges(
            "reflect",
            _decide_next,
            {
                "vector_search": "vector_search",
                "kg_search": "kg_search",
                "sql_query": "sql_query",
                "answer": "answer",
            },
        )
    graph.add_edge("answer", END)
    return await _compile_legacy(graph)


async def build_graph(enabled_tools=None):
    enabled_tools = enabled_tools or ["icp_diagnosis"]
    if _is_icp_agent(enabled_tools):
        return await _build_icp_graph()
    return await _build_legacy_graph(enabled_tools)


async def get_compiled_graph(agent_id: str, enabled_tools=None):
    cache_key = f"{agent_id}:{','.join(sorted(enabled_tools or []))}"
    if cache_key not in _compiled_graphs:
        _compiled_graphs[cache_key] = await build_graph(enabled_tools)
    return _compiled_graphs[cache_key]


def clear_graph_cache(agent_id: str = None):
    if agent_id:
        keys = [key for key in _compiled_graphs if key.startswith(f"{agent_id}:")]
        for key in keys:
            del _compiled_graphs[key]
    else:
        _compiled_graphs.clear()
