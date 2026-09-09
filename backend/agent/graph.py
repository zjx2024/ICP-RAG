"""LangGraph 状态图编译。

- ICP 专用 Agent：scope_router → plan → 定向检索 → 受限自适应补查/按需查图
  → FTA/RBD 确定性计算 → 证据门禁 → answer

ICP 会话历史已经由 PostgreSQL 消息表在每次请求时显式注入，因此专用图不再叠加
LangGraph checkpoint 状态，避免上一轮的工具调用、候选和消息被 reducer 重复累积。
"""
import logging
from typing import Dict

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes.plan import plan_node, route_after_plan
from backend.agent.nodes.answer import answer_node
from backend.agent.nodes.direct_response import direct_response_node
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.nodes.scope_router import (
    SCOPE_AGENT_CAPABILITY,
    SCOPE_CLARIFICATION,
    SCOPE_KNOWLEDGE_BASE,
    SCOPE_OUT_OF_SCOPE,
    route_after_scope_router,
    scope_router_node,
)
from backend.agent.nodes.adaptive_tools import (
    adaptive_retrieval_node,
    adaptive_tool_route_node,
    route_after_adaptive_retrieval,
    route_after_adaptive_tools,
)
from backend.agent.tools.icp_diagnosis import icp_diagnosis_node
from backend.agent.tools.icp_calculation import icp_calculation_node
from backend.agent.tools.graph_traversal import build_graph_traversal_node

logger = logging.getLogger(__name__)

_compiled_graphs: Dict[str, object] = {}


async def _build_icp_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scope_router", scope_router_node)
    graph.add_node("direct_response", direct_response_node)
    graph.add_node("plan", plan_node)
    graph.add_node("icp_diagnosis", icp_diagnosis_node)
    graph.add_node("adaptive_tools", adaptive_tool_route_node)
    graph.add_node("adaptive_retrieval", adaptive_retrieval_node)
    try:
        # Imported lazily so focused planner/tests do not open PostgreSQL at
        # module import time. Missing authority code does not relax security:
        # the traversal request requires reauthorization and therefore fails
        # closed when this callback is unavailable.
        from backend.icp.graph_authority import reauthorize_graph_relations
    except (ImportError, AttributeError) as exc:
        logger.warning("Graph relation reauthorization is unavailable: %s", exc)
        reauthorize_graph_relations = None
    graph.add_node(
        "graph_traversal",
        build_graph_traversal_node(
            reauthorize_relations=reauthorize_graph_relations,
        ),
    )
    graph.add_node("icp_calculation", icp_calculation_node)
    graph.add_node("evidence_gate", evidence_gate_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("scope_router")
    graph.add_conditional_edges(
        "scope_router",
        route_after_scope_router,
        {
            SCOPE_KNOWLEDGE_BASE: "plan",
            SCOPE_AGENT_CAPABILITY: "direct_response",
            SCOPE_OUT_OF_SCOPE: "direct_response",
            SCOPE_CLARIFICATION: "direct_response",
        },
    )
    graph.add_edge("direct_response", END)
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "icp_diagnosis": "icp_diagnosis",
            "direct_response": "direct_response",
        },
    )
    graph.add_edge("icp_diagnosis", "adaptive_tools")
    graph.add_conditional_edges(
        "adaptive_tools",
        route_after_adaptive_tools,
        {
            "adaptive_retrieval": "adaptive_retrieval",
            "graph_traversal": "graph_traversal",
            "icp_calculation": "icp_calculation",
        },
    )
    graph.add_conditional_edges(
        "adaptive_retrieval",
        route_after_adaptive_retrieval,
        {
            "graph_traversal": "graph_traversal",
            "icp_calculation": "icp_calculation",
        },
    )
    graph.add_edge("graph_traversal", "icp_calculation")
    graph.add_edge("icp_calculation", "evidence_gate")
    graph.add_edge("evidence_gate", "answer")
    graph.add_edge("answer", END)
    # 每轮使用 chat_router 从数据库加载的显式历史，避免 checkpoint 与 reducer 重复叠加。
    return graph.compile()


async def build_graph(enabled_tools=None):
    # 旧通用 Agent 路径（vector/kg/sql → reflect）已退役，所有会话统一走 ICP 专用图。
    return await _build_icp_graph()


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
