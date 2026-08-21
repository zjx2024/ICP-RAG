"""Agent 状态定义。

同时兼容旧通用 RAG Agent 与新的 ICP 故障诊断编排器。
"""
from typing import TypedDict, Annotated, List, Dict
import operator


class AgentState(TypedDict, total=False):
    # ── 输入 ──
    messages: Annotated[List[Dict], operator.add]
    query: str
    kb_id: str
    kb_ids: List[str]
    agent_id: str
    system_prompt: str
    chat_model_id: str
    embedding_model_id: str
    rerank_model_id: str

    # ── Agent 参数 ──
    temperature: float
    top_p: float
    max_tokens: int
    max_reflections: int
    retrieval_top_k: int
    kg_top_k: int
    sql_limit: int
    summary_threshold: int
    keep_recent_turns: int
    memory_enabled: bool

    # ── 记忆 ──
    memory_context: str

    # ── Plan 阶段 ──
    intent: str
    rewritten_query: str
    entity_keywords: List[str]
    type_keywords: List[str]
    tools_plan: List[Dict]
    plan_reasoning: str

    # ── 旧通用执行结果（兼容）──
    vector_results: List[Dict]
    kg_summary: str
    sql_results: List[Dict]

    # ── ICP 专用执行结果 ──
    domain_results: List[Dict]
    diagnostic_context: str
    candidate_faults: List[Dict]
    diagnostic_confidence: float
    clarification_question: str

    # ── Reflect 阶段 ──
    reflection: str
    need_more: bool
    next_action: str

    # ── 元信息 ──
    tool_calls: Annotated[List[Dict], operator.add]
    citations: List[Dict]
    iteration: int
    final_answer: str


# 默认状态工厂
def create_initial_state(query: str, agent_id: str, kb_id: str, kb_ids: List[str] = None,
                         system_prompt: str = "", memory_context: str = "",
                         history: List[Dict] = None,
                         chat_model_id: str = "", embedding_model_id: str = "",
                         rerank_model_id: str = "",
                         temperature: float = 0.7, top_p: float = 0.9,
                         max_tokens: int = 2048, max_reflections: int = 3,
                         retrieval_top_k: int = 5, kg_top_k: int = 5,
                         sql_limit: int = 5, summary_threshold: int = 12,
                         keep_recent_turns: int = 6, memory_enabled: bool = True) -> AgentState:
    return {
        "messages": history or [],
        "query": query,
        "kb_id": kb_id,
        "kb_ids": [k for k in (kb_ids or ([] if not kb_id else [kb_id])) if k],
        "agent_id": agent_id,
        "system_prompt": system_prompt,
        "chat_model_id": chat_model_id,
        "embedding_model_id": embedding_model_id,
        "rerank_model_id": rerank_model_id,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_reflections": max_reflections,
        "retrieval_top_k": retrieval_top_k,
        "kg_top_k": kg_top_k,
        "sql_limit": sql_limit,
        "summary_threshold": summary_threshold,
        "keep_recent_turns": keep_recent_turns,
        "memory_enabled": memory_enabled,
        "memory_context": memory_context,
        "intent": "",
        "rewritten_query": query,
        "entity_keywords": [],
        "type_keywords": [],
        "tools_plan": [],
        "plan_reasoning": "",
        "vector_results": [],
        "kg_summary": "",
        "sql_results": [],
        "domain_results": [],
        "diagnostic_context": "",
        "candidate_faults": [],
        "diagnostic_confidence": 0.0,
        "clarification_question": "",
        "reflection": "",
        "need_more": False,
        "next_action": "done",
        "tool_calls": [],
        "citations": [],
        "iteration": 0,
        "final_answer": "",
    }
