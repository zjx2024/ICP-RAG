"""Agent 状态定义。

基于方案的 AgentState，使用 LangGraph TypedDict。
"""
from typing import TypedDict, Annotated, List, Dict
import operator


class AgentState(TypedDict, total=False):
    # ── 输入 ──
    messages: Annotated[List[Dict], operator.add]   # 对话历史（累加）
    query: str                                        # 用户原始问题
    kb_id: str                                        # 主知识库ID（兼容旧逻辑）
    kb_ids: List[str]                                 # 关联知识库列表（多库检索用）
    agent_id: str                                     # Agent 配置ID
    system_prompt: str                                # Agent 系统提示词
    chat_model_id: str                                # 用户选择的对话模型ID（覆盖 agent_models 默认）
    embedding_model_id: str                           # 用户选择的向量模型ID（覆盖系统默认）
    rerank_model_id: str                              # 用户选择的重排序模型ID（未配置则不使用重排序）

    # ── Agent 参数（来自 agent_configs / agent_retrieval_params）──
    temperature: float                                # LLM 采样温度
    top_p: float                                      # LLM Top P
    max_tokens: int                                   # LLM 最大生成 token 数
    max_reflections: int                              # 最大反思轮次
    retrieval_top_k: int                              # 向量检索返回条数（vector_search）
    kg_top_k: int                                     # 知识图谱检索返回条数（kg_search）
    sql_limit: int                                    # 数据库查询返回条数（sql_query）
    summary_threshold: int                            # 对话摘要阈值
    keep_recent_turns: int                            # 保留最近对话轮数
    memory_enabled: bool                              # 长期记忆开关（false 时不注入不抽取）

    # ── 记忆 ──
    memory_context: str                               # 注入的长期记忆

    # ── Plan 阶段 ──
    intent: str                                       # document|graph|table|hybrid
    rewritten_query: str                              # 改写后的查询
    entity_keywords: List[str]                        # 实体关键词
    type_keywords: List[str]                          # 类型关键词

    # ── Execute 阶段 ──
    vector_results: List[Dict]                        # ES 向量+文本检索
    kg_summary: str                                   # HG 图谱检索摘要
    sql_results: List[Dict]                           # PG 元数据查询

    # ── Reflect 阶段 ──
    reflection: str
    need_more: bool
    next_action: str                                  # retry_document|retry_graph|retry_table|done

    # ── 元信息 ──
    tool_calls: Annotated[List[Dict], operator.add]   # 工具调用记录（累加）
    citations: List[Dict]                             # 引用来源
    iteration: int                                    # 迭代次数（防无限循环）
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
        # 多库关联：显式 kb_ids 优先，否则兼容 kb_id 单库
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
        "vector_results": [],
        "kg_summary": "",
        "sql_results": [],
        "reflection": "",
        "need_more": False,
        "next_action": "done",
        "tool_calls": [],
        "citations": [],
        "iteration": 0,
        "final_answer": "",
    }
