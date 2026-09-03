"""Agent 状态定义。

同时兼容旧通用 RAG Agent 与新的 ICP 故障诊断编排器。
"""
from typing import TypedDict, Annotated, List, Dict
import operator


class AgentState(TypedDict, total=False):
    messages: Annotated[List[Dict], operator.add]
    query: str
    kb_id: str
    kb_ids: List[str]
    agent_id: str
    system_prompt: str
    chat_model_id: str
    embedding_model_id: str
    rerank_model_id: str

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

    memory_context: str

    # 输入范围先于领域意图判断。非知识库请求在图入口直接结束，不能
    # 进入规划器、检索工具或证据门禁。
    agent_name: str
    scope_route: str
    scope_route_source: str
    scope_route_reason: str
    scope_confidence: float

    intent: str
    # A question may request cause + repair + history at the same time.  intent
    # remains the primary answer contract; query_targets drive tool coverage.
    query_targets: List[Dict]
    # Calculation authority is parsed only from the raw user request.
    calculation_requests: List[str]
    # 规划模型对“完整回答是否需要跨文档关系补全”的语义判断。
    # 它不授予任意图查询权限；实际执行仍需 authorized_tools 和业务 seed。
    relation_expansion: Dict
    # 稳定的用户回答契约；由确定性 intent 映射产生，不由回答模型自由选择。
    answer_mode: str
    rewritten_query: str
    entity_keywords: List[str]
    type_keywords: List[str]
    query_frame: Dict
    # Planner topology intent enriched after authoritative entity resolution.
    # ``canonical_business_ids`` may only come from persisted structured IDs;
    # the answer layer uses it to authorize one exact RBD redundancy group.
    topology_dependency: Dict
    # ``query_frame`` may carry a model-authored complete-collection
    # obligation. It becomes public only after PostgreSQL proves the set is
    # closed; semantic recall alone can never satisfy it.
    collection_resolution: Dict
    # 每轮根据持久化会话历史重新生成的指代消解结果。它只为检索提供
    # 当前讨论对象和本轮所问维度，不能作为工程事实或证据来源。
    dialogue_context: Dict
    # 最近回答中实际使用过的证据身份。规划器只能从 offered anchors 中
    # 选择，执行节点随后回 PostgreSQL 重读；旧答案正文不作为事实复用。
    conversation_anchors: List[Dict]
    selected_conversation_anchors: List[Dict]
    # 面向后续节点的稳定任务摘要；它只整理 planner 已输出的结构化槽位，
    # 不额外推断实体、关系或工具权限。
    task_profile: Dict
    normalized_terms: List[str]
    # Planner-authored retrieval wording per query tool; never grants authority.
    tool_queries: Dict[str, List[str]]
    # Exact document families explicitly requested by the user (for example
    # “请补查 RBD”).  This is a source-coverage obligation, not an intent or
    # calculator authorization; the answer layer reports an empty query rather
    # than silently substituting another document type.
    requested_source_tools: List[str]
    tools_plan: List[Dict]
    # tools_plan 用于展示 LLM 的规划；authorized_tools 是意图授权上限，
    # allowed_tools 才是策略校验后本轮实际选择/执行的工具集。
    authorized_tools: List[str]
    allowed_tools: List[str]
    authorized_doc_types: List[str]
    hybrid_doc_types: List[str]
    retrieval_scope: Dict
    plan_reasoning: str

    vector_results: List[Dict]
    kg_summary: str
    sql_results: List[Dict]

    domain_results: List[Dict]
    # OpenSearch 只负责发现工程图；这里保留一个有界、只含稳定定位字段的
    # handoff，供自适应补查按 diagram_id 回到 PostgreSQL 验真。它不是证据
    # 真源，也不能直接生成图结论。
    search_locator_hits: List[Dict]
    diagnostic_context: str
    candidate_faults: List[Dict]
    # 证据支持度不是故障概率；diagnostic_confidence 仅保留给旧链路兼容。
    diagnostic_support: float
    diagnostic_confidence: float
    clarification_question: str
    calculation_results: List[Dict]
    confirmed_graphs: List[Dict]

    # PostgreSQL 查询结果绑定到可重建图投影的稳定 logical ID。语义召回
    # 只能产生 candidate 绑定；只有精确业务代码可以标记为 exact。
    graph_seed_bindings: List[Dict]
    graph_traversal_request: Dict
    graph_paths: List[Dict]
    # 首轮检索后的受限自适应补查；原始模型输出不对外展示。
    pending_tools: List[str]
    adaptive_graph_requested: bool
    adaptive_round: int
    adaptive_decision: Dict

    # 新 ICP 证据链路。列表 reducer 允许检索与计算节点逐步追加。
    tool_results: Annotated[List[Dict], operator.add]
    evidence_registry: Annotated[List[Dict], operator.add]
    context_evidence_ids: List[str]
    approved_evidence_ids: List[str]
    safety_decision: Dict

    reflection: str
    need_more: bool
    next_action: str

    tool_calls: Annotated[List[Dict], operator.add]
    # 回答实际引用与过程候选证据分开：citations 只包含正文中的 [^n]，
    # evidence_catalog 保留本轮受限证据窗口，供执行步骤按稳定 evidence_id 关联。
    citations: List[Dict]
    evidence_catalog: List[Dict]
    used_evidence_ids: List[str]
    iteration: int
    final_answer: str
    # 答案完成度状态：full=必答字段完整，partial=存在未从已批准证据确认的字段。
    # 用于区分“部分确认答案”与“系统错误”，前端据此展示而不是当作失败。
    answer_completion_status: str
    answer_unconfirmed_fields: List[str]


def create_initial_state(query: str, agent_id: str, kb_id: str, kb_ids: List[str] = None,
                         system_prompt: str = "", memory_context: str = "",
                         history: List[Dict] = None,
                         agent_name: str = "",
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
        "agent_name": agent_name or "ICP故障诊断智能体",
        "scope_route": "unclassified",
        "scope_route_source": "",
        "scope_route_reason": "",
        "scope_confidence": 0.0,
        "intent": "",
        "query_targets": [],
        "calculation_requests": [],
        "relation_expansion": {"needed": False, "reason": ""},
        "answer_mode": "fault_diagnosis",
        "rewritten_query": query,
        "entity_keywords": [],
        "type_keywords": [],
        "query_frame": {},
        "topology_dependency": {},
        "collection_resolution": {
            "requested": False,
            "closed": False,
            "member_kind": "",
            "subject": "",
            "reason": "",
        },
        "dialogue_context": {
            "contextualized_query": query,
            "subjects": [],
            "requested_facets": [],
            "required_answer_fields": [],
            "context_used": False,
            "status": "standalone",
        },
        "conversation_anchors": [],
        "selected_conversation_anchors": [],
        "task_profile": {},
        "normalized_terms": [],
        "tool_queries": {},
        "tools_plan": [],
        "authorized_tools": [],
        "allowed_tools": [],
        "authorized_doc_types": [],
        "hybrid_doc_types": [],
        "retrieval_scope": {},
        "plan_reasoning": "",
        "vector_results": [],
        "kg_summary": "",
        "sql_results": [],
        "domain_results": [],
        "search_locator_hits": [],
        "diagnostic_context": "",
        "candidate_faults": [],
        "diagnostic_support": 0.0,
        "diagnostic_confidence": 0.0,
        "clarification_question": "",
        "calculation_results": [],
        "confirmed_graphs": [],
        "graph_seed_bindings": [],
        "graph_traversal_request": {},
        "graph_paths": [],
        "pending_tools": [],
        "adaptive_graph_requested": False,
        "adaptive_round": 0,
        "adaptive_decision": {},
        "tool_results": [],
        "evidence_registry": [],
        "context_evidence_ids": [],
        "approved_evidence_ids": [],
        "safety_decision": {},
        "reflection": "",
        "need_more": False,
        "next_action": "done",
        "tool_calls": [],
        "citations": [],
        "evidence_catalog": [],
        "used_evidence_ids": [],
        "iteration": 0,
        "final_answer": "",
        "answer_completion_status": "full",
        "answer_unconfirmed_fields": [],
    }
