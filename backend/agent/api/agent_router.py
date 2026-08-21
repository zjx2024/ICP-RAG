"""Agent 配置 + 模型角色绑定 API。"""
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agent.db import agent_db
from backend.agent.tools.registry import get_available_tools
from backend.agent.graph import clear_graph_cache

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    kb_id: Optional[str] = None
    kb_ids: Optional[List[str]] = None  # 关联多个知识库（优先于 kb_id）
    enabled_tools: List[str] = ["vector_search", "kg_search", "sql_query"]
    chat_model_id: Optional[str] = None  # 用户选择的对话模型（model_type=llm），为空则自动绑定默认 LLM
    embedding_model_id: Optional[str] = None  # 用户选择的向量模型（model_type=embedding），为空则自动绑定默认 embedding
    rerank_model_id: Optional[str] = None  # 用户选择的重排序模型（model_type=rerank），为空则不启用重排序
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    max_reflections: int = 3
    retrieval_top_k: int = 5
    summary_threshold: int = 12
    keep_recent_turns: int = 6
    streaming: bool = True
    memory_enabled: bool = True  # 长期记忆开关（默认启用）
    kg_top_k: int = 5       # 知识图谱检索返回条数（kg_search 工具）
    sql_limit: int = 5      # 数据库查询返回条数（sql_query 工具）


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    kb_id: Optional[str] = None
    kb_ids: Optional[List[str]] = None  # 关联多个知识库（优先于 kb_id）
    enabled_tools: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_reflections: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    summary_threshold: Optional[int] = None
    keep_recent_turns: Optional[int] = None
    streaming: Optional[bool] = None
    memory_enabled: Optional[bool] = None  # 长期记忆开关
    is_active: Optional[bool] = None
    kg_top_k: Optional[int] = None
    sql_limit: Optional[int] = None


class ModelBind(BaseModel):
    role: str  # plan | reflect | answer | embedding | rerank
    model_id: str
    is_default: bool = False


@router.post("/agent")
def create_agent(body: AgentCreate):
    """创建 Agent"""
    try:
        agent = agent_db.create_agent(
            name=body.name, description=body.description, system_prompt=body.system_prompt,
            kb_id=body.kb_id, kb_ids=body.kb_ids, enabled_tools=body.enabled_tools,
            chat_model_id=body.chat_model_id, embedding_model_id=body.embedding_model_id,
            rerank_model_id=body.rerank_model_id,
            temperature=body.temperature, top_p=body.top_p, max_tokens=body.max_tokens,
            max_reflections=body.max_reflections, retrieval_top_k=body.retrieval_top_k,
            summary_threshold=body.summary_threshold, keep_recent_turns=body.keep_recent_turns,
            streaming=body.streaming, memory_enabled=body.memory_enabled,
            kg_top_k=body.kg_top_k, sql_limit=body.sql_limit,
        )
        return agent
    except Exception as e:
        logger.error(f"创建 Agent 失败: {e}")
        raise HTTPException(500, f"创建失败: {e}")


@router.get("/agent")
def list_agents(active_only: bool = False):
    """Agent 列表"""
    return agent_db.list_agents(active_only=active_only)


@router.get("/agent/tools")
def list_tools():
    """列出可用工具"""
    return get_available_tools()


@router.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    """Agent 详情（含角色模型映射）"""
    agent = agent_db.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent 不存在")
    agent["model_roles"] = agent_db.list_agent_models(agent_id)
    return agent


@router.put("/agent/{agent_id}")
def update_agent(agent_id: str, body: AgentUpdate):
    """更新 Agent"""
    data = body.dict(exclude_none=True)
    if not data:
        raise HTTPException(400, "无更新字段")
    agent = agent_db.update_agent(agent_id, **data)
    if not agent:
        raise HTTPException(404, "Agent 不存在")
    clear_graph_cache(agent_id)
    return agent


@router.delete("/agent/{agent_id}")
def delete_agent(agent_id: str):
    """删除 Agent"""
    ok = agent_db.delete_agent(agent_id)
    if not ok:
        raise HTTPException(404, "Agent 不存在")
    clear_graph_cache(agent_id)
    return {"deleted": True}


# ── 模型角色绑定（引用现有 llm_models，不修改原表）──
@router.post("/agent/{agent_id}/models")
def bind_model(agent_id: str, body: ModelBind):
    """为 Agent 绑定角色模型（model_id 引用现有 llm_models）"""
    if not agent_db.get_agent(agent_id):
        raise HTTPException(404, "Agent 不存在")
    if body.role not in ("plan", "reflect", "answer", "embedding", "rerank"):
        raise HTTPException(400, "role 必须是 plan/reflect/answer/embedding/rerank")
    return agent_db.bind_model(agent_id, body.role, body.model_id, body.is_default)


@router.get("/agent/{agent_id}/models")
def list_agent_models(agent_id: str):
    """查询 Agent 的角色模型映射"""
    if not agent_db.get_agent(agent_id):
        raise HTTPException(404, "Agent 不存在")
    return agent_db.list_agent_models(agent_id)


@router.delete("/agent/{agent_id}/models/{role}")
def unbind_model(agent_id: str, role: str):
    """解绑角色模型"""
    ok = agent_db.unbind_model(agent_id, role)
    if not ok:
        raise HTTPException(404, "未找到该角色绑定")
    return {"unbound": True}
