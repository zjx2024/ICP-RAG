"""ICP 故障诊断 Agent 配置与模型角色绑定 API。"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
    kb_ids: Optional[List[str]] = None
    enabled_tools: List[str] = Field(default_factory=lambda: ["icp_diagnosis"])
    chat_model_id: Optional[str] = None
    embedding_model_id: Optional[str] = None
    rerank_model_id: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 4096
    max_reflections: int = 1
    retrieval_top_k: int = 12
    summary_threshold: int = 20
    keep_recent_turns: int = 8
    streaming: bool = True
    memory_enabled: bool = False
    kg_top_k: int = 12
    sql_limit: int = 12


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    kb_id: Optional[str] = None
    kb_ids: Optional[List[str]] = None
    enabled_tools: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_reflections: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    summary_threshold: Optional[int] = None
    keep_recent_turns: Optional[int] = None
    streaming: Optional[bool] = None
    memory_enabled: Optional[bool] = None
    is_active: Optional[bool] = None
    kg_top_k: Optional[int] = None
    sql_limit: Optional[int] = None


class ModelBind(BaseModel):
    role: str
    model_id: str
    is_default: bool = False


@router.post("/agent")
def create_agent(body: AgentCreate):
    try:
        agent = agent_db.create_agent(
            name=body.name,
            description=body.description,
            system_prompt=body.system_prompt,
            kb_id=body.kb_id,
            kb_ids=body.kb_ids,
            enabled_tools=body.enabled_tools,
            chat_model_id=body.chat_model_id,
            embedding_model_id=body.embedding_model_id,
            rerank_model_id=body.rerank_model_id,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            max_reflections=body.max_reflections,
            retrieval_top_k=body.retrieval_top_k,
            summary_threshold=body.summary_threshold,
            keep_recent_turns=body.keep_recent_turns,
            streaming=body.streaming,
            memory_enabled=body.memory_enabled,
            kg_top_k=body.kg_top_k,
            sql_limit=body.sql_limit,
        )
        return agent
    except Exception as exc:
        logger.error("创建 Agent 失败: %s", exc)
        raise HTTPException(500, f"创建失败: {exc}") from exc


@router.get("/agent")
def list_agents(active_only: bool = False):
    return agent_db.list_agents(active_only=active_only)


@router.get("/agent/tools")
def list_tools():
    return get_available_tools()


@router.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    agent = agent_db.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent 不存在")
    agent["model_roles"] = agent_db.list_agent_models(agent_id)
    return agent


@router.put("/agent/{agent_id}")
def update_agent(agent_id: str, body: AgentUpdate):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "无更新字段")
    agent = agent_db.update_agent(agent_id, **data)
    if not agent:
        raise HTTPException(404, "Agent 不存在")
    clear_graph_cache(agent_id)
    return agent


@router.delete("/agent/{agent_id}")
def delete_agent(agent_id: str):
    ok = agent_db.delete_agent(agent_id)
    if not ok:
        raise HTTPException(404, "Agent 不存在")
    clear_graph_cache(agent_id)
    return {"deleted": True}


@router.post("/agent/{agent_id}/models")
def bind_model(agent_id: str, body: ModelBind):
    if not agent_db.get_agent(agent_id):
        raise HTTPException(404, "Agent 不存在")
    if body.role not in ("plan", "reflect", "answer", "embedding", "rerank"):
        raise HTTPException(400, "role 必须是 plan/reflect/answer/embedding/rerank")
    return agent_db.bind_model(agent_id, body.role, body.model_id, body.is_default)


@router.get("/agent/{agent_id}/models")
def list_agent_models(agent_id: str):
    if not agent_db.get_agent(agent_id):
        raise HTTPException(404, "Agent 不存在")
    return agent_db.list_agent_models(agent_id)


@router.delete("/agent/{agent_id}/models/{role}")
def unbind_model(agent_id: str, role: str):
    ok = agent_db.unbind_model(agent_id, role)
    if not ok:
        raise HTTPException(404, "未找到该角色绑定")
    return {"unbound": True}
