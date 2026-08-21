"""会话管理 API。"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agent.db import agent_db

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionCreate(BaseModel):
    agent_id: str
    user_id: str = "default"
    title: str = "新会话"


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


@router.post("/session")
def create_session(body: SessionCreate):
    """新建会话（id 即 LangGraph thread_id）"""
    if not agent_db.get_agent(body.agent_id):
        raise HTTPException(404, "Agent 不存在")
    return agent_db.create_session(body.agent_id, body.user_id, body.title)


@router.get("/session")
def list_sessions(agent_id: Optional[str] = None, user_id: Optional[str] = None,
                  archived: bool = False, q: Optional[str] = None, limit: int = 100):
    """会话列表"""
    return agent_db.list_sessions(agent_id=agent_id, user_id=user_id, archived=archived,
                                  search=q, limit=limit)


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """会话详情（含消息历史）"""
    session = agent_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    session["messages"] = agent_db.list_messages(session_id)
    return session


@router.put("/session/{session_id}")
def update_session(session_id: str, body: SessionUpdate):
    """更新会话（重命名/置顶/归档）"""
    data = body.dict(exclude_none=True)
    if not data:
        raise HTTPException(400, "无更新字段")
    session = agent_db.update_session(session_id, **data)
    if not session:
        raise HTTPException(404, "会话不存在")
    return session


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """删除会话（级联清理 messages，Checkpoint 自动隔离）"""
    ok = agent_db.delete_session(session_id)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"deleted": True}


@router.post("/session/{session_id}/title")
def generate_title(session_id: str):
    """根据首条消息生成会话标题"""
    session = agent_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    messages = agent_db.list_messages(session_id, limit=2)
    if not messages:
        return session

    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if not first_user:
        return session

    content = (first_user.get("content") or "")[:50]
    title = content if content else "新会话"
    updated = agent_db.update_session(session_id, title=title)
    return updated or session
