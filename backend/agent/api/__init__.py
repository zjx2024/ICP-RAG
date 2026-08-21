"""Agent API 路由聚合。"""
from fastapi import APIRouter

from backend.agent.api.agent_router import router as agent_router
from backend.agent.api.session_router import router as session_router
from backend.agent.api.chat_router import router as chat_router
from backend.agent.api.memory_router import router as memory_router

# 聚合路由，由 main.py 挂载到 /api 前缀
router = APIRouter()
router.include_router(agent_router, tags=["Agent 配置"])
router.include_router(session_router, tags=["会话管理"])
router.include_router(chat_router, tags=["对话"])
router.include_router(memory_router, tags=["长期记忆"])
