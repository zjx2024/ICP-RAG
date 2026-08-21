"""长期记忆管理 API。"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agent.memory import list_memories, delete_memory, clear_memories

logger = logging.getLogger(__name__)
router = APIRouter()


class MemoryUpdate(BaseModel):
    value: str


@router.get("/memory")
async def get_memories(user_id: str = "default"):
    """当前用户长期记忆列表"""
    return await list_memories(user_id=user_id)


@router.delete("/memory")
async def clear_all_memories(user_id: str = "default"):
    """清空记忆"""
    ok = await clear_memories(user_id=user_id)
    if not ok:
        raise HTTPException(500, "清空失败（Store 未初始化或出错）")
    return {"cleared": True}


@router.delete("/memory/{key}")
async def delete_one_memory(key: str, user_id: str = "default"):
    """删除单条记忆"""
    ok = await delete_memory(user_id, key)
    if not ok:
        raise HTTPException(404, "记忆不存在或 Store 未初始化")
    return {"deleted": True, "key": key}
