"""三层记忆：短期(Checkpoint自动) + 摘要(超阈值压缩) + 长期(Store)。

设计原则：
- 短期记忆由 LangGraph Checkpoint 自动管理，无需代码
- 摘要记忆：对话超过阈值时用 LLM 压缩旧消息
- 长期记忆：使用 LangGraph Store（PostgresStore）跨会话存储用户偏好/关键事实
- 只读调用现有 llm_service（不修改）
"""
import logging
from typing import Dict, List, Optional

from backend.core.llm_service import llm_service  # 只读引用，不修改
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.checkpoint import get_store

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """请将以下对话历史压缩成简洁的摘要，保留关键事实、用户偏好和重要结论。用中文输出，不超过300字。

对话历史：
{history}

摘要："""

EXTRACT_MEMORY_PROMPT = """从以下对话中提取值得长期记住的用户偏好或关键事实（如用户身份、常用知识库、专业领域、偏好语言等）。
如果没有值得记忆的内容，返回空 JSON。

对话：
用户: {query}
助手: {answer}

输出 JSON 格式：
{{
  "memories": [
    {{"key": "简短标识", "value": "记忆内容"}}
  ]
}}
"""


async def maybe_summarize(state, summary_threshold: int = 12, keep_recent: int = 6,
                          agent_id: str = None) -> dict:
    """超阈值自动摘要压缩上下文

    Args:
        state: AgentState
        summary_threshold: 触发摘要的消息数阈值
        keep_recent: 保留最近的消息数
        agent_id: 用于取摘要模型
    """
    msgs = state.get("messages", [])
    # 每条消息算1条，阈值按消息条数
    if len(msgs) < summary_threshold * 2:
        return {}

    to_summarize = msgs[:-keep_recent * 2] if keep_recent * 2 < len(msgs) else msgs
    recent = msgs[-keep_recent * 2:] if keep_recent * 2 < len(msgs) else []

    # 构造摘要输入
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
        for m in to_summarize
    )

    # 取摘要模型（回退到默认 LLM）
    model = None
    if agent_id:
        model = agent_model_role_service.get_model_by_role(agent_id, "reflect")
    try:
        summary = llm_service.call_model(
            [{"role": "user", "content": SUMMARY_PROMPT.format(history=history_text)}],
            model
        )
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}")
        return {}

    summary_msg = {"role": "system", "content": f"[历史摘要] {summary}"}
    return {"messages": [summary_msg] + recent} if recent else {"messages": [summary_msg]}


async def inject_memory(state, config, user_id: str = "default") -> dict:
    """从 Store 注入长期记忆到上下文"""
    store = await get_store()
    if not store:
        return {"memory_context": ""}

    try:
        namespace = (user_id, "profile")
        items = await store.asearch(namespace, limit=5)
        if not items:
            return {"memory_context": ""}

        memory_parts = []
        for item in items:
            val = item.value if hasattr(item, "value") else item.get("value", "")
            memory_parts.append(f"- {val}")
        return {"memory_context": "\n".join(memory_parts)}
    except Exception as e:
        logger.warning(f"注入长期记忆失败: {e}")
        return {"memory_context": ""}


async def extract_and_save_memory(state, config, user_id: str = "default") -> None:
    """异步抽取长期记忆并保存到 Store（不阻塞返回）"""
    store = await get_store()
    if not store:
        return

    query = state.get("query", "")
    answer = state.get("final_answer", "")
    if not query or not answer:
        return

    agent_id = state.get("agent_id", "")
    model = None
    if agent_id:
        model = agent_model_role_service.get_model_by_role(agent_id, "reflect")

    try:
        response = llm_service.call_model(
            [{"role": "user", "content": EXTRACT_MEMORY_PROMPT.format(query=query, answer=answer)}],
            model
        )
        import json
        import re
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        memories = data.get("memories", [])
        if not memories:
            return

        namespace = (user_id, "profile")
        for m in memories:
            key = m.get("key", "")
            value = m.get("value", "")
            if key and value:
                import time
                await store.aput(namespace, f"{key}_{int(time.time() * 1000)}", {"key": key, "value": value})
    except Exception as e:
        logger.warning(f"抽取长期记忆失败: {e}")


async def list_memories(user_id: str = "default", limit: int = 50) -> List[Dict]:
    """列出长期记忆（供 API 调用）"""
    store = await get_store()
    if not store:
        return []
    try:
        namespace = (user_id, "profile")
        items = await store.asearch(namespace, limit=limit)
        result = []
        for item in items:
            store_key = item.key if hasattr(item, "key") else item.get("key", "")
            # item.value 是保存时的 dict {"key": ..., "value": ...}，需取出字符串字段
            value_data = item.value if hasattr(item, "value") else item.get("value", {})
            if isinstance(value_data, dict):
                display_key = value_data.get("key", "")
                display_value = value_data.get("value", "")
            else:
                display_key = value_data if isinstance(value_data, str) else ""
                display_value = value_data if isinstance(value_data, str) else ""
            result.append({
                "key": display_key,
                "value": display_value,
                "store_key": store_key,
            })
        return result
    except Exception as e:
        logger.warning(f"列出长期记忆失败: {e}")
        return []


async def delete_memory(user_id: str, key: str) -> bool:
    """删除长期记忆"""
    store = await get_store()
    if not store:
        return False
    try:
        namespace = (user_id, "profile")
        await store.adelete(namespace, key)
        return True
    except Exception as e:
        logger.warning(f"删除长期记忆失败: {e}")
        return False


async def clear_memories(user_id: str = "default") -> bool:
    """清空长期记忆"""
    store = await get_store()
    if not store:
        return False
    try:
        namespace = (user_id, "profile")
        items = await store.asearch(namespace, limit=1000)
        for item in items:
            k = item.key if hasattr(item, "key") else item.get("key", "")
            if k:
                await store.adelete(namespace, k)
        return True
    except Exception as e:
        logger.warning(f"清空长期记忆失败: {e}")
        return False
