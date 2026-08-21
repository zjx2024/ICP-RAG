"""对话 API（SSE 流式 + 普通）。

设计原则：
- 复用 LangGraph 编译后的图，通过 thread_id（=session_id）自动恢复上下文
- SSE 流式输出 Plan/工具调用/反思/答案等事件
- 支持停止生成、重新生成
"""
import json
import uuid
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.db import agent_db
from backend.agent.state import create_initial_state
from backend.agent.graph import get_compiled_graph

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    agent_id: str
    session_id: str
    query: str
    user_id: str = "default"
    chat_model_id: str = ""
    embedding_model_id: str = ""
    rerank_model_id: str = ""


# 运行中的任务（用于停止生成）
_running_tasks: dict = {}


def _sse_event(data: dict) -> str:
    """构造 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_graph_events(graph, state, thread_id: str, agent_id: str,
                               stream_queue: asyncio.Queue, graph_config: dict):
    """运行图并将节点级事件推送到 stream_queue。

    节点内部的 token 级事件（thinking_delta / answer_delta）由节点自身直接推送到 queue，
    与此处的节点级事件（plan / tool_call / reflect / answer）交替到达，实现实时流式。
    stream_queue 通过 config 传递（不放入 state，避免 LangGraph checkpoint 序列化失败）。
    """
    config = graph_config

    try:
        async for event in graph.astream(state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue

                if node_name == "plan":
                    await stream_queue.put({
                        "type": "plan",
                        "intent": update.get("intent", ""),
                        "rewritten_query": update.get("rewritten_query", ""),
                        "entity_keywords": update.get("entity_keywords", []),
                        "type_keywords": update.get("type_keywords", []),
                        "tools_plan": update.get("tools_plan", []),
                        "reasoning": update.get("plan_reasoning", ""),
                    })

                elif node_name in ("vector_search", "kg_search", "sql_query"):
                    tool_calls = update.get("tool_calls", [])
                    for tc in tool_calls:
                        await stream_queue.put({"type": "tool_call", **tc})

                elif node_name == "reflect":
                    await stream_queue.put({
                        "type": "reflect",
                        "need_more": update.get("need_more", False),
                        "next_action": update.get("next_action", "done"),
                        "reflection": update.get("reflection", ""),
                        "result_summary": update.get("result_summary", ""),
                        "tried_tools": update.get("tried_tools", []),
                    })

                elif node_name == "answer":
                    final_answer = update.get("final_answer", "")
                    citations = update.get("citations", [])
                    await stream_queue.put({"type": "answer", "content": final_answer})
                    if citations:
                        await stream_queue.put({"type": "citations", "items": citations})

    except asyncio.CancelledError:
        logger.info(f"会话 {thread_id} 生成被取消")
        raise
    except Exception as e:
        logger.error(f"图运行失败: {e}")
        await stream_queue.put({"type": "error", "message": str(e)})
    finally:
        # 哨兵：通知 event_stream 循环图已结束
        await stream_queue.put(None)


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """SSE 流式对话"""
    agent = agent_db.get_agent(body.agent_id)
    if not agent:
        raise HTTPException(404, "Agent 不存在")

    session = agent_db.get_session(body.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    # 保存用户消息
    agent_db.add_message(body.session_id, "user", body.query)

    # 取历史消息作为上下文
    history = agent_db.list_messages_for_state(body.session_id, limit=20)

    # 构造初始状态（注入 agent 配置的参数：模型采样参数 + 各工具检索参数）
    state = create_initial_state(
        query=body.query,
        agent_id=body.agent_id,
        kb_id=agent.get("kb_id", ""),
        kb_ids=agent.get("kb_ids") or [],
        system_prompt=agent.get("system_prompt", ""),
        history=history,
        chat_model_id=body.chat_model_id,
        embedding_model_id=body.embedding_model_id,
        rerank_model_id=body.rerank_model_id,
        temperature=agent.get("temperature", 0.7),
        top_p=agent.get("top_p", 0.9),
        max_tokens=agent.get("max_tokens", 2048),
        max_reflections=agent.get("max_reflections", 3),
        retrieval_top_k=agent.get("retrieval_top_k", 5),
        kg_top_k=agent.get("kg_top_k", 5),
        sql_limit=agent.get("sql_limit", 5),
        summary_threshold=agent.get("summary_threshold", 12),
        keep_recent_turns=agent.get("keep_recent_turns", 6),
        memory_enabled=bool(agent.get("memory_enabled", True)),
    )

    enabled_tools = agent.get("enabled_tools", ["vector_search", "kg_search", "sql_query"])
    graph = await get_compiled_graph(body.agent_id, enabled_tools)

    # 保存助手消息的回调（带 citations 和 thinking_events）
    async def save_assistant_message(final_answer: str, citations: list = None, thinking_events: list = None):
        return agent_db.add_message(
            body.session_id, "assistant", final_answer,
            citations=citations, thinking_events=thinking_events
        )

    # 思考过程事件类型列表（存入 DB 的 thinking_events）
    # 含 thinking_text：plan/reflect 节点流式输出的详细思考原文，否则历史消息只剩粗粒度 plan/tool_call/reflect 骨架
    THINKING_EVENT_TYPES = {"plan", "tool_call", "reflect", "thinking_text"}

    async def event_stream():
        final_answer = ""
        citations = []
        thinking_events = []
        # 创建 queue（通过 config 传递给节点，不放入 state 避免 checkpoint 序列化失败）
        stream_queue = asyncio.Queue()
        # 图运行配置：thread_id=session_id，LangGraph 按 thread 恢复上下文
        graph_config = {
            "configurable": {
                "thread_id": body.session_id,
                "agent_id": body.agent_id,
                "stream_queue": stream_queue,
            }
        }

        # 后台任务：运行图，将节点级事件推送到同一个 queue
        graph_task = asyncio.create_task(
            _stream_graph_events(graph, state, body.session_id, body.agent_id, stream_queue, graph_config)
        )
        _running_tasks[body.session_id] = graph_task

        try:
            while True:
                item = await stream_queue.get()
                # None 哨兵：图运行结束
                if item is None:
                    break

                evt_type = item.get("type")

                # 捕获最终答案和思考事件用于持久化
                if evt_type == "answer":
                    final_answer = item.get("content", "")
                elif evt_type == "citations":
                    citations = item.get("items", []) or []
                elif evt_type in THINKING_EVENT_TYPES:
                    thinking_events.append(item)

                # 实时 yield SSE 事件（包括 thinking_delta / answer_delta）
                yield _sse_event(item)

            # 等待图任务完成
            await graph_task

            # 保存助手消息（带 citations 和 thinking_events）
            saved_msg = None
            if final_answer:
                saved_msg = await save_assistant_message(
                    final_answer,
                    citations=citations if citations else None,
                    thinking_events=thinking_events if thinking_events else None
                )

            # 抽取长期记忆（对话已完整结束，用最终答案可靠保存；智能体关闭了长期记忆则跳过）
            if final_answer and state.get("memory_enabled", True):
                try:
                    from backend.agent.memory import extract_and_save_memory
                    mem_state = dict(state)
                    mem_state["final_answer"] = final_answer
                    await extract_and_save_memory(mem_state, graph_config)
                except Exception as e:
                    logger.warning(f"抽取长期记忆失败: {e}")

            done_payload = {"type": "done", "session_id": body.session_id}
            if saved_msg:
                done_payload["message_id"] = saved_msg.get("id")
                done_payload["thinking_events"] = thinking_events
                done_payload["citations"] = citations
            yield _sse_event(done_payload)
        except asyncio.CancelledError:
            graph_task.cancel()
            try:
                await graph_task
            except Exception:
                pass
            saved_msg = None
            if final_answer:
                saved_msg = await save_assistant_message(
                    final_answer,
                    citations=citations if citations else None,
                    thinking_events=thinking_events if thinking_events else None
                )
            stopped_payload = {"type": "stopped", "session_id": body.session_id}
            if saved_msg:
                stopped_payload["message_id"] = saved_msg.get("id")
                stopped_payload["thinking_events"] = thinking_events
                stopped_payload["citations"] = citations
            yield _sse_event(stopped_payload)
            raise
        finally:
            _running_tasks.pop(body.session_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/stop")
def stop_chat(session_id: str):
    """停止生成（标记取消）"""
    # 简化实现：前端断开 SSE 连接即可触发 CancelledError
    return {"stopped": True, "session_id": session_id}


@router.post("/chat/{session_id}/regenerate")
def regenerate(session_id: str):
    """重新生成最后一条助手消息（删除后重新触发）"""
    session = agent_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    return {"session_id": session_id, "note": "请前端删除最后一条助手消息后重新 POST /chat"}


@router.post("/session/{session_id}/export")
def export_session(session_id: str, fmt: str = "md"):
    """导出对话"""
    session = agent_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    messages = agent_db.list_messages(session_id)
    if fmt == "json":
        return {"session": session, "messages": messages}
    # md 格式
    lines = [f"# {session.get('title', '对话')}\n"]
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"**用户:** {content}\n")
        elif role == "assistant":
            lines.append(f"**助手:** {content}\n")
    return {"format": "md", "content": "\n".join(lines)}


@router.post("/message/{message_id}/feedback")
def message_feedback(message_id: str, feedback: str):
    """消息反馈（like/dislike）"""
    if feedback not in ("like", "dislike"):
        raise HTTPException(400, "feedback 必须是 like 或 dislike")
    ok = agent_db.update_feedback(message_id, feedback)
    if not ok:
        raise HTTPException(404, "消息不存在")
    return {"message_id": message_id, "feedback": feedback}


@router.post("/session/{session_id}/compress")
def compress_session(session_id: str):
    """压缩上下文：用 LLM 摘要旧对话，保留摘要 + 最近几轮"""
    session = agent_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    messages = agent_db.list_messages(session_id, limit=200)
    # 保留最近 N 条消息，摘要其余的（N 读 agent 配置 keep_recent_turns，默认 6）
    agent_cfg = agent_db.get_agent(session.get("agent_id", "")) if session.get("agent_id") else None
    keep_recent = int((agent_cfg or {}).get("keep_recent_turns", 6) or 6)
    if len(messages) <= keep_recent:
        return {"session_id": session_id, "before_tokens": 0, "after_tokens": 0,
                "note": "对话轮数较少，无需压缩"}

    old_msgs = messages[:-keep_recent]
    recent_msgs = messages[-keep_recent:]

    # 计算 before tokens（粗略估算）
    def _est_tokens(text):
        cn = len([c for c in text if '\u4e00' <= c <= '\u9fa5'])
        return int(cn / 1.5 + (len(text) - cn) / 4) if text else 0

    before_tokens = sum(_est_tokens(m.get("content", "")) for m in messages)

    # 用 LLM 生成摘要
    summary_text = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}: {m.get('content','')[:500]}"
        for m in old_msgs
    ])

    try:
        from backend.core.llm_service import llm_service
        summary_prompt = f"请用中文简洁总结以下对话的关键信息（实体、结论、上下文），不超过300字：\n\n{summary_text}"
        summary = llm_service._call_model(
            [{"role": "user", "content": summary_prompt}], None
        )
    except Exception as e:
        logger.warning(f"压缩摘要生成失败，使用简单截断: {e}")
        summary = f"[历史对话摘要生成失败，保留前200字] {summary_text[:200]}"

    # 删除旧消息
    from backend.storage.postgres_client import postgres_client
    old_ids = [m["id"] for m in old_msgs if m.get("id")]
    with postgres_client.conn.cursor() as cur:
        if old_ids:
            cur.execute(
                "DELETE FROM agent_messages WHERE id = ANY(%s)",
                (old_ids,)
            )
        # 在最近消息之前插入摘要
        if recent_msgs:
            first_recent_time = recent_msgs[0].get("created_at")
            msg_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO agent_messages (id, session_id, role, content, created_at)
                VALUES (%s, %s, 'system', %s, %s)
            """, (msg_id, session_id, f"[上下文摘要] {summary}", first_recent_time))
        postgres_client.conn.commit()

    # 清除 LangGraph checkpoint（让 agent 从摘要重新开始）
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (session_id,))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (session_id,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (session_id,))
            postgres_client.conn.commit()
    except Exception as e:
        logger.warning(f"清除 checkpoint 失败（不影响压缩）: {e}")
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass

    after_tokens = _est_tokens(summary) + sum(
        _est_tokens(m.get("content", "")) for m in recent_msgs
    )

    return {
        "session_id": session_id,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "deleted_count": len(old_msgs),
        "summary": summary[:200],
    }
