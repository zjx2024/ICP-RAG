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
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.agent.db import agent_db
from backend.agent.state import create_initial_state
from backend.agent.graph import get_compiled_graph

logger = logging.getLogger(__name__)
router = APIRouter()


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


CHAT_TOTAL_TIMEOUT_SECONDS = _positive_env_float("ICP_AGENT_TOTAL_TIMEOUT_SECONDS", 240.0)
SSE_HEARTBEAT_SECONDS = _positive_env_float("ICP_AGENT_SSE_HEARTBEAT_SECONDS", 12.0)
# Answer 最多使用 25 条提示证据；工具事件最多公开另外 75 个稳定 ID，
# 从而都能落入 answer.py 的 100 条受限 evidence_catalog。
PROCESS_EVIDENCE_EVENT_LIMIT = 75


class ChatRequest(BaseModel):
    agent_id: str
    session_id: str
    query: str = Field(min_length=1, max_length=12000)
    user_id: str = "default"
    chat_model_id: str = ""
    embedding_model_id: str = ""
    rerank_model_id: str = ""

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("问题不能为空")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in normalized):
            raise ValueError("问题包含不支持的控制字符")
        return normalized


# 运行中的任务（用于停止生成）
_running_tasks: dict = {}
_stop_requests: set[str] = set()
_active_sessions: set[str] = set()


def _sse_event(data: dict) -> str:
    """构造 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class _PublicStreamQueue:
    """Filter model-private token events before they enter the client queue."""

    def __init__(self, target: asyncio.Queue):
        self._target = target

    async def put(self, item):
        if isinstance(item, dict) and item.get("type") == "thinking_delta":
            return
        await self._target.put(item)


async def _stream_graph_events(graph, state, thread_id: str, agent_id: str,
                               stream_queue: asyncio.Queue, graph_config: dict):
    """运行图并将节点级事件推送到 stream_queue。

    节点内部的 token 级事件会直接写入 queue。浏览器仅接收整理后的
    answer_delta；thinking_delta 属于模型逐字思考，由队列代理在生产端丢弃。
    stream_queue 通过 config 传递（不放入 state，避免 LangGraph checkpoint 序列化失败）。
    """
    config = graph_config

    configurable = config.setdefault("configurable", {})
    # Nodes receive the proxy rather than the browser-facing queue.  This is a
    # producer-side boundary, so a fast model cannot leak thinking tokens before
    # the SSE consumer has a chance to inspect them.
    configurable["stream_queue"] = _PublicStreamQueue(stream_queue)
    shared_process_ids = configurable.setdefault("process_evidence_ids", [])
    normalized_shared = list(dict.fromkeys(
        str(value) for value in shared_process_ids if str(value)
    ))[:PROCESS_EVIDENCE_EVENT_LIMIT]
    shared_process_ids[:] = normalized_shared
    exposed_process_evidence_ids: set[str] = set(normalized_shared)

    def expose_process_ids(values) -> list[str]:
        exposed: list[str] = []
        for raw_value in values or []:
            evidence_id = str(raw_value or "").strip()
            if not evidence_id or evidence_id in exposed:
                continue
            if evidence_id in exposed_process_evidence_ids:
                exposed.append(evidence_id)
                continue
            if len(exposed_process_evidence_ids) >= PROCESS_EVIDENCE_EVENT_LIMIT:
                continue
            exposed_process_evidence_ids.add(evidence_id)
            shared_process_ids.append(evidence_id)
            exposed.append(evidence_id)
        return exposed

    try:
        iterator = graph.astream(state, config=config, stream_mode="updates").__aiter__()
        deadline = asyncio.get_running_loop().time() + CHAT_TOTAL_TIMEOUT_SECONDS
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue

                if node_name == "scope_router":
                    route = str(update.get("scope_route") or "clarification")
                    public_summary = {
                        "knowledge_base": "属于当前工程知识范围，继续分析具体信息需求。",
                        "agent_capability": "这是智能体能力或使用方式问题，无需查询工程资料。",
                        "out_of_scope": "这个问题不在当前工程知识范围内，不调用知识工具。",
                        "clarification": "需要先确认一项必要信息，再决定是否查询工程资料。",
                    }.get(route, "需要先确认问题范围。")
                    # 只公开稳定路径和产品级摘要；模型理由、置信度和提示词
                    # 留在服务端状态中，不作为“思考过程”发送。
                    await stream_queue.put({
                        "type": "stage",
                        "stage": "scope_routing",
                        "status": "completed",
                        "route": route,
                        "summary": public_summary,
                    })

                elif node_name == "plan":
                    await stream_queue.put({
                        "type": "plan",
                        "intent": update.get("intent", ""),
                        "query_targets": update.get("query_targets", []),
                        "calculation_requests": update.get("calculation_requests", []),
                        "relation_expansion": update.get("relation_expansion", {}),
                        "answer_mode": update.get("answer_mode", ""),
                        "rewritten_query": update.get("rewritten_query", ""),
                        "entity_keywords": update.get("entity_keywords", []),
                        "type_keywords": update.get("type_keywords", []),
                        "query_frame": update.get("query_frame", {}),
                        "tool_queries": update.get("tool_queries", {}),
                        "tools_plan": update.get("tools_plan", []),
                    })
                    if str(update.get("scope_route") or "") == "clarification":
                        # 缺失输入引导与指代不明共用 clarification 通道，但摘要
                        # 必须如实区分，避免把“说明还缺什么信息”误报为对象不明。
                        clarification_summary = (
                            "上一轮查询未满足输入要求，本轮直接说明还需要提供的最少信息。"
                            if str(update.get("clarification_kind") or "") == "missing_input"
                            else "上下文中的对象仍不明确，本轮不执行工程资料宽检索。"
                        )
                        await stream_queue.put({
                            "type": "stage",
                            "stage": "scope_routing",
                            "status": "completed",
                            "route": "clarification",
                            "summary": clarification_summary,
                        })

                elif node_name in (
                    "vector_search", "kg_search", "sql_query",
                    "icp_diagnosis", "adaptive_retrieval", "graph_traversal",
                    "icp_calculation",
                ):
                    tool_calls = update.get("tool_calls", [])
                    result_queues: dict[str, list[dict]] = {}
                    for result in update.get("tool_results", []) or []:
                        if not isinstance(result, dict):
                            continue
                        result_queues.setdefault(str(result.get("tool") or ""), []).append(result)
                    for tc in tool_calls:
                        tool_name = str(tc.get("tool") or "")
                        matching_results = result_queues.get(tool_name) or []
                        tool_result = matching_results.pop(0) if matching_results else {}
                        result_evidence_ids = list(dict.fromkeys(
                            str(value)
                            for value in (tool_result.get("evidence_ids") or [])
                            if str(value)
                        ))
                        # Keep the visible relevance order, then append every
                        # ToolResult ID so a compact preview never hides evidence
                        # that the answer may later adopt.
                        call_evidence_ids = list(dict.fromkeys(
                            str(value)
                            for value in (tc.get("evidence_ids") or [])
                            if str(value)
                        ))
                        raw_evidence_ids = list(dict.fromkeys([
                            *call_evidence_ids,
                            *result_evidence_ids,
                        ]))
                        exposed_ids = expose_process_ids(raw_evidence_ids)
                        evidence_total = max(
                            len(raw_evidence_ids),
                            int(tc.get("evidence_count", 0) or 0),
                        )
                        await stream_queue.put({
                            "type": "tool_call",
                            **tc,
                            "status": tool_result.get("status", tc.get("status", "ok")),
                            "evidence_ids": exposed_ids,
                            "evidence_count": evidence_total,
                            "evidence_ids_total": evidence_total,
                            "evidence_ids_truncated": bool(
                                tc.get("evidence_ids_truncated")
                                or len(exposed_ids) < len(raw_evidence_ids)
                                or len(raw_evidence_ids) < len(result_evidence_ids)
                            ),
                        })

                elif node_name == "adaptive_tools":
                    decision = update.get("adaptive_decision") or {}
                    await stream_queue.put({
                        "type": "reflect",
                        "need_more": bool(update.get("need_more")),
                        "next_action": update.get("next_action", "done"),
                        "tried_tools": list(decision.get("executed_tools") or []),
                        "selected_tools": list(decision.get("next_tools") or []),
                        "result_summary": update.get("reflection", ""),
                    })

                elif node_name == "evidence_gate":
                    decision = update.get("safety_decision") or {}
                    approved = list(decision.get("approved_evidence_ids") or [])
                    blocked = list(decision.get("blocked_evidence_ids") or [])
                    reviewed = list(dict.fromkeys(
                        str(value)
                        for value in [
                            *(decision.get("context_evidence_ids") or []),
                            *approved,
                            *blocked,
                        ]
                        if str(value)
                    ))
                    approved = list(dict.fromkeys(str(value) for value in approved if str(value)))
                    blocked = list(dict.fromkeys(str(value) for value in blocked if str(value)))
                    exposed_reviewed = expose_process_ids(reviewed)
                    admitted = exposed_process_evidence_ids
                    exposed_approved = [value for value in approved if value in admitted]
                    exposed_blocked = [value for value in blocked if value in admitted]
                    decision_status = str(decision.get("status") or "answer")
                    await stream_queue.put({
                        "type": "stage",
                        "stage": "evidence_review",
                        "status": (
                            "blocked"
                            if decision_status == "blocked"
                            else "partial" if decision_status == "clarify" else "completed"
                        ),
                        "metrics": {
                            "reviewed_count": len(reviewed),
                            "approved_count": len(set(approved)),
                            "excluded_count": len(set(blocked)),
                            "requires_clarification": bool(
                                decision.get("requires_clarification")
                            ),
                        },
                        "reviewed_evidence_ids": exposed_reviewed,
                        "reviewed_evidence_total": len(reviewed),
                        "reviewed_evidence_ids_truncated": len(exposed_reviewed) < len(reviewed),
                        "approved_evidence_ids": exposed_approved,
                        "approved_evidence_total": len(approved),
                        "approved_evidence_ids_truncated": len(exposed_approved) < len(approved),
                        "blocked_evidence_ids": exposed_blocked,
                        "blocked_evidence_total": len(blocked),
                        "blocked_evidence_ids_truncated": len(exposed_blocked) < len(blocked),
                    })

                elif node_name == "reflect":
                    await stream_queue.put({
                        "type": "reflect",
                        "need_more": update.get("need_more", False),
                        "next_action": update.get("next_action", "done"),
                        "reflection": update.get("reflection", ""),
                        "result_summary": update.get("result_summary", ""),
                        "tried_tools": update.get("tried_tools", []),
                    })

                elif node_name in {"answer", "direct_response"}:
                    final_answer = update.get("final_answer", "")
                    citations = update.get("citations", [])
                    evidence_catalog = update.get("evidence_catalog", [])
                    used_evidence_ids = update.get("used_evidence_ids", [])
                    # Answer 与 citations 原子入队；取消时不会出现已保存答案缺引用。
                    # completion_status 用于区分“部分字段未确认”的正常答案与系统错误，
                    # 前端据此展示提示而不是把降级答案当成失败。
                    await stream_queue.put({
                        "type": "answer",
                        "content": final_answer,
                        "citations": citations,
                        "evidence_catalog": evidence_catalog,
                        "used_evidence_ids": used_evidence_ids,
                        "completion_status": update.get("answer_completion_status", "full"),
                        "unconfirmed_fields": update.get("answer_unconfirmed_fields", []),
                    })

    except asyncio.CancelledError:
        logger.info(f"会话 {thread_id} 生成被取消")
        raise
    except asyncio.TimeoutError:
        logger.warning("会话 %s 超过 %.1fs 总预算", thread_id, CHAT_TOTAL_TIMEOUT_SECONDS)
        await stream_queue.put({
            "type": "error",
            "message": (
                f"本轮处理超过 {CHAT_TOTAL_TIMEOUT_SECONDS:g} 秒总时间预算，"
                "未生成经过大模型整理的答案，请稍后重试或缩小问题范围。"
            ),
            "terminal_reason": "total_timeout",
        })
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

    if body.session_id in _active_sessions:
        raise HTTPException(409, "该会话已有诊断任务正在运行，请先停止或等待完成")
    _active_sessions.add(body.session_id)
    try:
        # 保存用户消息
        agent_db.add_message(body.session_id, "user", body.query)

        # 取历史消息作为上下文
        history = agent_db.list_messages_for_state(body.session_id, limit=20)

        # 构造初始状态（注入 agent 配置的参数：模型采样参数 + 各工具检索参数）
        state = create_initial_state(
            query=body.query,
            agent_id=body.agent_id,
            agent_name=str(agent.get("name") or "ICP故障诊断智能体"),
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
    except BaseException:
        _active_sessions.discard(body.session_id)
        raise

    # 保存助手消息的回调（带 citations 和 thinking_events）
    async def save_assistant_message(
        final_answer: str,
        citations: list = None,
        thinking_events: list = None,
        evidence_catalog: list = None,
    ):
        return agent_db.add_message(
            body.session_id, "assistant", final_answer,
            citations=citations,
            thinking_events=thinking_events,
            evidence_catalog=evidence_catalog,
        )

    # 只持久化面向用户的可审计步骤。thinking_delta 等模型原始输出不落库、
    # 不展示；stage 保存证据数量和稳定 ID，供界面定位证据目录，但不暴露
    # 内部门禁推理或模型逐字思维链。
    THINKING_EVENT_TYPES = {"plan", "tool_call", "reflect", "stage"}

    async def event_stream():
        final_answer = ""
        citations = []
        evidence_catalog = []
        used_evidence_ids = []
        pending_answer_event = None
        pending_answer_delta = ""
        thinking_events = []
        terminal_error = False
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
        _stop_requests.discard(body.session_id)
        _running_tasks[body.session_id] = graph_task

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        stream_queue.get(),
                        timeout=SSE_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    yield ": heartbeat\n\n"
                    continue
                # None 哨兵：图运行结束
                if item is None:
                    break

                evt_type = item.get("type")

                # 模型逐 token 的思考过程既不是可审计结论，也可能暴露
                # 内部提示词。只保留 plan/tool/stage/reflect 的结构化摘要。
                if evt_type == "thinking_delta":
                    continue

                # 捕获最终答案和思考事件用于持久化
                if evt_type == "answer":
                    final_answer = item.get("content", "")
                    citations = item.get("citations", []) or []
                    evidence_catalog = item.get("evidence_catalog", []) or []
                    used_evidence_ids = item.get("used_evidence_ids", []) or []
                    pending_answer_event = item
                    continue
                elif evt_type == "answer_delta":
                    # answer_node 已缓冲完整答案；继续在服务端缓存，直到
                    # 助手消息和证据原子落库成功后再交给浏览器。
                    pending_answer_delta += str(item.get("content") or "")
                    continue
                elif evt_type == "citations":
                    citations = item.get("items", []) or []
                    continue
                elif evt_type in THINKING_EVENT_TYPES:
                    thinking_events.append(item)
                elif evt_type == "error":
                    terminal_error = True

                # 实时 yield 面向用户的结构化事件和最终 answer_delta。
                yield _sse_event(item)

            # 等待图任务完成
            await graph_task
            if terminal_error:
                return

            # 保存助手消息（带 citations 和 thinking_events）
            saved_msg = None
            if final_answer:
                try:
                    saved_msg = await save_assistant_message(
                        final_answer,
                        citations=citations if citations else None,
                        thinking_events=thinking_events if thinking_events else None,
                        evidence_catalog=evidence_catalog if evidence_catalog else None,
                    )
                    if not isinstance(saved_msg, dict) or not saved_msg.get("id"):
                        raise RuntimeError("assistant message insert returned no message id")
                except Exception as exc:
                    logger.error("助手消息持久化失败，会话 %s: %s", body.session_id, exc)
                    yield _sse_event({
                        "type": "error",
                        "message": "回答保存失败，本次结果未发送；请稍后重试。",
                        "terminal_reason": "message_persistence_failed",
                    })
                    return

            # 抽取长期记忆（对话已完整结束，用最终答案可靠保存；智能体关闭了长期记忆则跳过）
            if final_answer and state.get("memory_enabled", True):
                try:
                    from backend.agent.memory import extract_and_save_memory
                    mem_state = dict(state)
                    mem_state["final_answer"] = final_answer
                    await extract_and_save_memory(mem_state, graph_config)
                except Exception as e:
                    logger.warning(f"抽取长期记忆失败: {e}")

            # 持久化成功后再公开终态答案，保证用户看到的内容刷新后必然可恢复。
            if pending_answer_delta:
                yield _sse_event({"type": "answer_delta", "content": pending_answer_delta})
            if pending_answer_event is not None:
                yield _sse_event(pending_answer_event)
                if citations:
                    yield _sse_event({"type": "citations", "items": citations})

            done_payload = {"type": "done", "session_id": body.session_id}
            if saved_msg:
                done_payload["message_id"] = saved_msg.get("id")
                done_payload["thinking_events"] = thinking_events
                done_payload["citations"] = citations
                done_payload["evidence_catalog"] = evidence_catalog
                done_payload["used_evidence_ids"] = used_evidence_ids
            yield _sse_event(done_payload)
        except asyncio.CancelledError:
            explicit_stop = body.session_id in _stop_requests
            if not graph_task.done():
                graph_task.cancel()
            await asyncio.gather(graph_task, return_exceptions=True)
            stopped_payload = {"type": "stopped", "session_id": body.session_id}
            if explicit_stop:
                yield _sse_event(stopped_payload)
                return
            raise
        finally:
            if _running_tasks.get(body.session_id) is graph_task:
                _running_tasks.pop(body.session_id, None)
            _stop_requests.discard(body.session_id)
            _active_sessions.discard(body.session_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/stop")
async def stop_chat(session_id: str):
    """取消当前会话的图任务，并让 Ollama 流式请求收到协程取消。"""
    task = _running_tasks.get(session_id)
    if task is None or task.done():
        return {"stopped": False, "session_id": session_id, "reason": "no_active_generation"}
    _stop_requests.add(session_id)
    task.cancel()
    # 让取消先传播到图节点/HTTP 流；最终清理由 event_stream 的 finally 完成。
    await asyncio.sleep(0)
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
