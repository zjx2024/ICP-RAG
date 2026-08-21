"""Answer 节点：生成 ICP 故障诊断回答并提取证据引用。"""
import json
import logging
from typing import Dict, List

from backend.core.llm_service import llm_service
from backend.agent.model_role_service import agent_model_role_service

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """你是 ICP 航空电子系统故障诊断智能体。请只依据检索证据回答。

系统设定：
{system_prompt}

对话历史：
{history}

用户问题：
{query}

诊断置信度：{confidence}
候选故障：
{candidates}

需要补充的问题：
{clarification}

检索证据：
{context}

回答规则：
1. 用中文回答，不得编造 HMC、原因、影响、维修动作、可靠度或文档来源。
2. 先写“初步判断”，列出最多 3 个候选，并说明每个候选的匹配依据。
3. 分开写：资料明确记载、跨文档关联推断、尚未确认的信息。
4. 有完整 HMC 精确结果时说明是精确代码命中；没有时不得假装唯一定位。
5. 涉及原因时引用 FTA/FMECA；涉及影响和严酷度时引用 FMECA；涉及维修/API 时引用 HMC/MFL；涉及经验时引用故障历史；涉及可靠性时引用 RBD。
6. 置信度不足或候选接近时，最后必须提出“下一步需要确认”的问题，不得直接下唯一结论。
7. 维修内容仅作为文档知识检索结果，需提醒实际处置应以适用构型、有效版本和正式维修程序为准。
8. 引用具体证据后标注 [^编号]，编号对应检索结果顺序。
9. 不需要展示图片画廊；仅当证据本身是 FTA/RBD/架构图时，可以说明存在相关图证据。
10. 回答结构建议：已知现象 → 候选故障 → 原因/影响 → 建议检查 → 下一步确认 → 适用性说明。
"""


def _format_history(messages, max_turns=8):
    if not messages:
        return "无"
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    parts = []
    for message in recent:
        role = message.get("role", "")
        content = (message.get("content") or "")[:600]
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
        elif role == "system":
            parts.append(f"系统: {content}")
    return "\n".join(parts) if parts else "无"


def _format_table_data(meta_data) -> str:
    if not meta_data:
        return ""
    if isinstance(meta_data, str):
        try:
            meta_data = json.loads(meta_data)
        except Exception:
            return meta_data[:1000]
    data = meta_data.get("data", []) if isinstance(meta_data, dict) else meta_data
    if not isinstance(data, list):
        return ""
    lines = []
    for row in data[:20]:
        if isinstance(row, list):
            lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def _format_context(state) -> str:
    diagnostic_context = state.get("diagnostic_context", "")
    if diagnostic_context:
        return diagnostic_context

    parts: List[str] = []
    vector_results = state.get("vector_results", [])
    if vector_results:
        parts.append("【文档检索】")
        for index, item in enumerate(vector_results[:15], start=1):
            parts.append(
                f"[{index}] {item.get('title', '')}（来源: {item.get('doc_name', '')}）\n"
                f"{(item.get('content') or '')[:700]}"
            )

    kg_summary = state.get("kg_summary", "")
    if kg_summary:
        parts.append(f"【知识图谱】\n{kg_summary[:5000]}")

    sql_results = state.get("sql_results", [])
    if sql_results:
        parts.append("【结构化数据库】")
        for index, item in enumerate(sql_results[:10], start=1):
            item_type = item.get("type", "")
            if item_type == "table":
                table_data = _format_table_data(item.get("meta_data"))
                parts.append(
                    f"[{index}] {item.get('title', '')}: {(item.get('description') or '')[:300]}\n{table_data}"
                )
            else:
                parts.append(f"[{index}] {json.dumps(item, ensure_ascii=False, default=str)[:1000]}")
    return "\n\n".join(parts) if parts else "无检索结果"


def _format_candidates(candidates: List[Dict]) -> str:
    if not candidates:
        return "无明确候选"
    lines = []
    for index, item in enumerate(candidates[:8], start=1):
        evidence = ", ".join(
            f"{ev.get('source', '')}:{ev.get('id', '')}" for ev in item.get("evidence", [])[:5]
        )
        lines.append(
            f"{index}. {item.get('fault', '')}；综合得分={item.get('score', 0)}；证据={evidence or '无'}"
        )
    return "\n".join(lines)


def _extract_citations(state) -> list:
    citations = []
    seen = set()
    source_rows = state.get("domain_results", []) or state.get("vector_results", [])
    for item in source_rows[:25]:
        key = (item.get("doc_id"), item.get("chunk_id"), item.get("title"), (item.get("content") or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "id": len(citations) + 1,
                "doc_id": item.get("doc_id", ""),
                "doc_name": item.get("doc_name", ""),
                "title": item.get("title", ""),
                "chunk_id": item.get("chunk_id", ""),
                "content": (item.get("content") or "")[:3500],
                "type": item.get("source_type") or item.get("content_type") or "document",
            }
        )
        if len(citations) >= 20:
            break
    return citations


async def answer_node(state, config=None) -> Dict:
    agent_id = state.get("agent_id", "")
    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "answer") if agent_id else None
    model = agent_model_role_service.apply_agent_llm_params(state, model)

    context = _format_context(state)
    candidates = state.get("candidate_faults", []) or []
    confidence = float(state.get("diagnostic_confidence", 0.0) or 0.0)
    clarification = state.get("clarification_question", "") or "无"

    prompt = ANSWER_PROMPT.format(
        system_prompt=state.get("system_prompt", "你是 ICP 故障诊断智能体。"),
        history=_format_history(state.get("messages", [])),
        query=state.get("query", ""),
        confidence=f"{confidence:.2f}",
        candidates=_format_candidates(candidates),
        clarification=clarification,
        context=context,
    )

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    # ICP 编排器是一个 LangGraph 节点，但内部完成多个领域工具调用。旧 chat_router
    # 只识别三个通用节点，因此在回答前把细粒度调用记录显式推送给前端。
    if stream_queue is not None and state.get("diagnostic_context"):
        for tool_call in state.get("tool_calls", []) or []:
            try:
                await stream_queue.put({"type": "tool_call", **tool_call})
            except Exception:
                pass

    try:
        if stream_queue is not None:
            final_answer = ""
            async for token in llm_service.stream_model([{"role": "user", "content": prompt}], model):
                final_answer += token
                try:
                    await stream_queue.put({"type": "answer_delta", "content": token})
                except Exception:
                    pass
        else:
            final_answer = llm_service._call_model([{"role": "user", "content": prompt}], model)
    except Exception as exc:
        logger.error("Answer node failed: %s", exc)
        final_answer = f"生成回答时出错：{exc}"

    if not final_answer or final_answer.startswith("[LLM未配置]") or final_answer.startswith("[AI调用失败]"):
        candidate_text = _format_candidates(candidates)
        final_answer = (
            "### 初步判断\n"
            f"当前诊断置信度为 **{confidence:.2f}**。\n\n"
            f"### 候选故障\n{candidate_text}\n\n"
            f"### 检索证据\n{context[:5000]}\n\n"
            f"### 下一步需要确认\n{clarification}"
        )

    citations = _extract_citations(state)
    return {"final_answer": final_answer, "citations": citations}
