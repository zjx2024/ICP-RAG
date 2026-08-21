"""Answer 节点：生成最终回答（含流式输出与引用提取）。

设计原则：只读调用现有 llm_service，不修改原文件。
长期记忆抽取统一由 chat_router 在对话完整结束后执行（此处 state 尚无 final_answer）。
"""
import re
import json
import logging
from typing import Dict

from backend.core.llm_service import llm_service  # 只读引用，不修改
from backend.agent.model_role_service import agent_model_role_service

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """基于检索结果和对话历史回答用户问题。

{system_prompt}

对话历史:
{history}

用户问题: {query}
用户记忆: {memory}

检索结果:
{context}

回答要求:
1. 用中文回答
2. 结合对话历史理解用户问题的上下文（如指代词、省略内容）
3. 基于检索结果，不要编造
4. 引用检索内容时，把引用的具体内容直接写在正文中，并在引用的内容后面标注来源编号 [^编号]（编号对应当前检索结果列表中的序号，如 [^1]）。系统会把 [^编号] 渲染成可点击的来源标签，点击可展开引用的原文内容
5. 信息不足时如实说明
6. 条理清晰，分点叙述
7. 如果检索结果中包含图片URL（image_url字段），在回答中用 HTML img 标签展示图片:
   <img src="图片URL" alt="图片标题" />
   当用户要求查看图片、放映图片，或检索结果中有相关图片时，务必将图片展示出来。
   直接使用 image_url 字段的完整 URL 作为 src 值，不要编造 URL。
   重要：src 值不要用反引号、引号或其他符号包裹，直接写纯 URL。
   正确示例: <img src="http://192.168.1.100:9999/images/xxx.png" alt="图片标题" />
   错误示例: <img src="`http://...`" alt="图片标题" />
8. 如果检索结果中包含表格数据（table_data字段），用 markdown 表格语法输出原始数据:
   | 列1 | 列2 | 列3 |
   |-----|-----|-----|
   | 值1 | 值2 | 值3 |
   当用户要求查看表格、原始数据时，务必将表格内容完整展示。
"""


def _format_history(messages, max_turns=6):
    """格式化对话历史为文本（最近 max_turns*2 条消息）"""
    if not messages:
        return "无"
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    parts = []
    for m in recent:
        role = m.get("role", "")
        content = (m.get("content") or "")[:500]
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
        elif role == "system":
            parts.append(f"系统: {content}")
    return "\n".join(parts) if parts else "无"


def _clean_img_tags(text: str) -> str:
    """清洗 LLM 输出中的 <img> 标签：
    - 去除 src 属性值中的反引号和多余空格
    - 确保图片 URL 可直接用于渲染
    """
    def _fix_src(m):
        tag = m.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if src_match:
            clean_url = src_match.group(1).replace('`', '').strip()
            tag = tag[:src_match.start(1)] + clean_url + tag[src_match.end(1):]
        return tag
    return re.sub(r'<img\s+[^>]*src=["\'][^"\']+["\'][^>]*\/?>', _fix_src, text, flags=re.IGNORECASE)


def _format_table_data(meta_data) -> str:
    """将 meta_data（{"data": [[row1...], [row2...]]}）格式化为可读文本"""
    if not meta_data:
        return ""
    if isinstance(meta_data, str):
        try:
            meta_data = json.loads(meta_data)
        except Exception:
            return meta_data[:500]
    if isinstance(meta_data, dict):
        data = meta_data.get("data", [])
    elif isinstance(meta_data, list):
        data = meta_data
    else:
        return str(meta_data)[:500]

    if not data or not isinstance(data, list):
        return ""

    # 转为 markdown 表格文本，让 LLM 能直接引用
    lines = []
    for row in data[:20]:  # 最多 20 行，避免太长
        if isinstance(row, list):
            lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def _format_context(state) -> str:
    """格式化检索结果为上下文"""
    parts = []
    vec_results = state.get("vector_results", [])
    if vec_results:
        parts.append("【文档检索】")
        for i, r in enumerate(vec_results[:15], 1):
            content = r.get("content", "")[:500]
            title = r.get("title", "")
            doc_name = r.get("doc_name", "")
            part = f"[{i}] {title}（来源: {doc_name}）\n{content}"
            # 图片分块附带 image_url，便于 LLM 直接渲染图片
            if r.get("content_type") == "image" and r.get("image_url"):
                part += f"\nimage_url: {r['image_url']}"
            parts.append(part)

    kg_summary = state.get("kg_summary", "")
    if kg_summary:
        # kg_summary 顺序：故障树因果链 → 社区报告 → 相关实体 → 关系
        # 从开头截断 4000 字符，优先保留故障树因果链与社区报告等关键信息
        parts.append(f"【知识图谱】\n{kg_summary[:4000]}")

    sql_results = state.get("sql_results", [])
    if sql_results:
        parts.append("【数据库】")
        for i, r in enumerate(sql_results[:5], 1):
            rtype = r.get("type", "")
            title = r.get("title", "")
            desc = r.get("description", "")
            if rtype == "image" and r.get("image_url"):
                parts.append(f"[{i}] [{rtype}] {title}: {desc[:200]}\nimage_url: {r['image_url']}")
            elif rtype == "table":
                table_data = _format_table_data(r.get("meta_data"))
                parts.append(f"[{i}] [{rtype}] {title}: {desc[:200]}")
                if table_data:
                    parts.append(f"table_data:\n{table_data}")
            else:
                parts.append(f"[{i}] [{rtype}] {title}: {desc[:200]}")

    return "\n\n".join(parts) if parts else "无检索结果"


def _extract_citations(state) -> list:
    """从检索结果提取引用来源"""
    citations = []
    for i, r in enumerate(state.get("vector_results", [])[:15], 1):
        citations.append({
            "id": i,
            "doc_id": r.get("doc_id", ""),
            "doc_name": r.get("doc_name", ""),
            "title": r.get("title", ""),
            "chunk_id": r.get("chunk_id", ""),
            "content": (r.get("content") or "")[:3000],
            "type": "document",
        })
    for r in state.get("sql_results", [])[:5]:
        if r.get("type") == "image":
            citations.append({
                "id": len(citations) + 1,
                "doc_id": r.get("doc_id", ""),
                "doc_name": r.get("doc_name", ""),
                "title": r.get("title", ""),
                "content": (r.get("description") or "")[:500],
                "type": "image",
            })
        elif r.get("type") == "table":
            citations.append({
                "id": len(citations) + 1,
                "doc_id": r.get("doc_id", ""),
                "doc_name": r.get("doc_name", ""),
                "title": r.get("title", ""),
                "content": (r.get("description") or "")[:500],
                "type": "table",
            })
    return citations


async def answer_node(state, config=None) -> Dict:
    """Answer 节点：生成最终回答，流式输出 + 异步抽取记忆"""
    agent_id = state.get("agent_id", "")
    # 优先用用户选择的 chat_model_id，回退到 agent_models 默认
    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "answer") if agent_id else None
    # 用智能体配置的模型采样参数覆盖模型默认值
    model = agent_model_role_service.apply_agent_llm_params(state, model)

    context = _format_context(state)
    prompt = ANSWER_PROMPT.format(
        system_prompt=state.get("system_prompt", "你是知识图谱助手，基于检索结果回答问题。"),
        history=_format_history(state.get("messages", [])),
        query=state.get("query", ""),
        memory=state.get("memory_context", "") or "无",
        context=context,
    )

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    try:
        # 优先使用流式调用：逐 token 推送到 SSE，同时累积完整回答
        if stream_queue is not None:
            final_answer = ""
            async for token in llm_service.stream_model(
                [{"role": "user", "content": prompt}], model
            ):
                final_answer += token
                try:
                    await stream_queue.put({"type": "answer_delta", "content": token})
                except Exception:
                    pass
        else:
            final_answer = llm_service._call_model(
                [{"role": "user", "content": prompt}], model
            )
    except Exception as e:
        logger.error(f"Answer 节点调用失败: {e}")
        final_answer = f"抱歉，生成回答时出错: {e}"
        if stream_queue is not None:
            try:
                await stream_queue.put({"type": "answer_delta", "content": final_answer})
            except Exception:
                pass

    # 后端兜底：清洗 LLM 输出中 img 标签的反引号/空格
    final_answer = _clean_img_tags(final_answer)

    citations = _extract_citations(state)

    # 长期记忆抽取统一在 chat_router 对话完整结束后执行（此处 state 尚无 final_answer，无法可靠抽取）
    return {
        "final_answer": final_answer,
        "citations": citations,
    }
