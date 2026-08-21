"""Plan 节点：意图分析 + 问题改写 + 关键词提取。

设计原则：只读调用现有 llm_service._call_model()，不修改原文件。
"""
import json
import logging
from typing import Dict

from backend.core.llm_service import llm_service  # 只读引用，不修改
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.memory import inject_memory, maybe_summarize

logger = logging.getLogger(__name__)

PLAN_PROMPT = """你是一个意图分析专家。分析用户问题，提取检索信息并规划检索工具。
可用的工具（工具名: 用途）：
- vector_search: 向量检索文档分块（故障现象描述、标准条款、操作说明等文本内容）
- kg_search: 知识图谱检索（实体关系、故障因果链、故障树拓扑、社区报告摘要）
- sql_query: 数据库查询（表格数据、图片记录、参数规格）

用户问题: {query}
对话历史: {history}
用户记忆: {memory}
系统设定: {system_prompt}

注意：如果用户问题包含指代词（如"他""那个""继续""上面提到的"），请结合对话历史理解真实意图，并在 rewritten_query 中补全完整信息。

输出严格的 JSON（不要有其他文字）:
{{
  "intent": "document|graph|table|hybrid",
  "rewritten_query": "改写后的检索查询词（简洁的检索关键词组合）",
  "entity_keywords": ["实体关键词1", "实体关键词2"],
  "type_keywords": ["类型关键词1"],
  "tools_plan": [{{"tool": "kg_search", "reason": "计划用该工具检索的内容，例如：实体关系、故障树因果链、社区报告"}}],
  "reasoning": "简要说明判断意图、规划工具的理由（30-80字，让用户理解你的思考过程）"
}}

意图规则:
- document: 查文档内容、故障现象、标准条款、说明文字
- graph: 查实体关系、故障原因链、组件关系、故障树因果链
- table: 查表格数据、参数规格
- hybrid: 综合诊断或问题涉及多个方面

tools_plan 规划规则（必须在 reasoning 之外，明确列出将使用的每个工具）:
- 诊断类问题（为什么/原因/导致/故障/怎么回事）：必须包含 kg_search（图谱含故障树因果链与社区报告），并常配合 vector_search 补充文档细节
- 查文档内容/标准条款：vector_search
- 查表格/参数/图片：sql_query
- 综合问题：多个工具并列，并注明各工具侧重检索的内容

诊断类问题（包含"为什么""怎么回事""原因""导致""由于""造成""故障"等词）应判定为 graph 或 hybrid，
并在 entity_keywords 中同时提取：① 设备/组件名（如"三相异步电机"）② 故障现象词（如"过热"），
两者都要作为独立关键词，便于图谱同时匹配设备实体与故障节点。

实体关键词: 问题中提到的具体实体（设备名、故障名、组件名），诊断类问题务必包含故障现象本身
类型关键词: 实体所属类别（如"故障""原因""组件""测试"等）
"""


def _format_history(messages, max_turns=6):
    """格式化对话历史为文本（最近 max_turns*2 条消息）"""
    if not messages:
        return "无"
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    parts = []
    for m in recent:
        role = m.get("role", "")
        content = (m.get("content") or "")[:300]
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
        elif role == "system":
            parts.append(f"系统: {content}")
    return "\n".join(parts) if parts else "无"


def _parse_json_response(text: str) -> dict:
    """解析 LLM 返回的 JSON（容错处理）"""
    try:
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        # 尝试提取 JSON 块
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                s = s[start:end + 1]
        return json.loads(s)
    except Exception as e:
        logger.warning(f"Plan JSON 解析失败: {e}, 原始: {text[:200]}")
        return {
            "intent": "hybrid",
            "rewritten_query": "",
            "entity_keywords": [],
            "type_keywords": [],
        }


async def plan_node(state, config=None) -> Dict:
    """Plan 节点：意图分析 + 问题改写 + 注入长期记忆 + 自动摘要压缩"""
    agent_id = state.get("agent_id", "")
    query = state.get("query", "")
    system_prompt = state.get("system_prompt", "")

    # 注入长期记忆（智能体关闭了长期记忆则跳过注入）
    memory_ctx = state.get("memory_context", "")
    if not memory_ctx and state.get("memory_enabled", True):
        try:
            mem_result = await inject_memory(state, config or {})
            memory_ctx = mem_result.get("memory_context", "")
        except Exception:
            memory_ctx = ""

    # 自动摘要压缩：对话超过阈值时，用 LLM 压缩旧消息（阈值/保留轮数读 agent 配置）
    messages = state.get("messages", [])
    summary_threshold = int(state.get("summary_threshold", 12) or 12)
    keep_recent = int(state.get("keep_recent_turns", 6) or 6)
    if len(messages) >= summary_threshold:
        try:
            summary_result = await maybe_summarize(
                state, summary_threshold=summary_threshold, keep_recent=keep_recent, agent_id=agent_id
            )
            if summary_result:
                logger.info(f"Plan 节点触发自动摘要: {len(messages)} → {len(summary_result.get('messages', messages))} 条消息")
        except Exception as e:
            logger.warning(f"自动摘要失败（不影响流程）: {e}")

    # 取 Plan 角色模型：优先用用户选择的 chat_model_id，回退到 agent_models 默认
    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "plan") if agent_id else None
    # 用智能体配置的模型采样参数覆盖模型默认值
    model = agent_model_role_service.apply_agent_llm_params(state, model)

    prompt = PLAN_PROMPT.format(
        query=query,
        history=_format_history(state.get("messages", [])),
        memory=memory_ctx or "无",
        system_prompt=system_prompt or "通用知识问答",
    )

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    try:
        # 流式调用：逐 token 推送到前端思考面板，同时累积完整响应用于 JSON 解析
        if stream_queue is not None:
            response = ""
            async for token in llm_service.stream_model(
                [{"role": "user", "content": prompt}], model
            ):
                response += token
                try:
                    await stream_queue.put({"type": "thinking_delta", "source": "plan", "content": token})
                except Exception:
                    pass
        else:
            response = llm_service._call_model(
                [{"role": "user", "content": prompt}], model
            )
        parsed = _parse_json_response(response)
    except Exception as e:
        logger.error(f"Plan 节点调用失败: {e}")
        parsed = {
            "intent": "hybrid",
            "rewritten_query": query,
            "entity_keywords": [],
            "type_keywords": [],
            "reasoning": f"Plan 调用失败，使用默认 hybrid 意图: {e}",
        }

    return {
        "intent": parsed.get("intent", "hybrid"),
        "rewritten_query": parsed.get("rewritten_query") or query,
        "entity_keywords": parsed.get("entity_keywords", []),
        "type_keywords": parsed.get("type_keywords", []),
        "tools_plan": parsed.get("tools_plan", []),
        "plan_reasoning": parsed.get("reasoning", ""),
        "memory_context": memory_ctx,
        "iteration": state.get("iteration", 0) + 1,
    }
