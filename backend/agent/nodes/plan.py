"""Plan 节点：ICP 故障诊断意图分析、问题改写和工具规划。"""
import json
import logging
from typing import Dict

from backend.core.llm_service import llm_service
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.memory import inject_memory, maybe_summarize

logger = logging.getLogger(__name__)

PLAN_PROMPT = """你是 ICP 航空电子系统故障诊断的检索规划器。
用户通常只能描述模糊故障现象，不一定知道模块名或 HMC。你的任务是补全检索表达并规划七类知识工具。

可用知识工具：
- function_locator：将现象映射到系统、模块和功能
- hybrid_search：在七类文档中做关键词 + 向量混合召回
- hmc_query：按 LCN/HAMC/HSC 或故障描述查 HMC 和维修方法
- fmeca_query：查故障模式、原因、影响、严酷度、检测方法
- fta_query：查顶事件、原因链和 AND/OR 逻辑
- rbd_query：查可靠性框图、串并联结构、成功判据和参数
- mfl_query：查模块/SUR板/功能电路故障、API 和维修定位
- fault_history_query：查历史排故案例、步骤、根因和最终方案
- fault_reasoning：融合证据、排序候选、判断是否需要追问

用户问题：{query}
对话历史：{history}
会话记忆：{memory}
系统设定：{system_prompt}

输出严格 JSON，不要输出其他内容：
{{
  "intent": "exact_code|symptom_diagnosis|function_location|cause_analysis|impact_analysis|maintenance_query|reliability_query|history_query|hybrid",
  "rewritten_query": "面向检索的完整故障描述，保留告警码、时间阶段、现象和已执行操作",
  "entity_keywords": ["设备/模块/功能/故障现象/告警码"],
  "type_keywords": ["原因|影响|维修|可靠性|历史等"],
  "tools_plan": [
    {{"tool": "function_locator", "reason": "为什么需要该工具"}},
    {{"tool": "hybrid_search", "reason": "为什么需要该工具"}}
  ],
  "reasoning": "30-100字，说明检索规划，不输出未经证据支持的诊断结论"
}}

规划规则：
1. 出现完整或近似 HMC/LCN/HAMC/HSC：intent=exact_code，必须包含 hmc_query。
2. 仅有模糊现象：intent=symptom_diagnosis，先 function_locator + hybrid_search，再由 fault_reasoning 汇总。
3. 问“为什么/原因/导致”：必须包含 fta_query；通常同时包含 fmeca_query。
4. 问“影响/严重程度/危害”：必须包含 fmeca_query。
5. 问“怎么办/维修/检查/API”：包含 hmc_query、mfl_query；有历史经验诉求时增加 fault_history_query。
6. 问“可靠度/串并联/冗余/成功判据”：必须包含 rbd_query。
7. 问过去发生过什么、怎么排过：必须包含 fault_history_query。
8. 不得把“故障”一词本身当作有区分度的实体关键词；优先抽取现象词、阶段词、告警文字和设备名。
9. 有指代词时必须结合对话历史补全 rewritten_query。
"""


def _format_history(messages, max_turns=6):
    if not messages:
        return "无"
    recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    parts = []
    for message in recent:
        role = message.get("role", "")
        content = (message.get("content") or "")[:500]
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
        elif role == "system":
            parts.append(f"系统: {content}")
    return "\n".join(parts) if parts else "无"


def _parse_json_response(text: str) -> dict:
    try:
        value = text.strip()
        if value.startswith("```json"):
            value = value[7:]
        if value.startswith("```"):
            value = value[3:]
        if value.endswith("```"):
            value = value[:-3]
        value = value.strip()
        if not value.startswith("{"):
            start = value.find("{")
            end = value.rfind("}")
            if start >= 0 and end > start:
                value = value[start:end + 1]
        return json.loads(value)
    except Exception as exc:
        logger.warning("Plan JSON parse failed: %s; raw=%s", exc, text[:200])
        return {
            "intent": "symptom_diagnosis",
            "rewritten_query": "",
            "entity_keywords": [],
            "type_keywords": ["故障诊断"],
            "tools_plan": [
                {"tool": "function_locator", "reason": "先定位系统功能"},
                {"tool": "hybrid_search", "reason": "从七类文档召回候选"},
                {"tool": "fault_reasoning", "reason": "融合候选并判断是否追问"},
            ],
        }


async def plan_node(state, config=None) -> Dict:
    agent_id = state.get("agent_id", "")
    query = state.get("query", "")
    system_prompt = state.get("system_prompt", "")

    memory_context = state.get("memory_context", "")
    if not memory_context and state.get("memory_enabled", False):
        try:
            result = await inject_memory(state, config or {})
            memory_context = result.get("memory_context", "")
        except Exception:
            memory_context = ""

    messages = state.get("messages", [])
    summary_threshold = int(state.get("summary_threshold", 20) or 20)
    keep_recent = int(state.get("keep_recent_turns", 8) or 8)
    if len(messages) >= summary_threshold:
        try:
            await maybe_summarize(
                state,
                summary_threshold=summary_threshold,
                keep_recent=keep_recent,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning("Conversation summary failed: %s", exc)

    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "plan") if agent_id else None
    model = agent_model_role_service.apply_agent_llm_params(state, model)

    prompt = PLAN_PROMPT.format(
        query=query,
        history=_format_history(messages),
        memory=memory_context or "无",
        system_prompt=system_prompt or "ICP 故障诊断",
    )

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    try:
        if stream_queue is not None:
            response = ""
            async for token in llm_service.stream_model([{"role": "user", "content": prompt}], model):
                response += token
                try:
                    await stream_queue.put({"type": "thinking_delta", "source": "plan", "content": token})
                except Exception:
                    pass
        else:
            response = llm_service._call_model([{"role": "user", "content": prompt}], model)
        parsed = _parse_json_response(response)
    except Exception as exc:
        logger.error("Plan node failed: %s", exc)
        parsed = _parse_json_response("")
        parsed["rewritten_query"] = query
        parsed["reasoning"] = f"规划模型不可用，采用 ICP 默认混合检索流程：{exc}"

    return {
        "intent": parsed.get("intent", "symptom_diagnosis"),
        "rewritten_query": parsed.get("rewritten_query") or query,
        "entity_keywords": parsed.get("entity_keywords", []),
        "type_keywords": parsed.get("type_keywords", []),
        "tools_plan": parsed.get("tools_plan", []),
        "plan_reasoning": parsed.get("reasoning", ""),
        "memory_context": memory_context,
        "iteration": state.get("iteration", 0) + 1,
    }
