"""ICP 对话入口的范围路由。

该节点只判断请求应该进入哪条产品流程，不做领域检索规划，也不调用
任何知识工具。领域任务仍由 ``plan`` 负责；能力说明、范围外问题和无法
消解的指代则在图入口直接结束，避免把未知请求降级成七类资料宽召回。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Mapping, Optional

from backend.agent.model_role_service import agent_model_role_service
from backend.agent.model_runtime_policy import bounded_model_transport, remaining_seconds, TURN_TIMEOUT_SECONDS
from backend.core.llm_service import llm_service

logger = logging.getLogger(__name__)


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


SCOPE_ROUTER_TIMEOUT_SECONDS = TURN_TIMEOUT_SECONDS

SCOPE_KNOWLEDGE_BASE = "knowledge_base"
SCOPE_AGENT_CAPABILITY = "agent_capability"
SCOPE_OUT_OF_SCOPE = "out_of_scope"
SCOPE_CLARIFICATION = "clarification"
VALID_SCOPE_ROUTES = {
    SCOPE_KNOWLEDGE_BASE,
    SCOPE_AGENT_CAPABILITY,
    SCOPE_OUT_OF_SCOPE,
    SCOPE_CLARIFICATION,
}

_ROUTE_ALIASES = {
    "knowledge": SCOPE_KNOWLEDGE_BASE,
    "knowledge_related": SCOPE_KNOWLEDGE_BASE,
    "agent_help": SCOPE_AGENT_CAPABILITY,
    "capability": SCOPE_AGENT_CAPABILITY,
    "uncertain": SCOPE_CLARIFICATION,
    "clarify": SCOPE_CLARIFICATION,
}

def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _has_prior_knowledge_turn(messages: Any, current_query: str) -> bool:
    """Conservatively identify the immediately preceding knowledge turn."""

    if not isinstance(messages, list):
        return False
    items = [item for item in messages if isinstance(item, Mapping)]
    current = _compact_text(current_query).strip("，。！？?!")
    while items:
        last = items[-1]
        if (
            str(last.get("role") or "") == "user"
            and _compact_text(last.get("content")).strip("，。！？?!") == current
        ):
            items.pop()
            continue
        break

    previous_user_index = next(
        (
            index
            for index in range(len(items) - 1, -1, -1)
            if str(items[index].get("role") or "") == "user"
        ),
        -1,
    )
    if previous_user_index < 0:
        return False

    for item in items[previous_user_index + 1 :]:
        if str(item.get("scope_route") or "") == SCOPE_KNOWLEDGE_BASE:
            return True
        events = item.get("thinking_events")
        if isinstance(events, list) and any(
            isinstance(event, Mapping)
            and str(event.get("route") or "") == SCOPE_KNOWLEDGE_BASE
            for event in events
        ):
            return True
        if str(item.get("role") or "") != "assistant":
            continue
        if item.get("conversation_anchors"):
            return True
        if item.get("answer_delivery_manifest") or item.get("citations"):
            return True
    return False

SCOPE_ROUTER_PROMPT = """你是 ICP 航空电子知识智能体的入口路由器。只判断处理路径，不回答问题，不规划工具。

互斥路径：
- knowledge_base：需要查询已上传的 ICP 七类工程资料才能回答。包括系统/子系统/模块/功能/接口基础信息，架构组成，FUNC/HMC 等业务编号，故障现象、原因、影响、维修、历史、FTA 或 RBD。
- agent_capability：只询问智能体是谁、能做什么、资料范围、如何使用或如何提问；普通问候也走此路径。
- out_of_scope：与 ICP 航空电子知识、故障诊断及本智能体使用均无关。
- clarification：结合最近对话仍无法确定用户指代的对象或所问事项；只在缺少这项信息导致连检索对象都无法确定时使用。一个简短但明确的 ICP 故障现象仍属于 knowledge_base。

判断规则：
1. 按用户真正要获取的信息判断，不按关键词判断。“你能查一下 MOD-PWR-PSUA 吗”是 knowledge_base，不是能力询问。
2. ID、名称、缩写和自然语言描述只是实体表达方式。“MOD-PWR-PSUA 是什么”和“电源 A 模块是什么”都属于 knowledge_base。
3. 结合对话历史消解“它、这个模块、那怎么修”等追问；助手先前的判断不能当作工程事实。
   这一条适用于所有知识问题：定义后问“它重要吗”、架构后问“它上游是谁”、异常后问“这种情况什么级别”都是 knowledge_base。
4. 上一轮是知识查询时，用户指出漏查或要求补查某类工程资料，属于对证据覆盖的纠正，走 knowledge_base。这适用于系统架构与功能说明、HMC、FMECA、FTA、RBD、MFL 和故障历史全部七类资料。
5. “你能查 RBD 吗”“你能查哪些资料”只询问能力，仍走 agent_capability；“你不查 RBD 文档吗”“请补查 RBD 依据”“为什么没查 FTA”是在要求实际补查，走 knowledge_base。
6. 询问某类资料本身“能证明什么、不能替代什么、能不能直接给出根因”问的是资料内容边界，属于 knowledge_base；只有主语是“你/智能体”的能力问法才走 agent_capability。
7. 上一轮知识查询未命中或证据不足后，问“为了继续查还需要我提供什么信息”仍属于该诊断对话，走 knowledge_base。
8. 如果一句话包含实质性的 ICP 知识问题，即使同时有礼貌表达，也优先 knowledge_base。
9. 不得输出工具名、检索计划、答案、思考过程或工程结论。

示例：
- “FUNC-705 的具体功能是什么” → knowledge_base
- “测量值突然跳变，优先怀疑哪里” → knowledge_base
- 上一轮刚查询故障影响，接着问“为什么没查 FTA” → knowledge_base
- “故障树分析能证明什么、不能替代什么” → knowledge_base
- 上一轮查询未命中后问“为了得出结论还需要我补充哪些材料” → knowledge_base
- “你是谁，能查哪些资料” → agent_capability
- “你能查 RBD 吗” → agent_capability
- “怎么使用这个智能体” → agent_capability
- “帮我写一篇咖啡店宣传文案” → out_of_scope
- 没有任何可用上下文时只说“它呢” → clarification

用户问题：{query}
最近对话：{history}

只输出 JSON：
{{
  "route": "knowledge_base|agent_capability|out_of_scope|clarification",
  "confidence": 0.0,
  "reason": "不超过60字的分类依据，仅供内部审计",
  "clarification_question": "仅 clarification 时给出一个简短问题，其他路径为空"
}}
"""


def _format_history(messages: Any, *, max_turns: int | None = None) -> str:
    from backend.agent.answer_context_binding import conversation_review_context

    context = conversation_review_context(
        {"messages": messages}, max_messages=None if max_turns is None else max_turns * 2,
    )
    return json.dumps(context["messages"], ensure_ascii=False) if context["messages"] else "无"


def _extract_json_object(value: Any) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_scope_response(value: Any) -> Optional[Dict[str, Any]]:
    """Validate the small model contract without inferring a route from text."""

    parsed = _extract_json_object(value)
    if not parsed:
        return None
    raw_route = str(parsed.get("route") or "").strip()
    route = _ROUTE_ALIASES.get(raw_route, raw_route)
    if route not in VALID_SCOPE_ROUTES:
        return None
    try:
        confidence = min(max(float(parsed.get("confidence") or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    question = " ".join(str(parsed.get("clarification_question") or "").split())
    if route == SCOPE_CLARIFICATION and not question:
        question = "请说明你指的是哪个系统、模块、代码或故障现象。"
    if route != SCOPE_CLARIFICATION:
        question = ""
    return {
        "route": route,
        "confidence": confidence,
        "reason": " ".join(str(parsed.get("reason") or "").split())[:300],
        "clarification_question": question,
    }


def _prepare_scope_model(model: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not model:
        return None
    prepared = dict(model)
    if prepared.get("provider") == "ollama":
        prepared["reasoning"] = False
        prepared["temperature"] = 0.0
        prepared["format"] = "json"
    prepared["stream"] = False
    return bounded_model_transport(prepared, SCOPE_ROUTER_TIMEOUT_SECONDS)


def _model_for_state(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    chat_model_id = str(state.get("chat_model_id") or "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client

        model = postgres_client.get_model(chat_model_id)
    else:
        agent_id = str(state.get("agent_id") or "")
        model = (
            agent_model_role_service.get_model_by_role(agent_id, "plan")
            if agent_id
            else None
        )
    model = agent_model_role_service.apply_agent_llm_params(dict(state), model)
    return _prepare_scope_model(model)


async def scope_router_node(state: Mapping[str, Any], config=None) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    if not query:
        return {
            "scope_route": SCOPE_CLARIFICATION,
            "scope_route_source": "deterministic_empty_input",
            "scope_route_reason": "用户问题为空",
            "scope_confidence": 1.0,
            "clarification_question": "请描述要查询的系统、模块、代码或故障现象。",
            "authorized_tools": [],
            "allowed_tools": [],
            "tools_plan": [],
        }

    prompt = SCOPE_ROUTER_PROMPT.format(
        query=query,
        history=_format_history(state.get("messages")),
    )
    parsed: Optional[Dict[str, Any]] = None
    failure_reason = ""
    failure_kind = ""
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm_service.call_model,
                [{"role": "user", "content": prompt}],
                _model_for_state(state),
            ),
            timeout=remaining_seconds(),
        )
        parsed = parse_scope_response(response)
        if parsed is None:
            failure_reason = "范围分类模型没有返回有效 JSON"
            failure_kind = "invalid_output"
    except asyncio.TimeoutError:
        failure_reason = f"范围分类超过 {SCOPE_ROUTER_TIMEOUT_SECONDS:g} 秒时间预算"
        failure_kind = "timeout"
    except Exception as exc:
        failure_reason = f"范围分类模型不可用：{exc}"
        failure_kind = "model_error"

    if parsed is None:
        logger.warning(failure_reason)
        try:
            from backend.agent.tracing import record_degradation
            record_degradation("scope_router", detail=failure_kind)
        except Exception as exc:
            logger.debug("Unable to trace scope fallback: %s", type(exc).__name__)
        # 分类器不可用时不能把未知请求直接送入七类资料宽召回。让用户
        # 重试或明确对象，比错误调用领域工具更安全。
        return {
            "scope_route": SCOPE_CLARIFICATION,
            "scope_route_source": "safe_fallback",
            "scope_failure_kind": failure_kind,
            "scope_route_reason": failure_reason,
            "scope_confidence": 0.0,
            "clarification_question": (
                "入口模型暂时未能完成问题分类，本轮尚未查询工程资料。"
                "请稍后重试；这不表示你的问题缺少对象或不在知识库范围内。"
            ),
            "authorized_tools": [],
            "allowed_tools": [],
            "tools_plan": [],
        }

    route = str(parsed["route"])
    update: Dict[str, Any] = {
        "scope_route": route,
        "scope_route_source": "model",
        "scope_failure_kind": "",
        "scope_route_reason": parsed["reason"],
        "scope_confidence": parsed["confidence"],
        "clarification_question": parsed["clarification_question"],
    }
    if route != SCOPE_KNOWLEDGE_BASE:
        update.update(
            {
                "authorized_tools": [],
                "allowed_tools": [],
                "tools_plan": [],
                "calculation_authorizations": [],
                "calculation_requests": [],
                "relation_expansion": {"needed": False, "reason": ""},
            }
        )
    return update


def route_after_scope_router(state: Mapping[str, Any]) -> str:
    route = str(state.get("scope_route") or "")
    return route if route in VALID_SCOPE_ROUTES else SCOPE_CLARIFICATION


__all__ = [
    "SCOPE_AGENT_CAPABILITY",
    "SCOPE_CLARIFICATION",
    "SCOPE_KNOWLEDGE_BASE",
    "SCOPE_OUT_OF_SCOPE",
    "VALID_SCOPE_ROUTES",
    "parse_scope_response",
    "route_after_scope_router",
    "scope_router_node",
]
