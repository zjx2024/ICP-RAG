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
from backend.core.llm_service import llm_service

logger = logging.getLogger(__name__)


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


SCOPE_ROUTER_TIMEOUT_SECONDS = _positive_env_float(
    "ICP_AGENT_SCOPE_TIMEOUT_SECONDS", 20.0
)
SCOPE_ROUTER_MAX_PREDICTION_TOKENS = 256

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

# 这里只识别用户明确点名的七类产品资料。它不是检索词扩展表，也不会
# 决定后续调用什么工具；唯一用途是区分“补查某类资料”和“询问智能体
# 是否具备该能力”，防止入口路由在两者之间短路到能力介绍。
_DOCUMENT_SOURCE_ALIASES = (
    (
        "function_locator",
        (
            "function locator",
            "function-locator",
            "系统架构与功能说明",
            "系统架构文档",
            "功能说明文档",
            "架构文档",
        ),
    ),
    ("hmc", ("hmc", "hmc故障代码手册", "故障代码手册")),
    ("fta", ("fta", "fta故障树分析", "故障树分析", "故障树文档")),
    (
        "rbd",
        ("rbd", "rbd可靠性框图分析", "可靠性框图分析", "可靠性框图文档", "可靠性框图"),
    ),
    (
        "fmeca",
        ("fmeca", "fmeca故障模式影响及危害性分析", "故障模式影响及危害性分析"),
    ),
    ("mfl", ("mfl", "mfl维修故障清单", "维修故障清单")),
    (
        "fault_history",
        ("故障历史查询文档", "故障历史文档", "故障历史", "历史排故案例"),
    ),
)

_KNOWLEDGE_TURN_PATTERN = re.compile(
    r"(?i)(?:"
    r"\b(?:FUNC|MOD|SUB|LCN|HAMC|HSC|HMC|FM|FTA|RBD|MFL)[-_ *A-Z0-9]*\b"
    r"|系统架构|子系统|模块|部件|接口|通道|电源|电容|可靠性"
    r"|故障|异常|损坏|失效|告警|影响|原因|维修|修复|重启|复位"
    r")"
)


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _mentioned_document_sources(query: str) -> tuple[str, ...]:
    compact = _compact_text(query)
    found = []
    for doc_type, aliases in _DOCUMENT_SOURCE_ALIASES:
        if any(_compact_text(alias) in compact for alias in aliases):
            found.append(doc_type)
    return tuple(found)


def _is_pure_source_capability_question(query: str) -> bool:
    """Return whether the user only asks what sources the agent *can* query.

    The check is deliberately narrow. Content-bearing requests such as
    “能查一下 RBD 里的 A 路影响吗” retain the words after the source name and
    therefore do not satisfy this contract.
    """

    compact = _compact_text(query).strip("，。！？?!")
    prefix = re.match(
        r"^(?:你|这个智能体|该智能体|本系统|系统)?"
        r"(?:是否能|能不能|能否|能|是否可以|可不可以|可以|是否支持|支持|会不会|会)"
        r"(?:查一下|查|查询|检索|读取|处理|使用|看)?",
        compact,
    )
    if prefix is None:
        return False
    remainder = compact[prefix.end() :].strip("，。！？?!吗么呢")
    if re.fullmatch(r"(?:哪些|什么|这几类|这些)?(?:资料|文档|知识|知识库内容)", remainder):
        return True
    if not _mentioned_document_sources(remainder):
        return False
    for _, aliases in _DOCUMENT_SOURCE_ALIASES:
        for alias in sorted(aliases, key=len, reverse=True):
            remainder = remainder.replace(_compact_text(alias), "")
    remainder = re.sub(r"(?:文档|资料|数据|内容|和|及|与|或|、|,|，|/)", "", remainder)
    return not remainder


def _source_followup_kind(query: str) -> str:
    """Classify an explicit named-source retrieval correction, if present."""

    if not _mentioned_document_sources(query) or _is_pure_source_capability_question(query):
        return ""
    compact = _compact_text(query)
    if re.search(
        r"(?:请|麻烦)?(?:你)?(?:补查|补充查询|补充检索|追加查询|追加检索|"
        r"重新查询|重新检索|再查|再查询|再检索|一并查询|一并检索)",
        compact,
    ):
        return "supplement"
    if re.search(
        r"(?:请|麻烦)(?:你)?(?:查|查询|检索|参考|结合|使用|用)"
        r"|(?:补充|追加).*(?:依据|证据|来源|资料|文档)"
        r"|(?:结合|参考|使用|用).*(?:再|重新|补充)(?:回答|分析|判断)",
        compact,
    ):
        return "supplement"
    if re.search(
        r"(?:为什么|为何|怎么|咋)(?:还|却|就|都)?(?:没|没有|不曾|不)"
        r"(?:去)?(?:查|查询|检索|看|参考|用|使用|结合|引用)",
        compact,
    ):
        return "omission"
    if re.search(
        r"(?:没|没有|不)(?:去)?(?:查|查询|检索|看|参考|用|使用|结合|引用)"
        r".*(?:吗|呢|[?？])$",
        compact,
    ):
        return "omission"
    return ""


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

    previous_query = str(items[previous_user_index].get("content") or "")
    if _is_pure_source_capability_question(previous_query):
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
        if re.search(r"\[\^\d+\]", str(item.get("content") or "")):
            return True

    return bool(
        _mentioned_document_sources(previous_query)
        or _KNOWLEDGE_TURN_PATTERN.search(previous_query)
    )


def _is_named_source_followup(query: str, messages: Any) -> bool:
    return bool(
        _source_followup_kind(query)
        and _has_prior_knowledge_turn(messages, query)
    )


# 询问点名资料“本身能证明什么/不能替代什么”是对该类资料内容边界的知识
# 问题，需要检索该类资料后回答。主语是“你/智能体”的能力问法不在此列。
_SOURCE_BOUNDARY_QUESTION_PATTERN = re.compile(
    r"(?:能证明什么|不能证明什么|证明什么|能替代什么|不能替代什么|"
    r"能不能直接给出|能否直接给出|能直接给出|能直接给|不能直接给|"
    r"能不能替代|能否替代|能替代|不能替代|能说明什么|不能说明什么)"
)

# 上一轮知识/诊断查询后，询问“继续排查还缺什么信息”是对该轮证据边界
# 的自然延续；路由到能力介绍会打断诊断并丢掉会话上下文。
_CONTINUE_INVESTIGATION_PATTERN = re.compile(
    r"(?:为了继续|继续查|继续排查|继续定位|接着查|接着排查|往下查|"
    r"接下来(?:要|需要)?(?:查|排查|定位|做什么)|"
    r"(?:最少|至少)?还需要(?:我)?(?:提供|补充|给出)(?:什么|哪些)(?:原始)?"
    r"(?:信息|资料|输入|线索)|"
    r"还(?:缺|需要)(?:什么|哪些)(?:信息|资料))"
)


def _is_named_source_boundary_question(query: str) -> bool:
    compact = _compact_text(query)
    if re.match(r"^(?:你|这个智能体|该智能体|本系统|系统|我)", compact):
        return False
    if not _mentioned_document_sources(query):
        return False
    return bool(_SOURCE_BOUNDARY_QUESTION_PATTERN.search(compact))


def _is_continue_investigation_followup(query: str, messages: Any) -> bool:
    compact = _compact_text(query)
    if not _CONTINUE_INVESTIGATION_PATTERN.search(compact):
        return False
    return _has_prior_knowledge_turn(messages, query)

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


def _format_history(messages: Any, *, max_turns: int = 4) -> str:
    if not isinstance(messages, list) or not messages:
        return "无"
    recent = messages[-max_turns * 2 :]
    lines = []
    for item in recent:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = re.sub(r"\[\^\d+\]", "", str(item.get("content") or ""))
        content = " ".join(content.split())[:500]
        if content:
            lines.append(f"{'用户' if role == 'user' else '助手'}：{content}")
    return "\n".join(lines) if lines else "无"


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
    question = " ".join(str(parsed.get("clarification_question") or "").split())[:300]
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
    prepared["num_predict"] = SCOPE_ROUTER_MAX_PREDICTION_TOKENS
    if prepared.get("provider") == "ollama":
        prepared["reasoning"] = False
        prepared["temperature"] = 0.0
        prepared["format"] = "json"
    return prepared


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

    # “补查/漏查某类资料”是上一轮知识任务的继续，不是询问智能体能力。
    # 该确定性复核只决定入口路径；具体查哪些记录仍由 plan 授权。
    if _is_named_source_followup(query, state.get("messages")):
        return {
            "scope_route": SCOPE_KNOWLEDGE_BASE,
            "scope_route_source": "deterministic_source_followup",
            "scope_route_reason": "用户承接上一轮知识问题，要求补查明确点名的工程资料",
            "scope_confidence": 1.0,
            "clarification_question": "",
        }

    # 询问点名资料自身“能证明什么、不能替代什么”问的是资料内容边界，
    # 属于知识问题；分类模型容易把“能不能”误读成能力询问，这里确定性
    # 纠正。主语为“你/智能体”的问法仍由能力路径处理。
    if _is_named_source_boundary_question(query):
        return {
            "scope_route": SCOPE_KNOWLEDGE_BASE,
            "scope_route_source": "deterministic_source_boundary",
            "scope_route_reason": "用户询问点名工程资料自身的证明边界，需检索该类资料回答",
            "scope_confidence": 1.0,
            "clarification_question": "",
        }

    # 上一轮知识/诊断查询后询问“继续查还需要提供什么”，是对本轮证据
    # 边界的追问，必须留在知识流程内，不能短路到能力自我介绍。
    if _is_continue_investigation_followup(query, state.get("messages")):
        return {
            "scope_route": SCOPE_KNOWLEDGE_BASE,
            "scope_route_source": "deterministic_continue_investigation",
            "scope_route_reason": "用户承接上一轮诊断，询问继续排查所需的补充信息",
            "scope_confidence": 1.0,
            "clarification_question": "",
        }

    # 仅覆盖形态非常明确的能力问法，避免模型把“能查某记录吗”一类实际
    # 知识请求误送检索；带有来源内部对象或问题维度的请求不会命中这里。
    if _is_pure_source_capability_question(query):
        return {
            "scope_route": SCOPE_AGENT_CAPABILITY,
            "scope_route_source": "deterministic_source_capability",
            "scope_route_reason": "用户只询问可查询的资料范围",
            "scope_confidence": 1.0,
            "clarification_question": "",
            "authorized_tools": [],
            "allowed_tools": [],
            "tools_plan": [],
            "calculation_requests": [],
            "relation_expansion": {"needed": False, "reason": ""},
        }

    prompt = SCOPE_ROUTER_PROMPT.format(
        query=query,
        history=_format_history(state.get("messages")),
    )
    parsed: Optional[Dict[str, Any]] = None
    failure_reason = ""
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm_service._call_model,
                [{"role": "user", "content": prompt}],
                _model_for_state(state),
            ),
            timeout=SCOPE_ROUTER_TIMEOUT_SECONDS,
        )
        parsed = parse_scope_response(response)
        if parsed is None:
            failure_reason = "范围分类模型没有返回有效 JSON"
    except asyncio.TimeoutError:
        failure_reason = f"范围分类超过 {SCOPE_ROUTER_TIMEOUT_SECONDS:g} 秒时间预算"
    except Exception as exc:
        failure_reason = f"范围分类模型不可用：{exc}"

    if parsed is None:
        logger.warning(failure_reason)
        # 分类器不可用时不能把未知请求直接送入七类资料宽召回。让用户
        # 重试或明确对象，比错误调用领域工具更安全。
        return {
            "scope_route": SCOPE_CLARIFICATION,
            "scope_route_source": "safe_fallback",
            "scope_route_reason": failure_reason,
            "scope_confidence": 0.0,
            "clarification_question": (
                "我暂时无法确认这个问题是否属于当前知识库范围。"
                "请补充相关系统、模块、FUNC/HMC 编号或故障现象后重试。"
            ),
            "authorized_tools": [],
            "allowed_tools": [],
            "tools_plan": [],
        }

    route = str(parsed["route"])
    update: Dict[str, Any] = {
        "scope_route": route,
        "scope_route_source": "model",
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
