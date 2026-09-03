"""ICP 智能体的确定性工具授权与执行策略。

LLM 负责提出最小工具计划；本模块负责限制授权上限、校验实际执行集、
约束 OpenSearch 文档类型，并防止模型在用户没有要求数值计算时启动
FTA/RBD 计算器。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

from backend.agent.answer_contract import (
    ensure_intent_answer_fields,
    is_focused_answer_contract,
    normalize_answer_fields,
    tools_for_answer_fields,
)
from backend.agent.query_understanding import extract_business_codes


DOC_TYPE_BY_TOOL = {
    "function_locator": "function_locator",
    "hmc_query": "hmc",
    "fmeca_query": "fmeca",
    "fta_query": "fta",
    "rbd_query": "rbd",
    "mfl_query": "mfl",
    "fault_history_query": "fault_history",
}

# A document name written by the user is an explicit evidence-coverage
# instruction, not a guess about the question's semantic intent.  For example,
# “导致 MSM 损坏的原因，去查 RBD” is still a cause question, but it also asks us
# to inspect the RBD source before answering.  Keep this lexical mapping small
# and only map a named, user-visible document family to its own read tool.  It
# never authorizes an FTA/RBD calculator.
_EXPLICIT_SOURCE_TOOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("function_locator", re.compile(r"系统架构(?:与功能说明)?|功能定位|Function\s*Locator", re.IGNORECASE)),
    # A code such as ``MFL-SYN-041`` is an object-identity query, not the
    # user's instruction to inspect the whole MFL document family.  Keep the
    # named-source detector from treating the prefix of any business ID as an
    # explicit coverage demand.
    ("hmc_query", re.compile(r"(?<![A-Za-z0-9])HMC(?![-_A-Za-z0-9])|故障代码(?:手册)?", re.IGNORECASE)),
    ("fmeca_query", re.compile(r"(?<![A-Za-z0-9])FMECA(?![-_A-Za-z0-9])|失效模式(?:与影响分析)?", re.IGNORECASE)),
    ("fta_query", re.compile(r"(?<![A-Za-z0-9])FTA(?![-_A-Za-z0-9])|故障树", re.IGNORECASE)),
    ("rbd_query", re.compile(r"(?<![A-Za-z0-9])RBD(?![-_A-Za-z0-9])|可靠性框图", re.IGNORECASE)),
    # “维修清单”是用户对 MFL 维修故障清单的日常叫法（如“维修清单给了哪
    # 些可执行核查”），与完整名称同等视为点名该资料来源。
    ("mfl_query", re.compile(r"(?<![A-Za-z0-9])MFL(?![-_A-Za-z0-9])|维修(?:故障)?清单", re.IGNORECASE)),
    ("fault_history_query", re.compile(r"故障历史(?:查询文档|资料|记录)?|历史(?:案例|记录|排故)", re.IGNORECASE)),
)

# These fields make the user-requested source visible in the final answer when
# that source actually returns an approved record.  They are deliberately a
# bounded source summary, not a request to expand every column of the document.
_SOURCE_COVERAGE_FIELDS: Dict[str, tuple[str, ...]] = {
    "function_locator": ("architecture_hierarchy", "architecture_function"),
    "hmc_query": ("hmc_identity", "hmc_symptom"),
    "fmeca_query": ("failure_mode", "fault_cause"),
    "fta_query": ("structural_cause_chain",),
    "rbd_query": ("rbd_success_criterion", "rbd_path_structure"),
    "mfl_query": ("maintenance_locator", "maintenance_action"),
    "fault_history_query": ("history_time", "history_root", "history_solution", "history_result"),
}

# These fields make MFL the authoritative locator: 板级/SUR/功能子电路定位、
# API 上报与平台摸底状态只存在于 MFL 权威记录中。目标模型已把这些字段写进
# required_fields 后，强制 MFL 查询只是合同补全；HMC/FMECA 仍按各自字段提供
# 内容，这里不重写意图、也不把 MFL 扩成无边界清单。
_MFL_CONTRACT_FIELDS = frozenset({
    "maintenance_action",
    "maintenance_locator",
    "maintenance_api",
    "platform_test_status",
    "platform_test_result",
})

CANONICAL_TOOL_NAMES = {
    "function_locator",
    "hybrid_search",
    "hmc_query",
    "fmeca_query",
    "fta_query",
    "fta_analyzer",
    "rbd_query",
    "rbd_calculator",
    "mfl_query",
    "fault_history_query",
    "graph_traversal",
    "evidence_synthesis",
}

TOOL_ALIASES = {
    # 已保存 Agent 和旧规划仍可以使用这个名称。
    "fault_reasoning": "evidence_synthesis",
}

INTENT_ALIASES = {
    "document": "hybrid",
    "graph": "hybrid",
    "table": "hybrid",
    "function_query": "function_location",
}

# ``allowed`` 是模型可以选择的授权上限；``fallback`` 只在规划失败或
# 模型没有给出任何可执行查询时使用。不能再把 fallback 当作每轮的
# 固定工具套餐，否则 ``tools_plan`` 只会沦为展示字段。
INTENT_POLICY: Dict[str, Dict[str, Sequence[str]]] = {
    "entity_profile": {
        "fallback": ("function_locator",),
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "mfl_query",
            "fault_history_query", "hybrid_search", "graph_traversal",
        ),
        "hybrid_allowed": (
            "function_locator", "hmc_query", "fmeca_query", "mfl_query",
            "fault_history_query",
        ),
    },
    "architecture_query": {
        "fallback": ("function_locator", "hybrid_search"),
        "allowed": ("function_locator", "hybrid_search", "graph_traversal"),
    },
    "function_location": {
        "fallback": ("function_locator", "hybrid_search"),
        "allowed": ("function_locator", "hybrid_search", "graph_traversal"),
    },
    "exact_code": {
        "fallback": ("hmc_query",),
        "allowed": ("hmc_query", "hybrid_search"),
    },
    "symptom_diagnosis": {
        "fallback": ("hybrid_search",),
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "fault_history_query",
            "hybrid_search", "fta_query", "mfl_query", "graph_traversal",
        ),
    },
    "cause_analysis": {
        "fallback": ("fmeca_query", "fta_query", "hybrid_search"),
        # The answer stays cause-first, while the adaptive completion phase may
        # enrich a described fault with documented handling and comparable
        # history. This is evidence coverage, not a change to query_targets.
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "fta_query",
            "mfl_query", "fault_history_query", "hybrid_search",
            "graph_traversal",
        ),
    },
    "impact_analysis": {
        "fallback": ("fmeca_query", "hybrid_search"),
        "allowed": (
            "function_locator", "fmeca_query", "fta_query", "rbd_query",
            "hmc_query", "hybrid_search", "mfl_query",
            "fault_history_query", "graph_traversal",
        ),
        # FTA/RBD are structural subqueries selected explicitly by the
        # planner.  Merely selecting broad impact recall must not turn into an
        # unconditional seven-document scan; direct fta_query/rbd_query calls
        # still remain authorized above.
        "hybrid_allowed": (
            "function_locator", "fmeca_query", "hmc_query", "hybrid_search",
            "mfl_query", "fault_history_query", "graph_traversal",
        ),
    },
    "maintenance_query": {
        "fallback": ("hmc_query", "mfl_query", "hybrid_search"),
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "mfl_query",
            "fault_history_query", "hybrid_search", "graph_traversal",
        ),
    },
    "reliability_query": {
        "fallback": ("rbd_query", "hybrid_search"),
        "allowed": (
            "function_locator", "rbd_query", "rbd_calculator", "hybrid_search",
            "graph_traversal",
        ),
    },
    "history_query": {
        "fallback": ("fault_history_query", "hybrid_search"),
        "allowed": (
            "fault_history_query", "hmc_query", "fmeca_query", "mfl_query",
            "hybrid_search", "graph_traversal",
        ),
    },
    "hybrid": {
        "fallback": ("hybrid_search",),
        "allowed": tuple(sorted(CANONICAL_TOOL_NAMES - {"fta_analyzer", "rbd_calculator"})),
    },
}

QUERY_TOOL_NAMES = {
    "function_locator",
    "hybrid_search",
    "hmc_query",
    "fmeca_query",
    "fta_query",
    "rbd_query",
    "mfl_query",
    "fault_history_query",
}

_CALCULATION_REQUEST_FTA_CUT_SETS = "fta_cut_sets"
_CALCULATION_REQUEST_FTA_PROBABILITY = "fta_probability"
_CALCULATION_REQUEST_RBD_RELIABILITY = "rbd_reliability"

_HMC_COMPACT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<lcn>\d{6,10})\s*[-_/]?\s*"
    r"(?P<hamc>\*\*|[A-Za-z0-9]{1,4})\s*[-_/]?\s*"
    r"(?P<hsc>\d{1,6})(?![A-Za-z0-9])"
)
_HMC_FIELD_LABEL = re.compile(
    r"(?<![A-Za-z0-9_])(?:LCN|HAMC|HSC)(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
_HMC_FIELD_PATTERNS = {
    "lcn": re.compile(
        r"(?<![A-Za-z0-9_])LCN\s*(?:[:=]\s*)?(\d{6,10})(?!\d)",
        flags=re.IGNORECASE,
    ),
    "hamc": re.compile(
        r"(?<![A-Za-z0-9_])HAMC\s*(?:[:=]\s*)?"
        r"(?!LCN\b|HAMC\b|HSC\b)(\*\*|[A-Za-z0-9]{1,4})(?![A-Za-z0-9*])",
        flags=re.IGNORECASE,
    ),
    "hsc": re.compile(
        r"(?<![A-Za-z0-9_])HSC\s*(?:[:=]\s*)?(\d{1,6})(?!\d)",
        flags=re.IGNORECASE,
    ),
}


class ToolPolicyDecision(TypedDict):
    intent: str
    query_targets: List[Dict[str, Any]]
    calculation_requests: List[str]
    authorized_tools: List[str]
    allowed_tools: List[str]
    tools_plan: List[Dict[str, Any]]
    authorized_doc_types: List[str]
    hybrid_doc_types: List[str]
    rejected_tools: List[str]
    notes: List[str]
    # User explicitly named one or more document families to inspect.  This
    # is separate from the semantic intent and is propagated to answer output
    # so an empty requested source cannot disappear silently.
    requested_source_tools: List[str]
    source_coverage_fields: List[str]


def _canonical_tool(value: Any) -> str:
    name = str(value or "").strip()
    return TOOL_ALIASES.get(name, name)


def detect_explicit_source_tools(text: Any) -> List[str]:
    """Return document tools explicitly named in the user's wording.

    This is not an intent router and does not infer a source from a symptom.
    It only honours an exact name such as ``RBD``/``可靠性框图`` that the user
    wrote themselves.  Scope routing handles pure capability questions before
    this policy runs, so a real knowledge request such as “去查 RBD” reaches
    the corresponding read tool.
    """

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    result: List[str] = []
    for tool, pattern in _EXPLICIT_SOURCE_TOOL_PATTERNS:
        if pattern.search(normalized) and tool not in result:
            result.append(tool)
    return result


def _looks_like_definition_request(text: str) -> bool:
    return bool(re.search(r"是什么意思|什么含义|介绍一下|解释概念|定义是什么", text))


def detect_calculation_requests(text: Any) -> List[str]:
    """Authorize deterministic calculations from the raw user text only.

    Query rewrites and planner keywords are intentionally excluded: an LLM may
    broaden recall, but it cannot grant itself permission to calculate.
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    requests: List[str] = []
    definition_only = _looks_like_definition_request(value)

    cut_set_request = bool(
        re.search(r"(?:计算|求|列出|找出).{0,8}最小割集|最小割集.{0,6}(?:有哪些|是什么组合)", value)
        or re.search(
            r"(?:最少|最小).{0,8}(?:哪些|哪几种)?.{0,6}(?:原因|故障|事件).{0,12}(?:组合|同时发生)",
            value,
        )
        or re.search(
            r"(?:哪些|哪几种).{0,12}(?:原因|故障|事件).{0,12}(?:组合起来|同时发生).{0,8}(?:导致|引起|造成)",
            value,
        )
    )
    if cut_set_request and not definition_only:
        requests.append(_CALCULATION_REQUEST_FTA_CUT_SETS)

    asks_documented_parameter = bool(re.search(r"(?:文档|资料|手册).{0,12}(?:记载|记录|给出).{0,12}(?:参数|概率)", value))
    probability_request = bool(
        re.search(
            r"(?:计算|求|算出).{0,12}(?:顶事件|顶层故障|系统故障).{0,12}(?:概率|可能性)",
            value,
        )
        or re.search(
            r"(?:顶事件|顶层故障|系统故障).{0,12}(?:发生概率|可能性).{0,6}(?:多少|多大)",
            value,
        )
    )
    if probability_request and not definition_only and not asks_documented_parameter:
        requests.append(_CALCULATION_REQUEST_FTA_PROBABILITY)

    reliability_subject = bool(
        re.search(
            r"可靠度|系统可靠性|成功概率|任务成功率|任务.{0,40}(?:成功|完成)|正常工作|完成任务|持续运行|继续运行",
            value,
        )
    )
    numeric_request = bool(
        re.search(r"重新计算|复算|计算|求|算出|多少|多大|概率|可能性|几率|数值|最低|最高", value)
    )
    explicit_recompute = bool(re.search(r"重新计算|复算|重算|按.{0,20}(?:参数|失效率|故障率).{0,8}计算", value))
    if (
        reliability_subject
        and numeric_request
        and not definition_only
        and (not asks_documented_parameter or explicit_recompute)
    ):
        requests.append(_CALCULATION_REQUEST_RBD_RELIABILITY)
    return list(dict.fromkeys(requests))


def extract_hmc_components(text: Any) -> Dict[str, str]:
    """Parse labelled HMC fields or a complete compact LCN/HAMC/HSC code.

    Labelled ``LCN + HSC`` is intentionally retained without inventing a
    missing HAMC.  Once any field label is present, compact parsing is not
    attempted: otherwise ``LCN 91810001 HSC 0001`` is misread as if the word
    ``HSC`` were the HAMC value.
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("—", "-").replace("–", "-")
    labelled: Dict[str, str] = {}
    for key, pattern in _HMC_FIELD_PATTERNS.items():
        match = pattern.search(value)
        if match:
            labelled[key] = match.group(1).upper()

    if "hsc" in labelled and labelled["hsc"].isdigit() and len(labelled["hsc"]) < 4:
        labelled["hsc"] = labelled["hsc"].zfill(4)
    if "lcn" in labelled and "hsc" in labelled:
        if "hamc" not in labelled:
            return labelled
        labelled["hmc_code"] = f"{labelled['lcn']}{labelled['hamc']}{labelled['hsc']}"
        return labelled

    # A label makes the input field-oriented.  Fail closed instead of letting
    # the permissive compact expression consume another label as a code part.
    if _HMC_FIELD_LABEL.search(value):
        return {}

    match = _HMC_COMPACT_PATTERN.search(value)
    if not match:
        return {}
    result = {key: item.upper() for key, item in match.groupdict().items()}
    if result["hsc"].isdigit() and len(result["hsc"]) < 4:
        result["hsc"] = result["hsc"].zfill(4)
    result["hmc_code"] = f"{result['lcn']}{result['hamc']}{result['hsc']}"
    return result


def _normalize_proposals(proposed_plan: Any) -> List[Dict[str, str]]:
    if not isinstance(proposed_plan, list):
        return []
    normalized: List[Dict[str, str]] = []
    seen = set()
    for item in proposed_plan:
        if isinstance(item, str):
            tool = _canonical_tool(item)
            reason = "规划模型建议"
        elif isinstance(item, dict):
            tool = _canonical_tool(item.get("tool"))
            reason = str(item.get("reason") or "规划模型建议").strip()
        else:
            continue
        if not tool or tool in seen:
            continue
        seen.add(tool)
        normalized.append({"tool": tool, "reason": reason})
    return normalized


def _normalize_query_targets(
    *,
    intent: str,
    query_targets: Any,
    raw_user_query: str,
) -> tuple[str, List[Dict[str, Any]]]:
    targets: List[Dict[str, Any]] = []
    by_intent: Dict[str, Dict[str, Any]] = {}

    def add(value: Any, source: str, reason: str) -> None:
        if isinstance(value, dict):
            raw_intent = value.get("intent")
            # Source is assigned by the caller; planner JSON cannot promote
            # itself to a deterministic or authoritative classification.
            source_value = source
            reason_value = str(value.get("reason") or reason)
            required_fields = normalize_answer_fields(
                value.get("required_fields") or value.get("answer_fields")
            )
        else:
            raw_intent = value
            source_value = source
            reason_value = reason
            required_fields = []
        normalized = INTENT_ALIASES.get(str(raw_intent or "").strip(), str(raw_intent or "").strip())
        if normalized not in INTENT_POLICY:
            return
        if normalized in by_intent:
            current = by_intent[normalized]
            merged_fields = normalize_answer_fields(
                [*(current.get("required_fields") or []), *required_fields]
            )
            if merged_fields:
                current["required_fields"] = merged_fields
            return
        item = {"intent": normalized, "source": source_value, "reason": reason_value}
        if required_fields:
            item["required_fields"] = required_fields
        by_intent[normalized] = item
        targets.append(item)

    normalized_intent = INTENT_ALIASES.get(str(intent or "").strip(), str(intent or "").strip())
    if normalized_intent not in INTENT_POLICY:
        normalized_intent = "hybrid"
    add(normalized_intent, "planner", "规划模型识别的主要信息需求")
    for item in query_targets if isinstance(query_targets, list) else []:
        add(item, "planner", "规划模型识别的补充信息需求")

    structured_hmc = extract_hmc_components(raw_user_query)
    if structured_hmc:
        add("exact_code", "structured_code", "原始问题包含可结构化查询的 HMC 字段")

    # A broad compatibility intent must not expand a valid structured target
    # back to all seven sources.  The planner's first specific target becomes
    # the answer shape while all targets still contribute required tools.
    specific = [
        item["intent"]
        for item in targets
        if item["intent"] not in {"symptom_diagnosis", "hybrid", "exact_code"}
    ]
    if specific or structured_hmc:
        targets = [
            item
            for item in targets
            if item["intent"] not in {"symptom_diagnosis", "hybrid"}
        ]
        by_intent.pop("symptom_diagnosis", None)
        by_intent.pop("hybrid", None)
    if normalized_intent in {"symptom_diagnosis", "hybrid"} and specific:
        normalized_intent = specific[0]
    elif structured_hmc and not specific and normalized_intent in {"symptom_diagnosis", "hybrid"}:
        normalized_intent = "exact_code"

    # Keep the compatibility primary intent first; the remaining targets retain
    # discovery order so the UI can explain why each source was queried.
    primary_item = by_intent.get(normalized_intent)
    if primary_item is None:
        add(normalized_intent, "policy", "工具策略选择的兼容主目标")
        primary_item = by_intent[normalized_intent]
    targets = [primary_item, *(item for item in targets if item is not primary_item)]
    return normalized_intent, targets


_TOOL_ORDER = (
    "function_locator",
    "hmc_query",
    "fmeca_query",
    "fta_query",
    "rbd_query",
    "mfl_query",
    "fault_history_query",
    "hybrid_search",
    "graph_traversal",
    "fta_analyzer",
    "rbd_calculator",
    "evidence_synthesis",
)


def _tool_targets(tool: str, target_intents: Sequence[str]) -> List[str]:
    return [
        target
        for target in target_intents
        if tool in INTENT_POLICY[target]["fallback"] or tool in INTENT_POLICY[target]["allowed"]
    ]


def resolve_tool_policy(
    *,
    intent: str,
    proposed_plan: Any = None,
    query: str = "",
    type_keywords: Any = None,
    query_targets: Any = None,
    raw_user_query: Optional[str] = None,
    planner_failed: bool = False,
    coverage_scope: str = "unknown",
    collection_member_kind: str = "unknown",
    required_answer_fields: Any = None,
    diagnostic_closure: Any = None,
    relation_expansion_required: bool = False,
) -> ToolPolicyDecision:
    """将规划模型的结构化语义结果转换为可执行白名单。

    本函数不从自然语言猜测意图。它只校验 planner 给出的目标、识别可
    机械解析的 HMC 字段，并用原始用户文字做保守的数值计算授权。
    ``allowed_tools`` 是唯一执行依据；``tools_plan`` 只用于界面展示。
    """
    original_normalized_intent = INTENT_ALIASES.get(
        str(intent or "").strip(), str(intent or "").strip()
    )
    notes: List[str] = []
    if original_normalized_intent not in INTENT_POLICY:
        notes.append(f"未知意图 {original_normalized_intent or '<empty>'}，已降级为 hybrid")
        original_normalized_intent = "hybrid"

    # ``query`` and ``type_keywords`` remain in the public signature for saved
    # agents and older callers, but they never classify semantic intent here.
    authorization_text = str(query if raw_user_query is None else raw_user_query)
    normalized_intent, normalized_targets = _normalize_query_targets(
        intent=original_normalized_intent,
        query_targets=query_targets,
        raw_user_query=authorization_text,
    )
    target_intents = [item["intent"] for item in normalized_targets]
    requested_source_tools = detect_explicit_source_tools(authorization_text)
    source_coverage_fields = normalize_answer_fields(
        [
            field
            for tool in requested_source_tools
            for field in _SOURCE_COVERAGE_FIELDS.get(tool, ())
        ]
    )
    contract_fields = ensure_intent_answer_fields(
        [
            required_answer_fields or [],
            source_coverage_fields,
            *(
                item.get("required_fields") or []
                for item in normalized_targets
                if isinstance(item, dict)
            ),
        ],
        intents=target_intents,
    )
    if contract_fields and normalized_targets:
        normalized_targets[0]["required_fields"] = list(contract_fields)
    if "exact_code" in target_intents and normalized_intent != "exact_code":
        notes.append("检测到 HMC 字段，保留用户的主要问题目标并追加结构化 HMC 查询")

    allowed_by_intent: set[str] = set()
    hybrid_allowed_by_intent: set[str] = set()
    fallback_tools: List[str] = []
    for target in target_intents:
        policy = INTENT_POLICY[target]
        allowed_by_intent.update(policy["allowed"])
        hybrid_allowed_by_intent.update(policy.get("hybrid_allowed", policy["allowed"]))
        for tool in policy["fallback"]:
            if tool not in fallback_tools:
                fallback_tools.append(tool)
    allowed_by_intent.add("evidence_synthesis")

    if requested_source_tools:
        # The user has named this source themselves.  Preserve that coverage
        # even when the semantic target is narrower (for example a cause
        # question that explicitly asks to inspect RBD).  This grants the read
        # query only; deterministic calculators retain their own raw-query
        # numeric authorization below.
        allowed_by_intent.update(requested_source_tools)
        hybrid_allowed_by_intent.update(requested_source_tools)
        notes.append(
            "用户明确要求回查资料：" + "、".join(
                DOC_TYPE_BY_TOOL.get(tool, tool) for tool in requested_source_tools
            )
        )

    # A complete-collection request is a response contract, not a broader
    # diagnostic guess.  PostgreSQL Function Locator + FMECA rows are the
    # *only* sources that can prove membership closure.  Do not however turn
    # that proof requirement into a blanket ban on another explicitly planned
    # direct source: a question may ask for a complete FMECA set and also ask
    # how those already-listed items are repaired.  Extra sources can add
    # context, but never contribute to the N-of-N count.
    collection_contract = (
        str(coverage_scope or "").strip() == "complete_collection"
        and str(collection_member_kind or "").strip() in {"failure_mode", "cause"}
    )
    if collection_contract:
        allowed_by_intent.update({"function_locator", "fmeca_query"})
        # Search copies never establish collection membership.  Keep their
        # scope narrow even when another accepted direct tool is present.
        hybrid_allowed_by_intent.intersection_update(
            {"function_locator", "fmeca_query"}
        )
        fallback_tools = list(dict.fromkeys([
            "function_locator", "fmeca_query", *fallback_tools,
        ]))
        notes.append("完整集合成员仅由架构身份与FMECA权威记录闭合；其他资料不参与计数")

    diagnostic_closure_required = bool(
        isinstance(diagnostic_closure, Mapping)
        and diagnostic_closure.get("required") is True
    )
    focused_field_contract = is_focused_answer_contract(
        fields=contract_fields,
        coverage_scope=str(coverage_scope or "").strip(),
        intents=target_intents,
        diagnostic_closure=diagnostic_closure_required,
    )
    topology_entity_resolution = bool(
        "reliability_query" in target_intents
        and "rbd_single_failure_boundary" in contract_fields
    )
    if focused_field_contract:
        contract_tools = tools_for_answer_fields(contract_fields)
        direct_contract_tools = {
            tool for tool in contract_tools if tool in CANONICAL_TOOL_NAMES
        }
        # A validated answer field is a semantic request for the tool family
        # that owns that field.  Do not let an otherwise correct field such as
        # ``maintenance_locator`` be discarded merely because a compact model
        # labelled the question ``function_location`` instead of
        # ``maintenance_query``.  Calculators are deliberately absent from
        # answer-field mappings and continue to require raw-query authority.
        allowed_by_intent.update(direct_contract_tools)
        hybrid_allowed_by_intent.update(direct_contract_tools)
        # A single-path capability question needs one authoritative object
        # identity before FMECA and RBD can be compared.  Function Locator is
        # an identity resolver here, not an additional answer dimension; keep
        # it inside the focused boundary instead of falling back to name-only
        # matches in two unrelated document families.
        if topology_entity_resolution:
            direct_contract_tools.add("function_locator")
        # FTA/RBD are source structures.  A Function Locator lookup is allowed
        # as a bounded identity bridge when the planner already proposed it;
        # it avoids losing an explicit module name merely because the final
        # source obligation is a diagram.
        if {"fta_query", "rbd_query"} & set(requested_source_tools):
            direct_contract_tools.add("function_locator")
        direct_contract_tools.add("evidence_synthesis")
        allowed_by_intent.intersection_update(direct_contract_tools)
        # Hybrid retrieval is limited to the document families that can carry
        # the requested field, rather than inheriting the broad intent scope.
        hybrid_allowed_by_intent.intersection_update(set(contract_tools))
        contract_fallback = [
            tool
            for tool in contract_tools
            if tool in QUERY_TOOL_NAMES and tool in allowed_by_intent
        ]
        if contract_fallback:
            # Prefer PostgreSQL/domain tools first.  OpenSearch remains an
            # authorized adaptive fallback when the direct field lookup is
            # empty; running both up front widens a scalar follow-up into an
            # unnecessary candidate list.
            direct_fallback = [
                tool for tool in contract_fallback if tool != "hybrid_search"
            ]
            fallback_tools = direct_fallback or contract_fallback
        notes.append(
            "focused_detail 已按必答字段收窄工具范围："
            + "、".join(contract_fields)
        )

    # The target model has already established that the answer needs a
    # cross-document relation or multi-hop expansion.  This is deliberately a
    # typed planner signal rather than a raw-word trigger for “关系/路径/图谱”.
    # Keep the sources needed to bind an authoritative graph seed and, for a
    # causal/propagation question, the FTA structure that explains the chain.
    # HugeGraph remains a projection and is still executed only after those
    # sources provide stable business IDs.
    if relation_expansion_required:
        relation_tools = {"function_locator", "graph_traversal"}
        if set(target_intents) & {
            "cause_analysis", "symptom_diagnosis", "impact_analysis",
        }:
            relation_tools.add("fta_query")
        allowed_by_intent.update(relation_tools)
        # A focused scalar field must not erase the authoritative bridge the
        # model says is necessary to answer it completely.
        hybrid_allowed_by_intent.update(
            tool for tool in relation_tools if tool != "graph_traversal"
        )
        notes.append("语义关系合同已保留实体定位、结构来源和多跳关系查询")

    calculation_requests = detect_calculation_requests(authorization_text)
    explicit_fta_calculation = any(value.startswith("fta_") for value in calculation_requests)
    explicit_rbd_calculation = _CALCULATION_REQUEST_RBD_RELIABILITY in calculation_requests
    forced_tools: List[str] = []
    reliability_target = "reliability_query" in target_intents
    if topology_entity_resolution and "function_locator" in allowed_by_intent:
        forced_tools.append("function_locator")
        notes.append("单路失效边界查询先通过 Function Locator 绑定权威对象身份")
    if reliability_target and "rbd_query" in allowed_by_intent:
        # ``reliability_query`` is already a semantic, model-authored target.
        # Requiring its authoritative structure lookup is contract completion,
        # not a raw-text keyword rule.  It also prevents an impact-first plan
        # from silently answering a named-path failure with FMECA alone.
        forced_tools.append("rbd_query")
        notes.append("可靠性目标已保留 RBD 成功判据与单路失效边界查询")
    maintenance_contract_fields = _MFL_CONTRACT_FIELDS & set(contract_fields)
    if maintenance_contract_fields and "mfl_query" in allowed_by_intent:
        # Same contract-completion pattern as the RBD rule above: the target
        # model has already committed to maintenance fields whose authoritative
        # owner is MFL.  A compact plan that answers them with hybrid recall
        # alone would leave the documented 板级定位/处置清单 absent in the
        # first pass; adaptive completion cannot repair a source that was
        # never authorized to run.
        if "mfl_query" not in forced_tools:
            forced_tools.append("mfl_query")
            notes.append(
                "维修合同已保留 MFL 权威定位查询："
                + "、".join(sorted(maintenance_contract_fields))
            )
    for tool in requested_source_tools:
        if tool in allowed_by_intent and tool not in forced_tools:
            forced_tools.append(tool)
    structured_hmc = extract_hmc_components(authorization_text)
    if structured_hmc:
        forced_tools.append("hmc_query")
    # A syntactically explicit business ID can be resolved exactly without
    # guessing its document family from a prefix or trusting the planner to
    # choose the right table. Probe the bounded identity-capable PostgreSQL
    # sources; only an exact stored ID is allowed to survive downstream.
    record_identity_resolution = (
        "record_identity" in contract_fields
        and "entity_profile" in target_intents
        and bool(extract_business_codes(authorization_text))
        and not structured_hmc
    )
    if record_identity_resolution:
        for tool in (
            "function_locator",
            "fmeca_query",
            "mfl_query",
            "fault_history_query",
            "hmc_query",
        ):
            if tool in allowed_by_intent and tool not in forced_tools:
                forced_tools.append(tool)
        notes.append(
            "显式业务ID按权威身份列跨表精确解析，不依据前缀猜测文档类型"
        )
    if collection_contract:
        for tool in ("function_locator", "fmeca_query"):
            if tool not in forced_tools:
                forced_tools.append(tool)
        # 集合主体（模块/功能）已由上方确定性身份查询绑定出业务 ID；以该
        # ID 为种子的图投影聚合跨文档成员视图。闭合计数仍只来自 FMECA
        # 权威记录，图结果不参与 N-of-N 计数，也不得凭图补造成员。
        if "graph_traversal" in allowed_by_intent and "graph_traversal" not in forced_tools:
            forced_tools.append("graph_traversal")
            notes.append("完整集合成员已附加跨文档图聚合视图（不参与闭合计数）")

    if relation_expansion_required:
        for tool in ("function_locator", "fta_query", "graph_traversal"):
            if tool in allowed_by_intent and tool not in forced_tools:
                forced_tools.append(tool)

    if explicit_fta_calculation:
        allowed_by_intent.update({"fta_query", "fta_analyzer"})
        for tool in ("fta_query", "fta_analyzer"):
            if tool not in forced_tools:
                forced_tools.append(tool)
    if explicit_rbd_calculation:
        allowed_by_intent.update({"rbd_query", "rbd_calculator"})
        for tool in ("rbd_query", "rbd_calculator"):
            if tool not in forced_tools:
                forced_tools.append(tool)

    # 规划 JSON 已被标记为失败时，其中的残留工具名也不再可信。
    proposals = [] if planner_failed else _normalize_proposals(proposed_plan)
    accepted: List[Dict[str, str]] = []
    rejected: List[str] = []
    for item in proposals:
        tool = item["tool"]
        if tool not in CANONICAL_TOOL_NAMES or tool not in allowed_by_intent:
            rejected.append(tool)
            continue
        if tool == "fta_analyzer" and not explicit_fta_calculation:
            rejected.append(tool)
            continue
        if tool == "rbd_calculator" and not explicit_rbd_calculation:
            rejected.append(tool)
            continue
        accepted.append(item)

    if focused_field_contract and any(
        item["tool"] in QUERY_TOOL_NAMES and item["tool"] != "hybrid_search"
        for item in accepted
    ):
        accepted = [item for item in accepted if item["tool"] != "hybrid_search"]
        notes.append("聚焦字段先查权威结构化记录；混合检索仅在未命中时补查")

    # 完整 HMC 的单一精确查询必须先走 PostgreSQL 权威字段匹配。
    # 即使规划模型同时选择了 OpenSearch，也将其保留为后续授权能力，
    # 仅在 HMC 未命中时由自适应阶段补查，避免首轮无条件扩大召回。
    if structured_hmc and set(target_intents) == {"exact_code"}:
        deferred = [
            item["tool"]
            for item in accepted
            if item["tool"] in QUERY_TOOL_NAMES and item["tool"] != "hmc_query"
        ]
        if deferred:
            accepted = [
                item
                for item in accepted
                if item["tool"] not in set(deferred)
            ]
            notes.append(
                "完整 HMC 先执行权威精确查询；宽召回仅在未命中时自适应补查"
            )

    plan_by_tool: Dict[str, Dict[str, Any]] = {item["tool"]: dict(item) for item in accepted}

    selected_query_tools = {
        item["tool"] for item in accepted if item["tool"] in QUERY_TOOL_NAMES
    }
    if planner_failed or not selected_query_tools.union(
        tool for tool in forced_tools if tool in QUERY_TOOL_NAMES
    ):
        for tool in fallback_tools:
            if tool not in forced_tools:
                forced_tools.append(tool)
        notes.append(
            "规划模型未给出可执行查询，已按当前信息目标使用最小安全召回"
        )

    for tool in forced_tools:
        canonical = _canonical_tool(tool)
        if canonical in CANONICAL_TOOL_NAMES:
            plan_by_tool.setdefault(
                canonical,
                {
                    "tool": canonical,
                    "reason": (
                        "用户明确要求回查该类资料"
                        if canonical in requested_source_tools
                        else (
                            "原始问题包含可结构化字段或明确计算请求"
                        if canonical in {"hmc_query", "fta_analyzer", "rbd_calculator"}
                        else (
                            "可靠性目标必须回读 RBD 结构和成功判据"
                            if canonical == "rbd_query" and reliability_target
                            else (
                                "显式业务ID需在权威结构化身份列中精确解析"
                                if record_identity_resolution
                                else (
                                    "完整集合合同要求回查架构身份与权威FMECA记录"
                                    if collection_contract and canonical in {"function_locator", "fmeca_query"}
                                    else (
                                        "完整集合合同附加跨文档图聚合视图"
                                        if collection_contract and canonical == "graph_traversal"
                                        else (
                                            "维修合同必须回读 MFL 权威定位与处置清单"
                                            if canonical == "mfl_query" and maintenance_contract_fields
                                            else "规划不可用时的最小安全召回"
                                        )
                                    )
                                )
                            )
                        )
                        )
                    ),
                    "targets": _tool_targets(canonical, target_intents),
                },
            )

    # 证据汇总是内部固定阶段，不允许规划模型跳过。
    plan_by_tool.setdefault(
        "evidence_synthesis",
        {"tool": "evidence_synthesis", "reason": "汇总已检索证据并产生可审核候选"},
    )

    requested_tools = {
        _canonical_tool(tool)
        for tool in [*forced_tools, *(item["tool"] for item in accepted), "evidence_synthesis"]
    }
    ordered_tools = [tool for tool in _TOOL_ORDER if tool in requested_tools and tool in plan_by_tool]

    authorized_doc_types: List[str] = []
    for tool in _TOOL_ORDER:
        if tool not in allowed_by_intent:
            continue
        doc_type = DOC_TYPE_BY_TOOL.get(tool)
        if doc_type and doc_type not in authorized_doc_types:
            authorized_doc_types.append(doc_type)

    hybrid_doc_types = (
        [
            doc_type
            for tool in _TOOL_ORDER
            if tool in hybrid_allowed_by_intent
            for doc_type in [DOC_TYPE_BY_TOOL.get(tool)]
            if doc_type
        ]
        if "hybrid_search" in ordered_tools
        else []
    )
    authorized_tools = [
        tool for tool in _TOOL_ORDER if tool in allowed_by_intent
    ]

    return {
        "intent": normalized_intent,
        "query_targets": normalized_targets,
        "calculation_requests": calculation_requests,
        "authorized_tools": authorized_tools,
        "allowed_tools": ordered_tools,
        "tools_plan": [plan_by_tool[tool] for tool in ordered_tools],
        "authorized_doc_types": authorized_doc_types,
        "hybrid_doc_types": hybrid_doc_types,
        "rejected_tools": list(dict.fromkeys(rejected)),
        "notes": notes,
        "requested_source_tools": requested_source_tools,
        "source_coverage_fields": source_coverage_fields,
    }
