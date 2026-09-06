"""ICP 智能体的确定性工具授权与执行策略。

LLM 负责提出最小工具计划；本模块负责限制授权上限、校验实际执行集、
约束 OpenSearch 文档类型，并防止模型在用户没有要求数值计算时启动
FTA/RBD 计算器。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict

from backend.agent.answer_contract import (
    RBD_ANSWER_FIELDS,
    ensure_intent_answer_fields,
    is_focused_answer_contract,
    normalize_diagnostic_closure,
    normalize_answer_fields,
    tools_for_answer_fields,
)
from backend.agent.query_understanding import extract_business_codes
from backend.agent.hmc_identity import extract_hmc_components, normalize_typed_hmc_components
from backend.agent.user_context_binding import selected_user_hmc_components


DOC_TYPE_BY_TOOL = {
    "function_locator": "function_locator",
    "hmc_query": "hmc",
    "fmeca_query": "fmeca",
    "fta_query": "fta",
    "rbd_query": "rbd",
    "mfl_query": "mfl",
    "fault_history_query": "fault_history",
}

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

# ``allowed`` 只是模型可选的意图安全上限，不是默认工具套餐。
# 实际执行必须由模型显式计划、字段/来源合同或严格的确定性字段授权。
INTENT_POLICY: Dict[str, Dict[str, Sequence[str]]] = {
    "entity_profile": {
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
        "allowed": ("function_locator", "hybrid_search", "graph_traversal"),
    },
    "function_location": {
        "allowed": ("function_locator", "hybrid_search", "graph_traversal"),
    },
    "exact_code": {
        "allowed": ("hmc_query", "hybrid_search"),
    },
    "symptom_diagnosis": {
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "fault_history_query",
            "hybrid_search", "fta_query", "mfl_query", "graph_traversal",
        ),
    },
    "cause_analysis": {
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
        "allowed": (
            "function_locator", "hmc_query", "fmeca_query", "mfl_query",
            "fault_history_query", "hybrid_search", "graph_traversal",
        ),
    },
    "reliability_query": {
        "allowed": (
            "function_locator", "rbd_query", "rbd_calculator", "hybrid_search",
            "graph_traversal",
        ),
    },
    "history_query": {
        "allowed": (
            "fault_history_query", "hmc_query", "fmeca_query", "mfl_query",
            "hybrid_search", "graph_traversal",
        ),
    },
    "hybrid": {
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

# Calculation meaning is classified by the two planner model stages.  Code
# deliberately validates only this small protocol and never tries to enumerate
# all natural-language ways a user may request a calculation.
CALCULATION_OPERATION_TARGETS = {
    _CALCULATION_REQUEST_FTA_CUT_SETS: "fta",
    _CALCULATION_REQUEST_FTA_PROBABILITY: "fta",
    _CALCULATION_REQUEST_RBD_RELIABILITY: "rbd",
}
CALCULATION_AUTHORIZATION_MODES = {"calculate", "explain", "read_document"}



class ToolPolicyDecision(TypedDict):
    intent: str
    query_targets: List[Dict[str, Any]]
    calculation_authorizations: List[Dict[str, str]]
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
    requested_sources: List[Dict[str, Any]]
    source_obligations: List[Dict[str, Any]]
    relation_expansion: Dict[str, Any]


def _canonical_tool(value: Any) -> str:
    name = str(value or "").strip()
    return TOOL_ALIASES.get(name, name)


_TOOL_BY_DOC_TYPE = {doc_type: tool for tool, doc_type in DOC_TYPE_BY_TOOL.items()}
_SOURCE_QUESTION_ROLES = frozenset(
    {"evidence", "capability", "limitation", "comparison"}
)

# Relation expansion is an execution authorization, not a prose hint. Keep its
# semantic kinds closed while allowing the model to understand arbitrary user
# wording before emitting one of these stable values.
RELATION_KINDS = frozenset(
    {"relation_path", "structural_cause_chain", "impact_propagation"}
)
RELATION_REFERENCE_MODES = frozenset({"current_query", "selected_anchor"})
RELATION_TOOLS: Dict[str, frozenset[str]] = {
    "relation_path": frozenset({"function_locator", "graph_traversal"}),
    "structural_cause_chain": frozenset(
        {"function_locator", "fta_query", "graph_traversal"}
    ),
    "impact_propagation": frozenset(
        {"function_locator", "fmeca_query", "fta_query", "graph_traversal"}
    ),
}


def normalize_relation_expansion_contract(
    value: Any,
    *,
    raw_user_query: Any,
    offered_anchor_ids: Any = (),
) -> Dict[str, Any]:
    """Validate the only contract that may authorize graph traversal.

    ``needed`` alone is deliberately insufficient. A positive contract must
    name one closed relation kind and carry a literal, contiguous span from the
    current user turn. A deictic follow-up additionally names one or more
    server-offered citation anchors; planner text and prior answer prose cannot
    mint those identities.

    A well-formed negative decision is retained so an earlier semantic stage
    can explicitly deny a later retrieval planner's attempt to widen scope.
    """

    if not isinstance(value, Mapping):
        return {}
    needed = value.get("needed")
    if needed is False:
        return {
            "needed": False,
            "relation_kind": "",
            "reference_mode": "",
            "basis": "",
            "selected_anchor_ids": [],
            "reason": "",
        }
    if needed is not True:
        return {}

    relation_kind = str(value.get("relation_kind") or "").strip()
    reference_mode = str(value.get("reference_mode") or "").strip()
    basis = str(value.get("basis") or "").strip()[:500]
    query = str(raw_user_query or "")
    if (
        relation_kind not in RELATION_KINDS
        or reference_mode not in RELATION_REFERENCE_MODES
        or not basis
        or basis not in query
    ):
        return {}

    result: Dict[str, Any] = {
        "needed": True,
        "relation_kind": relation_kind,
        "reference_mode": reference_mode,
        "basis": basis,
        "selected_anchor_ids": [],
        "reason": " ".join(str(value.get("reason") or "").split())[:500],
    }
    if reference_mode == "current_query":
        return result

    offered = {
        str(anchor_id or "").strip()
        for anchor_id in (offered_anchor_ids or [])
        if str(anchor_id or "").strip()
    }
    selected: List[str] = []
    raw_selected = value.get("selected_anchor_ids")
    if not isinstance(raw_selected, (list, tuple, set, frozenset)):
        return {}
    for anchor_id in raw_selected:
        normalized = str(anchor_id or "").strip()
        if normalized in offered and normalized not in selected:
            selected.append(normalized)
        if len(selected) >= 12:
            break
    if not selected:
        return {}
    result["selected_anchor_ids"] = selected
    return result


def _normalize_requested_sources(
    value: Any,
    *,
    raw_user_query: Any,
) -> List[Dict[str, Any]]:
    """Validate model-authored source requests; never infer them from wording."""

    if not isinstance(value, list):
        return []
    query = str(raw_user_query or "")
    result: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for raw in value[:16]:
        if not isinstance(raw, Mapping):
            continue
        doc_type = str(raw.get("doc_type") or "").strip()
        tool = _TOOL_BY_DOC_TYPE.get(doc_type, "")
        role = str(raw.get("question_role") or "").strip()
        basis = str(raw.get("basis") or "").strip()[:500]
        if not tool or role not in _SOURCE_QUESTION_ROLES or not basis or basis not in query:
            continue
        fields = normalize_answer_fields(raw.get("requested_fields"))
        identity = (doc_type, basis, role, tuple(fields))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            {
                "doc_type": doc_type,
                "basis": basis,
                "requested_fields": fields,
                "question_role": role,
            }
        )
    return result


def normalize_calculation_authorizations(
    value: Any,
    *,
    raw_user_query: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Validate the closed calculation protocol without classifying language.

    When ``raw_user_query`` is supplied, every basis must be one literal,
    contiguous span of the current user turn.  NFKC or fuzzy matching is not
    used here: a rewrite, prior answer, or bag of matching words must not become
    provenance for a calculation request.
    """

    if not isinstance(value, list):
        return []
    current_query = None if raw_user_query is None else str(raw_user_query)
    normalized: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            continue
        operation = str(raw_item.get("operation") or "").strip()
        mode = str(raw_item.get("mode") or "").strip()
        target = str(raw_item.get("target") or "").strip()
        basis = str(raw_item.get("basis") or "").strip()
        if operation not in CALCULATION_OPERATION_TARGETS:
            continue
        if mode not in CALCULATION_AUTHORIZATION_MODES:
            continue
        if target != CALCULATION_OPERATION_TARGETS[operation]:
            continue
        if not basis or (current_query is not None and basis not in current_query):
            continue
        identity = (operation, mode, target, basis)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "operation": operation,
                "mode": mode,
                "target": target,
                "basis": basis,
            }
        )
    # Ambiguous modes never let calculate win, even from the sole semantic
    # owner. Keep this invariant in the shared validator for every caller.
    modes: Dict[str, set[str]] = {}
    for item in normalized:
        modes.setdefault(item["operation"], set()).add(item["mode"])
    return [item for item in normalized if len(modes[item["operation"]]) == 1]


def calculation_requests_from_authorizations(
    value: Any,
    *,
    raw_user_query: str,
) -> List[str]:
    """Project validated ``calculate`` objects to legacy answer field names."""

    return list(
        dict.fromkeys(
            item["operation"]
            for item in normalize_calculation_authorizations(
                value,
                raw_user_query=raw_user_query,
            )
            if item["mode"] == "calculate"
        )
    )




def grounded_typed_hmc_components(
    value: Any,
    *,
    raw_user_query: Any,
    source: Any,
) -> Dict[str, str]:
    """Accept a typed HMC object only as an audited view of user input.

    A syntactically valid mapping is not evidence that the user supplied the
    code.  The planner writes this slot for downstream audit, but exact-query
    authority still comes from the deterministic parse of the current raw
    turn.  Conversation citations use their separate server-offered anchor
    path and are not represented by this helper.
    """

    if str(source or "").strip() != "current_query":
        return {}
    typed = normalize_typed_hmc_components(value)
    extracted = extract_hmc_components(raw_user_query)
    if not typed or typed != extracted:
        return {}
    return typed


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
        if tool in INTENT_POLICY[target]["allowed"]
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
    relation_expansion: Any = None,
    selected_anchor_ids: Any = None,
    # Compatibility-only input for saved callers. A bare boolean has no
    # provenance and therefore cannot authorize graph traversal.
    relation_expansion_required: bool = False,
    selected_anchor_doc_types: Any = None,
    calculation_authorizations: Any = None,
    requested_sources: Any = None,
    user_context_state: Optional[Mapping[str, Any]] = None,
) -> ToolPolicyDecision:
    """将规划模型的结构化语义结果转换为可执行白名单。

    本函数不从自然语言猜测意图。它只校验 planner 给出的目标、识别可
    机械解析的 HMC 字段，并验证两个模型阶段已经达成共识的计算合同。
    ``allowed_tools`` 是唯一执行依据；``tools_plan`` 只用于界面展示。
    """
    del relation_expansion_required
    original_normalized_intent = INTENT_ALIASES.get(
        str(intent or "").strip(), str(intent or "").strip()
    )
    notes: List[str] = []
    invalid_intent = original_normalized_intent not in INTENT_POLICY
    if invalid_intent:
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
    normalized_requested_sources = _normalize_requested_sources(
        requested_sources,
        raw_user_query=authorization_text,
    )
    validated_relation_expansion = normalize_relation_expansion_contract(
        relation_expansion,
        raw_user_query=authorization_text,
        offered_anchor_ids=selected_anchor_ids,
    )
    relation_expansion_required = bool(
        validated_relation_expansion.get("needed") is True
    )
    requested_source_tools = list(
        dict.fromkeys(
            _TOOL_BY_DOC_TYPE[item["doc_type"]]
            for item in normalized_requested_sources
        )
    )
    source_coverage_fields = normalize_answer_fields(
        [
            field
            for item in normalized_requested_sources
            for field in item.get("requested_fields") or []
        ]
    )
    validated_diagnostic_closure = normalize_diagnostic_closure(
        diagnostic_closure,
        raw_user_query=authorization_text,
    )
    closure_dimensions = list(
        validated_diagnostic_closure.get("required_dimensions") or []
    )
    contract_fields = ensure_intent_answer_fields(
        [
            required_answer_fields or [],
            source_coverage_fields,
            closure_dimensions,
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
    allowed_by_intent.add("evidence_synthesis")
    if not relation_expansion_required:
        # Intent ceilings, answer-field ownership and a planner-selected tool
        # are not graph authority. They may retain direct PostgreSQL sources,
        # but only the grounded relation contract can open HugeGraph.
        allowed_by_intent.discard("graph_traversal")
        hybrid_allowed_by_intent.discard("graph_traversal")

    # Canonical answer dimensions, rather than their coarse intent label,
    # authorize the source families that may complete this turn.  This also
    # lets a narrowly mislabelled target retain an explicitly requested field
    # without reopening every source associated with the neighbouring intent.
    # Requested dimensions are not claims about source capability. A user
    # asking source A for field X authorizes A's read, not an unsolicited read
    # of source B merely because B usually stores X. Preserve X as an answer
    # obligation even when A's actual rows have no support for it.
    scoped_fields = set(source_coverage_fields)
    query_contract_tools = list(dict.fromkeys([*tools_for_answer_fields([
        field for field in contract_fields if field not in scoped_fields
    ]), *requested_source_tools]))
    contract_owner_tools = {
        tool
        for tool in query_contract_tools
        if tool in CANONICAL_TOOL_NAMES
    }
    allowed_by_intent.update(contract_owner_tools)
    hybrid_allowed_by_intent.update(contract_owner_tools)

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
        and "closed_collection_members" in contract_fields
    )
    if collection_contract:
        allowed_by_intent.update({"function_locator", "fmeca_query"})
        # Search copies never establish collection membership.  Keep their
        # scope narrow even when another accepted direct tool is present.
        hybrid_allowed_by_intent.intersection_update(
            {"function_locator", "fmeca_query"}
        )
        notes.append("完整集合成员仅由架构身份与FMECA权威记录闭合；其他资料不参与计数")

    diagnostic_closure_required = bool(validated_diagnostic_closure)
    focused_field_contract = is_focused_answer_contract(
        fields=contract_fields,
        coverage_scope=str(coverage_scope or "").strip(),
        intents=(),
        diagnostic_closure=diagnostic_closure_required,
    )
    topology_entity_resolution = bool(
        "rbd_single_failure_boundary" in contract_fields
    )
    if focused_field_contract:
        contract_tools = query_contract_tools
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
            fallback_tools = (direct_fallback or contract_fallback)[:1]
        notes.append(
            "focused_detail 已按必答字段收窄工具范围："
            + "、".join(contract_fields)
        )

    if not fallback_tools and contract_fields:
        field_candidates = [
            tool
            for tool in tools_for_answer_fields(contract_fields)
            if tool in QUERY_TOOL_NAMES and tool in allowed_by_intent
        ]
        direct_candidates = [
            tool for tool in field_candidates if tool != "hybrid_search"
        ]
        fallback_tools = (direct_candidates or field_candidates)[:1]

    # The target model has already established that the answer needs a
    # cross-document relation or multi-hop expansion.  This is deliberately a
    # typed planner signal rather than a raw-word trigger for “关系/路径/图谱”.
    # Keep the sources needed to bind an authoritative graph seed and, for a
    # causal/propagation question, the FTA structure that explains the chain.
    # HugeGraph remains a projection and is still executed only after those
    # sources provide stable business IDs.
    if relation_expansion_required:
        relation_tools = set(RELATION_TOOLS.get(
            str(validated_relation_expansion.get("relation_kind") or ""),
            (),
        ))
        allowed_by_intent.update(relation_tools)
        # A focused scalar field must not erase the authoritative bridge the
        # model says is necessary to answer it completely.
        hybrid_allowed_by_intent.update(
            tool for tool in relation_tools if tool != "graph_traversal"
        )
        notes.append("语义关系合同已保留实体定位、结构来源和多跳关系查询")

    validated_calculation_authorizations = normalize_calculation_authorizations(
        calculation_authorizations,
        raw_user_query=authorization_text,
    )
    calculation_requests = calculation_requests_from_authorizations(
        validated_calculation_authorizations,
        raw_user_query=authorization_text,
    )
    explicit_fta_calculation = any(value.startswith("fta_") for value in calculation_requests)
    explicit_rbd_calculation = _CALCULATION_REQUEST_RBD_RELIABILITY in calculation_requests
    forced_tools: List[str] = []
    tool_by_doc_type = {doc_type: tool for tool, doc_type in DOC_TYPE_BY_TOOL.items()}
    contract_tools = set(query_contract_tools)
    continuation_anchor_tools = {
        tool_by_doc_type.get(str(doc_type or "").strip(), "")
        for doc_type in (selected_anchor_doc_types or [])
    }
    continuation_anchor_tools.discard("")
    # The selected anchor is a server-issued evidence identity. Its owning
    # source may be re-read even when the model chose a neighbouring coarse
    # intent; this does not authorize any other source under that intent.
    allowed_by_intent.update(continuation_anchor_tools)
    hybrid_allowed_by_intent.update(continuation_anchor_tools)
    # A resolved follow-up must re-read the selected record's owner whenever
    # that document can supply a field requested in this turn.  The selected
    # anchor was offered by the server from a prior citation; this rule neither
    # trusts old answer prose nor widens to unrelated document families.
    for tool in _TOOL_ORDER:
        if (
            tool in continuation_anchor_tools
            and tool in contract_tools
            and tool in allowed_by_intent
            and tool not in forced_tools
        ):
            forced_tools.append(tool)
    if forced_tools:
        notes.append(
            "续轮已回读所选对象的权威来源："
            + "、".join(DOC_TYPE_BY_TOOL.get(tool, tool) for tool in forced_tools)
        )
    if topology_entity_resolution and "function_locator" in allowed_by_intent:
        forced_tools.append("function_locator")
        notes.append("单路失效边界查询先通过 Function Locator 绑定权威对象身份")
    if RBD_ANSWER_FIELDS.intersection(contract_fields) and "rbd_query" in allowed_by_intent:
        forced_tools.append("rbd_query")
        notes.append("模型显式 RBD 回答字段已保留权威结构查询")
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
    if not structured_hmc and user_context_state and user_context_state.get("query") == authorization_text:
        # Revalidate selected USER spans against the actual bounded history.
        # This is lookup identity only; calculation authority above still
        # uses the unchanged current question, never a historical rewrite.
        structured_hmc = selected_user_hmc_components(user_context_state)
    if structured_hmc:
        forced_tools.append("hmc_query")
        allowed_by_intent.update({"hmc_query", "hybrid_search"})
        hybrid_allowed_by_intent.add("hmc_query")
    # A syntactically explicit business ID can be resolved exactly without
    # guessing its document family from a prefix or trusting the planner to
    # choose the right table. Probe the bounded identity-capable PostgreSQL
    # sources; only an exact stored ID is allowed to survive downstream.
    record_identity_resolution = (
        "record_identity" in contract_fields
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

    if relation_expansion_required:
        for tool in ("function_locator", "graph_traversal"):
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

    calculation_contract_tools: set[str] = set()
    for authorization in validated_calculation_authorizations:
        target = str(authorization.get("target") or "").strip()
        if target == "fta":
            calculation_contract_tools.add("fta_query")
            if authorization.get("mode") == "calculate":
                calculation_contract_tools.add("fta_analyzer")
        elif target == "rbd":
            calculation_contract_tools.add("rbd_query")
            if authorization.get("mode") == "calculate":
                calculation_contract_tools.add("rbd_calculator")
    allowed_by_intent.update(calculation_contract_tools)

    normalized_selected_anchor_ids = {
        str(value or "").strip()
        for value in (selected_anchor_ids or [])
        if str(value or "").strip()
    }
    has_selected_anchor_contract = bool(
        normalized_selected_anchor_ids and continuation_anchor_tools
    )
    has_execution_contract = bool(
        contract_fields
        or requested_source_tools
        or has_selected_anchor_contract
        or relation_expansion_required
        or validated_calculation_authorizations
        or structured_hmc
    )
    if has_execution_contract:
        # A contract authorizes only the source families it names or owns. It
        # must not act as a global switch that lets every planner proposal
        # under the coarse intent ceiling execute. This distinction matters
        # most on follow-ups: selecting one prior RBD citation authorizes its
        # RBD re-read, not neighbouring MFL/FMECA/history lookups.
        execution_contract_tools = {
            *contract_owner_tools,
            *requested_source_tools,
            *continuation_anchor_tools,
            *calculation_contract_tools,
            *forced_tools,
            "evidence_synthesis",
        }
        if structured_hmc:
            # The complete code authorizes one exact PostgreSQL lookup and the
            # documented same-family OpenSearch fallback on exact miss.
            execution_contract_tools.add("hybrid_search")
        if relation_expansion_required:
            execution_contract_tools.update(RELATION_TOOLS.get(
                str(validated_relation_expansion.get("relation_kind") or ""),
                (),
            ))
        allowed_by_intent.intersection_update(execution_contract_tools)
        hybrid_allowed_by_intent.intersection_update(execution_contract_tools)
    if not has_execution_contract:
        # A coarse intent is only a safety ceiling. Without a canonical answer
        # field, explicit source, selected citation, grounded relation, or
        # calculation contract, arbitrary planner-selected document families
        # have no execution provenance. Preserve progress through one generic
        # seven-family recall; its hits remain candidates until a later exact
        # PostgreSQL binding is established.
        allowed_by_intent.intersection_update(
            {"hybrid_search", "evidence_synthesis"}
        )
        hybrid_allowed_by_intent.intersection_update({"hybrid_search"})
        fallback_tools = ["hybrid_search"]
        forced_tools = [
            tool for tool in forced_tools
            if tool not in QUERY_TOOL_NAMES or tool == "hybrid_search"
        ]
        notes.append("缺少可验证执行合同，仅保留一个通用混合召回")

    if not relation_expansion_required:
        # Some canonical answer fields list graph traversal as one possible
        # way to satisfy them. Field ownership is not graph authorization:
        # adaptive execution consumes ``authorized_tools`` even when the
        # planner did not put a tool in the initial plan, so close both the
        # immediate and deferred paths at this final policy boundary.
        allowed_by_intent.discard("graph_traversal")
        hybrid_allowed_by_intent.discard("graph_traversal")
        contract_owner_tools.discard("graph_traversal")
        forced_tools = [
            tool for tool in forced_tools if tool != "graph_traversal"
        ]
        fallback_tools = [
            tool for tool in fallback_tools if tool != "graph_traversal"
        ]

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
        if tool == "graph_traversal" and not relation_expansion_required:
            rejected.append(tool)
            continue
        accepted.append(item)

    deferred_authorized_tools: set[str] = set()
    if focused_field_contract and any(
        item["tool"] in QUERY_TOOL_NAMES and item["tool"] != "hybrid_search"
        for item in [*accepted, *({"tool": tool} for tool in forced_tools)]
    ):
        if "hybrid_search" in allowed_by_intent:
            # Defer execution, not authorization. A keyword hit with the right
            # fields can still be about the wrong object. The adaptive model
            # may request one same-source semantic lookup after inspecting it.
            deferred_authorized_tools.add("hybrid_search")
        accepted = [item for item in accepted if item["tool"] != "hybrid_search"]
        notes.append("聚焦字段先查权威记录；保留同源语义补查，由模型检查相关性和缺口")

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
            deferred_authorized_tools.update(deferred)
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
    if (
        (planner_failed or invalid_intent)
        and not fallback_tools
        and "hybrid_search" in allowed_by_intent
    ):
        # One generic lexical/semantic lookup is the only no-contract recovery
        # path.  It does not infer a document family or answer dimension.
        fallback_tools = ["hybrid_search"]
    if planner_failed or invalid_intent or not selected_query_tools.union(
        tool for tool in forced_tools if tool in QUERY_TOOL_NAMES
    ):
        added_fallback = False
        for tool in fallback_tools:
            if tool not in forced_tools:
                forced_tools.append(tool)
                added_fallback = True
        if added_fallback:
            notes.append("规划模型未给出可执行查询，已使用一个已授权的有界召回")

    def forced_tool_reason(canonical: str) -> str:
        if canonical in requested_source_tools:
            return "用户明确要求回查该类资料"
        if canonical in continuation_anchor_tools:
            return "续轮所选对象需从同一权威记录回读本轮字段"
        if canonical in {"hmc_query", "fta_analyzer", "rbd_calculator"}:
            return "原始问题包含可结构化字段或明确计算请求"
        if canonical == "function_locator" and topology_entity_resolution:
            return "显式单路失效边界字段需要先绑定权威对象身份"
        if record_identity_resolution:
            return "显式业务ID需在权威结构化身份列中精确解析"
        if collection_contract and canonical in {"function_locator", "fmeca_query"}:
            return "完整集合合同要求回查架构身份与权威FMECA记录"
        if canonical == "graph_traversal" and relation_expansion_required:
            return "经当前轮原文或已选引用校验的关系合同要求多跳查询"
        if canonical == "mfl_query" and maintenance_contract_fields:
            return "维修合同必须回读 MFL 权威定位与处置清单"
        if canonical in contract_owner_tools:
            return "模型显式回答字段需要该权威资料"
        return "规划不可用时的单一有界召回"

    if (
        contract_fields
        and "hybrid_search" in allowed_by_intent
        and not structured_hmc
        and not extract_business_codes(authorization_text)
        and not continuation_anchor_tools
        and any(tool in DOC_TYPE_BY_TOOL for tool in [*forced_tools, *(item["tool"] for item in accepted)])
    ):
        # Without a stable identity, a keyword hit is only one recall channel.
        # Search the already authorized families now; source fields being
        # nonempty cannot establish semantic relevance or suppress this pool.
        forced_tools.append("hybrid_search")
        notes.append("自然描述没有稳定记录身份：同源词法与语义联合召回，再回读权威记录")

    for tool in forced_tools:
        canonical = _canonical_tool(tool)
        if canonical in CANONICAL_TOOL_NAMES:
            plan_by_tool.setdefault(
                canonical,
                {
                    "tool": canonical,
                    "reason": forced_tool_reason(canonical),
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

    explicitly_authorized_tools = {
        *contract_owner_tools,
        *requested_source_tools,
        *forced_tools,
        *deferred_authorized_tools,
        *(item["tool"] for item in accepted),
        "evidence_synthesis",
    }
    explicitly_authorized_tools.intersection_update(allowed_by_intent)

    authorized_doc_types: List[str] = []
    for tool in _TOOL_ORDER:
        if tool not in explicitly_authorized_tools:
            continue
        doc_type = DOC_TYPE_BY_TOOL.get(tool)
        if doc_type and doc_type not in authorized_doc_types:
            authorized_doc_types.append(doc_type)

    hybrid_scope_tools = {
        tool
        for tool in explicitly_authorized_tools
        if tool in DOC_TYPE_BY_TOOL and tool != "hybrid_search"
    }
    if "hybrid_search" in ordered_tools and not hybrid_scope_tools:
        # A sole generic fallback stays bounded to the seven product document
        # families but does not guess one family from a coarse intent.
        hybrid_scope_tools = set(DOC_TYPE_BY_TOOL)
    hybrid_doc_types = [
        doc_type
        for tool in _TOOL_ORDER
        if tool in hybrid_scope_tools
        for doc_type in [DOC_TYPE_BY_TOOL.get(tool)]
        if doc_type
    ] if "hybrid_search" in ordered_tools else []
    authorized_tools = [
        tool for tool in _TOOL_ORDER if tool in explicitly_authorized_tools
    ]

    return {
        "intent": normalized_intent,
        "query_targets": normalized_targets,
        "calculation_authorizations": validated_calculation_authorizations,
        "calculation_requests": calculation_requests,
        "relation_expansion": validated_relation_expansion,
        "authorized_tools": authorized_tools,
        "allowed_tools": ordered_tools,
        "tools_plan": [plan_by_tool[tool] for tool in ordered_tools],
        "authorized_doc_types": authorized_doc_types,
        "hybrid_doc_types": hybrid_doc_types,
        "rejected_tools": list(dict.fromkeys(rejected)),
        "notes": notes,
        "requested_source_tools": requested_source_tools,
        "source_coverage_fields": source_coverage_fields,
        "requested_sources": normalized_requested_sources,
        "source_obligations": [
            {
                "tool": _TOOL_BY_DOC_TYPE[item["doc_type"]],
                "doc_type": item["doc_type"],
                "fields": list(item.get("requested_fields") or []),
                "question_role": item["question_role"],
                "basis": item["basis"],
            }
            for item in normalized_requested_sources
        ],
    }
