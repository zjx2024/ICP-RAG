"""Stable, model-authored answer-field contracts for ICP questions.

The planning model decides what the user is asking for.  This module only
validates that structured decision against a small domain ontology and maps it
to evidence/tool fields.  It deliberately never classifies the raw user query:
doing so here would recreate the brittle keyword router that the planner
replaced.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence


RESPONSE_FOCUS_FIELDS: dict[str, tuple[str, ...]] = {
    "importance": ("responsibility",),
    "immediate_impact": ("local_impact", "final_impact"),
    "count": ("closed_collection_members",),
    "completeness_check": ("closed_collection_members",),
    "first_action": ("detection_method", "maintenance_action", "history_steps"),
    "maintenance_action": ("maintenance_action",),
    "status_interface": ("maintenance_api",),
    "exact_identity": (
        "maintenance_locator", "record_identity", "hmc_identity",
        "record_name", "record_type",
    ),
    "history_conditions": ("history_time", "history_steps"),
    "exclusion_basis": ("history_steps",),
    "recovery_verification": ("history_result",),
    # “怎么验证” may refer either to the present incident's detection method
    # or to a historical recovery result.  The planner's current-turn field
    # contract disambiguates which of these is in scope.
    "verification_criteria": ("detection_method", "history_result"),
    "history_root_cause": ("history_root",),
    "history_solution": ("history_solution",),
    "cause": ("fault_cause", "structural_cause_chain"),
}


# 板级/SUR/功能子电路定位、维修处置、API 上报与平台摸底状态只存在于 MFL
# 权威记录中。目标模型把这些字段写进 required_fields 后，它们同时是回答
# 契约和 MFL 的所有权声明：其他资料族（如 FMECA 的补偿措施）可以提供
# 相近内容，但权威清单始终由 MFL 回读闭合。
MFL_CONTRACT_FIELDS = frozenset({
    "maintenance_action",
    "maintenance_locator",
    "maintenance_api",
    "platform_test_status",
    "platform_test_result",
})


# These are user-facing information dimensions, not database column names.
# Composite dimensions (for example ``architecture_function``) may be backed
# by several source columns, while scalar FMECA dimensions stay independent so
# an answer mentioning an impact cannot accidentally satisfy a severity query.
ANSWER_FIELD_SPECS = {
    "record_name": {
        "label": "标准名称",
        "tools": ("function_locator", "hybrid_search"),
    },
    "record_type": {
        "label": "对象类型",
        "tools": ("function_locator", "hybrid_search"),
    },
    "record_location": {
        "label": "所属位置",
        "tools": ("function_locator", "graph_traversal", "hybrid_search"),
    },
    "responsibility": {
        "label": "职责",
        "tools": ("function_locator", "hybrid_search"),
    },
    "record_identity": {
        "label": "名称、类型与身份",
        "tools": (
            "function_locator",
            "hmc_query",
            "fmeca_query",
            "mfl_query",
            "fault_history_query",
            "hybrid_search",
        ),
    },
    "architecture_hierarchy": {
        "label": "所属层级",
        "tools": ("function_locator", "graph_traversal", "hybrid_search"),
    },
    "architecture_function": {
        "label": "功能与职责",
        "tools": ("function_locator", "hybrid_search"),
    },
    "architecture_interface": {
        "label": "接口与输入输出",
        "tools": ("function_locator", "graph_traversal", "hybrid_search"),
    },
    "relation_path": {
        "label": "关系与路径",
        "tools": ("function_locator", "graph_traversal", "hybrid_search"),
    },
    "closed_collection_members": {
        "label": "完整集合成员",
        "tools": ("function_locator", "fmeca_query", "hybrid_search"),
    },
    "hmc_identity": {
        "label": "HMC代码与组成",
        "tools": ("hmc_query", "hybrid_search"),
    },
    "hmc_symptom": {
        "label": "HMC故障现象",
        "tools": ("hmc_query", "hybrid_search"),
    },
    "failure_mode": {
        "label": "故障模式",
        "tools": ("fmeca_query", "mfl_query", "hybrid_search"),
    },
    "fault_cause": {
        "label": "故障原因",
        "tools": ("fmeca_query", "fta_query", "hybrid_search"),
    },
    "structural_cause_chain": {
        "label": "结构原因链",
        "tools": ("fta_query", "hybrid_search"),
    },
    "fault_location": {
        "label": "故障定位环节",
        "tools": (
            "function_locator", "mfl_query", "graph_traversal", "hybrid_search",
        ),
    },
    "local_impact": {
        "label": "局部影响",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "final_impact": {
        "label": "最终影响",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "impact_propagation": {
        "label": "影响传播链",
        "tools": (
            "function_locator", "fmeca_query", "fta_query", "graph_traversal",
            "hybrid_search",
        ),
    },
    "severity": {
        "label": "严酷度",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "failure_probability": {
        "label": "故障概率",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "failure_probability_level": {
        "label": "故障概率等级",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "detection_method": {
        "label": "检测与判别方法",
        "tools": ("fmeca_query", "mfl_query", "hybrid_search"),
    },
    "mission_phase": {
        "label": "任务阶段与工作方式",
        "tools": ("fmeca_query", "fault_history_query", "hybrid_search"),
    },
    "design_improvement": {
        "label": "设计改进措施",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "compensation_action": {
        "label": "使用补偿措施",
        "tools": ("fmeca_query", "hybrid_search"),
    },
    "maintenance_locator": {
        "label": "维修定位",
        "tools": ("mfl_query", "hybrid_search"),
    },
    "maintenance_action": {
        "label": "检查与处置",
        "tools": ("hmc_query", "mfl_query", "fmeca_query", "hybrid_search"),
    },
    "maintenance_api": {
        "label": "状态与API信息",
        "tools": ("mfl_query", "hybrid_search"),
    },
    "platform_test_status": {
        "label": "平台级实验状态",
        "tools": ("mfl_query", "hybrid_search"),
    },
    "platform_test_result": {
        "label": "平台级实验结果",
        "tools": ("mfl_query", "hybrid_search"),
    },
    "history_time": {
        "label": "历史发生时间",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_root": {
        "label": "历史确认根因",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_steps": {
        "label": "历史处理步骤",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_solution": {
        "label": "历史最终方案",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_result": {
        "label": "历史验证结果",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_personnel": {
        "label": "历史排故人员",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "history_preventive_action": {
        "label": "历史预防措施",
        "tools": ("fault_history_query", "hybrid_search"),
    },
    "rbd_success_criterion": {
        "label": "RBD成功判据",
        "tools": ("rbd_query", "hybrid_search"),
    },
    "rbd_path_structure": {
        "label": "RBD路径结构",
        "tools": ("rbd_query", "hybrid_search"),
    },
    "rbd_redundancy_structure": {
        "label": "RBD冗余结构",
        "tools": ("rbd_query", "hybrid_search"),
    },
    "rbd_single_failure_boundary": {
        "label": "单路失效能力边界",
        "tools": ("rbd_query", "hybrid_search"),
    },
    "rbd_missing_parameters": {
        "label": "可靠性缺失参数",
        "tools": ("rbd_query", "hybrid_search"),
    },
}


RBD_ANSWER_FIELDS = frozenset(
    {
        "rbd_success_criterion",
        "rbd_path_structure",
        "rbd_redundancy_structure",
        "rbd_single_failure_boundary",
        "rbd_missing_parameters",
    }
)

# Response breadth is an open-language decision made by the target model.  The
# code boundary only accepts these protocol values and a literal current-turn
# basis; it never tries to recover the decision from keywords in the question.
RESPONSE_MODES = frozenset(
    {"structure_only", "focused", "diagnostic", "explanation", "unknown"}
)


def _current_turn_basis(value: Any, raw_user_query: Any, *, limit: int = 500) -> str:
    basis = str(value or "").strip()[:limit]
    query = str(raw_user_query or "")
    return basis if basis and basis in query else ""


def normalize_diagnostic_closure(
    value: Any,
    *,
    raw_user_query: Any,
) -> dict[str, Any]:
    """Validate a model-authored, field-level diagnostic closure request.

    A bare ``required=true`` has no retrieval meaning.  The model must bind the
    decision to this user turn and name the exact answer dimensions to close.
    Invalid dimensions are discarded; if none remain the whole authorization
    fails closed.
    """

    if not isinstance(value, Mapping) or value.get("required") is not True:
        return {}
    basis = _current_turn_basis(value.get("basis"), raw_user_query)
    dimensions = normalize_answer_fields(value.get("required_dimensions"))
    if not basis or not dimensions:
        return {}
    return {
        "required": True,
        "required_dimensions": dimensions,
        "basis": basis,
        "reason": " ".join(str(value.get("reason") or "").split())[:360],
    }


def normalize_response_mode(
    value: Any,
    *,
    raw_user_query: Any,
) -> dict[str, str]:
    """Validate response breadth without classifying the user's wording."""

    if not isinstance(value, Mapping):
        return {}
    mode = str(value.get("mode") or "").strip().casefold()
    basis = _current_turn_basis(value.get("basis"), raw_user_query)
    if mode not in RESPONSE_MODES or not basis:
        return {}
    return {"mode": mode, "basis": basis}


def ensure_intent_answer_fields(
    fields: Any,
    *,
    intents: Iterable[Any],
) -> List[str]:
    """Validate fields without deriving answer obligations from an intent.

    ``intents`` remains keyword-only for source compatibility with saved agents
    and older callers.  It is deliberately ignored: a coarse intent labels the
    topic but cannot decide whether this turn asks for a success criterion,
    topology, a scalar value, or no RBD field at all.
    """

    del intents
    return normalize_answer_fields(fields)


def _iter_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_values(item)


def normalize_answer_fields(
    value: Any,
    *,
    requested_facets: Any = (),
    limit: int = 24,
) -> List[str]:
    """Validate canonical model fields; free-form facets never grant duties."""

    result: List[str] = []

    def add(field: str) -> None:
        if field in ANSWER_FIELD_SPECS and field not in result and len(result) < limit:
            result.append(field)

    for raw in _iter_values(value):
        field = str(raw).strip().casefold().replace("-", "_").replace(" ", "_")
        if field in ANSWER_FIELD_SPECS:
            add(field)

    # ``requested_facets`` remains in the signature and query frame as an
    # audit/recall hint for saved agents.  Its values are open model language;
    # accepting aliases or substrings here would silently create answer and
    # tool obligations that neither structured model contract requested.
    del requested_facets
    return result


def required_answer_fields(state: Mapping[str, Any]) -> List[str]:
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    return normalize_answer_fields(
        frame.get("required_answer_fields"),
        requested_facets=frame.get("requested_facets") or [],
    )


def answer_field_labels(fields: Sequence[str]) -> List[str]:
    return [
        str(ANSWER_FIELD_SPECS[field]["label"])
        for field in fields
        if field in ANSWER_FIELD_SPECS
    ]


def tools_for_answer_fields(fields: Sequence[str]) -> List[str]:
    result: List[str] = []
    for field in fields:
        for tool in ANSWER_FIELD_SPECS.get(field, {}).get("tools", ()):
            if tool not in result:
                result.append(tool)
    return result


def is_focused_answer_contract(
    *,
    fields: Sequence[str],
    coverage_scope: str,
    intents: Sequence[str],
    diagnostic_closure: bool = False,
) -> bool:
    # Kept in the signature for compatibility with saved callers only.  Intent
    # is a routing/capability label and cannot broaden or narrow an answer.
    del intents
    if diagnostic_closure:
        return False
    return bool(normalize_answer_fields(fields)) and str(coverage_scope or "") == "focused_detail"


def incremental_followup_contract(state: Mapping[str, Any]) -> bool:
    """Whether this turn is a bounded, incremental continuation.

    A follow-up may legitimately span more than one diagnostic dimension --
    for example, ``先怎么检查`` plus ``旧案例如何验证恢复``.  Treating that
    as a fresh broad diagnosis makes the completion pipeline replay causes,
    impacts and every history field.  The plan node records two pieces of
    model-independent state for a resolved continuation: fields actually
    delivered by the previous answer and the current question's response
    semantics.  Together with a focused coverage scope they form a safe
    incremental contract.

    This changes presentation/completion breadth only.  It neither reuses an
    old conclusion nor relaxes evidence authorization; every fact still has
    to be read and approved again in the current turn.  Explicit requests to
    restate a complete diagnosis, closed collections and calculations retain
    their existing contracts.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    prior = frame.get("prior_delivered_fields")
    semantics = str(frame.get("question_semantics") or "").strip()
    fields = normalize_answer_fields(frame.get("required_answer_fields"))
    if not isinstance(prior, list) or not prior or not semantics or not fields:
        return False
    if str(frame.get("coverage_scope") or "") != "focused_detail":
        return False
    if "closed_collection_members" in fields or state.get("calculation_requests"):
        return False
    # Whether the user asked for an incremental answer or a full restatement is
    # open discourse semantics.  It is classified once by the target model and
    # validated as an enum; this layer must not maintain a phrase dictionary.
    return str(frame.get("response_scope") or "") == "incremental"


def focused_answer_contract(state: Mapping[str, Any]) -> bool:
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if incremental_followup_contract(state):
        return True
    closure = frame.get("diagnostic_closure")
    diagnostic_closure = bool(
        isinstance(closure, Mapping)
        and closure.get("required") is True
        and str(closure.get("basis") or "").strip()
        and normalize_answer_fields(closure.get("required_dimensions"))
    )
    # Intent and requested source names describe routing/capability, not the
    # breadth of the public answer.  Only the validated response/field
    # contract may make a turn focused or diagnostic.
    return is_focused_answer_contract(
        fields=required_answer_fields(state),
        coverage_scope=str(frame.get("coverage_scope") or ""),
        intents=(),
        diagnostic_closure=diagnostic_closure,
    )
