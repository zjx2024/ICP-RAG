"""Stable, model-authored answer-field contracts for ICP questions.

The planning model decides what the user is asking for.  This module only
validates that structured decision against a small domain ontology and maps it
to evidence/tool fields.  It deliberately never classifies the raw user query:
doing so here would recreate the brittle keyword router that the planner
replaced.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, List, Mapping, Sequence


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


def ensure_intent_answer_fields(
    fields: Any,
    *,
    intents: Iterable[Any],
) -> List[str]:
    """Keep a structured information target and its answer contract coherent.

    The semantic target classifier remains responsible for deciding whether a
    question is about reliability.  Once it has made that decision, however, a
    malformed or older field contract must not make the RBD source disappear
    during focused-tool narrowing.  A generic success criterion is the smallest
    RBD field that is valid for every reliability question; topology-dependent
    single-path questions add their more specific boundary fields in ``plan``.
    """

    result = normalize_answer_fields(fields)
    normalized_intents = {
        str(value or "").strip()
        for value in intents
        if str(value or "").strip()
    }
    if (
        "reliability_query" in normalized_intents
        and not RBD_ANSWER_FIELDS.intersection(result)
    ):
        result = normalize_answer_fields([*result, "rbd_success_criterion"])
    return result


def _identity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


# Compatibility aliases are applied only to planner-authored facets, never to
# raw user text.  This lets an older planner response such as ``严酷度`` enter
# the same typed contract as a newer response that emits ``severity``.
_EXACT_FACET_ALIASES = {
    "严酷度": ("severity",),
    "严重等级": ("severity",),
    "异常级别": ("severity",),
    "故障概率": ("failure_probability",),
    "概率等级": ("failure_probability_level",),
    "故障概率等级": ("failure_probability_level",),
    "故障模式": ("failure_mode",),
    "原因": ("fault_cause",),
    "根因": ("fault_cause",),
    "原因链": ("structural_cause_chain",),
    "故障链": ("structural_cause_chain",),
    "故障定位": ("fault_location",),
    "定位环节": ("fault_location",),
    "局部影响": ("local_impact",),
    "最终影响": ("final_impact",),
    "影响": ("local_impact", "final_impact"),
    "后果": ("local_impact", "final_impact"),
    "危害": ("local_impact", "final_impact", "severity"),
    "传播链": ("impact_propagation",),
    "影响传播": ("impact_propagation",),
    "检测方法": ("detection_method",),
    "判别方法": ("detection_method",),
    "任务阶段": ("mission_phase",),
    "工作方式": ("mission_phase",),
    "设计改进": ("design_improvement",),
    "使用补偿": ("compensation_action",),
    "维修": ("maintenance_action",),
    "处置": ("maintenance_action",),
    "api": ("maintenance_api",),
    "平台实验": ("platform_test_status", "platform_test_result"),
    "平台试验": ("platform_test_status", "platform_test_result"),
    "名称": ("record_name",),
    "类型": ("record_type",),
    "所属位置": ("record_location",),
    "职责": ("responsibility",),
    "功能": ("architecture_function",),
    "组成": ("closed_collection_members",),
    "接口": ("architecture_interface",),
    "上下游": ("relation_path",),
    "关系": ("relation_path",),
    "路径": ("relation_path",),
    "可靠性": ("rbd_success_criterion", "rbd_path_structure"),
    "历史": (
        "history_time",
        "history_root",
        "history_steps",
        "history_solution",
        "history_result",
    ),
    "预防措施": ("history_preventive_action",),
}
_NORMALIZED_EXACT_FACET_ALIASES = {
    _identity(key): value for key, value in _EXACT_FACET_ALIASES.items()
}


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
    """Validate canonical fields and bridge older planner facet labels."""

    result: List[str] = []

    def add(field: str) -> None:
        if field in ANSWER_FIELD_SPECS and field not in result and len(result) < limit:
            result.append(field)

    for raw in _iter_values(value):
        field = str(raw).strip().casefold().replace("-", "_").replace(" ", "_")
        if field in ANSWER_FIELD_SPECS:
            add(field)

    # A canonical field list is the planner's explicit answer contract.
    # ``requested_facets`` is only a compatibility bridge for older planner
    # responses that did not emit that list; mixing the two lets a broad
    # retrieval hint such as “影响” silently widen an explicit ``severity``
    # follow-up into local and final impact requirements.
    if result:
        return result

    for raw in _iter_values(requested_facets):
        normalized = _identity(raw)
        if not normalized:
            continue
        exact = _NORMALIZED_EXACT_FACET_ALIASES.get(normalized)
        if exact:
            for field in exact:
                add(field)
            continue
        # Compound planner facets such as ``严酷度与影响`` are accepted by
        # matching their semantic labels.  Specific labels are processed
        # before generic ones to retain both requested dimensions.
        for marker, fields in sorted(
            _NORMALIZED_EXACT_FACET_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if len(marker) >= 2 and marker in normalized:
                for field in fields:
                    add(field)
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
    normalized_intents = {str(value or "").strip() for value in intents}
    # A first-pass symptom diagnosis is intentionally a complete diagnostic
    # closure even if the semantic planner also names one preferred output
    # field.  Narrow field contracts are for follow-up/detail questions, not a
    # way to turn an initial fault description into a one-column lookup.
    if "symptom_diagnosis" in normalized_intents:
        return False
    # The target model may distinguish an observed, current engineering
    # abnormality from a detached knowledge explanation.  The former still
    # needs a bounded diagnostic closure (cause/effect/handling/history when
    # evidence exists), even if the user worded the immediate question as one
    # scalar such as "what is the cause".  This remains a model-authored
    # semantic contract rather than a raw-text keyword rule.
    if diagnostic_closure:
        return False
    diagnostic_dimensions = normalized_intents & {
        "cause_analysis",
        "impact_analysis",
        "maintenance_query",
        "history_query",
    }
    # A question spanning several diagnostic dimensions needs candidate
    # selection and cross-source synthesis.  Treating it as a scalar/detail
    # lookup would force every retrieved row into a field-by-field dump.
    if len(diagnostic_dimensions) >= 2:
        return False
    return bool(normalize_answer_fields(fields)) and str(coverage_scope or "") == "focused_detail"


def focused_answer_contract(state: Mapping[str, Any]) -> bool:
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    intents = [str(state.get("intent") or "").strip()]
    for target in state.get("query_targets") or []:
        value = target.get("intent") if isinstance(target, Mapping) else target
        intents.append(str(value or "").strip())
    requested_source_tools = {
        str(value or "").strip()
        for value in (
            state.get("requested_source_tools")
            or frame.get("requested_source_tools")
            or []
        )
        if str(value or "").strip()
    }
    # “为什么会坏，同时查 RBD”仍是原因分析，不应因为额外的 RBD
    # 覆盖要求退化成逐字段的图结构清单。RBD 在这类问题中补充的是
    # 成功路径/冗余边界，最终回答仍须先围绕用户的问题组织原因和结论。
    # 这不会改变真正的 reliability_query 或数值计算合同。
    if "rbd_query" in requested_source_tools and {
        "cause_analysis",
        "impact_analysis",
        "maintenance_query",
        "history_query",
        "symptom_diagnosis",
    }.intersection(intents):
        return False
    return is_focused_answer_contract(
        fields=required_answer_fields(state),
        coverage_scope=str(frame.get("coverage_scope") or ""),
        intents=intents,
        diagnostic_closure=bool(
            isinstance(frame.get("diagnostic_closure"), Mapping)
            and frame["diagnostic_closure"].get("required") is True
        ),
    )
