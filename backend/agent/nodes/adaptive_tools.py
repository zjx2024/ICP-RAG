"""ICP 首轮检索后的受限自适应工具路由。

规划模型先选择最小工具集，本节点再只基于已执行工具的状态和
有界证据摘要，决定是否需要一次补查或多跳关系查询。它不改变用户
的信息目标，不授权 FTA/RBD 计算，也不允许模型构造图实体 ID。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from backend.agent.answer_contract import (
    focused_answer_contract,
    normalize_answer_fields,
    normalize_diagnostic_closure,
    tools_for_answer_fields,
)
from backend.agent.contracts import binding_basis_rank
from backend.agent.history_focus import merge_history_parent_bindings
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.query_understanding import extract_business_codes
from backend.agent.rbd_contract import merge_rbd_query_bindings
from backend.agent.diagram_query_scope import merge_diagram_scopes, diagram_in_scope
from backend.agent.tool_policy import (
    DOC_TYPE_BY_TOOL,
    QUERY_TOOL_NAMES,
    extract_hmc_components,
    normalize_calculation_authorizations,
    normalize_relation_expansion_contract,
)
from backend.agent.tools.icp_diagnosis import _graph_request, icp_diagnosis_node
from backend.core.llm_service import llm_service

logger = logging.getLogger(__name__)


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


ADAPTIVE_TOOL_TIMEOUT_SECONDS = _positive_env_float(
    "ICP_AGENT_ADAPTIVE_TOOL_TIMEOUT_SECONDS", 30.0
)
ADAPTIVE_TOOL_MAX_PREDICTION_TOKENS = 384
MAX_FOLLOWUP_QUERY_TOOLS = 5

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
)

_TOOL_PURPOSES = {
    "function_locator": "补充系统、模块、功能、接口或上下游定位",
    "hmc_query": "补充 HMC 代码、故障现象或维修方法的权威记录",
    "fmeca_query": "补充故障模式、原因、影响、严酷度或检测方法",
    "fta_query": "补充明确的 FTA 事件、原因链和逻辑门资料",
    "rbd_query": "补充 RBD 串并联结构、成功判据和参数",
    "mfl_query": "补充维修故障清单、功能电路或 API 定位",
    "fault_history_query": "补充可比的历史排故时间、步骤、根因和处理结果",
    "hybrid_search": "在已授权文档类型中补充原文语义召回",
    "graph_traversal": "从已绑定业务 ID 出发查询能增加信息的跨文档多跳路径",
}

_TRUSTED_COMPLETION_AUTHORITIES = {
    "authoritative_record",
    "source_artifact",
    "confirmed_relation",
    "validated_relation",
    "deterministic_derivation",
}

# A record re-hydrated from a semantic search hit is authoritative *about the
# record*, but a very weak semantic hit is not authoritative about relevance to
# the current question.  Keep it in the audit/evidence registry while forcing
# a scoped structured follow-up before declaring an answer field complete.
SEMANTIC_COMPLETION_MIN_SCORE = 0.25

# Answer completion is checked at field level.  A successful query only says
# that rows were returned; it does not say that those rows contain the field
# the user asked for (an MFL row containing only API information is the common
# counterexample).  These stable slot names are also accepted in
# ``metadata.answer_fields`` so producers can advertise coverage without the
# router parsing user-facing summaries.
_HISTORY_SLOTS = (
    "history_time",
    "history_root",
    "history_steps",
    "history_solution",
    "history_result",
)

_ANSWER_FIELD_ALIASES = {
    "record_name": {
        "component_name", "function_name", "signal_name", "title",
        "failure_mode",
    },
    "record_type": {"component_type", "entity_type", "record_type"},
    "record_location": {
        "system_name", "subsystem_name", "module_name", "parent_component_id",
    },
    "responsibility": {
        "responsibility", "description", "function", "function_name",
    },
    "record_identity": {
        "component_id", "component_type", "component_name", "function_id",
        "function_name", "interface_id", "signal_name", "record_code",
        "failure_mode_id", "failure_mode", "module_failure", "sur_failure",
        "subcircuit_failure", "case_id", "title", "fault_symptom",
        "hmc_code", "lcn", "hamc", "hsc",
    },
    "architecture_hierarchy": {
        "system_name", "subsystem_name", "module_name", "component_id",
        "component_name", "component_type", "parent_component_id",
    },
    "architecture_function": {
        "function_id", "function_name", "function", "description",
        "responsibility",
    },
    "architecture_interface": {
        "inputs", "outputs", "interface_id", "source_component_id",
        "target_component_id", "signal_name", "direction", "protocol",
    },
    "relation_path": {
        "source_entity", "source_name", "target_entity", "target_name",
        "relation", "relation_type", "path_summary", "path_nodes",
        "path_relations",
    },
    "closed_collection_members": {
        "collection_closed", "collection_parent_id", "collection_member_kind",
        "collection_member_count", "collection_member_id",
        "collection_member_name", "collection_role",
    },
    "hmc_identity": {"hmc_code", "lcn", "hamc", "hsc"},
    "hmc_symptom": {"fault_symptom"},
    "entity_profile": {
        "component_id", "component_type", "component_name", "abbreviation",
        "parent_component_id", "function_id", "system_name", "subsystem_name",
        "module_name", "function_name", "description", "inputs", "outputs",
    },
    "cause": {
        "cause", "causes", "fault_cause", "failure_cause", "root_cause",
        "cause_chain", "basic_events",
    },
    "impact": {
        "impact", "impacts", "effect", "effects", "local_impact",
        "final_impact", "failure_effect", "consequence", "consequences",
    },
    "failure_mode": {
        "failure_mode", "module_failure", "sur_failure", "subcircuit_failure",
    },
    "fault_cause": {"fault_cause", "root_cause"},
    "structural_cause_chain": {
        "structural_cause_chain", "cause_chain", "top_event",
        "basic_events", "intermediate_events", "logic_gates",
        "structure_summary",
    },
    "fault_location": {
        "system_name", "subsystem_name", "module_name", "component_id",
        "component_name", "function_id", "function_name", "record_code",
        "module_failure", "sur_failure", "subcircuit_failure",
    },
    "local_impact": {"local_impact"},
    "final_impact": {"final_impact"},
    "impact_propagation": {
        "impact_propagation", "path_summary", "path_nodes", "path_relations",
        "source_entity", "source_name", "target_entity", "target_name",
        "relation", "relation_type",
    },
    "severity": {"severity", "severity_raw", "severity_normalized"},
    "failure_probability": {
        "failure_probability", "failure_probability_normalized",
    },
    "failure_probability_level": {"failure_probability_level"},
    "detection_method": {
        "detection_method", "detect_method", "diagnosis_method",
    },
    "mission_phase": {
        "mission_phase", "mission_stage", "task_stage", "task_phase",
        "operation_stage",
    },
    "design_improvement": {"design_improvement", "design_improve"},
    "compensation_action": {"compensation_action", "compensate_action"},
    "detect": {
        "detect", "detection", "detect_method", "detection_method",
        "diagnosis_method", "inspection_method", "test_method",
    },
    "hmc_repair": {
        "hmc_repair", "hmc_repair_method", "repair_method",
        "maintenance_action",
    },
    "mfl_repair": {
        "mfl_repair", "mfl_repair_method", "repair_method",
        "maintenance_action",
    },
    "mfl_api": {
        "mfl_api", "api", "api_name", "api_method", "api_usage",
        "api_supported", "api_direct_report", "api_direct_reportable",
    },
    "maintenance_locator": {
        "record_code", "module_failure", "sur_failure", "subcircuit_failure",
    },
    "maintenance_action": {
        "maintenance_action", "repair_method", "compensation_action",
        "final_solution",
    },
    "maintenance_api": {
        "api_reportable", "api_reportable_raw", "api_name", "api_method",
        "api_usage",
    },
    "platform_test_status": {"platform_test_status"},
    "platform_test_result": {"platform_test", "platform_test_raw"},
    "history_time": {
        "history_time", "occurred_at", "occurrence_time", "event_time",
    },
    "history_root": {
        "history_root", "root_cause",
    },
    "history_steps": {
        "history_steps", "steps", "step_records", "diagnosis_steps",
        "handling_steps",
    },
    "history_solution": {
        "history_solution", "final_solution", "resolution", "solution",
    },
    "history_result": {
        "history_result", "verification_result", "handling_result",
        "outcome",
    },
    "history_personnel": {
        "operator", "personnel", "maintenance_personnel", "troubleshooter",
    },
    "history_preventive_action": {"preventive_action"},
    "architecture_path": {
        "architecture_path", "interface_path", "data_path", "signal_path",
        "upstream", "downstream",
    },
    "rbd": {
        "rbd", "rbd_structure", "reliability", "reliability_parameters",
        "success_criterion", "paths", "series_paths", "parallel_branches",
    },
    "rbd_success_criterion": {"success_criterion"},
    "rbd_path_structure": {
        "structure_description", "structure_expression", "path_structure",
        "connection_types",
    },
    "rbd_redundancy_structure": {"redundancy_structure"},
    "rbd_single_failure_boundary": {"single_failure_boundary"},
    "rbd_missing_parameters": {"missing_parameters"},
}

_ANSWER_FIELD_CONTAINERS = (
    "answer_fields",
    "record",
    "row",
    "fields",
    "source_record",
    "original",
    "raw_data",
)

_STRUCTURED_SUPPORT_FIELDS = {
    "source_component_id",
    "target_component_id",
}

def _field_has_value(value: Any) -> bool:
    """Return whether a structured field is present, without reading its prose."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    # ``False`` is meaningful for fields such as API support, and zero is a
    # valid reliability/occurrence value.
    return True


def _normalized_field_name(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _structured_answer_fields(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Collect declared/raw structured fields from one evidence item.

    Evidence producers may put fields directly on the evidence, in metadata,
    or in one of a few explicitly named metadata containers.  We intentionally
    do not inspect ``content`` or ``display_summary``: those are presentation
    text and cannot safely prove that a database column was present.
    """

    collected: Dict[str, Any] = {}
    pending: List[Mapping[str, Any]] = [evidence]
    metadata = evidence.get("metadata")
    if isinstance(metadata, Mapping):
        pending.append(metadata)
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        for raw_key, value in current.items():
            key = _normalized_field_name(raw_key)
            if not key:
                continue
            if key == "answer_fields" and isinstance(value, (list, tuple, set, frozenset)):
                for declared in value:
                    declared_key = _normalized_field_name(declared)
                    if declared_key:
                        collected.setdefault(declared_key, True)
                continue
            if key in _ANSWER_FIELD_CONTAINERS and isinstance(value, Mapping):
                pending.append(value)
                continue
            if (
                key in _STRUCTURED_SUPPORT_FIELDS
                or any(key in aliases for aliases in _ANSWER_FIELD_ALIASES.values())
            ):
                collected.setdefault(key, value)
    return collected


def _can_complete_answer_slot(evidence: Mapping[str, Any]) -> bool:
    if str(evidence.get("authority") or "") not in _TRUSTED_COMPLETION_AUTHORITIES:
        return False
    if str(evidence.get("match_type") or "") != "semantic_id_hydration":
        return True
    raw_score = evidence.get("rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = evidence.get("retrieval_score")
    return (
        isinstance(raw_score, (int, float))
        and float(raw_score) >= SEMANTIC_COMPLETION_MIN_SCORE
    )


def _coverage_slots(state: Mapping[str, Any]) -> set[str]:
    """Return answer fields proved by trusted structured evidence."""

    coverage: set[str] = set()
    for item in state.get("evidence_registry") or []:
        if not isinstance(item, Mapping):
            continue
        if not _can_complete_answer_slot(item):
            continue
        claim_type = _normalized_field_name(item.get("claim_type"))
        doc_type = _normalized_field_name(item.get("doc_type"))

        # Claim types with one unambiguous semantic contract can prove a slot
        # on their own.  Broad row claims (fmeca_fact, historical_case and
        # maintenance_locator) deliberately cannot.
        if claim_type in {"fault_cause", "cause_fact", "fta_structure"}:
            coverage.add("cause")
        if claim_type in {"fault_cause", "cause_fact"}:
            coverage.add("fault_cause")
        if claim_type == "fta_structure":
            coverage.add("structural_cause_chain")
        if claim_type in {"failure_impact", "impact_fact"}:
            coverage.add("impact")
        if claim_type in {"detection_method", "diagnosis_method"}:
            coverage.add("detect")
            coverage.add("detection_method")
        if claim_type == "maintenance_action":
            coverage.add("maintenance_action")
            if doc_type == "hmc":
                coverage.add("hmc_repair")
            elif doc_type == "mfl":
                coverage.add("mfl_repair")
        if claim_type in {"interface_fact", "relation_path", "architecture_path"}:
            coverage.add("architecture_path")
        if claim_type == "interface_fact":
            coverage.add("architecture_interface")
        if claim_type == "relation_path":
            coverage.add("relation_path")
            coverage.add("impact_propagation")
        # An RBD claim family identifies the source, not which structural
        # dimension the row actually proves.  Only typed answer_fields below
        # may satisfy path, criterion, redundancy or boundary obligations.

        fields = _structured_answer_fields(item)

        def any_field(slot: str) -> bool:
            return any(
                alias in fields and _field_has_value(fields[alias])
                for alias in _ANSWER_FIELD_ALIASES[slot]
            )

        def has_field_pair(identifier: str, description: str) -> bool:
            if not all(
                key in fields and _field_has_value(fields[key])
                for key in (identifier, description)
            ):
                return False
            # Evidence display titles often default to the business ID.  Such
            # a duplicate is not a human-readable definition of the record.
            return str(fields[identifier]).strip().casefold() != str(
                fields[description]
            ).strip().casefold()

        def has_record_identity() -> bool:
            """Require an ID and its human-readable meaning from one row.

            A bare code is not enough to answer “what is this record?”, and a
            description without its code cannot bind the answer to the object
            the user named.  Each document family therefore declares its own
            explicit identity pair instead of inheriting Function Locator's
            column shape.
            """

            if doc_type == "function_locator":
                return any(
                    has_field_pair(identifier, description)
                    for identifier, description in (
                        ("component_id", "component_name"),
                        ("function_id", "function_name"),
                        ("interface_id", "signal_name"),
                    )
                )
            if doc_type == "fmeca":
                return any(
                    has_field_pair(identifier, "failure_mode")
                    for identifier in ("record_code", "failure_mode_id")
                )
            if doc_type == "mfl":
                return any(
                    has_field_pair("record_code", description)
                    for description in (
                        "module_failure",
                        "sur_failure",
                        "subcircuit_failure",
                    )
                )
            if doc_type == "fault_history":
                return has_field_pair("case_id", "fault_symptom")
            if doc_type == "hmc":
                return has_field_pair("hmc_code", "fault_symptom")
            return False

        if doc_type in {"fmeca", "fault_history", "fta"} and any_field("cause"):
            coverage.add("cause")
        if doc_type == "fmeca" and any_field("impact"):
            coverage.add("impact")
        if doc_type in {"fmeca", "fault_history"} and any_field("detect"):
            coverage.add("detect")
        if doc_type == "hmc" and any_field("hmc_repair"):
            coverage.add("hmc_repair")
        if doc_type == "mfl":
            if any_field("mfl_repair"):
                coverage.add("mfl_repair")
            if any_field("mfl_api"):
                coverage.add("mfl_api")
        if doc_type == "fault_history" or claim_type == "historical_case":
            for slot in _HISTORY_SLOTS:
                if any_field(slot):
                    coverage.add(slot)
        if doc_type == "function_locator":
            if any_field("entity_profile"):
                coverage.add("entity_profile")
            has_endpoints = all(
                key in fields and _field_has_value(fields[key])
                for key in ("source_component_id", "target_component_id")
            )
            if has_endpoints or any_field("architecture_path"):
                coverage.add("architecture_path")
        if doc_type == "rbd" and any_field("rbd"):
            coverage.add("rbd")
        if doc_type == "function_locator":
            for slot in (
                "record_name",
                "record_type",
                "record_location",
                "responsibility",
                "architecture_hierarchy",
                "architecture_function",
                "architecture_interface",
                "relation_path",
                "closed_collection_members",
            ):
                if any_field(slot):
                    coverage.add(slot)
        if has_record_identity():
            coverage.add("record_identity")
        if doc_type == "fmeca":
            for slot in (
                "failure_mode",
                "fault_cause",
                "local_impact",
                "final_impact",
                "severity",
                "failure_probability",
                "failure_probability_level",
                "detection_method",
                "mission_phase",
                "design_improvement",
                "compensation_action",
                "closed_collection_members",
            ):
                if any_field(slot):
                    coverage.add(slot)
        if doc_type == "hmc":
            for slot in ("hmc_identity", "hmc_symptom", "maintenance_action"):
                if any_field(slot):
                    coverage.add(slot)
        if doc_type == "mfl":
            for slot in (
                "failure_mode",
                "detection_method",
                "maintenance_locator",
                "maintenance_action",
                "maintenance_api",
                "fault_location",
                "platform_test_status",
                "platform_test_result",
            ):
                if any_field(slot):
                    coverage.add(slot)
        if doc_type == "fault_history":
            # History fields are deliberately scoped to history evidence; a
            # current FMECA root cause must not satisfy a historical-root turn.
            for slot in (*_HISTORY_SLOTS, "mission_phase", "history_personnel", "history_preventive_action"):
                if any_field(slot):
                    coverage.add(slot)
        if doc_type == "function_locator" and any_field("fault_location"):
            coverage.add("fault_location")
        if doc_type == "fta" and any_field("structural_cause_chain"):
            coverage.add("structural_cause_chain")
        if claim_type == "relation_path" and any_field("impact_propagation"):
            coverage.add("impact_propagation")
        if doc_type == "rbd":
            for slot in (
                "rbd_success_criterion",
                "rbd_path_structure",
                "rbd_redundancy_structure",
                "rbd_single_failure_boundary",
                "rbd_missing_parameters",
            ):
                if any_field(slot):
                    coverage.add(slot)
    return coverage


def _required_answer_slots(state: Mapping[str, Any]) -> List[str]:
    """Return only model-authored, validated answer dimensions."""

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    result = normalize_answer_fields(frame.get("required_answer_fields"))
    targets = [
        item for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping)
    ]
    for target in targets:
        for slot in normalize_answer_fields(
            target.get("required_fields") or target.get("answer_fields")
        ):
            if slot not in result:
                result.append(slot)
    closure = normalize_diagnostic_closure(
        frame.get("diagnostic_closure"),
        raw_user_query=state.get("query"),
    )
    for slot in closure.get("required_dimensions") or []:
        if slot not in result:
            result.append(slot)
    return result


def _slot_is_covered(slot: str, coverage: set[str]) -> bool:
    if slot == "repair":
        return bool(coverage & {"hmc_repair", "mfl_repair"})
    return slot in coverage


def _tools_for_slot(slot: str) -> tuple[str, ...]:
    return tuple(
        tool
        for tool in tools_for_answer_fields([slot])
        if tool in QUERY_TOOL_NAMES
    )

ADAPTIVE_TOOL_PROMPT = """你是 ICP 诊断智能体的工具路由器，只决定是否需要补查，不回答用户问题。

用户问题：{query}
本轮唯一任务视图（只读，包含消解对象及自然语言分问）：{targets}
已执行工具：{executed_tools}
已有结果：
{result_summary}

当前仍可选的工具：
{available_tools}

系统识别的跨文档扩展需要：
{graph_need}

约束：
1. 按本轮自然语言分问判断缺口。字段只帮助定位资料，字段有值不等于已回答条件、解释或核对问题；不得从 intent 扩展原因、影响、处置或历史套餐。
2. 先检查已有记录是否对应本轮对象、现象和限定条件，再看所需字段。权威记录真实、字段齐全不代表相关；只有泛词命中或记录与现象冲突时，可选已授权的同源 hybrid_search 补查。已有相关直接证据覆盖所需信息时选 answer；检索数量本身不代表答案完整。
3. 只有规划阶段已给出 relation_expansion 或首轮已明确选择 graph_traversal 时才可查图；仍须先有可追溯业务实体。不得从粗粒度 intent 重新推断查图需要。
4. 只能从“当前仍可选的工具”中选择；不得输出实体 ID、SQL、Gremlin、计算器或新的信息目标。
5. 补查最多选 {max_followup_tools} 个直接查询工具，同一工具不重复执行。

只输出 JSON：
{{
  "decision": "answer|retrieve|graph|retrieve_then_graph",
  "next_tools": ["fmeca_query"],
  "reason": "简短说明哪个分问仍缺少相关证据，或为什么无需补查",
  "tool_queries": {{"hybrid_search": ["保留当前对象和区分条件、针对缺口的查询"]}}
}}
"""


def _unique_strings(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))


def _completion_followups(
    state: Mapping[str, Any],
    *,
    executed: Sequence[str],
    available: Sequence[str],
) -> List[str]:
    """Return only the tools needed to fill uncovered answer fields.

    Coarse intents are deliberately absent from this decision. A tool is added
    only while a canonical field explicitly named by the model (directly or in
    a grounded diagnostic-closure contract) is still empty.
    """

    available_set = set(available)
    executed_set = set(executed)
    coverage = _coverage_slots(state)

    result: List[str] = []

    def add(tool: str) -> None:
        if (
            tool in available_set
            and tool not in executed_set
            and tool not in result
            and len(result) < MAX_FOLLOWUP_QUERY_TOOLS
        ):
            result.append(tool)

    required_slots = [
        slot for slot in _required_answer_slots(state)
        if not _slot_is_covered(slot, coverage)
    ]

    # First give every explicit target one best available tool. This prevents
    # the secondary FTA view for a cause target from exhausting the budget
    # before a simultaneously requested repair/history/path target is covered.
    primary_by_slot: Dict[str, str] = {}
    for slot in required_slots:
        for tool in _tools_for_slot(slot):
            if tool in available_set and tool not in executed_set:
                add(tool)
                primary_by_slot[slot] = tool
                break

    # Then add the remaining explicit-source candidates while budget remains.
    # Cause questions normally benefit from both the FMECA row and an explicit
    # FTA cause chain; repair questions may need both MFL and HMC because either
    # table can legitimately omit the actual action.
    for slot in required_slots:
        primary = primary_by_slot.get(slot)
        for tool in _tools_for_slot(slot):
            if tool != primary:
                add(tool)

    return result


def _prepare_model(model: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not model:
        return None
    prepared = dict(model)
    prepared["num_predict"] = ADAPTIVE_TOOL_MAX_PREDICTION_TOKENS
    if prepared.get("provider") == "ollama":
        prepared["reasoning"] = False
        prepared["temperature"] = 0.0
        prepared["format"] = "json"
    from backend.agent.model_runtime_policy import bounded_model_transport
    prepared["stream"] = False
    return bounded_model_transport(prepared, ADAPTIVE_TOOL_TIMEOUT_SECONDS)


def _parse_json_object(value: Any) -> Dict[str, Any] | None:
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


def _executed_tools(state: Mapping[str, Any]) -> List[str]:
    tools = []
    for item in state.get("tool_results", []) or []:
        if not isinstance(item, Mapping):
            continue
        tool = str(item.get("tool") or "").strip()
        if tool in QUERY_TOOL_NAMES or tool == "graph_traversal":
            tools.append(tool)
    return _unique_strings(tools)


def _evidence_count(state: Mapping[str, Any]) -> int:
    ids = {
        str(item.get("evidence_id") or "")
        for item in (state.get("evidence_registry", []) or [])
        if isinstance(item, Mapping) and str(item.get("evidence_id") or "")
    }
    return len(ids)


def _hmc_codes_from_evidence(evidence: Mapping[str, Any]) -> set[str]:
    """Return non-conflicting structured HMC identities from one PG fact."""

    metadata = (
        evidence.get("metadata")
        if isinstance(evidence.get("metadata"), Mapping)
        else {}
    )
    answer_fields = (
        metadata.get("answer_fields")
        if isinstance(metadata.get("answer_fields"), Mapping)
        else {}
    )
    candidates: List[Any] = [
        evidence.get("record_id"),
        metadata.get("hmc_code"),
        answer_fields.get("hmc_code"),
    ]
    for container in (metadata, answer_fields):
        lcn = str(container.get("lcn") or "").strip()
        hamc = str(container.get("hamc") or "").strip()
        hsc = str(container.get("hsc") or "").strip()
        if lcn and hamc and hsc:
            candidates.append(f"{lcn}-{hamc}-{hsc}")

    codes: set[str] = set()
    for candidate in candidates:
        components = extract_hmc_components(candidate)
        if all(components.get(key) for key in ("lcn", "hamc", "hsc")):
            codes.add(str(components.get("hmc_code") or ""))
    return {code for code in codes if code}


def _validated_source_requirements(
    state: Mapping[str, Any],
) -> List[tuple[str, str, tuple[str, ...]]]:
    """Return structurally valid source obligations carried by the plan.

    Natural-language source detection is deliberately absent. The plan has
    already validated provenance; this boundary only rechecks the closed
    document/tool pairing and canonical requested fields before allowing a
    shortcut to declare those obligations complete.
    """

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    tool_by_doc_type = {
        doc_type: tool for tool, doc_type in DOC_TYPE_BY_TOOL.items()
    }
    requirements: Dict[tuple[str, str], List[str]] = {}

    def add(*, tool: Any = "", doc_type: Any = "", fields: Any = ()) -> None:
        normalized_tool = str(tool or "").strip()
        normalized_doc_type = str(doc_type or "").strip()
        if not normalized_tool and normalized_doc_type:
            normalized_tool = tool_by_doc_type.get(normalized_doc_type, "")
        expected_doc_type = DOC_TYPE_BY_TOOL.get(normalized_tool, "")
        if (
            not expected_doc_type
            or normalized_doc_type not in {"", expected_doc_type}
        ):
            return
        key = (normalized_tool, expected_doc_type)
        current = requirements.setdefault(key, [])
        for field in normalize_answer_fields(fields):
            if field not in current:
                current.append(field)

    for container in (frame, state):
        for raw in container.get("source_obligations") or []:
            if isinstance(raw, Mapping):
                add(
                    tool=raw.get("tool"),
                    doc_type=raw.get("doc_type"),
                    fields=raw.get("fields") or raw.get("requested_fields") or [],
                )
        for raw in container.get("requested_sources") or []:
            if isinstance(raw, Mapping):
                add(
                    doc_type=raw.get("doc_type"),
                    fields=raw.get("requested_fields") or raw.get("fields") or [],
                )
    for tool in [
        *(state.get("requested_source_tools") or []),
        *(frame.get("requested_source_tools") or []),
    ]:
        add(tool=tool)

    return [
        (tool, doc_type, tuple(fields))
        for (tool, doc_type), fields in requirements.items()
    ]


def _source_requirements_are_covered(state: Mapping[str, Any]) -> bool:
    requirements = _validated_source_requirements(state)
    if not requirements:
        return True

    evidence_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in (state.get("evidence_registry") or [])
        if isinstance(item, Mapping) and str(item.get("evidence_id") or "")
    }
    results_by_tool: Dict[str, List[Mapping[str, Any]]] = {}
    for result in state.get("tool_results") or []:
        if not isinstance(result, Mapping):
            continue
        tool = str(result.get("tool") or "").strip()
        if tool in QUERY_TOOL_NAMES:
            results_by_tool.setdefault(tool, []).append(result)

    for tool, doc_type, fields in requirements:
        results = results_by_tool.get(tool, [])
        completed_results = [
            result
            for result in results
            if str(result.get("status") or "") in {"ok", "empty", "partial"}
        ]
        if not completed_results:
            return False
        linked_evidence = [
            evidence_by_id[evidence_id]
            for result in completed_results
            for evidence_id in _unique_strings(result.get("evidence_ids") or [])
            if evidence_id in evidence_by_id
            and str(evidence_by_id[evidence_id].get("doc_type") or "") == doc_type
            and _can_complete_answer_slot(evidence_by_id[evidence_id])
        ]
        if fields:
            source_coverage = _coverage_slots(
                {"evidence_registry": linked_evidence}
            )
            if any(
                not _slot_is_covered(field, source_coverage)
                for field in fields
            ):
                return False
        elif not linked_evidence and not any(
            str(result.get("status") or "") == "empty"
            for result in completed_results
        ):
            return False
    return True


def _complete_exact_hmc_answer_available(state: Mapping[str, Any]) -> bool:
    """Whether one exact PG HMC row closes every validated obligation.

    Intent and target labels are intentionally ignored. The shortcut is an
    evidence/contract decision: a full code in the original turn must resolve
    to exactly one matching PostgreSQL identity, every canonical answer/source
    obligation must already be covered, and no relation or calculation work
    may remain.
    """

    query = str(state.get("query") or "")
    query_components = extract_hmc_components(query)
    if not all(query_components.get(key) for key in ("lcn", "hamc", "hsc")):
        return False
    expected_code = str(query_components.get("hmc_code") or "")
    explicit_hmc_codes = {expected_code}
    for business_code in extract_business_codes(query):
        components = extract_hmc_components(business_code)
        if all(components.get(key) for key in ("lcn", "hamc", "hsc")):
            explicit_hmc_codes.add(str(components.get("hmc_code") or ""))
    if len({code for code in explicit_hmc_codes if code}) != 1:
        # A comparison or batch query containing two complete codes cannot be
        # closed by the first exact row merely because the coarse target label
        # happens to look singular.
        return False

    successful_hmc_evidence_ids = {
        evidence_id
        for result in (state.get("tool_results") or [])
        if isinstance(result, Mapping)
        and str(result.get("tool") or "") == "hmc_query"
        and str(result.get("status") or "") == "ok"
        for evidence_id in _unique_strings(result.get("evidence_ids") or [])
    }
    if not successful_hmc_evidence_ids:
        return False

    exact_identities: set[tuple[str, str, str]] = set()
    matching_identity_evidence_ids: set[str] = set()
    for evidence in state.get("evidence_registry") or []:
        if not isinstance(evidence, Mapping):
            continue
        if not (
            str(evidence.get("source_kind") or "") == "postgres"
            and str(evidence.get("authority") or "") == "authoritative_record"
            and str(evidence.get("doc_type") or "") == "hmc"
            and str(evidence.get("match_type") or "") == "exact_code"
        ):
            continue
        codes = _hmc_codes_from_evidence(evidence)
        # Conflicting or absent structured identities cannot prove an exact
        # match, even if the evidence label itself says ``exact_code``.
        if len(codes) != 1:
            return False
        code = next(iter(codes))
        identity = (
            str(evidence.get("doc_id") or "").strip(),
            str(evidence.get("record_id") or code).strip(),
            str(evidence.get("source_ref") or "").strip(),
        )
        exact_identities.add(identity)
        if code != expected_code:
            return False
        evidence_id = str(evidence.get("evidence_id") or "").strip()
        if evidence_id in successful_hmc_evidence_ids:
            matching_identity_evidence_ids.add(evidence_id)

    if len(exact_identities) != 1 or not matching_identity_evidence_ids:
        return False

    coverage = _coverage_slots(state)
    if any(
        not _slot_is_covered(slot, coverage)
        for slot in _required_answer_slots(state)
    ):
        return False
    if not _source_requirements_are_covered(state):
        return False

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    selected_anchor_ids = [
        str(anchor.get("anchor_id") or "").strip()
        for anchor in (state.get("selected_conversation_anchors") or [])
        if isinstance(anchor, Mapping) and str(anchor.get("anchor_id") or "").strip()
    ]
    raw_relation = (
        state.get("relation_expansion")
        if isinstance(state.get("relation_expansion"), Mapping)
        else frame.get("relation_expansion")
    )
    relation = normalize_relation_expansion_contract(
        raw_relation,
        raw_user_query=query,
        offered_anchor_ids=selected_anchor_ids,
    )
    if relation.get("needed") is True:
        return False

    if normalize_calculation_authorizations(
        state.get("calculation_authorizations")
        or frame.get("calculation_authorizations"),
        raw_user_query=query,
    ):
        return False
    if any(
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
    ):
        return False
    return True


def _result_summary(state: Mapping[str, Any], executed: Sequence[str]) -> str:
    result_lines: List[str] = []
    for item in state.get("tool_results", []) or []:
        if not isinstance(item, Mapping):
            continue
        tool = str(item.get("tool") or "").strip()
        if tool not in executed:
            continue
        data = item.get("data") if isinstance(item.get("data"), Mapping) else {}
        raw_count = data.get("count", len(item.get("evidence_ids") or []))
        try:
            count = max(0, int(raw_count or 0))
        except (TypeError, ValueError):
            count = len(item.get("evidence_ids") or [])
        result_lines.append(
            f"- {tool}: status={str(item.get('status') or 'unknown')}, count={count}"
        )

    # Round-robin records, not the first eight claims from one table. Otherwise
    # a rich HMC row can hide every history case from the gap decision.
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in state.get("evidence_registry", []) or []:
        if not isinstance(item, Mapping):
            continue
        from backend.agent.record_identity import RecordIdentity
        doc_type = str(item.get("doc_type") or "unknown")
        metadata = item.get("metadata") or {}
        identity = RecordIdentity.from_row(doc_type, {
            **item, "candidate_scope": item.get("scope") or {},
            "revision_id": metadata.get("revision_id"), "variant_id": metadata.get("variant_id"),
        }, fallback=str(item.get("evidence_id") or len(groups))).candidate_id
        record = groups.setdefault(doc_type, {}).setdefault(identity, {
            "doc_type": doc_type, "record_id": item.get("record_id"),
            "title": item.get("title"), "match_type": item.get("match_type"),
            "fields": {}, "summary": str(item.get("display_summary") or "")[:600],
        })
        record["fields"].update(_structured_answer_fields(item))
    from itertools import zip_longest
    previews = []
    for row in zip_longest(*(list(records.values())[:4] for records in groups.values())):
        for record in row:
            if record is not None:
                previews.append(json.dumps(record, ensure_ascii=False, default=str)[:1800])
        if len(previews) >= 14:
            break
    previews = previews[:14]
    result_lines.append(f"- 可追溯候选证据: {_evidence_count(state)} 条")
    doc_type_counts: Dict[str, int] = {}
    for item in state.get("evidence_registry", []) or []:
        if not isinstance(item, Mapping):
            continue
        doc_type = str(item.get("doc_type") or "unknown").strip()
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    if doc_type_counts:
        result_lines.append(
            "- 已覆盖文档类型: "
            + ", ".join(f"{key}={value}" for key, value in sorted(doc_type_counts.items()))
        )
    seed_counts: Dict[str, int] = {}
    for item in state.get("graph_seed_bindings", []) or []:
        if not isinstance(item, Mapping):
            continue
        label = "/".join(
            filter(
                None,
                (
                    str(item.get("entity_kind") or "entity"),
                    str(item.get("source_tool") or "unknown"),
                    str(item.get("binding_status") or "candidate"),
                ),
            )
        )
        seed_counts[label] = seed_counts.get(label, 0) + 1
    if seed_counts:
        result_lines.append(
            "- 可用于关系扩展的实体锚点: "
            + ", ".join(f"{key}={value}" for key, value in sorted(seed_counts.items()))
        )
    candidate_count = len(state.get("candidate_faults", []) or [])
    if candidate_count:
        result_lines.append(f"- 当前候选故障: {candidate_count} 项")
    if previews:
        result_lines.append("- 候选摘要: " + " | ".join(previews))
    return "\n".join(result_lines)


def _model_for_state(state: Mapping[str, Any]) -> Dict[str, Any] | None:
    chat_model_id = str(state.get("chat_model_id") or "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client

        model = postgres_client.get_model(chat_model_id)
    else:
        agent_id = str(state.get("agent_id") or "")
        model = agent_model_role_service.get_model_by_role(agent_id, "reflect") if agent_id else None
    model = agent_model_role_service.apply_agent_llm_params(dict(state), model)
    return _prepare_model(model)


def _fallback_tools(
    *,
    state: Mapping[str, Any],
    remaining_query_tools: Sequence[str],
    graph_available: bool,
) -> List[str]:
    # 二次路由模型不可用时，只尊重首轮已明确选择的图计划，
    # 或在完全无证据时补一次受限混合召回；不自行猜测其他领域工具。
    if graph_available and "graph_traversal" in set(state.get("allowed_tools") or []):
        return ["graph_traversal"]
    if _evidence_count(state) == 0 and "hybrid_search" in remaining_query_tools:
        return ["hybrid_search"]
    return []


_MAX_GRAPH_SEEDS = 8
_MAX_CANDIDATE_SEEDS_PER_TOOL = 3


def _graph_expansion_reason(state: Mapping[str, Any]) -> str:
    """Return why the current information goal benefits from graph expansion.

    This deliberately uses the planner's structured semantic targets rather
    than matching words such as “关系” or “路径” in the user's text.  It only
    decides whether graph expansion is useful; a traversal still cannot run
    before a query tool has produced a traceable business entity ID.
    """

    calculation_requests = {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
        if str(value or "").strip()
    }
    if calculation_requests & {"fta_cut_sets", "fta_probability", "rbd_reliability"}:
        # Formal FTA/RBD structures and deterministic Python calculators are
        # already the authoritative path. HugeGraph adds no numeric authority
        # and only increases noise for these requests.
        return ""

    authorized = set(state.get("authorized_tools") or state.get("allowed_tools") or [])
    if "graph_traversal" not in authorized:
        return ""
    relation_expansion = state.get("relation_expansion")
    raw_relation_needed = (
        relation_expansion.get("needed")
        if isinstance(relation_expansion, Mapping)
        else False
    )
    relation_needed = (
        raw_relation_needed is True
        or str(raw_relation_needed or "").strip().casefold() in {"1", "true", "yes"}
    )
    if isinstance(relation_expansion, Mapping) and relation_needed:
        reason = str(relation_expansion.get("reason") or "").strip()
        return reason or "规划阶段识别出完整回答需要跨文档关系补全"
    if "graph_traversal" in set(state.get("allowed_tools") or []):
        return "规划阶段已识别出需要跨文档或多跳信息"
    return ""


def _seed_owner_tools(state: Mapping[str, Any]) -> List[str]:
    """Return graph-seed owners from explicit field/source contracts only."""

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    owners = {
        tool
        for tool in tools_for_answer_fields(_required_answer_slots(state))
        if tool in QUERY_TOOL_NAMES
    }
    for raw in frame.get("source_obligations") or []:
        if not isinstance(raw, Mapping):
            continue
        tool = str(raw.get("tool") or "").strip()
        if tool in QUERY_TOOL_NAMES:
            owners.add(tool)
    owners.update(
        tool
        for tool in (frame.get("requested_source_tools") or [])
        if tool in QUERY_TOOL_NAMES
    )
    authorized = set(state.get("authorized_tools") or state.get("allowed_tools") or [])
    return [
        tool for tool in _TOOL_ORDER
        if tool in owners and tool in authorized
    ]


def _candidate_seed_tools(state: Mapping[str, Any]) -> List[str]:
    executed = set(_executed_tools(state))
    return [tool for tool in _seed_owner_tools(state) if tool in executed]


def _eligible_graph_seed_bindings(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Select bounded graph anchors whose origin matches the information goal.

    A complete HMC code is an exact identity and is always eligible.  Keyword
    HMC matches are deliberately excluded: a shared word such as “外设” must
    not turn an unrelated HMC row into the center of a multi-hop answer.
    Other semantic candidates are allowed only when their source tool was both
    executed/authorized and owns an explicit field or source obligation. They
    are capped per source so one broad table query cannot monopolize traversal.
    """

    allowed_candidate_tools = set(_candidate_seed_tools(state))
    record_binding = (
        state.get("structured_record_binding")
        if isinstance(state.get("structured_record_binding"), Mapping)
        else {}
    )
    fmeca_record_ambiguous = record_binding.get("ambiguous") is True
    fmeca_relevance_only = record_binding.get("relevance_only") is True
    entity_binding = (
        state.get("structured_entity_binding")
        if isinstance(state.get("structured_entity_binding"), Mapping)
        else {}
    )
    structured_entity_ambiguous = entity_binding.get("ambiguous") is True
    exact: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    per_tool: Dict[str, int] = {}
    seen = set()
    for raw in state.get("graph_seed_bindings", []) or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        logical_id = str(item.get("logical_id") or "").strip()
        if not logical_id or logical_id in seen:
            continue
        binding_status = str(item.get("binding_status") or "candidate")
        source_tool = str(item.get("source_tool") or "").strip()
        if (
            (fmeca_record_ambiguous or fmeca_relevance_only)
            and source_tool == "fmeca_query"
        ):
            # A later adaptive subset must not turn one side of a preserved
            # FMECA identity tie or relevance-only match into a graph anchor.
            continue
        if structured_entity_ambiguous and source_tool == "function_locator":
            # Equal-authority object conflicts are audit facts, not permission
            # to traverse both unrelated entities.
            continue
        if binding_status == "exact":
            seen.add(logical_id)
            exact.append(item)
            continue
        if source_tool not in allowed_candidate_tools:
            continue
        if per_tool.get(source_tool, 0) >= _MAX_CANDIDATE_SEEDS_PER_TOOL:
            continue
        per_tool[source_tool] = per_tool.get(source_tool, 0) + 1
        seen.add(logical_id)
        candidates.append(item)
    return [*exact, *candidates][:_MAX_GRAPH_SEEDS]


def _preferred_seed_tools(
    state: Mapping[str, Any],
    available_query_tools: Sequence[str],
) -> List[str]:
    available = set(available_query_tools)
    return [tool for tool in _seed_owner_tools(state) if tool in available]


async def adaptive_tool_route_node(state: Mapping[str, Any], config=None) -> Dict[str, Any]:
    """在首轮工具结果上做一次受限补查决策。

    这里故意只允许一批补查：模型可以自适应补充遗漏证据，同时不会
    形成不可控的 ReAct 无限循环。
    """

    executed = _executed_tools(state)
    if _complete_exact_hmc_answer_available(state):
        summary = "HMC 权威记录已精确命中，无需扩大语义检索或查询多跳关系。"
        return {
            "pending_tools": [],
            "adaptive_graph_requested": False,
            "graph_traversal_request": {},
            "adaptive_round": 1,
            "adaptive_decision": {
                "decision": "answer",
                "next_tools": [],
                "executed_tools": executed,
                "source": "exact_hmc_shortcut",
            },
            "allowed_tools": [
                tool for tool in (state.get("allowed_tools") or [])
                if tool != "graph_traversal"
            ],
            "need_more": False,
            "next_action": "done",
            "reflection": summary,
        }
    authorized = set(state.get("authorized_tools") or state.get("allowed_tools") or [])
    graph_expansion_reason = _graph_expansion_reason(state)
    remaining_query_tools = [
        tool for tool in _TOOL_ORDER
        if tool in QUERY_TOOL_NAMES and tool in authorized and tool not in executed
    ]
    eligible_graph_seeds = _eligible_graph_seed_bindings(state)
    graph_available = bool(
        "graph_traversal" in authorized and eligible_graph_seeds
    )
    available = [*remaining_query_tools]
    if graph_available:
        available.append("graph_traversal")

    if not available:
        summary = f"已检查 {len(executed)} 项查询，当前没有需要或可执行的补充工具。"
        return {
            "pending_tools": [],
            "adaptive_graph_requested": False,
            "graph_traversal_request": {},
            "adaptive_round": 1,
            "adaptive_decision": {
                "decision": "answer",
                "next_tools": [],
                "executed_tools": executed,
                "source": "no_options",
            },
            "need_more": False,
            "next_action": "done",
            "reflection": summary,
        }

    from backend.agent.task_context import current_task
    contract_fields = _required_answer_slots(state)
    available_text = "\n".join(
        f"- {tool}: {_TOOL_PURPOSES[tool]}" for tool in available
    )
    result_summary = _result_summary(state, executed)
    prompt = ADAPTIVE_TOOL_PROMPT.format(
        query=str(state.get("query") or ""),
        targets=json.dumps(current_task(state), ensure_ascii=False),
        executed_tools=", ".join(executed) or "无",
        result_summary=result_summary,
        available_tools=available_text,
        graph_need=graph_expansion_reason or "没有确定性要求；仅在现有证据仍缺少必要连接时选择查图",
        max_followup_tools=MAX_FOLLOWUP_QUERY_TOOLS,
    )

    parsed: Dict[str, Any] | None = None
    model_error = ""
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm_service.call_model,
                [{"role": "user", "content": prompt}],
                _model_for_state(state),
            ),
            timeout=ADAPTIVE_TOOL_TIMEOUT_SECONDS,
        )
        parsed = _parse_json_object(response)
        if parsed is None:
            model_error = "补查路由模型输出不是有效 JSON"
    except asyncio.TimeoutError:
        model_error = f"补查路由模型超过 {ADAPTIVE_TOOL_TIMEOUT_SECONDS:g} 秒时间预算"
    except Exception as exc:
        model_error = f"补查路由模型不可用：{exc}"

    selected: List[str] = []
    if parsed and str(parsed.get("decision") or "") != "answer":
        for tool in _unique_strings(parsed.get("next_tools") or []):
            if tool in available and tool not in selected:
                selected.append(tool)

    if focused_answer_contract(state):
        contract_tools = set(tools_for_answer_fields(contract_fields))
        # The route model may suggest an authorized but irrelevant source.
        # Field contracts are the narrower execution boundary for this turn.
        selected = [tool for tool in selected if tool in contract_tools]

    if not parsed:
        # B1 降级计数：路由输出不可用退回安全候选工具集。
        try:
            from backend.agent.tracing import record_degradation

            record_degradation(
                "adaptive_fallback",
                detail=str(model_error or "route model output unusable")[:200],
            )
        except Exception:
            pass
        selected = _fallback_tools(
            state=state,
            remaining_query_tools=remaining_query_tools,
            graph_available=graph_available,
        )
        if model_error:
            logger.warning(model_error)

    # The route model chooses the most useful next step, but it must not stop a
    # contract answer after finding only one explicitly requested dimension.
    # Add only the bounded, field-owned evidence families that remain available.
    completion_followups = _completion_followups(
        state,
        executed=executed,
        available=remaining_query_tools,
    )
    selected = _unique_strings([*completion_followups, *selected])

    # If cross-document completion is needed but the current anchors are only
    # unrelated keyword candidates, obtain a target-appropriate structured
    # anchor first.  This is a bounded coverage fallback, not a fixed full scan.
    selected_direct_tools = [tool for tool in selected if tool in QUERY_TOOL_NAMES]
    preferred_seed_tools = _preferred_seed_tools(state, remaining_query_tools)
    if (
        graph_expansion_reason
        and not graph_available
        and preferred_seed_tools
        and not any(tool in preferred_seed_tools for tool in selected_direct_tools)
    ):
        for tool in preferred_seed_tools[:2]:
            if tool not in selected:
                selected.append(tool)

    direct_tools = [
        tool for tool in selected if tool in QUERY_TOOL_NAMES
    ][:MAX_FOLLOWUP_QUERY_TOOLS]
    from backend.agent.query_understanding import normalize_tool_queries
    from backend.agent.task_context import task_query
    # Follow-up queries fill the same task's gap; they cannot grant tools,
    # replace exact selectors or mutate dialogue_context/query_frame.
    queries = normalize_tool_queries(state.get("tool_queries"))
    proposed_queries = normalize_tool_queries((parsed or {}).get("tool_queries"))
    for tool in direct_tools:
        queries[tool] = list(dict.fromkeys([
            task_query(state), *proposed_queries.get(tool, []),
        ]))[:4]
    # 明确的语义目标可以自动触发图扩展，不要求用户使用图谱术语，也不
    # 依赖二次路由模型再次猜中同一个需要。没有业务实体时仍然不能立刻
    # 查图；若本轮还有直接补查，则只记录“取得实体后再查图”的受限请求。
    graph_requested = graph_available and bool(
        graph_expansion_reason or "graph_traversal" in selected
    )
    graph_deferred = bool(graph_expansion_reason and not graph_available and direct_tools)
    adaptive_graph_requested = graph_requested or graph_deferred

    current_allowed = [
        tool for tool in (state.get("allowed_tools") or [])
        if tool != "graph_traversal"
    ]
    cumulative_allowed = _unique_strings([
        *current_allowed,
        *direct_tools,
        *(["graph_traversal"] if adaptive_graph_requested else []),
    ])
    request = (
        _graph_request(eligible_graph_seeds, cumulative_allowed)
        if graph_requested and not direct_tools
        else {}
    )
    if direct_tools and adaptive_graph_requested:
        action = "retrieve_then_graph"
    elif direct_tools:
        action = "retrieve"
    elif graph_requested:
        action = "graph"
    else:
        action = "answer"

    public_parts = [f"已检查 {len(executed)} 项查询和 {_evidence_count(state)} 条候选证据"]
    if direct_tools:
        public_parts.append("需要补查 " + "、".join(direct_tools))
    if graph_requested:
        public_parts.append("需要检查受限的跨文档关系")
    elif graph_deferred:
        public_parts.append("获得可追溯业务实体后检查跨文档关系")
    if not direct_tools and not adaptive_graph_requested:
        public_parts.append("现有资料已进入证据核对")
    public_summary = "；".join(public_parts) + "。"

    return {
        "pending_tools": direct_tools,
        "tool_queries": queries,
        "adaptive_graph_requested": adaptive_graph_requested,
        "graph_traversal_request": request,
        "adaptive_round": 1,
        "adaptive_decision": {
            "decision": action,
            "next_tools": [*direct_tools, *(["graph_traversal"] if graph_requested else [])],
            "executed_tools": executed,
            "source": "model" if parsed else "safe_fallback",
            "graph_trigger": graph_expansion_reason,
            "model_reason": str((parsed or {}).get("reason") or "")[:500],
            "completion_tools": completion_followups,
            "warning": model_error,
        },
        "allowed_tools": cumulative_allowed,
        "need_more": bool(direct_tools or graph_requested),
        "next_action": "adaptive_retrieval" if direct_tools else (
            "graph_traversal" if graph_requested else "done"
        ),
        "reflection": public_summary,
    }


def _merge_dict_rows(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    limit: int = 80,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for raw in [*first, *second]:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        identity = tuple(str(item.get(key) or "") for key in keys)
        if not any(identity):
            identity = (json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)[:1000],)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _merge_candidates(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_fault: Dict[str, Dict[str, Any]] = {}
    for raw in [*first, *second]:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        fault = str(item.get("fault") or "").strip()
        if not fault:
            continue
        current = by_fault.get(fault)
        if current is None:
            by_fault[fault] = item
            continue
        current["score"] = max(float(current.get("score") or 0), float(item.get("score") or 0))
        current["evidence"] = _merge_dict_rows(
            current.get("evidence") or [], item.get("evidence") or [], keys=("source", "id"), limit=8
        )
        current["supporting_evidence_ids"] = _unique_strings([
            *(current.get("supporting_evidence_ids") or []),
            *(item.get("supporting_evidence_ids") or []),
        ])[:12]
        for key in ("exact_hmc_match", "structured_hmc_match", "authoritative_support"):
            current[key] = bool(current.get(key) or item.get(key))
    ranked = list(by_fault.values())
    ranked.sort(
        key=lambda item: (
            bool(item.get("exact_hmc_match")),
            bool(item.get("structured_hmc_match")),
            bool(item.get("authoritative_support")),
            float(item.get("score") or 0),
        ),
        reverse=True,
    )
    return ranked[:8]


def _merge_seed_bindings(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in [*first, *second]:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        logical_id = str(item.get("logical_id") or "").strip()
        if not logical_id:
            continue
        current = by_id.get(logical_id)
        if current is None or (
            current.get("binding_status") != "exact" and item.get("binding_status") == "exact"
        ):
            by_id[logical_id] = item
    return list(by_id.values())[:24]


def _merge_structured_entity_binding(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Keep one compatible, strongest object hand-off across adaptive batches.

    ``icp_diagnosis_node`` is also used for the bounded adaptive batch.  A
    batch that does not run Function Locator legitimately returns an unbound
    value, but that absence must not erase an exact binding established by the
    first batch.  When both batches bind the same object family, merge their
    IDs/terms.  For disjoint IDs, the more authoritative binding wins; equal
    or unknown authority fails closed instead of inventing an entity union.
    Every conflict remains attached to the result for audit/debugging.
    """

    previous = dict(first or {})
    current = dict(second or {})
    previous_bound = previous.get("bound") is True
    current_bound = current.get("bound") is True

    def normalized_ids(binding: Mapping[str, Any]) -> set[str]:
        return {
            str(value or "").strip().upper()
            for value in (binding.get("business_ids") or [])
            if str(value or "").strip()
        }

    def inherited_conflicts(*bindings: Mapping[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for binding in bindings:
            for raw in binding.get("binding_conflicts") or []:
                if not isinstance(raw, Mapping):
                    continue
                item = dict(raw)
                if item not in result:
                    result.append(item)
        return result[:8]

    if not current_bound:
        return dict(previous if previous_bound else current)
    if not previous_bound:
        return dict(current)

    previous_ids = normalized_ids(previous)
    current_ids = normalized_ids(current)
    if not previous_ids or not current_ids or not (previous_ids & current_ids):
        previous_basis = str(previous.get("basis") or "").strip()
        current_basis = str(current.get("basis") or "").strip()
        previous_rank = binding_basis_rank(previous_basis)
        current_rank = binding_basis_rank(current_basis)
        conflict = {
            "reason": (
                "missing_business_ids"
                if not previous_ids or not current_ids
                else "disjoint_business_ids"
            ),
            "previous": {
                "basis": previous_basis,
                "rank": previous_rank,
                "business_ids": sorted(previous_ids),
            },
            "current": {
                "basis": current_basis,
                "rank": current_rank,
                "business_ids": sorted(current_ids),
            },
        }
        conflicts = [*inherited_conflicts(previous, current), conflict][-8:]
        if previous_rank > current_rank:
            conflict["resolution"] = "kept_stronger_previous"
            result = dict(previous)
            result["binding_conflicts"] = conflicts
            return result
        if current_rank > previous_rank:
            conflict["resolution"] = "kept_stronger_current"
            result = dict(current)
            result["binding_conflicts"] = conflicts
            return result
        conflict["resolution"] = "ambiguous_equal_or_unknown_authority"
        return {
            "bound": False,
            "ambiguous": True,
            "business_ids": [],
            "terms": [],
            "row_indexes": [],
            "basis": "ambiguous_adaptive_binding_conflict",
            "binding_conflicts": conflicts,
        }

    merged = dict(previous)
    for key in ("business_ids", "terms", "row_indexes"):
        values: List[Any] = []
        seen: set[str] = set()
        for value in [*(previous.get(key) or []), *(current.get(key) or [])]:
            identity = str(value).strip().casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            values.append(value)
        merged[key] = values
    merged["bound"] = True
    merged["ambiguous"] = False
    previous_basis = str(previous.get("basis") or "").strip()
    current_basis = str(current.get("basis") or "").strip()
    merged["basis"] = (
        current_basis
        if binding_basis_rank(current_basis) > binding_basis_rank(previous_basis)
        else previous_basis or current_basis
    )
    conflicts = inherited_conflicts(previous, current)
    if conflicts:
        merged["binding_conflicts"] = conflicts
    return merged


def _merge_structured_record_binding(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Preserve one exact record target across bounded adaptive batches.

    An adaptive batch often sees only a subset of the first batch's rows.  A
    unique result in that subset therefore cannot resolve an earlier tie (or
    erase a newly discovered tie).  Any ambiguity fails closed and carries all
    candidate identities forward for audit.  Only compatible bound targets are
    merged as one record.
    """

    previous = dict(first or {})
    current = dict(second or {})
    previous_bound = previous.get("bound") is True
    current_bound = current.get("bound") is True
    previous_ambiguous = previous.get("ambiguous") is True
    current_ambiguous = current.get("ambiguous") is True
    previous_relevance_only = previous.get("relevance_only") is True
    current_relevance_only = current.get("relevance_only") is True

    if previous_ambiguous or current_ambiguous:
        candidate_entries: List[Dict[str, Any]] = []
        seen_candidates: set[tuple[str, tuple[str, ...], str]] = set()

        def append_candidates(binding: Mapping[str, Any]) -> None:
            doc_type = str(binding.get("doc_type") or "").strip()
            raw_candidates = binding.get("candidates") or []
            candidates = (
                raw_candidates
                if isinstance(raw_candidates, Sequence)
                and not isinstance(raw_candidates, (str, bytes))
                else []
            )
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_ids = candidate.get("record_ids")
                if candidate_ids is None:
                    candidate_ids = [candidate.get("record_id")]
                normalized_ids = tuple(
                    sorted(
                        {
                            str(value or "").strip().upper()
                            for value in (candidate_ids or [])
                            if str(value or "").strip()
                        }
                    )
                )
                candidate_type = str(candidate.get("doc_type") or doc_type).strip()
                basis = str(candidate.get("basis") or binding.get("basis") or "").strip()
                identity = (candidate_type, normalized_ids, basis)
                if not normalized_ids or identity in seen_candidates:
                    continue
                seen_candidates.add(identity)
                candidate_entries.append(
                    {
                        "doc_type": candidate_type,
                        "record_ids": list(normalized_ids),
                        "basis": basis,
                    }
                )
            bound_ids = sorted(
                {
                    str(value or "").strip().upper()
                    for value in (binding.get("record_ids") or [])
                    if str(value or "").strip()
                }
            )
            identity = (
                doc_type,
                tuple(bound_ids),
                str(binding.get("basis") or "").strip(),
            )
            if bound_ids and identity not in seen_candidates:
                seen_candidates.add(identity)
                candidate_entries.append(
                    {
                        "doc_type": doc_type,
                        "record_ids": bound_ids,
                        "basis": identity[2],
                    }
                )

        append_candidates(previous)
        append_candidates(current)
        candidate_types = {
            str(item.get("doc_type") or "").strip()
            for item in candidate_entries
            if str(item.get("doc_type") or "").strip()
        }
        return {
            "doc_type": candidate_types.pop() if len(candidate_types) == 1 else "",
            "bound": False,
            "ambiguous": True,
            "record_ids": [],
            "row_indexes": [],
            "basis": "ambiguous_adaptive_record_binding_conflict",
            "candidates": candidate_entries,
        }

    if previous_relevance_only or current_relevance_only:
        # Relevance is not identity, but it is a sticky safety boundary.  A
        # later batch that did not run FMECA returns an empty default binding;
        # that absence must not erase the first batch's graph prohibition.
        # A genuinely exact current/previous binding may supersede relevance.
        if current_bound:
            return current
        if previous_bound:
            return previous
        candidates: List[Dict[str, Any]] = []
        seen_candidates: set[tuple[str, str]] = set()
        for binding in (previous, current):
            for raw in binding.get("candidates") or []:
                if not isinstance(raw, Mapping):
                    continue
                item = dict(raw)
                identity = (
                    str(item.get("record_id") or "").strip().upper(),
                    str(item.get("row_index") or "0"),
                )
                if not identity[0] or identity in seen_candidates:
                    continue
                seen_candidates.add(identity)
                candidates.append(item)
        source = previous if previous_relevance_only else current
        merged = dict(source)
        merged.update(
            {
                "bound": False,
                "ambiguous": False,
                "record_ids": [],
                "row_indexes": [],
                "basis": "literal_field_relevance_only",
                "candidates": candidates[:16],
                "relevance_only": True,
            }
        )
        return merged

    if not current_bound:
        return previous if previous_bound else current
    if not previous_bound:
        return current
    previous_ids = {
        str(value or "").strip().upper()
        for value in (previous.get("record_ids") or [])
        if str(value or "").strip()
    }
    current_ids = {
        str(value or "").strip().upper()
        for value in (current.get("record_ids") or [])
        if str(value or "").strip()
    }
    previous_type = str(previous.get("doc_type") or "").strip()
    current_type = str(current.get("doc_type") or "").strip()
    if (
        previous_ids
        and current_ids
        and previous_ids == current_ids
        and previous_type == current_type
    ):
        merged = dict(previous)
        merged.update(current)
        merged["record_ids"] = list(previous.get("record_ids") or current.get("record_ids") or [])
        return merged
    return {
        "doc_type": previous_type if previous_type == current_type else "",
        "bound": False,
        "ambiguous": True,
        "record_ids": [],
        "row_indexes": [],
        "basis": "ambiguous_adaptive_record_binding_conflict",
        "candidates": [
            {
                "doc_type": previous_type,
                "record_ids": sorted(previous_ids),
                "basis": str(previous.get("basis") or ""),
            },
            {
                "doc_type": current_type,
                "record_ids": sorted(current_ids),
                "basis": str(current.get("basis") or ""),
            },
        ],
    }


def _merge_hmc_mfl_binding(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Keep an explicit HMC-to-MFL ownership decision across batches."""

    previous = dict(first or {})
    current = dict(second or {})
    previous_required = previous.get("required") is True
    current_required = current.get("required") is True
    if not previous_required and not current_required:
        return current
    # Treat a batch which did not exercise this relationship as a neutral
    # required record, then run the same structural validation as a two-sided
    # merge.  Returning the other dictionary verbatim would let a malformed
    # ``bound=True`` flag bypass validation merely because the adaptive batch
    # happened not to query HMC/MFL.
    if not previous_required:
        previous = {"required": True}
    if not current_required:
        current = {"required": True}

    def values(binding: Mapping[str, Any], key: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for value in binding.get(key) or []:
            rendered = str(value or "").strip()
            if rendered:
                result.setdefault(rendered.casefold(), rendered)
        return result

    def rendered(values_by_id: Mapping[str, str]) -> List[str]:
        return [values_by_id[key] for key in sorted(values_by_id)]

    previous_hmc = values(previous, "hmc_codes")
    current_hmc = values(current, "hmc_codes")
    all_hmc = {**current_hmc, **previous_hmc}
    all_mfl = {
        **values(current, "mfl_codes"),
        **values(previous, "mfl_codes"),
    }
    trusted_mfl: Dict[str, str] = {}
    for binding in (current, previous):
        binding_hmc = values(binding, "hmc_codes")
        binding_mfl = values(binding, "mfl_codes")
        if (
            binding.get("bound") is True
            and str(binding.get("basis") or "").strip()
            == "explicit_authoritative_reference"
            and binding_hmc
            and binding_mfl
        ):
            trusted_mfl.update(binding_mfl)
    all_excluded = {
        **values(current, "excluded_mfl_codes"),
        **values(previous, "excluded_mfl_codes"),
        **{
            identity: value
            for identity, value in all_mfl.items()
            if identity not in trusted_mfl
        },
    }
    if previous.get("ambiguous") is True or current.get("ambiguous") is True:
        return {
            "required": True,
            "bound": False,
            "ambiguous": True,
            "basis": "ambiguous_adaptive_hmc_mfl_binding",
            "hmc_codes": rendered(all_hmc),
            "mfl_codes": [],
            "excluded_mfl_codes": rendered({**all_excluded, **all_mfl}),
        }
    if (
        previous_hmc
        and current_hmc
        and not (set(previous_hmc) & set(current_hmc))
    ):
        return {
            "required": True,
            "bound": False,
            "ambiguous": True,
            "basis": "ambiguous_adaptive_hmc_target_conflict",
            "hmc_codes": rendered(all_hmc),
            "mfl_codes": [],
            "excluded_mfl_codes": rendered({**all_excluded, **all_mfl}),
        }

    excluded = {
        identity: value
        for identity, value in all_excluded.items()
        if identity not in trusted_mfl
    }
    return {
        "required": True,
        "bound": bool(all_hmc and trusted_mfl),
        "ambiguous": False,
        "basis": (
            "explicit_authoritative_reference"
            if all_hmc and trusted_mfl
            else "no_explicit_authoritative_relation"
        ),
        "hmc_codes": rendered(all_hmc),
        "mfl_codes": rendered(trusted_mfl),
        "excluded_mfl_codes": rendered(excluded),
    }


def _merge_rbd_query_binding(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Delegate every adaptive batch to the shared RBD authority contract."""

    return merge_rbd_query_bindings(first, second)


async def adaptive_retrieval_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """执行自适应路由选中的唯一一批补查，并与首轮状态合并。"""

    pending = [
        tool for tool in _unique_strings(state.get("pending_tools") or [])
        if tool in QUERY_TOOL_NAMES
    ][:MAX_FOLLOWUP_QUERY_TOOLS]
    if not pending:
        return {"pending_tools": []}

    batch_state = dict(state)
    batch_state["allowed_tools"] = [*pending, "evidence_synthesis"]
    # A hybrid-only follow-up may discover and PostgreSQL-hydrate the exact
    # Function Locator row that the first direct batch could not name.  Carry
    # the already-authorized direct tools as a *hydration-only* ceiling so the
    # diagnosis node may re-read an exact linked record (for example FUNC-310
    # -> FMECA) without rerunning a broad tool or expanding planner authority.
    batch_state["structured_hydration_tools"] = list(
        state.get("allowed_tools") or []
    )
    if "hybrid_search" in pending:
        batch_state["hybrid_doc_types"] = list(state.get("authorized_doc_types") or [])
    else:
        batch_state["hybrid_doc_types"] = []
    update = await icp_diagnosis_node(batch_state)

    merged_seeds = _merge_seed_bindings(
        state.get("graph_seed_bindings") or [],
        update.get("graph_seed_bindings") or [],
    )
    cumulative_allowed = _unique_strings(state.get("allowed_tools") or [])
    graph_requested = bool(state.get("adaptive_graph_requested"))
    merged_state = dict(state)
    merged_state["graph_seed_bindings"] = merged_seeds
    # Candidate graph anchors are eligible only after their owner tool has
    # actually run.  Include this adaptive batch in that audit trail before
    # evaluating the newly returned bindings.
    merged_state["tool_results"] = [
        *(state.get("tool_results") or []),
        *(update.get("tool_results") or []),
    ]
    update["graph_seed_bindings"] = merged_seeds
    update["structured_entity_binding"] = _merge_structured_entity_binding(
        state.get("structured_entity_binding")
        if isinstance(state.get("structured_entity_binding"), Mapping)
        else {},
        update.get("structured_entity_binding")
        if isinstance(update.get("structured_entity_binding"), Mapping)
        else {},
    )
    merged_state["structured_entity_binding"] = update[
        "structured_entity_binding"
    ]
    update["structured_record_binding"] = _merge_structured_record_binding(
        state.get("structured_record_binding")
        if isinstance(state.get("structured_record_binding"), Mapping)
        else {},
        update.get("structured_record_binding")
        if isinstance(update.get("structured_record_binding"), Mapping)
        else {},
    )
    merged_state["structured_record_binding"] = update[
        "structured_record_binding"
    ]
    update["history_parent_binding"] = merge_history_parent_bindings(
        state.get("history_parent_binding")
        if isinstance(state.get("history_parent_binding"), Mapping)
        else {},
        update.get("history_parent_binding")
        if isinstance(update.get("history_parent_binding"), Mapping)
        else {},
    )
    merged_state["history_parent_binding"] = update[
        "history_parent_binding"
    ]
    update["hmc_mfl_binding"] = _merge_hmc_mfl_binding(
        state.get("hmc_mfl_binding")
        if isinstance(state.get("hmc_mfl_binding"), Mapping)
        else {},
        update.get("hmc_mfl_binding")
        if isinstance(update.get("hmc_mfl_binding"), Mapping)
        else {},
    )
    merged_state["hmc_mfl_binding"] = update["hmc_mfl_binding"]
    update["rbd_query_binding"] = _merge_rbd_query_binding(
        state.get("rbd_query_binding")
        if isinstance(state.get("rbd_query_binding"), Mapping)
        else {},
        update.get("rbd_query_binding")
        if isinstance(update.get("rbd_query_binding"), Mapping)
        else {},
    )
    merged_state["rbd_query_binding"] = update["rbd_query_binding"]
    eligible_graph_seeds = _eligible_graph_seed_bindings(merged_state)
    update["graph_traversal_request"] = (
        _graph_request(eligible_graph_seeds, cumulative_allowed) if graph_requested else {}
    )
    update["candidate_faults"] = _merge_candidates(
        state.get("candidate_faults") or [], update.get("candidate_faults") or []
    )
    update["diagnostic_support"] = max(
        float(state.get("diagnostic_support") or 0),
        float(update.get("diagnostic_support") or 0),
    )
    update["diagnostic_confidence"] = update["diagnostic_support"]
    update["domain_results"] = _merge_dict_rows(
        state.get("domain_results") or [],
        update.get("domain_results") or [],
        keys=("source_type", "doc_id", "chunk_id", "title"),
    )
    update["search_locator_hits"] = _merge_dict_rows(
        state.get("search_locator_hits") or [],
        update.get("search_locator_hits") or [],
        keys=("doc_type_kwd", "diagram_id_kwd", "doc_id", "business_id_kwd"),
        limit=16,
    )
    update["vector_results"] = list(update["domain_results"])
    update["confirmed_graphs"] = _merge_dict_rows(
        state.get("confirmed_graphs") or [],
        update.get("confirmed_graphs") or [],
        keys=("doc_id", "revision_id", "diagram_id"),
    )
    update["fta_query_binding"] = merge_diagram_scopes(state.get("fta_query_binding"), update.get("fta_query_binding"))
    if update["fta_query_binding"]:
        update["confirmed_graphs"] = [row for row in update["confirmed_graphs"]
            if str(row.get("diagram_type") or "") != "fta"
            or diagram_in_scope(row, update["fta_query_binding"], query=str(state.get("query") or ""))]
    update["allowed_tools"] = cumulative_allowed
    update["pending_tools"] = []
    return update


def route_after_adaptive_tools(state: Mapping[str, Any]) -> str:
    if state.get("pending_tools"):
        return "adaptive_retrieval"
    if state.get("graph_traversal_request", {}).get("seed_ids"):
        return "graph_traversal"
    return "icp_calculation"


def route_after_adaptive_retrieval(state: Mapping[str, Any]) -> str:
    if state.get("graph_traversal_request", {}).get("seed_ids"):
        return "graph_traversal"
    return "icp_calculation"


__all__ = [
    "adaptive_retrieval_node",
    "adaptive_tool_route_node",
    "route_after_adaptive_retrieval",
    "route_after_adaptive_tools",
]
