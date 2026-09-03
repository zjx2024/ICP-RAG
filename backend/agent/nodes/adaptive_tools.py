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
    required_answer_fields,
    tools_for_answer_fields,
)
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.tool_policy import QUERY_TOOL_NAMES
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

_INTENT_REQUIRED_SLOTS = {
    "entity_profile": ("entity_profile",),
    "symptom_diagnosis": ("cause", "impact", "detect"),
    "cause_analysis": ("cause",),
    "impact_analysis": ("impact",),
    "maintenance_query": ("repair",),
    "history_query": _HISTORY_SLOTS,
    "architecture_query": ("architecture_path",),
    "function_location": ("architecture_path",),
    "reliability_query": ("rbd",),
}

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

_FAULT_COMPLETION_INTENTS = {
    "symptom_diagnosis",
    "cause_analysis",
    "impact_analysis",
    "maintenance_query",
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
        if claim_type == "fta_structure":
            coverage.add("structural_cause_chain")
        if claim_type in {"failure_impact", "impact_fact"}:
            coverage.add("impact")
        if claim_type in {"detection_method", "diagnosis_method"}:
            coverage.add("detect")
        if claim_type == "maintenance_action":
            if doc_type == "hmc":
                coverage.add("hmc_repair")
            elif doc_type == "mfl":
                coverage.add("mfl_repair")
        if claim_type in {"interface_fact", "relation_path", "architecture_path"}:
            coverage.add("architecture_path")
        if claim_type == "relation_path":
            coverage.add("impact_propagation")
        if claim_type in {"rbd_structure", "rbd_reliability", "rbd_calculation"}:
            coverage.add("rbd")

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
    """Return explicit user-goal slots in query-target order."""

    contract_fields = required_answer_fields(state)
    if contract_fields:
        return contract_fields

    result: List[str] = []
    targets = [
        item for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping)
    ]
    if not targets:
        targets = [{"intent": state.get("intent")}]
    for target in targets:
        # Future target classifiers may explicitly emit field contracts.  Use
        # them when present, while keeping the intent defaults for compatibility.
        declared = target.get("answer_fields") or target.get("required_fields")
        declared_slots: List[str] = []
        if isinstance(declared, str):
            declared = [declared]
        if isinstance(declared, (list, tuple, set, frozenset)):
            for value in declared:
                key = _normalized_field_name(value)
                if key == "history":
                    declared_slots.extend(_HISTORY_SLOTS)
                elif key in _ANSWER_FIELD_ALIASES or key == "repair":
                    declared_slots.append(key)
        slots = declared_slots or list(
            _INTENT_REQUIRED_SLOTS.get(str(target.get("intent") or ""), ())
        )
        for slot in slots:
            if slot not in result:
                result.append(slot)
    return result


def _slot_is_covered(slot: str, coverage: set[str]) -> bool:
    if slot == "repair":
        return bool(coverage & {"hmc_repair", "mfl_repair"})
    return slot in coverage


def _tools_for_slot(slot: str) -> tuple[str, ...]:
    contract_tools = tuple(
        tool
        for tool in tools_for_answer_fields([slot])
        if tool in QUERY_TOOL_NAMES
    )
    if contract_tools:
        return contract_tools
    return {
        "cause": ("fmeca_query", "fta_query"),
        "impact": ("fmeca_query",),
        "detect": ("fmeca_query",),
        "repair": ("mfl_query", "hmc_query"),
        "hmc_repair": ("hmc_query",),
        "mfl_repair": ("mfl_query",),
        "mfl_api": ("mfl_query",),
        "history_time": ("fault_history_query",),
        "history_root": ("fault_history_query",),
        "history_steps": ("fault_history_query",),
        "history_solution": ("fault_history_query",),
        "history_result": ("fault_history_query",),
        "history_personnel": ("fault_history_query",),
        "history_preventive_action": ("fault_history_query",),
        "architecture_path": ("function_locator",),
        "entity_profile": ("function_locator",),
        "rbd": ("rbd_query",),
    }.get(slot, ())

ADAPTIVE_TOOL_PROMPT = """你是 ICP 诊断智能体的工具路由器，只决定是否需要补查，不回答用户问题。

用户问题：{query}
本轮明确目标：{targets}
已执行工具：{executed_tools}
已有结果：
{result_summary}

当前仍可选的工具：
{available_tools}

系统识别的跨文档扩展需要：
{graph_need}

约束：
1. 先回答用户当前明确目标；若系统给出了字段级必答合同，只补合同仍缺少的资料。只有首次模糊诊断或用户明确要求完整分析时，才补原因、影响、处置和相似历史。
2. 已有权威直接证据覆盖所需信息时选 answer；OpenSearch 检索投影的数量本身不代表答案已经完整。
3. 判断 graph_traversal 看问题的实际信息需求，不看用户有没有说“关系、路径、关联、图谱”。如果原因定位、故障诊断、架构定位或多个信息目标需要把不同文档实体串起来，就应查图；精确代码的单字段查询、已有直接证据足够的普通解释不查图。
4. 只能从“当前仍可选的工具”中选择；不得输出实体 ID、SQL、Gremlin、计算器或新的信息目标。
5. 补查最多选 {max_followup_tools} 个直接查询工具，同一工具不重复执行。

只输出 JSON：
{{
  "decision": "answer|retrieve|graph|retrieve_then_graph",
  "next_tools": ["fmeca_query"],
  "reason": "简短说明为什么需要或不需要补查"
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

    The previous implementation appended every evidence family associated with
    an intent.  That reproduced most of the old fixed full-scan workflow and
    increased noise even when an authoritative result was already available.
    Here the answer contract is expressed as evidence *slots*: diagnosis,
    cause chain, handling, history, architecture and reliability.  A tool is
    added only while its slot is still empty.
    """

    available_set = set(available)
    executed_set = set(executed)
    trusted = [
        item
        for item in (state.get("evidence_registry") or [])
        if isinstance(item, Mapping)
        and _can_complete_answer_slot(item)
    ]
    coverage = _coverage_slots(state)
    has_any_trusted = bool(trusted)

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

    intents = set(_target_intents(state))
    focused_contract = focused_answer_contract(state)
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    collection_contract = str(frame.get("coverage_scope") or "") == "complete_collection"
    if (
        intents & _FAULT_COMPLETION_INTENTS
        and not focused_contract
        and not collection_contract
    ):
        # Fault-oriented answers always get one bounded structured FMECA lookup
        # unless it already ran.  A semantic hydration can prove that one record
        # exists, but it may cover only one side of a comparison (for example,
        # line-open versus A/D saturation) and therefore cannot replace the
        # table query used to assemble the cause/effect alternatives.
        # This is a single structured follow-up, not a broad full scan; add()
        # also prevents repeating fmeca_query when it already ran this round.
        add("fmeca_query")
    if (
        intents & _FAULT_COMPLETION_INTENTS
        and not focused_contract
        and not collection_contract
    ):
        # A fault answer should include documented handling when available.
        # MFL API coordinates alone are not a repair, so they intentionally do
        # not satisfy this condition.
        if not (coverage & {"hmc_repair", "mfl_repair"}):
            add("mfl_query")
            add("hmc_query")
        # A history row is only complete when it can answer when, why, what was
        # done, the final solution and the verified result.
        if not all(slot in coverage for slot in _HISTORY_SLOTS):
            add("fault_history_query")

    # Hybrid retrieval is a discovery fallback.  It must not run merely to add
    # volume after structured sources have already covered the answer.
    if not has_any_trusted and not result:
        add("hybrid_search")
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
    return prepared


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


def _complete_exact_hmc_answer_available(state: Mapping[str, Any]) -> bool:
    """完整 HMC 精确命中已经满足“这条记录是什么”。

    这个确定性快速路径只适用于单一 exact_code 目标。一旦用户同时
    询问原因、维修或关联路径，仍交给自适应路由决定是否补查。
    """

    targets = {
        str(item.get("intent") or "")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping) and str(item.get("intent") or "")
    }
    if str(state.get("intent") or "") != "exact_code" or targets - {"exact_code"}:
        return False
    hmc_ok = any(
        isinstance(item, Mapping)
        and str(item.get("tool") or "") == "hmc_query"
        and str(item.get("status") or "") in {"ok", "partial"}
        for item in (state.get("tool_results") or [])
    )
    exact_evidence = any(
        isinstance(item, Mapping)
        and str(item.get("doc_type") or "") == "hmc"
        and str(item.get("match_type") or "") == "exact_code"
        for item in (state.get("evidence_registry") or [])
    )
    return hmc_ok and exact_evidence


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

    previews: List[str] = []
    seen = set()
    for item in state.get("evidence_registry", []) or []:
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and evidence_id in seen:
            continue
        if evidence_id:
            seen.add(evidence_id)
        title = str(item.get("title") or item.get("doc_name") or "").strip()
        summary = str(item.get("display_summary") or "").strip()
        if not summary:
            content = str(item.get("content") or "").strip()
            summary = "" if content.startswith(("{", "[")) else content
        text = " ".join(part for part in (title, summary[:220]) if part).strip()
        if text:
            previews.append(text[:280])
        if len(previews) >= 8:
            break
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


_IMPLICIT_GRAPH_INTENTS = {
    "architecture_query",
    "cause_analysis",
    "symptom_diagnosis",
}

_CANDIDATE_SEED_TOOLS_BY_INTENT = {
    "entity_profile": ("function_locator",),
    "architecture_query": ("function_locator",),
    "function_location": ("function_locator",),
    "symptom_diagnosis": (
        "function_locator", "fmeca_query", "fta_query", "mfl_query",
        "fault_history_query",
    ),
    "cause_analysis": ("fmeca_query", "fta_query", "function_locator"),
    "impact_analysis": ("fmeca_query", "function_locator", "fta_query"),
    "maintenance_query": (
        "mfl_query", "fault_history_query", "function_locator", "fmeca_query",
    ),
    "reliability_query": ("rbd_query", "function_locator"),
    "history_query": (
        "fault_history_query", "fmeca_query", "mfl_query", "function_locator",
    ),
    "hybrid": (
        "function_locator", "fmeca_query", "fta_query", "rbd_query",
        "mfl_query", "fault_history_query",
    ),
}

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
    targets = {
        str(item.get("intent") or "")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping) and str(item.get("intent") or "")
    }
    intent = str(state.get("intent") or "")
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
    if len(targets) >= 2:
        return "当前问题同时包含多个信息目标，需要检查它们在不同文档中的连接"
    if intent in _IMPLICIT_GRAPH_INTENTS:
        return {
            "architecture_query": "架构定位需要补全实体层级、功能和上下游连接",
            "cause_analysis": "原因分析需要检查故障、原因、功能和模块之间的证据链",
            "symptom_diagnosis": "模糊故障定位需要检查候选现象在多类资料中的交叉连接",
        }[intent]
    return ""


def _target_intents(state: Mapping[str, Any]) -> List[str]:
    targets = _unique_strings(
        item.get("intent")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping)
    )
    intent = str(state.get("intent") or "").strip()
    if intent and intent not in targets:
        targets.insert(0, intent)
    return targets


def _candidate_seed_tools(state: Mapping[str, Any]) -> List[str]:
    result: List[str] = []
    for intent in _target_intents(state):
        for tool in _CANDIDATE_SEED_TOOLS_BY_INTENT.get(intent, ()):
            if tool not in result:
                result.append(tool)
    return result


def _eligible_graph_seed_bindings(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Select bounded graph anchors whose origin matches the information goal.

    A complete HMC code is an exact identity and is always eligible.  Keyword
    HMC matches are deliberately excluded: a shared word such as “外设” must
    not turn an unrelated HMC row into the center of a multi-hop answer.
    Other semantic candidates are allowed only when their source tool is
    appropriate for the planner's structured target, and are capped per source
    so one broad table query cannot monopolize the traversal.
    """

    allowed_candidate_tools = set(_candidate_seed_tools(state))
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
    return [tool for tool in _candidate_seed_tools(state) if tool in available]


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

    targets = [
        str(item.get("intent") or "")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping) and str(item.get("intent") or "")
    ]
    contract_fields = required_answer_fields(state)
    available_text = "\n".join(
        f"- {tool}: {_TOOL_PURPOSES[tool]}" for tool in available
    )
    result_summary = _result_summary(state, executed)
    prompt = ADAPTIVE_TOOL_PROMPT.format(
        query=str(state.get("query") or ""),
        targets=", ".join([*targets, *(f"必答字段:{field}" for field in contract_fields)])
        or str(state.get("intent") or "hybrid"),
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
                llm_service._call_model,
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
        selected = _fallback_tools(
            state=state,
            remaining_query_tools=remaining_query_tools,
            graph_available=graph_available,
        )
        if model_error:
            logger.warning(model_error)

    # The route model chooses the most useful next step, but it must not stop a
    # fault answer after finding only one dimension (for example, a cause with
    # no documented handling or historical outcome). Add only the bounded,
    # intent-specific evidence families that remain available.
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
    eligible_graph_seeds = _eligible_graph_seed_bindings(merged_state)
    update["graph_seed_bindings"] = merged_seeds
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
