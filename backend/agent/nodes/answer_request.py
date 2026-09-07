"""Validated current-turn request views and presentation boundaries.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import required_answer_fields
from backend.agent.nodes.answer_field_catalog import _INCREMENTAL_RESPONSE_FOCUS_FIELDS
from backend.agent.nodes.answer_text import _normalize_generated_markdown
from backend.agent.tool_policy import DOC_TYPE_BY_TOOL
from backend.agent.tool_policy import _SOURCE_COVERAGE_FIELDS
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
import json
import re


from backend.agent.task_context import current_task as _answer_task_context
from backend.agent.task_context import task_query as _answer_query_text


def _current_user_conditions(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose supported user conditions, explicitly leaving absent ones open."""
    from backend.agent.user_context_binding import selected_user_input_quotes
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    quotes = selected_user_input_quotes(state)
    source_texts = [str(state.get("query") or ""), *quotes]
    result: Dict[str, Any] = {"selected_user_inputs": quotes, "authority": "unverified_user_input"}
    for key in ("operating_stage", "temporal_pattern"):
        basis = frame.get(key + "_basis")
        valid = isinstance(basis, str) and bool(basis.strip()) and any(basis in text for text in source_texts)
        result[key] = frame.get(key) if valid else "未知"
        result[key + "_basis"] = basis if valid else ""
    return result


def _incremental_closed_collection_followup(state: Mapping[str, Any]) -> bool:
    """Recognize an incremental collection turn from the planner contract.

    Count and completeness wording are open discourse semantics.  The answer
    layer therefore consumes only the planner's validated enums; it must not
    require a legacy delivery flag, maintain a question-phrase dictionary or emit
    canned prose for those meanings.
    """

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    focuses = {
        str(value or "").strip()
        for value in (frame.get("response_focus") or [])
        if str(value or "").strip()
    }
    return bool(
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and str(frame.get("response_scope") or "") == "incremental"
        and focuses & {"count", "completeness_check"}
    )


def _incremental_response_focus_fields(
    state: Mapping[str, Any],
    focuses: set[str],
    required_ordered: Sequence[str],
) -> List[str] | None:
    """Map this turn's explicit response slices to their canonical fields.

    Intent defaults may append broad fields such as a maintenance locator or a
    historical root cause.  They remain retrieval/completion context, while
    this mapping records only the fields relevant to each requested slice.
    Open semantic slices still require the semantic judge; this mapping merely
    attaches their supporting evidence.  An unknown focus or one without a
    corresponding current field fails closed.
    """

    required = set(required_ordered)
    expected: List[str] = []
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    source_obligations = [
        item
        for item in (frame.get("source_obligations") or [])
        if isinstance(item, Mapping)
    ]
    for focus in focuses:
        if focus == "source_limitation":
            # An epistemic boundary has no single affirmative source field.
            continue
        if focus == "source_capability":
            source_fields = {
                str(field or "").strip()
                for item in source_obligations
                for field in (item.get("fields") or [])
                if str(field or "").strip()
            }
            mapped = [
                field
                for field in required_ordered
                if not source_fields or field in source_fields
            ]
        else:
            candidates = _INCREMENTAL_RESPONSE_FOCUS_FIELDS.get(focus)
            if candidates is None:
                return None
            mapped = [field for field in candidates if field in required]
        if not mapped:
            # A source-boundary question can explicitly ask about a dimension
            # which this source does not carry (for example root cause in an
            # RBD).  In that case ``source_limitation`` is the answer contract;
            # requiring a nonexistent affirmative field would suppress the
            # deterministic boundary altogether.
            if "source_limitation" in focuses and focus in _INCREMENTAL_RESPONSE_FOCUS_FIELDS:
                continue
            return None
        for field in mapped:
            if field not in expected:
                expected.append(field)
    return expected




def _diagnostic_closure_required(state: Mapping[str, Any]) -> bool:
    """Return the planner-approved current-incident answer contract.

    This is intentionally not a raw wording classifier.  The planning model
    has already separated a detached engineering explanation from a user
    reporting a present abnormality; answer-time code only consumes that
    structured decision to keep the final response from losing a documented
    handling step behind a generic "check" sentence.
    """

    frame = state.get("query_frame")
    if not isinstance(frame, Mapping):
        return False
    contract = frame.get("diagnostic_closure")
    return bool(
        isinstance(contract, Mapping)
        and contract.get("required") is True
        and str(contract.get("basis") or "").strip()
        and required_answer_fields(
            {"query_frame": {"required_answer_fields": contract.get("required_dimensions")}}
        )
    )


def _is_fta_structure_only_request(state: Mapping[str, Any]) -> bool:
    """Return whether this turn asks to explain an FTA structure only.

    An FTA structure follow-up is still routed through the diagnostic graph so
    the requested tree can be found, but it is not a request to diagnose the
    current incident.  Treating it as a normal diagnostic closure causes the
    terminal fault/history completion helpers to append nearby HMC, FMECA or
    maintenance rows that the user did not ask for.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    required_fields = set(required_answer_fields(dict(state)))
    response_mode = (
        frame.get("response_mode")
        if isinstance(frame.get("response_mode"), Mapping)
        else {}
    )
    if (
        str(response_mode.get("mode") or "") != "structure_only"
        or not str(response_mode.get("basis") or "").strip()
    ):
        return False
    structural_fields = {
        "structural_cause_chain",
        "relation_path",
    }
    fta_source_fields = {
        str(field or "").strip()
        for obligation in (frame.get("source_obligations") or [])
        if isinstance(obligation, Mapping)
        and (
            str(obligation.get("tool") or "") == "fta_query"
            or str(obligation.get("doc_type") or "") == "fta"
        )
        for field in (obligation.get("fields") or [])
        if str(field or "").strip()
    }
    declared_fields = required_fields | fta_source_fields
    return bool(
        "structural_cause_chain" in declared_fields
        and not (required_fields - structural_fields)
        and not state.get("calculation_requests")
    )


def _source_obligation_map(
    state: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return bounded, planner-authored facet obligations per source tool."""

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    obligations: Dict[str, Dict[str, Any]] = {}
    for raw in frame.get("source_obligations") or []:
        if not isinstance(raw, Mapping):
            continue
        tool = str(raw.get("tool") or "").strip()
        doc_type = str(raw.get("doc_type") or DOC_TYPE_BY_TOOL.get(tool) or "").strip()
        if tool not in DOC_TYPE_BY_TOOL or not doc_type:
            continue
        obligations[tool] = {
            "doc_type": doc_type,
            "fields": [
                str(value or "").strip()
                for value in (raw.get("fields") or [])
                if str(value or "").strip()
            ],
        }
    # Backward compatibility for in-flight states produced before the
    # facet-scoped contract was introduced.
    requested = (
        state.get("requested_source_tools")
        or frame.get("requested_source_tools")
        or []
    )
    for raw_tool in requested:
        tool = str(raw_tool or "").strip()
        doc_type = DOC_TYPE_BY_TOOL.get(tool)
        if not doc_type or tool in obligations:
            continue
        obligations[tool] = {
            "doc_type": doc_type,
            "fields": list(_SOURCE_COVERAGE_FIELDS.get(tool, ())),
        }
    return obligations


def _clarification_questions(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    raw_values: List[Any] = [state.get("clarification_question")]
    decision = state.get("safety_decision") or {}
    if isinstance(decision, dict):
        raw_values.extend(decision.get("questions", []) or [])
    for raw in raw_values:
        text = _normalize_generated_markdown(str(raw or ""))
        text = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in values:
            values.append(text)
    return values


_NEXT_STEP_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"下一步(?:需要)?确认(?:的问题)?\s*(?:\*\*)?\s*[:：]?\s*$"
)


_LEGACY_UNCITED_ASSERTION_PREFIX = "待复核（未绑定已批准直接证据）："


_UNIQUE_DIAGNOSIS_BOUNDARY = (
    "证据边界：当前证据不足以支持唯一故障结论，以下仅为候选和可复核线索。"
)


_NUMERIC_BOUNDARY = "数值边界：确定性计算未通过完整性门禁，不提供正式概率或可靠度数值。"


_TOPOLOGY_BOUNDARY = "拓扑边界：相关工程图结构尚未通过当前有效修订验证，只能作为参考线索。"


_NO_SUPPORTED_FACTS_MESSAGE = "当前已批准证据不足以支持可直接陈述的事实结论。"


_APPLICABILITY_NOTICE = "实际处置以适用构型、有效版本和正式维修程序为准。"


_MULTI_CANDIDATE_FRAMING = "以下是待排查的候选方向，不表示这些故障同时发生。"


def _authoritative_source_warning(decision: Dict[str, Any]) -> str:
    blocked_claims = {str(value) for value in decision.get("blocked_claims", [])}
    if "authoritative_source_unavailable" not in blocked_claims:
        return ""
    unavailable_tools = [
        str(value).strip()
        for value in decision.get("unavailable_authoritative_tools", [])
        if str(value).strip()
    ]
    tool_text = f"（{', '.join(unavailable_tools)}）" if unavailable_tools else ""
    return (
        f"本次所需的 PostgreSQL 权威事实查询失败{tool_text}，"
        "相关内容只能作为参考线索，不能作为正式结论。"
    )




def _is_applicability_notice_line(value: Any) -> bool:
    """Recognize a standalone semantic equivalent of the safety footer."""

    text = re.sub(r"\[\^\d+\]", "", str(value or "")).strip()
    if not text or len(text) > 180:
        return False
    return all(
        (
            re.search(r"实际|现场|诊断|维修|处置", text),
            re.search(r"适用.{0,6}(?:构型|机型|配置)", text),
            re.search(r"有效.{0,6}(?:版本|文档)|版本.{0,6}有效", text),
            re.search(r"正式.{0,8}(?:维修|维护|放行)?程序|正式维修手册", text),
            re.search(r"为准|复核|遵循|结合|依据", text),
        )
    )


def _topology_dependency(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated implicit-topology contract, if this turn has one.

    The compact target classifier owns this object.  The answer layer must not
    recreate it from query keywords because doing so would make evidence
    authorization depend on another ad-hoc intent classifier.
    """

    # The diagnosis node may enrich the planner contract with exact business
    # IDs re-read from PostgreSQL. Prefer that execution-side contract over
    # the original semantic frame; the latter deliberately cannot invent IDs.
    value = state.get("topology_dependency")
    if isinstance(value, Mapping) and value.get("required") is True:
        return value
    for container_name in ("query_frame", "dialogue_context"):
        container = state.get(container_name)
        if not isinstance(container, Mapping):
            continue
        value = container.get("topology_dependency")
        if isinstance(value, Mapping) and value.get("required") is True:
            return value
    return {}


def _closed_collection_obligation_active(
    answer: str,
    state: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
) -> bool:
    """Return only the planner-validated N-of-N coverage decision.

    Answer prose and retrieved members cannot create a new exhaustiveness
    obligation.  They are outputs/evidence, not authority for widening the
    current turn's contract.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    del answer, manifest
    return str(frame.get("coverage_scope") or "") == "complete_collection"


_CLARIFICATION_SLOT_LABELS = {
    "identity_slot": "对象身份",
    "calculation_parameter": "计算参数",
    "applicability_scope": "适用范围",
}

_CLARIFICATION_REASON_LABELS = {
    "insufficient_evidence": "证据不足",
    "ambiguous_identity": "对象身份不唯一",
    "unresolved_identity": "对象身份未解析",
    "unresolved_relation": "跨资料关联未确认",
    "scope_conflict": "范围条件冲突",
    "authoritative_source_unavailable": "权威资料暂不可用",
    "authoritative_record_not_found": "权威记录未命中",
    "unvalidated_graph": "工程图未通过验证",
    "incomplete_calculation": "计算参数不完整",
}


def _describe_clarification_identity(identity: Mapping[str, Any]) -> str:
    parts: List[str] = []
    doc_type = str(identity.get("doc_type") or "").strip()
    if doc_type:
        parts.append(doc_type)
    for record_id in (identity.get("record_ids") or [])[:6]:
        value = str(record_id or "").strip()
        if value:
            parts.append(value)
    fields = identity.get("fields") if isinstance(identity.get("fields"), Mapping) else {}
    for key, value in list(fields.items())[:6]:
        text = str(value or "").strip()
        if text:
            parts.append(f"{key}={text}")
    return "、".join(parts)


def render_clarification_only_answer(state: Mapping[str, Any]) -> str:
    """Deterministically render a clarification-list answer from slot contracts.

    最后防线专用：当修订循环失败且 salvage 零存活、而本轮确实存在已校验的
    澄清缺口合同时，发布由缺口确定性渲染的澄清清单，而不是空答案错误。
    内容只来自 missing_input_requirements / unresolved_identity_slots /
    clarification_reasons 的闭合字段，不经过模型，不含工程结论，因此不需要
    引用；没有可用缺口时返回空串，调用方继续走原拒发路径（不得发明澄清）。
    """

    from backend.agent.clarification_contract import (
        normalize_clarification_reasons,
        normalize_missing_input_requirements,
    )

    decision = (
        state.get("safety_decision")
        if isinstance(state.get("safety_decision"), Mapping)
        else {}
    )

    def value(name: str) -> Any:
        return decision.get(name) if name in decision else state.get(name)

    missing = normalize_missing_input_requirements(value("missing_input_requirements"))
    unresolved = normalize_missing_input_requirements(
        value("unresolved_identity_slots")
    )
    reasons = normalize_clarification_reasons(value("clarification_reasons"))
    if not (missing or unresolved or reasons):
        return ""

    lines: List[str] = [
        "本轮未能发布经过核验的完整回答。为继续查询，还需要补充以下信息："
    ]
    seen: set = set()
    for item in missing + unresolved:
        slot = str(item.get("slot") or "").strip()
        kind = str(item.get("kind") or "").strip()
        identity = (
            item.get("identity")
            if isinstance(item.get("identity"), Mapping)
            else {}
        )
        label = _CLARIFICATION_SLOT_LABELS.get(kind, kind or "信息")
        described = _describe_clarification_identity(identity)
        basis = str(item.get("basis") or "").strip()
        parts = [f"{label}：{slot}" if slot else label]
        if described:
            parts.append(f"（涉及：{described}）")
        if basis:
            parts.append(f"——{basis}")
        line = "".join(parts)
        if line not in seen:
            seen.add(line)
            lines.append(f"- {line}")
    for item in reasons:
        kind = str(item.get("kind") or "").strip()
        identity = (
            item.get("identity")
            if isinstance(item.get("identity"), Mapping)
            else {}
        )
        label = _CLARIFICATION_REASON_LABELS.get(kind, kind or "条件")
        blocked = [
            str(value or "").strip()
            for value in (item.get("blocked_claims") or [])[:4]
            if str(value or "").strip()
        ]
        described = _describe_clarification_identity(identity)
        parts = [f"{label}"]
        if described:
            parts.append(f"（涉及：{described}）")
        if blocked:
            parts.append("；在其确认前不能给出：" + "、".join(blocked))
        line = "".join(parts)
        if line not in seen:
            seen.add(line)
            lines.append(f"- {line}")
    lines.append("补充以上信息后，可以基于权威资料重新检索并给出有来源的回答。")
    return "\n".join(lines)
