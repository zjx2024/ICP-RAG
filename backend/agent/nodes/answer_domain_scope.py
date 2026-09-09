"""Approved evidence scope across requested source domains.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import required_answer_fields
from backend.agent.nodes.answer_fault_rules import _fault_candidate_fit
from backend.agent.nodes.answer_payload import _binding_identity_codes
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_payload import _evidence_object_business_ids
from backend.agent.nodes.answer_payload import _evidence_object_identity_codes
from backend.agent.nodes.answer_payload import _normalize_business_id
from backend.agent.nodes.answer_payload import _selected_anchor_record_ids
from backend.agent.nodes.answer_presence import normalize_presence_text
from backend.agent.nodes.answer_record_scope import _narrow_semantic_table_families
from backend.agent.nodes.answer_request import _is_fta_structure_only_request
from backend.agent.nodes.answer_request import _source_obligation_map
from backend.agent.nodes.answer_request import _topology_dependency
from backend.agent.nodes.answer_structure_views import _fta_structure_graph
from backend.agent.nodes.answer_structure_views import _rbd_matches_topology_dependency
from backend.agent.nodes.answer_text import _compact_semantic_text
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
import ast


def _filter_answer_domain_evidence(
    state: Dict[str, Any], evidence: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Apply authority-bearing source, identity and applicability boundaries.

    Records removed here remain in the audit catalog.  Raw question words are
    deliberately not a deletion authority: ``数据``/``显示``/``复位`` and
    direction wording are open semantics already represented by the planner's
    structured bindings.  Only those bindings, exact IDs, explicit source
    scope and deterministic applicability conflicts may narrow the answer
    window.
    """
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    complete_collection = str(frame.get("coverage_scope") or "") == "complete_collection"
    topology_dependency = _topology_dependency(state)
    canonical_business_ids = {
        normalized
        for value in (topology_dependency.get("canonical_business_ids") or [])
        if (normalized := _normalize_business_id(value))
    }
    structured_binding = (
        state.get("structured_entity_binding")
        if isinstance(state.get("structured_entity_binding"), Mapping)
        else {}
    )
    focused_business_ids = {
        normalized
        for value in (structured_binding.get("business_ids") or [])
        if structured_binding.get("bound") is True
        and (normalized := _normalize_business_id(value))
    }
    focused_identity_codes = _binding_identity_codes(structured_binding)
    scoped_business_ids = canonical_business_ids | focused_business_ids
    requested_source_tools = {
        str(value or "").strip()
        for value in (
            state.get("requested_source_tools")
            or frame.get("requested_source_tools")
            or []
        )
        if str(value or "").strip()
    }
    response_focus = {
        str(value or "").strip()
        for value in (frame.get("response_focus") or [])
        if str(value or "").strip()
    }
    source_obligations = _source_obligation_map(state)
    source_boundary_doc_types = {
        str(obligation.get("doc_type") or "")
        for obligation in source_obligations.values()
        if str(obligation.get("doc_type") or "")
    }
    source_boundary_fields = {
        str(field or "").strip()
        for obligation in source_obligations.values()
        for field in (obligation.get("fields") or [])
        if str(field or "").strip()
    }
    non_source_focus = response_focus - {"source_capability", "source_limitation"}
    requested_fields = set(required_answer_fields(state))
    source_boundary_locked = bool(
        source_boundary_doc_types
        and response_focus & {"source_capability", "source_limitation"}
        and not (requested_fields - source_boundary_fields)
        and not non_source_focus
    )
    source_owned_goals = bool(source_boundary_fields and requested_fields
                             and not (requested_fields - source_boundary_fields))
    source_boundary_locked = source_boundary_locked or source_owned_goals
    # A reread anchor may identify the object for a named-source follow-up;
    # a neighbouring document/record is not that anchor. Independent goals
    # outside the named source keep their own families. Explicit exclusivity
    # never opens an anchor exception.
    allow_source_anchor = source_owned_goals and not any(
        isinstance(request, Mapping) and (request.get("scope") == "exclusive"
                                         or request.get("question_role", "evidence") != "evidence")
        for request in frame.get("source_obligations") or []
    )
    source_anchors = [anchor for anchor in state.get("selected_conversation_anchors") or []
                      if isinstance(anchor, Mapping) and anchor.get("doc_id") and anchor.get("record_id")]
    # 边界/元问题（“RBD 能证明什么”“哪些结论不能说”）由 plan 判定后把
    # evidence_domain_locked 与上一轮已批准 doc_type 放进 query_frame。
    # 这里把答案窗口锁到这些来源族：跨域的强语义邻居（例如 function_locator
    # 的网络接入记录）保留审计目录，但不能进入答案上下文，避免元问题被
    # 其他文档族的记录带偏。用户本轮显式点名新来源（requested_source_tools
    # 非空）时以用户为准，不锁定。
    locked_doc_types = {
        str(value or "").strip()
        for value in (frame.get("locked_doc_types") or [])
        if str(value or "").strip()
    }
    followup_domain_locked = (
        frame.get("evidence_domain_locked") is True
        and bool(locked_doc_types)
        and not requested_source_tools
    )
    required_fields = set(required_answer_fields(state))
    # 用户显式点名某类来源（“请查 RBD/FTA”）时，证据门禁批准的该类记录就是
    # 本轮的主要回答对象：主题收敛和拓扑依赖筛选是为“隐式来源选择”服务的，
    # 不能在显式请求下把已批准记录整类丢弃——那会把可引用 ledger 清空，
    # 使后续引用修复只能输出墙句。豁免仅限门禁已批准且被点名 doc_type 的
    # 记录；未批准的投影/候选仍按原规则过滤。
    decision_meta = state.get("safety_decision") or {}
    approved_evidence_ids = {
        str(value)
        for value in (
            decision_meta.get("approved_evidence_ids") or []
            if isinstance(decision_meta, dict)
            else []
        )
        if str(value)
    }
    # When the user explicitly asks to inspect an FTA and the only requested
    # answer dimension is its structural chain, FMECA/HMC side rows are not
    # answer evidence.  They are useful in the audit catalogue, but passing
    # them to the synthesis model invites an unrelated fault mode to be
    # appended beside the requested tree.  Keep this source boundary narrow:
    # a mixed question that also asks for causes, impacts or repair remains
    # free to use the corresponding structured sources.
    fta_structure_only = _is_fta_structure_only_request(state) and any(
        str(item.get("doc_type") or "") == "fta"
        and str(item.get("claim_type") or "") in {"fta_structure", "fta_calculation"}
        for item in evidence
    )
    # 已批准 FTA 树的叶子事件是其显式组成部分。当 FMECA 行的故障模式与某
    # 个叶子事件原文完全一致、或其业务 ID 显式出现在图顶点时，该行是树的
    # 证据而不只是语义邻居；这类行不应因产品/功能标志里没有当前轮绑定的对
    # 象 ID 而被对象边界检查误删。匹配只使用精确文本/ID，不做相似度合并，
    # 且前提是 FTA 树本身已被门禁批准。
    approved_fta_leaf_names: set[str] = set()
    approved_fta_leaf_ids: set[str] = set()
    allow_fta_leaf_identity = bool(
        {"structural_cause_chain", "fault_cause"} & required_fields
    )
    for item in (evidence if allow_fta_leaf_identity else ()):
        if (
            str(item.get("evidence_id") or "") not in approved_evidence_ids
            or str(item.get("doc_type") or "") != "fta"
        ):
            continue
        graph = _fta_structure_graph(item)
        for vertex in graph.get("vertices") or graph.get("nodes") or []:
            if not isinstance(vertex, Mapping):
                continue
            properties = (
                vertex.get("properties")
                if isinstance(vertex.get("properties"), Mapping)
                else {}
            )
            label = _compact_semantic_text(
                vertex.get("label") or properties.get("type")
            ).casefold()
            if label not in {"basicevent", "basic_event", "basic event", "基本事件"}:
                continue
            for value in (
                vertex.get("business_id"),
                vertex.get("module_id"),
                vertex.get("component_id"),
                vertex.get("function_id"),
                properties.get("business_id"),
                properties.get("failure_mode_id"),
            ):
                if normalized := _normalize_business_id(value):
                    approved_fta_leaf_ids.add(normalized)
            name = _compact_semantic_text(
                properties.get("name")
                or properties.get("event_name")
                or vertex.get("name")
                or vertex.get("title")
            )
            if name:
                approved_fta_leaf_names.add(name)
        # 答案侧窗口常常只带被有界的 basic_events 投影，而不是原始图 JSON。
        basic_projection = _evidence_answer_fields(item).get("basic_events")
        if isinstance(basic_projection, (list, tuple)):
            for entry in basic_projection:
                if entry is None:
                    continue
                if isinstance(entry, str):
                    try:
                        entry = ast.literal_eval(entry)
                    except (SyntaxError, ValueError):
                        entry = None
                if not isinstance(entry, Mapping):
                    continue
                name = _compact_semantic_text(
                    entry.get("name") or entry.get("event_name")
                )
                if name:
                    approved_fta_leaf_names.add(name)
    fta_linked_fmeca_ids: set[str] = set()
    if approved_fta_leaf_names or approved_fta_leaf_ids:
        for item in evidence:
            if str(item.get("doc_type") or "") != "fmeca":
                continue
            record_id = _normalize_business_id(item.get("record_id"))
            mode = _compact_semantic_text(
                _evidence_answer_fields(item).get("failure_mode")
            )
            if (record_id and record_id in approved_fta_leaf_ids) or (
                mode and mode in approved_fta_leaf_names
            ):
                fta_linked_fmeca_ids.add(str(item.get("evidence_id") or ""))
    filtered: List[Dict[str, Any]] = []
    for item in evidence:
        _dropped_by = ""
        exact_source_anchor = allow_source_anchor and str(item.get("evidence_id") or "") in approved_evidence_ids and any(
            all(item.get(key) == anchor.get(key) for key in ("doc_type", "doc_id", "record_id"))
            and (not anchor.get("revision_id") or anchor["revision_id"] == (item.get("metadata") or {}).get("revision_id"))
            for anchor in source_anchors
        )
        if (
            source_boundary_locked
            and str(item.get("doc_type") or "") not in source_boundary_doc_types
            and not exact_source_anchor
        ):
            _dropped_by = "source_boundary"
        elif (
            followup_domain_locked
            and str(item.get("doc_type") or "") not in locked_doc_types
        ):
            _dropped_by = "followup_domain"
        elif fta_structure_only and str(item.get("doc_type") or "") != "fta":
            _dropped_by = "fta_structure_only"
        elif (
            focused_business_ids
            and not complete_collection
            and str(frame.get("coverage_scope") or "") == "focused_detail"
            and str(item.get("doc_type") or "") == "function_locator"
            and not (
                _evidence_object_business_ids(item) & focused_business_ids
            )
        ):
            # The execution binder has already selected one exact Function
            # Locator chain.  Semantic/hybrid siblings remain in the audit
            # catalogue, but cannot become answer rows for that focused
            # object.  This is exact structured-ID containment, not fuzzy
            # alias resolution.
            _dropped_by = "focused_function_identity"
        elif (
            str(item.get("doc_type") or "") == "rbd"
            and not _rbd_matches_topology_dependency(item, state)
        ):
            _dropped_by = "rbd_topology"
        if _dropped_by:
            continue
        if not _dropped_by and (
            scoped_business_ids
            and str(item.get("doc_type") or "") == "fmeca"
        ):
            # PostgreSQL authority proves that a FMECA row exists, not that it
            # describes the named path.  Once Function Locator has bound this
            # turn to durable IDs, an impact row must carry one of those exact
            # object IDs in its structured product/function fields.  This
            # prevents a high semantic score for another A-channel record from
            # displacing the actual FUNC-* row while leaving the rejected item
            # visible in the audit catalog.
            object_ids = _evidence_object_business_ids(item)
            object_codes = _evidence_object_identity_codes(item)
            exact_prior_target = (
                normalize_presence_text(item.get("record_id"))
                in _selected_anchor_record_ids(state, "fmeca")
            )
            object_matches = bool(object_ids & scoped_business_ids) or bool(
                focused_identity_codes & object_codes
            )
            fta_leaf_matches = bool(
                not focused_business_ids
                and not focused_identity_codes
                and str(item.get("evidence_id") or "") in fta_linked_fmeca_ids
            )
            if not exact_prior_target and not object_matches and not fta_leaf_matches:
                _dropped_by = "fmeca_ids"
        if _dropped_by:
            continue
        # Exact FTA linkage establishes identity, not relevance to the current
        # operating phase or occurrence pattern.  Explicitly contradictory
        # FMECA rows stay in the audit catalog but cannot enter answer context.
        fields = _evidence_answer_fields(item)
        is_closed_member = (
            fields.get("collection_closed") is True
            and str(fields.get("collection_role") or "") == "member"
        )
        if (
            not (complete_collection and is_closed_member)
            and _fault_candidate_fit(state, item) == "contradictory"
        ):
            continue
        filtered.append(item)
    return _narrow_semantic_table_families(state, filtered)
