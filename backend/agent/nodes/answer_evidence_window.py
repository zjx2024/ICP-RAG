"""Bounded evidence-window selection and allocation policies.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import required_answer_fields
from backend.agent.answer_contract import tools_for_answer_fields
from backend.agent.contracts import match_type_is_exact
from backend.agent.history_focus import history_case_identity
from backend.agent.nodes.answer_collections import _architecture_role_indices
from backend.agent.nodes.answer_collections import _requested_closed_collection_group
from backend.agent.nodes.answer_fault_rules import _FAULT_FIT_RANK
from backend.agent.nodes.answer_fault_rules import _fault_candidate_fit
from backend.agent.nodes.answer_history_rules import _history_collection_requested
from backend.agent.nodes.answer_history_rules import _history_is_primary_target
from backend.agent.nodes.answer_history_rules import _history_subject_matches_state
from backend.agent.nodes.answer_payload import _answer_relevance_score
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer_presence import normalize_presence_text
from backend.agent.tool_policy import DOC_TYPE_BY_TOOL
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple


# The gate may retain a large audit set, but the answer/reviewer should see a
# small, source-balanced working set.  A bounded window keeps multi-turn
# context within the model budget while the complete registry remains
# available for audit and later targeted rereads.
_EVIDENCE_WINDOW_LIMIT = 12
_EVIDENCE_WINDOW_HARD_CAP = 25


_EVIDENCE_TYPE_ROUND_LIMIT = 8


_TARGET_TYPE_RESERVE = 3


_DIRECT_EVIDENCE_AUTHORITIES = {
    "authoritative_record",
    "source_artifact",
    "confirmed_relation",
    "validated_relation",
    "deterministic_derivation",
}


_DEFAULT_EVIDENCE_TYPE_ORDER = (
    "function_locator",
    "hmc",
    "fmeca",
    "fta",
    "rbd",
    "mfl",
    "fault_history",
)


def _evidence_target_types(state: Dict[str, Any]) -> List[str]:
    """Return evidence families authorized by structured answer contracts."""

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    result: List[str] = []
    tools = tools_for_answer_fields(required_answer_fields(state))
    for raw in frame.get("source_obligations") or []:
        if not isinstance(raw, Mapping):
            continue
        tool = str(raw.get("tool") or "").strip()
        fields = required_answer_fields(
            {"query_frame": {"required_answer_fields": raw.get("fields")}}
        )
        if tool in DOC_TYPE_BY_TOOL and fields and tool not in tools:
            tools.append(tool)
    calculation_requests = {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
        if str(value or "").strip()
    }
    if calculation_requests & {"fta_cut_sets", "fta_probability"}:
        tools.extend(tool for tool in ("fta_query", "fta_analyzer") if tool not in tools)
    if "rbd_reliability" in calculation_requests:
        tools.extend(tool for tool in ("rbd_query", "rbd_calculator") if tool not in tools)
    for tool in tools:
        doc_type = DOC_TYPE_BY_TOOL.get(tool)
        if doc_type and doc_type not in result:
            result.append(doc_type)
    return result


def _evidence_priority(
    item: Dict[str, Any],
    *,
    index: int,
    approved_ids: set[str],
    calculation_requested: bool = False,
) -> Tuple[int, int, int, int, float, int]:
    """Sort formal, gate-approved facts before projections within one family."""

    evidence_id = str(item.get("evidence_id") or "")
    authority = str(item.get("authority") or "")
    level = str(item.get("evidence_level") or "")
    raw_score = item.get("answer_rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("retrieval_score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
    authority_rank = 0 if authority in _DIRECT_EVIDENCE_AUTHORITIES else (
        1 if authority == "unconfirmed_extraction" else 2
    )
    level_rank = {
        "L1_DIRECT": 0,
        "L2_CORROBORATED": 1,
        "L3_INFERRED": 2,
        "L4_UNRESOLVED": 3,
    }.get(level, 4)
    return (
        0 if evidence_id in approved_ids else 1,
        0
        if calculation_requested and authority == "deterministic_derivation"
        else 1,
        authority_rank,
        level_rank,
        -score,
        index,
    )


def _select_evidence_window(
    state: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    *,
    limit: int = _EVIDENCE_WINDOW_LIMIT,
) -> List[Dict[str, Any]]:
    """Select a bounded, source-balanced evidence window for answer and UI.

    Small result sets retain their original order for compatibility.  Once the
    gate exposes more evidence than the model can safely consume, matching
    evidence families receive reserved positions and the rest are selected in
    a round-robin pass.  A final fill pass may exceed the soft per-family quota
    but can never displace the reserved evidence.
    """

    if limit <= 0:
        return []
    # ``limit`` is the normal (soft) model-context budget.  A collection that
    # PostgreSQL has explicitly marked closed is one indivisible answer
    # obligation: keeping only 12 of 15 members would make an exhaustive
    # question silently incomplete.  Let that one verified collection grow
    # the window, but never beyond the global hard safety cap.
    bounded_limit = min(int(limit), _EVIDENCE_WINDOW_LIMIT)
    query_frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    # The evidence contract, rather than a particular wording/classifier
    # result, decides whether a closed collection is present.  Architecture
    # inventories use the same contract even when their query frame predates
    # the generic ``complete_collection`` field.
    requested_group = _requested_closed_collection_group(evidence, state)
    if requested_group is not None:
        closed_collection_groups = [list(requested_group["indexes"])]
    else:
        # Closed metadata in retrieved rows cannot create a collection answer
        # contract.  Without a validated matching coverage scope, retain such
        # rows only through the ordinary bounded evidence policy.
        closed_collection_groups = []
    effective_limit = min(
        _EVIDENCE_WINDOW_HARD_CAP,
        max(
            bounded_limit,
            max((len(indexes) for indexes in closed_collection_groups), default=0),
        ),
    )
    has_answer_rerank = any(
        isinstance(item.get("answer_rerank_score"), (int, float))
        for item in evidence
    )
    has_fault_fit_constraints = any(
        str(query_frame.get(key) or "").strip() not in {"", "未知"}
        for key in ("operating_stage", "temporal_pattern")
    )
    if (
        len(evidence) <= effective_limit
        and not (state.get("calculation_requests") or [])
        and not has_answer_rerank
        and not has_fault_fit_constraints
    ):
        # A direct “有没有相近案例” question is not an inventory request.
        # Keep at most two history rows even when the total evidence set fits
        # the soft window; otherwise every broad SQL hit is handed to the
        # answer model and the response becomes an unrelated case dump.
        if _history_is_primary_target(state) and not _history_collection_requested(state):
            history_rows = [
                item
                for item in evidence
                if str(item.get("doc_type") or "") == "fault_history"
            ]
            scoped_history_rows = [
                item
                for item in history_rows
                if _history_subject_matches_state(item, state, evidence)
            ]
            if scoped_history_rows:
                history_rows = scoped_history_rows
                scoped_history_ids = {
                    str(item.get("evidence_id") or "")
                    for item in scoped_history_rows
                    if str(item.get("evidence_id") or "")
                }
                evidence = [
                    item
                    for item in evidence
                    if str(item.get("doc_type") or "") != "fault_history"
                    or str(item.get("evidence_id") or "") in scoped_history_ids
                ]
            # Different retrieval paths can expose the same PostgreSQL case
            # under separate evidence IDs.  One parent consumes one answer
            # slot; keep its strongest representation while retaining all
            # duplicates in the audit registry.
            strongest_by_case: Dict[str, Dict[str, Any]] = {}
            for item in history_rows:
                case_id = history_case_identity(item) or str(
                    item.get("evidence_id") or ""
                )
                previous = strongest_by_case.get(case_id)
                if previous is None or (
                    match_type_is_exact(item.get("match_type")),
                    _answer_relevance_score(item),
                ) > (
                    match_type_is_exact(previous.get("match_type")),
                    _answer_relevance_score(previous),
                ):
                    strongest_by_case[case_id] = item
            history_rows = list(strongest_by_case.values())
            strongest_history_ids = {
                str(item.get("evidence_id") or "")
                for item in history_rows
                if str(item.get("evidence_id") or "")
            }
            evidence = [
                item
                for item in evidence
                if str(item.get("doc_type") or "") != "fault_history"
                or str(item.get("evidence_id") or "") in strongest_history_ids
            ]
            if len(history_rows) > 2:
                history_ids = {
                    str(item.get("evidence_id") or "")
                    for item in history_rows[:2]
                }
                return [
                    item
                    for item in evidence
                    if str(item.get("doc_type") or "") != "fault_history"
                    or str(item.get("evidence_id") or "") in history_ids
                ]
        return list(evidence)

    decision = state.get("safety_decision") or {}
    approved_ids = approved_id_set(decision.get("approved_evidence_ids") or [])
    calculation_requested = bool(state.get("calculation_requests") or [])
    history_cap = (
        2
        if _history_is_primary_target(state) and not _history_collection_requested(state)
        else None
    )
    indexed = list(enumerate(evidence))

    # A direct history request can still produce several authoritative rows
    # sharing a generic condition such as “high load”.  If at least one case
    # is tied to the current subject (including an explicitly selected prior
    # citation), keep other history rows in the audit ledger but exclude them
    # from this bounded answer window.  Without this filter the generic
    # target-family reservation below can reinsert an already rejected case.
    if _history_is_primary_target(state) and not _history_collection_requested(state):
        scoped_history_ids = {
            str(item.get("evidence_id") or "")
            for item in evidence
            if str(item.get("doc_type") or "") == "fault_history"
            and _history_subject_matches_state(item, state, evidence)
            and str(item.get("evidence_id") or "")
        }
        if scoped_history_ids:
            indexed = [
                pair
                for pair in indexed
                if str(pair[1].get("doc_type") or "") != "fault_history"
                or str(pair[1].get("evidence_id") or "") in scoped_history_ids
            ]

    architecture_role_ids: set[str] = set()
    architecture_role_ordered_ids: List[str] = []
    exact_anchor_ordered_ids: List[str] = []
    history_role_ids: set[str] = set()
    history_role_ordered_ids: List[str] = []
    query_frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    anchor_keys = {
        (
            str(anchor.get("doc_type") or "").strip(),
            normalize_presence_text(anchor.get("record_id")),
        )
        for anchor in (state.get("selected_conversation_anchors") or [])
        if isinstance(anchor, Mapping)
        and normalize_presence_text(anchor.get("record_id"))
    }
    record_binding = (
        state.get("structured_record_binding")
        if isinstance(state.get("structured_record_binding"), Mapping)
        else {}
    )
    if (
        record_binding.get("bound") is True
        and record_binding.get("ambiguous") is not True
        and record_binding.get("relevance_only") is not True
    ):
        binding_doc_type = str(record_binding.get("doc_type") or "").strip()
        anchor_keys.update(
            (binding_doc_type, normalize_presence_text(record_id))
            for record_id in (record_binding.get("record_ids") or [])
            if normalize_presence_text(record_id)
        )
    for _index, item in indexed:
        item_doc_type = str(item.get("doc_type") or "").strip()
        item_record_id = normalize_presence_text(
            item.get("record_id") or item.get("title")
        )
        if not item_record_id:
            continue
        if any(
            record_id == item_record_id
            and (not doc_type or doc_type == item_doc_type)
            for doc_type, record_id in anchor_keys
        ):
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in exact_anchor_ordered_ids:
                exact_anchor_ordered_ids.append(evidence_id)
    if str(query_frame.get("coverage_scope") or "") == "complete_collection":
        # A PostgreSQL closed inventory is a single answer obligation. Reserve
        # all of its siblings before score-based balancing, otherwise a dense
        # set of leaf functions can make a whole-system answer look complete
        # while silently omitting low-ranked direct subsystems.
        for indexes in closed_collection_groups:
            for index in indexes[:effective_limit]:
                if not 0 <= index < len(evidence):
                    continue
                evidence_id = str(evidence[index].get("evidence_id") or "")
                if evidence_id:
                    architecture_role_ids.add(evidence_id)
                    if evidence_id not in architecture_role_ordered_ids:
                        architecture_role_ordered_ids.append(evidence_id)
    if query_frame.get("architecture_roles"):
        for indexes in _architecture_role_indices(state, evidence).values():
            for index in indexes:
                if 0 <= index < len(evidence):
                    evidence_id = str(evidence[index].get("evidence_id") or "")
                    if evidence_id:
                        architecture_role_ids.add(evidence_id)
                        if evidence_id not in architecture_role_ordered_ids:
                            architecture_role_ordered_ids.append(evidence_id)

    # An explicit history request is a direct answer obligation, even when the
    # same turn also asks for maintenance advice.  Without a reserved slot, a
    # high-scoring FMECA/MFL pair can consume the bounded window and the
    # PostgreSQL history row remains visible only in the audit catalog.  Keep
    # at most two distinct authoritative cases; broad history lists are still
    # bounded by the normal evidence window and never let search projections
    # displace a hydrated record.
    if _history_is_primary_target(state):
        history_candidates = [
            pair
            for pair in indexed
            if str(pair[1].get("doc_type") or "") == "fault_history"
            and _history_subject_matches_state(pair[1], state, evidence)
            and str(pair[1].get("evidence_id") or "") in approved_ids
            and str(pair[1].get("record_id") or pair[1].get("title") or "").strip()
            and str(pair[1].get("match_type") or "")
            in {"exact_history_reference", "structured_query", "semantic_id_hydration"}
        ]
        history_candidates.sort(
            key=lambda pair: (
                {
                    "exact_history_reference": 0,
                    "structured_query": 1,
                    "semantic_id_hydration": 2,
                }.get(str(pair[1].get("match_type") or ""), 3),
                -_answer_relevance_score(pair[1])[0],
                pair[0],
            )
        )
        for _index, item in history_candidates[:2]:
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id:
                history_role_ids.add(evidence_id)
                if evidence_id not in history_role_ordered_ids:
                    history_role_ordered_ids.append(evidence_id)

    def evidence_window_priority(
        pair: Tuple[int, Dict[str, Any]],
    ) -> Tuple[int, int, int, int, int, int, float, int]:
        base = _evidence_priority(
            pair[1],
            index=pair[0],
            approved_ids=approved_ids,
            calculation_requested=calculation_requested,
        )
        evidence_id = str(pair[1].get("evidence_id") or "")
        # Gate approval remains the first key.  Within the same approval class,
        # reserve the explicit roles requested by a multi-part architecture
        # question before generic hierarchy rows with a slightly higher score.
        fit_rank = _FAULT_FIT_RANK.get(
            _fault_candidate_fit(state, pair[1]),
            _FAULT_FIT_RANK["unknown"],
        )
        return (
            base[0],
            0 if evidence_id in architecture_role_ids else 1,
            fit_rank,
            *base[1:],
        )

    indexed.sort(
        key=evidence_window_priority
    )

    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for pair in indexed:
        doc_type = str(pair[1].get("doc_type") or "unknown")
        groups.setdefault(doc_type, []).append(pair)

    target_types = _evidence_target_types(state)
    group_order = [doc_type for doc_type in target_types if doc_type in groups]
    group_order.extend(
        doc_type
        for doc_type in _DEFAULT_EVIDENCE_TYPE_ORDER
        if doc_type in groups and doc_type not in group_order
    )
    group_order.extend(doc_type for doc_type in groups if doc_type not in group_order)

    selected: List[Tuple[int, Dict[str, Any]]] = []
    selected_ids: set[str] = set()
    type_counts: Dict[str, int] = {}
    record_counts: Dict[Tuple[str, str], int] = {}

    def add(pair: Tuple[int, Dict[str, Any]], *, enforce_record_limit: bool = True) -> bool:
        item = pair[1]
        evidence_id = str(item.get("evidence_id") or "")
        if (
            not evidence_id
            or evidence_id in selected_ids
            or len(selected) >= effective_limit
        ):
            return False
        doc_type = str(item.get("doc_type") or "unknown")
        if (
            history_cap is not None
            and doc_type == "fault_history"
            and type_counts.get(doc_type, 0) >= history_cap
        ):
            return False
        record_id = str(item.get("record_id") or item.get("chunk_id") or evidence_id)
        record_key = (doc_type, record_id)
        if enforce_record_limit and record_counts.get(record_key, 0) >= 2:
            return False
        selected.append(pair)
        selected_ids.add(evidence_id)
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        record_counts[record_key] = record_counts.get(record_key, 0) + 1
        return True

    # Multi-role architecture questions have a finite set of explicit slots.
    # Reserve each approved slot before generic per-document balancing so two
    # low-score parallel receivers cannot be displaced by many broad component
    # rows from the same Function Locator document.
    pair_by_evidence_id = {
        str(pair[1].get("evidence_id") or ""): pair for pair in indexed
    }
    for evidence_id in exact_anchor_ordered_ids:
        if evidence_id not in approved_ids:
            continue
        pair = pair_by_evidence_id.get(evidence_id)
        if pair is not None:
            add(pair)
    for evidence_id in architecture_role_ordered_ids:
        if evidence_id not in approved_ids:
            continue
        pair = pair_by_evidence_id.get(evidence_id)
        if pair is not None:
            add(pair)

    for evidence_id in history_role_ordered_ids:
        pair = pair_by_evidence_id.get(evidence_id)
        if pair is not None:
            add(pair)

    # The model-recognized goal gets guaranteed coverage first.  Approved
    # PostgreSQL/validated evidence wins within each family through the sort.
    for doc_type in target_types:
        reserved = 0
        for pair in groups.get(doc_type, []):
            if add(pair):
                reserved += 1
            if reserved >= _TARGET_TYPE_RESERVE or len(selected) >= effective_limit:
                break

    # Round-robin prevents HMC rows (often fault + maintenance pairs) or a wide
    # FMECA table from monopolizing the remaining window.
    made_progress = True
    while len(selected) < effective_limit and made_progress:
        made_progress = False
        for doc_type in group_order:
            if type_counts.get(doc_type, 0) >= _EVIDENCE_TYPE_ROUND_LIMIT:
                continue
            for pair in groups[doc_type]:
                if add(pair):
                    made_progress = True
                    break
            if len(selected) >= effective_limit:
                break

    # If the available families cannot fill the window under the soft quota,
    # retain more relevant evidence without undoing the reserved coverage.
    for pair in indexed:
        if len(selected) >= effective_limit:
            break
        add(pair)
    for pair in indexed:
        if len(selected) >= effective_limit:
            break
        add(pair, enforce_record_limit=False)

    return [item for _, item in selected]
