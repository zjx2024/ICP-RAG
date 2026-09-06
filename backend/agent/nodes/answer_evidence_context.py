"""Current approved evidence context and internal citation ledger.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import required_answer_fields
from backend.agent.contracts import deduplicate_evidence
from backend.agent.nodes.answer_domain_scope import _filter_answer_domain_evidence
from backend.agent.nodes.answer_evidence_window import _select_evidence_window
from backend.agent.nodes.answer_history_rules import _history_is_primary_target
from backend.agent.nodes.answer_history_rules import _history_subject_matches_state
from backend.agent.nodes.answer_history_rules import _supplemental_history_evidence_ids
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer_request import _diagnostic_closure_required
from backend.agent.nodes.answer_structure_views import _rbd_structure_answer_fields_for_state
from backend.agent.nodes.answer_source_fields import source_owned_field_view
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping


def _context_evidence(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [source_owned_field_view(item, state) for item in _selected_context_evidence(state)]


def _selected_context_evidence(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the exact de-duplicated evidence list exposed by the gate."""
    evidence = deduplicate_evidence(state.get("evidence_registry", []) or [])
    decision = state.get("safety_decision") or {}
    has_gate_selection = "context_evidence_ids" in state or (
        isinstance(decision, dict) and "context_evidence_ids" in decision
    )
    if not has_gate_selection:
        return _select_evidence_window(
            state,
            _filter_answer_domain_evidence(state, evidence),
        )
    selected = state.get("context_evidence_ids")
    if selected is None and isinstance(decision, dict):
        selected = decision.get("context_evidence_ids")
    selected_ids = {str(value) for value in (selected or []) if str(value)}
    approved_ids = approved_id_set(decision.get("approved_evidence_ids") or [])
    gate_evidence = [
        item
        for item in evidence
        if str(item.get("evidence_id") or "") in selected_ids
    ]
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        # ``context_evidence_ids`` is normally identical to the gate-approved
        # direct rows, but older/running states and a focused gate may expose a
        # smaller Top-K window.  A closed collection is indivisible: restore
        # every *approved* member from the full registry before any answer-side
        # semantic narrowing.  This cannot promote a blocked row and closure
        # is still rechecked from parent/kind/count by the window selector.
        requested_kind = str(frame.get("collection_member_kind") or "").strip()
        stored_kind = "failure_mode" if requested_kind == "cause" else requested_kind
        resolution = (
            state.get("collection_resolution")
            if isinstance(state.get("collection_resolution"), Mapping)
            else {}
        )
        resolved_parent_id = str(resolution.get("parent_id") or "").strip()
        present_ids = {
            str(item.get("evidence_id") or "") for item in gate_evidence
        }
        for item in evidence:
            evidence_id = str(item.get("evidence_id") or "")
            if not evidence_id or evidence_id in present_ids or evidence_id not in approved_ids:
                continue
            fields = _evidence_answer_fields(item)
            if not (
                fields.get("collection_closed") is True
                and str(fields.get("collection_role") or "") == "member"
                and (
                    not stored_kind
                    or str(fields.get("collection_member_kind") or "") == stored_kind
                )
                and (
                    not resolved_parent_id
                    or str(fields.get("collection_parent_id") or "") == resolved_parent_id
                )
            ):
                continue
            gate_evidence.append(item)
            present_ids.add(evidence_id)
    gate_evidence = _filter_answer_domain_evidence(state, gate_evidence)
    # For a fault/cause/impact answer, a broad PostgreSQL history keyword row
    # is not enough to call an old case "similar".  Supplemental history must
    # have been found semantically and then re-hydrated by exact storage ID.
    # Explicit history questions retain structured rows because history itself
    # is the user's primary information target.  The excluded rows remain in
    # the UI audit catalog; only the answer context is narrowed here.
    answer_fields = set(required_answer_fields(state))
    history_fields = {
        field for field in answer_fields if field.startswith("history_")
    }
    non_history_fields = answer_fields - history_fields
    # A direct business-record lookup keeps its selected history row as the
    # requested primary fact, not as a weak supplemental precedent.
    direct_record_identity = "record_identity" in answer_fields
    mixed_current_incident_history = bool(
        _history_is_primary_target(state)
        and _diagnostic_closure_required(state)
        and history_fields
        and non_history_fields
    )
    if (
        not direct_record_identity
        and (
            not _history_is_primary_target(state)
            or mixed_current_incident_history
        )
    ):
        if mixed_current_incident_history:
            # A structured diagnostic+history contract may show a bounded set
            # of query-relative cases, but it does not turn every hydrated row
            # into a co-equal parent.  Exact stored references dominate; in
            # their absence use the ordinary absolute+relative semantic limit.
            supplemental_history_ids = _supplemental_history_evidence_ids(
                gate_evidence
            )
            supplemental_history_ids = {
                evidence_id
                for evidence_id in supplemental_history_ids
                if any(
                    str(item.get("evidence_id") or "") == evidence_id
                    and _history_subject_matches_state(
                        item,
                        state,
                        gate_evidence,
                        apply_relative_focus=False,
                    )
                    for item in gate_evidence
                    if str(item.get("doc_type") or "") == "fault_history"
                )
            }
        else:
            supplemental_history_ids = _supplemental_history_evidence_ids(gate_evidence)
            supplemental_history_ids = {
                evidence_id
                for evidence_id in supplemental_history_ids
                if any(
                    str(item.get("evidence_id") or "") == evidence_id
                    and _history_subject_matches_state(item, state, gate_evidence)
                    for item in gate_evidence
                    if str(item.get("doc_type") or "") == "fault_history"
                )
            }
        gate_evidence = [
            item
            for item in gate_evidence
            if str(item.get("doc_type") or "") != "fault_history"
            or str(item.get("evidence_id") or "") in supplemental_history_ids
        ]
    approved = [
        item
        for item in gate_evidence
        if str(item.get("evidence_id") or "") in approved_ids
    ]
    clues = [
        item
        for item in gate_evidence
        if str(item.get("evidence_id") or "") not in approved_ids
    ]
    # The full registry remains available to the UI audit catalog. The answer
    # model sees all approved facts plus only a few high-priority clues, so a
    # broad graph expansion cannot drown two useful PostgreSQL records in
    # twenty blocked paths.
    clue_budget = 4 if approved else 8
    return _select_evidence_window(state, [*approved, *clues[:clue_budget]])


def _citation_ledger(state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    approved_ids = approved_id_set((state.get("safety_decision") or {}).get("approved_evidence_ids") or [])
    return {
        index: _citation_source_item(item, state)
        for index, item in enumerate(_context_evidence(state), start=1)
        if str(item.get("evidence_id") or "") in approved_ids
    }


def _citation_source_item(item: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Use the identical state-authorized structure for generation and proof.

    The raw RBD graph is kept in state for deterministic revalidation, but it
    cannot bypass the requested object's projection through a summary field.
    Nor should the citation reviewer lose a formal expression merely because
    the public summary happened to omit it.
    """
    if item.get("doc_type") != "rbd" or item.get("claim_type") != "rbd_structure":
        return item
    fields = _rbd_structure_answer_fields_for_state(item, state)
    summary = str(fields.get("structure_description") or fields.get("structure_expression")
                  or fields.get("success_criterion") or "")
    return {**item, "_citation_fields_only": True,
            "display_summary": summary or "当前对象未获得可引用的正式结构字段。", "metadata": {
        **(item.get("metadata") or {}), "answer_fields": dict(fields),
    }}
