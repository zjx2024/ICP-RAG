"""Current approved evidence context and internal citation ledger.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.contracts import deduplicate_evidence
from backend.agent.nodes.answer_domain_scope import _filter_answer_domain_evidence
from backend.agent.nodes.answer_evidence_window import _select_evidence_window
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer_structure_views import _rbd_structure_answer_fields_for_state
from backend.agent.nodes.answer_source_fields import source_owned_field_view
from backend.agent.nodes.answer_request import _topology_dependency
from typing import Any
from typing import Dict
from typing import List


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
    # The gate approves provenance, not semantic exclusivity. Keep all
    # approved rows even when its presentation IDs contain fewer candidates.
    # The same identity coordinates are available to author and critic.
    gate_evidence = [
        item for item in evidence
        if str(item.get("evidence_id") or "") in (selected_ids | approved_ids)
        and str(item.get("evidence_id") or "") not in set(decision.get("blocked_evidence_ids") or [])
    ]
    gate_evidence = _filter_answer_domain_evidence(state, gate_evidence)
    return _select_evidence_window(state, gate_evidence)


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
    metadata = {**(item.get("metadata") or {}), "answer_fields": dict(fields)}
    limitations = list(item.get("limitations") or [])
    if fields and _topology_dependency(state) and not fields.get("single_failure_boundary"):
        metadata["usage_scope"] = "model_content_only"
        limitations.append("本视图只证明该模型自身记载的成功判据与结构；未取得本轮指定失效状态的已验证可生存性关系，不能直接授权该状态下的正式拓扑、数值或继续运行结论。")
    summary = str(fields.get("structure_description") or fields.get("structure_expression")
                  or fields.get("success_criterion") or "")
    return {**item, "_citation_fields_only": True,
            "display_summary": summary or "当前对象未获得可引用的正式结构字段。",
            "limitations": limitations, "metadata": metadata}
