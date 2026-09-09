"""Exact record selection; visibility never establishes a cross-source link."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.agent.answer_contract import required_answer_fields
from backend.agent.nodes.answer_collections import _closed_collection_groups
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_presence import normalize_presence_text
from backend.agent.query_understanding import extract_business_codes


def _narrow_semantic_table_families(
    state: Mapping[str, Any], evidence: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Narrow only a family's explicitly selected records, never its neighbours.

    A selected FMECA row cannot delete independent MFL sources. Nor can the
    highest scoring semantic candidate choose the user's object. The writer
    compares visible records, while source bindings and review separately
    govern whether a relationship or diagnosis can be asserted.
    """
    decision = state.get("safety_decision")
    rows = list(evidence)
    if isinstance(decision, Mapping) and "approved_evidence_ids" in decision:
        approved = set(decision.get("approved_evidence_ids") or [])
        rows = [item for item in rows if item.get("evidence_id") in approved]
    frame = state.get("query_frame") or {}
    if (frame.get("coverage_scope") == "complete_collection" and _closed_collection_groups(rows)) or {
        "structural_cause_chain", "relation_path",
    } & set(required_answer_fields(state)) or (state.get("structured_record_binding") or {}).get("ambiguous"):
        return rows
    current_codes = {
        normalize_presence_text(value) for value in extract_business_codes(
            str(frame.get("original_query") or state.get("query") or ""), limit=48,
        )
    }

    def record_id(item: Mapping[str, Any]) -> str:
        return normalize_presence_text(
            item.get("record_id") or _evidence_answer_fields(item).get("record_code")
        )

    selections: dict[str, set[str]] = {}
    for family in ("fmeca", "mfl"):
        members = [item for item in rows if item.get("doc_type") == family]
        available = {record_id(item) for item in members}
        explicit = available & current_codes
        # Only a current user-authored record ID narrows this source family.
        # Prior citations and neighbouring explicit links remain useful evidence,
        # but do not establish that other candidates are unrelated to this turn.
        selected = explicit
        if selected:
            selections[family] = selected
    return [item for item in rows if item.get("doc_type") not in selections
            or record_id(item) in selections[item["doc_type"]]]
