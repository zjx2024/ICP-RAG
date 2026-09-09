"""Order the current candidate view; ranks never decide identity or coverage.

The request's model-context budget owns size. This module deliberately has no
per-family quotas, top-two history focus, or raw cross-retriever score cutoffs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from backend.agent.nodes.answer_collections import _requested_closed_collection_group
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer_fault_rules import _fault_candidate_fit


def _select_evidence_window(
    state: Dict[str, Any], evidence: List[Dict[str, Any]], *, limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Preserve all offered candidates, order by one request-local rerank.

    The legacy limit argument does not authorize silent selection. A complete
    prompt is checked at the shared model-capacity boundary; oversized work
    is reported explicitly rather than deleting records here.
    """
    if limit is not None and limit <= 0:
        return []
    decision = state.get("safety_decision") or {}
    approved = approved_id_set(decision.get("approved_evidence_ids") or [])
    anchors = {
        (str(a.get("doc_type") or ""), str(a.get("record_id") or ""))
        for a in state.get("selected_conversation_anchors") or []
        if isinstance(a, Mapping)
    }
    collection = _requested_closed_collection_group(evidence, state)
    collection_indexes = set(collection["indexes"]) if collection else set()
    has_rerank = any(isinstance(row.get("answer_rerank_score"), (int, float)) for row in evidence)
    calculating = bool(state.get("calculation_requests"))

    def priority(pair):
        index, row = pair
        score = row.get("answer_rerank_score")
        return (
            0 if row.get("evidence_id") in approved else 1,
            0 if index in collection_indexes else 1,
            {"direct": 0, "conditional": 1, "unknown": 1, "contradictory": 2}.get(_fault_candidate_fit(state, row), 1),
            0 if (str(row.get("doc_type") or ""), str(row.get("record_id") or "")) in anchors else 1,
            0 if calculating and row.get("authority") == "deterministic_derivation" else 1,
            -float(score) if has_rerank and isinstance(score, (int, float)) else 0,
            index,
        )

    return [row for _, row in sorted(enumerate(evidence), key=priority)]
