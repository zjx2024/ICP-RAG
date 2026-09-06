"""Rank retrieval candidates by scoped record identity, never by fault wording.

These scores only order retrieval clues. They cannot establish diagnosis,
cross-document equivalence, evidence approval or calculation permission.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from backend.agent.ranking_policy import CandidateRankingWeights
from backend.agent.record_identity import RecordIdentity


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _compatible_scope(identity: RecordIdentity, row: Mapping[str, Any]) -> bool:
    if row.get("doc_id") and identity.doc_id != _text(row["doc_id"]):
        return False
    if row.get("revision_id") and identity.revision_id != _text(row["revision_id"]):
        return False
    if row.get("variant_id") and identity.variant_id != _text(row["variant_id"]):
        return False
    scope = dict(identity.scope)
    return all(scope.get(key) == _text(value) for key, value in (row.get("candidate_scope") or {}).items() if _text(value))


def _supporting_ids(identity: RecordIdentity, row: Mapping[str, Any], evidence, *, search: bool) -> list[str]:
    result = []
    for item in evidence:
        if item.get("doc_type") != identity.doc_type or item.get("claim_type") == "maintenance_action":
            continue
        if identity.doc_id and item.get("doc_id") != identity.doc_id:
            continue
        if search:
            if not row.get("chunk_id") or item.get("chunk_id") != row["chunk_id"]:
                continue
        elif not identity.record_id or _text(item.get("record_id")) != identity.record_id:
            continue
        item_variant = _text((item.get("metadata") or {}).get("variant_id"))
        if identity.variant_id and item_variant != identity.variant_id:
            continue
        item_scope = item.get("scope") or {}
        if any(item_scope.get(key) and item_scope[key] != value for key, value in identity.scope):
            continue
        item_revision = (item.get("metadata") or {}).get("revision_id")
        if identity.revision_id and item_revision and identity.revision_id != _text(item_revision):
            continue
        evidence_id = _text(item.get("evidence_id"))
        if evidence_id and evidence_id not in result:
            result.append(evidence_id)
    return result


def rank_candidates(
    hmc_rows: Sequence[Mapping[str, Any]], fmeca_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]],
    *, weights: CandidateRankingWeights, structured_hmc_matches: frozenset[str],
) -> list[dict[str, Any]]:
    candidates: dict[RecordIdentity, dict[str, Any]] = {}

    def candidate(identity, label):
        return candidates.setdefault(identity, {
            "candidate_id": identity.candidate_id, "record_identity": identity.public(),
            "fault": label, "evidence": [], "supporting_evidence_ids": [],
            "exact_hmc_match": False, "structured_hmc_match": False, "authoritative_support": False,
            "contributions": {},
        })

    def add_support(item, source, record_id, ids, contribution, score):
        reference = {"source": source, "id": record_id}
        if reference not in item["evidence"]:
            item["evidence"].append(reference)
        item["supporting_evidence_ids"] = list(dict.fromkeys([*item["supporting_evidence_ids"], *ids]))
        # Repeated rows/chunks are not independent corroboration. Their best
        # retrieval score is retained; duplicates cannot accumulate authority.
        item["contributions"][contribution] = max(item["contributions"].get(contribution, 0), score)

    for doc_type, rows in (("hmc", hmc_rows), ("fmeca", fmeca_rows)):
        for index, row in enumerate(rows):
            label = _text(row.get("fault_desc") if doc_type == "hmc" else row.get("fault_mode")) or _text(row.get("record_id"))
            if not label:
                continue
            identity = RecordIdentity.from_row(doc_type, row, fallback=f"unresolved-row:{index}")
            item = candidate(identity, label)
            match = _text(row.get("match_type"))
            exact = bool(identity.record_id) and doc_type == "hmc" and match in {"exact_code", "exact_conversation_reference"}
            structured = bool(identity.record_id) and doc_type == "hmc" and match in structured_hmc_matches
            item["exact_hmc_match"] |= exact
            item["structured_hmc_match"] |= structured
            item["authoritative_support"] |= bool(identity.record_id)
            score = weights.fmeca if doc_type == "fmeca" else (
                weights.hmc_exact if exact else weights.hmc_components if match == "exact_components" else weights.hmc_keyword)
            add_support(item, doc_type.upper(), identity.record_id,
                        _supporting_ids(identity, row, evidence, search=False), "record", score)

    authoritative = list(candidates)
    for index, row in enumerate(search_rows[:12]):
        doc_type = _text(row.get("doc_type"))
        if doc_type not in {"hmc", "fmeca"}:
            continue
        title = _text(row.get("title"))
        if not title:
            continue
        # A title is accepted as a code only when it is literally an existing
        # PG business ID. Unknown source scope must not choose among duplicates.
        code = _text(row.get("record_id")) or title
        matches = [identity for identity in authoritative
                   if identity.doc_type == doc_type and identity.record_id == code
                   and _compatible_scope(identity, row)]
        if len(matches) == 1:
            identity = matches[0]
        else:
            identity = RecordIdentity.from_row(doc_type, row, fallback=f"unresolved-hit:{index}")
        item = candidate(identity, title)
        try:
            score = float(row.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = min(max(score, 0), 1) if math.isfinite(score) else 0
        add_support(item, doc_type, row.get("chunk_id"),
                    _supporting_ids(identity, row, evidence, search=True), "search",
                    weights.search_base + weights.search_score * score)

    ranked = []
    for item in candidates.values():
        scores = item.pop("contributions")
        ranked.append({**item, "score": round(min(sum(scores.values()), 0.99), 3),
                       "evidence": item["evidence"][:6],
                       "supporting_evidence_ids": item["supporting_evidence_ids"][:8]})
    ranked.sort(key=lambda item: (-int(item["exact_hmc_match"]), -int(item["structured_hmc_match"]),
                                  -int(item["authoritative_support"]), -item["score"], item["candidate_id"]))
    return ranked[:8]
