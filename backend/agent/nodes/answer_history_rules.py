"""History applicability and identity rules independent of answer orchestration.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import required_answer_fields
from backend.agent.history_focus import history_case_identity
from backend.agent.history_focus import resolve_history_parent_cases
from backend.agent.nodes.answer_payload import _answer_relevance_score
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_payload import _evidence_object_business_ids
from backend.agent.nodes.answer_payload import _normalize_business_id
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.query_understanding import extract_business_codes
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
import re


HISTORY_SUPPLEMENT_MIN_SCORE = 0.15


HISTORY_SUPPLEMENT_RELATIVE_RATIO = 0.75


HISTORY_SUPPLEMENT_MAX_CASES = 2


def _history_relative_focus_ids(
    state: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> set[str] | None:
    """Resolve the history candidates chosen once for this turn.

    Exact identities and selected conversation anchors remain authoritative.
    Retrieval-ranked candidates may be bounded for presentation, but never
    become exact parent identities merely because their semantic score is high.
    """

    if not _history_is_primary_target(state) or _history_collection_requested(state):
        return None
    candidates = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "fault_history"
        and str(item.get("evidence_id") or "")
    ]
    selection = resolve_history_parent_cases(
        state,
        candidates,
        limit=HISTORY_SUPPLEMENT_MAX_CASES,
    )
    if not selection.selected_case_ids:
        return set()
    return {
        str(item.get("evidence_id") or "")
        for item in candidates
        if history_case_identity(item) in selection.selected_case_ids
    }


def _history_evidence_is_strong(item: Mapping[str, Any]) -> bool:
    """Whether a history row is relevant enough for supplemental prose.

    Exact-reference rows are created only from an explicit business-code
    linkage.  Semantic identity hydration additionally needs a retrieval score
    threshold: PostgreSQL identity confirms the row exists, but does not by
    itself make a weakly recalled case similar to the user's fault.
    """

    match_type = str(item.get("match_type") or "")
    if match_type == "exact_history_reference":
        return True
    if match_type != "semantic_id_hydration":
        return False
    # 公开引用压缩会剔除私有 rerank 字段，重新打上的 answer_rerank_score
    # 才是答案窗口实际使用的分数。优先读它，避免健康案例被判为弱匹配。
    raw_score = item.get("answer_rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("retrieval_score")
    return (
        isinstance(raw_score, (int, float))
        and float(raw_score) >= HISTORY_SUPPLEMENT_MIN_SCORE
    )


def _focused_history_explicit_references(
    state: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> Dict[str, set[str]] | None:
    """Return links owned by the one history case selected for this turn.

    A continuation can carry several exact record identities from the preceding
    answer.  Those identities are retrieval anchors, not proof that every old
    candidate remains the subject after the user explicitly asks for a similar
    history case.  Once the answer-side history selector resolves exactly one
    authoritative case, that case's stored cross-document references form a
    stronger boundary for its FMECA/MFL children.

    The boundary is intentionally fail-closed: it is absent for history
    inventories, optional/implicit history enrichment, multiple distinct cases,
    unapproved rows, or rows without a durable case identity.  Duplicate
    representations of the same case are merged, because they do not make the
    user request multi-object.
    """

    target_intents = {
        str(target.get("intent") or "").strip()
        for target in (state.get("query_targets") or [])
        if isinstance(target, Mapping) and str(target.get("intent") or "").strip()
    }
    primary_intent = str(state.get("intent") or "").strip()
    explicitly_targeted = (
        "history_query" in target_intents or primary_intent == "history_query"
    )
    if (
        not explicitly_targeted
        or not _history_is_primary_target(state)
        or _history_collection_requested(state)
    ):
        return None

    decision = (
        state.get("safety_decision")
        if isinstance(state.get("safety_decision"), Mapping)
        else {}
    )
    approval_is_explicit = "approved_evidence_ids" in decision
    approved_ids = approved_id_set(decision.get("approved_evidence_ids") or [])
    eligible_items: List[Mapping[str, Any]] = []
    direct_match_types = {
        "exact_conversation_reference",
        "exact_history_reference",
        "exact_record_identity",
        "exact_structured_identity",
        "structured_query",
    }
    for item in evidence:
        if (
            str(item.get("doc_type") or "") != "fault_history"
            or str(item.get("authority") or "") != "authoritative_record"
        ):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if approval_is_explicit and evidence_id not in approved_ids:
            continue
        match_type = str(item.get("match_type") or "")
        if match_type not in direct_match_types and not _history_evidence_is_strong(item):
            continue
        eligible_items.append(item)

    selection = resolve_history_parent_cases(
        state,
        eligible_items,
        require_explicit_target=True,
        limit=HISTORY_SUPPLEMENT_MAX_CASES,
    )
    if not selection.authorizes_exact_links or len(selection.selected_case_ids) != 1:
        return None

    selected_case_id = next(iter(selection.selected_case_ids))
    representations = [
        item
        for item in eligible_items
        if history_case_identity(item) == selected_case_id
    ]
    if not representations:
        return None
    case_id = _normalize_business_id(
        _evidence_answer_fields(representations[0]).get("case_id")
        or representations[0].get("record_id")
        or representations[0].get("title")
    )
    result: Dict[str, set[str]] = {
        "case_ids": {case_id},
        "fmeca": set(),
        "mfl": set(),
    }
    for item in representations:
        references = _evidence_answer_fields(item).get("explicit_references")
        if not isinstance(references, Mapping):
            continue
        for doc_type in ("fmeca", "mfl"):
            raw_values = references.get(doc_type)
            values = (
                list(raw_values)
                if isinstance(raw_values, (list, tuple, set))
                else re.split(r"[,，;；\s]+", str(raw_values or ""))
            )
            result[doc_type].update(
                str(value or "").strip().upper()
                for value in values
                if str(value or "").strip()
            )
    return result


def _supplemental_history_evidence_ids(
    evidence: Sequence[Mapping[str, Any]],
    *,
    limit: int = HISTORY_SUPPLEMENT_MAX_CASES,
) -> set[str]:
    """Select a small, query-relative set of genuinely similar cases.

    An exact cross-document history reference is stronger than a semantic
    neighbor.  Semantic scores from different queries are not globally
    calibrated, therefore an absolute floor alone is insufficient: after the
    floor, candidates must also stay close to the best case retrieved for this
    query.  Selection is by unique case identity so duplicate OpenSearch and
    PostgreSQL representations cannot consume the answer budget twice.
    """

    bounded_limit = max(1, int(limit or HISTORY_SUPPLEMENT_MAX_CASES))

    def score(item: Mapping[str, Any]) -> float:
        # Keep this identical to _history_evidence_is_strong's score source.
        # answer_rerank_score is re-applied by the answer window after public
        # citation compaction strips the private rerank fields; without it a
        # healthy hydrated case falls back to -1.0 and the relative floor
        # below silently drops the closest case.
        for key in ("answer_rerank_score", "rerank_score", "retrieval_score"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return -1.0

    exact = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "fault_history"
        and str(item.get("match_type") or "") == "exact_history_reference"
    ]
    # An exact stored relationship is already the selected historical parent
    # for this turn.  Semantic neighbours remain auditable candidates, but
    # must not be co-presented as though they shared that exact binding.
    if exact:
        semantic = []
    else:
        semantic = [
            item
            for item in evidence
            if str(item.get("doc_type") or "") == "fault_history"
            and _history_evidence_is_strong(item)
        ]
    if semantic:
        best_score = max(score(item) for item in semantic)
        relative_floor = max(
            HISTORY_SUPPLEMENT_MIN_SCORE,
            best_score * HISTORY_SUPPLEMENT_RELATIVE_RATIO,
        )
        semantic = [item for item in semantic if score(item) >= relative_floor]

    ranked = [
        *sorted(exact, key=score, reverse=True),
        *sorted(semantic, key=score, reverse=True),
    ]
    selected_ids: set[str] = set()
    selected_records: set[str] = set()
    for item in ranked:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id:
            continue
        record_identity = str(
            item.get("record_id") or item.get("title") or evidence_id
        ).strip()
        if record_identity in selected_records:
            continue
        selected_ids.add(evidence_id)
        selected_records.add(record_identity)
        if len(selected_records) >= bounded_limit:
            break
    return selected_ids


def _history_subject_matches_state(
    history_item: Mapping[str, Any],
    state: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    apply_relative_focus: bool = True,
) -> bool:
    """Keep an automatic history supplement tied to this turn's subject.

    A high semantic score such as ``高负载`` is not enough to call a case
    similar: the power-path incident and a thermal-controller incident can
    share those words while requiring different checks and repairs.  When the
    current evidence already binds a concrete FMECA/HMC/MFL subject, prefer
    explicit cross-document IDs and module/function IDs.  If the history row
    carries explicit links which all disagree with that subject, it is audit
    context only and must not be written as a previous occurrence.
    """

    history_fields = _evidence_answer_fields(history_item)
    focus_ids = (
        _history_relative_focus_ids(state, evidence)
        if apply_relative_focus
        else None
    )
    current_evidence_id = str(history_item.get("evidence_id") or "")
    if focus_ids is not None and current_evidence_id not in focus_ids:
        return False

    anchors = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") in {"fmeca", "hmc", "mfl", "function_locator"}
        and str(item.get("record_id") or "").strip()
    ]
    if not anchors:
        return True

    explicit = history_fields.get("explicit_references")
    explicit_values: set[str] = set()
    if isinstance(explicit, Mapping):
        for raw_values in explicit.values():
            values = (
                list(raw_values)
                if isinstance(raw_values, (list, tuple, set))
                else re.split(r"[,，;；\s]+", str(raw_values or ""))
            )
            explicit_values.update(
                str(value or "").strip().upper()
                for value in values
                if str(value or "").strip()
            )

    # A follow-up may select several citations from the preceding answer while
    # referring back to the same observed phenomenon.  An explicit stored link
    # from a history case to one of those selected records is stronger than a
    # fresh semantic score.
    selected_anchor_record_ids = {
        str(anchor.get("record_id") or "").strip().upper()
        for anchor in (state.get("selected_conversation_anchors") or [])
        if isinstance(anchor, Mapping)
        and str(anchor.get("record_id") or "").strip()
    }
    if explicit_values & selected_anchor_record_ids:
        return True

    current_record_ids = {
        str(item.get("record_id") or "").strip().upper()
        for item in anchors
        if str(item.get("record_id") or "").strip()
    }
    # When retrieval produced several FMECA rows, only the highest-ranked
    # subject is the current incident anchor.  Letting any lower-ranked row
    # authorize its linked history would reintroduce a different module's
    # case merely because both mention a generic condition such as “高负载”.
    fmeca_anchors = [
        item for item in anchors if str(item.get("doc_type") or "") == "fmeca"
    ]
    current_anchor_record_ids = current_record_ids
    if fmeca_anchors:
        best_fmeca = max(fmeca_anchors, key=_answer_relevance_score)
        best_id = str(best_fmeca.get("record_id") or "").strip().upper()
        if best_id:
            current_anchor_record_ids = {best_id}
    current_object_ids: set[str] = set()
    for item in anchors:
        # Use the same primary FMECA anchor as the record-ID check below.  A
        # lower-ranked row from another module must not authorize its own
        # history merely because its FUNC-* ID is present in the broad recall.
        if fmeca_anchors and item is not best_fmeca:
            continue
        current_object_ids.update(_evidence_object_business_ids(item))
    history_object_ids = {
        _normalize_business_id(value)
        for value in extract_business_codes(
            [
                history_fields.get("related_module"),
                history_fields.get("related_function"),
                history_fields.get("related_interface"),
                history_fields.get("hmc_code"),
            ],
            limit=48,
        )
        if _normalize_business_id(value)
    }

    if explicit_values:
        if explicit_values & current_anchor_record_ids:
            return True
        if history_object_ids & current_object_ids:
            return True
        # Explicit links exist but none point to the current bound subject.
        # Do not let lexical similarity turn a different module's case into a
        # historical precedent.
        return False

    if history_object_ids and current_object_ids:
        return bool(history_object_ids & current_object_ids)
    # With a concrete current object, a semantic neighbour without any stored
    # relationship or shared business identity remains audit-only.  Open text
    # similarity may rank candidates, but it cannot establish that two records
    # concern the same engineering object.
    return False


def _history_is_primary_target(state: Mapping[str, Any]) -> bool:
    """Whether this turn explicitly requests history as an answer dimension.

    A mixed question such as “先怎么处置？以前相近案例怎么验证？” has
    ``maintenance_query`` as its primary intent but still makes the history
    record a direct answer target.  Treating history as supplemental in that
    case lets a FMECA citation masquerade as a historical case.  Automatic
    diagnostic closure without an explicit history target remains supplemental.
    """

    fields = set(required_answer_fields(state))
    if fields & {
        "history_time",
        "history_root",
        "history_steps",
        "history_solution",
        "history_result",
        "history_personnel",
        "history_preventive_action",
    }:
        return True
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    history_focuses = {
        "history_conditions",
        "recovery_verification",
        "exclusion_basis",
        "verification_criteria",
        "history_root_cause",
        "history_solution",
    }
    if history_focuses & {
        str(value or "").strip()
        for value in (frame.get("response_focus") or [])
        if str(value or "").strip()
    }:
        return True
    return any(
        isinstance(item, Mapping)
        and str(item.get("doc_type") or "") == "fault_history"
        and bool(item.get("fields"))
        for item in (frame.get("source_obligations") or [])
    )


def _history_collection_requested(state: Mapping[str, Any]) -> bool:
    """Whether the user asks for an exhaustive history inventory."""

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    return bool(
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and str(frame.get("collection_member_kind") or "") == "record"
    )
