"""Pure, query-relative selection of fault-history parent records.

Both retrieval hydration and answer composition need to decide which history
case the current turn is about.  Keeping that decision here avoids importing
the answer node from a tool module (which would create a cycle) and, more
importantly, prevents the two stages from choosing different parents.

The selector never treats a semantic-search score as an identity.  Current
user-authored IDs and already selected history citations have priority; normal
language relevance remains the responsibility of retrieval/rerank models and
is carried only as a bounded candidate set.  Ambiguity stays visible to callers
so exact child links cannot be manufactured from an unselected case.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.agent.query_understanding import extract_business_codes


HISTORY_FOCUS_MAX_CASES = 2


HISTORY_PARENT_BINDING_VERSION = 1


# These match types are created only after a durable record identity or an
# explicit stored relationship has been verified.  A normal structured query
# still proves that the returned row exists, but it does not prove that this
# row is the object meant by an open-language question.
_IDENTITY_MATCH_TYPES = frozenset(
    {
        "exact_conversation_reference",
        "exact_history_reference",
        "exact_record_identity",
        "exact_structured_identity",
    }
)


@dataclass(frozen=True)
class HistoryParentSelection:
    """Auditable result of choosing history parent records."""

    selected_case_ids: frozenset[str] = frozenset()
    source: str = "none"
    comparison: bool = False
    ambiguous: bool = False
    conflict: bool = False
    # ``identity_authorized`` is deliberately independent from relevance.
    # A semantic candidate may be suitable for a clearly labelled comparison,
    # yet it must never authorize exact HMC/FMECA/MFL child hydration.
    identity_authorized: bool = False

    @property
    def authorizes_exact_links(self) -> bool:
        return bool(
            self.identity_authorized
            and self.selected_case_ids
            and not self.ambiguous
            and not self.conflict
        )


def history_parent_binding(
    selection: HistoryParentSelection,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Serialize one tool-stage parent decision for downstream consumers.

    The binding contains only stable identities and decision metadata.  It is
    not evidence itself and never carries a previous answer's prose.
    """

    selected = set(selection.selected_case_ids)
    row_ids: List[str] = []
    for row in rows:
        if history_case_identity(row) not in selected:
            continue
        value = str(row.get("id") or "").strip()
        if value and value not in row_ids:
            row_ids.append(value)
    if selection.conflict:
        authority = "conflict"
    elif selection.ambiguous:
        authority = "ambiguous"
    elif selection.identity_authorized:
        authority = "identity"
    elif selected:
        authority = "candidate"
    else:
        authority = "unresolved"
    return {
        "version": HISTORY_PARENT_BINDING_VERSION,
        "case_ids": sorted(selected),
        "row_ids": row_ids,
        "source": selection.source,
        "authority": authority,
        "comparison": bool(selection.comparison),
        "ambiguous": bool(selection.ambiguous),
        "conflict": bool(selection.conflict),
        "authorizes_exact_links": selection.authorizes_exact_links,
    }


def history_parent_selection_from_binding(
    state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> HistoryParentSelection | None:
    """Validate and restore the tool-stage parent decision from state.

    Returning ``None`` means no binding was produced in this turn and callers
    may run the selector.  A present but malformed/stale binding fails closed;
    it is never silently replaced by a new semantic choice in the answer node.
    """

    raw = state.get("history_parent_binding")
    if not isinstance(raw, Mapping) or not raw:
        return None
    try:
        version = int(raw.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version != HISTORY_PARENT_BINDING_VERSION:
        return HistoryParentSelection(
            source="invalid_binding",
            ambiguous=True,
            conflict=True,
        )

    requested_ids = {
        _stable_identity(value)
        for value in (raw.get("case_ids") or [])
        if _stable_identity(value)
    }
    available_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        case_id = history_case_identity(row)
        if case_id:
            available_rows.setdefault(case_id, []).append(row)
    available_ids = set(available_rows)
    conflict = bool(raw.get("conflict")) or bool(requested_ids - available_ids)
    ambiguous = bool(raw.get("ambiguous"))
    authority = str(raw.get("authority") or "").strip()

    # A serialized decision is an audit record, not a capability token.  When
    # it claims identity authority, rebuild the decision from the current raw
    # turn, selected citations and the rows that are actually available now.
    # This prevents a forged/stale ``authority=identity`` boolean from turning
    # a semantic history candidate into exact HMC/FMECA/MFL child lookups.
    identity_authorized = False
    recomputed: HistoryParentSelection | None = None
    if authority == "identity" and raw.get("authorizes_exact_links") is True:
        unbound_state = dict(state)
        unbound_state.pop("history_parent_binding", None)
        recomputed = select_history_parent_cases(unbound_state, rows)

        claimed_row_ids = {
            str(value or "").strip()
            for value in (raw.get("row_ids") or [])
            if str(value or "").strip()
        }
        available_row_ids = {
            str(row.get("id") or "").strip()
            for case_id in requested_ids
            for row in available_rows.get(case_id, [])
            if str(row.get("id") or "").strip()
        }
        row_identity_valid = bool(
            (not available_row_ids and not claimed_row_ids)
            or (
                claimed_row_ids
                and claimed_row_ids.issubset(available_row_ids)
            )
        )
        identity_authorized = bool(
            recomputed.authorizes_exact_links
            and requested_ids == set(recomputed.selected_case_ids)
            and row_identity_valid
            and not conflict
            and not ambiguous
        )
        if not identity_authorized:
            conflict = True
            ambiguous = True

    comparison = (
        bool(recomputed.comparison)
        if recomputed is not None
        else _comparison_requested(state)
    )
    if len(requested_ids) > 1 and not comparison:
        ambiguous = True
        identity_authorized = False
    return HistoryParentSelection(
        selected_case_ids=frozenset(requested_ids & available_ids),
        source=str(raw.get("source") or "persisted_binding"),
        comparison=comparison,
        ambiguous=ambiguous or (bool(requested_ids) and not bool(requested_ids & available_ids)),
        conflict=conflict,
        identity_authorized=identity_authorized,
    )


def merge_history_parent_bindings(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
    *,
    limit: int = HISTORY_FOCUS_MAX_CASES,
) -> Dict[str, Any]:
    """Merge bounded adaptive batches without changing identity authority."""

    previous = dict(first or {})
    current = dict(second or {})
    if not previous:
        return current
    if not current:
        return previous

    def ids(binding: Mapping[str, Any], key: str) -> List[str]:
        result: List[str] = []
        for value in binding.get(key) or []:
            normalized = str(value or "").strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    previous_cases = ids(previous, "case_ids")
    current_cases = ids(current, "case_ids")
    previous_authority = str(previous.get("authority") or "unresolved")
    current_authority = str(current.get("authority") or "unresolved")

    if previous.get("conflict") is True or current.get("conflict") is True:
        return {
            "version": HISTORY_PARENT_BINDING_VERSION,
            "case_ids": [],
            "row_ids": [],
            "source": "adaptive_binding_conflict",
            "authority": "conflict",
            "comparison": False,
            "ambiguous": True,
            "conflict": True,
            "authorizes_exact_links": False,
        }

    previous_identity = (
        previous_authority == "identity"
        and previous.get("authorizes_exact_links") is True
    )
    current_identity = (
        current_authority == "identity"
        and current.get("authorizes_exact_links") is True
    )
    if previous_identity and current_identity:
        if set(previous_cases) != set(current_cases):
            return {
                "version": HISTORY_PARENT_BINDING_VERSION,
                "case_ids": [],
                "row_ids": [],
                "source": "adaptive_identity_conflict",
                "authority": "conflict",
                "comparison": False,
                "ambiguous": True,
                "conflict": True,
                "authorizes_exact_links": False,
            }
        merged = dict(previous)
        merged["row_ids"] = list(
            dict.fromkeys([*ids(previous, "row_ids"), *ids(current, "row_ids")])
        )
        return merged
    if previous_identity:
        return previous
    if current_identity:
        return current

    # An empty adaptive batch cannot erase a useful earlier candidate set.
    if not current_cases:
        return previous
    if not previous_cases:
        return current
    candidates = list(dict.fromkeys([*previous_cases, *current_cases]))
    bounded = candidates[: max(1, int(limit))]
    return {
        "version": HISTORY_PARENT_BINDING_VERSION,
        "case_ids": bounded,
        "row_ids": list(
            dict.fromkeys([*ids(previous, "row_ids"), *ids(current, "row_ids")])
        ),
        "source": "adaptive_candidates",
        "authority": "candidate" if len(candidates) == 1 else "ambiguous",
        "comparison": bool(previous.get("comparison") or current.get("comparison")),
        "ambiguous": len(candidates) != 1,
        "conflict": False,
        "authorizes_exact_links": False,
    }


def resolve_history_parent_cases(
    state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    require_explicit_target: bool = False,
    limit: int = HISTORY_FOCUS_MAX_CASES,
) -> HistoryParentSelection:
    """Use the persisted tool decision, or select once when none exists."""

    bound = history_parent_selection_from_binding(state, rows)
    if bound is not None:
        return bound
    return select_history_parent_cases(
        state,
        rows,
        require_explicit_target=require_explicit_target,
        limit=limit,
    )


def _stable_identity(value: Any) -> str:
    return str(value or "").strip().casefold()


def _non_history_code_key(value: Any) -> str:
    """Normalize only deterministic non-history code punctuation.

    HMC manuals commonly render the same LCN/HAMC/HSC identity with or
    without separators.  Treating those display variants as a new history
    case would incorrectly invalidate a genuinely selected history anchor.
    No natural-language aliases or approximate matching are used here.
    """

    stable = _stable_identity(value)
    compact = re.sub(r"[-_/\.\s]+", "", stable)
    if re.fullmatch(r"\d{6,}\*{2}[a-z0-9]+", compact):
        return f"hmc:{compact}"
    return stable


def _code_namespace(value: Any) -> str:
    text = _stable_identity(value)
    parts = re.split(r"[-_./]", text, maxsplit=1)
    return parts[0] if len(parts) == 2 and parts[0] else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def history_answer_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Read normalized fields from raw PG rows and public evidence alike."""

    result = dict(row)
    for container_name in ("metadata", "document_metadata"):
        container = _mapping(row.get(container_name))
        for key, value in container.items():
            result.setdefault(str(key), value)
        answer_fields = _mapping(container.get("answer_fields"))
        result.update(answer_fields)
    result.update(_mapping(row.get("answer_fields")))
    return result


def history_case_identity(row: Mapping[str, Any]) -> str:
    """Return one stable case identity without erasing ID punctuation."""

    fields = history_answer_fields(row)
    value = fields.get("case_id") or row.get("case_id")
    if not str(value or "").strip():
        value = row.get("record_id") or row.get("title")
    return _stable_identity(value)


def history_turn_explicitly_targeted(state: Mapping[str, Any]) -> bool:
    """Whether planner output makes history a requested current-turn facet."""

    frame = _mapping(state.get("query_frame"))
    intents: set[str] = set()
    for raw_targets in (state.get("query_targets"), frame.get("query_targets")):
        for target in raw_targets or []:
            if isinstance(target, Mapping):
                value = target.get("intent")
            else:
                value = target
            if str(value or "").strip():
                intents.add(str(value).strip())
    if str(state.get("intent") or "").strip():
        intents.add(str(state.get("intent")).strip())
    if "history_query" in intents:
        return True

    fields: List[str] = []
    for values in (
        state.get("required_answer_fields"),
        frame.get("required_answer_fields"),
        frame.get("response_focus"),
    ):
        fields.extend(str(value or "").strip() for value in values or [])
    return any(value.startswith("history_") for value in fields)


def _original_query(state: Mapping[str, Any]) -> str:
    frame = _mapping(state.get("query_frame"))
    return str(frame.get("original_query") or state.get("query") or "").strip()


def _case_score(rows: Sequence[Mapping[str, Any]]) -> float:
    """Use answer-stage relevance only; raw semantic recall is not authority."""

    values: List[float] = []
    for row in rows:
        raw = row.get("answer_rerank_score")
        if not isinstance(raw, (int, float)):
            raw = row.get("rerank_score")
        if not isinstance(raw, (int, float)):
            raw = row.get("retrieval_score")
        if not isinstance(raw, (int, float)):
            raw = row.get("retrieval_match_score")
        if isinstance(raw, (int, float)):
            values.append(float(raw))
    return max(values, default=-1.0)


def _comparison_requested(state: Mapping[str, Any]) -> bool:
    frame = _mapping(state.get("query_frame"))
    return str(frame.get("question_semantics") or "").strip() == "compare"


def _selected_history_anchor_ids(state: Mapping[str, Any]) -> set[str]:
    return {
        _stable_identity(anchor.get("record_id") or anchor.get("case_id"))
        for anchor in (state.get("selected_conversation_anchors") or [])
        if isinstance(anchor, Mapping)
        and str(anchor.get("doc_type") or "").strip() == "fault_history"
        and _stable_identity(anchor.get("record_id") or anchor.get("case_id"))
    }


_NON_HISTORY_ANCHOR_ID_FIELDS = frozenset({
    "record_code",
    "hmc_code",
    "lcn",
    "hamc",
    "hsc",
    "mfl_code",
    "component_id",
    "module_id",
    "function_id",
    "interface_id",
    "event_id",
    "diagram_id",
    "diagram_key",
})


def _selected_non_history_anchor_codes(state: Mapping[str, Any]) -> set[str]:
    """Return current non-history identities that explain user-authored codes.

    A mixed question may name an HMC/FMECA/MFL object while also asking for a
    similar case.  Such a code must not be mistaken for a new history-case ID.
    Only persisted identity columns from selected, non-history citations may
    exempt it; titles and narrative fields are deliberately excluded.
    """

    result: set[str] = set()
    for anchor in state.get("selected_conversation_anchors") or []:
        if (
            not isinstance(anchor, Mapping)
            or str(anchor.get("doc_type") or "").strip() == "fault_history"
        ):
            continue
        values: List[Any] = [anchor.get("record_id")]
        fields = _mapping(anchor.get("answer_fields"))
        values.extend(fields.get(key) for key in _NON_HISTORY_ANCHOR_ID_FIELDS)
        for value in values:
            if isinstance(value, (list, tuple, set, frozenset)):
                candidates = value
            else:
                candidates = (value,)
            for candidate in candidates:
                normalized = _stable_identity(candidate)
                if normalized:
                    result.add(_non_history_code_key(normalized))
                result.update(
                    _non_history_code_key(code)
                    for code in extract_business_codes(candidate, limit=16)
                    if _non_history_code_key(code)
                )
    return result


def select_history_parent_cases(
    state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    require_explicit_target: bool = False,
    limit: int = HISTORY_FOCUS_MAX_CASES,
) -> HistoryParentSelection:
    """Select current history parents from authoritative candidate rows.

    Multiple current IDs or selected citations are allowed only for an
    explicit comparison.  For free-text focus a close second case may be
    returned as candidate context, but no semantic score authorizes an exact
    child lookup.
    """

    comparison = _comparison_requested(state)
    if require_explicit_target and not history_turn_explicitly_targeted(state):
        return HistoryParentSelection(source="no_history_target", comparison=comparison)

    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        case_id = history_case_identity(row)
        if case_id:
            groups.setdefault(case_id, []).append(row)
    if not groups:
        return HistoryParentSelection(source="no_identity", comparison=comparison)

    def scope_conflict(case_ids: set[str], *, anchors: bool = False) -> bool:
        for case_id in case_ids:
            documents = {str(row.get("doc_id") or "") for row in groups.get(case_id, [])}
            if len(documents) > 1:
                return True
            if anchors:
                expected = {str(anchor.get("doc_id"))
                            for anchor in state.get("selected_conversation_anchors") or []
                            if isinstance(anchor, Mapping) and anchor.get("doc_type") == "fault_history"
                            and _stable_identity(anchor.get("record_id")) == case_id and anchor.get("doc_id")}
                if expected and documents != expected:
                    return True
        return False

    original_codes = {
        _stable_identity(value)
        for value in extract_business_codes(_original_query(state), limit=48)
        if _stable_identity(value)
    }
    namespaces = {
        namespace
        for case_id in groups
        if (namespace := _code_namespace(case_id))
    }
    current_case_codes = {
        code
        for code in original_codes
        if code in groups or _code_namespace(code) in namespaces
    }
    if current_case_codes:
        unresolved = current_case_codes - set(groups)
        selected = current_case_codes & set(groups)
        if unresolved or not selected or scope_conflict(selected):
            return HistoryParentSelection(
                source="current_query_id",
                comparison=comparison,
                ambiguous=True,
                conflict=True,
            )
        if len(selected) > 1 and not comparison:
            return HistoryParentSelection(
                source="current_query_id",
                comparison=False,
                ambiguous=True,
            )
        return HistoryParentSelection(
            selected_case_ids=frozenset(selected),
            source="current_query_id",
            comparison=comparison,
            identity_authorized=True,
        )

    anchor_ids = _selected_history_anchor_ids(state)
    selected_non_history_codes = _selected_non_history_anchor_codes(state)
    unexplained_current_codes = {
        code
        for code in original_codes
        if _non_history_code_key(code) not in selected_non_history_codes
    }
    if anchor_ids and unexplained_current_codes:
        # The current user turn contains a durable identity which is neither a
        # retrieved history parent nor an explicitly selected non-history
        # object.  In particular, a different namespace must not make that ID
        # disappear and silently re-authorize an older history anchor.  Fail
        # closed until the current ID can be re-read or clarified.
        return HistoryParentSelection(
            source="current_query_unresolved_id",
            comparison=comparison,
            ambiguous=True,
            conflict=True,
        )
    if anchor_ids:
        unresolved = anchor_ids - set(groups)
        selected = anchor_ids & set(groups)
        if unresolved or not selected or scope_conflict(selected, anchors=True):
            return HistoryParentSelection(
                source="conversation_anchor",
                comparison=comparison,
                ambiguous=True,
                conflict=True,
            )
        if len(selected) > 1 and not comparison:
            return HistoryParentSelection(
                source="conversation_anchor",
                comparison=False,
                ambiguous=True,
            )
        return HistoryParentSelection(
            selected_case_ids=frozenset(selected),
            source="conversation_anchor",
            comparison=comparison,
            identity_authorized=True,
        )

    if len(groups) == 1:
        if scope_conflict(set(groups)):
            return HistoryParentSelection(source="scoped_identity_conflict", comparison=comparison,
                                          ambiguous=True, conflict=True)
        only_rows = next(iter(groups.values()))
        identity_authorized = any(
            str(row.get("match_type") or "") in _IDENTITY_MATCH_TYPES
            for row in only_rows
        )
        return HistoryParentSelection(
            selected_case_ids=frozenset(groups),
            source=("verified_singleton" if identity_authorized else "candidate_singleton"),
            comparison=comparison,
            identity_authorized=identity_authorized,
        )

    verified_groups = [
        case_id
        for case_id, group_rows in groups.items()
        if any(
            str(row.get("match_type") or "") in _IDENTITY_MATCH_TYPES
            for row in group_rows
        )
    ]
    if verified_groups:
        if scope_conflict(set(verified_groups)):
            return HistoryParentSelection(source="scoped_identity_conflict", comparison=comparison,
                                          ambiguous=True, conflict=True)
        if len(verified_groups) > 1 and not comparison:
            return HistoryParentSelection(
                source="verified_records",
                comparison=False,
                ambiguous=True,
            )
        return HistoryParentSelection(
            selected_case_ids=frozenset(
                verified_groups[: max(1, int(limit))]
            ),
            source="verified_records",
            comparison=comparison,
            ambiguous=(
                len(verified_groups) > max(1, int(limit))
                or (len(verified_groups) > 1 and not comparison)
            ),
            identity_authorized=(
                len(verified_groups) == 1
                or (comparison and len(verified_groups) <= max(1, int(limit)))
            ),
        )

    # Open-language relevance belongs to the retrieval/rerank models.  This
    # identity layer deliberately has no Chinese phrase dictionary, direction
    # parser or substring scorer.  It only carries a bounded set of candidates
    # forward and keeps their non-authoritative status explicit.
    ranked: List[Tuple[float, int, str]] = [
        (_case_score(group_rows), index, case_id)
        for index, (case_id, group_rows) in enumerate(groups.items())
        if _case_score(group_rows) >= 0
    ]
    if not ranked:
        return HistoryParentSelection(
            source="retrieval_candidates", comparison=comparison, ambiguous=True
        )
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    best_score = ranked[0][0]
    floor = max(best_score * 0.82, best_score - 2.5)
    selected = [entry[2] for entry in ranked if entry[0] >= floor][:
        max(1, int(limit))
    ]
    return HistoryParentSelection(
        selected_case_ids=frozenset(selected),
        source="retrieval_candidates",
        comparison=comparison,
        ambiguous=len(selected) != 1,
        identity_authorized=False,
    )
