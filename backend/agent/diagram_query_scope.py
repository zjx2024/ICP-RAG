"""Current-question graph identity, separate from validity and calculation permission."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from backend.agent.current_model_selection import model_candidates, query_fingerprint, selection_matches_model
from backend.agent.diagram_identity import _requested_graph_selectors, _selector_match_score
from backend.agent.model_selection_service import select_current_models

SCOPE_VERSION = "diagram-query-scope/v1"


def _value(item: Mapping[str, Any], key: str) -> str:
    for layer in (item, item.get("metadata"), item.get("document_metadata"), item.get("scope")):
        if isinstance(layer, Mapping) and layer.get(key) not in (None, ""):
            return str(layer[key]).strip().casefold()
    return ""


def diagram_in_scope(item: Mapping[str, Any], binding: Mapping[str, Any], *, query: str) -> bool:
    if binding.get("contract_version") != SCOPE_VERSION or binding.get("query_fingerprint") != query_fingerprint(query):
        return False
    if binding.get("status") != "selected":
        return False
    for chosen in binding.get("models") or []:
        if not isinstance(chosen, Mapping):
            continue
        if not all(_value(item, key) and _value(item, key) == _value(chosen, key) for key in ("doc_id", "revision_id")):
            continue
        if all(not _value(chosen, key) or _value(item, key) == _value(chosen, key)
               for key in ("diagram_id", "diagram_key", "configuration")):
            return True
    return False


async def bind_current_diagrams(state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, doc_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = str(state.get("query") or "")
    binding = {"contract_version": SCOPE_VERSION, "required": True, "doc_type": doc_type,
               "query_fingerprint": query_fingerprint(query), "status": "unavailable", "models": []}
    if not model_candidates(rows, doc_type):
        return [], {**binding, "status": "no_candidates"}
    # An adaptive batch cannot widen the object selected earlier this turn.
    prior = state.get(f"{doc_type}_query_binding")
    if isinstance(prior, Mapping) and prior.get("contract_version") == SCOPE_VERSION and prior.get("query_fingerprint") == query_fingerprint(query):
        return [dict(row) for row in rows if diagram_in_scope(row, prior, query=query)], dict(prior)
    raw_selectors = _requested_graph_selectors({**state, "selected_conversation_anchors": []}, doc_type)
    selectors = raw_selectors or _requested_graph_selectors(state, doc_type)
    candidates = list(rows)
    if selectors:
        resolved = []
        for selector in selectors:
            scored = [(_selector_match_score(row, selector), row) for row in candidates]
            best = max((score for score, row in scored), default=0)
            matches = [row for score, row in scored if score == best and best > 0]
            if len(matches) != 1:
                return [], {**binding, "status": "ambiguous" if len(matches) > 1 else "not_represented"}
            if matches[0] not in resolved:
                resolved.append(matches[0])
        candidates = resolved
        # Prior multi-citations define a reread universe, not a requirement to
        # answer with every model. Current explicit identities take priority.
        if raw_selectors or len(candidates) == 1:
            return [dict(row) for row in candidates], {**binding, "status": "selected",
                "basis": "current_identity" if raw_selectors else "selected_citation",
                "models": model_candidates(candidates, doc_type)}
    selection = await select_current_models(state, candidates, doc_type=doc_type)
    selected = [dict(row) for row in candidates if selection_matches_model(selection, row, query=query)]
    return selected, {**binding, "status": selection.get("status", "unavailable"),
                      "basis": "current_semantic_selection", "models": model_candidates(selected, doc_type),
                      "reason": selection.get("reason", "")}


def merge_diagram_scopes(first: Any, second: Any) -> dict[str, Any]:
    first = dict(first) if isinstance(first, Mapping) and first.get("required") else {}
    second = dict(second) if isinstance(second, Mapping) and second.get("required") else {}
    if not first or not second:
        return first or second
    if first == second:
        return first
    return {**first, "status": "conflict", "models": [], "reason": "adaptive_scope_conflict"}
