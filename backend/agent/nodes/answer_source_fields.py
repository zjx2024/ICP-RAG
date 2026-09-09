"""Facet-scoped source ownership, shared by drafting and completeness proof.

The planner assigns dimensions to sources. This projection enforces that
typed contract, without interpreting user wording or changing stored facts.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from backend.agent.answer_contract import required_answer_fields
from backend.agent.tool_policy import source_request_is_scoped
from backend.agent.nodes.answer_field_catalog import _TARGET_COMPLETENESS_FIELD_KEYS
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.icp.document_types import DOCUMENT_TYPES


def source_owned_field_view(item: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    frame = state.get("query_frame")
    if not isinstance(frame, Mapping):
        return item
    owners: dict[str, set[str]] = {}
    doc_types = {row[0] for row in DOCUMENT_TYPES}
    for obligation in frame.get("source_obligations") or []:
        if not isinstance(obligation, Mapping) or obligation.get("doc_type") not in doc_types:
            continue
        if not source_request_is_scoped(obligation):
            continue
        facets = obligation.get("fields")
        if not isinstance(facets, list):
            continue
        for facet in facets:
            if isinstance(facet, str) and facet in _TARGET_COMPLETENESS_FIELD_KEYS:
                owners.setdefault(facet, set()).add(obligation["doc_type"])
    if not owners:
        return item
    # Shared aliases can serve an independently requested dimension. E.g. a
    # historical root cause remains available when only the *current* cause
    # is assigned to FMECA. There is no whole-turn document-type lock.
    requested = set(required_answer_fields(state)) | set(owners)
    permitted = set().union(*(
        _TARGET_COMPLETENESS_FIELD_KEYS.get(facet, set())
        for facet in requested if facet not in owners or item.get("doc_type") in owners[facet]
    ))
    excluded = set().union(*(
        _TARGET_COMPLETENESS_FIELD_KEYS[facet]
        for facet, sources in owners.items() if item.get("doc_type") not in sources
    )) - permitted
    fields = _evidence_answer_fields(item)
    if not excluded.intersection(fields):
        return item
    visible = {key: value for key, value in fields.items() if key not in excluded}
    # Raw prose could reintroduce the very fields excluded above. Keep the
    # original registry untouched, but give all answer consumers this same
    # field-only view rather than a contradictory full-record summary.
    summary = json.dumps(visible, ensure_ascii=False, default=str)
    return {**item, "_citation_fields_only": True, "content": summary,
            "display_summary": summary, "metadata": {
                **(item.get("metadata") or {}), "answer_fields": visible,
                "source_field_exclusions": sorted(excluded.intersection(fields)),
            }}
