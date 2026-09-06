"""Read-only evidence organization; no query semantics or authority promotion.

Packets group exact scoped PG records, not similarly worded candidates. Claim
views retain their own citation IDs, gates and bounded quote coordinates.
Neither a complete view nor an empty bounded query proves source-wide absence.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.agent.citation_binding import quote_field_view
from backend.agent.record_identity import RecordIdentity


def returned_projection_read(row: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the returned projection, not unread columns or the Word file."""
    def empty(value):
        return value is None or (isinstance(value, str) and not value.strip()) or (
            isinstance(value, (list, tuple, dict)) and not value
        )

    return {
        "status": "returned_projection",
        "record_completeness": "unknown",
        "returned_fields": sorted(str(key) for key in row),
        "empty_fields": sorted(str(key) for key, value in row.items() if empty(value)),
    }


def build_evidence_packets(
    ledger: Mapping[int, Mapping[str, Any]], *, approved_ids: set[str], blocked_ids: set[str],
) -> list[dict[str, Any]]:
    packets: dict[tuple, dict[str, Any]] = {}
    for number, item in ledger.items():
        metadata = item.get("metadata") or {}
        evidence_id = str(item.get("evidence_id") or "")
        source_kind = str(item.get("source_kind") or "")
        identity = RecordIdentity.from_row(str(item.get("doc_type") or ""), {
            **item, "candidate_scope": item.get("scope") or {},
            "revision_id": metadata.get("revision_id"), "variant_id": metadata.get("variant_id"),
        }, fallback=f"citation:{number}")
        # A search projection is not a second authoritative copy of a PG row;
        # missing document/business identity cannot authorize any merge.
        isolated = "" if source_kind == "postgres" and identity.doc_id and identity.record_id else number
        key = (source_kind, identity, isolated)
        packet = packets.setdefault(key, {
            "record_identity": identity.public(), "source_kind": source_kind,
            "cross_record_relation": "not_asserted", "claims": [],
        })
        view = quote_field_view(item)
        excluded = list(metadata.get("source_field_exclusions") or [])
        read = metadata.get("source_read")
        # Only the producer can report what it actually read. Legacy evidence
        # stays unknown, even if authority=authoritative_record.
        source_read = dict(read) if isinstance(read, Mapping) else {
            "status": "unknown", "record_completeness": "unknown",
        }
        packet["claims"].append({
            "citation_id": number, "evidence_id": evidence_id,
            "citation_label": f"[^{number}]",
            "claim_type": item.get("claim_type", ""),
            "title": item.get("title", ""),
            "authority": item.get("authority", ""),
            "evidence_level": item.get("evidence_level", ""),
            "review_status": item.get("review_status", ""),
            "usage": "blocked" if evidence_id in blocked_ids else (
                "approved" if evidence_id in approved_ids else "context_only"
            ),
            "source_ref": item.get("source_ref") or item.get("record_id") or item.get("chunk_id"),
            "doc_name": item.get("doc_name") or item.get("doc_id"),
            "limitations": list(item.get("limitations") or []),
            "source_read": source_read,
            "view_state": {
                "status": "truncated" if view["truncated_fields"] else "complete_selected_view",
                "source_restricted": bool(excluded or item.get("_citation_fields_only")),
                "excluded_answer_fields": excluded,
            },
            # This is exactly what the quote verifier sees, never raw hidden
            # fields or the unfiltered registry copied alongside the view.
            "quote_view": view,
        })
    return list(packets.values())


def query_coverage(
    results: Sequence[Mapping[str, Any]], *, requested_tools: Sequence[str] = (),
) -> list[dict[str, Any]]:
    coverage = []
    executed = set()
    for result in results:
        tool = str(result.get("tool") or "")
        if not tool:
            continue
        executed.add(tool)
        params = result.get("params") or {}
        data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
        diagnostics = result.get("diagnostics") or {}
        status = str(result.get("status") or "unknown")
        # Counts and limits refer only to this invocation. Less than limit
        # does not prove an exhaustive scan (filters/hydration may drop rows).
        coverage.append({
            "tool": tool, "execution_status": status,
            "backend": diagnostics.get("backend", ""),
            "returned_count": data.get("count", data.get("postgres_count")),
            "requested_limit": params.get("limit", params.get("top_k")),
            "query_scope": dict(params),
            "matching_set_completeness": "not_established",
            "anchor_reads": list(data.get("anchor_reads") or []),
            "errors_present": bool(diagnostics.get("errors")),
            "degraded": status == "partial" or bool(diagnostics.get("fallbacks")),
        })
    coverage.extend({
        "tool": tool, "execution_status": "not_queried", "matching_set_completeness": "not_established",
    } for tool in dict.fromkeys(requested_tools) if tool and tool not in executed)
    return coverage
