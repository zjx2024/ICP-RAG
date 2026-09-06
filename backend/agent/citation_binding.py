"""Deterministic source-field coordinates for model-proposed claim bindings.

The model judges meaning. Code verifies the offered record, field and literal
excerpt; it does not classify claims through an open-language word list.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

# A source view is shared by answer generation and the read-only reviewer.
# Keeping this budget small is a context-engineering boundary: it prevents a
# broad retrieval row from crowding out the current question and neighboring
# records.  Identity fields are still inserted first, and truncation is
# explicit so the model cannot mistake the view for a complete source read.
MAX_QUOTE_FIELDS = 80
MAX_QUOTE_VALUE_CHARS = 1200
MAX_QUOTE_RECORD_CHARS = 3600

def quote_field_view(item: Mapping[str, Any]) -> dict[str, Any]:
    """Build the same bounded public values for prompting and verification."""

    fields: dict[str, str] = {}
    truncated: list[str] = []
    size = 0

    def add(path: str, value: Any, depth: int = 0) -> None:
        nonlocal size
        if value is None or value == "":
            return
        if len(fields) >= MAX_QUOTE_FIELDS or size >= MAX_QUOTE_RECORD_CHARS or depth > 5:
            truncated.append(path)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                add(f"{path}/{escaped}", child, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                add(f"{path}/{index}", child, depth + 1)
            return
        if not isinstance(value, (str, int, float, bool)):
            return
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        limit = min(MAX_QUOTE_VALUE_CHARS, MAX_QUOTE_RECORD_CHARS - size)
        text = raw[:limit]
        if text:
            fields[path] = text
            size += len(text)
        if len(raw) > len(text):
            truncated.append(path)

    # Stable identity precedes descriptions, so a long record cannot hide the
    # business code needed to reject an otherwise similar neighbouring row.
    for key in ("record_id", "title", "doc_id", "doc_name"):
        add(f"/{key}", item.get(key))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    answer_fields = metadata.get("answer_fields")
    if isinstance(answer_fields, Mapping):
        if item.get("claim_type") == "hmc_fault":
            # This evidence identity represents the symptom. Maintenance has
            # its own approved identity and must not be borrowed from hidden data.
            allowed = {"record_code", "hmc_code", "lcn", "hamc", "hsc", "fault_symptom", "failure_mode"}
            answer_fields = {key: value for key, value in answer_fields.items() if key in allowed}
        add("/answer_fields", answer_fields)
    if item.get("authority") == "deterministic_derivation":
        add("/calculation", metadata.get("calculation"))
    add("/scope", item.get("scope"))
    for key in ("system", "configuration", "document_number", "document_version", "data_status", "data_source",
                "revision_id", "diagram_id", "diagram_key", "validation_status", "is_active", "projection_status"):
        add(f"/source/{key}", metadata.get(key))
    if item.get("_citation_fields_only") is not True:
        add("/content", item.get("display_summary") or item.get("content"))
    return {"fields": fields, "truncated_fields": list(dict.fromkeys(truncated))[:MAX_QUOTE_FIELDS]}


def unit_record_requirements(text: str, ledger: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Surface finite, exact record identities before the model chooses sources."""

    groups: dict[str, list[int]] = {}
    for number, item in ledger.items():
        record_id = str(item.get("record_id") or "").strip()
        if not record_id:
            continue
        pattern = r"(?<![A-Za-z0-9_.-])" + re.escape(record_id) + r"(?![A-Za-z0-9_.-])"
        if re.search(pattern, text, re.IGNORECASE):
            groups.setdefault(record_id, []).append(number)
    return [{"record_id": identity, "citation_ids": numbers} for identity, numbers in groups.items()]
