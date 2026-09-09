"""Deterministic source-field coordinates for model-proposed claim bindings.

The model judges meaning. Code verifies the offered record, field and literal
excerpt; it does not classify claims through an open-language word list.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Mapping


EXACT_IDENTITY_PATTERN = (
    r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]*(?:[-_][A-Za-z0-9*]+)+(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_.])\d{6,}[A-Za-z0-9*._-]*(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+(?![A-Za-z0-9_])"
)


def exact_identities(visible_or_literal_text: str) -> set[str]:
    """Closed code syntax; no natural-language topic or question classifier."""
    return set(re.findall(EXACT_IDENTITY_PATTERN, visible_or_literal_text))


def _html_text(value: str) -> str:
    from html.parser import HTMLParser

    class Text(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    parser = Text()
    parser.feed(value)
    return "".join(parser.parts)


def _markdown_parser():
    from markdown_it import MarkdownIt

    # CommonMark block semantics, with the table/strike extensions used by
    # the frontend. Parse the document before inspecting individual units.
    return MarkdownIt("commonmark").enable(["table", "strikethrough"])


def rendered_identity_text(value: str) -> str:
    """Read displayed identifiers; authoritative fields remain literal data."""
    return _html_text(_markdown_parser().render(str(value))).rstrip("\n")


def markdown_review_units(draft: str) -> list[dict[str, Any]]:
    """Keep source coordinates while reading identities in document context.

    An indented child list is not an indented code block. Rendering each line
    separately loses that distinction. Full-document tokens also preserve
    multiline emphasis and literal code. If inline normalization collapses
    source lines (e.g. a multiline code span), review that complete block as
    one unit rather than assigning its identity to a guessed line.
    """
    lines = list(re.finditer(r"[^\r\n]*(?:\r\n|[\r\n]|$)", draft))
    lines = [line for line in lines if line.start() != line.end()]
    visible: dict[int, list[str]] = {}
    merged: dict[int, int] = {}
    md = _markdown_parser()
    for token in md.parse(draft):
        if token.map is None:
            continue
        start, end = token.map
        if token.type == "inline":
            text = _html_text(md.renderer.renderInline(token.children or [], md.options, {}))
        elif token.type in {"code_block", "fence"}:
            text = token.content.removesuffix("\n")
            if token.type == "fence":
                start += 1
            end = start + len(text.split("\n"))
        elif token.type == "html_block":
            text = _html_text(token.content).removesuffix("\n")
        else:
            continue
        parts = text.split("\n")
        if len(parts) == end - start:
            for offset, part in enumerate(parts):
                visible.setdefault(start + offset, []).append(part)
        else:
            merged[start] = max(merged.get(start, start), end)
            visible.setdefault(start, []).append(text)
    units = []
    index = 0
    while index < len(lines):
        end = merged.get(index, index + 1)
        for inner in range(index + 1, end):
            end = max(end, merged.get(inner, end))
        first, last = lines[index], lines[end - 1]
        stop = last.end() - len(last.group()) + len(last.group().rstrip("\r\n"))
        text = draft[first.start():stop]
        if text.strip():
            units.append({"unit_id": f"u{len(units) + 1}", "text": text,
                          "start": first.start(), "end": stop,
                          "identity_text": "\n".join(" ".join(visible.get(i, []))
                                                       for i in range(index, end)).rstrip("\n")})
        index = end
    return units


def quote_field_view(item: Mapping[str, Any]) -> dict[str, Any]:
    """Build the same bounded public values for prompting and verification."""

    fields: dict[str, str] = {}
    truncated: list[str] = []

    def add(path: str, value: Any, depth: int = 0) -> None:
        if value is None or value == "":
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
        if raw:
            fields[path] = raw

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
    return {"fields": fields, "truncated_fields": list(dict.fromkeys(truncated))}


def unit_record_requirements(text: str, ledger: Mapping[int, Mapping[str, Any]], *,
                             visible_text: str | None = None) -> list[dict[str, Any]]:
    """Surface finite, exact record identities before the model chooses sources."""

    text = rendered_identity_text(text) if visible_text is None else visible_text
    groups: dict[str, list[int]] = {}
    for number, item in ledger.items():
        record_id = str(item.get("record_id") or "").strip()
        if not record_id:
            continue
        pattern = r"(?<![A-Za-z0-9_.-])" + re.escape(record_id) + r"(?![A-Za-z0-9_.-])"
        if re.search(pattern, text, re.IGNORECASE):
            groups.setdefault(record_id, []).append(number)
    return [{"record_id": identity, "citation_ids": numbers} for identity, numbers in groups.items()]


class CitationBindingError(ValueError):
    """An author-owned citation cannot support its immutable text span."""

    def __init__(self, message, *, span=None, unsupported_identities=()):
        self.span = dict(span or {})
        self.unsupported_identities = sorted(unsupported_identities)
        super().__init__(message)

def citation_spans(draft: str) -> list[dict[str, Any]]:
    """Bind each inline citation group to preceding text, not a physical row.

    The author can compare models within one paragraph. Adjacent citation
    markers form a single group; they do not prove pairwise value ownership.
    Uncited tails remain visible to semantic review, never silently dropped.
    """
    spans = []
    for unit in markdown_review_units(draft):
        cursor = 0
        visible_cursor = 0
        pattern = r"(?:\[\^\d+\][ \t]*)+"
        matches = list(re.finditer(pattern, unit["text"]))
        displayed = list(re.finditer(pattern, unit["identity_text"]))
        if [re.findall(r"\d+", m[0]) for m in matches] != [re.findall(r"\d+", m[0]) for m in displayed]:
            raise CitationBindingError("citation coordinates do not match rendered document")
        for match, visible_match in zip(matches, displayed):
            stop = match.end()
            spans.append({"start": unit["start"] + cursor, "end": unit["start"] + stop,
                          "text": unit["text"][cursor:stop],
                          "identity_text": unit["identity_text"][visible_cursor:visible_match.start()],
                          "citation_ids": list(dict.fromkeys(int(n) for n in re.findall(r"\[\^(\d+)\]", match[0])))})
            cursor = stop
            visible_cursor = visible_match.end()
        if unit["text"][cursor:].strip():
            spans.append({"start": unit["start"] + cursor, "end": unit["end"],
                          "text": unit["text"][cursor:],
                          "identity_text": unit["identity_text"][visible_cursor:], "citation_ids": []})
    return spans


_NUMERIC_LITERAL = r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9_.])"

def _numeric_values(text: str) -> set[Decimal]:
    text = re.sub(EXACT_IDENTITY_PATTERN, "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    return {Decimal(value) for value in re.findall(_NUMERIC_LITERAL, text)}


def _calculation_values(item: Mapping, *, parameters_only=False, cut_sets_only=False) -> set[Decimal]:
    from backend.agent.evidence_display import _number

    fields = quote_field_view(item)["fields"]
    # Top-level results and input parameters are distinct from intermediate
    # node calculations. A neighbour's block value cannot support a system result.
    paths = {"/calculation/results/system_reliability", "/calculation/results/top_event_probability",
             "/calculation/reliability", "/calculation/probability",
             "/calculation/results/minimal_cut_set_count"}
    if cut_sets_only:
        paths = {"/calculation/results/minimal_cut_set_count"}
    values = set().union(*(_numeric_values(value) for path, value in fields.items()
                          if path.startswith("/calculation/parameters/")
                          or (not parameters_only and path in paths)))
    return values | {Decimal(_number(value)) for value in values if _number(value)}


def inspect_record_citations(draft: str, ledger: Mapping[int, Mapping], *,
                             calculation_requests=()):
    """Collect independent binding errors without repairing or changing text.

    No question vocabulary or critic-selected field list is involved. Source
    qualification and formal-claim permission are separate semantic boundaries.
    """
    spans = citation_spans(draft)
    errors = []
    for span in spans:
        def reject(message, *, unsupported_identities=()):
            errors.append(CitationBindingError(message, span=span,
                                               unsupported_identities=unsupported_identities))

        ids = span["citation_ids"]
        if any(number not in ledger for number in ids):
            reject("unknown or unapproved citation")
            continue  # No authority or value check may dereference an unknown source.
        if not ids:
            continue
        items = [ledger[number] for number in ids]
        visible = span["identity_text"]
        source = "\n".join(value for item in items for value in quote_field_view(item)["fields"].values())
        unsupported = exact_identities(visible) - exact_identities(source)
        if unsupported:
            reject("cited span contains identity not supported by its records", unsupported_identities=unsupported)
        derived = [item for item in items if item.get("authority") == "deterministic_derivation"]
        for item in derived:
            required = {"rbd": {"rbd_reliability"}, "fta": {"fta_probability", "fta_cut_sets"}}.get(item.get("doc_type"), set())
            if not required.intersection(calculation_requests):
                reject("calculation not authorized")
            meta = item.get("metadata") or {}
            if ({item.get("review_status"), meta.get("validation_status")} & {"provisional", "quarantined"}
                    or meta.get("usage_scope") == "model_content_only"):
                reject("calculation evidence not formally validated")
        numbers = _numeric_values(visible)
        if not derived or not numbers:
            continue
        cut_sets_only = all(item.get("doc_type") == "fta" for item in derived) and "fta_probability" not in calculation_requests
        if not cut_sets_only and any((item.get("metadata") or {}).get("numeric_complete") is False for item in derived):
            reject("incomplete numeric calculation")
        if cut_sets_only:
            # Cut-set indices/IDs are not probability values. Numeric results
            # cannot borrow cut-set-only execution authority.
            if any(not numbers <= _calculation_values(item, cut_sets_only=True) for item in derived):
                reject("numeric result not authorized")
        identities = {(item.get("doc_id"), (item.get("metadata") or {}).get("revision_id"),
                       item.get("record_id")) for item in derived}
        if len(identities) > 1:
            if all(numbers <= _calculation_values(item, parameters_only=True) for item in derived):
                continue
            reject("ambiguous numeric ownership: cite each result at its own text span")
            continue
        if not numbers <= _calculation_values(derived[0]):
            reject("numeric value not supported by this calculation")
    return spans, errors


def validate_record_citations(draft: str, ledger: Mapping[int, Mapping], *,
                              calculation_requests=()) -> list[dict[str, Any]]:
    """Compatibility entry; identical checks, fail on any collected error."""
    spans, errors = inspect_record_citations(draft, ledger, calculation_requests=calculation_requests)
    if errors:
        raise errors[0]
    return spans
