"""Schema-guided table discovery for Word and Excel engineering documents.

The engine deliberately separates layout recognition from domain
normalization.  It searches beyond a fixed cover window, supports repeated and
multi-row headers, honours alias priority, preserves unknown columns, joins
continuation rows, and attaches a physical source location to every record.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.core.parsers.document_ir import CellIR, TableIR, clean_text, load_document_ir
from backend.core.parsers.records import NormalizationIssue, RecordDraft, SourceRef


def normalize_header(value: object) -> str:
    # NFKC makes full-width ASCII/digits and compatibility punctuation match
    # the same controlled alias table without weakening domain matching.
    text = unicodedata.normalize("NFKC", clean_text(value)).upper()
    return re.sub(r"[\s_\-—–/\\:：;；,，.。()（）\[\]【】]+", "", text)


@dataclass(frozen=True)
class ColumnRule:
    name: str
    aliases: Tuple[str, ...]


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Tuple[ColumnRule, ...]
    required_fields: Tuple[str, ...] = ()
    identity_groups: Tuple[Tuple[str, ...], ...] = ()
    min_mapped_fields: int = 2
    max_header_depth: int = 6
    continuation_fields: Tuple[str, ...] = ()
    merged_cell_inherit_fields: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        name: str,
        columns: Mapping[str, Sequence[str]],
        *,
        required_fields: Sequence[str] = (),
        identity_groups: Sequence[Sequence[str]] = (),
        min_mapped_fields: int = 2,
        max_header_depth: int = 6,
        continuation_fields: Sequence[str] = (),
        merged_cell_inherit_fields: Sequence[str] = (),
    ) -> "TableSchema":
        return cls(
            name=name,
            columns=tuple(
                ColumnRule(field_name, tuple(str(alias) for alias in aliases))
                for field_name, aliases in columns.items()
            ),
            required_fields=tuple(required_fields),
            identity_groups=tuple(tuple(group) for group in identity_groups),
            min_mapped_fields=min_mapped_fields,
            max_header_depth=max_header_depth,
            continuation_fields=tuple(continuation_fields),
            merged_cell_inherit_fields=tuple(merged_cell_inherit_fields),
        )


@dataclass
class TableSource:
    rows: List[List[str]]
    source: SourceRef
    row_numbers: List[int]
    cells: Optional[List[List[CellIR]]] = None
    hidden_rows: Tuple[int, ...] = ()
    hidden_columns: Tuple[int, ...] = ()

    @classmethod
    def from_ir(cls, table: TableIR) -> "TableSource":
        rows = table.matrix()
        row_numbers = [
            int(row[0].source.row_index or index)
            if row
            else index
            for index, row in enumerate(table.rows, start=1)
        ]
        return cls(
            rows=rows,
            source=table.source,
            row_numbers=row_numbers,
            cells=table.rows,
            hidden_rows=table.hidden_rows,
            hidden_columns=table.hidden_columns,
        )


@dataclass(frozen=True)
class HeaderCandidate:
    start: int
    end: int
    mapping: Dict[str, int]
    labels: Tuple[str, ...]
    score: int
    ambiguous_fields: Tuple[str, ...] = ()


@dataclass
class TableRecord:
    values: Dict[str, str]
    raw_data: Dict[str, Any]
    source_refs: List[SourceRef]
    header: HeaderCandidate
    issues: List[NormalizationIssue] = field(default_factory=list)

    def to_draft(self, record_type: str, record_key: str = "") -> RecordDraft:
        return RecordDraft(
            record_type=record_type,
            values=dict(self.values),
            raw_data=dict(self.raw_data),
            source_refs=list(self.source_refs),
            issues=list(self.issues),
            record_key=record_key,
        )


def _dedupe_parts(values: Iterable[object]) -> List[str]:
    parts: List[str] = []
    for value in values:
        text = clean_text(value)
        if text and (not parts or normalize_header(text) != normalize_header(parts[-1])):
            parts.append(text)
    return parts


def _alias_score(text: str, alias: str, alias_index: int, *, leaf: bool) -> int:
    normalized_text = normalize_header(text)
    normalized_alias = normalize_header(alias)
    if not normalized_text or not normalized_alias:
        return 0
    priority = max(0, 200 - alias_index)
    if normalized_text == normalized_alias:
        return 5000 + priority + (300 if leaf else 0)
    # Containment is useful for headers such as “故障描述及现象”, but short
    # aliases create many false positives in long data cells.
    contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", normalized_alias))
    if contains_cjk and len(normalized_alias) >= 2 and normalized_alias in normalized_text:
        return 1200 + priority + len(normalized_alias) * 5 + (100 if leaf else 0)
    return 0


def _match_column(parts: Sequence[str], schema: TableSchema) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    combined = " ".join(parts)
    for rule in schema.columns:
        best = 0
        for alias_index, alias in enumerate(rule.aliases):
            for part_index, part in enumerate(reversed(parts)):
                score = _alias_score(part, alias, alias_index, leaf=part_index == 0)
                best = max(best, score)
            best = max(best, _alias_score(combined, alias, alias_index, leaf=False))
        if best:
            candidates.append((best, rule.name))
    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def _candidate_for_segment(
    rows: Sequence[Sequence[object]], start: int, depth: int, schema: TableSchema
) -> HeaderCandidate:
    segment = rows[start : start + depth]
    width = max((len(row) for row in segment), default=0)
    labels: List[str] = []
    choices: Dict[str, List[Tuple[int, int]]] = {}
    for column_index in range(width):
        parts = _dedupe_parts(
            row[column_index] if column_index < len(row) else "" for row in segment
        )
        label = " / ".join(parts)
        labels.append(label)
        matches = _match_column(parts, schema)
        if matches:
            score, field_name = matches[0]
            choices.setdefault(field_name, []).append((score, column_index))

    mapping: Dict[str, int] = {}
    ambiguous: List[str] = []
    score = 0
    for field_name, field_choices in choices.items():
        ordered = sorted(field_choices, key=lambda item: (-item[0], item[1]))
        score += ordered[0][0]
        mapping[field_name] = ordered[0][1]
        if len(ordered) > 1:
            ambiguous.append(field_name)
    score += len(mapping) * 100
    return HeaderCandidate(
        start=start,
        end=start + depth - 1,
        mapping=mapping,
        labels=tuple(labels),
        score=score,
        ambiguous_fields=tuple(sorted(ambiguous)),
    )


def _valid_header(candidate: HeaderCandidate, schema: TableSchema) -> bool:
    fields = set(candidate.mapping)
    if len(fields) < schema.min_mapped_fields:
        return False
    if not set(schema.required_fields).issubset(fields):
        return False
    if schema.identity_groups and not any(set(group).issubset(fields) for group in schema.identity_groups):
        return False
    return True


def find_headers(rows: Sequence[Sequence[object]], schema: TableSchema) -> List[HeaderCandidate]:
    """Return non-overlapping header bands, including repeated page headers."""

    candidates: List[HeaderCandidate] = []
    for start in range(len(rows)):
        max_depth = min(schema.max_header_depth, len(rows) - start)
        for depth in range(1, max_depth + 1):
            candidate = _candidate_for_segment(rows, start, depth, schema)
            if _valid_header(candidate, schema):
                candidates.append(candidate)
    if not candidates:
        return []

    # Overlap clustering is unsafe here: a six-row candidate can bridge a data
    # row and the repeated header on the next printed page, collapsing both
    # regions into one.  Prefer dense/short bands first, then greedily retain
    # disjoint regions.  A genuine two-level header still wins when neither
    # individual row satisfies the schema on its own.
    candidates.sort(
        key=lambda item: (
            -(len(item.mapping) / (item.end - item.start + 1)),
            -len(item.mapping),
            len(item.ambiguous_fields),
            item.end - item.start,
            -item.score,
            item.start,
        )
    )
    selected: List[HeaderCandidate] = []
    for candidate in candidates:
        if any(
            candidate.start <= existing.end and existing.start <= candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


def _raw_row(row: Sequence[str], labels: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for index, value in enumerate(row):
        label = clean_text(labels[index] if index < len(labels) else "") or f"column_{index + 1}"
        key = label
        duplicate = 2
        while key in result:
            key = f"{label}__{duplicate}"
            duplicate += 1
        result[key] = clean_text(value)
    return result


def _json_safe_raw(value: Any) -> Any:
    """Preserve a cell's original scalar without breaking JSONB persistence."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_raw(item())
        except Exception:
            pass
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _cell_metadata(cell: CellIR) -> Dict[str, Any]:
    return {
        "raw_value": _json_safe_raw(cell.raw_value),
        "raw_type": type(cell.raw_value).__name__,
        "display_text": cell.text,
        "formula": cell.formula,
        "source_ref": str(cell.source),
        "anchor_row": cell.anchor_row,
        "anchor_column": cell.anchor_column,
        "is_merge_continuation": cell.is_merge_continuation,
    }


def _raw_cells(
    cells: Optional[Sequence[CellIR]], labels: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if cells is None:
        return {}, {}
    raw_values: Dict[str, Any] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for index, cell in enumerate(cells):
        label = clean_text(labels[index] if index < len(labels) else "") or f"column_{index + 1}"
        key = label
        duplicate = 2
        while key in raw_values:
            key = f"{label}__{duplicate}"
            duplicate += 1
        raw_values[key] = _json_safe_raw(cell.raw_value)
        metadata[key] = _cell_metadata(cell)
    return raw_values, metadata


def _has_identity(values: Mapping[str, str], schema: TableSchema) -> bool:
    if not schema.identity_groups:
        return any(values.values())
    return any(all(clean_text(values.get(field)) for field in group) for group in schema.identity_groups)


def _merge_continuation(target: TableRecord, incoming: TableRecord, schema: TableSchema) -> None:
    fields = schema.continuation_fields or tuple(incoming.values)
    for field_name in fields:
        value = clean_text(incoming.values.get(field_name))
        if not value:
            continue
        previous = clean_text(target.values.get(field_name))
        if not previous:
            target.values[field_name] = value
        elif value != previous:
            target.values[field_name] = f"{previous}\n{value}"
    continuation_rows = target.raw_data.setdefault("_continuation_rows", [])
    continuation_rows.append(dict(incoming.raw_data))
    target.source_refs.extend(incoming.source_refs)
    target.issues.append(
        NormalizationIssue(
            code="continuation_row_joined",
            message="无业务标识的续行已与上一条记录合并",
        )
    )


def extract_records(source: TableSource, schema: TableSchema) -> List[TableRecord]:
    headers = find_headers(source.rows, schema)
    records: List[TableRecord] = []
    if not headers:
        return records

    for header_index, header in enumerate(headers):
        stop = headers[header_index + 1].start if header_index + 1 < len(headers) else len(source.rows)
        pending: Optional[TableRecord] = None
        for physical_index in range(header.end + 1, stop):
            row = [clean_text(value) for value in source.rows[physical_index]]
            if not any(row):
                continue
            # Defensive skip for a header variant that did not win its overlap cluster.
            one_row = _candidate_for_segment(source.rows, physical_index, 1, schema)
            if _valid_header(one_row, schema):
                continue
            source_cells = (
                source.cells[physical_index]
                if source.cells is not None and physical_index < len(source.cells)
                else None
            )
            values: Dict[str, str] = {}
            merge_issues: List[NormalizationIssue] = []
            field_raw_values: Dict[str, Any] = {}
            field_cell_metadata: Dict[str, Dict[str, Any]] = {}
            for field_name, column_index in header.mapping.items():
                value = row[column_index] if column_index < len(row) else ""
                cell = (
                    source_cells[column_index]
                    if source_cells is not None and column_index < len(source_cells)
                    else None
                )
                if cell is not None and cell.is_merge_continuation and value:
                    if field_name in schema.merged_cell_inherit_fields:
                        merge_issues.append(
                            NormalizationIssue(
                                code="merged_cell_inherited",
                                message=(
                                    f"字段 {field_name} 从合并单元格锚点 "
                                    f"R{cell.anchor_row}C{cell.anchor_column} 受控继承"
                                ),
                                field=field_name,
                            )
                        )
                    else:
                        # A visual merge does not automatically authorize copying
                        # a business identifier or fact into another physical row.
                        # The row may still be joined as a configured continuation.
                        value = ""
                        merge_issues.append(
                            NormalizationIssue(
                                code="merged_cell_not_inherited",
                                message=(
                                    f"字段 {field_name} 位于合并单元格续格，未在继承白名单中；"
                                    "保留原始锚点证据但不生成重复事实"
                                ),
                                field=field_name,
                            )
                        )
                values[field_name] = value
                if cell is not None:
                    field_raw_values[field_name] = _json_safe_raw(cell.raw_value)
                    field_cell_metadata[field_name] = _cell_metadata(cell)
            row_number = (
                source.row_numbers[physical_index]
                if physical_index < len(source.row_numbers)
                else physical_index + 1
            )
            source_ref = source.source.child(element_type="row", row_index=row_number)
            raw_values, cell_metadata = _raw_cells(source_cells, header.labels)
            record = TableRecord(
                values=values,
                raw_data={
                    "columns": _raw_row(row, header.labels),
                    "raw_values": raw_values,
                    "cell_metadata": cell_metadata,
                    "field_raw_values": field_raw_values,
                    "field_cell_metadata": field_cell_metadata,
                    "header_labels": list(header.labels),
                    "physical_row": row_number,
                },
                source_refs=[source_ref],
                header=header,
                issues=merge_issues,
            )
            if header.ambiguous_fields:
                record.issues.append(
                    NormalizationIssue(
                        code="ambiguous_header_columns",
                        message=f"表头存在重复候选列: {', '.join(header.ambiguous_fields)}",
                    )
                )

            if _has_identity(values, schema):
                if pending is not None:
                    records.append(pending)
                pending = record
            elif pending is not None and any(values.values()):
                _merge_continuation(pending, record, schema)
            elif any(values.get(field) for field in schema.required_fields):
                record.issues.append(
                    NormalizationIssue(
                        code="orphan_row",
                        message="数据行缺少业务标识，无法与前一条记录可靠合并",
                        severity="error",
                    )
                )
                records.append(record)
        if pending is not None:
            records.append(pending)
    return records


def table_sources(
    file_path: str, *, source_name: Optional[str] = None
) -> List[TableSource]:
    document = load_document_ir(file_path, source_name=source_name)
    return [TableSource.from_ir(table) for table in document.tables]


def extract_table_drafts(
    file_path: str,
    schema: TableSchema,
    *,
    record_type: Optional[str] = None,
    source_name: Optional[str] = None,
) -> List[RecordDraft]:
    drafts: List[RecordDraft] = []
    for source in table_sources(file_path, source_name=source_name):
        for record in extract_records(source, schema):
            drafts.append(record.to_draft(record_type or schema.name))
    return drafts


__all__ = [
    "ColumnRule",
    "HeaderCandidate",
    "TableRecord",
    "TableSchema",
    "TableSource",
    "extract_records",
    "extract_table_drafts",
    "find_headers",
    "normalize_header",
    "table_sources",
]
