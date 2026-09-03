"""Lossless parser records shared by the seven ICP adapters.

The persistence layer still accepts ordinary dictionaries.  ``RecordDraft``
keeps extraction provenance and normalization diagnostics next to that legacy
shape so parsers can become auditable before the PostgreSQL schema is expanded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _clean_part(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def source_file_name(file_path: str, source_name: Optional[str] = None) -> str:
    """Return a stable, basename-only evidence name.

    Parser input paths can be short-lived ArtifactStore materializations such
    as ``/tmp/.../materialized-abc.docx``.  A caller-provided original name is
    therefore preferred, while both POSIX and Windows path separators are
    stripped so source references never expose a host path.
    """

    candidate = str(source_name or "").strip() or str(file_path or "").strip()
    basename = candidate.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if basename:
        return basename
    fallback = str(file_path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return fallback or "document"


@dataclass(frozen=True)
class SourceRef:
    """Stable, human-readable location inside one source artifact.

    DOCX has no reliable page number until rendered, so table/row and ordered
    block positions are used.  Excel locations retain sheet and physical row.
    """

    file_name: str = ""
    element_type: str = ""
    sequence: Optional[int] = None
    section_path: Tuple[str, ...] = ()
    block_id: str = ""
    sheet_name: str = ""
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    image_index: Optional[int] = None

    def child(
        self,
        *,
        element_type: Optional[str] = None,
        row_index: Optional[int] = None,
        column_index: Optional[int] = None,
        image_index: Optional[int] = None,
    ) -> "SourceRef":
        return SourceRef(
            file_name=self.file_name,
            element_type=element_type if element_type is not None else self.element_type,
            sequence=self.sequence,
            section_path=self.section_path,
            block_id=self.block_id,
            sheet_name=self.sheet_name,
            table_index=self.table_index,
            row_index=row_index if row_index is not None else self.row_index,
            column_index=column_index if column_index is not None else self.column_index,
            image_index=image_index if image_index is not None else self.image_index,
        )

    def render(self) -> str:
        parts: List[str] = []
        if self.section_path:
            parts.append("/".join(_clean_part(item) for item in self.section_path if _clean_part(item)))
        if self.sheet_name:
            parts.append(f"sheet={_clean_part(self.sheet_name)}")
        if self.block_id:
            parts.append(f"block={_clean_part(self.block_id)}")
        if self.table_index is not None:
            parts.append(f"table={self.table_index}")
        if self.row_index is not None:
            parts.append(f"row={self.row_index}")
        if self.column_index is not None:
            parts.append(f"col={self.column_index}")
        if self.image_index is not None:
            parts.append(f"image={self.image_index}")
        if self.sequence is not None and not any(
            value is not None for value in (self.table_index, self.image_index)
        ):
            parts.append(f"seq={self.sequence}")
        location = ";".join(parts)
        file_name = _clean_part(self.file_name)
        return f"{file_name}#{location}" if location else file_name

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class NormalizationIssue:
    code: str
    message: str
    field: str = ""
    severity: str = "warning"

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "severity": self.severity,
        }


@dataclass
class RecordDraft:
    """One source record before it is promoted to an authoritative PG row."""

    record_type: str
    values: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    source_refs: List[SourceRef] = field(default_factory=list)
    issues: List[NormalizationIssue] = field(default_factory=list)
    record_key: str = ""

    @property
    def errors(self) -> List[NormalizationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def source_ref(self) -> str:
        return str(self.source_refs[0]) if self.source_refs else ""

    def add_issue(
        self,
        code: str,
        message: str,
        *,
        field: str = "",
        severity: str = "warning",
    ) -> None:
        self.issues.append(
            NormalizationIssue(code=code, message=message, field=field, severity=severity)
        )

    def extend_sources(self, sources: Iterable[SourceRef]) -> None:
        known = {str(item) for item in self.source_refs}
        for source in sources:
            if str(source) not in known:
                self.source_refs.append(source)
                known.add(str(source))

    def to_row(self) -> Dict[str, Any]:
        """Return the legacy dictionary plus forward-compatible audit fields."""

        row = dict(self.values)
        row["raw_data"] = dict(self.raw_data)
        row["source_ref"] = self.source_ref
        row["source_refs"] = [str(source) for source in self.source_refs]
        row["normalization_warnings"] = [issue.as_dict() for issue in self.issues]
        if self.record_key:
            row["record_key"] = self.record_key
        return row


__all__ = ["NormalizationIssue", "RecordDraft", "SourceRef", "source_file_name"]
