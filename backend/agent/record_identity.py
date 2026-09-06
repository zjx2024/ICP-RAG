"""Scoped business identity shared by ranking and evidence organization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


@dataclass(frozen=True)
class RecordIdentity:
    doc_type: str
    doc_id: str
    record_id: str
    scope: tuple[tuple[str, str], ...] = ()
    revision_id: str = ""
    variant_id: str = ""
    locator: str = ""

    @classmethod
    def from_row(cls, doc_type: str, row: Mapping[str, Any], *, fallback: str):
        scope = row.get("candidate_scope") or {}
        record_id = _text(row.get("record_id"))
        return cls(doc_type, _text(row.get("doc_id")), record_id,
                   tuple(sorted((key, _text(value)) for key, value in scope.items() if _text(value))),
                   _text(row.get("revision_id")), _text(row.get("variant_id")),
                   "" if record_id else _text(row.get("chunk_id") or row.get("source_ref")) or fallback)

    def public(self) -> dict[str, Any]:
        return {**asdict(self), "scope": dict(self.scope)}

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(self.public(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "candidate-" + hashlib.sha256(payload.encode()).hexdigest()[:24]
