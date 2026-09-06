"""Render a reviewed answer without interpreting or rewriting its prose."""
from __future__ import annotations

import re
from typing import Any, Collection, Mapping, Sequence


class CitationRenderingError(ValueError):
    """An unresolved reference makes the entire publication invalid."""


def render_citations(
    answer: str,
    prompt_citations: Sequence[Mapping[str, Any]],
    *,
    approved_evidence_ids: Collection[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Assign display numbers in occurrence order; all other bytes survive.

    The caller owns semantic review. Rendering may reject an invalid binding,
    never silently remove its marker/claim or infer approval from its presence.
    None is reserved for legacy citations without the ICP evidence gate.
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    for item in prompt_citations:
        number = item.get("id")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0 or number in by_id:
            raise CitationRenderingError("引用登记含无效或重复编号")
        by_id[number] = item
    mapping: dict[int, int] = {}
    citations: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        item = by_id.get(old)
        if item is None:
            raise CitationRenderingError(f"正文引用未登记编号：{old}")
        if approved_evidence_ids is not None and str(item.get("evidence_id") or "") not in approved_evidence_ids:
            raise CitationRenderingError(f"正文引用未经批准的证据：{old}")
        if old not in mapping:
            mapping[old] = len(mapping) + 1
            citations.append({**item, "id": mapping[old]})
        return f"[^{mapping[old]}]"

    return re.sub(r"\[\^(\d+)\]", replace, answer), citations
