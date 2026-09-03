"""Build the user-facing, source-ordered view of an ICP document.

This module is intentionally read-only.  It reconstructs display content from
the source artifact without invoking models or writing extraction results.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from backend.core.parsers.document_ir import (
    ImageIR,
    ParagraphIR,
    TableIR,
    load_document_ir,
)


def _image_blocks(
    images: Iterable[ImageIR],
    *,
    doc_type: str,
    doc_id: str,
    sequence: int,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for image in images:
        image_index = int(image.source.image_index or 0)
        if image_index <= 0:
            continue
        blocks.append(
            {
                "id": f"img-{image_index}",
                "kind": "image",
                "sequence": sequence,
                "title": f"图片 {image_index}",
                "level": 0,
                "text": "",
                "image_index": image_index,
                "image_name": image.original_name,
                "content_type": image.content_type,
                "source_ref": str(image.source),
                "image_url": (
                    f"/api/v1/document-types/{doc_type}/documents/{doc_id}"
                    f"/embedded-images/{image_index}"
                ),
            }
        )
    return blocks


def build_original_content(
    file_path: str,
    *,
    doc_type: str,
    doc_id: str,
    file_name: str,
    page: int = 1,
    size: int = 100,
) -> Dict[str, Any]:
    """Return source-ordered paragraphs, tables and embedded pictures.

    Pagination is applied only after the complete ordered block list has been
    built, so totals and image positions remain stable on every page.  Table
    matrices are not truncated; the UI owns scrolling and presentation.
    """

    if page < 1 or size < 1:
        raise ValueError("page 和 size 必须为正整数")

    document = load_document_ir(file_path, source_name=file_name)
    blocks: List[Dict[str, Any]] = []
    for element in document.elements:
        sequence = int(element.source.sequence or len(blocks) + 1)
        if isinstance(element, ParagraphIR):
            raw_text = str(element.raw_text or "").strip()
            if raw_text:
                is_heading = element.heading_level > 0
                blocks.append(
                    {
                        "id": element.source.block_id or f"p-{sequence}",
                        "kind": "paragraph",
                        "sequence": sequence,
                        "title": element.text if is_heading else "",
                        "level": element.heading_level if is_heading else 0,
                        "text": "" if is_heading else raw_text,
                        "source_ref": str(element.source),
                    }
                )
            blocks.extend(
                _image_blocks(
                    element.images,
                    doc_type=doc_type,
                    doc_id=doc_id,
                    sequence=sequence,
                )
            )
            continue

        if isinstance(element, TableIR):
            table_index = int(element.source.table_index or 0)
            sheet_name = str(element.source.sheet_name or "").strip()
            blocks.append(
                {
                    "id": element.source.block_id or f"tbl-{table_index or sequence}",
                    "kind": "table",
                    "sequence": sequence,
                    "title": f"工作表：{sheet_name}" if sheet_name else f"表格 {table_index or sequence}",
                    "level": 0,
                    "text": "",
                    "table_index": table_index or None,
                    "row_count": len(element.rows),
                    "column_count": element.width,
                    "source_ref": str(element.source),
                    "table": element.matrix(),
                }
            )
            blocks.extend(
                _image_blocks(
                    element.images,
                    doc_type=doc_type,
                    doc_id=doc_id,
                    sequence=sequence,
                )
            )

    total = len(blocks)
    start = (page - 1) * size
    page_blocks = blocks[start : start + size]
    return {
        "doc_type": doc_type,
        "file_name": file_name,
        "total": total,
        "page": page,
        "size": size,
        "counts": {
            "paragraphs": sum(block.get("kind") == "paragraph" for block in blocks),
            "tables": sum(block.get("kind") == "table" for block in blocks),
            "images": sum(block.get("kind") == "image" for block in blocks),
        },
        "blocks": page_blocks,
    }


_BLOCK_REF_RE = re.compile(r"(?:^|;)block=([^;]+)")
_ROW_REF_RE = re.compile(r"(?:^|;)row=(\d+)")


def _source_section(value: Any) -> str:
    """Return the human section part of a source locator.

    Materialized parser names and original upload names can differ, therefore
    matching is intentionally based on the part after ``#`` and before the
    block/row coordinates rather than on the filename.
    """

    text = str(value or "").strip()
    if "#" in text:
        text = text.split("#", 1)[1]
    return text.split(";block=", 1)[0].strip()


def _compact_source_text(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"


def _table_row_for_evidence(
    table: List[List[Any]],
    *,
    title: str,
    source_ref: str,
) -> Tuple[List[str], List[str]]:
    if not table:
        return [], []
    rows = [[_compact_source_text(cell, 240) for cell in row] for row in table]
    headers = rows[0]
    if len(rows) == 1:
        return headers, []

    row_match = _ROW_REF_RE.search(source_ref)
    if row_match:
        # Source locators use the original document's one-based row number.
        index = max(1, int(row_match.group(1)) - 1)
        if index < len(rows):
            return headers, rows[index]

    normalized_title = "".join(str(title or "").split()).lower()
    if normalized_title:
        for row in rows[1:]:
            values = ["".join(cell.split()).lower() for cell in row]
            if any(
                normalized_title == value
                or (len(normalized_title) >= 4 and normalized_title in value)
                for value in values
            ):
                return headers, row
    return headers, rows[1]


def build_evidence_preview(
    blocks: Iterable[Mapping[str, Any]],
    *,
    source_ref: str = "",
    title: str = "",
    structure_summary: str = "",
) -> Dict[str, Any]:
    """Select a small, source-faithful preview for one evidence card.

    The result deliberately contains at most one table row and one nearby
    picture.  It is meant for quick verification in the diagnosis sidebar,
    not as another full-document viewer.
    """

    items = [dict(block) for block in blocks]
    block_match = _BLOCK_REF_RE.search(str(source_ref or ""))
    block_id = block_match.group(1) if block_match else ""
    requested_section = _source_section(source_ref)

    matched: Optional[Dict[str, Any]] = next(
        (item for item in items if block_id and str(item.get("id") or "") == block_id),
        None,
    )
    if matched is None and source_ref:
        matched = next(
            (
                item
                for item in items
                if str(item.get("source_ref") or "").strip() == str(source_ref).strip()
            ),
            None,
        )

    normalized_title = "".join(str(title or "").split()).lower()
    if matched is None and normalized_title:
        for item in items:
            searchable: List[str] = [
                str(item.get("title") or ""),
                str(item.get("text") or ""),
            ]
            searchable.extend(
                str(cell or "")
                for row in (item.get("table") or [])
                for cell in (row or [])
            )
            if any(normalized_title in "".join(value.split()).lower() for value in searchable):
                matched = item
                break

    matched_sequence = int((matched or {}).get("sequence") or 0)
    matched_section = _source_section((matched or {}).get("source_ref")) or requested_section

    table_block: Optional[Dict[str, Any]] = None
    if matched and (matched.get("kind") == "table" or matched.get("table")):
        table_block = matched
    else:
        table_candidates = [item for item in items if item.get("table")]
        same_section = [
            item for item in table_candidates
            if matched_section and _source_section(item.get("source_ref")) == matched_section
        ]
        candidates = same_section or table_candidates
        if candidates and matched_sequence:
            table_block = min(
                candidates,
                key=lambda item: abs(int(item.get("sequence") or 0) - matched_sequence),
            )

    table_preview: Optional[Dict[str, Any]] = None
    if table_block:
        headers, row = _table_row_for_evidence(
            list(table_block.get("table") or []),
            title=title,
            source_ref=source_ref,
        )
        if headers or row:
            table_preview = {
                "title": str(table_block.get("title") or ""),
                "headers": headers,
                "row": row,
                "source_ref": str(table_block.get("source_ref") or ""),
            }

    anchor_sequence = int((table_block or matched or {}).get("sequence") or 0)
    anchor_section = _source_section((table_block or matched or {}).get("source_ref")) or matched_section
    images = [item for item in items if item.get("kind") == "image" and item.get("image_url")]
    same_section_images = [
        item for item in images
        if anchor_section and _source_section(item.get("source_ref")) == anchor_section
    ]
    image_candidates = same_section_images
    if not image_candidates and anchor_sequence:
        image_candidates = [
            item
            for item in images
            if abs(int(item.get("sequence") or 0) - anchor_sequence) <= 2
        ]
    image_block = (
        min(
            image_candidates,
            key=lambda item: abs(int(item.get("sequence") or 0) - anchor_sequence),
        )
        if image_candidates
        else None
    )
    image_preview = None
    if image_block:
        image_preview = {
            "title": str(image_block.get("title") or ""),
            "image_url": str(image_block.get("image_url") or ""),
            "source_ref": str(image_block.get("source_ref") or ""),
        }

    paragraph_candidates = []
    for item in items:
        if item.get("kind") != "paragraph":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if anchor_section and _source_section(item.get("source_ref")) != anchor_section:
            continue
        distance = abs(int(item.get("sequence") or 0) - anchor_sequence) if anchor_sequence else 0
        if anchor_sequence and distance > 4:
            continue
        paragraph_candidates.append((distance, int(item.get("sequence") or 0), text))
    paragraph_candidates.sort(key=lambda item: (item[0], item[1]))
    excerpt = _compact_source_text(paragraph_candidates[0][2]) if paragraph_candidates else ""

    resolved_source_ref = str(
        (table_block or matched or {}).get("source_ref") or source_ref or ""
    ).strip()
    summary = _compact_source_text(structure_summary, 1200)
    return {
        "found": bool(excerpt or table_preview or image_preview or summary),
        "resolved_source_ref": resolved_source_ref,
        "structure_summary": summary,
        "excerpt": excerpt,
        "table": table_preview,
        "image": image_preview,
    }


def read_original_docx_image(
    file_path: str,
    image_index: int,
    *,
    source_name: str = "",
) -> Tuple[bytes, str, str]:
    """Read one embedded DOCX image using the same index as DocumentIR."""

    if Path(file_path).suffix.lower() != ".docx":
        raise ValueError("只有 DOCX 文档支持读取内嵌图片")
    if image_index <= 0:
        raise IndexError("图片序号必须为正整数")

    document_ir = load_document_ir(file_path, source_name=source_name)
    image = next(
        (
            candidate
            for candidate in document_ir.images
            if candidate.source.image_index == image_index
        ),
        None,
    )
    if image is None:
        raise IndexError("图片不存在")

    from docx import Document

    document = Document(file_path)
    related = getattr(document.part, "related_parts", {}).get(image.relation_id)
    blob = bytes(getattr(related, "blob", b"") or b"") if related is not None else b""
    if not blob:
        raise IndexError("图片内容不存在")
    content_type = str(getattr(related, "content_type", "") or image.content_type or "application/octet-stream")
    return blob, content_type, image.original_name


__all__ = [
    "build_evidence_preview",
    "build_original_content",
    "read_original_docx_image",
]
