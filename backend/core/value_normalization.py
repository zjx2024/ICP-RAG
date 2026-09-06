"""Small, storage-independent value normalizers shared by ICP adapters."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_ABSENT_IDENTIFIER_MARKERS = frozenset(
    {
        "na",
        "none",
        "null",
        "不适用",
        "无",
        "未定义",
        "未提供",
    }
)
_SEPARATOR_ONLY_RE = re.compile(r"^[\s\-‐‑‒–—―_/\\.·]+$")


def normalize_optional_identifier(value: Any) -> str:
    """Normalize an optional identifier while retaining meaningful spelling.

    Table authors commonly use a dash, ``N/A`` or an equivalent explicit null
    marker in identifier columns.  Those cells are absence, not business IDs;
    callers retain the original cell separately in their raw audit payload.
    """

    if value is None:
        return ""
    text = " ".join(
        unicodedata.normalize("NFKC", str(value))
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )
    if not text or _SEPARATOR_ONLY_RE.fullmatch(text):
        return ""
    compact = re.sub(r"[\s_/\\.\-]+", "", text).casefold()
    if compact in _ABSENT_IDENTIFIER_MARKERS:
        return ""
    return text


__all__ = ["normalize_optional_identifier"]
