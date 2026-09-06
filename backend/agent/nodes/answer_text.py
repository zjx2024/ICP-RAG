"""Markdown and quote-text primitives; no evidence selection or model calls.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from typing import Any
from typing import List
from typing import Sequence
import re
import unicodedata


def split_top_level_clauses(text: str) -> list[str]:
    """Split prose punctuation without breaking syntax or detaching citations.

    Parenthesized expressions, quoted values, inline code and Markdown table
    rows are atomic here. Their semantics belong to the evidence reviewer,
    not a language-specific list of conjunctions in a string splitter.
    """
    if text.lstrip().startswith("|"):
        return [text]
    pairs = {"(": ")", "（": "）", "[": "]", "【": "】", "{": "}", "“": "”", "‘": "’", '"': '"'}
    stack: list[str] = []
    code_delimiter = ""
    parts: list[str] = []
    start = index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "`":
            end = index + 1
            while end < len(text) and text[end] == "`":
                end += 1
            delimiter = text[index:end]
            if not code_delimiter:
                code_delimiter = delimiter
            elif code_delimiter == delimiter:
                code_delimiter = ""
            index = end
            continue
        if code_delimiter:
            index += 1
            continue
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        if char in "。！？；" and not stack:
            end = index + 1
            # A footnote belongs to the clause that precedes it, including
            # whitespace and multiple markers; never split inside [^123].
            while match := re.match(r"\s*\[\^\d+\]", text[end:]):
                end += match.end()
            parts.append(text[start:end].strip())
            start = index = end
        else:
            index += 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts or [text]


def strip_generated_footnote_definitions(answer: str) -> str:
    """Only the server's citation ledger may define evidence footnotes.

    Drop Markdown definitions, including indented continuation paragraphs,
    before marker stripping can turn their invented contents into fact units.
    Inline reference markers and ordinary colon-prefixed prose are untouched.
    """
    lines = str(answer or "").splitlines()
    result: list[str] = []
    in_definition = False
    for line in lines:
        if re.match(r"^ {0,3}\[\^[^\]\n]+\]:", line):
            in_definition = True
            continue
        if in_definition and (not line.strip() or line.startswith(("    ", "\t"))):
            continue
        in_definition = False
        result.append(line)
    return "\n".join(result)


def _normalize_generated_markdown(answer: str) -> str:
    """Normalize common Markdown escaping emitted by local chat models.

    The diagnosis renderer intentionally supports a small Markdown subset. A
    model sometimes emits JSON-style escapes such as ``\\###``, ``1\\.`` or a
    backslash at every line ending; those are displayed literally by the UI.
    Only structural Markdown escapes are removed so HMC values remain untouched.
    """
    result = str(answer or "").replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"\\+[ \t]*\n", "\n", result)
    result = re.sub(r"(?m)^([ \t]*)\\+(?=#{1,6}(?:\s|$))", r"\1", result)
    result = re.sub(r"(?m)^([ \t]*)\\+(?=[*+-]\s)", r"\1", result)
    result = re.sub(r"(?m)^([ \t]*\d+)\\+\.(?=\s)", r"\1.", result)
    result = re.sub(r"\\+([*#_-])", r"\1", result)
    # 空全角括号是模型套用了“（字段）”模板但该字段为空的残留，没有信息量；
    # 只清全角形态，避免碰到“GetXXX()”这类合法半角 API 写法。
    result = re.sub(r"（\s*）", "", result)
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


_SECTION_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+\S.*|\*\*[^*\n]+\*\*\s*[:：]?\s*)$"
)


def _section_title(line: str) -> str:
    value = re.sub(r"^\s*#{1,6}\s*", "", str(line or "")).strip()
    value = re.sub(r"^\*\*|\*\*$", "", value).strip()
    return value.rstrip(":：").strip()


_STRUCTURAL_HEADING_TITLES = {
    "直接回答",
    "初步判断",
    "候选故障",
    "原因分析",
    "原因与影响",
    "建议检查",
    "建议检查与处理",
    "建议处理",
    "处理建议",
    "处置建议",
    "怎么处理",
    "相似历史",
    "历史处理",
    "历史案例",
    "下一步需要确认",
    "适用性说明",
    "系统架构与功能",
    "层级与组成",
    "功能与接口",
    "处理子系统",
    "后续处理子系统",
    "来源与适用范围",
    "资料边界",
    "资料缺口",
    "hmc 查询结果",
    "诊断结论",
    "候选故障模式的区分与处理",
}


_NUMBERED_BOLD_SECTION = re.compile(
    r"^(?P<indent>\s*)(?P<mark>\*\*|__)\s*\d+\s*[.、)]\s*"
    r"(?P<title>.+?)(?P=mark)(?P<citations>(?:\s*\[\^\d+\])*)\s*$"
)


def _cleanup_numbered_bold_sections(answer: str) -> str:
    """Drop empty generated sections and remove brittle section numbering.

    Citation repair can deliberately remove an unsupported paragraph.  If a
    generated ``**2. ...**`` heading is left immediately before another
    heading (or at the end), showing it as an empty section is misleading.
    Surviving headings lose their model-generated number so later removals do
    not leave a visible ``1, 3`` sequence.
    """
    lines = (answer or "").splitlines()
    remove_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if not _NUMBERED_BOLD_SECTION.match(line):
            continue
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if (
            next_index >= len(lines)
            or _NUMBERED_BOLD_SECTION.match(lines[next_index])
            or lines[next_index].lstrip().startswith("#")
        ):
            remove_indexes.add(index)

    cleaned: List[str] = []
    for index, line in enumerate(lines):
        if index in remove_indexes:
            continue
        match = _NUMBERED_BOLD_SECTION.match(line)
        if match:
            line = (
                f"{match.group('indent')}{match.group('mark')}"
                f"{match.group('title').strip()}{match.group('mark')}"
                f"{match.group('citations')}"
            )
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


_ATX_SECTION = re.compile(r"^(?P<indent>\s{0,3})(?P<marks>#{1,6})\s+\S")


def _cleanup_incomplete_markdown_tables(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            cleaned.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines):
            current = lines[end].strip()
            if not (current.startswith("|") and current.endswith("|")):
                break
            end += 1
        block = lines[index:end]
        separator_indexes = [
            offset
            for offset, value in enumerate(block)
            if re.fullmatch(r"\|?(?:\s*:?-{3,}:?\s*\|)+\s*", value.strip())
        ]
        if separator_indexes:
            separator_index = separator_indexes[0]
            has_header = separator_index > 0
            has_data = any(value.strip() for value in block[separator_index + 1:])
            if has_header and has_data:
                cleaned.extend(block)
        index = end
    return cleaned


def _cleanup_orphan_atx_sections(lines: List[str]) -> List[str]:
    remove_indexes: set[int] = set()
    for index, line in enumerate(lines):
        match = _ATX_SECTION.match(line)
        if not match:
            continue
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index >= len(lines):
            remove_indexes.add(index)
            continue
        next_heading = _ATX_SECTION.match(lines[next_index])
        if next_heading and len(next_heading.group("marks")) <= len(match.group("marks")):
            remove_indexes.add(index)
    return [line for index, line in enumerate(lines) if index not in remove_indexes]


def _renumber_top_level_numeric_headings(lines: List[str]) -> List[str]:
    """Close numbering gaps left when an unsupported section is removed."""

    pattern = re.compile(
        r"^(?P<marks>#{1,6})\s+(?P<number>\d{1,2})(?:[.、])?\s+(?P<title>\S.*)$"
    )
    matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := pattern.match(line.strip()))
    ]
    if len(matches) < 2:
        return lines
    top_level = min(len(match.group("marks")) for _, match in matches)
    result = list(lines)
    ordinal = 0
    for index, match in matches:
        if len(match.group("marks")) != top_level:
            continue
        ordinal += 1
        result[index] = f"{match.group('marks')} {ordinal}. {match.group('title')}"
    return result


def _cleanup_generated_markdown_structure(answer: str) -> str:
    result = _cleanup_numbered_bold_sections(answer)
    lines = _cleanup_incomplete_markdown_tables(result.splitlines())
    lines = _cleanup_orphan_atx_sections(lines)
    lines = _renumber_top_level_numeric_headings(lines)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _insert_into_matching_markdown_section(
    answer: str,
    *,
    heading_pattern: re.Pattern[str],
    default_heading: str,
    body_lines: Sequence[str],
) -> str:
    """Insert a completion into an existing semantic section when available."""

    clean_body = [str(line) for line in body_lines if str(line).strip()]
    if not clean_body:
        return str(answer or "")
    lines = str(answer or "").splitlines()
    match_index = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if _SECTION_HEADING.match(lines[index])
            and heading_pattern.search(_section_title(lines[index]))
        ),
        None,
    )
    if match_index is None:
        block = [default_heading, *clean_body]
        return _normalize_generated_markdown(
            f"{str(answer or '').rstrip()}\n\n" + "\n".join(block)
        )

    heading_match = re.match(r"^\s*(#{1,6})\s+", lines[match_index])
    level = len(heading_match.group(1)) if heading_match else 3
    insert_at = len(lines)
    for index in range(match_index + 1, len(lines)):
        next_match = re.match(r"^\s*(#{1,6})\s+", lines[index])
        if next_match and len(next_match.group(1)) <= level:
            insert_at = index
            break
    insertion = [""] if insert_at > 0 and lines[insert_at - 1].strip() else []
    insertion.extend(clean_body)
    return _normalize_generated_markdown(
        "\n".join([*lines[:insert_at], *insertion, *lines[insert_at:]])
    )


_HISTORY_BINDING_CONTEXT = "【历史案例上下文】"


_FTA_CUT_SET_BINDING_CONTEXT = "【FTA最小割集回答】"


_INTERNAL_BINDING_CONTEXTS = (
    _HISTORY_BINDING_CONTEXT,
    _FTA_CUT_SET_BINDING_CONTEXT,
)


def _normalized_quote_text(value: Any) -> str:
    text = str(value or "")
    for marker in _INTERNAL_BINDING_CONTEXTS:
        text = text.replace(marker, "")
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _compact_semantic_text(value: Any) -> str:
    return re.sub(
        r"[^0-9a-z\u4e00-\u9fff]+",
        "",
        _normalized_quote_text(value),
    )
