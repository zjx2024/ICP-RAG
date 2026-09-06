"""Fact-unit and section syntax shared by citation and completeness checks.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.nodes.answer_text import _HISTORY_BINDING_CONTEXT
from backend.agent.nodes.answer_text import _SECTION_HEADING
from backend.agent.nodes.answer_text import _STRUCTURAL_HEADING_TITLES
from backend.agent.nodes.answer_text import _section_title
from typing import Any
from typing import Dict
from typing import List
import re


def _line_is_factual_assertion(
    line: str,
    exempt_lines: set[str] | None = None,
) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"\|?(?:\s*:?-{3,}:?\s*\|)+\s*", stripped):
        return False
    is_table_row = stripped.startswith("|") and stripped.endswith("|")
    semantic_text = re.sub(r"^#{1,6}\s*", "", stripped).strip()
    semantic_text = re.sub(r"^(?:[-+*]|\d+[.)])\s*", "", semantic_text).strip()
    semantic_text = re.sub(r"\[\^\d+\]", "", semantic_text).strip()
    heading_text = re.sub(r"^(?:\*\*|__)|(?:\*\*|__)$", "", semantic_text).strip()
    if heading_text.casefold() in _STRUCTURAL_HEADING_TITLES:
        return False
    if semantic_text in (exempt_lines or set()):
        return False
    if is_table_row:
        return True
    # Grounded by default: every substantive generated line needs evidence.
    # Exact structural, clarification and safety text is exempted above.  A
    # keyword allow-list inevitably misses harmless paraphrases such as
    # “等效串联电阻会增大”, which would otherwise bypass citation review.
    return bool(re.search(r"[0-9A-Za-z\u4e00-\u9fff]", semantic_text))


def _line_has_uncited_assertion(
    line: str,
    approved_numbers: set[int],
    exempt_lines: set[str] | None = None,
) -> bool:
    if not _line_is_factual_assertion(line, exempt_lines):
        return False
    cited_numbers = {int(value) for value in re.findall(r"\[\^(\d+)\]", line)}
    return not bool(cited_numbers & approved_numbers)


def _cited_fact_units(
    answer: str,
    citation_id: int,
) -> List[Dict[str, Any]]:
    """Return citation-bound Markdown units with their structural context.

    Completeness checks used to inspect only the physical line containing a
    citation.  That misses natural prose whose entity or history context lives
    in a parent list item or section heading, causing a later editor to repeat
    an already stated field under a new title.  These lightweight logical units
    mirror the context inheritance used by citation repair without inferring
    any new domain relationship.
    """

    marker = f"[^{int(citation_id)}]"
    units: List[Dict[str, Any]] = []
    heading = ""
    active_history = False
    parent_labels: Dict[int, str] = {}
    for raw_line in str(answer or "").splitlines():
        if _SECTION_HEADING.match(raw_line):
            heading = _section_title(raw_line)
            active_history = _history_section_context(heading, active_history)
            parent_labels.clear()
            continue
        indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        content = raw_line.lstrip()
        indent_width = len(indent.expandtabs(4))
        bold_label_match = re.match(
            r"^(?:[-+*]|\d+[.)])\s+\*\*([^*]{2,80})\*\*\s*[:：]?",
            content,
        )
        if bold_label_match:
            parent_labels = {
                key: value for key, value in parent_labels.items() if key < indent_width
            }
            parent_labels[indent_width] = bold_label_match.group(1).strip().rstrip("：:")
        if marker not in raw_line:
            continue
        inherited = [
            value
            for key, value in sorted(parent_labels.items())
            if key < indent_width
        ]
        context = " / ".join([value for value in (heading, *inherited) if value])
        fact_text = re.sub(r"\[\^\d+\]", "", raw_line)
        fact_text = re.sub(r"[*_`#]", "", fact_text).strip()
        history_context = bool(
            active_history
            or _unit_requires_fault_history(raw_line)
            or re.search(r"(?:该|此|上述|本次)(?:历史)?(?:案例|记录)", raw_line)
        )
        units.append(
            {
                "text": fact_text,
                "context": f"{context}：{fact_text}" if context else fact_text,
                "history_context": history_context,
            }
        )
    return units


_HISTORICAL_ASSERTION = re.compile(
    # “故障历史记录”是 ICP 功能/模块名（如 FUNC-530），不是历史事件断言；
    # 负向后顾避免该实体名误触发 fault_history 归属要求，否则功能查询的
    # 正确答案会被整行删除并退化成墙句。
    r"(?:相似|既往|(?<!故障)历史).{0,12}(?:案例|记录|经验|处理)|"
    r"(?:曾经|曾|此前|以往|当时).{0,60}"
    r"(?:发生|出现|记录|处理|处置|修复|解决|通过|检查|排查|操作|测量|更换|验证)|"
    r"发生时间|\bHIS-[A-Za-z0-9._-]+",
    flags=re.IGNORECASE,
)


_HISTORICAL_DETAIL_LABEL = re.compile(
    r"(?:^|[\n；;])\s*(?:(?:[-+*]|\d+[.)])\s*)?(?:\*\*)?"
    r"(?:当时现象|当时处理|确认根因|当时根因|关键操作|最终方案|验证结果)"
    r"(?:\*\*)?\s*[:：]",
    flags=re.IGNORECASE,
)


_HISTORY_SECTION_TITLE = re.compile(
    # 同 _HISTORICAL_ASSERTION：“故障历史记录”实体名不应激活历史区块上下文。
    r"(?:相似|既往|(?<!故障)历史).{0,10}(?:案例|记录|经验|处理|处置)?|案例回顾",
    flags=re.IGNORECASE,
)


_HISTORY_DETAIL_SECTION_TITLES = {
    "当时现象",
    "根因",
    "确认根因",
    "关键操作",
    "最终方案",
    "验证结果",
}


def _unit_requires_fault_history(unit_text: str) -> bool:
    """Historical-event wording requires an actual fault-history record."""

    text = str(unit_text or "")
    return bool(
        _HISTORY_BINDING_CONTEXT in text
        or _HISTORICAL_ASSERTION.search(text)
        or _HISTORICAL_DETAIL_LABEL.search(text)
    )


def _history_section_context(title: str, active: bool = False) -> bool:
    normalized = re.sub(r"\s+", "", str(title or "")).strip("：:")
    if _HISTORY_SECTION_TITLE.search(normalized):
        return True
    return active and normalized in _HISTORY_DETAIL_SECTION_TITLES


_ACTION_CLAIM_RE = re.compile(
    r"检查|排查|处置|维修|修复|更换|校准|锁止|压接|回滚|切换|限制|"
    r"调用|读取|测量|监测|导出|隔离|恢复|重启|测试|验证",
)
