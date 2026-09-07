"""Deterministic backstop rendering at the publication boundary.

发布边界的确定性保底：本轮字段契约或计算合同要求的确定性内容，若模型
稿与审核幸存文本均未呈现，按已批准记录逐行转写补全。内容只来自已批准
证据的字段值与确定性计算摘要；引用编号沿用本轮 citation ledger，最终仍
由 render_citations 统一验证来源合法性。这不是第二次生成，与 salvage 的
确定性剪除同属发布边界的纯渲染操作：字段无幻觉、引用有绑定、数值有
确定性来源，因此不再经过 LLM 审核。
"""
from __future__ import annotations

from backend.agent.answer_contract import MFL_CONTRACT_FIELDS
from backend.agent.answer_contract import required_answer_fields
from backend.agent.nodes.answer_field_catalog import _TARGET_COMPLETENESS_FIELD_KEYS
from backend.agent.nodes.answer_payload import _flatten_target_values
from backend.agent.nodes.answer_target_presence import _target_value_is_present
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import json
import re


# MFL 行按上限渲染，防止异常宽的检索窗口把保底变成清单倾倒。
_MFL_BACKSTOP_MAX_LINES = 12
_API_PLACEHOLDER_VALUES = {"", "—", "-", "－", "无", "N/A", "n/a", "none", "None"}
# “不可直接上报”是 MFL 列（API是否可直接上报）的领域规范表述，也是本
# 模块自身的渲染短语；检查它不是在猜测用户措辞，而是在核对系统输出协议。
_NOT_REPORTABLE_MARKERS = ("不可直接上报", "无法直接上报", "不能直接上报")
_FIELD_VALUE_LIMIT = 240


def _record_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    fields = (
        metadata.get("answer_fields")
        if isinstance(metadata.get("answer_fields"), Mapping)
        else {}
    )
    return fields


def _first_value(fields: Mapping[str, Any], key: str) -> str:
    for value in _flatten_target_values(fields.get(key)):
        text = value.strip()
        if text and text not in _API_PLACEHOLDER_VALUES:
            return text[:_FIELD_VALUE_LIMIT]
    return ""


def _cited_lines_text(answer: str, citation_id: int) -> str:
    marker = f"[^{citation_id}]"
    return "\n".join(
        line for line in str(answer or "").splitlines() if marker in line
    )


def _is_reportable(fields: Mapping[str, Any]) -> bool | None:
    value = fields.get("api_reportable")
    if isinstance(value, bool):
        return value
    raw = str(fields.get("api_reportable_raw") or "").strip().casefold()
    if raw in {"是", "支持", "√", "y", "yes", "true"}:
        return True
    if raw in {"否", "n", "no", "false"}:
        return False
    return None


def _api_name(fields: Mapping[str, Any]) -> str:
    return _first_value(fields, "api_name")


def _api_attribute_present(cited_text: str, fields: Mapping[str, Any]) -> bool:
    """Whether the answer already expresses this row's API reporting status."""

    reportable = _is_reportable(fields)
    if reportable is None:
        return True
    if reportable:
        name = _api_name(fields)
        if not name:
            return True
        return _target_value_is_present(cited_text, name)
    return any(marker in str(cited_text or "") for marker in _NOT_REPORTABLE_MARKERS)


def _api_phrase(fields: Mapping[str, Any], *, detailed: bool) -> str:
    reportable = _is_reportable(fields)
    if reportable is None:
        return ""
    if not reportable:
        return "API可直接上报：否，该行没有可用的状态接口"
    name = _api_name(fields)
    if not detailed:
        return f"该行可通过API直接上报（{name}）" if name else "该行可通过API直接上报"
    parts = ["API可直接上报：是"]
    if name:
        parts.append(f"API名称：{name}")
    for key, label in (("api_method", "调用方法"), ("api_usage", "使用方法")):
        value = _first_value(fields, key)
        if value:
            parts.append(f"{label}：{value}")
    return "；".join(parts)


def _facet_phrase(facet: str, fields: Mapping[str, Any]) -> str:
    """Render one contract facet's values for a single approved MFL row."""

    if facet == "maintenance_action":
        for key in ("maintenance_action", "repair_method", "compensation_action"):
            value = _first_value(fields, key)
            if value:
                return f"资料建议处置：{value}"
        return ""
    if facet == "maintenance_locator":
        parts = []
        for key, label in (
            ("sur_failure", "SUR板故障模式"),
            ("subcircuit_failure", "功能子电路故障模式"),
        ):
            value = _first_value(fields, key)
            if value:
                parts.append(f"{label}：{value}")
        return "；".join(parts)
    if facet == "platform_test_status":
        value = _first_value(fields, "platform_test_status")
        return f"平台级实验状态：{value}" if value else ""
    if facet == "platform_test_result":
        value = _first_value(fields, "platform_test_raw") or _first_value(
            fields, "platform_test"
        )
        return f"平台级实验结果：{value}" if value else ""
    if facet == "maintenance_api":
        return _api_phrase(fields, detailed=True)
    # 未知 facet 不猜测渲染；调用方只传入 MFL 合同字段。
    return ""


def _facet_present(cited_text: str, facet: str, fields: Mapping[str, Any]) -> bool:
    if not str(cited_text or "").strip():
        return False
    if facet == "maintenance_api":
        return _api_attribute_present(cited_text, fields)
    keys = _TARGET_COMPLETENESS_FIELD_KEYS.get(facet) or ()
    values: List[str] = []
    for key in keys:
        for value in _flatten_target_values(fields.get(key)):
            text = value.strip()
            if text and text not in _API_PLACEHOLDER_VALUES:
                values.append(text)
    if not values:
        return True
    return any(_target_value_is_present(cited_text, value) for value in values)


def _record_label(fields: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    module_failure = _first_value(fields, "module_failure")
    record_code = _first_value(fields, "record_code") or str(
        item.get("record_id") or ""
    ).strip()
    if module_failure and record_code:
        return f"{module_failure}（{record_code}）"
    return module_failure or record_code


def _mfl_backstop_lines(
    answer: str,
    ledger: Mapping[int, Mapping[str, Any]],
    contract_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    mfl_fields = [
        field
        for field in (
            "maintenance_locator",
            "maintenance_action",
            "maintenance_api",
            "platform_test_status",
            "platform_test_result",
        )
        if field in MFL_CONTRACT_FIELDS and field in set(contract_fields)
    ]
    if not mfl_fields:
        return []
    lines: List[Dict[str, Any]] = []
    detailed_api = "maintenance_api" in mfl_fields
    for citation_id, item in ledger.items():
        if str(item.get("doc_type") or "") != "mfl":
            continue
        if len(lines) >= _MFL_BACKSTOP_MAX_LINES:
            break
        fields = _record_fields(item)
        label = _record_label(fields, item)
        if not label:
            continue
        cited_text = _cited_lines_text(answer, int(citation_id))
        parts: List[str] = []
        for facet in mfl_fields:
            if facet == "maintenance_api":
                continue
            phrase = _facet_phrase(facet, fields)
            if phrase and not _facet_present(cited_text, facet, fields):
                parts.append(phrase)
        api_phrase = _api_phrase(fields, detailed=detailed_api)
        if api_phrase and not _api_attribute_present(cited_text, fields):
            parts.append(api_phrase)
        if not parts:
            continue
        lines.append(
            {
                "citation_id": int(citation_id),
                "record_id": str(item.get("record_id") or ""),
                "text": f"{label}[^{citation_id}]：" + "；".join(parts) + "。",
            }
        )
    return lines


def _calculation_result_content(item: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        content = json.loads(item.get("content") or "{}")
    except Exception:
        content = {}
    return content if isinstance(content, dict) else {}


def _calculation_present(answer: str, item: Mapping[str, Any]) -> bool:
    """Whether the answer already carries this calculation's exact results."""

    answer_text = str(answer or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    calc = metadata.get("calculation") if isinstance(metadata.get("calculation"), Mapping) else {}
    results = calc.get("results") if isinstance(calc.get("results"), Mapping) else {}
    compact_digits = re.sub(r"\D", "", answer_text)
    for key in ("top_event_probability", "system_reliability"):
        value = results.get(key)
        if value is None:
            continue
        # 完整有效数字串必须逐位出现；四舍五入或百分比近似不算已呈现，
        # 保底行随后会补上权威数值。
        digits = re.sub(r"\D", "", str(value))
        if len(digits) >= 6 and digits not in compact_digits:
            return False
    if str(item.get("claim_type") or "") == "fta_calculation":
        content = _calculation_result_content(item)
        for cut in content.get("minimal_cut_sets") or []:
            members = cut if isinstance(cut, (list, tuple, set, frozenset)) else [cut]
            if len(members) < 2:
                continue
            if not all(
                _target_value_is_present(answer_text, str(member))
                for member in members
            ):
                return False
    return True


def _calculation_backstop_lines(
    answer: str,
    ledger: Mapping[int, Mapping[str, Any]],
    calculation_requests: Sequence[Any],
) -> List[Dict[str, Any]]:
    if not calculation_requests:
        return []
    lines: List[Dict[str, Any]] = []
    for citation_id, item in ledger.items():
        claim = str(item.get("claim_type") or "")
        if claim not in ("fta_calculation", "rbd_calculation"):
            continue
        summary = str(item.get("display_summary") or "").strip()
        if not summary or _calculation_present(answer, item):
            continue
        lines.append(
            {
                "citation_id": int(citation_id),
                "record_id": str(item.get("record_id") or ""),
                "claim_type": claim,
                "text": f"{summary}[^{citation_id}]",
            }
        )
    return lines


def answer_backstop(
    answer: str,
    state: Mapping[str, Any],
    ledger: Mapping[int, Mapping[str, Any]],
    prompt_citations: Sequence[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Append deterministic backstop lines for unconsumed approved facts.

    ``prompt_citations`` bounds the valid citation numbers: the backstop never
    introduces a marker that the public citation renderer cannot resolve.
    """

    valid_ids = {
        int(item.get("id"))
        for item in prompt_citations or []
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    bounded_ledger = {
        int(number): item
        for number, item in ledger.items()
        if int(number) in valid_ids
    }
    contract_fields = required_answer_fields(state)
    calculation_lines = _calculation_backstop_lines(
        answer, bounded_ledger, state.get("calculation_requests") or []
    )
    mfl_lines = _mfl_backstop_lines(answer, bounded_ledger, contract_fields)
    report: Dict[str, Any] = {}
    if not calculation_lines and not mfl_lines:
        return answer, report
    blocks: List[str] = []
    if calculation_lines:
        blocks.append(
            "补充确定性计算结果（转写自本轮获准的计算证据）：\n"
            + "\n".join(f"- {line['text']}" for line in calculation_lines)
        )
        report["calculation"] = [
            {"citation_id": line["citation_id"], "record_id": line["record_id"]}
            for line in calculation_lines
        ]
    if mfl_lines:
        blocks.append(
            "维修清单逐行补充（含API上报属性）：\n"
            + "\n".join(f"- {line['text']}" for line in mfl_lines)
        )
        report["mfl"] = [
            {"citation_id": line["citation_id"], "record_id": line["record_id"]}
            for line in mfl_lines
        ]
    return str(answer or "").rstrip() + "\n\n" + "\n\n".join(blocks), report
