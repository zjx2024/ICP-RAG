"""Answer 节点·计算(FTA/RBD)域策略模块（C1 批 6 从 answer.py 抽出）。

包含：FTA 结构限定问答、RBD/FTA 计算清单、缺失参数识别、
确定性计算输出核验与格式规整等策略函数。

依赖约定与其他域模块相同：共享 helper 留在 answer.py 内核。
"""

import ast
import json
import math
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.agent.answer_contract import required_answer_fields
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer_domain_fault import (
    _numbered_fault_candidate_blocks,
    _query_requests_parallel_failure_modes,
)
from backend.agent.nodes.answer import (
    _MULTI_CANDIDATE_FRAMING,
    _NEXT_STEP_HEADING,
    _SECTION_HEADING,
    _answer_query_text,
    _citation_ledger,
    _compact_semantic_text,
    _evidence_answer_fields,
    _is_applicability_notice_line,
    _is_fta_structure_only_request,
    _normalize_generated_markdown,
    _target_value_is_present,
)

def _strip_diagnostic_followup_for_calculation(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Remove generic diagnostic questions from a closed calculation answer."""

    if not (state.get("calculation_requests") or ""):
        return str(answer or "")
    lines = str(answer or "").splitlines()
    output: List[str] = []
    index = 0
    while index < len(lines):
        # Some models omit the Markdown heading while retaining the canned
        # diagnosis paragraph.  It is not a calculation limitation and should
        # not follow a deterministic result into the user-visible answer.
        if (
            "当前信息不足以唯一定位故障" in lines[index]
            or re.match(r"^\s*(?:请补充|需要补充).{0,12}(?:故障|告警|HMC|阶段)", lines[index])
        ):
            index += 1
            continue
        if not _NEXT_STEP_HEADING.match(lines[index]):
            output.append(lines[index])
            index += 1
            continue
        index += 1
        while index < len(lines):
            line = lines[index]
            if _SECTION_HEADING.match(line) or _is_applicability_notice_line(line):
                break
            index += 1
    return _normalize_generated_markdown("\n".join(output))


def _ensure_broad_fta_structure_context(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Keep an approved FTA topology visible in broad path explanations.

    FMECA rows explain individual modes, but a user asking where a data or
    function chain may fail also needs the topology that groups those modes.
    Once the LLM has produced at least two FMECA-backed branches, retain the
    approved FTA modelling note as a short structural lead-in. The sentence is
    copied from the validated source instead of inferring new edges or a unique
    diagnosis.
    """

    current = str(answer or "").strip()
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        return current
    if not current or _query_requests_parallel_failure_modes(state):
        return current

    # FTA may be queried internally while diagnosing an ordinary symptom.  A
    # topology note is useful in the answer only when the user asked for the
    # tree/causal structure (or the planner explicitly required that field).
    # Otherwise a generic FTA modelling sentence is noise and can make an
    # unrelated candidate look as if it were established by that tree.
    required_fields = set(required_answer_fields(dict(state)))
    query = _answer_query_text(dict(state))
    explicit_structure_wording = bool(re.search(
        r"FTA|故障树|因果链|原因链|最小割集|逻辑门", query, flags=re.IGNORECASE
    ))
    if "structural_cause_chain" not in required_fields and not explicit_structure_wording:
        # A broad multi-hop question can still have an FTA that is genuinely
        # about the reported path.  Permit that narrow case by requiring two
        # non-generic Chinese bigrams shared by the turn and the validated FTA
        # title/summary.  This keeps the useful implicit-topology behaviour
        # while rejecting an unrelated FTA note merely because the planner
        # happened to call ``fta_query``.
        generic_ngrams = {"系统", "设备", "数据", "故障", "异常", "有效"}

        def meaningful_ngrams(value: str) -> set[str]:
            compact = re.sub(r"\s+", "", str(value or ""))
            # Normalize the common conversational variants used in broad
            # path reports before comparing them with an FTA title.
            compact = (
                compact.replace("外面的", "外部")
                .replace("外面", "外部")
                .replace("有效内容", "有效业务数据")
            )
            return {
                compact[index : index + 2]
                for index in range(max(0, len(compact) - 1))
                if re.match(r"^[\u4e00-\u9fff]{2}$", compact[index : index + 2])
            } - generic_ngrams

        query_ngrams = meaningful_ngrams(query)
        ledger_for_match = _citation_ledger(dict(state))
        related_fta = False
        for item in ledger_for_match.values():
            if (
                str(item.get("doc_type") or "") != "fta"
                or str(item.get("claim_type") or "") != "fta_structure"
            ):
                continue
            summary = str(item.get("display_summary") or item.get("content") or "")
            if len(query_ngrams & meaningful_ngrams(summary)) >= 2:
                related_fta = True
                break
        if not related_fta:
            return current
    if len(_numbered_fault_candidate_blocks(current, state)) < 2:
        return current

    ledger = _citation_ledger(dict(state))
    cited_fmeca = {
        number
        for number, item in ledger.items()
        if str(item.get("doc_type") or "") == "fmeca"
        and f"[^{number}]" in current
    }
    if len(cited_fmeca) < 2:
        return current

    for number, item in ledger.items():
        if (
            str(item.get("doc_type") or "") != "fta"
            or str(item.get("claim_type") or "") != "fta_structure"
            or f"[^{number}]" in current
        ):
            continue
        summary = str(item.get("display_summary") or item.get("content") or "")
        match = re.search(r"建模说明\s*[:：]\s*(.+?)(?:[。\n]|$)", summary)
        if not match:
            continue
        modelling_note = re.sub(r"\s*;\s*", "；", match.group(1)).strip(" ；。")
        if not modelling_note:
            continue
        lead_in = f"**先看结构**：{modelling_note}。[^{number}]"
        if current.startswith(_MULTI_CANDIDATE_FRAMING):
            remainder = current[len(_MULTI_CANDIDATE_FRAMING):].lstrip()
            return _normalize_generated_markdown(
                f"{_MULTI_CANDIDATE_FRAMING}\n\n{lead_in}\n\n{remainder}"
            )
        return _normalize_generated_markdown(f"{lead_in}\n\n{current}")
    return current


def _fta_structure_graph(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read a canonical FTA graph from one approved evidence item."""

    raw = item.get("content")
    if isinstance(raw, Mapping):
        return raw
    text = str(raw or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _ensure_fta_structure_only_completeness(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Complete an explicit FTA-structure answer without opening diagnosis.

    The normal answer model sees a compact source summary.  That summary is
    intentionally bounded and can contain a phrase such as “以下三类条件：”
    without actually listing the nodes.  For a structure-only request the
    canonical graph itself is the safe source for a deterministic node list;
    this helper copies only names/descriptions that are present in that graph
    and never adds HMC, FMECA, history or maintenance facts.
    """

    current = str(answer or "").strip()
    if not _is_fta_structure_only_request(state):
        return current
    ledger = _citation_ledger(dict(state))
    selected: Tuple[int, Mapping[str, Any]] | None = None
    for number, item in ledger.items():
        if (
            str(item.get("doc_type") or "") == "fta"
            and str(item.get("claim_type") or "") == "fta_structure"
        ):
            selected = (number, item)
            break

    # Remove a dangling model sentence even when a legacy/summary-only row
    # does not carry the canonical JSON graph.  A promise followed by no list
    # is worse than a concise statement of the available structure.
    lines = [
        line
        for line in current.splitlines()
        if not re.search(
            r"(?:以下|下列)(?:[一二两三四五六七八九十\d]+)?(?:类|项)?"
            r"(?:下层|基本)?(?:条件|事件).{0,24}[：:]\s*$",
            line.strip(),
        )
    ]
    current = _normalize_generated_markdown("\n".join(lines))
    if selected is None:
        return current
    citation, item = selected
    graph = _fta_structure_graph(item)
    raw_vertices = graph.get("vertices") or graph.get("nodes") or []
    vertices = [node for node in raw_vertices if isinstance(node, Mapping)]

    def text_value(value: Any, limit: int = 180) -> str:
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return ""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    basic_events: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for node in vertices:
        properties = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        label = text_value(node.get("label") or properties.get("type"), 80).casefold()
        if label not in {"basicevent", "basic_event", "basic event", "基本事件"}:
            continue
        name = text_value(
            properties.get("name")
            or properties.get("event_name")
            or node.get("name")
            or node.get("title")
            or node.get("id"),
            220,
        )
        if not name or name in seen:
            continue
        seen.add(name)
        description = text_value(
            properties.get("description")
            or properties.get("failure_cause")
            or properties.get("cause")
            or properties.get("reason"),
            180,
        )
        basic_events.append((name, description))
    # The answer-side evidence window may already contain the public summary
    # rather than the raw graph.  In that case use the bounded structured
    # ``answer_fields.basic_events`` projection produced by the FTA tool; it
    # is still PostgreSQL-backed and avoids exposing JSON to the model/UI.
    if not basic_events:
        answer_fields = _evidence_answer_fields(item)
        projected = answer_fields.get("basic_events")
        if isinstance(projected, (list, tuple)):
            for entry in projected:
                if isinstance(entry, str):
                    try:
                        parsed_entry = ast.literal_eval(entry)
                    except (SyntaxError, ValueError):
                        parsed_entry = None
                    if isinstance(parsed_entry, Mapping):
                        entry = parsed_entry
                if not isinstance(entry, Mapping):
                    continue
                name = text_value(entry.get("name") or entry.get("event_name"), 220)
                if not name or name in seen:
                    continue
                seen.add(name)
                description = text_value(
                    entry.get("description")
                    or entry.get("failure_cause")
                    or entry.get("cause")
                    or entry.get("reason"),
                    180,
                )
                basic_events.append((name, description))
    if not basic_events:
        return current

    # The model often labels a complete canonical basic-event list as a
    # “selection”.  That wording makes the user think that additional events
    # were omitted, so normalize it to the count that was actually read from
    # the graph.
    current = re.sub(
        r"基本事件\s*[（(](?:节选|部分|底层原因|下层事件|基本原因)[）)]",
        f"基本事件（共 {len(basic_events)} 项）",
        current,
    )
    if re.search(r"基本事件", current) and not re.search(r"基本事件.*共\s*\d+\s*项", current):
        current = re.sub(
            r"基本事件",
            f"基本事件（共 {len(basic_events)} 项）",
            current,
            count=1,
        )
    # Some local models repeat the same complete list under a second heading
    # such as “基本事件详情”.  De-duplicate by the canonical event name while
    # preserving the first occurrence (which normally carries the richer
    # description and citation).
    event_names = [name for name, _description in basic_events]
    seen_event_lines: set[str] = set()
    deduplicated_lines: List[str] = []
    for line in current.splitlines():
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
            matched_name = next(
                (name for name in event_names if name and name in line),
                "",
            )
            if matched_name:
                if matched_name in seen_event_lines:
                    continue
                seen_event_lines.add(matched_name)
        deduplicated_lines.append(line)
    current = "\n".join(deduplicated_lines)
    current = re.sub(
        r"(?m)^\s*(?:#{1,6}\s*)?(?:基本事件详情|基本事件详细列表)\s*(?:\[\^\d+\])?\s*$\n?",
        "",
        current,
    )
    # Do not leave a section title with no body after citation repair.  This
    # is common when a local model emits ``建模说明`` as a placeholder but
    # the validated source only supplied the summary already shown above.
    cleaned_sections: List[str] = []
    structure_heading = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:建模说明|结构说明|逻辑门说明)\s*(?:\[\^\d+\])?\s*$"
    )
    for index, line in enumerate(current.splitlines()):
        if structure_heading.match(line):
            next_index = index + 1
            while next_index < len(current.splitlines()) and not current.splitlines()[next_index].strip():
                next_index += 1
            if next_index >= len(current.splitlines()) or re.match(
                r"^\s*#{1,6}\s+", current.splitlines()[next_index]
            ):
                continue
        cleaned_sections.append(line)
    current = "\n".join(cleaned_sections)
    if not re.search(r"基本事件", current):
        lines = current.splitlines()
        insert_at = next(
            (
                index
                for index, line in enumerate(lines)
                if any(name and name in line for name in event_names)
            ),
            len(lines),
        )
        lines.insert(insert_at, f"FTA基本事件（共 {len(basic_events)} 项）")
        current = "\n".join(lines)
    normalized_current = _compact_semantic_text(current)
    if all(_compact_semantic_text(name) in normalized_current for name, _ in basic_events):
        return current
    title = str(item.get("title") or item.get("record_id") or "该故障树").strip()
    header = f"### {title}的下层基本事件（共 {len(basic_events)} 项）"
    body = [header]
    for name, description in basic_events:
        suffix = f"：{description}" if description else ""
        body.append(f"- **{name}**{suffix}。[^{citation}]")
    return _normalize_generated_markdown(f"{current}\n\n" + "\n".join(body))


def _normalize_fta_calculation_structure(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Prevent a mixed-order cut-set list from being mislabeled as one order."""

    requests = {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
        if str(value or "").strip()
    }
    if "fta_cut_sets" not in requests:
        return str(answer or "")
    manifest = _fta_calculation_manifest(dict(state))
    if not manifest:
        return str(answer or "")
    current = str(answer or "")
    mixed_orders = any(
        len([part for part in re.split(r"\s*\+\s*|\s*[∧&]\s*", str(cut)) if part.strip()]) > 1
        for item in manifest
        for cut in (item.get("cut_sets") or [])
    ) and any(
        len([part for part in re.split(r"\s*\+\s*|\s*[∧&]\s*", str(cut)) if part.strip()]) == 1
        for item in manifest
        for cut in (item.get("cut_sets") or [])
    )
    if not mixed_orders:
        return current
    # A single heading claiming every result is first-order is false as soon
    # as one combination contains two basic events.  Use an order-neutral
    # heading and remove an unsupported “二阶无” sentence; the deterministic
    # list immediately below remains the authoritative complete result.
    current = re.sub(
        r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?一阶最小割集(?:\*\*)?[^\n]*$",
        lambda match: re.sub(r"一阶最小割集", "最小割集", match.group(0)),
        current,
    )
    current = re.sub(
        r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?二阶最小割集(?:\*\*)?\s*[：:]?\s*无[^\n]*\n?",
        "",
        current,
    )
    # The calculation reviewer may append its machine manifest after the
    # model has already presented the complete numbered list.  Keep the
    # user-facing list once; repeating all nine sets in a long inline sentence
    # makes the result look like two conflicting calculations.
    if re.search(r"(?m)最小割集(?:列表)?\s*(?:（[^\n]*）)?", current):
        current = re.sub(
            r"(?m)^\s*:?\s*FTA 确定性计算：[^\n]*完整列表：[^\n]*\n?",
            "",
            current,
        )
    # Do not publish empty model-generated sections.  A deterministic result
    # already tells the user what was computed; a heading with only a
    # citation marker looks like a missing tool result.
    lines = current.splitlines()
    empty_section_labels = {
        "组合含义与处置说明",
        "参数缺失说明",
        "需要补充的问题",
        "参数说明",
    }
    cleaned: List[str] = []
    for index, line in enumerate(lines):
        label = re.sub(r"\[\^\d+\]", "", line).strip()
        label = re.sub(r"^[#*\-\s]+", "", label).strip("：:")
        if label in empty_section_labels:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index >= len(lines) or re.sub(
                r"^[#*\-\s]+", "", re.sub(r"\[\^\d+\]", "", lines[next_index]).strip()
            ).strip("：:") in empty_section_labels:
                continue
        cleaned.append(line)
    current = "\n".join(cleaned)
    return _normalize_generated_markdown(current)


def _ensure_fta_calculation_output(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Keep the deterministic FTA result visible when the editor omits it."""

    if "fta_cut_sets" not in {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
    }:
        return str(answer or "")
    manifest = _fta_calculation_manifest(dict(state))
    if not manifest:
        return str(answer or "")
    current = str(answer or "").strip()
    additions: List[str] = []
    for item in manifest:
        if not item.get("include_cut_sets"):
            continue
        citation = int(item.get("citation_id") or 0)
        marker = f"[^{citation}]" if citation > 0 else ""
        cuts = [str(value or "").strip() for value in (item.get("cut_sets") or []) if str(value or "").strip()]
        if not cuts:
            continue
        normalized = _compact_semantic_text(current)
        if all(_normalize_calculation_identity(cut) in _normalize_calculation_identity(normalized) for cut in cuts):
            continue
        additions.extend(
            [
                f"### 最小割集完整清单（共 {int(item.get('cut_set_count') or len(cuts))} 个）{marker}",
                *(
                    f"{index}. {cut}{marker}"
                    for index, cut in enumerate(cuts, start=1)
                ),
            ]
        )
    if not additions:
        return current
    return _normalize_generated_markdown("\n\n".join([current, "\n".join(additions)]).strip())


def _rbd_calculation_manifest(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests = {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
        if str(value or "").strip()
    }
    if "rbd_reliability" not in requests:
        return []
    results: List[Dict[str, Any]] = []
    for number, item in _citation_ledger(state).items():
        if (
            str(item.get("doc_type") or "") != "rbd"
            or str(item.get("authority") or "") != "deterministic_derivation"
            or str(item.get("claim_type") or "") != "rbd_calculation"
        ):
            continue
        content = str(item.get("display_summary") or item.get("content") or "")
        match = re.search(r"系统可靠度\s*[：:]\s*(0(?:\.\d+)?|1(?:\.0+)?)", content)
        if not match:
            continue
        model_name = str(item.get("title") or "").replace(
            "RBD 确定性计算：", ""
        ).strip() or str(item.get("record_id") or "未命名 RBD 模型")
        results.append(
            {
                "citation_id": number,
                "record_id": str(item.get("record_id") or ""),
                "model": model_name,
                "reliability": match.group(1),
                "source_summary": content[:1200],
            }
        )
    return results


def _missing_rbd_calculation_values(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
) -> List[str]:
    text = str(answer or "")
    clauses = [
        part.strip()
        for part in re.split(r"[，,；;。！？!?\n]+", text)
        if part.strip()
    ]
    missing: List[str] = []
    for item in manifest:
        citation_id = item.get("citation_id")
        marker = f"[^{citation_id}]"
        model = str(item.get("model") or "").strip()
        reliability = str(item.get("reliability") or "").strip()
        normalized_aliases = _rbd_model_identity_aliases(model)
        value_pattern = re.compile(
            rf"(?<![\d.]){re.escape(reliability)}(?![\d.])"
        )
        complete = marker in text and any(
            bool(value_pattern.search(clause))
            and any(
                alias in _normalize_calculation_identity(clause)
                for alias in normalized_aliases
            )
            for clause in clauses
        )
        # The final semantic citation gate verifies the exact local binding.
        # Here the marker may sit at the end of a combined sentence; requiring
        # it inside every comma-delimited clause would cause needless editor
        # retries even though the model identity and value are already paired.
        if not complete:
            missing.append(f"{marker} {model}={reliability}")
    return missing


def _normalize_calculation_identity(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _rbd_model_identity_aliases(model: str) -> Tuple[str, ...]:
    """Return conservative display aliases for one deterministic RBD model.

    The calculation title commonly ends in ``可靠性模型`` while a readable
    answer says ``电源保障链``.  Only explicit display variants are allowed
    here; fuzzy similarity must not make one model's value belong to another.
    """

    raw = re.sub(r"^RBD\s*确定性计算\s*[：:]\s*", "", str(model or ""), flags=re.I).strip()
    candidates = [raw]
    for suffix in ("可靠性框图模型", "可靠性模型", "可靠度模型", "模型"):
        if raw.endswith(suffix) and len(raw) > len(suffix):
            candidates.append(raw[: -len(suffix)].strip())
    for candidate in list(candidates):
        for suffix in ("功能", "链"):
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                candidates.append(candidate[: -len(suffix)].strip())
    for candidate in list(candidates):
        if "外部" in candidate:
            candidates.append(candidate.replace("外部", "对外"))
        if "对外" in candidate:
            candidates.append(candidate.replace("对外", "外部"))
    aliases = [
        _normalize_calculation_identity(candidate)
        for candidate in candidates
        if len(_normalize_calculation_identity(candidate)) >= 2
    ]
    return tuple(dict.fromkeys(aliases))


def _parse_fta_cut_set_summary(summary: str) -> Tuple[int | None, List[str]]:
    text = str(summary or "")
    match = re.search(
        r"最小割集\s*(\d+)\s*个(?:\s*[（(](?:完整列表|前\s*\d+\s*个)\s*[：:]\s*(.*?)[）)])?",
        text,
    )
    if not match:
        return None, []
    count = int(match.group(1))
    raw_list = str(match.group(2) or "").strip()
    if not raw_list:
        return count, []
    values: List[str] = []
    for part in re.split(r"\s*[；;]\s*(?=\d+\s*[.、)])", raw_list):
        value = re.sub(r"^\s*\d+\s*[.、)]\s*", "", part).strip()
        value = re.sub(r"[；;…]+\s*$", "", value).strip()
        if value and value not in values:
            values.append(value)
    return count, values


def _parse_fta_cut_set_content(content: str) -> Tuple[int | None, List[str]]:
    try:
        payload = json.loads(str(content or ""), parse_float=str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, []
    if not isinstance(payload, Mapping):
        return None, []
    raw_sets = payload.get("minimal_cut_sets") or []
    values: List[str] = []
    if isinstance(raw_sets, (list, tuple)):
        for raw_set in raw_sets:
            members = (
                list(raw_set)
                if isinstance(raw_set, (list, tuple, set, frozenset))
                else [raw_set]
            )
            label = " + ".join(str(member or "").strip() for member in members if str(member or "").strip())
            if label and label not in values:
                values.append(label)
    raw_count = payload.get("minimal_cut_set_count")
    try:
        count = int(raw_count) if raw_count is not None else len(values)
    except (TypeError, ValueError):
        count = len(values)
    return count, values


def _fta_calculation_payload(content: Any) -> Mapping[str, Any]:
    text = str(content or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        payload = json.loads(text, parse_float=str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _calculation_number_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    text = str(value).strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return ""
    try:
        return text if math.isfinite(float(text)) else ""
    except (TypeError, ValueError):
        return ""


def _fta_input_parameters(payload: Mapping[str, Any]) -> List[str]:
    """Preserve explicitly returned inputs; never derive them from a result."""

    raw = next(
        (
            payload.get(key)
            for key in (
                "input_parameters",
                "basic_event_probabilities",
                "input_probabilities",
                "event_probabilities",
            )
            if payload.get(key) not in (None, "", [], {})
        ),
        None,
    )
    values: List[str] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            rendered = _calculation_number_text(value) or str(value or "").strip()
            if str(name or "").strip() and rendered:
                values.append(f"{str(name).strip()}={rendered}")
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, Mapping):
                name = str(item.get("name") or item.get("event") or item.get("id") or "").strip()
                value = item.get("probability", item.get("failure_rate", item.get("value")))
                rendered = _calculation_number_text(value) or str(value or "").strip()
                if name and rendered:
                    values.append(f"{name}={rendered}")
            else:
                rendered = str(item or "").strip()
                if rendered:
                    values.append(rendered)
    return list(dict.fromkeys(values))


def _fta_calculation_manifest(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests = {
        str(value or "").strip()
        for value in (state.get("calculation_requests") or [])
        if str(value or "").strip()
    }
    include_cut_sets = "fta_cut_sets" in requests
    include_probability = "fta_probability" in requests
    if not (include_cut_sets or include_probability):
        return []
    results: List[Dict[str, Any]] = []
    for number, item in _citation_ledger(state).items():
        if (
            str(item.get("doc_type") or "") != "fta"
            or str(item.get("authority") or "") != "deterministic_derivation"
            or str(item.get("claim_type") or "") != "fta_calculation"
        ):
            continue
        summary = str(item.get("display_summary") or item.get("content") or "")
        payload = _fta_calculation_payload(item.get("content"))
        count, cut_sets = _parse_fta_cut_set_summary(summary)
        content_count, content_cut_sets = _parse_fta_cut_set_content(
            str(item.get("content") or "")
        )
        if count is None:
            count = content_count
        # The human-readable summary maps graph IDs to event names.  Use the
        # structured result only when that projection is unavailable.
        if not cut_sets:
            cut_sets = content_cut_sets
        if include_cut_sets and count is None:
            continue
        top_event = str(payload.get("top_event") or "").strip()
        if not top_event:
            top_match = re.search(r"顶事件\s*[：:]\s*([^；;。]+)", summary)
            top_event = str(top_match.group(1) if top_match else "").strip()
        probability_present = "top_event_probability" in payload
        probability = _calculation_number_text(payload.get("top_event_probability"))
        if not probability_present:
            probability_match = re.search(
                r"顶事件概率\s*[：:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                summary,
            )
            if probability_match:
                probability_present = True
                probability = probability_match.group(1)
        mission_time = _calculation_number_text(payload.get("mission_time"))
        missing_present = "missing_probability_nodes" in payload
        missing_nodes = [
            str(value or "").strip()
            for value in (payload.get("missing_probability_nodes") or [])
            if str(value or "").strip()
        ]
        block_reason = str(payload.get("probability_block_reason") or "").strip()
        assumption = str(payload.get("assumption") or "").strip()
        limitations = [
            str(value or "").strip()
            for value in (item.get("limitations") or [])
            if str(value or "").strip()
            and str(value or "").strip() not in {assumption, block_reason}
        ]
        input_parameters = _fta_input_parameters(payload)
        if include_probability and not (
            probability_present
            or mission_time
            or missing_present
            or block_reason
            or assumption
            or input_parameters
            or limitations
        ):
            continue
        results.append(
            {
                "citation_id": number,
                "record_id": str(item.get("record_id") or ""),
                "model": str(item.get("title") or "").replace("FTA 确定性计算：", ""),
                "include_cut_sets": include_cut_sets,
                "include_probability": include_probability,
                "cut_set_count": count,
                "cut_sets": cut_sets,
                "top_event": top_event,
                "top_event_probability": probability,
                "probability_present": probability_present,
                "mission_time": mission_time,
                "input_parameters": input_parameters,
                "missing_parameters_present": missing_present,
                "missing_probability_nodes": missing_nodes,
                "probability_block_reason": block_reason,
                "assumption": assumption,
                "limitations": list(dict.fromkeys(limitations)),
                "source_summary": summary[:2400],
            }
        )
    return results


def _calculation_sentences(answer: str) -> List[str]:
    """Keep citations following terminal punctuation attached to the sentence."""

    return [
        match.group(0).strip()
        for match in re.finditer(
            r"[^。！？!?\n]+(?:[。！？!?]+)?(?:\s*\[\^\d+\])*",
            str(answer or ""),
        )
        if match.group(0).strip()
    ]


def _missing_fta_calculation_values(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
) -> List[str]:
    sentences = _calculation_sentences(answer)
    missing: List[str] = []
    for item in manifest:
        citation_id = item.get("citation_id")
        marker = f"[^{citation_id}]"
        cited_text = "\n".join(sentence for sentence in sentences if marker in sentence)
        if item.get("include_cut_sets"):
            count = int(item.get("cut_set_count") or 0)
            count_pattern = re.compile(
                rf"(?:最小割集[^。！？!?\n]{{0,24}}(?<!\d){count}(?!\d)\s*个|"
                rf"(?<!\d){count}(?!\d)\s*个[^。！？!?\n]{{0,12}}最小割集)"
            )
            if not any(marker in sentence and count_pattern.search(sentence) for sentence in sentences):
                missing.append(f"{marker} 最小割集总数={count}")
            for cut_set in item.get("cut_sets") or []:
                normalized_cut = _normalize_calculation_identity(cut_set)
                if normalized_cut and not any(
                    marker in sentence
                    and normalized_cut in _normalize_calculation_identity(sentence)
                    for sentence in sentences
                ):
                    missing.append(f"{marker} 最小割集={cut_set}")
        if not item.get("include_probability"):
            continue
        top_event = str(item.get("top_event") or "").strip()
        if top_event and not _target_value_is_present(cited_text, top_event):
            missing.append(f"{marker} 顶事件={top_event}")
        probability = str(item.get("top_event_probability") or "").strip()
        if probability:
            probability_pattern = re.compile(rf"(?<![\d.]){re.escape(probability)}(?![\d.])")
            if not probability_pattern.search(cited_text):
                missing.append(f"{marker} 顶事件概率={probability}")
        elif item.get("probability_present") and not re.search(
            r"未形成|无法(?:形成|给出|计算)|不能(?:形成|给出|计算)|缺少.{0,16}概率",
            cited_text,
        ):
            missing.append(f"{marker} 未形成顶事件概率结论")
        mission_time = str(item.get("mission_time") or "").strip()
        if mission_time and not re.search(
            rf"(?<![\d.]){re.escape(mission_time)}(?![\d.])\s*(?:小时|h|hr|hour)",
            cited_text,
            re.IGNORECASE,
        ):
            missing.append(f"{marker} 任务时间={mission_time}小时")
        for parameter in item.get("input_parameters") or []:
            if not _target_value_is_present(cited_text, str(parameter)):
                missing.append(f"{marker} 输入参数={parameter}")
        missing_nodes = item.get("missing_probability_nodes") or []
        if missing_nodes:
            if not re.search(r"缺少|缺失|未提供|参数不足", cited_text) or not all(
                _target_value_is_present(cited_text, str(value)) for value in missing_nodes
            ):
                missing.append(f"{marker} 缺失概率参数=" + "、".join(missing_nodes))
        elif item.get("missing_parameters_present") and not re.search(
            r"参数(?:完整|齐全)|无缺失.{0,8}(?:概率|输入)?参数|未缺少.{0,8}(?:概率|输入)?参数",
            cited_text,
        ):
            missing.append(f"{marker} 概率输入参数完整")
        for label, value in (
            ("计算限制", item.get("probability_block_reason")),
            ("计算假设", item.get("assumption")),
        ):
            text = str(value or "").strip()
            if text and not _target_value_is_present(cited_text, text):
                missing.append(f"{marker} {label}={text}")
        for limitation in item.get("limitations") or []:
            if not _target_value_is_present(cited_text, str(limitation)):
                missing.append(f"{marker} 计算限制={limitation}")
    return missing


def _format_rbd_calculation_manifest(
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    return "\n".join(
        f"- [^{item.get('citation_id')}] {item.get('model')}：系统可靠度 "
        f"{item.get('reliability')}（记录 {item.get('record_id') or '未标识'}）"
        for item in manifest
    ) or "无"


def _format_fta_calculation_manifest(
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    lines: List[str] = []
    for item in manifest:
        marker = f"[^{item.get('citation_id')}]"
        facts: List[str] = []
        if item.get("include_cut_sets"):
            cuts = item.get("cut_sets") or []
            cut_text = "；".join(
                f"{index}. {value}" for index, value in enumerate(cuts, start=1)
            ) or "证据未提供可逐项展示的列表"
            facts.append(f"最小割集 {item.get('cut_set_count')} 个；完整列表：{cut_text}")
        if item.get("include_probability"):
            if item.get("top_event"):
                facts.append(f"顶事件：{item.get('top_event')}")
            probability = str(item.get("top_event_probability") or "").strip()
            facts.append(
                f"顶事件概率：{probability}"
                if probability
                else "未形成顶事件概率结论"
            )
            if item.get("mission_time"):
                facts.append(f"任务时间：{item.get('mission_time')} 小时")
            if item.get("input_parameters"):
                facts.append("输入参数：" + "；".join(item.get("input_parameters") or []))
            if item.get("missing_parameters_present"):
                missing_nodes = item.get("missing_probability_nodes") or []
                facts.append(
                    "缺失概率参数：" + "、".join(missing_nodes)
                    if missing_nodes
                    else "概率输入参数：完整（确定性结果未报告缺项）"
                )
            if item.get("probability_block_reason"):
                facts.append(f"计算限制：{item.get('probability_block_reason')}")
            if item.get("assumption"):
                facts.append(f"计算假设：{item.get('assumption')}")
            for limitation in item.get("limitations") or []:
                facts.append(f"计算限制：{limitation}")
        lines.append(f"- {marker} {item.get('model') or 'FTA'}：" + "；".join(facts))
    return "\n".join(lines) or "无"


