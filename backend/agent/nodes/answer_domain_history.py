"""Answer 节点·历史案例域策略模块（C1 批 4 从 answer.py 抽出）。

包含：历史证据强度判定、历史补充召回、历史清单完整性、
历史详情/引用/去重等确定性策略函数。

依赖约定与 answer_domain_fault 相同：共享 helper 留在 answer.py 内核，
本模块顶部对内核的引用在 answer.py 完整加载后才执行（answer.py 在文件
底部再导入本模块，规避循环导入）。
"""

import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.agent.answer_contract import focused_answer_contract, required_answer_fields
from backend.agent.contracts import deduplicate_evidence
from backend.agent.query_understanding import extract_business_codes
from backend.agent.nodes.answer_presence import (
    approved_id_set,
    normalize_presence_text,
    semantic_tokens as _semantic_tokens,
)
from backend.agent.nodes.answer_domain_fault import _prompt_payload_as_evidence
from backend.agent.nodes.answer import (
    _answer_query_text,
    _answer_relevance_score,
    _citation_ledger,
    _compact_semantic_text,
    _context_evidence,
    _evidence_answer_fields,
    _evidence_object_business_ids,
    _evidence_payload,
    _history_section_context,
    _insert_into_matching_markdown_section,
    _normalize_business_id,
    _normalize_generated_markdown,
    _payload_direction,
    _section_title,
    _unit_requires_fault_history,
    _SECTION_HEADING,
)


# ---- 历史域专属常量（随本模块迁移）----
HISTORY_SUPPLEMENT_MIN_SCORE = 0.15


HISTORY_SUPPLEMENT_RELATIVE_RATIO = 0.75


HISTORY_SUPPLEMENT_MAX_CASES = 2


def _history_evidence_is_strong(item: Mapping[str, Any]) -> bool:
    """Whether a history row is relevant enough for supplemental prose.

    Exact-reference rows are created only from an explicit business-code
    linkage.  Semantic identity hydration additionally needs a retrieval score
    threshold: PostgreSQL identity confirms the row exists, but does not by
    itself make a weakly recalled case similar to the user's fault.
    """

    match_type = str(item.get("match_type") or "")
    if match_type == "exact_history_reference":
        return True
    if match_type != "semantic_id_hydration":
        return False
    # 公开引用压缩会剔除私有 rerank 字段，重新打上的 answer_rerank_score
    # 才是答案窗口实际使用的分数。优先读它，避免健康案例被判为弱匹配。
    raw_score = item.get("answer_rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("rerank_score")
    if not isinstance(raw_score, (int, float)):
        raw_score = item.get("retrieval_score")
    return (
        isinstance(raw_score, (int, float))
        and float(raw_score) >= HISTORY_SUPPLEMENT_MIN_SCORE
    )


def _supplemental_history_evidence_ids(
    evidence: Sequence[Mapping[str, Any]],
    *,
    limit: int = HISTORY_SUPPLEMENT_MAX_CASES,
) -> set[str]:
    """Select a small, query-relative set of genuinely similar cases.

    An exact cross-document history reference is stronger than a semantic
    neighbor.  Semantic scores from different queries are not globally
    calibrated, therefore an absolute floor alone is insufficient: after the
    floor, candidates must also stay close to the best case retrieved for this
    query.  Selection is by unique case identity so duplicate OpenSearch and
    PostgreSQL representations cannot consume the answer budget twice.
    """

    bounded_limit = max(1, int(limit or HISTORY_SUPPLEMENT_MAX_CASES))

    def score(item: Mapping[str, Any]) -> float:
        # Keep this identical to _history_evidence_is_strong's score source.
        # answer_rerank_score is re-applied by the answer window after public
        # citation compaction strips the private rerank fields; without it a
        # healthy hydrated case falls back to -1.0 and the relative floor
        # below silently drops the closest case.
        for key in ("answer_rerank_score", "rerank_score", "retrieval_score"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return -1.0

    exact = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "fault_history"
        and str(item.get("match_type") or "") == "exact_history_reference"
    ]
    semantic = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "fault_history"
        and _history_evidence_is_strong(item)
    ]
    if semantic:
        best_score = max(score(item) for item in semantic)
        relative_floor = max(
            HISTORY_SUPPLEMENT_MIN_SCORE,
            best_score * HISTORY_SUPPLEMENT_RELATIVE_RATIO,
        )
        semantic = [item for item in semantic if score(item) >= relative_floor]

    ranked = [
        *sorted(exact, key=score, reverse=True),
        *sorted(semantic, key=score, reverse=True),
    ]
    selected_ids: set[str] = set()
    selected_records: set[str] = set()
    for item in ranked:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id:
            continue
        record_identity = str(
            item.get("record_id") or item.get("title") or evidence_id
        ).strip()
        if record_identity in selected_records:
            continue
        selected_ids.add(evidence_id)
        selected_records.add(record_identity)
        if len(selected_records) >= bounded_limit:
            break
    return selected_ids


def _history_subject_matches_state(
    history_item: Mapping[str, Any],
    state: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> bool:
    """Keep an automatic history supplement tied to this turn's subject.

    A high semantic score such as ``高负载`` is not enough to call a case
    similar: the power-path incident and a thermal-controller incident can
    share those words while requiring different checks and repairs.  When the
    current evidence already binds a concrete FMECA/HMC/MFL subject, prefer
    explicit cross-document IDs and module/function IDs.  If the history row
    carries explicit links which all disagree with that subject, it is audit
    context only and must not be written as a previous occurrence.
    """

    # Apply explicit topology terms even for a standalone history query. If
    # no current FMECA/HMC anchor was retrieved, do not let every PWR case
    # pass solely because it shares the subsystem ID.
    history_fields = _evidence_answer_fields(history_item)
    query_text = _compact_semantic_text(_answer_query_text(state))
    branch_query = bool(re.search(r"主供电|备用|冗余|A路|B路|支路|接管|切换", query_text))
    history_text_for_scope = _compact_semantic_text(
        " ".join(
            str(history_fields.get(key) or "")
            for key in ("fault_symptom", "root_cause", "related_module", "related_function")
        )
    )
    if branch_query:
        if not re.search(
            r"A路|主供电",
            history_text_for_scope,
            re.IGNORECASE,
        ):
            return False
        if not re.search(
            r"接管|切换|备用|冗余|B路",
            history_text_for_scope,
            re.IGNORECASE,
        ):
            return False

    anchors = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") in {"fmeca", "hmc", "mfl", "function_locator"}
        and str(item.get("record_id") or "").strip()
    ]
    if not anchors:
        return True

    explicit = history_fields.get("explicit_references")
    explicit_values: set[str] = set()
    if isinstance(explicit, Mapping):
        for raw_values in explicit.values():
            values = (
                list(raw_values)
                if isinstance(raw_values, (list, tuple, set))
                else re.split(r"[,，;；\s]+", str(raw_values or ""))
            )
            explicit_values.update(
                str(value or "").strip().upper()
                for value in values
                if str(value or "").strip()
            )

    current_record_ids = {
        str(item.get("record_id") or "").strip().upper()
        for item in anchors
        if str(item.get("record_id") or "").strip()
    }
    # When retrieval produced several FMECA rows, only the highest-ranked
    # subject is the current incident anchor.  Letting any lower-ranked row
    # authorize its linked history would reintroduce a different module's
    # case merely because both mention a generic condition such as “高负载”.
    fmeca_anchors = [
        item for item in anchors if str(item.get("doc_type") or "") == "fmeca"
    ]
    current_anchor_record_ids = current_record_ids
    if fmeca_anchors:
        best_fmeca = max(fmeca_anchors, key=_answer_relevance_score)
        best_id = str(best_fmeca.get("record_id") or "").strip().upper()
        if best_id:
            current_anchor_record_ids = {best_id}
    current_object_ids: set[str] = set()
    for item in anchors:
        # Use the same primary FMECA anchor as the record-ID check below.  A
        # lower-ranked row from another module must not authorize its own
        # history merely because its FUNC-* ID is present in the broad recall.
        if fmeca_anchors and item is not best_fmeca:
            continue
        current_object_ids.update(_evidence_object_business_ids(item))
    history_object_ids = {
        _normalize_business_id(value)
        for value in extract_business_codes(
            [
                history_fields.get("related_module"),
                history_fields.get("related_function"),
                history_fields.get("related_interface"),
                history_fields.get("hmc_code"),
            ],
            limit=48,
        )
        if _normalize_business_id(value)
    }

    if explicit_values:
        if explicit_values & current_anchor_record_ids:
            return True
        if history_object_ids & current_object_ids:
            return True
        # Explicit links exist but none point to the current bound subject.
        # Do not let lexical similarity turn a different module's case into a
        # historical precedent.
        return False

    # A standalone history question may have no current FMECA/HMC anchor. Use
    # the user's explicit topology terms as a second subject boundary so a
    # generic “掉电” query does not return capacitor, display or startup
    # cases merely because they share that word.
    if history_object_ids and current_object_ids:
        return bool(history_object_ids & current_object_ids)

    history_text = _compact_semantic_text(
        " ".join(
            str(history_fields.get(key) or "")
            for key in ("fault_symptom", "root_cause", "related_module", "related_function")
        )
    )
    anchor_text = _compact_semantic_text(
        " ".join(
            str(value or "")
            for item in anchors
            for value in (
                item.get("title"),
                _evidence_answer_fields(item).get("failure_mode"),
                _evidence_answer_fields(item).get("fault_symptom"),
                _evidence_answer_fields(item).get("module_failure"),
            )
        )
    )
    if not history_text or not anchor_text:
        return True
    # Short distinctive phrases are safer than generic terms such as “异常” or
    # “高负载”.  This fallback is only used for records without explicit links.
    distinctive = [
        token
        for token in re.findall(r"[A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{3,}", anchor_text)
        if token not in {"系统", "模块", "功能", "异常", "故障", "高负载"}
    ]
    return any(token in history_text for token in distinctive)


def _history_is_primary_target(state: Mapping[str, Any]) -> bool:
    """Whether this turn explicitly requests history as an answer dimension.

    A mixed question such as “先怎么处置？以前相近案例怎么验证？” has
    ``maintenance_query`` as its primary intent but still makes the history
    record a direct answer target.  Treating history as supplemental in that
    case lets a FMECA citation masquerade as a historical case.  Automatic
    diagnostic closure without an explicit history target remains supplemental.
    """

    intents = {
        str(item.get("intent") or "")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping) and str(item.get("intent") or "")
    }
    if str(state.get("intent") or ""):
        intents.add(str(state.get("intent") or ""))
    if "history_query" in intents:
        # The planner may add a history target for an optional clause such as
        # “如有相似历史也说明”.  That is supplemental context, not a command
        # to admit every structured history row.  Treat an explicit request
        # for a past case/time/operation/result as primary; optional wording
        # remains subject to the semantic score and identity filters.
        query = _compact_semantic_text(_answer_query_text(state))
        optional_history = bool(
            re.search(r"(?:如有|如果有|若有|有的话|有则|可以的话).{0,12}(?:历史|案例|曾经|以前)", query)
        )
        if not optional_history:
            return True
    fields = set(required_answer_fields(state))
    return bool(fields & {
        "history_time",
        "history_root",
        "history_steps",
        "history_solution",
        "history_result",
        "history_personnel",
        "history_preventive_action",
    })


def _history_collection_requested(state: Mapping[str, Any]) -> bool:
    """Whether the user asks for an exhaustive history inventory."""

    query = _compact_semantic_text(_answer_query_text(dict(state)))
    return bool(
        re.search(r"所有|全部|每个|各个|逐条|清单|列表|列出.{0,8}(?:案例|记录)", query)
    )


def _history_review_manifest(
    answer: str,
    state: Dict[str, Any],
    *,
    limit: int = 2,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return history records actually referenced by the draft.

    Model-emitted citation numbers are only hints at this stage, but restricting
    the editor to gate-approved, explicitly referenced fault-history records
    prevents it from pulling an unrelated candidate into the public answer.
    """

    ledger = _citation_ledger(state)
    referenced = {
        int(value)
        for value in re.findall(r"\[\^(\d+)\]", str(answer or ""))
    }
    records: List[Dict[str, Any]] = []
    selected_numbers: set[int] = set()

    target_intents = {
        str(target.get("intent") or "")
        for target in (state.get("query_targets") or [])
        if isinstance(target, Mapping) and str(target.get("intent") or "")
    }
    if str(state.get("intent") or ""):
        target_intents.add(str(state.get("intent") or ""))
    explicit_history_query = _history_is_primary_target(state)

    def relevance(item: Mapping[str, Any]) -> float:
        for key in ("answer_rerank_score", "rerank_score", "retrieval_score"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return -1.0

    query_direction = _payload_direction(_answer_query_text(state))

    def direction_matches(item: Mapping[str, Any]) -> bool:
        evidence_direction = _payload_direction(
            item.get("display_summary") or item.get("content") or ""
        )
        return not (
            query_direction
            and evidence_direction
            and query_direction != evidence_direction
        )

    strong_history_ids = _supplemental_history_evidence_ids(
        list(ledger.values()),
        limit=limit,
    )
    # 业务锚点优先层：公开引用压缩会剔除私有 rerank 字段，仅靠分数可能让
    # 与本轮 FMECA/HMC 锚点绑定同一业务主体的历史案例（如 HIS-SYN-002）
    # 被同一子系统的高分噪声挤出。subject 匹配是比分数更强的相似信号。
    # 但只有本轮确已绑定具体业务主体（账本中存在 FMECA/HMC/MFL/架构锚点）
    # 时才启用：无锚点的纯历史问题仍完全按分数挑选，不能把锚点层变成
    # “所有历史记录都进池”的后门。
    anchor_priority_ids: set[str] = set()
    anchored_ledger_items = [
        item
        for item in ledger.values()
        if (
            str(item.get("doc_type") or "") in {"fmeca", "hmc", "mfl", "function_locator"}
            and str(item.get("record_id") or "").strip()
        )
    ]
    if anchored_ledger_items:
        history_ledger_items = [
            item
            for item in ledger.values()
            if (
                str(item.get("doc_type") or "") == "fault_history"
                and str(item.get("record_id") or item.get("title") or "").strip()
                and str(item.get("evidence_id") or "")
            )
        ]
        for item in history_ledger_items:
            if _history_subject_matches_state(item, state, anchored_ledger_items):
                anchor_priority_ids.add(str(item.get("evidence_id") or ""))

    def anchor_weight(item: Mapping[str, Any]) -> int:
        return 0 if str(item.get("evidence_id") or "") in anchor_priority_ids else 1

    strong_candidates = sorted(
        (
            (number, item)
            for number, item in ledger.items()
            if str(item.get("doc_type") or "") == "fault_history"
            and direction_matches(item)
            and (
                str(item.get("evidence_id") or "") in strong_history_ids
                or str(item.get("evidence_id") or "") in anchor_priority_ids
            )
        ),
        key=lambda pair: (anchor_weight(pair[1]), relevance(pair[1]), -pair[0]),
        reverse=True,
    )
    strong_numbers = {number for number, _item in strong_candidates}

    def add_record(number: int, item: Dict[str, Any]) -> None:
        if number in selected_numbers or len(records) >= max(1, limit):
            return
        item = ledger.get(number)
        if not item or str(item.get("doc_type") or "") != "fault_history":
            return
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        if not metadata and isinstance(item.get("document_metadata"), Mapping):
            # Public citation payloads carry the direct fields under
            # ``document_metadata`` while internal Evidence items use
            # ``metadata``.  Both shapes reach this helper, so keep the
            # answer_fields extraction identical to _evidence_answer_fields.
            metadata = item.get("document_metadata")
        fields = metadata.get("answer_fields") if isinstance(metadata.get("answer_fields"), Mapping) else {}
        selected_fields: Dict[str, Any] = {}
        for key in (
            "case_id",
            "occurred_at",
            "mission_phase",
            "fault_symptom",
            "root_cause",
            "diagnosis_method",
            "steps",
            "final_solution",
            "verification_result",
        ):
            value = fields.get(key)
            if value in (None, "", [], {}):
                continue
            selected_fields[key] = value[:8] if key == "steps" and isinstance(value, list) else value
        records.append(
            {
                "citation_id": number,
                "record_id": str(item.get("record_id") or item.get("title") or ""),
                "payload_direction": _payload_direction(
                    item.get("display_summary") or item.get("content") or ""
                ),
                "answer_fields": selected_fields,
                "source_summary": str(
                    item.get("display_summary") or item.get("content") or ""
                )[:1800],
            }
        )
        selected_numbers.add(number)

    # For an explicit history question, a semantically hydrated PostgreSQL
    # identity outranks broad structured rows even if the draft happened to
    # cite an irrelevant row first.  This is the same failure mode as a broad
    # SQL term such as “正常” returning a temperature case for an A-bus query.
    # When no semantic identity exists, retain the bounded structured fallback
    # so broad history-list questions still work.
    if explicit_history_query and strong_candidates:
        for number in sorted(referenced):
            if number in strong_numbers:
                add_record(number, ledger[number])
        for number, item in strong_candidates:
            add_record(number, item)
    else:
        for number in sorted(referenced):
            item = ledger.get(number)
            if not item or not direction_matches(item):
                continue
            # 显式历史目标下，草稿实际引用的账本记录就是答案目标本身。此处
            # 不再用 match_type/强匹配门控过滤已引用记录：该门控针对的是“未被
            # 引用记录”的补充性注入（防止普通故障答案强塞无关历史）；若对已
            # 引用记录也生效，混合检索召回的案例行（match_type=hybrid_search
            # 等）会被整条拒绝，导致引用修复后无任何历史字段可补，最终退化
            # 为纯 FMECA/HMC 回答（现象：NO_RECORDS 但账本含已引用 HIS 记录）。
            add_record(number, item)

    # If the initial draft omitted history entirely, one strongly matched,
    # exact-identity hydrated case may still be supplied to the completeness
    # editor.  An explicit history question may additionally use bounded,
    # gate-approved structured-query rows: those records are the direct answer
    # target even when the first draft forgot to cite them.  Generic structured
    # rows and weak semantic neighbors remain audit-only for ordinary fault
    # answers, preventing "history for every answer" noise.
    if not records:
        candidates = strong_candidates
        if explicit_history_query and not candidates:
            candidates = sorted(
                (
                    (number, item)
                    for number, item in ledger.items()
                    if str(item.get("doc_type") or "") == "fault_history"
                    and str(item.get("match_type") or "") == "structured_query"
                    and direction_matches(item)
                ),
                key=lambda pair: (relevance(pair[1]), -pair[0]),
                reverse=True,
            )
        for number, item in candidates:
            if str(item.get("doc_type") or "") != "fault_history":
                continue
            is_explicit_structured_record = (
                explicit_history_query
                and str(item.get("match_type") or "") == "structured_query"
            )
            if (
                not is_explicit_structured_record
                and str(item.get("evidence_id") or "") not in strong_history_ids
            ):
                continue
            if not direction_matches(item):
                continue
            add_record(number, item)
            if not explicit_history_query and records:
                break
    unsupported_history_tone = _unit_requires_fault_history(answer) and not records
    return records, unsupported_history_tone


def _format_history_review_manifest(records: Sequence[Mapping[str, Any]]) -> str:
    """Present historical facts to the editor without exposing JSON syntax."""

    field_labels = {
        "case_id": "案例编号",
        "occurred_at": "发生时间",
        "mission_phase": "阶段/模式",
        "fault_symptom": "当时现象",
        "root_cause": "确认根因",
        "diagnosis_method": "排故方法",
        "steps": "关键操作",
        "final_solution": "最终方案",
        "verification_result": "验证结果",
    }
    lines: List[str] = []
    for record in records:
        citation = record.get("citation_id")
        lines.append(
            f"- 故障历史证据 [^{citation}]｜记录 {record.get('record_id') or '未标识'}"
        )
        fields = record.get("answer_fields") if isinstance(record.get("answer_fields"), Mapping) else {}
        for key, value in fields.items():
            label = field_labels.get(str(key), str(key))
            if key == "steps" and isinstance(value, list):
                lines.append(f"  - {label}：")
                lines.extend(f"    {index}. {step}" for index, step in enumerate(value, start=1))
            else:
                lines.append(f"  - {label}：{value}")
    return "\n".join(lines) or "无"


def _occurrence_time_present(answer: str, occurred_at: Any) -> bool:
    source = str(occurred_at or "")
    match = re.search(
        r"(?P<year>20\d{2})[-/.年](?P<month>\d{1,2})[-/.月](?P<day>\d{1,2})日?"
        r"(?:\s+(?P<time>\d{1,2}:\d{2}))?",
        source,
    )
    if not match:
        return source.strip() in str(answer or "")
    year = match.group("year")
    month = str(int(match.group("month")))
    day = str(int(match.group("day")))
    text = str(answer or "")
    date_present = bool(
        re.search(
            rf"{year}(?:[-/.]0?{month}[-/.]0?{day}|年0?{month}月0?{day}日?)",
            text,
        )
    )
    time_value = match.group("time")
    return date_present and (not time_value or time_value in text)


def _history_citation_text(answer: str, citation: Any) -> str:
    """Return only answer blocks bound to one history citation.

    A generic recommendation earlier in the answer must not satisfy the
    completeness check for a later historical case.  Answer lines are the
    same units used by citation repair, so keeping only lines containing the
    record marker also aligns this check with the public evidence binding.
    """

    try:
        marker = rf"\[\^{int(citation)}\]"
    except (TypeError, ValueError):
        return ""
    lines = str(answer or "").splitlines()
    return "\n".join(line for line in lines if re.search(marker, line))


def _history_context_citation_text(answer: str, citation: Any) -> str:
    """Return cited units that are explicitly presented as past-case facts."""

    try:
        marker = rf"\[\^{int(citation)}\]"
    except (TypeError, ValueError):
        return ""
    result: List[str] = []
    active_history_section = False
    for line in str(answer or "").splitlines():
        if _SECTION_HEADING.match(line):
            active_history_section = _history_section_context(
                _section_title(line), active_history_section
            )
            continue
        if not re.search(marker, line):
            continue
        if (
            active_history_section
            or _unit_requires_fault_history(line)
            or re.search(r"(?:该|此|上述|本次)(?:历史)?(?:案例|记录)", line)
            or re.search(
                r"发生于|发生时间|当时|确认根因|根因(?:是|为)|关键操作|"
                r"排故方法|最终方案|验证(?:时|结果)|复测",
                line,
            )
        ):
            result.append(line)
    return "\n".join(result)


def _history_field_citation_text(
    answer: str,
    citation: Any,
    field: str,
) -> str:
    """Return citation-bound lines that explicitly present one history field.

    A sentence such as “当前建议先检查连接器” may cite an old case as
    its rationale, but it is not a report of what operators did in that case.
    Field labels and historical phrasing keep current advice from satisfying
    the history-detail contract accidentally.
    """

    try:
        marker = rf"\[\^{int(citation)}\]"
    except (TypeError, ValueError):
        return ""
    patterns = {
        "fault_symptom": re.compile(r"当时现象|故障现象|(?:历史|案例).{0,18}(?:表现|出现|发生)"),
        "root_cause": re.compile(r"确认根因|当时根因|经排故确认|确认故障|故障位于|根因(?:是|为|指向)"),
        "operations": re.compile(
            r"关键操作|当时处理|排故方法|"
            r"当时.{0,24}(?:检查|调用|抓帧|测量|比对|更换|修复|锁止|压接|配置|回放)"
        ),
        "final_solution": re.compile(r"最终方案|最终处理|当时.{0,20}(?:修复|更换|恢复|回滚|压接|锁止)"),
        "verification_result": re.compile(r"验证结果|完成修复后验证|复测|连续.{0,18}(?:无|正常|低于|小于|满足)"),
    }
    pattern = patterns.get(field)
    if pattern is None:
        return _history_citation_text(answer, citation)
    return "\n".join(
        line
        for line in str(answer or "").splitlines()
        if re.search(marker, line) and pattern.search(line)
    )


_HISTORY_DETAIL_STOPWORDS = {
    "完成",
    "结果",
    "验证",
    "复测",
    "测试",
    "修复",
    "处理",
    "处置",
    "检查",
    "排故",
    "最终",
    "方案",
    "确认",
    "故障",
    "系统",
    "正常",
    "进行",
}


_HISTORY_ACTION_TERM_RE = re.compile(
    r"检查|调用|抓帧|测量|比对|导出|复现|更换|修复|锁止|压接|"
    r"校准|回滚|恢复|重新发布|切换|隔离|导入|固定|调整|配置"
)


_HISTORY_NUMBERED_STEP_RE = re.compile(
    r"^\s*(?:[-+*]\s+)?(?:\d{1,3}\s*[.)、]|[（(]\d{1,3}[）)])\s*"
)


_HISTORY_CURRENT_ADVICE_RE = re.compile(
    r"建议|当前|现场(?:先|应|需|建议)|"
    r"(?:^|[：:，,；;。\s])(?:应当|应|需要|需|请|可先|优先)(?=[\u4e00-\u9fff])"
)


def _history_numbered_citation_text(answer: str, citation: Any) -> str:
    """Return numbered source-like steps carrying one history citation."""

    try:
        marker = rf"\[\^{int(citation)}\]"
    except (TypeError, ValueError):
        return ""
    return "\n".join(
        line
        for line in str(answer or "").splitlines()
        if re.search(marker, line)
        and _HISTORY_NUMBERED_STEP_RE.search(line)
        # Current recommendations must not masquerade as actions performed in
        # an old case even when they cite that case as their rationale.
        and not _HISTORY_CURRENT_ADVICE_RE.search(line)
    )


def _history_numbered_exact_detail_present(
    answer: str,
    citation: Any,
    value: Any,
) -> bool:
    """Whether a numbered cited step contains the exact structured field."""

    source_normalized = normalize_presence_text(value)
    if len(source_normalized) < 6:
        return False
    for line in _history_numbered_citation_text(answer, citation).splitlines():
        line_normalized = normalize_presence_text(line)
        if source_normalized in line_normalized:
            return True
    return False


def _history_detail_present(
    answer: str,
    value: Any,
    *,
    require_source_action: bool = False,
) -> bool:
    """Whether a cited history block contains source-specific field details."""

    values = value if isinstance(value, (list, tuple)) else [value]
    answer_normalized = normalize_presence_text(answer)
    if not answer_normalized:
        return False
    for raw_value in values:
        source = str(raw_value or "").strip()
        if not source:
            continue
        if require_source_action:
            source_actions = set(_HISTORY_ACTION_TERM_RE.findall(source))
            if source_actions and not any(action in str(answer or "") for action in source_actions):
                continue
        source_normalized = normalize_presence_text(source)
        if len(source_normalized) >= 6 and source_normalized in answer_normalized:
            return True
        # A compact answer may retain one concrete operation/result rather
        # than repeat a long source field verbatim.  Accept a meaningful
        # clause or several non-generic semantic anchors, but never a lone
        # word such as "检查" or "验证".
        for fragment in re.split(r"[+；;，,。\n]", source):
            normalized_fragment = normalize_presence_text(fragment)
            if len(normalized_fragment) >= 6 and normalized_fragment in answer_normalized:
                return True
        source_tokens = {
            token
            for token in _semantic_tokens(source)
            if token not in _HISTORY_DETAIL_STOPWORDS and len(token) >= 2
        }
        answer_tokens = _semantic_tokens(answer)
        overlap = source_tokens & answer_tokens
        required = min(4, max(2, math.ceil(len(source_tokens) * 0.15)))
        if len(overlap) >= required:
            return True
    return False


def _history_review_missing_fields(
    answer: str,
    records: Sequence[Mapping[str, Any]],
) -> List[str]:
    missing: List[str] = []
    for record in records:
        citation = record.get("citation_id")
        text = _history_context_citation_text(answer, citation)
        fields = record.get("answer_fields") if isinstance(record.get("answer_fields"), Mapping) else {}
        occurred_at = fields.get("occurred_at")
        if occurred_at and not _occurrence_time_present(text, occurred_at):
            missing.append(f"[^{citation}] occurred_at={occurred_at}")
        root_cause = fields.get("root_cause")
        root_text = _history_field_citation_text(answer, citation, "root_cause") or text
        if root_cause and not _history_detail_present(root_text, root_cause):
            missing.append(f"[^{citation}] root_cause={fields['root_cause']}")
        raw_steps = fields.get("steps")
        step_values = (
            list(raw_steps)
            if isinstance(raw_steps, (list, tuple))
            else [raw_steps]
        )
        operation_values = [
            value
            for value in [fields.get("diagnosis_method"), *step_values]
            if value not in (None, "", [], {})
        ]
        numbered_text = _history_numbered_citation_text(answer, citation)
        operation_text = _history_field_citation_text(
            answer, citation, "operations"
        ) or text
        operation_present = _history_detail_present(operation_text, operation_values)
        if numbered_text:
            operation_present = operation_present or _history_detail_present(
                numbered_text,
                operation_values,
                require_source_action=True,
            )
        if operation_values and not operation_present:
            missing.append(f"[^{citation}] steps/diagnosis_method")
        final_solution = fields.get("final_solution")
        final_text = _history_field_citation_text(
            answer, citation, "final_solution"
        ) or text
        final_present = _history_detail_present(
            final_text,
            final_solution,
            require_source_action=True,
        ) if final_solution else True
        if final_solution and not final_present and not _history_numbered_exact_detail_present(
            answer,
            citation,
            final_solution,
        ):
            missing.append(f"[^{citation}] final_solution={fields['final_solution']}")
        verification_text = _history_field_citation_text(
            answer, citation, "verification_result"
        ) or text
        if fields.get("verification_result") and not _history_detail_present(
            verification_text, fields.get("verification_result")
        ):
            missing.append(f"[^{citation}] verification_result={fields['verification_result']}")
    return missing


def _history_completion_missing_keys(
    answer: str,
    record: Mapping[str, Any],
) -> set[str]:
    """Return source fields still absent from this record's cited answer units.

    This is deliberately stricter than checking whether the answer contains a
    generic word such as ``根因`` or ``验证``.  A field is complete only when
    the prose carrying the same history citation retains concrete details from
    that PostgreSQL record.  The citation-local check prevents current repair
    advice from accidentally satisfying an old case's completeness contract.
    """

    citation = record.get("citation_id")
    text = _history_context_citation_text(answer, citation)
    fields = (
        record.get("answer_fields")
        if isinstance(record.get("answer_fields"), Mapping)
        else {}
    )
    missing: set[str] = set()
    occurred_at = fields.get("occurred_at")
    if occurred_at and not _occurrence_time_present(text, occurred_at):
        missing.add("occurred_at")
    fault_symptom = fields.get("fault_symptom")
    symptom_text = _history_field_citation_text(
        answer, citation, "fault_symptom"
    ) or text
    if fault_symptom and not _history_detail_present(symptom_text, fault_symptom):
        missing.add("fault_symptom")
    root_cause = fields.get("root_cause")
    root_text = _history_field_citation_text(answer, citation, "root_cause") or text
    if root_cause and (
        not root_text
        or not _history_detail_present(root_text, root_cause)
    ):
        missing.add("root_cause")
    raw_steps = fields.get("steps")
    steps = list(raw_steps) if isinstance(raw_steps, (list, tuple)) else [raw_steps]
    operation_values = [
        value
        for value in [fields.get("diagnosis_method"), *steps]
        if value not in (None, "", [], {})
    ]
    operation_text = _history_field_citation_text(answer, citation, "operations") or text
    numbered_text = _history_numbered_citation_text(answer, citation)
    operation_present = _history_detail_present(operation_text, operation_values)
    if numbered_text:
        operation_present = operation_present or _history_detail_present(
            numbered_text,
            operation_values,
            require_source_action=True,
        )
    if operation_values and not operation_present:
        missing.add("operations")
    final_solution = fields.get("final_solution")
    final_text = _history_field_citation_text(answer, citation, "final_solution") or text
    if final_solution and not _history_detail_present(
        final_text,
        final_solution,
        require_source_action=True,
    ) and not _history_numbered_exact_detail_present(
        answer,
        citation,
        final_solution,
    ):
        missing.add("final_solution")
    verification_result = fields.get("verification_result")
    verification_text = _history_field_citation_text(
        answer, citation, "verification_result"
    ) or text
    if verification_result and not _history_detail_present(
        verification_text, verification_result
    ):
        missing.add("verification_result")
    return missing


def _history_public_value(value: Any) -> str:
    """Flatten one authoritative field for compact Markdown without renaming it."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ensure_approved_history_completeness(
    answer: str,
    state: Dict[str, Any],
) -> str:
    """Fail closed while restoring omitted fields from an approved HIS record.

    The answer remains LLM-authored.  This final contract only appends source
    fields that the bounded editor or citation repair omitted.  Every appended
    fact is copied from one gate-approved PostgreSQL ``answer_fields`` payload
    and carries that exact record's citation; it never searches, infers, or
    borrows details from another case.
    """

    current = str(answer or "")
    records, _unsupported_history_tone = _history_review_manifest(current, state)
    completions: List[Tuple[Mapping[str, Any], set[str]]] = []
    for record in records:
        missing = _history_completion_missing_keys(current, record)
        if missing:
            completions.append((record, missing))
    if not completions:
        return current

    lines: List[str] = []
    for record, missing in completions:
        citation = record.get("citation_id")
        try:
            citation_number = int(citation)
        except (TypeError, ValueError):
            continue
        # Re-check the ledger at the point of emission.  This keeps the helper
        # safe if a caller passes a stale manifest or mutates the gate state.
        ledger_item = _citation_ledger(state).get(citation_number)
        if not ledger_item or str(ledger_item.get("doc_type") or "") != "fault_history":
            continue
        fields = (
            record.get("answer_fields")
            if isinstance(record.get("answer_fields"), Mapping)
            else {}
        )
        record_id = _history_public_value(
            fields.get("case_id") or record.get("record_id")
        )
        record_lines: List[str] = []
        if "occurred_at" in missing and fields.get("occurred_at"):
            record_lines.append(
                f"- **发生时间**：{_history_public_value(fields['occurred_at'])}。[^{citation_number}]"
            )
        if "fault_symptom" in missing and fields.get("fault_symptom"):
            record_lines.append(
                f"- **当时现象**：{_history_public_value(fields['fault_symptom'])}[^{citation_number}]"
            )
        if "root_cause" in missing and fields.get("root_cause"):
            record_lines.append(
                f"- **确认根因**：{_history_public_value(fields['root_cause'])}[^{citation_number}]"
            )
        if "operations" in missing:
            diagnosis_method = _history_public_value(fields.get("diagnosis_method"))
            raw_steps = fields.get("steps")
            step_values = (
                list(raw_steps)
                if isinstance(raw_steps, (list, tuple))
                else [raw_steps]
            )
            clean_steps = [
                _history_public_value(value)
                for value in step_values
                if value not in (None, "", [], {})
            ]
            diagnosis_identity = _compact_semantic_text(diagnosis_method)
            diagnosis_repeated_by_step = bool(
                diagnosis_identity
                and any(
                    diagnosis_identity in _compact_semantic_text(step)
                    or _compact_semantic_text(step) in diagnosis_identity
                    for step in clean_steps
                    if _compact_semantic_text(step)
                )
            )
            if diagnosis_method and not diagnosis_repeated_by_step:
                record_lines.append(
                    f"- **排故方法**：{diagnosis_method}"
                    f"[^{citation_number}]"
                )
            if clean_steps:
                record_lines.append("**处理过程**")
                final_solution = _history_public_value(fields.get("final_solution"))
                final_identity = _compact_semantic_text(final_solution)
                record_lines.extend(
                    f"{index}. {value}"
                    + (
                        "（最终方案）"
                        if "final_solution" in missing
                        and final_identity
                        and _compact_semantic_text(value) == final_identity
                        else ""
                    )
                    + f"[^{citation_number}]"
                    for index, value in enumerate(clean_steps, start=1)
                )
        final_in_steps = bool(
            "operations" in missing
            and "final_solution" in missing
            and _compact_semantic_text(fields.get("final_solution"))
            and any(
                _compact_semantic_text(value)
                == _compact_semantic_text(fields.get("final_solution"))
                for value in (
                    fields.get("steps")
                    if isinstance(fields.get("steps"), (list, tuple))
                    else [fields.get("steps")]
                )
            )
        )
        if (
            "final_solution" in missing
            and fields.get("final_solution")
            and not final_in_steps
        ):
            record_lines.append(
                f"- **最终方案**：{_history_public_value(fields['final_solution'])}[^{citation_number}]"
            )
        if "verification_result" in missing and fields.get("verification_result"):
            record_lines.append(
                f"- **验证结果**：{_history_public_value(fields['verification_result'])}[^{citation_number}]"
            )
        if not record_lines:
            continue
        if record_id:
            # The case identifier is itself an authoritative fact.  Bind the
            # subsection label to the same history citation so terminal
            # uncited-assertion cleanup cannot erase the identity while
            # leaving its time and steps behind.
            lines.append(f"#### {record_id} [^{citation_number}]")
        lines.extend(record_lines)

    if not lines:
        return current
    return _insert_into_matching_markdown_section(
        current,
        heading_pattern=re.compile(r"相似历史|相关历史|历史案例|既往案例|案例回顾"),
        default_heading="### 相关历史案例",
        body_lines=lines,
    )


def _remove_redundant_history_append(answer: str) -> str:
    """Drop a terminal history block when the same case is already complete.

    The bounded history editor may write a complete case under a natural
    heading, after which the deterministic field guard can conservatively add
    a second ``#### HIS-*`` block.  Keep the original prose when it already
    contains the case's time, symptom, cause, action and verification fields.
    """

    lines = str(answer or "").splitlines()
    heading_re = re.compile(r"^(\s*)(#{3,6})\s+([A-Za-z][A-Za-z0-9_-]{2,})")
    label_groups = (
        ("发生时间",),
        ("当时现象",),
        ("确认根因", "故障原因", "根因"),
        ("最终方案", "最终处理"),
        ("验证结果", "恢复标准"),
    )
    output: List[str] = []
    index = 0
    while index < len(lines):
        match = heading_re.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        level = len(match.group(2))
        case_id = match.group(3)
        previous_text = "\n".join(output)
        # Only remove an appended case heading if a prior history section is
        # visibly complete; otherwise retain it as a legitimate completion.
        prior_history = previous_text.rsplit("###", 1)[-1]
        complete = all(
            any(label in prior_history for label in group)
            for group in label_groups
        ) and bool(
            re.search(r"排故方法|关键操作|处理过程|操作步骤|验证恢复方法|验证步骤", prior_history)
        )
        if complete and re.search(r"(?:HIS|HISTORY|CASE)[-_]", case_id, re.IGNORECASE):
            index += 1
            while index < len(lines):
                next_match = heading_re.match(lines[index])
                if next_match and len(next_match.group(2)) <= level:
                    break
                index += 1
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output)


def _remove_redundant_action_lines(answer: str) -> str:
    """Remove shorter duplicate action lines while keeping the fuller one."""

    lines = str(answer or "").splitlines()
    candidates: List[Tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        match = re.search(
            r"(?:处置|可采取处置|可选补偿|文档处置|处置建议)\s*[：:]\s*(.+)",
            line,
        )
        if not match:
            continue
        body = re.sub(r"\[\^\d+\]", "", match.group(1)).strip(" 。；;，,")
        normalized = _compact_semantic_text(body)
        if normalized:
            candidates.append((index, normalized, line))
    remove: set[int] = set()
    for index, value, _line in candidates:
        for other_index, other_value, _other_line in candidates:
            if index == other_index or len(other_value) <= len(value):
                continue
            if value in other_value:
                remove.add(index)
                break
    return "\n".join(line for index, line in enumerate(lines) if index not in remove)


def _ensure_explicit_history_target(
    answer: str,
    prompt_citations: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Guarantee an explicit history question has one usable case citation.

    A model can answer the maintenance half of a mixed question and omit the
    requested historical case even though the history tool returned an
    approved PostgreSQL row.  Promote the best approved structured case into
    the prompt ledger and let the existing source-bound completion helper
    render its time, operations, solution and verification fields.
    """

    if not _history_is_primary_target(state):
        return str(answer or ""), [dict(item) for item in prompt_citations]
    existing = [
        item
        for item in prompt_citations
        if str(item.get("doc_type") or "") == "fault_history"
        and str(item.get("evidence_id") or "")
    ]
    if existing:
        return str(answer or ""), [dict(item) for item in prompt_citations]
    decision = state.get("safety_decision") if isinstance(state.get("safety_decision"), Mapping) else {}
    approved_ids = approved_id_set(decision.get("approved_evidence_ids") or [])
    candidates = [
        item
        for item in deduplicate_evidence(state.get("evidence_registry") or [])
        if str(item.get("doc_type") or "") == "fault_history"
        and str(item.get("evidence_id") or "") in approved_ids
        and str(item.get("record_id") or item.get("title") or "").strip()
    ]
    if not candidates:
        return str(answer or ""), [dict(item) for item in prompt_citations]
    candidates.sort(
        key=lambda item: (
            0
            if str(item.get("match_type") or "")
            in {"exact_history_reference", "structured_query"}
            else 1,
            -_answer_relevance_score(item)[0],
            str(item.get("record_id") or item.get("evidence_id") or ""),
        )
    )
    selected = candidates[0]
    extended = [dict(item) for item in prompt_citations]
    payload = _evidence_payload(selected)
    citation_number = len(extended) + 1
    payload["id"] = citation_number
    extended.append(payload)

    prompt_evidence = [
        _prompt_payload_as_evidence(item)
        for item in extended
        if isinstance(item, Mapping)
    ]
    prompt_ids = [
        str(item.get("evidence_id") or "")
        for item in prompt_evidence
        if str(item.get("evidence_id") or "")
    ]
    temp_state = {
        **dict(state),
        "evidence_registry": prompt_evidence,
        "context_evidence_ids": prompt_ids,
        "_terminal_public_ledger": True,
        "safety_decision": {
            **dict(decision),
            "approved_evidence_ids": [
                str(item.get("evidence_id") or "")
                for item in prompt_evidence
                if str(item.get("evidence_id") or "") in approved_ids
            ],
            "context_evidence_ids": prompt_ids,
        },
    }
    seed = f"{str(answer or '').rstrip()}\n\n### 相似历史案例\n引用案例 [^{citation_number}]"
    completed = _ensure_approved_history_completeness(seed, temp_state)
    completed = re.sub(
        rf"(?m)^\s*引用案例\s*\[\^{citation_number}\]\s*$\n?",
        "",
        completed,
    )
    return _normalize_generated_markdown(completed), extended


def _history_only_answer_request(state: Mapping[str, Any]) -> bool:
    """Whether history is the sole information dimension for this turn."""

    intents = {
        str(state.get("intent") or "").strip(),
        *{
            str(target.get("intent") or "").strip()
            for target in (state.get("query_targets") or [])
            if isinstance(target, Mapping)
        },
    }
    intents.discard("")
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    return bool(
        "history_query" in intents
        and intents <= {"history_query"}
        and not (
            isinstance(frame.get("diagnostic_closure"), Mapping)
            and frame["diagnostic_closure"].get("required") is True
        )
    )


def _render_history_only_answer(
    answer: str,
    state: Mapping[str, Any],
    prompt_citations: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render a readable, bounded history answer from approved records."""

    current = str(answer or "").strip()
    if not _history_only_answer_request(state):
        return current
    ledger = {
        index: dict(item)
        for index, item in enumerate(
            prompt_citations if prompt_citations is not None else _context_evidence(dict(state)),
            start=1,
        )
        if isinstance(item, Mapping)
    }
    evidence = list(ledger.values())
    candidates: List[Tuple[int, Mapping[str, Any]]] = []
    for citation, item in ledger.items():
        if str(item.get("doc_type") or "") != "fault_history":
            continue
        if not _history_subject_matches_state(item, state, evidence):
            continue
        if not _evidence_answer_fields(item):
            continue
        candidates.append((citation, item))
    if not candidates:
        return current
    if not _history_collection_requested(state):
        candidates = candidates[:2]

    body: List[str] = [f"### 相近历史案例（共 {len(candidates)} 个）"]
    body.append("以下内容均来自历史记录；用于参考排故过程，不等同于当前故障已确认。")
    field_specs = (
        ("发生时间", "occurred_at"),
        ("当时现象", "fault_symptom"),
        ("关联模块", "related_module"),
        ("确认根因", "root_cause"),
        ("排故方法", "diagnosis_method"),
    )
    for citation, item in candidates:
        fields = _evidence_answer_fields(item)
        case_id = _history_public_value(
            fields.get("case_id") or item.get("record_id") or item.get("title")
        )
        marker = f"[^{citation}]"
        body.extend(("", f"#### {case_id}{marker}" if case_id else f"#### 历史案例{marker}"))
        for label, key in field_specs:
            value = _history_public_value(fields.get(key))
            if value:
                body.append(f"- {label}：{value}{marker}")
        raw_steps = fields.get("steps")
        steps = (
            [value for value in raw_steps if value not in (None, "", [], {})]
            if isinstance(raw_steps, (list, tuple))
            else ([raw_steps] if raw_steps not in (None, "", [], {}) else [])
        )
        if steps:
            body.append(f"- 处理步骤：{marker}")
            body.extend(
                f"  {index}. {_history_public_value(value)}{marker}"
                for index, value in enumerate(steps, start=1)
            )
        for label, key in (("最终方案", "final_solution"), ("验证结果", "verification_result")):
            value = _history_public_value(fields.get(key))
            if value:
                body.append(f"- {label}：{value}{marker}")
    return _normalize_generated_markdown("\n".join(body))


