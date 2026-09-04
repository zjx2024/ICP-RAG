"""Answer 节点·故障域策略模块（C1 批 3 从 answer.py 抽出）。

包含：故障候选适配、平行故障模式分支、编号故障候选完整性、
当前故障（current incident）动作补全等确定性策略函数。

依赖约定：共享 helper 与常量仍留在 answer.py（内核）；本模块顶部
``from backend.agent.nodes.answer import ...`` 只允许在 answer.py 已
完整加载后执行，因此 answer.py 必须在文件**底部** re-export 本模块
（answer -> 本模块 -> answer 顺序导入，避免循环导入）。
"""

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.agent.answer_contract import focused_answer_contract
from backend.agent.contracts import deduplicate_evidence
from backend.agent.nodes.answer_presence import approved_id_set
from backend.agent.nodes.answer import (
    _answer_query_text,
    _answer_relevance_score,
    _citation_ledger,
    _cited_answer_line_supports_field,
    _cleanup_generated_markdown_structure,
    _compact_semantic_text,
    _diagnostic_closure_required,
    _evidence_answer_fields,
    _evidence_payload,
    _history_section_context,
    _insert_into_matching_markdown_section,
    _normalize_generated_markdown,
    _section_title,
    _target_value_is_present,
    _unit_requires_fault_history,
    _SECTION_HEADING,
    _ACTION_CLAIM_RE,
    _DIRECT_EVIDENCE_AUTHORITIES,
    _MULTI_CANDIDATE_FRAMING,
)


def _fault_candidate_fit(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str:
    """Compare planner-normalized constraints with one structured FMECA row.

    The planner remains responsible for understanding free-form user language.
    This deterministic layer only compares its closed ``query_frame`` slots
    with the source's own structured phase/impact fields.  Missing source data
    stays ``unknown``; only explicit conflicts are rejected.
    """

    if str(item.get("doc_type") or "") != "fmeca":
        return "unknown"
    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    query_stage = str(frame.get("operating_stage") or "").strip()
    query_temporal = str(frame.get("temporal_pattern") or "").strip()
    fields = _evidence_answer_fields(item)
    mission_phase = _compact_semantic_text(fields.get("mission_phase"))
    effect_text = _compact_semantic_text(
        " ".join(
            str(fields.get(key) or "")
            for key in ("failure_mode", "local_impact", "final_impact")
        )
    )

    stage_fit = "unknown"
    if query_stage and query_stage != "未知" and mission_phase:
        direct_terms: Dict[str, Tuple[str, ...]] = {
            "上电": ("上电", "加电", "启动"),
            "初始化": ("初始化", "自检"),
            "运行中": ("正常运行", "运行中", "正常工作"),
            "高负载": ("高负载", "满负荷", "大负载", "高通信负载"),
            "持续运行后": ("持续运行后", "运行一段时间", "工作一段时间"),
        }
        conditional_terms: Dict[str, Tuple[str, ...]] = {
            "运行中": (
                "高负载",
                "满负荷",
                "大负载",
                "高通信负载",
                "持续运行后",
                "运行一段时间",
                "配置更新",
                "同步切换",
            ),
            "高负载": ("正常运行", "运行中", "正常工作"),
            "持续运行后": ("正常运行", "运行中", "正常工作"),
        }
        contradictory_terms: Dict[str, Tuple[str, ...]] = {
            "上电": ("正常运行", "运行中", "正常工作"),
            "初始化": ("正常运行", "运行中", "正常工作"),
            "运行中": ("上电", "加电", "初始化", "自检", "启动"),
            "高负载": ("上电", "加电", "初始化", "自检", "启动"),
            "持续运行后": ("上电", "加电", "初始化", "自检", "启动"),
        }
        if any(term in mission_phase for term in direct_terms.get(query_stage, ())):
            stage_fit = "direct"
        elif any(
            term in mission_phase
            for term in conditional_terms.get(query_stage, ())
        ):
            stage_fit = "conditional"
        elif any(
            term in mission_phase
            for term in contradictory_terms.get(query_stage, ())
        ):
            stage_fit = "contradictory"

    temporal_fit = "unknown"
    intermittent_terms = ("间歇", "偶发", "时有时无", "不定期", "突发", "抖动")
    repeated_terms = ("反复", "重复", "循环", "多次")
    persistent_terms = ("持续", "长期", "一直", "始终")
    if query_temporal == "持续发生" and effect_text:
        if any(term in effect_text for term in intermittent_terms):
            temporal_fit = "contradictory"
        elif any(term in effect_text for term in persistent_terms):
            temporal_fit = "direct"
    elif query_temporal == "间歇发生" and effect_text:
        if any(term in effect_text for term in intermittent_terms):
            temporal_fit = "direct"
    elif query_temporal == "重复发生" and effect_text:
        if any(term in effect_text for term in repeated_terms):
            temporal_fit = "direct"

    fits = {stage_fit, temporal_fit}
    if "contradictory" in fits:
        return "contradictory"
    if "conditional" in fits:
        return "conditional"
    if "direct" in fits:
        return "direct"
    return "unknown"


# ---- 故障域专属常量（随本模块迁移，answer.py 不再引用）----
_FAULT_COMPLETENESS_INTENTS = {
    "symptom_diagnosis",
    "cause_analysis",
    "impact_analysis",
    "maintenance_query",
    "exact_code",
}


def _answer_has_fault_facet(answer: str, facet: str) -> bool:
    substantive_lines: List[str] = []
    for raw_line in str(answer or "").splitlines():
        if _SECTION_HEADING.match(raw_line):
            continue
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s*", "", raw_line).strip()
        line = re.sub(r"\[\^\d+\]", "", line).strip()
        # A bare mode label such as ``**A/D 采样饱和**`` carries no cause or
        # impact even when the preceding section title contains those words.
        if re.fullmatch(r"\*\*[^*]+\*\*\s*[:：]?", line):
            continue
        if line:
            substantive_lines.append(line)
    patterns = {
        "cause": (
            r"原因|根因|故障机理|机制在于|由于|源于",
            r"(?:不足|拥塞|老化|退化|损坏|失效|异常|错误|松动|开路|短路).{0,45}(?:会|可)?(?:直接)?(?:引起|导致|造成)",
        ),
        "impact": (
            r"局部影响|最终影响|影响|后果",
            r"(?:导致|造成|引起).{0,90}(?:中断|异常|失效|降级|丧失|不可用|缺失|延迟|错误|无法)",
        ),
        "resolution": (
            r"检查|排查|处置|处理(?:方法|措施|步骤|建议|方案)|维修(?:方法|措施|步骤|建议|方案)|修复|更换|校准|锁止|压接|回滚|切换|限制流量|调用.{0,30}API",
        ),
    }
    # Match within one factual line.  Joining the whole answer used to let a
    # cause phrase on one bullet and an unrelated word such as ``异常`` on a
    # later bullet accidentally satisfy the impact contract.
    return any(
        re.search(pattern, line, re.IGNORECASE)
        for line in substantive_lines
        for pattern in patterns.get(facet, ())
    )


def _nonhistory_fault_text(answer: str, state: Dict[str, Any]) -> str:
    """Exclude history-only lines when checking the current fault closure.

    A well-documented old case may contain a root cause and repair, but that
    does not answer “what should I suspect now?” by itself.  History remains in
    the final answer; it simply cannot make the current cause/impact/solution
    checklist appear complete.
    """

    ledger = _citation_ledger(state)
    lines: List[str] = []
    for line in str(answer or "").splitlines():
        numbers = {
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", line)
            if int(value) in ledger
        }
        if numbers and all(
            str(ledger[number].get("doc_type") or "") == "fault_history"
            for number in numbers
        ):
            continue
        # The history row may have been intentionally removed from the answer
        # context as an unrelated semantic neighbor, leaving an old marker
        # like ``[^3]`` unresolved.  Historical wording must still not satisfy
        # the current-incident cause/impact checklist in that case.
        if not numbers and _unit_requires_fault_history(line):
            continue
        lines.append(line)
    return "\n".join(lines)


_FAULT_MODE_FIELD_KEYS = (
    "failure_mode",
    "module_failure",
    "sur_failure",
    "subcircuit_failure",
)


_FAULT_CAUSE_FIELD_KEYS = ("fault_cause", "root_cause")


_FAULT_IMPACT_FIELD_KEYS = ("local_impact", "final_impact")


_FAULT_DISCRIMINATION_FIELD_KEYS = (
    "detection_method",
    "detect_method",
    "api_usage",
    "sur_failure",
    "subcircuit_failure",
)


_FAULT_ACTION_FIELD_KEYS = (
    "compensation_action",
    "maintenance_action",
    "repair_method",
    "final_solution",
)


def _query_requests_parallel_failure_modes(state: Mapping[str, Any]) -> bool:
    """Whether the user explicitly asks to distinguish multiple branches.

    This is a completeness-contract trigger, not a fault classifier.  The
    actual branch identities still come exclusively from approved structured
    FMECA/MFL fields below.
    """

    # "Explicitly asks" must be decided from the user's original wording.
    # Planner rewrites may mention that several candidates should be compared
    # as an internal retrieval tactic; treating that as user authorization
    # would wrongly switch broad path questions into exhaustive mode-by-mode
    # presentation and would also bypass compact candidate framing.
    text = re.sub(r"\s+", "", str(state.get("query") or ""))
    if not text:
        return False
    comparison = bool(
        re.search(r"区分|辨别|鉴别|分清|判别|对比|比较", text, re.IGNORECASE)
        or re.search(r"(?:到底|究竟|更像)?是?.{2,80}还是.{2,80}", text)
        or re.search(r"\bvs\.?\b", text, re.IGNORECASE)
    )
    multiple = bool(
        re.search(r"还是|两种|多个|不同(?:故障|模式|候选)|各自|分别", text)
        or re.search(r"(?:故障|异常|问题|模式|候选).{0,35}(?:和|与|或|、).{0,35}(?:故障|异常|问题|模式|候选)", text)
    )
    return comparison and multiple


def _fault_candidate_mode_text(candidate: Mapping[str, Any]) -> str:
    fields = (
        candidate.get("answer_fields")
        if isinstance(candidate.get("answer_fields"), Mapping)
        else {}
    )
    return "\n".join(
        str(fields.get(key) or "").strip()
        for key in _FAULT_MODE_FIELD_KEYS
        if str(fields.get(key) or "").strip()
    )


def _fault_mode_pair_matches(fmeca_mode: str, maintenance_mode: str) -> bool:
    """Pair records only when one normalized stored mode contains the other."""

    fmeca_key = _compact_semantic_text(fmeca_mode)
    maintenance_key = _compact_semantic_text(maintenance_mode)
    return bool(
        len(fmeca_key) >= 4
        and len(maintenance_key) >= 4
        and (fmeca_key in maintenance_key or maintenance_key in fmeca_key)
    )


def _parallel_failure_mode_branches(
    candidates: Sequence[Dict[str, Any]],
    state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build deterministic FMECA/MFL mode pairs for an explicit comparison."""

    if not _query_requests_parallel_failure_modes(state):
        return []
    query_text = _compact_semantic_text(_answer_query_text(dict(state)))
    fmeca = [
        item
        for item in candidates
        if item.get("doc_type") == "fmeca"
        and str(
            (
                item.get("answer_fields")
                if isinstance(item.get("answer_fields"), Mapping)
                else {}
            ).get("failure_mode")
            or ""
        ).strip()
    ]
    maintenance = [item for item in candidates if item.get("doc_type") == "mfl"]
    branches: List[Dict[str, Any]] = []
    seen_modes: set[str] = set()
    for fmeca_item in fmeca:
        fields = fmeca_item.get("answer_fields") or {}
        mode = str(fields.get("failure_mode") or "").strip()
        mode_key = _compact_semantic_text(mode)
        # Exact graph/record expansions can be valid context without being one
        # of the alternatives the user asked to compare.  Keep such a row only
        # when the query itself names its mode.
        if (
            str(fmeca_item.get("match_type") or "").startswith("exact_")
            and mode_key not in query_text
        ):
            continue
        matched_mfl = [
            item
            for item in maintenance
            if _fault_mode_pair_matches(mode, _fault_candidate_mode_text(item))
        ]
        if not matched_mfl or not mode_key or mode_key in seen_modes:
            continue
        seen_modes.add(mode_key)
        branches.append(
            {
                "mode": mode,
                "fmeca": fmeca_item,
                "maintenance": matched_mfl,
                "items": [fmeca_item, *matched_mfl],
            }
        )
    return branches if len(branches) >= 2 else []


def _branch_mode_is_present(
    answer: str,
    mode: str,
    items: Sequence[Mapping[str, Any]],
) -> bool:
    for item in items:
        marker = f"[^{int(item.get('citation_id') or 0)}]"
        for line in str(answer or "").splitlines():
            if marker in line and _target_value_is_present(line, mode):
                return True
    return False


def _branch_fields_are_present(
    answer: str,
    items: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    require_all: bool,
) -> Tuple[bool, bool]:
    """Return ``(supported, present)`` for fields bound to branch citations."""

    checks: List[bool] = []
    for item in items:
        fields = (
            item.get("answer_fields")
            if isinstance(item.get("answer_fields"), Mapping)
            else {}
        )
        citation_id = int(item.get("citation_id") or 0)
        for key in keys:
            value = fields.get(key)
            if value in (None, "", [], {}):
                continue
            checks.append(
                _cited_answer_line_supports_field(answer, citation_id, value)
            )
    if not checks:
        return False, True
    return True, (all(checks) if require_all else any(checks))


_NUMBERED_FAULT_BRANCH_RE = re.compile(
    r"^(?P<number>\d{1,2})[.、)]\s+(?P<body>\S.*)$"
)


_TOP_LEVEL_BULLET_RE = re.compile(r"^[-+*]\s+")


def _numbered_fault_candidate_blocks(
    answer: str,
    state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return top-level numbered candidate blocks bound to approved FMECA rows.

    The answer model is free to choose readable group names, so an exact failure
    mode string is not required in the heading.  A top-level number becomes a
    candidate branch only when its own heading cites an approved FMECA record;
    this avoids mistaking numbered maintenance steps or history steps for fault
    alternatives.
    """

    lines = str(answer or "").splitlines()
    ledger = _citation_ledger(dict(state))
    fmeca_numbers = {
        number
        for number, item in ledger.items()
        if str(item.get("doc_type") or "") == "fmeca"
    }
    starts: List[int] = []
    for index, line in enumerate(lines):
        match = _NUMBERED_FAULT_BRANCH_RE.match(line)
        if not match:
            continue
        cited = {int(value) for value in re.findall(r"\[\^(\d+)\]", line)}
        if cited & fmeca_numbers:
            starts.append(index)
    if len(starts) < 2:
        return []

    blocks: List[Dict[str, Any]] = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(lines)
        # A model sometimes writes one detached, top-level "总线层/网络层" action
        # list after all numbered candidates.  It is not part of the final
        # numbered branch and must not make that branch look complete.
        for index in range(start + 1, end):
            if _TOP_LEVEL_BULLET_RE.match(lines[index]) or re.match(
                r"^#{1,6}\s+", lines[index]
            ):
                end = index
                break
        block_lines = lines[start:end]
        citation_numbers = list(
            dict.fromkeys(
                int(value)
                for line in block_lines
                for value in re.findall(r"\[\^(\d+)\]", line)
                if int(value) in fmeca_numbers
            )
        )
        if not citation_numbers:
            continue
        heading = _NUMBERED_FAULT_BRANCH_RE.match(lines[start])
        heading_body = str(heading.group("body") if heading else lines[start])
        bold_title = re.search(r"\*\*(.+?)\*\*", heading_body)
        title = str(bold_title.group(1) if bold_title else heading_body)
        title = re.sub(r"\[\^\d+\]", "", title).strip(" ：:")
        sources = [
            {
                "citation_id": number,
                "item": ledger[number],
                "answer_fields": _evidence_answer_fields(ledger[number]),
            }
            for number in citation_numbers
            if number in ledger
        ]
        blocks.append(
            {
                "start": start,
                "end": end,
                "title": title,
                "text": "\n".join(block_lines),
                "citation_ids": citation_numbers,
                "sources": sources,
            }
        )
    return blocks if len(blocks) >= 2 else []


def _numbered_fault_branch_missing_facets(
    block: Mapping[str, Any],
) -> List[str]:
    """Check cause, impact and resolution inside one displayed branch only."""

    text = str(block.get("text") or "")
    sources = [
        source
        for source in (block.get("sources") or [])
        if isinstance(source, Mapping)
    ]
    facet_keys = {
        "cause": _FAULT_CAUSE_FIELD_KEYS,
        "impact": _FAULT_IMPACT_FIELD_KEYS,
        "resolution": (
            *_FAULT_DISCRIMINATION_FIELD_KEYS,
            *_FAULT_ACTION_FIELD_KEYS,
        ),
    }
    missing: List[str] = []
    for facet, keys in facet_keys.items():
        supported = False
        present = False
        for source in sources:
            citation = int(source.get("citation_id") or 0)
            fields = (
                source.get("answer_fields")
                if isinstance(source.get("answer_fields"), Mapping)
                else {}
            )
            for key in keys:
                value = fields.get(key)
                if value in (None, "", [], {}):
                    continue
                supported = True
                if _cited_answer_line_supports_field(text, citation, value):
                    present = True
        if supported and not present:
            missing.append(facet)
    return missing


def _fault_completion_manifest(
    answer: str,
    state: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        # Complete inventories have an independent N-of-N contract in the
        # target/collection guard.  Treating their rows as ranked diagnostic
        # candidates activates the 1-to-3 candidate editor and conflicts with
        # the user's request to see every authoritative member.
        return [], []
    if focused_answer_contract(state):
        # A focused follow-up is closed by its typed answer fields.  Reusing
        # the broad diagnostic checklist here would turn “什么级别” into a new
        # cause/impact/repair/history report and bury the requested value.
        return [], []
    intents = {
        str(state.get("intent") or ""),
        *(
            str(item.get("intent") or "")
            for item in (state.get("query_targets") or [])
            if isinstance(item, Mapping)
        ),
    }
    if not (intents & _FAULT_COMPLETENESS_INTENTS):
        return [], []

    candidates: List[Dict[str, Any]] = []
    support = {"cause": False, "impact": False, "resolution": False, "action": False}
    for number, item in _citation_ledger(state).items():
        doc_type = str(item.get("doc_type") or "")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        fields = metadata.get("answer_fields") if isinstance(metadata.get("answer_fields"), Mapping) else {}
        cause_value = fields.get("fault_cause") or fields.get("root_cause")
        impact_value = fields.get("local_impact") or fields.get("final_impact")
        resolution_value = (
            fields.get("maintenance_action")
            or fields.get("repair_method")
            or fields.get("detection_method")
            or fields.get("detect_method")
            or fields.get("compensation_action")
            or fields.get("final_solution")
        )
        claim_type = str(item.get("claim_type") or "")
        supports_cause = doc_type in {"fmeca", "fta"} and bool(
            cause_value or claim_type in {"fta_structure", "fault_cause", "cause_fact"}
        )
        supports_impact = doc_type == "fmeca" and bool(
            impact_value or claim_type in {"failure_impact", "impact_fact"}
        )
        supports_resolution = (
            doc_type in {"hmc", "mfl", "fmeca"}
            and (
                bool(resolution_value)
                or claim_type
                in {"maintenance_action", "maintenance_locator", "hmc_maintenance"}
            )
        )
        # A check/detection field is useful, but it is not the same thing as
        # a documented handling action.  Current-incident answers need that
        # distinction so “monitor the status word” cannot incorrectly close
        # the repair part of the answer.
        supports_action = doc_type in {"hmc", "mfl", "fmeca"} and bool(
            fields.get("maintenance_action")
            or fields.get("repair_method")
            or fields.get("compensation_action")
            or fields.get("final_solution")
        )
        if not (supports_cause or supports_impact or supports_resolution or supports_action):
            continue
        support["cause"] = support["cause"] or supports_cause
        support["impact"] = support["impact"] or supports_impact
        support["resolution"] = support["resolution"] or supports_resolution
        support["action"] = support["action"] or supports_action
        candidates.append(
            {
                "citation_id": number,
                "record_id": str(item.get("record_id") or item.get("title") or ""),
                "doc_type": doc_type,
                "match_type": str(item.get("match_type") or ""),
                "supports": [
                    name
                    for name, enabled in (
                        ("cause", supports_cause),
                        ("impact", supports_impact),
                        ("resolution", supports_resolution),
                        ("action", supports_action),
                    )
                    if enabled
                ],
                "answer_fields": {
                    key: value
                    for key, value in fields.items()
                    if key
                    in {
                        "failure_mode",
                        "module_failure",
                        "sur_failure",
                        "subcircuit_failure",
                        "fault_cause",
                        "root_cause",
                        "mission_phase",
                        "local_impact",
                        "final_impact",
                        "detection_method",
                        "detect_method",
                        "compensation_action",
                        "maintenance_action",
                        "repair_method",
                        "final_solution",
                        "api_name",
                        "api_method",
                        "api_usage",
                    }
                    and value not in (None, "", [], {})
                },
                "source_summary": str(
                    item.get("display_summary") or item.get("content") or ""
                )[:1400],
                "missing_supports": [],
            }
        )
    facet_text = _nonhistory_fault_text(answer, state)
    missing = [
        facet
        for facet in ("cause", "impact", "resolution")
        if support[facet] and not _answer_has_fault_facet(facet_text, facet)
    ]
    if _diagnostic_closure_required(state) and support["action"]:
        action_present = False
        for candidate in candidates:
            if "action" not in set(candidate.get("supports") or []):
                continue
            citation = int(candidate.get("citation_id") or 0)
            fields = (
                candidate.get("answer_fields")
                if isinstance(candidate.get("answer_fields"), Mapping)
                else {}
            )
            if citation and any(
                _cited_answer_line_supports_field(answer, citation, fields.get(key))
                for key in _FAULT_ACTION_FIELD_KEYS
                if fields.get(key) not in (None, "", [], {})
            ):
                action_present = True
                break
        if not action_present:
            missing.append("action")

    parallel_branches = _parallel_failure_mode_branches(candidates, state)
    if parallel_branches:
        # The comparison contract is evidence-led: once two approved FMECA
        # modes each have a deterministic MFL mode match, every branch is
        # checked even when the first draft omitted its canonical name.
        for branch in parallel_branches:
            mode = str(branch.get("mode") or "").strip()
            fmeca_items = list(branch.get("fmeca") and [branch["fmeca"]] or [])
            branch_items = list(branch.get("items") or [])
            mode_present = _branch_mode_is_present(answer, mode, branch_items)
            cause_supported, cause_present = _branch_fields_are_present(
                answer,
                fmeca_items,
                _FAULT_CAUSE_FIELD_KEYS,
                require_all=False,
            )
            impact_supported, impact_present = _branch_fields_are_present(
                answer,
                fmeca_items,
                _FAULT_IMPACT_FIELD_KEYS,
                require_all=True,
            )
            discrimination_supported, discrimination_present = (
                _branch_fields_are_present(
                    answer,
                    branch_items,
                    _FAULT_DISCRIMINATION_FIELD_KEYS,
                    require_all=False,
                )
            )
            action_supported, action_present = _branch_fields_are_present(
                answer,
                branch_items,
                _FAULT_ACTION_FIELD_KEYS,
                require_all=False,
            )
            resolution_supported = discrimination_supported or action_supported
            resolution_present = (
                (not discrimination_supported or discrimination_present)
                and (not action_supported or action_present)
            )
            branch_status = (
                ("cause", cause_supported, cause_present),
                ("impact", impact_supported, impact_present),
                ("resolution", resolution_supported, resolution_present),
            )
            branch_missing = [
                facet
                for facet, supported, present in branch_status
                if supported and (not mode_present or not present)
            ]
            for item in branch_items:
                item["branch_mode"] = mode
                item["missing_supports"] = [
                    facet
                    for facet in branch_missing
                    if facet in set(item.get("supports") or [])
                ]
            for facet in branch_missing:
                if facet not in missing:
                    missing.append(facet)
    else:
        # Outside an explicit comparison, keep the narrower legacy behavior:
        # only modes named by the user or answer receive a per-mode contract.
        named_text = _compact_semantic_text(
            "\n".join((str(state.get("query") or ""), str(answer or "")))
        )
        for candidate in candidates:
            if candidate.get("doc_type") != "fmeca":
                continue
            fields = candidate.get("answer_fields") if isinstance(candidate.get("answer_fields"), Mapping) else {}
            mode = str(fields.get("failure_mode") or candidate.get("record_id") or "").strip()
            if not mode or _compact_semantic_text(mode) not in named_text:
                continue
            citation_id = int(candidate.get("citation_id") or 0)
            mode_missing: List[str] = []
            cause_value = fields.get("fault_cause") or fields.get("root_cause")
            if cause_value and not _cited_answer_line_supports_field(
                answer, citation_id, cause_value
            ):
                mode_missing.append("cause")
            impact_values = [
                fields.get("local_impact"),
                fields.get("final_impact"),
            ]
            if any(impact_values) and any(
                not _cited_answer_line_supports_field(answer, citation_id, value)
                for value in impact_values
                if value
            ):
                mode_missing.append("impact")
            candidate["missing_supports"] = mode_missing
            for facet in mode_missing:
                if facet not in missing:
                    missing.append(facet)
        # A broad path answer can be globally complete while one numbered
        # candidate is only a bare heading and another branch owns all checks.
        # Validate each displayed branch locally so the existing bounded editor
        # gets a chance to move/fill facts before terminal rendering.
        candidates_by_citation = {
            int(item.get("citation_id") or 0): item for item in candidates
        }
        for block in _numbered_fault_candidate_blocks(answer, state):
            branch_missing = _numbered_fault_branch_missing_facets(block)
            if not branch_missing:
                continue
            for citation in block.get("citation_ids") or []:
                candidate = candidates_by_citation.get(int(citation))
                if not candidate:
                    continue
                candidate["branch_label"] = str(block.get("title") or "")
                candidate["missing_supports"] = list(
                    dict.fromkeys(
                        [
                            *(candidate.get("missing_supports") or []),
                            *(
                                facet
                                for facet in branch_missing
                                if facet in set(candidate.get("supports") or [])
                            ),
                        ]
                    )
                )
            for facet in branch_missing:
                if facet not in missing:
                    missing.append(facet)
    if not missing:
        return [], []
    if parallel_branches:
        useful = [item for item in candidates if item.get("branch_mode")][:12]
    else:
        branch_useful = [item for item in candidates if item.get("branch_label")]
        other_useful = [
            item
            for item in candidates
            if not item.get("branch_label")
            if set(item.get("missing_supports") or item["supports"]) & set(missing)
        ]
        useful = [*branch_useful, *other_useful][:12]
    return missing, useful


def _format_fault_completion_manifest(
    manifest: Sequence[Mapping[str, Any]],
) -> str:
    """Render editor input as prose so a small model cannot echo raw JSON."""

    lines: List[str] = []
    for item in manifest:
        citation = item.get("citation_id")
        branch_mode = str(item.get("branch_mode") or "").strip()
        branch_label = str(item.get("branch_label") or "").strip()
        lines.append(
            f"- 证据 [^{citation}]｜{item.get('doc_type') or 'unknown'}｜"
            f"记录 {item.get('record_id') or '未标识'}｜可支持 "
            + "、".join(str(value) for value in (item.get("supports") or []))
            + (f"｜对比分支 {branch_mode}" if branch_mode else "")
            + (f"｜候选分支 {branch_label}" if branch_label else "")
            + (
                "｜本条仍缺 "
                + "、".join(str(value) for value in (item.get("missing_supports") or []))
                if item.get("missing_supports")
                else ""
            )
        )
        fields = item.get("answer_fields") if isinstance(item.get("answer_fields"), Mapping) else {}
        for key, value in fields.items():
            lines.append(f"  - {key}：{value}")
        if not fields and item.get("source_summary"):
            lines.append(f"  - 来源摘要：{item.get('source_summary')}")
    return "\n".join(lines) or "无"


def _fault_public_scalar(value: Any) -> str:
    """Return one source scalar without exposing an internal field payload."""

    if isinstance(value, (Mapping, list, tuple, set)):
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _append_missing_parallel_failure_mode_branches(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> str:
    """Close an explicit multi-mode comparison from approved source fields.

    The LLM remains responsible for the normal prose.  This terminal fallback
    is used only after two independently approved FMECA modes can each be
    paired with an MFL mode by the deterministic identity rule above.  Every
    emitted value is copied from its own citation, so one mode can never borrow
    another mode's cause, impact, discriminator, or repair action.
    """

    candidates = [dict(item) for item in manifest if isinstance(item, Mapping)]
    branches = _parallel_failure_mode_branches(candidates, state)
    if len(branches) < 2:
        return str(answer or "")

    current = str(answer or "")
    ledger = _citation_ledger(dict(state))
    branch_sections: List[str] = []

    def approved_item(
        item: Mapping[str, Any], expected_doc_type: str
    ) -> Tuple[int, Mapping[str, Any]]:
        try:
            citation = int(item.get("citation_id") or 0)
        except (TypeError, ValueError):
            return 0, {}
        ledger_item = ledger.get(citation)
        if not ledger_item or str(ledger_item.get("doc_type") or "") != expected_doc_type:
            return 0, {}
        return citation, _evidence_answer_fields(ledger_item)

    def first_scalar(
        sources: Sequence[Tuple[int, Mapping[str, Any]]],
        keys: Sequence[str],
    ) -> Tuple[int, str]:
        for citation, fields in sources:
            for key in keys:
                value = _fault_public_scalar(fields.get(key))
                if value:
                    return citation, value
        return 0, ""

    for branch in branches:
        fmeca_candidate = branch.get("fmeca")
        if not isinstance(fmeca_candidate, Mapping):
            continue
        fmeca_citation, fmeca_fields = approved_item(fmeca_candidate, "fmeca")
        mode = _fault_public_scalar(fmeca_fields.get("failure_mode"))
        if not fmeca_citation or not mode:
            continue

        maintenance_sources = [
            approved_item(item, "mfl")
            for item in (branch.get("maintenance") or [])
            if isinstance(item, Mapping)
        ]
        maintenance_sources = [
            (citation, fields)
            for citation, fields in maintenance_sources
            if citation and fields
        ]
        # Re-validate the pairing against the authoritative ledger fields at
        # emission time rather than trusting a potentially stale manifest.
        maintenance_sources = [
            (citation, fields)
            for citation, fields in maintenance_sources
            if _fault_mode_pair_matches(
                mode,
                "\n".join(
                    _fault_public_scalar(fields.get(key))
                    for key in _FAULT_MODE_FIELD_KEYS
                    if _fault_public_scalar(fields.get(key))
                ),
            )
        ]
        if not maintenance_sources:
            continue

        facts: List[Tuple[str, int, str]] = []

        def add_missing(label: str, citation: int, value: str) -> None:
            if not citation or not value:
                return
            if _cited_answer_line_supports_field(current, citation, value):
                return
            compact = _compact_semantic_text(value)
            if any(
                existing_citation == citation
                and _compact_semantic_text(existing_value) == compact
                for _label, existing_citation, existing_value in facts
            ):
                return
            facts.append((label, citation, value))

        cause_citation, cause = first_scalar(
            [(fmeca_citation, fmeca_fields)], _FAULT_CAUSE_FIELD_KEYS
        )
        add_missing("原因", cause_citation, cause)
        add_missing(
            "局部影响",
            fmeca_citation,
            _fault_public_scalar(fmeca_fields.get("local_impact")),
        )
        add_missing(
            "最终影响",
            fmeca_citation,
            _fault_public_scalar(fmeca_fields.get("final_impact")),
        )

        detection_sources = [(fmeca_citation, fmeca_fields), *maintenance_sources]
        detection_citation, detection = first_scalar(
            detection_sources,
            ("detection_method", "detect_method", "api_usage"),
        )
        if not detection:
            detection_citation, detection = first_scalar(
                maintenance_sources,
                ("sur_failure", "subcircuit_failure"),
            )
        add_missing("检测/判别", detection_citation, detection)

        action_citation, action = first_scalar(
            maintenance_sources,
            ("maintenance_action", "repair_method", "final_solution"),
        )
        add_missing("文档处置", action_citation, action)

        mode_present = _branch_mode_is_present(current, mode, branch.get("items") or [])
        if not facts and mode_present:
            continue
        section = [f"- **故障模式：{mode}**[^{fmeca_citation}]"]
        section.extend(
            f"  - **{label}**：{value}[^{citation}]"
            for label, citation, value in facts
        )
        branch_sections.append("\n".join(section))

    if not branch_sections:
        return current
    supplement = "### 候选故障模式的区分与处理\n\n" + "\n\n".join(branch_sections)
    return _normalize_generated_markdown(f"{current.rstrip()}\n\n{supplement}")


def _append_missing_anchored_single_fault_details(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Close one already-cited FMECA mode without promoting nearby recalls.

    A broad retrieval commonly returns adjacent FMECA rows from the same
    subsystem.  Only a row whose *own citation* is already attached to its
    failure mode or fault cause is considered selected by the answer.  This
    lets the terminal fallback restore that row's omitted impact or handling
    fields while preventing a semantically nearby row from silently becoming
    another diagnosis candidate.

    MFL actions use the same conservative identity rule as the parallel-mode
    contract: the normalized stored failure-mode strings must contain one
    another.  No vector/name similarity is used at this stage.
    """

    intents = {
        str(state.get("intent") or ""),
        *(
            str(item.get("intent") or "")
            for item in (state.get("query_targets") or [])
            if isinstance(item, Mapping)
        ),
    }
    if not (intents & _FAULT_COMPLETENESS_INTENTS):
        return str(answer or "")
    if _query_requests_parallel_failure_modes(state):
        return str(answer or "")

    current = str(answer or "")
    ledger = _citation_ledger(dict(state))
    # A cited history row may bind the current symptom to one exact FMECA
    # business ID.  That explicit cross-document key is stronger than text
    # similarity and is therefore a safe single-mode anchor.  The history row
    # must already be cited; its exact business-ID link may restore the
    # corresponding approved FMECA row even when citation repair dropped that
    # row from the draft.  Nearby recalls without that exact link stay unused.
    history_fmeca_ids: set[str] = set()
    for history_citation, history_item in ledger.items():
        if str(history_item.get("doc_type") or "") != "fault_history":
            continue
        if f"[^{history_citation}]" not in current:
            continue
        history_fields = _evidence_answer_fields(history_item)
        references = history_fields.get("explicit_references")
        if not isinstance(references, Mapping):
            continue
        raw_references = references.get("fmeca")
        reference_values = (
            list(raw_references)
            if isinstance(raw_references, (list, tuple, set))
            else re.split(r"[,，;；\s]+", str(raw_references or ""))
        )
        history_fmeca_ids.update(
            str(value or "").strip().upper()
            for value in reference_values
            if str(value or "").strip()
        )
    maintenance_sources: List[Tuple[int, Mapping[str, Any]]] = [
        (number, _evidence_answer_fields(item))
        for number, item in ledger.items()
        if str(item.get("doc_type") or "") == "mfl"
    ]
    # Resolve a *single* selected mode before generating any supplement.  The
    # old implementation appended every FMECA row whose own cause happened to
    # be cited.  A broad data-flow answer naturally cites several such rows,
    # so it was incorrectly expanded into a second, repetitive fault-mode
    # catalogue.  Exact history references and a mode named verbatim by the
    # user are safe strong anchors; ordinary answer anchors are only safe when
    # exactly one distinct mode remains.
    user_query_compact = _compact_semantic_text(state.get("query") or "")
    anchored_by_mode: Dict[str, Dict[str, Any]] = {}

    for citation, item in ledger.items():
        if str(item.get("doc_type") or "") != "fmeca":
            continue
        fields = _evidence_answer_fields(item)
        mode = _fault_public_scalar(fields.get("failure_mode"))
        cause = next(
            (
                value
                for value in (
                    _fault_public_scalar(fields.get("fault_cause")),
                    _fault_public_scalar(fields.get("root_cause")),
                )
                if value
            ),
            "",
        )
        if not mode:
            continue

        mode_anchor = bool(
            len(_compact_semantic_text(mode)) >= 4
            and _cited_answer_line_supports_field(current, citation, mode)
        )
        cause_anchor = bool(
            len(_compact_semantic_text(cause)) >= 4
            and _cited_answer_line_supports_field(current, citation, cause)
        )
        record_id = str(item.get("record_id") or "").strip().upper()
        history_reference_anchor = bool(
            record_id
            and record_id in history_fmeca_ids
        )
        if not (mode_anchor or cause_anchor or history_reference_anchor):
            continue

        mode_identity = _compact_semantic_text(mode)
        query_mode_anchor = bool(
            len(mode_identity) >= 4 and mode_identity in user_query_compact
        )
        candidate = {
            "citation": citation,
            "fields": fields,
            "mode": mode,
            "cause": cause,
            "mode_anchor": mode_anchor,
            "history_reference_anchor": history_reference_anchor,
            "query_mode_anchor": query_mode_anchor,
        }
        existing = anchored_by_mode.get(mode_identity)
        candidate_priority = (
            int(history_reference_anchor),
            int(query_mode_anchor),
            int(mode_anchor),
            -citation,
        )
        existing_priority = (
            int(bool(existing and existing.get("history_reference_anchor"))),
            int(bool(existing and existing.get("query_mode_anchor"))),
            int(bool(existing and existing.get("mode_anchor"))),
            -int(existing.get("citation") or 0) if existing else 0,
        )
        if existing is None or candidate_priority > existing_priority:
            anchored_by_mode[mode_identity] = candidate

    anchored = list(anchored_by_mode.values())
    history_anchored = [
        item for item in anchored if item.get("history_reference_anchor")
    ]
    query_anchored = [item for item in anchored if item.get("query_mode_anchor")]
    if len(query_anchored) == 1:
        selected = query_anchored
    elif len(query_anchored) > 1:
        return current
    elif len(anchored) > 1:
        # Multiple ordinary anchors mean the draft is presenting alternatives
        # or stages.  Even one exact history link must not select a single row
        # for a second appendix after the draft has already anchored several
        # current candidates; doing so would make that historical branch look
        # artificially privileged and would repeat information already grouped
        # by the answer model.  A mode named verbatim by the user remains the
        # only safe override above.
        return current
    elif len(history_anchored) == 1:
        selected = history_anchored
    elif len(anchored) == 1:
        selected = anchored
    else:
        return current

    sections: List[str] = []
    for selected_item in selected:
        citation = int(selected_item.get("citation") or 0)
        fields = selected_item.get("fields") or {}
        mode = str(selected_item.get("mode") or "").strip()
        cause = str(selected_item.get("cause") or "").strip()

        facts: List[Tuple[str, int, str]] = []

        def add_missing(label: str, source_citation: int, value: Any) -> None:
            text = _fault_public_scalar(value)
            if not source_citation or not text:
                return
            if _cited_answer_line_supports_field(current, source_citation, text):
                return
            compact = _compact_semantic_text(text)
            if any(
                existing_citation == source_citation
                and _compact_semantic_text(existing) == compact
                for _label, existing_citation, existing in facts
            ):
                return
            facts.append((label, source_citation, text))

        add_missing("原因", citation, cause)
        add_missing("局部影响", citation, fields.get("local_impact"))
        add_missing("最终影响", citation, fields.get("final_impact"))
        detection = fields.get("detection_method") or fields.get("detect_method")
        add_missing("检测/判别", citation, detection)
        add_missing("使用补偿", citation, fields.get("compensation_action"))

        paired_maintenance = [
            (source_citation, source_fields)
            for source_citation, source_fields in maintenance_sources
            if _fault_mode_pair_matches(
                mode,
                "\n".join(
                    _fault_public_scalar(source_fields.get(key))
                    for key in _FAULT_MODE_FIELD_KEYS
                    if _fault_public_scalar(source_fields.get(key))
                ),
            )
        ]
        if paired_maintenance:
            source_citation, source_fields = paired_maintenance[0]
            action = (
                source_fields.get("maintenance_action")
                or source_fields.get("repair_method")
                or source_fields.get("final_solution")
            )
            add_missing("文档处置", source_citation, action)

        # A cause-only anchor may omit the canonical mode name altogether.
        # Emit it with the same FMECA citation so the supplemented facts have
        # an explicit and locally verifiable entity boundary.
        if not facts:
            continue
        section = [f"- **故障模式：{mode}**[^{citation}]"]
        section.extend(
            f"  - **{label}**：{value}[^{source_citation}]"
            for label, source_citation, value in facts
        )
        sections.append("\n".join(section))

    if not sections:
        return current
    supplement = "### 故障模式的原因、影响与处理\n\n" + "\n\n".join(sections)
    return _normalize_generated_markdown(f"{current.rstrip()}\n\n{supplement}")


def _ensure_current_incident_actions(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Restore a compact, source-bound handling action for a live incident.

    The answer model normally writes the action itself.  This terminal guard
    covers the failure mode observed in smoke tests: it cited the FMECA cause
    and a generic detection method while dropping an approved HMC/MFL repair
    record.  It only runs for the planner's ``diagnostic_closure`` contract,
    starts from a failure mode already cited in the answer, and pairs an
    HMC/MFL row solely by stored mode/symptom containment.  It therefore does
    not use a semantic neighbour as a cross-document repair instruction.
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    current = str(answer or "")
    ledger = _citation_ledger(dict(state))
    anchored_modes = _current_incident_anchored_modes(current, ledger)

    if not anchored_modes:
        return current

    appended: List[str] = []
    emitted_actions: set[str] = set()
    # One automatically restored action is enough to make a live-incident
    # answer executable.  More candidates remain visible as alternatives, but
    # their repairs must be selected by the answer model rather than appended
    # as a noisy catch-all checklist.
    for _mode_citation, mode in anchored_modes[:1]:
        matches: List[Tuple[int, int, str, str]] = []
        for citation, item in ledger.items():
            doc_type = str(item.get("doc_type") or "")
            if doc_type not in {"hmc", "mfl", "fmeca"}:
                continue
            fields = _evidence_answer_fields(item)
            if doc_type == "hmc":
                identity = _fault_public_scalar(fields.get("fault_symptom"))
            else:
                identity = "\n".join(
                    _fault_public_scalar(fields.get(key))
                    for key in _FAULT_MODE_FIELD_KEYS
                    if _fault_public_scalar(fields.get(key))
                )
            if doc_type == "fmeca" and citation == _mode_citation:
                identity = mode
            if not identity or not _fault_mode_pair_matches(mode, identity):
                continue
            action = next(
                (
                    _fault_public_scalar(fields.get(key))
                    for key in (
                        "maintenance_action",
                        "repair_method",
                        "final_solution",
                        "compensation_action",
                    )
                    if _fault_public_scalar(fields.get(key))
                ),
                "",
            )
            if not action or _cited_answer_line_supports_field(current, citation, action):
                continue
            # Prefer a directly matching maintenance/HMC record over a FMECA
            # compensation field.  Within one type, keep existing citation
            # order stable for reproducibility.
            source_rank = {"hmc": 0, "mfl": 1, "fmeca": 2}[doc_type]
            matches.append((source_rank, citation, doc_type, action))
        if not matches:
            continue
        _rank, citation, doc_type, action = min(matches, key=lambda item: (item[0], item[1]))
        action_key = _compact_semantic_text(action)
        if not action_key or action_key in emitted_actions:
            continue
        emitted_actions.add(action_key)
        label = "文档处置" if doc_type in {"hmc", "mfl"} else "使用补偿"
        appended.append(f"- **{mode}的{label}**：{action}[^{citation}]")

    if not appended:
        return current
    return _insert_into_matching_markdown_section(
        current,
        heading_pattern=re.compile(r"处置|维修|处理|建议"),
        default_heading="### 可执行的文档处置",
        body_lines=appended,
    )


def _restore_hollow_current_incident_candidate(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Fill a visible HMC candidate card only from its exact paired records.

    Local models can correctly select an HMC maintenance record for the first
    (and usually most relevant) candidate, then lose its body while citation
    repair removes a mixed sentence.  The ordinary numbered-card guard only
    owns headings cited by FMECA, so such an HMC heading used to survive as an
    empty ``1. ...`` line.  This repair is deliberately narrower: it accepts
    a heading only when its literal text contains the HMC symptom and a cited
    FMECA failure mode contains that same stored symptom.  It then copies the
    FMECA cause/impact/check and the HMC/MFL action as independent, cited
    fields.  No semantic-neighbour matching or inferred relationship is used.
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    current = str(answer or "").strip()
    if not current:
        return current

    ledger = _citation_ledger(dict(state))
    fmeca_rows: List[Tuple[int, Mapping[str, Any], str]] = []
    for citation, item in ledger.items():
        if str(item.get("doc_type") or "") != "fmeca":
            continue
        mode = _fault_public_scalar(_evidence_answer_fields(item).get("failure_mode"))
        if mode:
            fmeca_rows.append((citation, item, mode))
    if not fmeca_rows:
        return current

    lines = current.splitlines()
    for start, line in enumerate(lines):
        heading = _NUMBERED_FAULT_BRANCH_RE.match(line)
        if not heading:
            continue
        hmc_citations = [
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", line)
            if int(value) in ledger
            and str(ledger[int(value)].get("doc_type") or "") == "hmc"
        ]
        if not hmc_citations:
            continue

        # This is a candidate card only when the visible title names the same
        # stored FMECA mode that is literally contained in the HMC symptom.
        # A heading may expand ``BCM 复位信号卡滞`` into
        # ``启动控制模块（BCM）的复位信号卡滞``; requiring the shorter, stored
        # mode on both sides keeps that expansion readable without treating a
        # merely similar HMC phrase as the same event.
        title = re.sub(r"\[\^\d+\]", "", str(heading.group("body") or ""))
        title = re.sub(r"[*_`#]", "", title).strip(" ：:")
        end = start + 1
        while end < len(lines) and not _NUMBERED_FAULT_BRANCH_RE.match(lines[end]):
            if lines[end].lstrip().startswith("#"):
                break
            end += 1
        if any(value.strip() for value in lines[start + 1:end]):
            # It already has a body.  The usual source-local citation guard
            # remains responsible for any individual unsupported fact.
            continue

        selected: Tuple[int, Mapping[str, Any], str, int, Mapping[str, Any]] | None = None
        for hmc_citation in hmc_citations:
            hmc_item = ledger[hmc_citation]
            symptom = _fault_public_scalar(
                _evidence_answer_fields(hmc_item).get("fault_symptom")
            )
            if not symptom:
                continue
            for fmeca_citation, fmeca_item, mode in fmeca_rows:
                if (
                    _fault_mode_pair_matches(mode, symptom)
                    and _fault_mode_pair_matches(mode, title)
                ):
                    selected = (
                        hmc_citation,
                        hmc_item,
                        symptom,
                        fmeca_citation,
                        fmeca_item,
                    )
                    break
            if selected:
                break
        if not selected:
            # An HMC row can be the only approved source for a candidate.  It
            # cannot support a FMECA-style cause/impact explanation, but its
            # own symptom and maintenance action are still useful.  Replace a
            # bare model-generated heading with that exact source wording so a
            # user never sees an unexplained second candidate.
            hmc_citation = hmc_citations[0]
            hmc_fields = _evidence_answer_fields(ledger[hmc_citation])
            symptom = _fault_public_scalar(hmc_fields.get("fault_symptom"))
            action = _fault_public_scalar(
                hmc_fields.get("maintenance_action")
                or hmc_fields.get("repair_method")
                or hmc_fields.get("final_solution")
            )
            if not symptom and not action:
                continue
            # A bare HMC row is not automatically a second diagnosis branch.
            # Keep it only when its stored symptom is explicitly tied to the
            # current wording (or the user supplied an exact HMC lookup).  This
            # prevents a generic high-load hit such as a HUM hold-time issue
            # from appearing beside an otherwise unrelated PSU candidate.
            query_text = _compact_semantic_text(
                " ".join(
                    (
                        str(state.get("query") or ""),
                        str(state.get("rewritten_query") or ""),
                    )
                )
            )
            hmc_identity = str(ledger[hmc_citation].get("record_id") or "").strip()
            exact_hmc_in_query = bool(
                hmc_identity
                and _compact_semantic_text(hmc_identity) in query_text
            )
            if not exact_hmc_in_query and not _fault_mode_pair_matches(symptom, query_text):
                # The model exposed a bare HMC heading but the stored symptom
                # is neither an exact lookup nor a lexical match for this
                # incident.  Remove that orphan branch instead of showing a
                # second, unexplained candidate.
                lines[start:end] = []
                return _normalize_generated_markdown("\n".join(lines))
            source_title = symptom or title
            replacement = [
                f"**文档记录：{source_title}**[^{hmc_citation}]",
            ]
            if symptom:
                replacement.append(
                    f"  - **记录现象**：{symptom}[^{hmc_citation}]"
                )
            if action:
                replacement.append(
                    f"  - **文档处置**：{action}[^{hmc_citation}]"
                )
            replacement.append("")
            lines[start:end] = replacement
            return _normalize_generated_markdown("\n".join(lines))

        hmc_citation, hmc_item, _symptom, fmeca_citation, fmeca_item = selected
        fmeca_fields = _evidence_answer_fields(fmeca_item)
        hmc_fields = _evidence_answer_fields(hmc_item)
        details: List[str] = []
        cause = _fault_public_scalar(
            fmeca_fields.get("fault_cause") or fmeca_fields.get("root_cause")
        )
        impact = _fault_public_scalar(
            fmeca_fields.get("final_impact") or fmeca_fields.get("local_impact")
        )
        check = _fault_public_scalar(
            fmeca_fields.get("detection_method") or fmeca_fields.get("detect_method")
        )
        action = _fault_public_scalar(
            hmc_fields.get("maintenance_action")
            or hmc_fields.get("repair_method")
            or hmc_fields.get("final_solution")
        )
        action_citation = hmc_citation
        if not action:
            # An HMC row may contain only the symptom.  An MFL action is still
            # eligible only when its stored failure field contains that exact
            # HMC symptom / FMECA mode.
            for candidate_citation, candidate in ledger.items():
                if str(candidate.get("doc_type") or "") != "mfl":
                    continue
                candidate_fields = _evidence_answer_fields(candidate)
                candidate_mode = "\n".join(
                    _fault_public_scalar(candidate_fields.get(key))
                    for key in _FAULT_MODE_FIELD_KEYS
                    if _fault_public_scalar(candidate_fields.get(key))
                )
                candidate_action = _fault_public_scalar(
                    candidate_fields.get("maintenance_action")
                    or candidate_fields.get("repair_method")
                    or candidate_fields.get("final_solution")
                )
                if candidate_action and _fault_mode_pair_matches(_symptom, candidate_mode):
                    action = candidate_action
                    action_citation = candidate_citation
                    break

        if cause:
            details.append(f"  - **原因**：{cause}[^{fmeca_citation}]")
        if impact:
            details.append(f"  - **可能影响**：{impact}[^{fmeca_citation}]")
        if check:
            details.append(f"  - **如何确认**：{check}[^{fmeca_citation}]")
        if action:
            details.append(f"  - **文档处置**：{action}[^{action_citation}]")
        if not details:
            # Do not publish a heading that cannot be made informative from
            # direct source fields.
            lines[start:end] = []
            return _normalize_generated_markdown("\n".join(lines))

        # Make the source-bound branch a priority card rather than a first
        # list item.  This avoids the misleading ``1., 1., 2.`` sequence when
        # the following FMECA candidates retain their own ordered list.
        priority = f"**优先定位：{title}**[^{hmc_citation}]"
        replacement = [priority, *details, ""]
        lines[start:end] = replacement
        return _normalize_generated_markdown("\n".join(lines))
    return current


def _strip_fta_only_current_incident_candidate_cards(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Do not present an FTA-only event as a detailed repair candidate.

    An FTA proves the validated Boolean structure around a top event.  It does
    not by itself provide a board-level cause, impact, inspection or repair
    instruction.  In a live-incident answer, a numbered candidate with only
    FTA citations can therefore look precise while borrowing its prose from a
    different source.  Keep the FTA available as structural context, but drop
    that *detailed candidate card* unless it has a direct FMECA/HMC/MFL record
    of its own.
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    lines = str(answer or "").splitlines()
    ledger = _citation_ledger(dict(state))
    remove_indexes: set[int] = set()
    index = 0
    while index < len(lines):
        if not _NUMBERED_FAULT_BRANCH_RE.match(lines[index]):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not _NUMBERED_FAULT_BRANCH_RE.match(lines[end]):
            if lines[end].lstrip().startswith("#"):
                break
            end += 1
        citations = {
            int(value)
            for candidate_line in lines[index:end]
            for value in re.findall(r"\[\^(\d+)\]", candidate_line)
            if int(value) in ledger
        }
        doc_types = {
            str(ledger[citation].get("doc_type") or "") for citation in citations
        }
        if "fta" in doc_types and not (doc_types & {"fmeca", "hmc", "mfl"}):
            remove_indexes.update(range(index, end))
        index = end
    if not remove_indexes:
        return str(answer or "")
    return _normalize_generated_markdown(
        "\n".join(line for index, line in enumerate(lines) if index not in remove_indexes)
    )


def _renumber_current_incident_candidate_cards(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Renumber only source-bound diagnostic cards after a priority card."""

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    ledger = _citation_ledger(dict(state))
    direct_types = {"fmeca", "hmc", "mfl"}
    lines = str(answer or "").splitlines()
    ordinal = 0
    for index, line in enumerate(lines):
        match = _NUMBERED_FAULT_BRANCH_RE.match(line)
        if not match:
            continue
        # A model sometimes numbers the following history section as if it
        # were another fault candidate (for example ``4. 相似历史案例``).
        # That leaves the visible list as ``1, 4, 2`` after source-bound
        # candidates are renumbered.  History is a separate answer section,
        # so convert the accidental list marker to a heading and leave it out
        # of the candidate ordinal sequence.
        body = str(match.group("body") or "")
        if re.search(
            r"相似历史|相关历史|历史案例|既往案例|案例回顾|\bHIS[-_]\w+",
            body,
            flags=re.IGNORECASE,
        ):
            lines[index] = f"### {body.strip()}"
            continue
        citations = {
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", line)
            if int(value) in ledger
        }
        if not any(
            str(ledger[citation].get("doc_type") or "") in direct_types
            for citation in citations
        ):
            continue
        ordinal += 1
        lines[index] = re.sub(r"^\d{1,2}[.、)]", f"{ordinal}.", line, count=1)
    return _normalize_generated_markdown("\n".join(lines))


def _align_current_incident_candidate_count(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Keep a model-written “以下 N 个候选” lead-in aligned with visible cards."""

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    ledger = _citation_ledger(dict(state))
    direct_types = {"fmeca", "hmc", "mfl"}
    lines = str(answer or "").splitlines()
    count = sum(1 for line in lines if _CURRENT_INCIDENT_PRIORITY_CARD_RE.match(line))
    for line in lines:
        if not (_NUMBERED_FAULT_BRANCH_RE.match(line) and "**" in line):
            continue
        citations = {
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", line)
            if int(value) in ledger
        }
        if any(
            str(ledger[citation].get("doc_type") or "") in direct_types
            for citation in citations
        ):
            count += 1
    if count < 1:
        return str(answer or "")
    chinese_counts = {
        1: "一",
        2: "两",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    count_text = chinese_counts.get(count, str(count))
    pattern = re.compile(
        r"(?P<prefix>(?:优先定位|重点排查|需要排查).{0,12}?以下|以下)"
        r"(?P<number>[一二两三四五六七八九十\d]+)(?P<measure>个|项|处)?"
        r"(?P<unit>板级环节|候选(?:方向)?|排查方向)"
    )

    def replace(match: re.Match[str]) -> str:
        measure = str(match.group("measure") or "个")
        return f"{match.group('prefix')}{count_text}{measure}{match.group('unit')}"

    return pattern.sub(replace, str(answer or ""))


def _deduplicate_current_incident_action_lines(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Remove a repeated terminal repair line already shown in a priority card."""

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    seen_actions: set[str] = set()
    output: List[str] = []
    action_line = re.compile(r"^\s*[-+*]\s+.*(?:文档处置|可采取处置).*[：:]\s*(.+)$")
    for line in str(answer or "").splitlines():
        match = action_line.match(line)
        if not match:
            output.append(line)
            continue
        action = re.sub(r"\[\^\d+\]", "", match.group(1)).strip(" 。；;")
        key = _compact_semantic_text(action)
        if key and key in seen_actions:
            continue
        if key:
            seen_actions.add(key)
        output.append(line)
    return _normalize_generated_markdown("\n".join(output))


_CURRENT_INCIDENT_PRIORITY_CARD_RE = re.compile(
    r"^\*\*优先定位：(?P<title>.+?)\*\*(?P<citations>(?:\s*\[\^\d+\])*)\s*$"
)


_CURRENT_INCIDENT_MODE_SUPPLEMENT_RE = re.compile(
    r"^\s*[-+*]\s+\*\*故障模式：(?P<mode>.+?)\*\*"
    r"(?P<citations>(?:\s*\[\^\d+\])*)\s*$"
)


_CURRENT_INCIDENT_COMPENSATION_RE = re.compile(
    r"^\s*[-+*]\s+\*\*(?P<mode>.+?)的使用补偿\*\*\s*[:：]\s*"
    r"(?P<value>.+)$"
)


def _merge_current_incident_mode_supplements(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Keep one fault mode's check/handling facts in one visible card.

    The bounded single-mode completeness editor may append a safe source-bound
    ``故障模式：...`` supplement after a short candidate card.  That is correct
    but visually fragmented.  For a live incident, move that supplement into
    the preceding card only when the two locations cite the *same* approved
    FMECA row and its stored failure mode matches the supplement title.
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    lines = str(answer or "").splitlines()
    if not lines:
        return ""
    ledger = _citation_ledger(dict(state))

    def citation_numbers(value: str) -> set[int]:
        return {
            int(number)
            for number in re.findall(r"\[\^(\d+)\]", value)
            if int(number) in ledger
        }

    def source_mode(citation: int) -> str:
        item = ledger.get(citation) or {}
        if str(item.get("doc_type") or "") != "fmeca":
            return ""
        return _fault_public_scalar(_evidence_answer_fields(item).get("failure_mode"))

    card_starts: List[int] = []
    for index, line in enumerate(lines):
        if _CURRENT_INCIDENT_PRIORITY_CARD_RE.match(line) or (
            _NUMBERED_FAULT_BRANCH_RE.match(line) and "**" in line
        ):
            card_starts.append(index)
    if not card_starts:
        return str(answer or "")

    cards: List[Dict[str, Any]] = []
    for offset, start in enumerate(card_starts):
        end = card_starts[offset + 1] if offset + 1 < len(card_starts) else len(lines)
        for position in range(start + 1, end):
            if _CURRENT_INCIDENT_MODE_SUPPLEMENT_RE.match(lines[position]):
                end = position
                break
            if lines[position].lstrip().startswith("#"):
                end = position
                break
        block = lines[start:end]
        citations = citation_numbers("\n".join(block))
        modes = {
            mode
            for citation in citations
            if (mode := source_mode(citation))
        }
        if not modes:
            continue
        cards.append(
            {
                "start": start,
                "end": end,
                "last_content": max(
                    (position for position in range(start, end) if lines[position].strip()),
                    default=start,
                ),
                "citations": citations,
                "modes": modes,
                "text": "\n".join(block),
            }
        )
    if not cards:
        return str(answer or "")

    remove_indexes: set[int] = set()
    insertions: Dict[int, List[str]] = {}

    def target_card(mode: str, citations: set[int], before: int) -> Dict[str, Any] | None:
        for card in reversed(cards):
            if int(card["start"]) >= before:
                continue
            shared_modes = {
                source_mode(citation)
                for citation in citations & set(card["citations"])
                if source_mode(citation)
            }
            if any(_fault_mode_pair_matches(mode, candidate) for candidate in shared_modes):
                return card
        return None

    index = 0
    while index < len(lines):
        match = _CURRENT_INCIDENT_MODE_SUPPLEMENT_RE.match(lines[index])
        if not match:
            index += 1
            continue
        mode = str(match.group("mode") or "").strip()
        citations = citation_numbers(match.group("citations") or "")
        if not mode or not citations or not all(
            source_mode(citation) and _fault_mode_pair_matches(mode, source_mode(citation))
            for citation in citations
        ):
            index += 1
            continue
        target = target_card(mode, citations, index)
        if target is None:
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            if not lines[end].strip():
                break
            if _NUMBERED_FAULT_BRANCH_RE.match(lines[end]) or lines[end].lstrip().startswith("#"):
                break
            end += 1
        additions: List[str] = []
        target_text = str(target["text"])
        for child in lines[index + 1:end]:
            transformed = re.sub(r"^\s*[-+*]\s+", "   - ", child)
            transformed = transformed.replace("**检测/判别**", "**如何确认**")
            transformed = transformed.replace("**使用补偿**", "**可采取处置**")
            candidate_value = re.sub(r"\[\^\d+\]", "", transformed)
            if candidate_value and _compact_semantic_text(candidate_value) in _compact_semantic_text(target_text):
                continue
            additions.append(transformed)
        if additions:
            insertions.setdefault(int(target["last_content"]), []).extend(additions)
        remove_indexes.update(range(index, end))
        index = end

    # FMECA compensation occasionally appears as one detached terminal line.
    # Move it into the exact same card for the same reason, but never treat a
    # free-text similarity as a mode match.
    for index, line in enumerate(lines):
        if index in remove_indexes:
            continue
        match = _CURRENT_INCIDENT_COMPENSATION_RE.match(line)
        if not match:
            continue
        mode = str(match.group("mode") or "").strip()
        citations = citation_numbers(line)
        if not mode or not citations or not all(
            source_mode(citation) and _fault_mode_pair_matches(mode, source_mode(citation))
            for citation in citations
        ):
            continue
        target = target_card(mode, citations, index)
        if target is None:
            continue
        transformed = re.sub(r"^\s*[-+*]\s+", "  - ", line)
        transformed = transformed.replace(f"**{mode}的使用补偿**", "**可选补偿**")
        target_text = str(target["text"])
        if _compact_semantic_text(re.sub(r"\[\^\d+\]", "", transformed)) not in _compact_semantic_text(target_text):
            insertions.setdefault(int(target["last_content"]), []).append(transformed)
        remove_indexes.add(index)

    if not remove_indexes and not insertions:
        return str(answer or "")
    output: List[str] = []
    for index, line in enumerate(lines):
        if index not in remove_indexes:
            output.append(line)
        for addition in insertions.get(index, []):
            output.append(addition)
    return _normalize_generated_markdown("\n".join(output))


def _normalize_current_incident_fmeca_cards(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Render a visible FMECA candidate from its own source fields only.

    This is a terminal presentation guard for an already selected diagnostic
    candidate, not a candidate generator.  It activates only for a numbered
    card whose heading itself cites an approved FMECA row.  Re-rendering that
    card from the row's cause/impact/detection/compensation fields prevents a
    model from mixing an FTA sibling event into the same card (for example,
    describing both A-path and B-path power failures beneath a B-path title).
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    lines = str(answer or "").splitlines()
    ledger = _citation_ledger(dict(state))
    replacements: List[Tuple[int, int, List[str]]] = []
    for start, line in enumerate(lines):
        heading = _NUMBERED_FAULT_BRANCH_RE.match(line)
        if not heading or "**" not in line:
            continue
        cited_fmeca = [
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", line)
            if int(value) in ledger
            and str(ledger[int(value)].get("doc_type") or "") == "fmeca"
        ]
        if not cited_fmeca:
            continue
        citation = cited_fmeca[0]
        fields = _evidence_answer_fields(ledger[citation])
        mode = _fault_public_scalar(fields.get("failure_mode"))
        cause = _fault_public_scalar(fields.get("fault_cause") or fields.get("root_cause"))
        impact = _fault_public_scalar(fields.get("final_impact") or fields.get("local_impact"))
        check = _fault_public_scalar(fields.get("detection_method") or fields.get("detect_method"))
        action = _fault_public_scalar(fields.get("compensation_action"))
        if not mode or not (cause or impact or check or action):
            continue

        end = start + 1
        while end < len(lines) and not _NUMBERED_FAULT_BRANCH_RE.match(lines[end]):
            if lines[end].lstrip().startswith("#"):
                break
            end += 1
        original = "\n".join(lines[start:end])
        # Keep a compact, already source-local card untouched.  The renderer
        # is only needed for a mixed FTA/FMECA block or an incomplete one.
        required_values = [value for value in (cause, impact, check, action) if value]
        source_mixed = any(
            int(value) in ledger and str(ledger[int(value)].get("doc_type") or "") == "fta"
            for value in re.findall(r"\[\^(\d+)\]", original)
        )
        complete = all(
            _cited_answer_line_supports_field(original, citation, value)
            for value in required_values
        )
        if complete and not source_mixed:
            continue

        heading_body = str(heading.group("body") or "")
        title_match = re.search(r"\*\*(.+?)\*\*", heading_body)
        title = str(title_match.group(1) if title_match else mode).strip()
        card = [f"{heading.group('number')}. **{title}**[^{citation}]"]
        if cause:
            card.append(f"   - **原因**：{cause}[^{citation}]")
        if impact:
            card.append(f"   - **影响**：{impact}[^{citation}]")
        if check:
            card.append(f"   - **如何确认**：{check}[^{citation}]")
        if action:
            card.append(f"   - **可采取处置**：{action}[^{citation}]")
        replacements.append((start, end, card))

    if not replacements:
        return str(answer or "")
    for start, end, card in reversed(replacements):
        lines[start:end] = card
    return _normalize_generated_markdown("\n".join(lines))


def _deduplicate_current_incident_candidate_modes(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Keep one visible card for a mode cited through FMECA/HMC/MFL rows.

    Retrieval often returns the same stored mode three times: a FMECA cause
    row, an HMC repair row and an MFL/API row. They are complementary source
    fields, not three candidate failures. Merge child facts into the first
    card when their approved mode identities match exact stored text
    containment; never merge merely similar wording from different modes.
    """

    if not _diagnostic_closure_required(state):
        return str(answer or "")
    lines = str(answer or "").splitlines()
    ledger = _citation_ledger(dict(state))
    direct_types = {"fmeca", "hmc", "mfl"}
    starts: List[int] = []
    for index, line in enumerate(lines):
        if _CURRENT_INCIDENT_PRIORITY_CARD_RE.match(line) or (
            _NUMBERED_FAULT_BRANCH_RE.match(line) and "**" in line
        ):
            cited = {
                int(value)
                for value in re.findall(r"\[\^(\d+)\]", line)
                if int(value) in ledger
            }
            if cited and any(
                str(ledger[number].get("doc_type") or "") in direct_types
                for number in cited
            ):
                starts.append(index)
    if len(starts) < 2:
        return str(answer or "")

    cards: List[Dict[str, Any]] = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(lines)
        for position in range(start + 1, end):
            if lines[position].lstrip().startswith("#"):
                end = position
                break
        block_lines = list(lines[start:end])
        citations = {
            int(value)
            for value in re.findall(r"\[\^(\d+)\]", "\n".join(block_lines))
            if int(value) in ledger
        }
        mode_values: List[str] = []
        for citation in citations:
            if str(ledger[citation].get("doc_type") or "") not in direct_types:
                continue
            fields = _evidence_answer_fields(ledger[citation])
            preferred = [
                _fault_public_scalar(fields.get(key))
                for key in ("failure_mode", "module_failure", "fault_symptom")
            ]
            fallback = [
                _fault_public_scalar(fields.get(key))
                for key in ("sur_failure", "subcircuit_failure")
            ]
            for value in (*preferred, *fallback):
                if value and value not in mode_values:
                    mode_values.append(value)
        if mode_values:
            cards.append(
                {
                    "start": start,
                    "end": end,
                    "lines": block_lines,
                    "mode_values": mode_values,
                }
            )
    if len(cards) < 2:
        return str(answer or "")

    def same_mode(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return any(
            _fault_mode_pair_matches(str(a), str(b))
            for a in (left.get("mode_values") or [])
            for b in (right.get("mode_values") or [])
            if str(a).strip() and str(b).strip()
        )

    kept: List[Dict[str, Any]] = []
    duplicate_indexes: set[int] = set()
    for card in cards:
        target = next(
            (candidate for candidate in kept if same_mode(candidate, card)),
            None,
        )
        if target is None:
            kept.append(card)
            continue
        existing = {
            _compact_semantic_text(re.sub(r"\[\^\d+\]", "", line))
            for line in target["lines"]
            if _compact_semantic_text(re.sub(r"\[\^\d+\]", "", line))
        }
        for child in card["lines"][1:]:
            identity = _compact_semantic_text(re.sub(r"\[\^\d+\]", "", child))
            if identity and identity in existing:
                continue
            if child.strip():
                target["lines"].append(child)
                existing.add(identity)
        duplicate_indexes.add(int(card["start"]))

    replacements = {
        int(card["start"]): list(card["lines"])
        for card in kept
        if list(card["lines"])
        != lines[int(card["start"]): int(card["end"])]
    }
    output: List[str] = []
    index = 0
    while index < len(lines):
        card = next(
            (item for item in cards if int(item["start"]) == index),
            None,
        )
        if card is None:
            output.append(lines[index])
            index += 1
            continue
        if index in duplicate_indexes:
            index = int(card["end"])
            continue
        output.extend(
            replacements.get(index, lines[index: int(card["end"])])
        )
        index = int(card["end"])
    return _normalize_generated_markdown("\n".join(output))


def _current_incident_anchored_modes(
    answer: str,
    ledger: Mapping[int, Mapping[str, Any]],
) -> List[Tuple[int, str]]:
    """Prefer an explicitly cited HMC symptom over a bare FMECA heading.

    A local answer model sometimes cites the HMC row for a candidate title and
    the matching FMECA row for the surrounding explanation.  The two rows may
    not share the same physical Markdown line, even though their stored mode
    and HMC symptom contain one another exactly.  This helper keeps that
    source-local pairing usable without treating semantically similar modes as
    identical.
    """

    result: List[Tuple[int, str]] = []
    seen: set[str] = set()

    def add(citation: int, mode: str) -> None:
        key = _compact_semantic_text(mode)
        if key and key not in seen:
            seen.add(key)
            result.append((citation, mode))

    fmeca_rows: List[Tuple[int, str]] = []
    for citation, item in ledger.items():
        if str(item.get("doc_type") or "") != "fmeca":
            continue
        mode = _fault_public_scalar(_evidence_answer_fields(item).get("failure_mode"))
        if mode:
            fmeca_rows.append((citation, mode))

    # An HMC fault symptom shown in the answer is a concrete current-fault
    # anchor.  Pair it only to a FMECA mode by exact normalized containment.
    for hmc_citation, item in ledger.items():
        if str(item.get("doc_type") or "") != "hmc":
            continue
        symptom = _fault_public_scalar(_evidence_answer_fields(item).get("fault_symptom"))
        if not symptom or not _cited_answer_line_supports_field(answer, hmc_citation, symptom):
            continue
        for fmeca_citation, mode in fmeca_rows:
            if _fault_mode_pair_matches(mode, symptom):
                add(fmeca_citation, mode)
                break

    for citation, mode in fmeca_rows:
        if _cited_answer_line_supports_field(answer, citation, mode):
            add(citation, mode)
    return result


def _prompt_payload_as_evidence(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Restore the small internal shape needed by terminal field guards."""

    return {
        "evidence_id": str(item.get("evidence_id") or ""),
        "doc_id": str(item.get("doc_id") or ""),
        "doc_name": str(item.get("doc_name") or ""),
        "title": str(item.get("title") or ""),
        "chunk_id": str(item.get("chunk_id") or ""),
        "record_id": str(item.get("record_id") or ""),
        "content": str(item.get("content") or ""),
        "display_summary": str(item.get("display_summary") or ""),
        "claim_type": str(item.get("type") or ""),
        "source_ref": str(item.get("source_ref") or ""),
        "source_kind": str(item.get("source_kind") or ""),
        "doc_type": str(item.get("doc_type") or ""),
        "authority": str(item.get("authority") or ""),
        "evidence_level": str(item.get("evidence_level") or ""),
        "review_status": str(item.get("review_status") or ""),
        "match_type": str(item.get("match_type") or ""),
        "metadata": dict(item.get("document_metadata") or {}),
    }


def _extend_incident_action_ledger(
    answer: str,
    prompt_citations: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Add one exact matching HMC/MFL action to the final citation ledger.

    The normal prompt window is intentionally bounded.  It can therefore keep
    the FMECA row that explains a candidate while a matching direct repair row
    remains only in the audit registry.  For a planner-approved live-incident
    closure, promote at most one HMC/MFL row *per already cited FMECA mode* by
    exact stored text containment.  No search result is promoted by score,
    embedding similarity, title similarity, or an inferred relationship.
    """

    current = str(answer or "")
    extended = [dict(item) for item in prompt_citations if isinstance(item, Mapping)]
    if not _diagnostic_closure_required(state) or not extended:
        return current, extended

    prompt_evidence = [_prompt_payload_as_evidence(item) for item in extended]
    prompt_ids = [
        str(item.get("evidence_id") or "") for item in prompt_evidence if str(item.get("evidence_id") or "")
    ]
    decision = state.get("safety_decision") if isinstance(state.get("safety_decision"), Mapping) else {}
    approved_ids = approved_id_set(decision.get("approved_evidence_ids") or [])
    temporary_state = {
        "query": state.get("query"),
        "intent": state.get("intent"),
        "query_targets": state.get("query_targets"),
        "query_frame": state.get("query_frame"),
        "evidence_registry": prompt_evidence,
        "context_evidence_ids": prompt_ids,
        "_terminal_public_ledger": True,
        "safety_decision": {
            "approved_evidence_ids": [
                evidence_id for evidence_id in prompt_ids if evidence_id in approved_ids
            ],
            "context_evidence_ids": prompt_ids,
        },
    }
    ledger = _citation_ledger(temporary_state)
    anchored_modes = [
        mode
        for _citation, mode in _current_incident_anchored_modes(current, ledger)
    ]

    if not anchored_modes:
        return current, extended

    raw_evidence = [
        item
        for item in deduplicate_evidence(state.get("evidence_registry") or [])
        if isinstance(item, Mapping)
        and str(item.get("evidence_id") or "") in approved_ids
        and str(item.get("doc_type") or "") in {"hmc", "mfl"}
        and str(item.get("authority") or "") in _DIRECT_EVIDENCE_AUTHORITIES
    ]
    known_ids = set(prompt_ids)
    for mode in anchored_modes[:1]:
        matching: List[Tuple[int, Dict[str, Any]]] = []
        for raw in raw_evidence:
            evidence_id = str(raw.get("evidence_id") or "")
            if not evidence_id or evidence_id in known_ids:
                continue
            fields = _evidence_answer_fields(raw)
            doc_type = str(raw.get("doc_type") or "")
            identity = (
                _fault_public_scalar(fields.get("fault_symptom"))
                if doc_type == "hmc"
                else "\n".join(
                    _fault_public_scalar(fields.get(key))
                    for key in _FAULT_MODE_FIELD_KEYS
                    if _fault_public_scalar(fields.get(key))
                )
            )
            action = next(
                (
                    _fault_public_scalar(fields.get(key))
                    for key in ("maintenance_action", "repair_method", "final_solution")
                    if _fault_public_scalar(fields.get(key))
                ),
                "",
            )
            if identity and action and _fault_mode_pair_matches(mode, identity):
                matching.append((0 if doc_type == "hmc" else 1, dict(raw)))
        if not matching:
            continue
        _rank, selected = min(
            matching,
            key=lambda pair: (
                pair[0],
                -_answer_relevance_score(pair[1])[0],
                str(pair[1].get("evidence_id") or ""),
            ),
        )
        payload = _evidence_payload(selected)
        payload["id"] = len(extended) + 1
        extended.append(payload)
        evidence_id = str(payload.get("evidence_id") or "")
        known_ids.add(evidence_id)
        prompt_evidence.append(_prompt_payload_as_evidence(payload))
        prompt_ids.append(evidence_id)

    if len(extended) == len(prompt_citations):
        return current, extended
    temporary_state["evidence_registry"] = prompt_evidence
    temporary_state["context_evidence_ids"] = prompt_ids
    temporary_state["safety_decision"]["context_evidence_ids"] = prompt_ids
    temporary_state["safety_decision"]["approved_evidence_ids"] = [
        evidence_id for evidence_id in prompt_ids if evidence_id in approved_ids
    ]
    return _ensure_current_incident_actions(current, temporary_state), extended


def _ensure_approved_single_fault_completeness(
    answer: str,
    state: Dict[str, Any],
) -> str:
    """Apply the terminal single-mode closure from approved cited fields."""

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        # The single-mode anchor is only a conversational seed in an
        # exhaustiveness follow-up. Appending its details would again answer
        # one member instead of the requested collection.
        return str(answer or "")
    current = _append_missing_anchored_single_fault_details(answer, state)
    # Preserve the most directly bound HMC candidate as a complete priority
    # card before the generic action guard decides whether a trailing action
    # section is needed.  Otherwise an HMC-only heading can survive while the
    # relevant repair instruction is appended far away from that candidate.
    current = _restore_hollow_current_incident_candidate(current, state)
    current = _strip_fta_only_current_incident_candidate_cards(current, state)
    current = _ensure_current_incident_actions(current, state)
    current = _merge_current_incident_mode_supplements(current, state)
    current = _normalize_current_incident_fmeca_cards(current, state)
    current = _deduplicate_current_incident_candidate_modes(current, state)
    current = _deduplicate_current_incident_action_lines(current, state)
    current = _renumber_current_incident_candidate_cards(current, state)
    return _align_current_incident_candidate_count(current, state)


def _ensure_numbered_fault_candidate_completeness(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Fail closed when a broad numbered answer still contains hollow branches.

    The normal path remains LLM-authored and the bounded completeness editor gets
    two chances to repair it.  This terminal guard runs only when at least two
    numbered, FMECA-cited branches remain incomplete.  It rebuilds those branch
    cards from their *own* approved structured fields, so citation repair cannot
    leave a visible heading with no explanation and one mode can never borrow
    another mode's cause, impact or action.
    """

    current = str(answer or "").strip()
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        # A closed PostgreSQL inventory is an authoritative N-of-N listing,
        # not a ranked set of diagnostic candidates.  Rebuilding it with the
        # normal 1-to-3 candidate-card guard would silently discard collection
        # members (or their source-side causes) after citation compaction.
        return current
    if not current or _query_requests_parallel_failure_modes(state):
        return current
    blocks = _numbered_fault_candidate_blocks(current, state)
    if not blocks:
        return current
    missing_by_start = {
        int(block.get("start") or 0): _numbered_fault_branch_missing_facets(block)
        for block in blocks
    }
    if not any(missing_by_start.values()):
        return current

    lines = current.splitlines()
    rebuilt: List[str] = list(lines[: int(blocks[0].get("start") or 0)])
    seen_actions: set[str] = set()

    def one_value_per_source(
        sources: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
        *,
        limit: int = 2,
    ) -> List[Tuple[int, str, str]]:
        values: List[Tuple[int, str, str]] = []
        seen: set[str] = set()
        for source in sources:
            fields = (
                source.get("answer_fields")
                if isinstance(source.get("answer_fields"), Mapping)
                else {}
            )
            value = next(
                (
                    _fault_public_scalar(fields.get(key))
                    for key in keys
                    if _fault_public_scalar(fields.get(key))
                ),
                "",
            )
            compact = _compact_semantic_text(value)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            values.append(
                (
                    int(source.get("citation_id") or 0),
                    _fault_public_scalar(fields.get("failure_mode")),
                    value,
                )
            )
            if len(values) >= limit:
                break
        return values

    def render_values(
        values: Sequence[Tuple[int, str, str]],
        *,
        identify_modes: bool,
    ) -> str:
        multiple = len(values) > 1
        parts: List[str] = []
        for citation, mode, value in values:
            label = f"{mode}：" if identify_modes and (multiple or mode) else ""
            parts.append(f"{label}{value}[^{citation}]")
        return "；".join(parts)

    rendered_count = 0
    for block in blocks[:3]:
        sources = [
            source
            for source in (block.get("sources") or [])
            if isinstance(source, Mapping)
        ][:2]
        if not sources:
            continue
        branch_missing = missing_by_start.get(int(block.get("start") or 0), [])
        if not branch_missing:
            original = lines[int(block["start"]): int(block["end"])]
            if original:
                original[0] = re.sub(
                    r"^\d{1,2}[.、)]",
                    f"{rendered_count + 1}.",
                    original[0],
                )
                if rebuilt and rebuilt[-1].strip():
                    rebuilt.append("")
                rebuilt.extend(original)
                rendered_count += 1
            continue

        causes = one_value_per_source(sources, _FAULT_CAUSE_FIELD_KEYS)
        impacts = one_value_per_source(sources, ("final_impact", "local_impact"))
        checks = one_value_per_source(
            sources,
            ("detection_method", "detect_method", "api_usage"),
        )
        actions = one_value_per_source(sources, _FAULT_ACTION_FIELD_KEYS)
        actions = [
            item
            for item in actions
            if _compact_semantic_text(item[2]) not in seen_actions
        ]
        if not causes or not impacts or not (checks or actions):
            # A hollow branch with no complete approved field set is safer to
            # omit than to expose as if it were a supported candidate.
            continue
        seen_actions.update(_compact_semantic_text(item[2]) for item in actions)
        rendered_count += 1
        title = str(block.get("title") or "待排查环节").strip()
        markers = "".join(
            f"[^{int(source.get('citation_id') or 0)}]" for source in sources
        )
        branch_lines = [
            f"{rendered_count}. **{title}**{markers}",
            f"   - **为什么可能**：{render_values(causes, identify_modes=True)}",
            f"   - **可能影响**：{render_values(impacts, identify_modes=len(impacts) > 1)}",
        ]
        if checks:
            branch_lines.append(
                f"   - **如何确认**：{render_values(checks, identify_modes=False)}"
            )
        if actions:
            branch_lines.append(
                f"   - **可采取处置**：{render_values(actions, identify_modes=False)}"
            )
        if rebuilt and rebuilt[-1].strip():
            rebuilt.append("")
        rebuilt.extend(branch_lines)

    region_end = max(int(block.get("end") or 0) for block in blocks)
    ledger = _citation_ledger(dict(state))
    for line in lines[region_end:]:
        citations = {
            int(value) for value in re.findall(r"\[\^(\d+)\]", line)
        }
        detached_fault_action = bool(
            _TOP_LEVEL_BULLET_RE.match(line)
            and _ACTION_CLAIM_RE.search(line)
            and citations
            and all(
                number in ledger
                and str(ledger[number].get("doc_type") or "") in {"fmeca", "hmc", "mfl"}
                for number in citations
            )
        )
        if detached_fault_action:
            continue
        rebuilt.append(line)

    if rendered_count < 1:
        return current
    return _normalize_generated_markdown("\n".join(rebuilt))


def _ensure_numbered_fault_candidate_conditions(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """Expose source-side phase requirements for conditional candidates."""

    current = str(answer or "").strip()
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        # Collection rows must remain a literal source inventory.  Conditional
        # candidate annotations belong to ranked diagnosis answers and must not
        # reinterpret or reshape an exhaustive list during public renumbering.
        return current
    blocks = _numbered_fault_candidate_blocks(current, state)
    if not blocks:
        return current
    lines = current.splitlines()
    insertions: List[Tuple[int, str]] = []
    for block in blocks:
        block_text = _compact_semantic_text(block.get("text"))
        conditions: List[str] = []
        seen: set[str] = set()
        for source in block.get("sources") or []:
            if not isinstance(source, Mapping):
                continue
            item = source.get("item") if isinstance(source.get("item"), Mapping) else {}
            if _fault_candidate_fit(state, item) != "conditional":
                continue
            fields = (
                source.get("answer_fields")
                if isinstance(source.get("answer_fields"), Mapping)
                else {}
            )
            phase = _fault_public_scalar(fields.get("mission_phase"))
            if not phase or _compact_semantic_text(phase) in block_text:
                continue
            mode = _fault_public_scalar(fields.get("failure_mode"))
            citation = int(source.get("citation_id") or 0)
            identity = _compact_semantic_text(f"{mode}{phase}")
            if not identity or identity in seen or citation <= 0:
                continue
            seen.add(identity)
            label = f"{mode}仅在{phase}时再查" if mode else f"仅在{phase}时再查"
            conditions.append(f"{label}。[^{citation}]")
        if conditions:
            insertions.append(
                (
                    int(block.get("start") or 0) + 1,
                    "   - **适用条件**：" + "；".join(conditions),
                )
            )
    for index, line in reversed(insertions):
        lines.insert(index, line)
    return _normalize_generated_markdown("\n".join(lines))


def _ensure_multi_candidate_framing(
    answer: str,
    state: Mapping[str, Any],
) -> str:
    """State that multiple cited FMECA rows are alternatives, not co-failures.

    This is a presentation invariant rather than a diagnostic inference.  It
    never changes, ranks or adds a fault fact; it only prevents a compact list
    of supported alternatives from reading like several simultaneously
    confirmed failures when the user did not request a formal comparison.
    """

    current = str(answer or "").strip()
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if str(frame.get("coverage_scope") or "") == "complete_collection":
        return current
    if not current or _query_requests_parallel_failure_modes(state):
        return current
    cited_modes: set[str] = set()
    for citation, item in _citation_ledger(dict(state)).items():
        if str(item.get("doc_type") or "") != "fmeca":
            continue
        if f"[^{citation}]" not in current:
            continue
        fields = _evidence_answer_fields(item)
        mode = _fault_public_scalar(fields.get("failure_mode"))
        identity = _compact_semantic_text(
            mode or item.get("record_id") or item.get("title") or ""
        )
        if identity:
            cited_modes.add(identity)
    if len(cited_modes) < 2:
        return current
    if re.search(
        r"(?:替代|并列|待排查).{0,10}(?:方向|候选)|"
        r"(?:不表示|不代表|不意味着|并非).{0,18}(?:同时|全部|都).{0,8}(?:发生|存在|故障)?",
        current,
    ):
        return current
    return _normalize_generated_markdown(f"{_MULTI_CANDIDATE_FRAMING}\n\n{current}")


def _coalesce_exact_repeated_cited_lines(answer: str) -> str:
    """Remove only exact, citation-identical repeated fact units.

    This final guard is intentionally strict.  It does not use fuzzy
    similarity and therefore never merges different failure modes, historical
    cases, directions or calculation models merely because their wording is
    close.  Field-aware completeness checks handle paraphrased coverage; this
    guard only catches literal duplicate rows that survived an editor.
    """

    lines = str(answer or "").splitlines()
    seen: set[Tuple[bool, Tuple[int, ...], str]] = set()
    active_history = False
    for index, line in enumerate(lines):
        if _SECTION_HEADING.match(line):
            active_history = _history_section_context(
                _section_title(line), active_history
            )
            continue
        citations = tuple(
            sorted({int(value) for value in re.findall(r"\[\^(\d+)\]", line)})
        )
        if not citations:
            continue
        body = re.sub(r"\[\^\d+\]", "", line)
        # Unordered bullet glyphs and Markdown emphasis are presentation only.
        # Keep numeric prefixes because repeated history steps at different
        # ordinals remain distinct facts.
        body = re.sub(r"^\s*[-+*]\s+", "", body)
        body = re.sub(r"[*_`#]", "", body).strip()
        identity = _compact_semantic_text(body)
        if len(identity) < 6:
            continue
        history_context = bool(
            active_history
            or _unit_requires_fault_history(line)
            or re.search(r"(?:该|此|上述|本次)(?:历史)?(?:案例|记录)", line)
        )
        key = (history_context, citations, identity)
        if key in seen:
            lines[index] = ""
            continue
        seen.add(key)
    return _cleanup_generated_markdown_structure("\n".join(lines))


def _apply_terminal_fault_completeness(
    answer: str,
    manifest: Sequence[Mapping[str, Any]],
    state: Dict[str, Any],
) -> str:
    """Run mutually exclusive parallel/single deterministic closures."""

    current = _append_missing_parallel_failure_mode_branches(answer, manifest, state)
    return _ensure_approved_single_fault_completeness(current, state)


def _ensure_approved_parallel_fault_completeness(
    answer: str,
    state: Dict[str, Any],
) -> str:
    """Apply the multi-mode terminal fallback only while a gap still exists."""

    missing, manifest = _fault_completion_manifest(answer, state)
    if not missing or not manifest:
        return answer
    return _append_missing_parallel_failure_mode_branches(answer, manifest, state)


def _fault_preflight_instruction(state: Dict[str, Any]) -> str:
    """Make the first LLM pass reserve room for every evidence-backed facet."""

    missing, manifest = _fault_completion_manifest("", state)
    if not missing or not manifest:
        return ""
    labels = {
        "cause": "原因",
        "impact": "影响",
        "resolution": "检查/处置",
        "action": "文档处置",
    }
    lines = [
        "【本轮回答闭环检查（内部指令，不要原样输出）】",
        "已批准证据具备：" + "、".join(labels[value] for value in missing) + "。最终回答必须全部覆盖；不要把篇幅全部用于罗列候选。",
    ]
    branch_modes = list(
        dict.fromkeys(
            str(item.get("branch_mode") or "").strip()
            for item in manifest
            if str(item.get("branch_mode") or "").strip()
        )
    )
    if len(branch_modes) >= 2:
        lines.append(
            "用户明确要求区分多个模式；必须逐支回答，不得漏项："
            + "、".join(branch_modes)
            + "。每支只使用同名模式证据中的原因、影响、判别和处置字段。"
        )
    evidence_modes = list(
        dict.fromkeys(
            _fault_candidate_mode_text(item).strip()
            for item in manifest
            if str(item.get("doc_type") or "") == "fmeca"
            and _fault_candidate_mode_text(item).strip()
        )
    )
    broad_candidate_presentation = bool(
        len(evidence_modes) >= 2 and not _query_requests_parallel_failure_modes(state)
    )
    if broad_candidate_presentation:
        lines.append(
            "本轮必须使用紧凑的候选路径结构：第一句明确说明‘以下是替代排查方向，不表示这些故障同时发生’。"
            "正文只能有 1～3 个编号主分支，严格不得出现第 4 个；若问题描述了数据流或功能链，按实际流程排序并合并相近模式。"
            "每个主分支内部必须直接写清原因、影响、一个最有区分力的检查以及有证据的处置；不允许只有标题，"
            "也不要在末尾另列统一检查清单。已批准 FTA 若明确给出串联单点、冗余或路径结构，应先用一句话说明它如何决定排查顺序。"
            "弱相关候选仅在有独特伴随条件时写成一条‘仅当……时再查’；同一模式只出现一次，"
            "禁止在分阶段解释后再追加故障模式清单。"
        )
        query_frame = (
            state.get("query_frame")
            if isinstance(state.get("query_frame"), Mapping)
            else {}
        )
        operating_stage = str(query_frame.get("operating_stage") or "").strip()
        temporal_pattern = str(query_frame.get("temporal_pattern") or "").strip()
        stated_constraints = [
            value
            for value in (
                f"发生阶段={operating_stage}"
                if operating_stage and operating_stage != "未知"
                else "",
                f"发生规律={temporal_pattern}"
                if temporal_pattern and temporal_pattern != "未知"
                else "",
            )
            if value
        ]
        if stated_constraints:
            lines.append(
                "用户已明确的现象约束："
                + "、".join(stated_constraints)
                + "。选择主分支时必须逐项核对证据中的任务阶段、工作方式和最终影响；"
                "与约束不一致的记录只能作为带有明确触发条件的次要线索，不能占据无条件主分支。"
            )
        fta_items = [
            item
            for item in manifest
            if str(item.get("doc_type") or "") == "fta"
            and str(item.get("source_summary") or "").strip()
        ]
        if fta_items:
            anchor = fta_items[0]
            lines.append(
                "本轮已有可采用的 FTA 结构依据，正文必须先用它说明真实结构再展开候选："
                f"[^{anchor.get('citation_id')}] "
                + str(anchor.get("source_summary") or "")[:700]
                + "。主分支应沿该结构中明确出现的环节组织，不得用一个抽象标题混合互不相同的环节。"
            )
    resolution_limit = 3 if broad_candidate_presentation else 4
    resolution_items = [
        item for item in manifest if "resolution" in set(item.get("supports") or [])
    ][:resolution_limit]
    if resolution_items:
        lines.append("可用于检查/处置的直接字段：")
        for item in resolution_items:
            fields = item.get("answer_fields") if isinstance(item.get("answer_fields"), Mapping) else {}
            action_values = [
                str(fields.get(key) or "").strip()
                for key in (
                    "detection_method",
                    "detect_method",
                    "compensation_action",
                    "maintenance_action",
                    "repair_method",
                )
                if str(fields.get(key) or "").strip()
            ]
            action = "；".join(action_values) or str(item.get("source_summary") or "")[:320]
            lines.append(
                f"- [^{item.get('citation_id')}] {item.get('record_id')}：{action}"
            )
    return "\n".join(lines)


