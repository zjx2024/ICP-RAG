"""Record-family narrowing without generating answers or merging identities.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.answer_contract import MFL_CONTRACT_FIELDS
from backend.agent.answer_contract import required_answer_fields
from backend.agent.contracts import match_type_is_exact
from backend.agent.nodes.answer_collections import _closed_collection_groups
from backend.agent.nodes.answer_fault_rules import _query_requests_parallel_failure_modes
from backend.agent.nodes.answer_history_rules import _focused_history_explicit_references
from backend.agent.nodes.answer_history_rules import _history_evidence_is_strong
from backend.agent.nodes.answer_payload import _answer_relevance_score
from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_payload import _explicit_function_ids
from backend.agent.nodes.answer_payload import _selected_anchor_record_ids
from backend.agent.nodes.answer_presence import normalize_presence_text
from backend.agent.nodes.answer_text import _compact_semantic_text
from backend.agent.tools.icp_diagnosis import _mfl_module_fault_segments
from backend.agent.query_understanding import extract_business_codes
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
import re


def _narrow_semantic_table_families(
    state: Mapping[str, Any],
    evidence: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep secondary FMECA/MFL rows tied to the strongest explicit entity.

    PostgreSQL authority proves that a row exists, not that it answers this
    question.  Broad lexical recall can return another FMECA row merely because
    both mention “滤波”.  For semantic rows, use the strongest FMECA record as
    an anchor and retain secondary modes only when they share an explicit
    ``FUNC-*`` business ID.  Cross-family rows are retained only through an
    exact record selection or a persisted exact history reference.  Equal or
    overlapping failure prose is not an identity key: two independently
    authored tables may use the same description for different branches.
    Exact code/FTA/history references are never narrowed here and discarded
    rows remain available in the audit catalog.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    if (
        str(frame.get("coverage_scope") or "") == "complete_collection"
        and _closed_collection_groups(evidence)
    ):
        # A PostgreSQL-closed set is already the relevance boundary. Choosing
        # one semantically strongest FUNC-* anchor here would silently turn an
        # N-member inventory back into a Top-K subset.
        return list(evidence)
    # 因果链/关系路径类问题要求枚举发送、链路、协议等多个并列环节；用
    # 单一 FUNC 锚点收窄 FMECA 会把其他环节的记录静默删掉，答案退化成
    # 单一故障模式。planner 的字段合同是这种枚举意图的显式信号。
    if {
        "structural_cause_chain",
        "relation_path",
    } & set(required_answer_fields(state)):
        return list(evidence)
    structured_record_binding = (
        state.get("structured_record_binding")
        if isinstance(state.get("structured_record_binding"), Mapping)
        else {}
    )
    if structured_record_binding.get("ambiguous") is True:
        # An ambiguous exact-ID lookup is itself the answer boundary.  Choosing
        # one table row by semantic score here would turn a required
        # clarification into a silent identity binding.
        return list(evidence)

    focused_history_references = None
    if not _query_requests_parallel_failure_modes(state):
        focused_history_references = _focused_history_explicit_references(
            state, evidence
        )

    exact_history_references: Dict[str, set[str]] = {"fmeca": set(), "mfl": set()}
    if focused_history_references is not None:
        for doc_type in ("fmeca", "mfl"):
            exact_history_references[doc_type].update(
                focused_history_references.get(doc_type) or set()
            )
    if not _query_requests_parallel_failure_modes(state):
        for history_item in (() if focused_history_references is not None else evidence):
            if (
                str(history_item.get("doc_type") or "") != "fault_history"
                or not _history_evidence_is_strong(history_item)
            ):
                continue
            fields = _evidence_answer_fields(history_item)
            references = fields.get("explicit_references")
            if not isinstance(references, Mapping):
                continue
            for doc_type in ("fmeca", "mfl"):
                raw_values = references.get(doc_type)
                values = (
                    list(raw_values)
                    if isinstance(raw_values, (list, tuple, set))
                    else re.split(r"[,，;；\s]+", str(raw_values or ""))
                )
                exact_history_references[doc_type].update(
                    str(value or "").strip().upper()
                    for value in values
                    if str(value or "").strip()
                )

    def business_record_id(item: Mapping[str, Any]) -> str:
        fields = _evidence_answer_fields(item)
        return str(
            item.get("record_id") or fields.get("record_code") or ""
        ).strip().upper()

    frame_query = str(frame.get("original_query") or state.get("query") or "")
    current_user_codes = {
        str(value or "").strip().upper()
        for value in extract_business_codes(frame_query, limit=48)
        if str(value or "").strip()
    }

    def code_namespace(value: Any) -> str:
        """Return a schema-derived code namespace without classifying it."""

        text = str(value or "").strip().upper()
        parts = re.split(r"[-_./]", text, maxsplit=1)
        return parts[0] if len(parts) == 2 and parts[0] else ""

    def focused_history_row_ids(
        doc_type: str, rows: Sequence[Mapping[str, Any]]
    ) -> set[str] | None:
        """Resolve one history parent's links, preserving current ID conflicts.

        The original user turn is the only source of conflict identities.  IDs
        copied into rewritten queries or conversation anchors must not block the
        newer history link.  Family membership is learned from authoritative
        row/reference namespaces rather than from a hard-coded prefix table.
        """

        if focused_history_references is None:
            return None
        references = {
            str(value or "").strip().upper()
            for value in (focused_history_references.get(doc_type) or set())
            if str(value or "").strip()
        }
        if not references:
            return None
        linked = {
            str(item.get("evidence_id") or "")
            for item in rows
            if business_record_id(item) in references
            and str(item.get("evidence_id") or "")
        }
        # A stored link with no re-read authoritative row cannot authorize a
        # replacement.  Leave normal selection intact instead of erasing the
        # family or pretending that the unresolved reference was retrieved.
        if not linked:
            return None

        family_ids = {business_record_id(item) for item in rows}
        namespaces = {
            namespace
            for value in (*family_ids, *references)
            if (namespace := code_namespace(value))
        }
        user_family_codes = {
            code
            for code in current_user_codes
            if code in family_ids or code_namespace(code) in namespaces
        }
        unresolved_conflicts = user_family_codes - family_ids - references
        if unresolved_conflicts:
            # The user named a conflicting family ID but its authoritative row
            # is unavailable.  Do not silently rebind the turn to the history
            # child; the existing fail-closed selection/reporting path owns it.
            return None
        explicit_rows = {
            str(item.get("evidence_id") or "")
            for item in rows
            if business_record_id(item) in user_family_codes
            and str(item.get("evidence_id") or "")
        }
        return linked | explicit_rows

    all_fmeca_rows = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "fmeca"
        and str(item.get("authority") or "") == "authoritative_record"
    ]
    focused_history_fmeca_ids = focused_history_row_ids("fmeca", all_fmeca_rows)
    selected_fmeca_ids = _selected_anchor_record_ids(state, "fmeca")
    current_fmeca_rows = [
        item
        for item in all_fmeca_rows
        if business_record_id(item) in current_user_codes
    ]
    exact_target_fmeca = [
        item
        for item in all_fmeca_rows
        if not current_fmeca_rows
        and (
            selected_fmeca_ids
            and normalize_presence_text(business_record_id(item))
            in selected_fmeca_ids
        )
        or (
            match_type_is_exact(item.get("match_type"))
            and str(item.get("match_type") or "") in {
            "exact_conversation_reference",
            "exact_record_identity",
            "exact_structured_identity",
            }
        )
    ]
    kept_fmeca_ids: set[str] = set()
    strong_target_anchor = bool(
        focused_history_fmeca_ids is not None
        or current_fmeca_rows
        or exact_target_fmeca
    )
    if focused_history_fmeca_ids is not None:
        kept_rows = [
            item
            for item in all_fmeca_rows
            if str(item.get("evidence_id") or "") in focused_history_fmeca_ids
        ]
    elif current_fmeca_rows:
        kept_rows = current_fmeca_rows
    elif exact_target_fmeca:
        kept_rows = exact_target_fmeca
    else:
        # Other exact relations (for example an FTA leaf reference) are exact
        # as provenance but are not necessarily the user's current table-row
        # target.  Keep them in relevance ranking instead of dropping every
        # ``exact_*`` row wholesale.
        fmeca_rows = list(all_fmeca_rows)
        if len(fmeca_rows) < 2 or not any(
            isinstance(item.get("answer_rerank_score"), (int, float))
            for item in fmeca_rows
        ):
            return list(evidence)
        history_linked_fmeca = [
            item
            for item in fmeca_rows
            if business_record_id(item) in exact_history_references["fmeca"]
        ]
        anchor = max(history_linked_fmeca or fmeca_rows, key=_answer_relevance_score)
        anchor_functions = _explicit_function_ids(anchor)
        if not anchor_functions:
            return list(evidence)
        kept_rows = [
            item
            for item in fmeca_rows
            if (
                bool(history_linked_fmeca)
                and business_record_id(item) in exact_history_references["fmeca"]
            )
            or (
                not history_linked_fmeca
                and (item is anchor or (_explicit_function_ids(item) & anchor_functions))
            )
        ]
    for item in kept_rows:
        kept_fmeca_ids.add(str(item.get("evidence_id") or ""))

    selected_mfl_ids = _selected_anchor_record_ids(state, "mfl")
    all_mfl_rows = [
        item
        for item in evidence
        if str(item.get("doc_type") or "") == "mfl"
        and str(item.get("authority") or "") == "authoritative_record"
    ]
    focused_history_mfl_ids = focused_history_row_ids("mfl", all_mfl_rows)
    current_mfl_rows = [
        item
        for item in all_mfl_rows
        if business_record_id(item) in current_user_codes
    ]
    exact_target_mfl = [
        item
        for item in all_mfl_rows
        if not current_mfl_rows
        and (
            selected_mfl_ids
            and normalize_presence_text(business_record_id(item)) in selected_mfl_ids
        )
        or (
            match_type_is_exact(item.get("match_type"))
            and str(item.get("match_type") or "") in {
            "exact_conversation_reference",
            "exact_record_identity",
            "exact_structured_identity",
            }
        )
    ]
    if focused_history_mfl_ids is not None:
        exact_target_mfl = [
            item
            for item in all_mfl_rows
            if str(item.get("evidence_id") or "") in focused_history_mfl_ids
        ]
        mfl_rows: List[Dict[str, Any]] = []
    elif current_mfl_rows:
        exact_target_mfl = current_mfl_rows
        mfl_rows = [
            item for item in all_mfl_rows if item not in exact_target_mfl
        ]
    else:
        mfl_rows = [
            item
            for item in all_mfl_rows
            if item not in exact_target_mfl
        ]
    history_linked_mfl = [
        item
        for item in mfl_rows
        if business_record_id(item) in exact_history_references["mfl"]
    ]

    def explicit_references(item: Mapping[str, Any], doc_type: str) -> set[str]:
        references = _evidence_answer_fields(item).get("explicit_references")
        if not isinstance(references, Mapping):
            return set()
        raw_values = references.get(doc_type)
        values = (
            list(raw_values)
            if isinstance(raw_values, (list, tuple, set))
            else re.split(r"[,，;；\s]+", str(raw_values or ""))
        )
        return {
            str(value or "").strip().upper()
            for value in values
            if str(value or "").strip()
        }

    kept_fmeca_record_ids = {
        business_record_id(item) for item in kept_rows if business_record_id(item)
    }
    explicitly_linked_mfl_ids = {
        str(mfl_item.get("evidence_id") or "")
        for mfl_item in mfl_rows
        if str(mfl_item.get("evidence_id") or "")
        and (
            business_record_id(mfl_item)
            in {
                reference
                for fmeca_item in kept_rows
                for reference in explicit_references(fmeca_item, "mfl")
            }
            or bool(
                explicit_references(mfl_item, "fmeca")
                & kept_fmeca_record_ids
            )
        )
    }
    matching_mfl_ids = {
        str(item.get("evidence_id") or "") for item in exact_target_mfl
    }
    matching_mfl_ids.update(explicitly_linked_mfl_ids)
    matching_mfl_ids.update(
        str(item.get("evidence_id") or "")
        for item in mfl_rows
        if (
            bool(history_linked_mfl)
            and business_record_id(item) in exact_history_references["mfl"]
        )
    )
    # MFL 合同维度在本轮被点名时，MFL 行是这些维度的权威属主。已获门禁批准、
    # 且与保留 FMECA 行共享同一故障模式词的 MFL 行不按“语义表族”丢弃：
    # ``module_fault`` 的权威存储形态是“模块名：故障模式”，其故障模式段与
    # FMECA ``failure_mode`` 逐字相等是两表共用的显式词法身份（与
    # ``_mfl_identity_terms`` 同一机制），不是相似度合并。无此保留时，一个
    # 强 FMECA 锚点会把整族 MFL 行从答案上下文剔除，维修合同只能靠 FMECA
    # 的补偿措施字面凑数。
    if MFL_CONTRACT_FIELDS & set(required_answer_fields(state)):
        approved_ids = {
            str(value)
            for value in (
                (state.get("safety_decision") or {}).get(
                    "approved_evidence_ids"
                )
                or []
            )
            if str(value)
        }
        kept_failure_modes = {
            mode
            for item in kept_rows
            if (
                mode := _compact_semantic_text(
                    _evidence_answer_fields(item).get("failure_mode")
                )
            )
        }
        if kept_failure_modes:
            for item in all_mfl_rows:
                evidence_id = str(item.get("evidence_id") or "")
                if (
                    evidence_id in matching_mfl_ids
                    or evidence_id not in approved_ids
                ):
                    continue
                # 分段必须用原始文本：_compact_semantic_text 会剥离全角冒号，
                # compact 之后再分段会把模块名并进故障模式段。
                module_name, fault_mode = _mfl_module_fault_segments(
                    _evidence_answer_fields(item).get("module_failure")
                )
                compact_mode = _compact_semantic_text(fault_mode)
                if compact_mode and compact_mode in kept_failure_modes:
                    matching_mfl_ids.add(evidence_id)

    narrowed: List[Dict[str, Any]] = []
    for item in evidence:
        doc_type = str(item.get("doc_type") or "")
        match_type = str(item.get("match_type") or "")
        evidence_id = str(item.get("evidence_id") or "")
        if (
            doc_type == "fmeca"
            and item in all_fmeca_rows
            and evidence_id not in kept_fmeca_ids
        ):
            continue
        if (
            doc_type == "mfl"
            and item in all_mfl_rows
            and evidence_id not in matching_mfl_ids
            and (strong_target_anchor or matching_mfl_ids or history_linked_mfl)
        ):
            continue
        narrowed.append(item)
    return narrowed
