"""ICP 专用故障诊断编排工具。

该节点不是让大模型自由猜测，而是根据问题类型调用七类文档的专用检索逻辑：
Function Locator、HMC、FMECA、FTA、RBD、MFL、故障历史，并输出可追溯候选。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.agent.answer_contract import required_answer_fields
from backend.agent.contracts import answer_mode_for_intent, make_evidence, make_tool_result
from backend.agent.evidence_display import (
    deterministic_graph_source_ref,
    summarize_diagram_locator,
    summarize_graph_structure,
)
from backend.agent.query_understanding import (
    bind_structured_function_entity,
    build_hybrid_query,
    build_tool_terms,
    closed_inventory_requested,
    diagram_match_score,
    extract_business_codes,
    normalize_search_term,
)
from backend.agent.tool_policy import extract_hmc_components, resolve_tool_policy
from backend.core.retrieval import knowledge_retriever
from backend.core.parsers.document_ir import clean_text, load_document_ir
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import (
    FORMAL_VALIDATION_STATUSES,
    VALIDATION_FAILED,
    VALIDATION_PROVISIONAL,
    VALIDATION_QUARANTINED,
)
from backend.icp.entity_resolution import normalize_business_id
from backend.icp.repository import icp_repository
from backend.icp.structured_graph import business_node_id, document_text_node_id
from backend.storage.postgres_client import postgres_client
from backend.storage.artifact_store import artifact_store, storage_key_from

logger = logging.getLogger(__name__)

ICP_DOC_TYPES = (
    "function_locator",
    "hmc",
    "fmeca",
    "fta",
    "rbd",
    "mfl",
    "fault_history",
)
_TOOL_CALL_EVIDENCE_LIMIT = 8
_TOOL_CALL_PREVIEW_SNIPPET_LIMIT = 260
_TOOL_CALL_PREVIEW_SOURCE_LIMIT = 320
_GRAPH_SEED_LIMIT = 20
_MAX_RECORD_HYDRATION_PER_TYPE = 16
# A natural-language symptom may cover several independent branches (for
# example an A/B supply pair) even when the user does not know their stored
# identifiers.  Semantic recall may discover those rows, but PostgreSQL must
# re-read two distinct top locators so a second directly relevant branch is
# not lost, while a third weak semantic neighbour cannot enter the diagnosis
# merely to fill the answer. These are per-family hard caps, not an unbounded
# table scan; history remains deliberately conservative.
_DIAGNOSTIC_RECORD_HYDRATION_LIMITS = {
    "hmc": 2,
    "fmeca": 2,
    "mfl": 2,
}
_STRUCTURED_HMC_MATCH_TYPES = {"exact_code", "exact_components"}
_SEMANTIC_HISTORY_REFERENCE_MIN_SCORE = 0.25
_FTA_DIAGRAM_BUSINESS_ID_RE = re.compile(
    r"FTA-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", re.IGNORECASE
)
_FTA_EVENT_BUSINESS_ID_RE = re.compile(
    r"(?:EVT|FTA)-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", re.IGNORECASE
)
_RBD_DIAGRAM_BUSINESS_ID_RE = re.compile(
    r"RBD-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", re.IGNORECASE
)
_FMECA_BUSINESS_ID_RE = re.compile(
    r"(?:FM|FMECA)-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.IGNORECASE,
)
_SQL_BUSINESS_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z]{2,}[A-Za-z0-9]*(?:[._/-][A-Za-z0-9*]+)+"
    r"|\d{4,}(?:\*{2}[A-Za-z0-9]+|[._/-][A-Za-z0-9*]+)+"
    r"|\d{6,}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# 召回预算取 OR 语义下的“覆盖面”而非精度：排名由 _rank_table_rows 负责。
# 8 个槽位时，规划模型先给出的业务 ID 与整句改写（各占 6-16 字符）就能
# 占满预算，把排在术语列表后部、真正与结构化单元格命中的短锚点（如
# “输出通道”）挤出 SQL 召回，权威行只在术语顺序恰好有利时才被召回。
# 13 列 × 16 词 = 208 个 ILIKE 参数，仍在 PostgreSQL 合理绑定范围内。
_SQL_TERM_BUDGET = 16
_SQL_TERM_SOURCE_LIMIT = 32
_SQL_SHORT_ANCHOR_MAX_LENGTH = 16

_IDENTITY_FIELDS_BY_TOOL = {
    "function_locator": ("function_id", "component_id", "interface_id"),
    "hmc_query": ("hmc_code",),
    "fmeca_query": ("code", "record_code", "failure_mode_id"),
    "mfl_query": ("mfl_code", "record_code"),
    "fault_history_query": ("case_id",),
}
_GRAPH_LOCATOR_CONTENT_TYPES = {
    "fta_structured_facts",
    "fta_diagram_candidate_summary",
    "rbd_structured_facts",
    "rbd_diagram_candidate_summary",
    "function_diagram_candidate_summary",
    # Legacy projections remain discoverable but are still re-authorized by PG.
    "fta_node",
    "fta_edge",
    "fta_diagram",
    "rbd_node",
    "rbd_edge",
    "rbd_diagram",
    "function_architecture_diagram",
    "function_architecture_edge",
}


def _graph_validation_status(row: Dict[str, Any]) -> str:
    """Return the machine-validation status, with a legacy human fallback.

    An explicit non-formal automatic status always wins over the historical
    ``review_status`` field.  Only old rows which predate validation metadata
    may fall back from ``review_status=confirmed`` to manual confirmation.
    """

    validation_status = str(row.get("validation_status") or "").strip()
    if validation_status in {VALIDATION_PROVISIONAL, VALIDATION_QUARANTINED, VALIDATION_FAILED}:
        return validation_status
    if validation_status in FORMAL_VALIDATION_STATUSES:
        return validation_status
    if row.get("review_status") == "confirmed":
        return "manual_confirmed"
    return validation_status or str(row.get("review_status") or "pending").strip()


def _is_context_graph_row(row: Dict[str, Any]) -> bool:
    return (
        _graph_validation_status(row)
        in {*FORMAL_VALIDATION_STATUSES, VALIDATION_PROVISIONAL}
        and row.get("projection_status") == "published"
        and row.get("is_active") is True
    )


def _is_formal_graph_row(row: Dict[str, Any]) -> bool:
    return (
        _graph_validation_status(row) in FORMAL_VALIDATION_STATUSES
        and row.get("projection_status") == "published"
        and row.get("is_active") is True
    )

def _unique_terms(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), dict) else {}
    for key in ("entities", "symptoms", "normalized_terms"):
        for value in frame.get(key, []) or []:
            if value:
                values.append(str(value).strip())
    for key in ("operating_stage", "temporal_pattern"):
        if frame.get(key):
            values.append(str(frame[key]).strip())
    for value in state.get("normalized_terms", []) or []:
        if value:
            values.append(str(value).strip())
    for value in state.get("entity_keywords", []) or []:
        if value:
            values.append(str(value).strip())
    rewritten = str(state.get("rewritten_query") or "").strip()
    original = str(state.get("query") or "").strip()
    if rewritten:
        values.append(rewritten)
    if original and original != rewritten:
        values.append(original)

    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        identity = normalize_search_term(value)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        result.append(value)
    return result[:32]


def _binding_can_drive_cross_document_expansion(
    state: Mapping[str, Any], binding: Mapping[str, Any]
) -> bool:
    """Whether a Function Locator match is specific enough to become a seed.

    A broad locator hit is useful for text recall, but it must not become the
    single authoritative hand-off for a graph/relation question.  Otherwise a
    phrase such as “供电模块” can make an arbitrary child (for example HUM)
    hide the A/B branches the user actually asked about.  This gate does not
    infer intent from wording: it consumes the planner's typed relation/topology
    contract and the binder's deterministic match basis.
    """

    if not binding.get("bound"):
        return False
    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    relation = state.get("relation_expansion")
    relation_needed = bool(
        isinstance(relation, Mapping)
        and (
            relation.get("needed") is True
            or str(relation.get("needed") or "").strip().casefold()
            in {"1", "true", "yes"}
        )
    )
    required_fields = {
        str(value or "").strip()
        for value in (frame.get("required_answer_fields") or [])
        if str(value or "").strip()
    }
    topology = frame.get("topology_dependency")
    relation_needed = relation_needed or bool(
        {"impact_propagation", "structural_cause_chain"} & required_fields
    ) or bool(isinstance(topology, Mapping) and topology.get("required") is True)
    if not relation_needed:
        return True
    # The identifier itself, an exact stored name, an explicit route/channel
    # marker, or an explicit same-row structural chain can safely carry a
    # cross-document query. Generic containment/similarity remains retrieval
    # context only and must wait for semantic rows to provide a real ID.
    return str(binding.get("basis") or "") in {
        "exact_business_id",
        "exact_identity_label",
        "explicit_branch_identity_overlap",
        "same_row_module_function_chain",
    }


def _apply_structured_entity_binding(
    state: Mapping[str, Any],
    function_rows: Sequence[Mapping[str, Any]],
    terms_by_tool: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Hand one authoritative Function Locator object to later queries.

    The binder only considers stored identity/name/abbreviation/alias fields.
    A successful binding moves those exact terms ahead of broad planner terms
    for FMECA and RBD.  It does not claim that a similarly worded record in
    another document is the same entity; formal cross-document links still
    require equal business IDs or PostgreSQL-authorized graph assertions.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    binding = bind_structured_function_entity(
        [
            frame.get("entities") or [],
            frame.get("normalized_terms") or [],
            terms_by_tool.get("function_locator") or [],
            state.get("rewritten_query") or "",
            state.get("query") or "",
        ],
        function_rows,
    )
    if not binding.get("bound"):
        return [dict(row) for row in function_rows], binding
    if not _binding_can_drive_cross_document_expansion(state, binding):
        return [dict(row) for row in function_rows], {
            "bound": False,
            "ambiguous": False,
            "business_ids": [],
            "terms": [],
            "row_indexes": [],
            "basis": "generic_locator_match_not_used_as_cross_document_seed",
        }

    entity_terms = [
        str(value or "").strip()
        for value in binding.get("terms") or []
        if str(value or "").strip()
    ]
    for tool in ("fmeca_query", "rbd_query"):
        ordered = [*entity_terms, *(terms_by_tool.get(tool) or [])]
        deduplicated: List[str] = []
        seen: set[str] = set()
        for value in ordered:
            identity = normalize_search_term(value)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(value)
        terms_by_tool[tool] = deduplicated[:24]

    # A Function Locator match is an authoritative identity hand-off.  Pass
    # only its stored exact IDs to the remaining record families; this expands
    # board/API/history diagnostics without creating a cross-document merge
    # from similar prose or a model-generated synonym.
    business_ids = [
        str(value or "").strip()
        for value in binding.get("business_ids") or []
        if str(value or "").strip()
    ]
    for tool in ("hmc_query", "mfl_query", "fault_history_query"):
        ordered = [*business_ids, *(terms_by_tool.get(tool) or [])]
        deduplicated = []
        seen = set()
        for value in ordered:
            identity = normalize_search_term(value)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(value)
        terms_by_tool[tool] = deduplicated[:24]

    selected_indexes = {
        int(value)
        for value in binding.get("row_indexes") or []
        if isinstance(value, int) or str(value).isdigit()
    }
    selected_rows = [
        dict(row) for index, row in enumerate(function_rows) if index in selected_indexes
    ]
    return selected_rows or [dict(row) for row in function_rows], binding


def _shared_fmeca_identity_terms(
    rows: Sequence[Mapping[str, Any]],
    *,
    structured_business_ids: Sequence[Any] = (),
) -> List[str]:
    """Return the uniquely dominant exact identity shared by FMECA hits.

    This is a bounded query handoff for an implicit topology question.  It
    does not merge records by wording: at least two PostgreSQL FMECA rows must
    carry the same complete set of explicit codes stored in ``product_flag``.
    A weak unrelated Top-K row must not erase that agreement, while two
    equally supported identity clusters fail closed instead of allowing result
    order to choose which RBD family is queried.
    """

    if len(rows) < 2:
        return []
    required_ids = {
        normalize_search_term(value)
        for value in structured_business_ids
        if normalize_search_term(value)
    }
    clusters: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    rendered_codes: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        row_codes: Dict[str, str] = {}
        for code in extract_business_codes(row.get("product_flag") or ""):
            normalized = normalize_search_term(code)
            if not normalized:
                continue
            row_codes.setdefault(normalized, str(code).strip())
        if not row_codes:
            continue
        if required_ids and not (set(row_codes) & required_ids):
            continue
        identity = tuple(sorted(row_codes))
        clusters.setdefault(identity, []).append(row)
        rendered_codes.setdefault(identity, row_codes)
    if not clusters:
        return []

    positive_score_present = any(
        _retrieval_score(row.get("retrieval_identity_score")) > 0
        for cluster_rows in clusters.values()
        for row in cluster_rows
    )
    voting_clusters: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for identity, cluster_rows in clusters.items():
        if positive_score_present:
            voters = [
                row
                for row in cluster_rows
                if _retrieval_score(row.get("retrieval_identity_score")) > 0
            ]
        else:
            # Tests and legacy direct callers may not carry a ranking score.
            # An explicit zero is still a non-vote; an absent field preserves
            # the former exact-code consensus behavior.
            voters = [
                row
                for row in cluster_rows
                if "retrieval_identity_score" not in row
                or row.get("retrieval_identity_score") is None
            ]
        if voters:
            voting_clusters[identity] = voters
    if not voting_clusters:
        return []

    ranked = sorted(
        voting_clusters.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(
                _retrieval_score(row.get("retrieval_identity_score"))
                for row in item[1]
            ),
            item[0],
        ),
    )
    winning_identity, winning_rows = ranked[0]
    winning_votes = len(winning_rows)
    if winning_votes < 2:
        return []
    if len(ranked) > 1 and len(ranked[1][1]) == winning_votes:
        return []

    terms = [
        rendered_codes[winning_identity][value]
        for value in winning_identity
    ]
    functions = {
        normalize_search_term(row.get("func_management"))
        for row in winning_rows
        if normalize_search_term(row.get("func_management"))
    }
    if len(functions) == 1:
        function_value = next(
            str(row.get("func_management") or "").strip()
            for row in winning_rows
            if normalize_search_term(row.get("func_management")) in functions
        )
        if function_value:
            terms.append(function_value)
    return terms[:8]


def _prepend_query_terms(
    existing: Sequence[Any], additions: Sequence[Any], *, limit: int = 24
) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in [*additions, *existing]:
        text = str(value or "").strip()
        normalized = normalize_search_term(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
        if len(result) >= max(0, int(limit)):
            break
    return result


def _enriched_topology_dependency(
    dependency: Any,
    structured_binding: Mapping[str, Any],
    fmeca_identity_terms: Sequence[Any],
) -> Dict[str, Any]:
    """Attach only persisted, exact business codes to a topology contract.

    Natural-language names stay retrieval hints. They are deliberately not
    copied into ``canonical_business_ids`` because the answer layer uses this
    field as the cross-document RBD authorization boundary.
    """

    if not isinstance(dependency, Mapping) or dependency.get("required") is not True:
        return {}
    candidates: List[Any] = [
        *(dependency.get("canonical_business_ids") or []),
        *(structured_binding.get("business_ids") or []),
        *extract_business_codes(structured_binding.get("terms") or []),
        *extract_business_codes(fmeca_identity_terms),
    ]
    business_ids: List[str] = []
    seen: set[str] = set()
    for value in candidates:
        rendered = str(value or "").strip()
        identity = normalize_search_term(rendered)
        if not rendered or not identity or identity in seen:
            continue
        seen.add(identity)
        business_ids.append(rendered)
    return {
        **dict(dependency),
        "canonical_business_ids": business_ids[:16],
        "identity_basis": (
            "PostgreSQL 结构化业务ID精确交接" if business_ids else "未绑定精确业务ID"
        ),
    }


def _closed_collection_scope_seeds(
    scope_candidates: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Tuple[List[str], List[str], str]:
    """Resolve an explicit collection subject to authoritative Function IDs.

    Exact business IDs and unique exact labels win.  A cautious lexical linker
    may then map a model-normalized entity to one returned component/function
    candidate, but only with a high score and clear margin.  A model may also
    explicitly list more than one current subject (for example the A and B
    supply branches).  Those exact, same-document identities are a bounded
    union; contextual Top-K children and stale conversation anchors are never
    unioned to prove a parent collection.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), Mapping) else {}
    subject = str(frame.get("collection_subject") or "").strip()
    raw_entities = frame.get("entities") or []
    entity_values = (
        [str(value or "").strip() for value in raw_entities]
        if isinstance(raw_entities, (list, tuple, set))
        else []
    )
    business_id_pattern = re.compile(
        r"(?:FUNC|MOD|SUB|SYS)-[A-Z0-9]+(?:[._-][A-Z0-9]+)*",
        re.IGNORECASE,
    )
    normalized_subject = normalize_search_term(subject)
    subject_business_ids = [
        match.group(0)
        for match in business_id_pattern.finditer(subject)
    ]
    entity_business_ids = [
        match.group(0)
        for value in entity_values
        for match in business_id_pattern.finditer(value)
    ]
    subject_related_entities = [
        value
        for value in entity_values
        if normalize_search_term(value)
        and normalized_subject
        and normalize_search_term(value) in normalized_subject
    ]
    if not normalized_subject and entity_values:
        subject_related_entities = entity_values[:1]

    anchor_parent_ids: List[str] = []
    anchor_fallback_ids: List[str] = []
    for anchor in state.get("selected_conversation_anchors") or []:
        if not isinstance(anchor, Mapping):
            continue
        fields = (
            anchor.get("answer_fields")
            if isinstance(anchor.get("answer_fields"), Mapping)
            else {}
        )
        parent_id = str(fields.get("collection_parent_id") or "").strip()
        if business_id_pattern.fullmatch(parent_id):
            anchor_parent_ids.append(parent_id)
        for raw_value in (
            anchor.get("record_id"),
            fields.get("component_id"),
            fields.get("function_id"),
        ):
            value = str(raw_value or "").strip()
            if business_id_pattern.fullmatch(value):
                anchor_fallback_ids.append(value)

    link_targets = list(
        dict.fromkeys(
            normalize_search_term(value)
            for value in [subject, *entity_values]
            if len(normalize_search_term(value)) >= 4
        )
    )
    if not any(
        (
            subject_business_ids,
            subject,
            anchor_parent_ids,
            anchor_fallback_ids,
            entity_business_ids,
            subject_related_entities,
            link_targets,
        )
    ):
        return [], [], ""

    candidates: Dict[Tuple[str, str, str], Dict[str, set[str]]] = {}
    for raw_row in scope_candidates:
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        if "active_scope_version_count" in row:
            try:
                active_version_count = int(row.get("active_scope_version_count") or 0)
            except (TypeError, ValueError):
                active_version_count = 0
            if (
                active_version_count != 1
                or not str(row.get("applicable_system") or "").strip()
                or not str(row.get("applicable_configuration") or "").strip()
            ):
                continue
        row_doc_id = str(row.get("doc_id") or "").strip()
        kind = str(row.get("collection_entity_kind") or "").strip().lower()
        entity_id = str(row.get("collection_entity_id") or "").strip()
        if not kind:
            if row.get("component_id"):
                kind, entity_id = "component", str(row.get("component_id") or "").strip()
            elif row.get("function_id"):
                kind, entity_id = "function", str(row.get("function_id") or "").strip()
        if kind not in {"component", "function"} or not entity_id or not row_doc_id:
            continue
        key = (kind, entity_id, row_doc_id)
        values = candidates.setdefault(key, {"ids": set(), "labels": set()})
        values["ids"].add(normalize_search_term(entity_id))
        label_fields = (
            (
                "collection_entity_name",
                "collection_entity_abbreviation",
                "name",
                "abbreviation",
                "module_name",
            )
            if kind == "component"
            else ("collection_entity_name", "function_name")
        )
        values["labels"].update(
            normalize_search_term(row.get(field))
            for field in label_fields
            if normalize_search_term(row.get(field))
        )

    def unique_id_match(values: Sequence[str]) -> Tuple[Optional[Tuple[str, str, str]], bool]:
        keys = {
            normalize_search_term(value)
            for value in values
            if normalize_search_term(value)
        }
        matches = [
            key for key, candidate in candidates.items() if keys & candidate["ids"]
        ]
        return (matches[0], False) if len(matches) == 1 else (None, len(matches) > 1)

    def unique_label_match(values: Sequence[str]) -> Tuple[Optional[Tuple[str, str, str]], bool]:
        keys = {
            normalize_search_term(value)
            for value in values
            if normalize_search_term(value)
        }
        matches = [
            key for key, candidate in candidates.items() if keys & candidate["labels"]
        ]
        return (matches[0], False) if len(matches) == 1 else (None, len(matches) > 1)

    def unique_explicit_entity_match(
        value: Any,
    ) -> Tuple[Optional[Tuple[str, str, str]], bool]:
        """Resolve one model-listed current entity without fuzzy expansion."""

        text = str(value or "").strip()
        if not text:
            return None, False
        ids = [match.group(0) for match in business_id_pattern.finditer(text)]
        if ids:
            matched_by_id, ambiguous_by_id = unique_id_match(ids)
            if matched_by_id is not None or ambiguous_by_id:
                return matched_by_id, ambiguous_by_id
        return unique_label_match([text])

    matched: Optional[Tuple[str, str, str]] = None

    # Resolution precedence is semantic, not result-order based:
    # 1) an ID explicitly present in the current collection subject;
    # 2) an exact unique label in that subject;
    # 3) the unique parent collection carried by answer-used citations;
    # 4) other planner entities and finally cautious lexical linking.
    # This prevents member FUNC IDs from outvoting their cited SUB parent while
    # still allowing a newly typed current-turn ID to override old context.
    if subject_business_ids:
        matched, ambiguous = unique_id_match(subject_business_ids)
        if ambiguous or matched is None:
            return [], [], ""

    if matched is None and subject:
        matched, ambiguous = unique_label_match([subject])
        if ambiguous:
            return [], [], ""

    anchor_parent_matches_subject = any(
        normalize_search_term(parent_id) in candidate.get("ids", set())
        and any(
            label and label in normalized_subject
            for label in candidate.get("labels", set())
        )
        for parent_id in anchor_parent_ids
        for candidate in candidates.values()
        if normalized_subject
    )
    if matched is None and not anchor_parent_matches_subject:
        # ``entities`` is authored from the *current* user turn by the target
        # model.  When it names several exact, independent Function Locator
        # objects, preserve each one instead of letting an older single-parent
        # anchor collapse a dual-path question to the first route.  Every
        # object must resolve uniquely and all must belong to the same active
        # architecture document/configuration; otherwise leave the collection
        # open rather than guessing a union.
        current_matches: List[Tuple[str, str, str]] = []
        seen_current_matches: set[Tuple[str, str, str]] = set()
        for value in entity_values:
            current_match, current_ambiguous = unique_explicit_entity_match(value)
            if current_ambiguous:
                # A broad label in the entity list (for example a subsystem
                # abbreviation) must not veto a separate, exact current
                # object.  It simply cannot participate in the union.
                continue
            if current_match is not None and current_match not in seen_current_matches:
                seen_current_matches.add(current_match)
                current_matches.append(current_match)
        current_docs = {item[2] for item in current_matches}
        if len(current_matches) >= 2 and len(current_docs) == 1:
            component_ids = [
                entity_id
                for kind, entity_id, _ in current_matches
                if kind == "component"
            ]
            function_ids = [
                entity_id
                for kind, entity_id, _ in current_matches
                if kind == "function"
            ]
            if component_ids or function_ids:
                return component_ids, function_ids, next(iter(current_docs))

    if matched is None:
        normalized_parent_ids = list(dict.fromkeys(
            normalize_search_term(value)
            for value in anchor_parent_ids
            if normalize_search_term(value)
        ))
        if len(normalized_parent_ids) > 1:
            return [], [], ""
        if normalized_parent_ids:
            matched, ambiguous = unique_id_match(anchor_parent_ids)
            if ambiguous or matched is None:
                return [], [], ""
        elif anchor_fallback_ids and not subject_business_ids:
            normalized_fallback_ids = list(dict.fromkeys(
                normalize_search_term(value)
                for value in anchor_fallback_ids
                if normalize_search_term(value)
            ))
            if len(normalized_fallback_ids) == 1:
                matched, ambiguous = unique_id_match(anchor_fallback_ids)
                if ambiguous:
                    return [], [], ""

    if matched is None and entity_business_ids:
        matched, ambiguous = unique_id_match(entity_business_ids)
        if ambiguous:
            return [], [], ""

    if matched is None and subject_related_entities:
        matched, ambiguous = unique_label_match(subject_related_entities)
        if ambiguous:
            return [], [], ""

    if matched is None:
        # This is query-to-entity linking, not cross-document entity merging.
        # It is evaluated against the complete active identity catalogue, so a
        # relevance window cannot hide a competing entity.  High confidence
        # and a clear global margin are both mandatory; otherwise the
        # collection stays open and the user can clarify the object.
        ranked: List[Tuple[float, Tuple[str, str, str]]] = []
        for key, values in candidates.items():
            labels = [value for value in values["labels"] if len(value) >= 4]
            score = max(
                (
                    SequenceMatcher(None, target, label).ratio()
                    for target in link_targets
                    for label in labels
                ),
                default=0.0,
            )
            ranked.append((score, key))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or ranked[0][0] < 0.90:
            return [], [], ""
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08:
            return [], [], ""
        matched = ranked[0][1]

    kind, entity_id, doc_id = matched
    if kind == "component":
        return [entity_id], [], doc_id
    if kind == "function":
        return [], [entity_id], doc_id
    return [], [], ""


def _extract_hmc(query: str) -> Dict[str, str]:
    return extract_hmc_components(query)


def _row_dicts(cur, rows) -> List[Dict[str, Any]]:
    columns = [desc[0] for desc in cur.description]
    return [{name: value for name, value in zip(columns, row)} for row in rows]


def _sql_candidate_terms(
    terms: Sequence[str], *, limit: int = _SQL_TERM_BUDGET
) -> List[str]:
    """Select bounded SQL recall terms without dropping precise anchors.

    Planner output may contain several complete natural-language questions
    before an entity ID such as ``FUNC-110``. Expanding every question into a
    raw and normalized form before applying the SQL budget used to consume all
    eight slots and silently discard that ID. Extract explicit codes from the
    entire bounded input first, then keep short anchors, and leave complete
    long questions as fallbacks. This is lexical prioritization only: it does
    not infer aliases, entity identity, or domain intent.
    """

    try:
        budget = max(int(limit), 0)
    except (TypeError, ValueError):
        budget = 0
    if budget == 0:
        return []

    raw_terms = [
        str(term or "").strip()
        for term in terms[:_SQL_TERM_SOURCE_LIMIT]
        if str(term or "").strip()
    ]
    identifiers: List[str] = []
    identifier_fallbacks: List[str] = []
    short_anchors: List[str] = []
    long_phrases: List[str] = []
    seen: set[str] = set()

    def append(bucket: List[str], value: Any) -> None:
        text = str(value or "").strip()
        identity = text.casefold()
        if len(text) < 2 or not identity or identity in seen:
            return
        seen.add(identity)
        bucket.append(text)

    # Scan all bounded planner terms before spending the candidate budget so
    # an ID near the end cannot be displaced by earlier question expansions.
    for raw in raw_terms:
        for match in _SQL_BUSINESS_CODE_RE.finditer(raw):
            code = match.group(0)
            append(identifiers, code)
            normalized_code = _normalize_match_term(code)
            if normalized_code != code.casefold():
                append(identifier_fallbacks, normalized_code)

    for raw in raw_terms:
        normalized = _normalize_match_term(raw)
        full_code = _SQL_BUSINESS_CODE_RE.fullmatch(raw)
        compact_technical_code = bool(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,15}", raw)
            and (raw.upper() == raw or any(char.isdigit() for char in raw))
        )
        if full_code or compact_technical_code:
            append(identifiers, raw)
            if normalized != raw.casefold():
                append(identifier_fallbacks, normalized)
            continue

        bucket = (
            short_anchors
            if len(normalized) <= _SQL_SHORT_ANCHOR_MAX_LENGTH
            else long_phrases
        )
        # Natural-language punctuation and whitespace are not stored in many
        # structured cells, so the normalized variant remains the primary one.
        append(bucket, normalized)
        append(bucket, raw)
        # 复合锚点常带拉丁模块前缀（“ICU 输入过压钳位失效”），而 MFL 等表
        # 只存纯中文片段（“输入调理模块：输入过压钳位失效”）。派生最长纯
        # CJK 片段属于词法归一化，不推断别名或实体身份；只对短锚点生效，
        # 避免从整句问题派生碎片、挤占 16 槽召回预算。
        if len(normalized) <= _SQL_SHORT_ANCHOR_MAX_LENGTH:
            cjk_runs = re.findall(r"[\u4e00-\u9fff]{5,}", normalized)
            if cjk_runs:
                fragment = max(cjk_runs, key=len)
                if fragment != normalized:
                    append(short_anchors, fragment)

    return [
        *identifiers,
        *short_anchors,
        *identifier_fallbacks,
        *long_phrases,
    ][:budget]


def _exact_identity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tool: str,
    business_codes: Sequence[str],
) -> List[Dict[str, Any]]:
    """Keep rows whose stored identity exactly matches an explicit code.

    The same syntax is used across document families, but no prefix-to-table
    mapping is assumed.  This final equality check is what makes the bounded
    cross-table probe authoritative instead of a five-source keyword search.
    """

    fields = _IDENTITY_FIELDS_BY_TOOL.get(tool, ())
    expected = {
        normalize_search_term(value)
        for value in business_codes
        if normalize_search_term(value)
    }
    if not fields or not expected:
        return []
    result: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        identities = {
            normalize_search_term(row.get(field))
            for field in fields
            if normalize_search_term(row.get(field))
        }
        if identities & expected:
            result.append(row)
    return result


def _hmc_from_terms(terms: Sequence[str]) -> Dict[str, str]:
    """Extract an LCN/HSC component pair from planner-produced search terms.

    The planner's ``terms`` are explicit tokens, not free-form prose.  When a
    rewrite dropped the ``LCN/HSC`` labels from the query text, this fallback
    rebuilds the exact-component match from unambiguous digit shapes: the
    longest 5-8 digit run is treated as LCN, the following 1-4 digit run as
    HSC (zero-padded to four).  Only digit combinations are accepted; any other
    occurrence leaves the query in keyword/fallback mode (fail-closed).
    """

    numbers: List[str] = []
    for term in terms:
        for match in re.finditer(r"(?<!\d)(\d{1,8})(?!\d)", str(term or "")):
            value = match.group(1)
            if value and (not numbers or value != numbers[0]):
                numbers.append(value)
    if len(numbers) < 2:
        return {}
    # 最长数字串视为 LCN，其余取第一个作为 HSC
    lcn = max(numbers, key=len)
    hsc_candidates = [value for value in numbers if value != lcn]
    if not lcn or not hsc_candidates:
        return {}
    hsc = hsc_candidates[0]
    if hsc.isdigit() and len(hsc) < 4:
        hsc = hsc.zfill(4)
    return {"lcn": lcn, "hsc": hsc, "hamc": ""}


def _query_hmc(
    query: str,
    terms: Sequence[str],
    limit: int,
    errors: Optional[List[str]] = None,
    *,
    include_repair_method: bool = True,
) -> List[Dict[str, Any]]:
    exact = _extract_hmc(query)
    if not exact:
        # 用户原文可能被规划器改写/归一化而丢失 LCN/HSC 标签，导致上面的
        # 标签解析失败。terms 由规划器显式拆出、不含自然语言噪音；当其中
        # 同时出现明确的 LCN（≥5 位数字）与 HSC（≤4 位数字）时，按字段等价
        # 匹配走组件精确查询，而不是退化成关键词/语义检索。没有明确数字
        # 组合时不猜代码，保持 fail-closed。
        exact = _hmc_from_terms(terms)
    try:
        with postgres_client.conn.cursor() as cur:
            if exact:
                if exact.get("hamc"):
                    match_type = "exact_code"
                    cur.execute(
                        """
                        SELECT h.* FROM fault_hmc h
                        JOIN icp_documents d ON d.doc_id=h.doc_id
                        WHERE d.status IN ('indexed','kg_pending','kg_done')
                          AND h.lcn=%s AND h.hamc=%s AND h.hsc=%s
                        ORDER BY h.updated_at DESC LIMIT %s
                        """,
                        (exact["lcn"], exact["hamc"], exact["hsc"], limit),
                    )
                else:
                    # HAMC was not supplied.  Match only the two fields the
                    # user actually provided; do not silently manufacture **.
                    match_type = "exact_components"
                    cur.execute(
                        """
                        SELECT h.* FROM fault_hmc h
                        JOIN icp_documents d ON d.doc_id=h.doc_id
                        WHERE d.status IN ('indexed','kg_pending','kg_done')
                          AND h.lcn=%s AND h.hsc=%s
                        ORDER BY h.updated_at DESC LIMIT %s
                        """,
                        (exact["lcn"], exact["hsc"], limit),
                    )
                rows = _row_dicts(cur, cur.fetchall())
                for row in rows:
                    row["match_type"] = match_type
                return rows

            # A cause/symptom question must not select an HMC row solely because
            # a generic user word occurs in its repair instruction.  For
            # example, “外设” in “检查外设响应” says nothing about whether the
            # row's fault phenomenon matches “数据没有传到外设”.  Maintenance
            # questions can opt into that column; all other keyword lookups use
            # code fields plus the recorded fault description only.
            searchable_columns = [
                "hmc_code", "lcn", "hamc", "hsc", "fault_desc",
            ]
            if include_repair_method:
                searchable_columns.append("repair_method")
            search_terms = _sql_candidate_terms(terms)
            clauses: List[str] = []
            params: List[Any] = []
            for term in search_terms:
                pattern = f"%{term}%"
                clauses.append(
                    "(" + " OR ".join(
                        f"CAST(h.{column} AS TEXT) ILIKE %s"
                        for column in searchable_columns
                    ) + ")"
                )
                params.extend([pattern] * len(searchable_columns))
            if not clauses:
                return []
            candidate_limit = min(max(int(limit) * 8, 64), 512)
            params.append(candidate_limit)
            cur.execute(
                "SELECT h.* FROM fault_hmc h "
                "JOIN icp_documents d ON d.doc_id=h.doc_id "
                "WHERE d.status IN ('indexed','kg_pending','kg_done') AND ("
                f"{' OR '.join(clauses)}) ORDER BY h.updated_at DESC LIMIT %s",
                tuple(params),
            )
            rows = _row_dicts(cur, cur.fetchall())
            rows = _rank_table_rows(
                rows,
                searchable_columns,
                [*search_terms, *terms],
                limit,
            )
            for row in rows:
                row["match_type"] = "keyword"
            return rows
    except Exception as exc:
        logger.warning("HMC query failed: %s", exc)
        if errors is not None:
            errors.append(str(exc))
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _query_table(
    table: str,
    columns: Sequence[str],
    terms: Sequence[str],
    limit: int,
    errors: Optional[List[str]] = None,
    *,
    priority_columns: Sequence[str] = (),
    priority_terms: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    if not terms:
        return []
    # SQL candidate recall must use the same question-noise normalization as
    # the deterministic ranker. Otherwise a natural question such as
    # “滤波电容有点老化了，会有什么影响吗” never recalls the row containing
    # the exact phrase “滤波电容老化”, leaving only the weaker search copy.
    search_terms = _sql_candidate_terms(terms)
    if not search_terms:
        return []
    clauses: List[str] = []
    specificity_terms: List[str] = []
    params: List[Any] = []
    for term in search_terms:
        pattern = f"%{term}%"
        clause = "(" + " OR ".join(f"CAST(t.{col} AS TEXT) ILIKE %s" for col in columns) + ")"
        clauses.append(clause)
        specificity_terms.append(
            f"CASE WHEN {clause} THEN {int(len(term) ** 2)} ELSE 0 END"
        )
        params.extend([pattern] * len(columns))
    # Fetch a bounded candidate pool before deterministic relevance sorting.
    # Ordering only by updated_at lets a generic token such as “老化” outrank
    # the row that contains the complete phrase “滤波电容老化”; worse, when
    # question-noise terms sweep most of the table, the authoritative row is
    # truncated out of the pool before the Python ranker ever sees it.  The
    # SQL therefore prefilters with the same len² specificity criterion as
    # _rank_table_rows so precise phrase hits survive the candidate cutoff;
    # updated_at remains only a tie-break.  Weights are derived from term
    # lengths (not user text) and stay server-side constants.
    order_params: List[Any] = []
    for term in search_terms:
        order_params.extend([f"%{term}%"] * len(columns))
    candidate_limit = min(max(int(limit) * 8, 64), 512)
    params.extend(order_params)
    params.append(candidate_limit)
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT t.* FROM {table} t "
                "JOIN icp_documents d ON d.doc_id=t.doc_id "
                "WHERE d.status IN ('indexed','kg_pending','kg_done') AND ("
                f"{' OR '.join(clauses)}) "
                f"ORDER BY ({' + '.join(specificity_terms)}) DESC, t.updated_at DESC LIMIT %s",
                tuple(params),
            )
            rows = _row_dicts(cur, cur.fetchall())
            return _rank_table_rows(
                rows,
                columns,
                [*search_terms, *terms],
                limit,
                priority_columns=priority_columns,
                priority_terms=priority_terms,
            )
    except Exception as exc:
        logger.warning("ICP table query failed (%s): %s", table, exc)
        if errors is not None:
            errors.append(str(exc))
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _normalize_match_term(value: Any) -> str:
    return normalize_search_term(value)


def _rank_table_rows(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    terms: Sequence[str],
    limit: int,
    *,
    priority_columns: Sequence[str] = (),
    priority_terms: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Rank structured rows by specific phrase coverage, preserving stable ties."""
    normalized_terms = list(
        dict.fromkeys(
            value
            for value in (_normalize_match_term(term) for term in terms[:16])
            if len(value) >= 2
        )
    )

    normalized_priority_terms = list(
        dict.fromkeys(
            value
            for value in (
                _normalize_match_term(term) for term in priority_terms[:16]
            )
            if len(value) >= 2
        )
    )

    def score(row: Dict[str, Any]) -> tuple[int, int, int, int]:
        cells = [_normalize_match_term(row.get(column)) for column in columns]
        matched = [
            term
            for term in normalized_terms
            if any(term in cell for cell in cells if cell)
        ]
        identity_cells = [
            _normalize_match_term(row.get(column)) for column in priority_columns
        ]
        identity_matches = [
            term
            for term in normalized_priority_terms
            if any(term in cell for cell in identity_cells if cell)
        ]
        # Squared length strongly rewards the most specific complete phrase;
        # number and longest match provide deterministic secondary ordering.
        # An entity term found in a stored business/object field outranks the
        # same generic word appearing only in an impact or narrative column.
        return (
            sum(len(term) ** 2 for term in identity_matches),
            sum(len(term) ** 2 for term in matched),
            max((len(term) for term in matched), default=0),
            len(matched),
        )

    ranked = [dict(row) for row in rows]
    scored = [(score(row), row) for row in ranked]
    # 相对特异性下限（仅对象锚定查询）：问题给出明确业务编码时，回答围绕
    # 该对象；只靠两三个字泛词（如“状态/检查”命中每一行的“状态字”列）凑分
    # 的行不得进入回答窗口，身份字段命中永远保留。阈值按最强命中的四分之一
    # 自适应，最强命中本身很弱时不设限。没有业务编码的症状描述型问题保留
    # 多候选语义，不设下限——那是“待排查方向”，不是对象锚定检索。
    anchored_query = bool(
        _SQL_BUSINESS_CODE_RE.search(
            " ".join(str(term or "") for term in terms)
        )
    )
    best_match = max((value[1] for value, _row in scored), default=0)
    if anchored_query and best_match >= 16:
        kept = [
            (value, row)
            for value, row in scored
            if value[0] > 0 or value[1] * 4 >= best_match
        ]
        if kept:
            scored = kept
    scored.sort(key=lambda pair: pair[0], reverse=True)
    for row_score, row in scored:
        row["retrieval_identity_score"] = row_score[0]
        row["retrieval_match_score"] = row_score[1]
    return [row for _, row in scored[: max(int(limit), 0)]]


def _query_fmeca(
    terms: Sequence[str],
    limit: int,
    errors: Optional[List[str]] = None,
    *,
    entity_terms: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    return _query_table(
        "fmea",
        (
            "code", "product_flag", "func_management", "fault_mode", "fault_cause",
            "mission_phase", "local_impact", "final_impact", "severity", "detect_method",
            "design_improve", "compensate_action",
        ),
        terms,
        limit,
        errors,
        priority_columns=("code", "product_flag", "func_management", "fault_mode"),
        priority_terms=entity_terms,
    )


def _query_fmeca_by_business_ids(
    business_ids: Sequence[Any],
    limit: int,
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Read FMECA rows by literal Function Locator business IDs.

    The SQL lookup remains bounded and every returned row is checked again
    against codes parsed from its authoritative ``product_flag``.  Natural
    language labels and retrieval similarity cannot authorize this handoff.
    """

    terms = _prepend_query_terms([], business_ids, limit=16)
    normalized_ids = {
        normalize_search_term(value) for value in terms if normalize_search_term(value)
    }
    if not normalized_ids:
        return []
    candidates = _query_fmeca(
        terms,
        max(8, min(max(int(limit), 0) * 2, 32)),
        errors,
        entity_terms=terms,
    )
    matched: List[Dict[str, Any]] = []
    for row in candidates:
        row_ids = {
            normalize_search_term(value)
            for value in extract_business_codes(row.get("product_flag") or "")
            if normalize_search_term(value)
        }
        if not (row_ids & normalized_ids):
            continue
        item = dict(row)
        item["match_type"] = "exact_structured_identity"
        item["structured_identity_terms"] = [
            value
            for value in terms
            if normalize_search_term(value) in row_ids
        ]
        matched.append(item)
        if len(matched) >= max(int(limit), 0):
            break
    return matched


def _query_mfl(
    terms: Sequence[str], limit: int, errors: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    return _query_table(
        "mfl",
        (
            "mfl_code", "module_fault", "sur_fault", "subcircuit_fault",
            "api_reportable", "api_reportable_raw", "api_name", "api_method",
            "api_usage", "platform_test", "platform_test_raw",
            "platform_test_status", "maintenance_action",
        ),
        terms,
        limit,
        errors,
    )


def _query_diagrams(
    doc_type: str,
    terms: Sequence[str],
    limit: int,
    errors: Optional[List[str]] = None,
    *,
    entity_terms: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    if doc_type not in ("fta", "rbd") or not terms:
        return []
    try:
        revisions = [
            revision
            for revision in diagram_repository.list_active_revisions(
                diagram_type=doc_type
            )
            if _is_context_graph_row(revision)
        ]
        matches: List[Tuple[int, Dict[str, Any]]] = []
        for revision in revisions:
            # 仓储会返回正式自动验证图和仅供浏览的 provisional 图。这里仍
            # 防御性排除 pending/quarantined/failed；provisional 可进入带边界
            # 的回答上下文，但后续计算节点会再次执行更严格的正式门禁。
            if not _is_context_graph_row(revision):
                continue
            haystack = (
                f"{revision.get('title', '')} {revision.get('diagram_key', '')} "
                f"{json.dumps(revision.get('canonical_graph') or {}, ensure_ascii=False, default=str)}"
            )
            score, matched_terms = diagram_match_score(haystack, terms)
            if score <= 0:
                continue
            graph = revision.get("canonical_graph") or {}
            identity_values: List[str] = []
            if isinstance(graph, Mapping):
                # Versioned diagram graphs use ``vertices``; a few legacy and
                # test payloads still use ``nodes``.  Read both shapes so the
                # identity ranker works against the actual persisted RBD data.
                graph_nodes: List[Mapping[str, Any]] = []
                for collection_name in ("vertices", "nodes"):
                    collection = graph.get(collection_name) or []
                    if isinstance(collection, (list, tuple)):
                        graph_nodes.extend(
                            item for item in collection if isinstance(item, Mapping)
                        )
                for node in graph_nodes:
                    properties = (
                        node.get("properties")
                        if isinstance(node.get("properties"), Mapping)
                        else {}
                    )
                    for value in (
                        node.get("id"),
                        node.get("business_id"),
                        properties.get("module_id"),
                        properties.get("function_id"),
                        properties.get("name"),
                        properties.get("abbreviation"),
                    ):
                        if str(value or "").strip():
                            identity_values.append(str(value).strip())
            identity_score, identity_matches = diagram_match_score(
                " ".join(identity_values), entity_terms
            )
            # 标题与顶事件是图的“身份文本”：与它们重合的词是主题级匹配，
            # 只出现在深层节点说明里的词是偶然匹配。排序必须身份优先，否则
            # 一棵不相关故障树里恰好包含用户措辞会压过真正命中的顶事件。
            top_event_name = ""
            root_id_text = str(graph.get("root_fault_id") or "").strip()
            for node in graph_nodes:
                node_id = str(node.get("id") or "").strip()
                node_label = str(node.get("label") or "").casefold().replace("_", "")
                if (root_id_text and node_id == root_id_text) or node_label == "topevent":
                    node_properties = (
                        node.get("properties")
                        if isinstance(node.get("properties"), Mapping)
                        else {}
                    )
                    top_event_name = str(
                        node_properties.get("name") or node.get("name") or ""
                    ).strip()[:200]
                    break
            heading_score, heading_terms = diagram_match_score(
                f"{revision.get('title', '')} {revision.get('diagram_key', '')} {top_event_name}",
                terms,
            )
            item = dict(revision)
            item["graph"] = item.get("canonical_graph") or {}
            item["image_title"] = item.get("title") or item.get("diagram_key") or ""
            item["retrieval_match_score"] = score
            item["retrieval_identity_score"] = identity_score
            item["retrieval_heading_score"] = heading_score
            item["retrieval_match_terms"] = list(
                dict.fromkeys([*matched_terms, *heading_terms])
            )
            item["retrieval_identity_terms"] = identity_matches
            matches.append(
                (identity_score * 1_000_000 + heading_score * 1000 + score, item)
            )
        matches.sort(key=lambda pair: pair[0], reverse=True)
        # 同义落空兜底：问题并列枚举多个对象（顿号枚举 + 分别/各自语义、
        # ≥3 个短锚点）而权威图结构恰好直接覆盖其中一个（例如用户说
        # “核心运算”，库内模型名为“核心处理功能”）时，按类别补齐当前
        # 文档族的其余活动图，让回答能逐项覆盖并对确实缺失的对象显式
        # 声明。完全零覆盖不得触发——那是真实的资料未命中，补齐全族反而
        # 制造跨域噪声。补齐只来自同一 doc_type 的权威修订列表，不引入
        # 相似度、别名或跨文档猜测；数量有界并排在直接命中之后。
        anchors = [
            value
            for value in (_normalize_match_term(term) for term in terms)
            if 2 <= len(value) <= _SQL_SHORT_ANCHOR_MAX_LENGTH
        ]
        covered: set[str] = set()
        for _, item in matches:
            covered.update(
                _normalize_match_term(value)
                for value in (item.get("retrieval_match_terms") or [])
            )
        enumeration_shape = bool(
            re.search(r"、[^。]{2,40}(?:分别|各自|都)|(?:分别|各自)[^。]{0,12}(?:依赖|包括|是哪些)", str(terms[0] or ""))
        )
        category_context = bool(
            enumeration_shape
            and len(anchors) >= 3
            and sum(1 for anchor in anchors if anchor in covered) == 1
        )
        if category_context:
            matched_ids = {
                str(item.get("revision_id") or "") for _, item in matches
            }
            for revision in revisions:
                if len(matches) >= 12:
                    break
                if str(revision.get("revision_id") or "") in matched_ids:
                    continue
                extra = dict(revision)
                extra["graph"] = extra.get("canonical_graph") or {}
                extra["image_title"] = (
                    extra.get("title") or extra.get("diagram_key") or ""
                )
                extra["retrieval_match_score"] = 0
                extra["retrieval_identity_score"] = 0
                extra["retrieval_match_terms"] = []
                extra["retrieval_identity_terms"] = []
                extra["category_context"] = True
                matches.append((0, extra))
        if category_context:
            return [item for _, item in matches]
        return [item for _, item in matches[: max(int(limit), 0)]]
    except Exception as exc:
        logger.warning("Diagram query failed (%s): %s", doc_type, exc)
        if errors is not None:
            errors.append(str(exc))
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _graph_basic_event_summaries(graph: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Extract a bounded, human-readable basic-event list from a graph.

    FTA evidence keeps the canonical graph in ``content`` for provenance, but
    answer prompts must not receive raw JSON.  This compact projection lets a
    structure-only answer list every basic event while the public citation
    still exposes only a readable summary.
    """

    if not isinstance(graph, Mapping):
        return []
    vertices: List[Mapping[str, Any]] = []
    for key in ("vertices", "nodes"):
        values = graph.get(key) or []
        if isinstance(values, (list, tuple)):
            vertices.extend(item for item in values if isinstance(item, Mapping))
    result: List[Dict[str, str]] = []
    seen: set[str] = set()
    for vertex in vertices:
        properties = vertex.get("properties") if isinstance(vertex.get("properties"), Mapping) else {}
        label = str(vertex.get("label") or properties.get("type") or "").strip().casefold()
        if label not in {"basicevent", "basic_event", "basic event", "基本事件"}:
            continue
        name = str(
            properties.get("name")
            or properties.get("event_name")
            or vertex.get("name")
            or vertex.get("title")
            or vertex.get("id")
            or ""
        ).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        description = str(
            properties.get("description")
            or properties.get("failure_cause")
            or properties.get("cause")
            or properties.get("reason")
            or ""
        ).strip()
        item = {"name": name}
        if description:
            item["description"] = description
        result.append(item)
    return result


def _diagram_explicit_block_business_ids(row: Mapping[str, Any]) -> set[str]:
    """Read only durable block IDs carried by a persisted diagram graph.

    Diagram-local vertex keys and human-readable labels are intentionally not
    identity evidence.  This helper is used to narrow an explicitly requested
    RBD lookup after Function Locator has supplied an exact business ID; it
    never promotes a lexical hit into a cross-document link.
    """

    graph = row.get("graph") or row.get("canonical_graph") or {}
    if not isinstance(graph, Mapping):
        return set()
    values: List[Any] = []
    for collection_name in ("vertices", "nodes"):
        collection = graph.get(collection_name) or []
        if not isinstance(collection, (list, tuple)):
            continue
        for vertex in collection:
            if not isinstance(vertex, Mapping):
                continue
            properties = (
                vertex.get("properties")
                if isinstance(vertex.get("properties"), Mapping)
                else {}
            )
            values.extend(
                (
                    vertex.get("business_id"),
                    vertex.get("module_id"),
                    vertex.get("component_id"),
                    vertex.get("function_id"),
                    properties.get("business_id"),
                    properties.get("module_id"),
                    properties.get("component_id"),
                    properties.get("function_id"),
                )
            )
    return {
        normalized
        for value in extract_business_codes(values, limit=96)
        if (normalized := normalize_business_id(value))
    }


def _filter_diagrams_by_explicit_object_ids(
    rows: Sequence[Mapping[str, Any]], business_ids: Sequence[Any]
) -> List[Dict[str, Any]]:
    """Keep diagrams that explicitly contain one bound Function Locator ID.

    An empty result is meaningful: the named diagram family was consulted but
    has no formally attributable graph for this object.  Callers can then
    report that fact instead of silently substituting another RBD model.
    """

    expected = {
        normalize_business_id(value)
        for value in business_ids
        if normalize_business_id(value)
    }
    if not expected:
        return [dict(row) for row in rows]
    return [
        dict(row)
        for row in rows
        if _diagram_explicit_block_business_ids(row) & expected
    ]


def _hybrid_search(
    state: Dict[str, Any],
    doc_types: Iterable[str],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    query = build_hybrid_query(state)
    if not query:
        return [], {"reranked": False, "fallback": "查询为空，未执行混合召回"}
    embedding_model_id = state.get("embedding_model_id") or None
    try:
        outcome = knowledge_retriever.search_hybrid_with_info(
            kb_id="",
            kb_ids=list(doc_types),
            query=query,
            top_k=top_k,
            keyword_weight=0.45,
            vector_weight=0.55,
            embedding_model_id=embedding_model_id,
            agent_id=state.get("agent_id", ""),
            rerank_model_id=state.get("rerank_model_id", ""),
        )
    except Exception as exc:
        logger.warning("ICP hybrid search failed: %s", exc)
        return [], {
            "reranked": False,
            "fallback": f"统一检索失败: {exc}",
            "error": str(exc),
        }
    return outcome.results, outcome.info()


def _tool_call(name: str, count: int, started: float, preview: List[Dict[str, Any]], **params) -> Dict[str, Any]:
    return {
        "tool": name,
        "count": count,
        "duration": round(time.time() - started, 3),
        "params": params,
        "results_preview": preview[:8],
        # Filled from the matching ToolResult before the node returns.  Keeping
        # the field present on empty/failed calls gives SSE consumers one stable
        # shape without exposing retrieval rows which were not normalized as
        # Evidence.
        "evidence_ids": [],
        "evidence_count": 0,
    }


def _preview_rows(rows: Sequence[Dict[str, Any]], title_keys: Sequence[str], body_keys: Sequence[str]) -> List[Dict[str, Any]]:
    preview = []
    for row in rows[:8]:
        title = next((str(row.get(key) or "").strip() for key in title_keys if row.get(key)), "")
        body = "；".join(str(row.get(key) or "").strip() for key in body_keys if row.get(key))
        preview.append({"title": title[:100], "snippet": body[:260]})
    return preview


def _attach_tool_call_evidence(
    tool_calls: Sequence[Dict[str, Any]],
    tool_results: Sequence[Dict[str, Any]],
    evidence_registry: Sequence[Dict[str, Any]],
) -> None:
    """Bind auditable, bounded Evidence references to tool-call SSE payloads.

    ``tool_calls`` are persisted and streamed to the UI, while ``ToolResult`` is
    the authoritative record of which normalized evidence a tool produced.  The
    two structures used to be independent, making every retrieval candidate look
    like an answer citation.  This binding keeps the existing event fields but
    adds stable evidence IDs and compact previews.  It intentionally exposes no
    planner reasoning and never copies a full evidence body into an event.
    """

    result_ids_by_tool: Dict[str, List[str]] = defaultdict(list)
    for result in tool_results or []:
        tool = str(result.get("tool") or "").strip()
        if not tool:
            continue
        for value in result.get("evidence_ids") or []:
            evidence_id = str(value or "").strip()
            if evidence_id and evidence_id not in result_ids_by_tool[tool]:
                result_ids_by_tool[tool].append(evidence_id)

    evidence_by_id = {
        str(item.get("evidence_id") or "").strip(): item
        for item in evidence_registry or []
        if str(item.get("evidence_id") or "").strip()
    }
    # ``fault_reasoning`` is the user-facing compatibility label for the
    # deterministic ``evidence_synthesis`` ToolResult.
    result_aliases = {"fault_reasoning": "evidence_synthesis"}

    for call in tool_calls or []:
        tool = str(call.get("tool") or "").strip()
        result_tool = result_aliases.get(tool, tool)
        all_ids = result_ids_by_tool.get(result_tool, [])
        allowed_ids = set(all_ids)

        # Candidate-fusion previews can already identify their supporting
        # evidence.  Prefer those links, then fill the bounded window in the
        # ToolResult order.  Every selected ID therefore still comes from the
        # matching ToolResult.
        preferred_ids: List[str] = []
        for raw_preview in call.get("results_preview") or []:
            values = [raw_preview.get("evidence_id")]
            values.extend(raw_preview.get("evidence_ids") or [])
            for value in values:
                evidence_id = str(value or "").strip()
                if (
                    evidence_id
                    and evidence_id in allowed_ids
                    and evidence_id not in preferred_ids
                ):
                    preferred_ids.append(evidence_id)
        selected_ids = [
            *preferred_ids,
            *(evidence_id for evidence_id in all_ids if evidence_id not in preferred_ids),
        ][:_TOOL_CALL_EVIDENCE_LIMIT]
        call["evidence_ids"] = selected_ids
        call["evidence_count"] = len(all_ids)

        linked_candidate_previews: List[Dict[str, Any]] = []
        for raw_preview in (call.get("results_preview") or [])[:8]:
            preview_ids = []
            for value in [raw_preview.get("evidence_id"), *(raw_preview.get("evidence_ids") or [])]:
                evidence_id = str(value or "").strip()
                if evidence_id in selected_ids and evidence_id not in preview_ids:
                    preview_ids.append(evidence_id)
            if not preview_ids:
                continue
            primary = evidence_by_id.get(preview_ids[0], {})
            display_summary = " ".join(
                str(primary.get("display_summary") or "").split()
            )[:_TOOL_CALL_PREVIEW_SNIPPET_LIMIT]
            snippet = display_summary or " ".join(
                str(raw_preview.get("snippet") or "").split()
            )[:_TOOL_CALL_PREVIEW_SNIPPET_LIMIT]
            linked_candidate_previews.append(
                {
                    "title": str(raw_preview.get("title") or "").strip()[:100],
                    "snippet": snippet,
                    "display_summary": display_summary,
                    "evidence_id": preview_ids[0],
                    "evidence_ids": preview_ids,
                    "source_ref": str(primary.get("source_ref") or "").strip()[
                        :_TOOL_CALL_PREVIEW_SOURCE_LIMIT
                    ],
                }
            )

        if linked_candidate_previews:
            call["results_preview"] = linked_candidate_previews
            continue

        evidence_previews: List[Dict[str, Any]] = []
        for evidence_id in selected_ids:
            item = evidence_by_id.get(evidence_id)
            if not item:
                continue
            display_summary = " ".join(
                str(item.get("display_summary") or "").split()
            )[:_TOOL_CALL_PREVIEW_SNIPPET_LIMIT]
            evidence_previews.append(
                {
                    "title": str(item.get("title") or item.get("doc_name") or "证据").strip()[:100],
                    "snippet": display_summary or " ".join(str(item.get("content") or "").split())[
                        :_TOOL_CALL_PREVIEW_SNIPPET_LIMIT
                    ],
                    "display_summary": display_summary,
                    "evidence_id": evidence_id,
                    "source_ref": str(item.get("source_ref") or "").strip()[
                        :_TOOL_CALL_PREVIEW_SOURCE_LIMIT
                    ],
                }
            )
        if evidence_previews:
            call["results_preview"] = evidence_previews


def _as_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _enrich_function_sources(
    rows: Sequence[Dict[str, Any]],
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Attach the authoritative document name/version/configuration to PG rows."""
    enriched = [dict(row) for row in rows]
    doc_ids = list(dict.fromkeys(str(row.get("doc_id") or "") for row in enriched if row.get("doc_id")))
    if not doc_ids:
        return enriched
    placeholders = ",".join(["%s"] * len(doc_ids))
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT doc_id, file_name, metadata FROM icp_documents WHERE doc_id IN ({placeholders})",
                tuple(doc_ids),
            )
            source_by_doc = {
                str(doc_id): {"doc_name": file_name or "", "document_metadata": _as_metadata(metadata)}
                for doc_id, file_name, metadata in cur.fetchall()
            }
    except Exception as exc:
        logger.warning("Function source metadata query failed: %s", exc)
        if errors is not None:
            errors.append(f"文档来源元数据查询失败: {exc}")
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return enriched

    for row in enriched:
        source = source_by_doc.get(str(row.get("doc_id") or ""), {})
        row["doc_name"] = source.get("doc_name", "")
        row["document_metadata"] = source.get("document_metadata", {})
    return enriched


def _architecture_domain_results(
    function_rows: Sequence[Dict[str, Any]],
    function_hybrid: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize PG truth and Function Locator projections without fault scoring."""
    results: List[Dict[str, Any]] = []
    for row in function_rows[:12]:
        content_type = str(row.get("content_type") or "function_record")
        hierarchy: Dict[str, Any] = {}
        if content_type == "function_component":
            record_id = row.get("component_id", "")
            title = row.get("name") or record_id
            content = _readable_direct_fields(
                (
                    ("组件ID", record_id),
                    ("类型", row.get("component_type")),
                    ("名称", row.get("name")),
                    ("简称", row.get("abbreviation")),
                    ("父组件ID", row.get("parent_component_id")),
                    ("说明", row.get("description")),
                )
            )
        elif content_type == "function_interface":
            record_id = row.get("interface_id", "")
            title = row.get("signal_name") or record_id
            content = _readable_direct_fields(
                (
                    ("接口ID", record_id),
                    ("源组件", row.get("source_component_id")),
                    ("目标组件", row.get("target_component_id")),
                    ("关系", row.get("relation_type")),
                    ("信号", row.get("signal_name")),
                    ("方向", row.get("direction")),
                    ("协议", row.get("protocol")),
                    ("说明", row.get("description")),
                )
            )
        else:
            hierarchy = {
                "system_name": row.get("system_name", ""),
                "subsystem_name": row.get("subsystem_name", ""),
                "module_name": row.get("module_name", ""),
                "function_name": row.get("function_name", ""),
            }
            aliases = row.get("aliases") or []
            if isinstance(aliases, str):
                aliases_text = aliases
            else:
                aliases_text = "、".join(str(value) for value in aliases if value)
            record_id = row.get("function_id", "")
            title = row.get("function_name") or record_id
            content = (
                f"系统：{hierarchy['system_name']}\n子系统：{hierarchy['subsystem_name']}\n"
                f"模块：{hierarchy['module_name']}\n功能：{hierarchy['function_name']}\n"
                f"说明：{row.get('description', '')}\n别名：{aliases_text}\n"
                f"输入：{'；'.join(row.get('inputs') or [])}\n"
                f"输出：{'；'.join(row.get('outputs') or [])}\n"
                f"典型失效表现：{row.get('failure_symptoms', '')}"
            )
        results.append(
            {
                "source_type": "function_locator_pg",
                "authority": "postgres_authoritative",
                "doc_id": row.get("doc_id", ""),
                "doc_name": row.get("doc_name", ""),
                "chunk_id": str(record_id or ""),
                "function_id": row.get("function_id", ""),
                "component_id": row.get("component_id", ""),
                "interface_id": row.get("interface_id", ""),
                "content_type": content_type,
                "title": title,
                "content": content,
                "source_ref": row.get("source_ref", ""),
                "document_metadata": _as_metadata(row.get("document_metadata")),
                "hierarchy": hierarchy,
            }
        )

    remaining = max(20 - len(results), 0)
    for row in function_hybrid[:remaining]:
        item = dict(row)
        item["source_type"] = "function_locator_search"
        item["authority"] = "opensearch_projection"
        item["document_metadata"] = _as_metadata(item.get("document_metadata"))
        item.setdefault("doc_id", "")
        item.setdefault("doc_name", "")
        item.setdefault("chunk_id", "")
        item.setdefault("title", "")
        item.setdefault("content", "")
        results.append(item)
    return results


def _metadata_summary(metadata: Dict[str, Any]) -> str:
    labels = (
        ("document_number", "文档编号"),
        ("applicable_system", "适用系统"),
        ("applicable_configuration", "适用构型"),
        ("document_version", "文档版本"),
        ("data_source", "数据来源"),
        ("data_status", "数据状态"),
        ("compiled_date", "编制日期"),
    )
    return "；".join(f"{label}={metadata.get(key)}" for key, label in labels if metadata.get(key))


def _scope(metadata: Any) -> Dict[str, str]:
    value = _as_metadata(metadata)
    aliases = {
        "system": ("applicable_system", "system", "system_name"),
        "configuration": ("applicable_configuration", "configuration", "configuration_name"),
        "document_number": ("document_number", "doc_number"),
        "document_version": ("document_version", "version"),
    }
    result: Dict[str, str] = {}
    for target, keys in aliases.items():
        selected = next((str(value.get(key) or "").strip() for key in keys if value.get(key)), "")
        if selected:
            result[target] = selected
    return result


def _tool_status(rows: Sequence[Any], errors: Sequence[str], *, partial: bool = False) -> str:
    if errors and not rows:
        return "failed"
    if errors or partial:
        return "partial"
    return "ok" if rows else "empty"


def _retrieval_diagnostics(info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """将检索管线的失败与可降级信息映射到 ToolResult。"""
    errors: List[str] = []
    fallbacks: List[str] = []
    error = str(info.get("error") or "").strip()
    if error:
        errors.append(error)
    for value in [info.get("fallback"), *(info.get("recall_fallbacks") or [])]:
        text = str(value or "").strip()
        if text and text not in fallbacks:
            fallbacks.append(text)
    rerank_error = str(info.get("rerank_error") or "").strip()
    if rerank_error and rerank_error not in fallbacks:
        fallbacks.append(rerank_error)
    return errors, fallbacks


def _record_id(row: Dict[str, Any], *keys: str) -> str:
    return next((str(row.get(key) or "").strip() for key in keys if row.get(key)), "")


def _search_doc_type(row: Dict[str, Any]) -> str:
    for key in ("doc_type_kwd", "kb_id", "section", "doc_type"):
        raw = row.get(key)
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        for item in values:
            value = str(item or "").strip()
            if value in ICP_DOC_TYPES:
                return value
    return ""


def _scalar_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return next((str(item or "").strip() for item in value if str(item or "").strip()), "")
    return str(value or "").strip()


def _is_graph_locator_hit(row: Dict[str, Any]) -> bool:
    return (
        _scalar_text(row.get("projection_role_kwd")) == "diagram_locator"
        or _scalar_text(row.get("content_type")) in _GRAPH_LOCATOR_CONTENT_TYPES
    )


def _search_locator_handoffs(
    rows: Sequence[Dict[str, Any]], *, limit: int = 24, limit_per_type: int = 4
) -> List[Dict[str, Any]]:
    """Keep stable PostgreSQL identities across adaptive query batches.

    The handoff deliberately excludes indexed payloads and lifecycle claims.
    Diagram locators are resolved through the versioned diagram repository;
    table records are resolved only through the stored ``doc_id + pg_id`` pair.
    Semantic text, titles and aliases never become an entity identity here.
    """

    result: List[Dict[str, Any]] = []
    seen = set()
    per_type: Dict[str, int] = defaultdict(int)
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        doc_type = _search_doc_type(dict(raw))
        if doc_type not in ICP_DOC_TYPES:
            continue
        if per_type[doc_type] >= max(int(limit_per_type), 0):
            continue
        doc_id = _scalar_text(raw.get("doc_id"))
        graph_locator = _is_graph_locator_hit(dict(raw)) and doc_type in {"fta", "rbd"}
        diagram_id = _scalar_text(raw.get("diagram_id_kwd") or raw.get("diagram_id"))
        business_id = _scalar_text(raw.get("business_id_kwd") or raw.get("diagram_key"))
        pg_id = _scalar_text(raw.get("pg_id_kwd") or raw.get("pg_id"))
        if graph_locator:
            stable_identity = diagram_id or (f"{doc_id}:{business_id}" if doc_id and business_id else "")
        else:
            stable_identity = f"{doc_id}:{pg_id}" if doc_id and pg_id else ""
        if not stable_identity:
            continue
        identity = (doc_type, stable_identity)
        if identity in seen:
            continue
        seen.add(identity)
        item = {
            "doc_type_kwd": doc_type,
            "doc_id": doc_id,
            "business_id_kwd": business_id,
            "pg_id_kwd": pg_id,
            "projection_role_kwd": "diagram_locator" if graph_locator else "record_locator",
            "content_type": _scalar_text(raw.get("content_type")),
            "chunk_id": _scalar_text(raw.get("chunk_id") or raw.get("id")),
            "title": _scalar_text(raw.get("title")),
            "score": raw.get("score"),
            "rerank_score": raw.get("rerank_score"),
        }
        if graph_locator:
            item["diagram_id_kwd"] = diagram_id
        result.append(item)
        per_type[doc_type] += 1
        if len(result) >= max(int(limit), 0):
            break
    return result


def _hydrate_diagram_search_hits(
    rows: Sequence[Dict[str, Any]],
    *,
    allowed_tools: Iterable[str],
    limit_per_type: int,
    errors: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Resolve semantic graph locators back to active PostgreSQL revisions.

    OpenSearch decides relevance only.  Identity, lifecycle and validation
    status are re-read from PostgreSQL.  A hit without a stable diagram ID (or
    the legacy exact ``doc_id + diagram_key`` pair) is not fuzzily attached to
    a graph.
    """

    allowed = set(allowed_tools)
    result: Dict[str, List[Dict[str, Any]]] = {"fta_query": [], "rbd_query": []}
    seen: Dict[str, set[str]] = {"fta_query": set(), "rbd_query": set()}
    identity_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for hit in rows:
        doc_type = _search_doc_type(hit)
        tool = "fta_query" if doc_type == "fta" else "rbd_query" if doc_type == "rbd" else ""
        if not tool or tool not in allowed or not _is_graph_locator_hit(hit):
            continue
        if len(result[tool]) >= max(int(limit_per_type), 0):
            continue

        diagram_id = _scalar_text(hit.get("diagram_id_kwd") or hit.get("diagram_id"))
        doc_id = _scalar_text(hit.get("doc_id"))
        diagram_key = _scalar_text(hit.get("business_id_kwd") or hit.get("diagram_key"))
        revision: Optional[Dict[str, Any]] = None
        try:
            if diagram_id:
                revision = diagram_repository.get_active_revision(diagram_id)
            elif doc_id and diagram_key:
                cache_key = (doc_id, doc_type)
                if cache_key not in identity_cache:
                    identity_cache[cache_key] = diagram_repository.list_active_revisions(
                        doc_id=doc_id,
                        diagram_type=doc_type,
                    )
                revision = next(
                    (
                        item
                        for item in identity_cache[cache_key]
                        if str(item.get("diagram_key") or "") == diagram_key
                    ),
                    None,
                )
        except Exception as exc:
            logger.warning("Semantic diagram hydration failed (%s): %s", doc_type, exc)
            if errors is not None:
                errors.setdefault(tool, []).append(str(exc))
            continue
        if not revision or not _is_context_graph_row(revision):
            continue
        if doc_id and str(revision.get("doc_id") or "") != doc_id:
            continue
        if str(revision.get("diagram_type") or "") != doc_type:
            continue

        revision_id = str(revision.get("revision_id") or "")
        if not revision_id or revision_id in seen[tool]:
            continue
        seen[tool].add(revision_id)
        item = dict(revision)
        item["graph"] = item.get("canonical_graph") or {}
        item["image_title"] = item.get("title") or item.get("diagram_key") or ""
        item["match_type"] = "semantic_locator_hydration"
        item["retrieval_match_score"] = hit.get("rerank_score", hit.get("score"))
        item["retrieval_source_chunk_id"] = hit.get("chunk_id") or hit.get("id") or ""
        result[tool].append(item)
    return result


def _hydrate_record_search_hits(
    rows: Sequence[Dict[str, Any]],
    *,
    limit_per_type: int = 1,
    limits_by_type: Optional[Mapping[str, int]] = None,
    errors: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Resolve top semantic record locators back to PostgreSQL by storage ID.

    This turns a useful OpenSearch recall into a citable fact without trusting
    the indexed copy.  Per-type overrides allow a small comparison set of
    distinct modes, while a hard cap prevents an unbounded family scan.  It
    does *not* merge entities or claim that similarly named records are
    identical: each returned row is the exact PostgreSQL row referenced by the
    search projection.
    """

    tool_by_type = {
        "function_locator": "function_locator",
        "hmc": "hmc_query",
        "fmeca": "fmeca_query",
        "mfl": "mfl_query",
        "fault_history": "fault_history_query",
    }

    def hydration_limit(doc_type: str) -> int:
        configured = (limits_by_type or {}).get(doc_type, limit_per_type)
        try:
            value = int(configured)
        except (TypeError, ValueError):
            value = int(limit_per_type)
        return min(max(value, 0), _MAX_RECORD_HYDRATION_PER_TYPE)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        if not isinstance(raw, Mapping) or _is_graph_locator_hit(dict(raw)):
            continue
        doc_type = _search_doc_type(dict(raw))
        if doc_type not in tool_by_type:
            continue
        if doc_type == "function_locator" and _scalar_text(raw.get("content_type")) not in {
            "function_record",
            "function_component",
            "function_interface",
        }:
            # Free-form Function Locator paragraphs remain search clues. They
            # need an exact source-artifact check before they can be cited and
            # therefore never enter the PostgreSQL record hydration path.
            continue
        if len(grouped[doc_type]) >= hydration_limit(doc_type):
            continue
        if not _scalar_text(raw.get("doc_id")) or not _scalar_text(
            raw.get("pg_id_kwd") or raw.get("pg_id")
        ):
            continue
        grouped[doc_type].append(dict(raw))

    result: Dict[str, List[Dict[str, Any]]] = {
        tool: [] for tool in tool_by_type.values()
    }
    for doc_type, locators in grouped.items():
        tool = tool_by_type[doc_type]
        type_limit = hydration_limit(doc_type)
        try:
            hydrated = icp_repository.get_records_by_storage_locators(
                doc_type,
                locators,
                limit=type_limit,
            )
        except Exception as exc:
            logger.warning("Semantic record hydration failed (%s): %s", doc_type, exc)
            if errors is not None:
                errors.setdefault(tool, []).append(str(exc))
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
            continue

        locator_by_id = {
            str(item.get("pg_id_kwd") or item.get("pg_id") or "").strip(): item
            for item in locators
        }
        for row in hydrated:
            item = dict(row)
            locator = locator_by_id.get(str(item.get("id") or ""), {})
            item["match_type"] = "semantic_id_hydration"
            item["retrieval_match_score"] = locator.get(
                "rerank_score", locator.get("score")
            )
            item["retrieval_source_chunk_id"] = locator.get("chunk_id") or ""
            result[tool].append(item)
    return result


def _source_section(value: Any) -> str:
    text = str(value or "").strip()
    if "#" in text:
        text = text.split("#", 1)[1]
    return text.split(";block=", 1)[0].strip()


def _verified_function_text_evidence(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int = 2,
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Promote an indexed architecture paragraph only after byte-source check.

    Function Locator narrative paragraphs can contain a data-flow fact which
    is not represented by the component/function tables.  OpenSearch may find
    that paragraph, but its copy is not authoritative.  We therefore reopen
    the registered ArtifactStore document, rebuild the same section grouping
    used during indexing and require exact whitespace-normalized equality.
    A title or semantic similarity alone can never pass this check.
    """

    candidates = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and _search_doc_type(dict(row)) == "function_locator"
        and _scalar_text(row.get("content_type")) == "function_text"
        and _scalar_text(row.get("doc_id"))
        and _scalar_text(row.get("source_ref"))
        and clean_text(row.get("content"))
    ]
    result: List[Dict[str, Any]] = []
    document_cache: Dict[str, Dict[str, str]] = {}
    for hit in candidates:
        if len(result) >= max(int(limit), 0):
            break
        doc_id = _scalar_text(hit.get("doc_id"))
        section = _source_section(hit.get("source_ref"))
        if not section:
            continue
        if doc_id not in document_cache:
            try:
                doc = postgres_client.get_icp_document(doc_id)
                if not doc or str(doc.get("status") or "") not in {
                    "indexed", "kg_pending", "kg_done"
                }:
                    document_cache[doc_id] = {}
                    continue
                file_name = str(doc.get("file_name") or "")
                if os.path.splitext(file_name)[1].lower() != ".docx":
                    document_cache[doc_id] = {}
                    continue
                with artifact_store.materialize(
                    storage_key_from(doc), suffix=".docx"
                ) as local_path:
                    document = load_document_ir(local_path, source_name=file_name)
                grouped: Dict[str, List[str]] = defaultdict(list)
                for paragraph in document.paragraphs:
                    text = clean_text(paragraph.text)
                    if not text:
                        continue
                    paragraph_section = (
                        " / ".join(
                            clean_text(value)
                            for value in paragraph.section_path
                            if clean_text(value)
                        )
                        or "文档说明"
                    )
                    grouped[paragraph_section].append(text)
                document_cache[doc_id] = {
                    key: "\n".join(values) for key, values in grouped.items()
                }
            except Exception as exc:
                logger.warning("Function source paragraph verification failed: %s", exc)
                if errors is not None:
                    errors.append(str(exc))
                document_cache[doc_id] = {}
                continue

        source_content = document_cache[doc_id].get(section, "")
        indexed_content = str(hit.get("content") or "")
        if not source_content or clean_text(source_content) != clean_text(indexed_content):
            continue
        doc = postgres_client.get_icp_document(doc_id) or {}
        source_ref = f"{str(doc.get('file_name') or hit.get('doc_name') or '').strip()}#{section}"
        result.append(
            make_evidence(
                source_kind="artifact",
                authority="source_artifact",
                evidence_level="L1_DIRECT",
                claim_type="source_text_fact",
                doc_type="function_locator",
                doc_id=doc_id,
                doc_name=str(doc.get("file_name") or hit.get("doc_name") or ""),
                chunk_id=str(hit.get("chunk_id") or hit.get("id") or ""),
                record_id=str(hit.get("chunk_id") or hit.get("id") or source_ref),
                source_ref=source_ref,
                title=str(hit.get("title") or section),
                content=source_content,
                display_summary=source_content,
                match_type="verified_source_text",
                review_status="not_required",
                scope=_scope(hit.get("document_metadata")),
                retrieval_score=hit.get("score"),
                rerank_score=hit.get("rerank_score"),
            )
        )
    return result


def _explicit_failure_mode_ids(fta_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Read explicit FTA→FMECA identifiers from formally validated graphs."""

    keys = {
        "failure_mode_id",
        "failure_mode_ids",
        "failure_mode_ref",
        "fmeca_code",
        "fmeca_codes",
        "fmeca_ref",
    }
    result: List[str] = []

    def visit(value: Any, *, depth: int = 0, active_key: str = "") -> None:
        if depth > 8 or len(result) >= 80:
            return
        if isinstance(value, Mapping):
            for raw_key, child in list(value.items())[:400]:
                key = str(raw_key or "").strip().casefold()
                visit(child, depth=depth + 1, active_key=key)
            return
        if isinstance(value, (list, tuple, set)):
            for child in list(value)[:400]:
                visit(child, depth=depth + 1, active_key=active_key)
            return
        if active_key not in keys:
            return
        for match in _FMECA_BUSINESS_ID_RE.finditer(str(value or "")):
            code = match.group(0).upper()
            # The structured repository uses FM-* as the canonical record ID;
            # retain legacy FMECA-* references only when no direct FM form is
            # present rather than guessing a numeric conversion.
            if code not in result:
                result.append(code)

    for row in fta_rows:
        if not _is_formal_graph_row(row):
            continue
        visit(row.get("graph") or row.get("canonical_graph") or {})
    return result


def _query_fmeca_by_codes(
    codes: Sequence[str],
    limit: int,
    errors: Optional[List[str]] = None,
    *,
    match_type: str = "exact_fta_reference",
    reference_context: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    values = list(dict.fromkeys(str(code or "").strip() for code in codes if str(code or "").strip()))
    if not values:
        return []
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                "SELECT f.* FROM fmea f "
                "JOIN icp_documents d ON d.doc_id=f.doc_id "
                "WHERE d.status IN ('indexed','kg_pending','kg_done') "
                "AND f.code = ANY(%s) ORDER BY f.code LIMIT %s",
                (values, max(int(limit), 0)),
            )
            rows = _row_dicts(cur, cur.fetchall())
            for row in rows:
                row["match_type"] = match_type
                context = dict((reference_context or {}).get(str(row.get("code") or "")) or {})
                # An exact business ID authorizes the identity link, not an
                # arbitrary relevance boost.  FTA callers intentionally pass
                # no context.  History callers carry forward only the score of
                # the semantically matched PostgreSQL history row which
                # explicitly stored this code.
                row["retrieval_match_score"] = context.get("retrieval_match_score")
                if context:
                    row["reference_source"] = "fault_history"
                    row["linked_history_record_ids"] = list(
                        context.get("history_record_ids") or []
                    )
                    row["linked_history_case_ids"] = list(
                        context.get("history_case_ids") or []
                    )
            return rows
    except Exception as exc:
        logger.warning("Business-code-linked FMECA query failed: %s", exc)
        if errors is not None:
            errors.append(str(exc))
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _conversation_fmeca_codes(state: Mapping[str, Any]) -> List[str]:
    """Return only FMECA identities selected from server-offered prior citations."""

    values: List[str] = []
    for anchor in state.get("selected_conversation_anchors") or []:
        if not isinstance(anchor, Mapping) or str(anchor.get("doc_type") or "") != "fmeca":
            continue
        fields = anchor.get("answer_fields")
        candidates: List[Any] = [anchor.get("record_id")]
        if isinstance(fields, Mapping):
            candidates.extend(fields.get(key) for key in ("record_code",))
        for candidate in candidates:
            code = str(candidate or "").strip()
            if _FMECA_BUSINESS_ID_RE.fullmatch(code) and code not in values:
                values.append(code)
    return values[:12]


def _conversation_hmc_queries(state: Mapping[str, Any]) -> List[str]:
    """Build exact HMC field queries only from selected prior citations."""

    values: List[str] = []
    for anchor in state.get("selected_conversation_anchors") or []:
        if not isinstance(anchor, Mapping) or str(anchor.get("doc_type") or "") != "hmc":
            continue
        fields = anchor.get("answer_fields")
        if not isinstance(fields, Mapping):
            fields = {}
        code = str(fields.get("hmc_code") or anchor.get("record_id") or "").strip()
        if code and _extract_hmc(code):
            if code not in values:
                values.append(code)
            continue
        lcn = str(fields.get("lcn") or "").strip()
        hsc = str(fields.get("hsc") or "").strip()
        hamc = str(fields.get("hamc") or "").strip()
        if lcn and hsc:
            field_query = f"LCN {lcn} "
            if hamc:
                field_query += f"HAMC {hamc} "
            field_query += f"HSC {hsc}"
            if _extract_hmc(field_query) and field_query not in values:
                values.append(field_query)
    return values[:8]


def _query_hmc_from_conversation_anchors(
    queries: Sequence[str],
    terms: Sequence[str],
    limit: int,
    errors: Optional[List[str]],
    *,
    include_repair_method: bool,
) -> List[Dict[str, Any]]:
    """Re-read selected historical HMC records; never reuse their old prose."""

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for query in queries:
        remaining = max(int(limit) - len(rows), 1)
        for row in _query_hmc(
            query,
            terms,
            remaining,
            errors,
            include_repair_method=include_repair_method,
        ):
            identity = (
                str(row.get("doc_id") or ""),
                str(row.get("lcn") or ""),
                str(row.get("hamc") or ""),
                str(row.get("hsc") or ""),
            )
            if identity in seen:
                continue
            seen.add(identity)
            enriched = dict(row)
            enriched["match_type"] = "exact_conversation_reference"
            rows.append(enriched)
            if len(rows) >= max(int(limit), 0):
                return rows
    return rows


_HISTORY_REFERENCE_TABLES: Dict[str, Tuple[str, str]] = {
    "hmc": ("fault_hmc", "hmc_code"),
    "mfl": ("mfl", "mfl_code"),
}


def _semantic_history_reference_context(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return explicit cross-document IDs from semantically hydrated history.

    Only a history row first discovered by OpenSearch and then re-read through
    its exact PostgreSQL storage identity may seed this expansion.  Keyword
    history rows are deliberately excluded: otherwise a broad history query
    could fan out into unrelated document families.
    """

    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if str(row.get("match_type") or "") != "semantic_id_hydration":
            continue
        record_id = str(row.get("id") or "").strip()
        case_id = str(row.get("case_id") or "").strip()
        score = row.get("retrieval_match_score")
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = 0.0
        if normalized_score < _SEMANTIC_HISTORY_REFERENCE_MIN_SCORE:
            continue
        for ref_type, codes in _explicit_history_references(row).items():
            for code in codes:
                current = result[ref_type].setdefault(
                    code,
                    {
                        "retrieval_match_score": score,
                        "history_record_ids": [],
                        "history_case_ids": [],
                    },
                )
                try:
                    new_score = float(score)
                except (TypeError, ValueError):
                    new_score = None
                try:
                    current_score = float(current.get("retrieval_match_score"))
                except (TypeError, ValueError):
                    current_score = None
                if new_score is not None and (
                    current_score is None or new_score > current_score
                ):
                    current["retrieval_match_score"] = score
                if record_id and record_id not in current["history_record_ids"]:
                    current["history_record_ids"].append(record_id)
                if case_id and case_id not in current["history_case_ids"]:
                    current["history_case_ids"].append(case_id)
    return {ref_type: dict(values) for ref_type, values in result.items()}


def _query_history_linked_rows(
    ref_type: str,
    reference_context: Mapping[str, Mapping[str, Any]],
    limit: int,
    errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch HMC/MFL rows by explicit history business code.

    Table and identifier names come exclusively from the closed internal map;
    user/history values remain bound SQL parameters.
    """

    table_spec = _HISTORY_REFERENCE_TABLES.get(ref_type)
    values = list(
        dict.fromkeys(
            str(code or "").strip()
            for code in reference_context
            if str(code or "").strip()
        )
    )
    if table_spec is None or not values:
        return []
    table, code_column = table_spec
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT t.* FROM {table} t "
                "JOIN icp_documents d ON d.doc_id=t.doc_id "
                "WHERE d.status IN ('indexed','kg_pending','kg_done') "
                f"AND t.{code_column} = ANY(%s) ORDER BY t.{code_column} LIMIT %s",
                (values, max(int(limit), 0)),
            )
            rows = _row_dicts(cur, cur.fetchall())
            for row in rows:
                context = dict(reference_context.get(str(row.get(code_column) or "")) or {})
                row["match_type"] = "exact_history_reference"
                row["retrieval_match_score"] = context.get("retrieval_match_score")
                row["reference_source"] = "fault_history"
                row["linked_history_record_ids"] = list(
                    context.get("history_record_ids") or []
                )
                row["linked_history_case_ids"] = list(
                    context.get("history_case_ids") or []
                )
            return rows
    except Exception as exc:
        logger.warning("History-linked %s query failed: %s", ref_type.upper(), exc)
        if errors is not None:
            errors.append(str(exc))
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass
        return []


def _retrieval_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _filter_search_scope(
    rows: Sequence[Dict[str, Any]],
    allowed_doc_types: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Keep OpenSearch hits inside the policy scope and make an implicit singleton explicit."""
    allowed = list(dict.fromkeys(value for value in allowed_doc_types if value in ICP_DOC_TYPES))
    allowed_set = set(allowed)
    accepted: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        item = dict(row)
        doc_type = _search_doc_type(item)
        if not doc_type and len(allowed) == 1:
            # Some older indexes omit doc_type from _source. Because the query itself
            # was restricted to one type, record that scope explicitly.
            item["doc_type_kwd"] = allowed[0]
            doc_type = allowed[0]
        if doc_type not in allowed_set:
            dropped += 1
            continue
        accepted.append(item)
    warnings = []
    if dropped:
        warnings.append(f"丢弃 {dropped} 条超出策略文档类型范围的 OpenSearch 命中")
    return accepted, warnings


def _readable_direct_fields(
    fields: Sequence[Tuple[str, Any]],
    *,
    steps: Any = None,
) -> str:
    """Render structured direct facts as compact, user-readable evidence.

    PostgreSQL remains the structured source of truth.  This projection only
    controls what the answer model and evidence panel read; empty columns are
    omitted instead of producing a dense ``field=；field=`` record.
    """

    lines = [
        f"{label}：{str(value).strip()}"
        for label, value in fields
        if value is not None and str(value).strip()
    ]
    if steps is None:
        return "\n".join(lines)

    if isinstance(steps, (str, bytes)):
        raw_steps: Sequence[Any] = [steps]
    elif isinstance(steps, Sequence):
        raw_steps = steps
    else:
        raw_steps = []

    step_lines: List[str] = []
    for fallback_no, raw_step in enumerate(raw_steps, start=1):
        if isinstance(raw_step, Mapping):
            content = str(
                raw_step.get("content")
                or raw_step.get("step")
                or raw_step.get("description")
                or ""
            ).strip()
            step_no = raw_step.get("step_no") or fallback_no
        else:
            content = str(raw_step or "").strip()
            step_no = fallback_no
        if content:
            step_lines.append(f"{step_no}. {content}")
    if step_lines:
        lines.append("操作步骤：")
        lines.extend(step_lines)
    return "\n".join(lines)


def _has_answer_value(value: Any) -> bool:
    """Keep meaningful false/zero values while omitting genuinely empty fields."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return bool(value)
    return True


def _first_answer_value(row: Mapping[str, Any], *keys: str) -> Any:
    """Return only a value explicitly carried by one of the compatible row keys."""

    for key in keys:
        if key in row and _has_answer_value(row.get(key)):
            return row.get(key)
    return None


def _answer_fields(**fields: Any) -> Dict[str, Any]:
    """Build a compact structured projection for downstream answer coverage checks."""

    result: Dict[str, Any] = {}
    for key, value in fields.items():
        if not _has_answer_value(value):
            continue
        if isinstance(value, Mapping):
            nested = _answer_fields(**dict(value))
            if nested:
                result[key] = nested
        elif isinstance(value, (list, tuple, set)):
            cleaned = [
                item if isinstance(item, (bool, int, float)) else str(item).strip()
                for item in value
                if _has_answer_value(item)
            ]
            if cleaned:
                result[key] = cleaned
        elif isinstance(value, (bool, int, float)):
            result[key] = value
        else:
            result[key] = str(value).strip()
    return result


def _yes_no(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value).strip()


def _explicit_history_references(row: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Collect only references explicitly stored on a history record.

    Parser-normalized ``document_links`` and legacy scalar columns are both
    accepted.  Values are not tokenized or similarity-matched here, so this
    projection cannot manufacture a cross-document association.
    """

    links = _as_metadata(row.get("document_links"))
    sources = {
        "hmc": (
            ("hmc_code", "hmc_ref", "initial_hmc"),
            ("hmc_codes", "hmc_refs"),
        ),
        "fmeca": (
            ("fmeca_code", "fmeca_ref", "failure_mode_id"),
            ("fmeca_refs", "fmeca_codes", "failure_mode_ids"),
        ),
        "fta": (("fta_ref", "fta_code"), ("fta_refs", "fta_codes")),
        "rbd": (("rbd_ref", "rbd_code"), ("rbd_refs", "rbd_codes")),
        "mfl": (("mfl_code", "mfl_ref"), ("mfl_refs", "mfl_codes")),
    }
    result: Dict[str, List[str]] = {}
    for ref_type, (row_keys, link_keys) in sources.items():
        values: List[Any] = [row.get(key) for key in row_keys if key in row]
        values.extend(links.get(key) for key in link_keys if key in links)
        flattened: List[str] = []
        for value in values:
            items = value if isinstance(value, (list, tuple, set)) else [value]
            for item in items:
                text = str(item or "").strip()
                if text and text not in flattened:
                    flattened.append(text)
        if flattened:
            result[ref_type] = flattened
    return result


def _direct_evidence(tool: str, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        doc_name = str(row.get("doc_name") or "")
        source_ref = str(row.get("source_ref") or row.get("source_section") or "")
        scope = _scope(row.get("scope") or row.get("document_metadata"))
        display_summary = ""
        direct_metadata: Dict[str, Any] = {}
        if row.get("reference_source") == "fault_history":
            direct_metadata.update(
                {
                    "reference_source": "fault_history",
                    "linked_history_record_ids": list(
                        row.get("linked_history_record_ids") or []
                    ),
                    "linked_history_case_ids": list(
                        row.get("linked_history_case_ids") or []
                    ),
                }
            )
        if tool == "function_locator":
            content_type = str(row.get("content_type") or "function_record")
            record_id = _record_id(row, "function_id", "component_id", "interface_id", "id")
            if content_type == "function_component":
                content = _readable_direct_fields(
                    (
                        ("组件ID", row.get("component_id")),
                        ("类型", row.get("component_type")),
                        ("名称", row.get("name")),
                        ("简称", row.get("abbreviation")),
                        ("父组件ID", row.get("parent_component_id")),
                        ("说明", row.get("description")),
                    )
                )
                title = str(row.get("name") or record_id)
                claim_type = "component_fact"
                direct_metadata["answer_fields"] = _answer_fields(
                    component_id=row.get("component_id"),
                    component_type=row.get("component_type"),
                    component_name=row.get("name"),
                    abbreviation=row.get("abbreviation"),
                    parent_component_id=row.get("parent_component_id"),
                    description=row.get("description"),
                    collection_closed=row.get("collection_closed"),
                    collection_parent_id=row.get("collection_parent_id"),
                    collection_parent_name=row.get("collection_parent_name"),
                    collection_member_kind=row.get("collection_member_kind"),
                    collection_member_count=row.get("collection_member_count"),
                    collection_role=row.get("collection_role"),
                    collection_member_id=row.get("collection_member_id"),
                    collection_member_name=row.get("collection_member_name"),
                    collection_member_description=row.get("collection_member_description"),
                )
            elif content_type == "function_interface":
                content = _readable_direct_fields(
                    (
                        ("接口ID", row.get("interface_id")),
                        ("源组件", row.get("source_component_id")),
                        ("目标组件", row.get("target_component_id")),
                        ("关系", row.get("relation_type")),
                        ("信号", row.get("signal_name")),
                        ("方向", row.get("direction")),
                        ("协议", row.get("protocol")),
                        ("说明", row.get("description")),
                    )
                )
                title = str(row.get("signal_name") or record_id)
                claim_type = "interface_fact"
                direct_metadata["answer_fields"] = _answer_fields(
                    interface_id=row.get("interface_id"),
                    source_component_id=row.get("source_component_id"),
                    target_component_id=row.get("target_component_id"),
                    relation_type=row.get("relation_type"),
                    signal_name=row.get("signal_name"),
                    direction=row.get("direction"),
                    protocol=row.get("protocol"),
                    description=row.get("description"),
                )
            else:
                content = _readable_direct_fields(
                    (
                        ("系统", row.get("system_name")),
                        ("子系统", row.get("subsystem_name")),
                        ("模块", row.get("module_name")),
                        ("功能", row.get("function_name")),
                        ("说明", row.get("description")),
                        ("输入", "；".join(row.get("inputs") or [])),
                        ("输出", "；".join(row.get("outputs") or [])),
                        ("典型失效表现", row.get("failure_symptoms")),
                    )
                )
                title = str(row.get("function_name") or row.get("name") or record_id)
                claim_type = "function_fact"
                direct_metadata["answer_fields"] = _answer_fields(
                    function_id=row.get("function_id"),
                    system_name=row.get("system_name"),
                    subsystem_name=row.get("subsystem_name"),
                    module_name=row.get("module_name"),
                    function_name=row.get("function_name"),
                    description=row.get("description"),
                    inputs=row.get("inputs"),
                    outputs=row.get("outputs"),
                    failure_symptoms=row.get("failure_symptoms"),
                )
            display_summary = content
            doc_type = "function_locator"
        elif tool == "hmc_query":
            record_id = _record_id(row, "hmc_code", "id")
            title = str(row.get("hmc_code") or record_id)
            doc_type = "hmc"
            hmc_metadata = _as_metadata(row.get("document_metadata"))
            hmc_metadata.update(
                {
                    "hmc_code": str(row.get("hmc_code") or ""),
                    "lcn": str(row.get("lcn") or ""),
                    "hamc": str(row.get("hamc") or ""),
                    "hsc": str(row.get("hsc") or ""),
                    "answer_fields": _answer_fields(
                        hmc_code=row.get("hmc_code"),
                        lcn=row.get("lcn"),
                        hamc=row.get("hamc"),
                        hsc=row.get("hsc"),
                        fault_symptom=row.get("fault_desc"),
                        maintenance_action=row.get("repair_method"),
                    ),
                }
            )
            hmc_metadata.update(direct_metadata)
            symptom = make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="hmc_fault",
                doc_type=doc_type,
                doc_id=doc_id,
                doc_name=doc_name,
                record_id=record_id,
                source_ref=source_ref,
                title=title,
                content=str(row.get("fault_desc") or ""),
                display_summary=_readable_direct_fields(
                    (("故障现象", row.get("fault_desc")),)
                ),
                match_type=str(row.get("match_type") or "structured_query"),
                scope=scope,
                retrieval_score=row.get("retrieval_match_score"),
                metadata=hmc_metadata,
            )
            evidence.append(symptom)
            repair = str(row.get("repair_method") or "").strip()
            if repair:
                evidence.append(
                    make_evidence(
                        source_kind="postgres",
                        authority="authoritative_record",
                        evidence_level="L1_DIRECT",
                        claim_type="maintenance_action",
                        doc_type=doc_type,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        record_id=record_id,
                        source_ref=source_ref,
                        title=f"{title} 维修方法",
                        content=repair,
                        display_summary=_readable_direct_fields(
                            (("维修措施", repair),)
                        ),
                        match_type=str(row.get("match_type") or "structured_query"),
                        scope=scope,
                        retrieval_score=row.get("retrieval_match_score"),
                        metadata=hmc_metadata,
                    )
                )
            continue
        elif tool == "fmeca_query":
            record_id = _record_id(row, "code", "id")
            title = str(row.get("fault_mode") or record_id)
            probability_raw = _first_answer_value(row, "probability_raw", "failure_probability_raw")
            probability = _first_answer_value(row, "probability", "failure_probability")
            probability_display = probability_raw if _has_answer_value(probability_raw) else probability
            content = _readable_direct_fields(
                (
                    ("产品/功能标志", row.get("product_flag")),
                    ("功能", row.get("func_management")),
                    ("故障模式", row.get("fault_mode")),
                    ("故障原因", row.get("fault_cause")),
                    ("任务阶段与工作方式", row.get("mission_phase")),
                    ("局部影响", row.get("local_impact")),
                    ("最终影响", row.get("final_impact")),
                    ("严酷度", row.get("severity")),
                    ("故障概率", probability_display),
                    ("故障概率等级", row.get("probability_level")),
                    ("检测方法", row.get("detect_method")),
                    ("设计改进措施", row.get("design_improve")),
                    ("使用补偿措施", row.get("compensate_action")),
                )
            )
            display_summary = content
            claim_type = "fmeca_fact"
            doc_type = "fmeca"
            direct_metadata["answer_fields"] = _answer_fields(
                record_code=row.get("code"),
                product_flag=row.get("product_flag"),
                function=row.get("func_management"),
                failure_mode=row.get("fault_mode"),
                fault_cause=row.get("fault_cause"),
                mission_phase=row.get("mission_phase"),
                local_impact=row.get("local_impact"),
                final_impact=row.get("final_impact"),
                severity=row.get("severity"),
                failure_probability=probability_display,
                failure_probability_normalized=(
                    probability
                    if _has_answer_value(probability_raw) and _has_answer_value(probability)
                    else None
                ),
                failure_probability_level=row.get("probability_level"),
                detection_method=row.get("detect_method"),
                design_improvement=row.get("design_improve"),
                compensation_action=row.get("compensate_action"),
                collection_closed=row.get("collection_closed"),
                collection_parent_id=row.get("collection_parent_id"),
                collection_parent_name=row.get("collection_parent_name"),
                collection_member_kind=row.get("collection_member_kind"),
                collection_member_count=row.get("collection_member_count"),
                collection_role=row.get("collection_role"),
                collection_member_id=row.get("collection_member_id"),
                collection_member_name=row.get("collection_member_name"),
                collection_member_description=row.get("collection_member_description"),
            )
        elif tool == "mfl_query":
            record_id = _record_id(row, "mfl_code", "id")
            title = str(row.get("mfl_code") or record_id)
            api_reportable = _first_answer_value(row, "api_reportable")
            api_reportable_raw = _first_answer_value(row, "api_reportable_raw")
            platform_test = _first_answer_value(row, "platform_test")
            platform_test_raw = _first_answer_value(row, "platform_test_raw")
            maintenance_action = _first_answer_value(
                row, "maintenance_action", "repair_action"
            )
            content = _readable_direct_fields(
                (
                    ("模块故障模式", row.get("module_fault")),
                    ("SUR板故障模式", row.get("sur_fault")),
                    ("功能子电路级故障模式", row.get("subcircuit_fault")),
                    ("API可直接上报", _yes_no(api_reportable)),
                    ("API上报原始值", api_reportable_raw),
                    ("API名称", row.get("api_name")),
                    ("API调用方法", row.get("api_method")),
                    ("API使用方法", row.get("api_usage")),
                    ("平台级实验状态", row.get("platform_test_status")),
                    ("平台级实验结果", _yes_no(platform_test)),
                    ("平台级实验原始值", platform_test_raw),
                    ("维修措施", maintenance_action),
                )
            )
            display_summary = content
            claim_type = "maintenance_locator"
            doc_type = "mfl"
            direct_metadata["answer_fields"] = _answer_fields(
                record_code=row.get("mfl_code"),
                module_failure=row.get("module_fault"),
                sur_failure=row.get("sur_fault"),
                subcircuit_failure=row.get("subcircuit_fault"),
                api_reportable=api_reportable,
                api_reportable_raw=api_reportable_raw,
                api_name=row.get("api_name"),
                api_method=row.get("api_method"),
                api_usage=row.get("api_usage"),
                platform_test=platform_test,
                platform_test_raw=platform_test_raw,
                platform_test_status=row.get("platform_test_status"),
                maintenance_action=maintenance_action,
            )
        elif tool == "fault_history_query":
            record_id = _record_id(row, "case_id", "id")
            title = str(row.get("case_id") or record_id)
            mission_phase = _first_answer_value(
                row, "mission_phase", "mission_stage", "task_stage", "task_phase", "operation_stage"
            )
            device_type = _first_answer_value(
                row, "device_type", "equipment_type", "device_model"
            )
            related_module = _first_answer_value(
                row, "related_module", "module_name", "fault_module"
            )
            related_function = _first_answer_value(
                row, "related_function", "function_id", "function_name"
            )
            operator = _first_answer_value(
                row, "personnel", "operator", "maintenance_personnel", "troubleshooter"
            )
            references = _explicit_history_references(row)
            content = _readable_direct_fields(
                (
                    ("发生时间", row.get("occurred_at")),
                    ("任务阶段/工作模式", mission_phase),
                    ("设备类型", device_type),
                    ("关联模块", related_module),
                    ("关联功能", related_function),
                    ("排故人员", operator),
                    ("故障现象", row.get("fault_symptom")),
                    ("根因", row.get("root_cause")),
                    ("排故方法", row.get("diagnosis_method")),
                    ("最终方案", row.get("final_solution")),
                    ("验证结果", row.get("verification_result")),
                    ("预防措施", row.get("preventive_action")),
                    ("关联 HMC", "；".join(references.get("hmc", []))),
                    ("关联 FMECA", "；".join(references.get("fmeca", []))),
                    ("关联 FTA", "；".join(references.get("fta", []))),
                    ("关联 RBD", "；".join(references.get("rbd", []))),
                    ("关联 MFL", "；".join(references.get("mfl", []))),
                ),
                steps=row.get("steps"),
            )
            display_summary = content
            claim_type = "historical_case"
            doc_type = "fault_history"
            direct_metadata["answer_fields"] = _answer_fields(
                case_id=row.get("case_id"),
                occurred_at=row.get("occurred_at"),
                mission_phase=mission_phase,
                device_type=device_type,
                related_module=related_module,
                related_function=related_function,
                operator=operator,
                fault_symptom=row.get("fault_symptom"),
                root_cause=row.get("root_cause"),
                diagnosis_method=row.get("diagnosis_method"),
                steps=row.get("steps"),
                final_solution=row.get("final_solution"),
                verification_result=row.get("verification_result"),
                preventive_action=row.get("preventive_action"),
                explicit_references=references,
            )
        elif tool in {"fta_query", "rbd_query"}:
            # The user-facing identity is the stable engineering diagram code
            # (for example FTA-SYN-003), while revision UUIDs remain provenance
            # metadata for exact source reconstruction.
            record_id = _record_id(
                row,
                "diagram_key",
                "revision_id",
                "diagram_id",
                "id",
            )
            title = str(row.get("title") or row.get("image_title") or record_id)
            graph = row.get("graph") or {}
            content = json.dumps(graph, ensure_ascii=False, default=str)
            claim_type = "fta_structure" if tool == "fta_query" else "rbd_structure"
            doc_type = "fta" if tool == "fta_query" else "rbd"
            display_summary = summarize_graph_structure(
                graph,
                diagram_type=doc_type,
                title=title,
            )
            validation_status = _graph_validation_status(row)
            if validation_status in FORMAL_VALIDATION_STATUSES:
                manually_confirmed = validation_status == "manual_confirmed"
                authority = "confirmed_relation" if manually_confirmed else "validated_relation"
                evidence_level = "L1_DIRECT" if manually_confirmed else "L2_CORROBORATED"
                limitations = (
                    []
                    if manually_confirmed
                    else ["工程图结构已通过确定性自动验证，但不等同于人工确认"]
                )
                if not source_ref:
                    source_ref = deterministic_graph_source_ref(graph)
            elif validation_status == VALIDATION_PROVISIONAL:
                authority = "unconfirmed_extraction"
                evidence_level = "L4_UNRESOLVED"
                limitations = ["工程图仅通过视觉结构检查，未达到正式自动验证等级"]
            else:
                # Quarantined/failed/pending graph revisions are audit records,
                # not reasoning context.  Fail closed even if a repository test
                # double or a stale index returns one unexpectedly.
                continue
            evidence.append(
                make_evidence(
                    source_kind="postgres",
                    authority=authority,
                    evidence_level=evidence_level,
                    claim_type=claim_type,
                    doc_type=doc_type,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    record_id=record_id,
                    source_ref=source_ref,
                    title=title,
                    content=content,
                    display_summary=display_summary,
                    review_status=validation_status,
                    scope=scope,
                    limitations=limitations,
                    metadata={
                        "is_active": row.get("is_active") is True,
                        "projection_status": str(row.get("projection_status") or ""),
                        "diagram_id": str(row.get("diagram_id") or ""),
                        "revision_id": str(row.get("revision_id") or ""),
                        "diagram_key": str(row.get("diagram_key") or ""),
                        "validation_status": validation_status,
                        "validation_tier": str(row.get("validation_tier") or ""),
                        "validation_method": str(row.get("validation_method") or ""),
                        "validator_version": str(row.get("validator_version") or ""),
                        # The canonical graph is deliberately kept in content
                        # for provenance.  This compact structured field lets
                        # answer completion recognize a validated FTA cause
                        # structure without ever treating JSON as user prose.
                        "answer_fields": _answer_fields(
                            structural_cause_chain=(
                                display_summary if tool == "fta_query" else None
                            ),
                            structure_summary=display_summary,
                            basic_events=(
                                _graph_basic_event_summaries(graph)
                                if tool == "fta_query"
                                else None
                            ),
                        ),
                    },
                )
            )
            continue
        else:
            continue
        evidence.append(
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type=claim_type,
                doc_type=doc_type,
                doc_id=doc_id,
                doc_name=doc_name,
                record_id=record_id,
                source_ref=source_ref,
                title=title,
                content=content,
                display_summary=display_summary,
                match_type=str(row.get("match_type") or "structured_query"),
                scope=scope,
                retrieval_score=row.get("retrieval_match_score"),
                metadata=direct_metadata,
            )
        )
    return evidence


def _search_evidence(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        content_type = str(row.get("content_type") or "")
        is_graph = _is_graph_locator_hit(row)
        validation_status = str(
            row.get("validation_status")
            or row.get("validation_status_kwd")
            or ""
        ).strip()
        if is_graph and validation_status in {VALIDATION_QUARANTINED, VALIDATION_FAILED}:
            # Search indexes are rebuildable projections.  A stale projection
            # must never re-introduce a graph revision which PostgreSQL has
            # explicitly quarantined or whose validator failed.
            continue
        result.append(
            make_evidence(
                source_kind="opensearch",
                authority="unconfirmed_extraction" if is_graph else "search_projection",
                evidence_level="L4_UNRESOLVED" if is_graph else "L3_INFERRED",
                claim_type="diagram_structure" if is_graph else "retrieval_hit",
                doc_type=_search_doc_type(row),
                doc_id=str(row.get("doc_id") or ""),
                doc_name=str(row.get("doc_name") or ""),
                chunk_id=str(row.get("chunk_id") or row.get("id") or ""),
                source_ref=str(row.get("source_ref") or row.get("image_url") or ""),
                title=str(row.get("title") or ""),
                content=str(row.get("content") or ""),
                display_summary=(
                    summarize_diagram_locator(
                        doc_type=_search_doc_type(row),
                        title=str(row.get("title") or ""),
                    )
                    if is_graph
                    else ""
                ),
                match_type="hybrid_search",
                review_status=(
                    VALIDATION_PROVISIONAL
                    if is_graph and validation_status == VALIDATION_PROVISIONAL
                    else "pending" if is_graph else "not_required"
                ),
                scope=_scope(row.get("document_metadata")),
                retrieval_score=row.get("score"),
                rerank_score=row.get("rerank_score"),
                limitations=["OpenSearch 是可重建检索投影，不是结构化事实真源"],
            )
        )
    return result


def _format_architecture_context(query: str, results: Sequence[Dict[str, Any]]) -> str:
    sections = [
        f"【用户架构问题】\n{query}",
        "【证据使用规则】\nPostgreSQL 功能记录为权威结构化事实；OpenSearch 结果是可重建检索投影。"
        "auto_validated/manual_confirmed 架构关系可作正式结构证据；provisional 只能作为视觉候选。",
    ]
    if not results:
        sections.append("【检索结果】\n未找到匹配的 Function Locator 权威记录或检索投影。")
    else:
        lines = []
        for index, item in enumerate(results, start=1):
            authority = item.get("authority")
            source_label = "PostgreSQL 权威记录" if authority == "postgres_authoritative" else "OpenSearch 检索投影"
            content_type = str(item.get("content_type") or "")
            validation_status = str(
                item.get("validation_status")
                or item.get("validation_status_kwd")
                or ""
            )
            review_note = (
                "；视觉候选·仅供浏览"
                if content_type in {"function_architecture_diagram", "function_architecture_edge"}
                and validation_status not in FORMAL_VALIDATION_STATUSES
                else ""
            )
            metadata_text = _metadata_summary(_as_metadata(item.get("document_metadata")))
            source_ref = item.get("source_ref") or item.get("image_url") or item.get("chunk_id") or ""
            lines.append(
                f"- [^{index}] {source_label}{review_note}｜{item.get('title', '')}\n"
                f"  {str(item.get('content') or '')[:900]}\n"
                f"  来源：{item.get('doc_name') or item.get('doc_id') or '未标注文档'}"
                f"；定位：{source_ref or '未标注'}"
                f"{f'；{metadata_text}' if metadata_text else ''}"
            )
        sections.append("【系统—子系统—模块—功能/接口证据】\n" + "\n".join(lines))
    sections.append(
        "【HugeGraph 关系补充】\nFunction Locator 的显式关系会自动生成 HugeGraph 可重建投影；"
        "本回答直接读取 PostgreSQL 权威组件/功能/接口事实，不用图投影反向覆盖真源。"
    )
    sections.append(
        "【回答边界】\n缺少接口、输入输出或上下游证据时应明确写“资料未记载/仍需确认”，不得按常识补造。"
    )
    return "\n\n".join(sections)


def _format_domain_context(
    query: str,
    function_rows: Sequence[Dict[str, Any]],
    hmc_rows: Sequence[Dict[str, Any]],
    fmeca_rows: Sequence[Dict[str, Any]],
    fta_rows: Sequence[Dict[str, Any]],
    rbd_rows: Sequence[Dict[str, Any]],
    mfl_rows: Sequence[Dict[str, Any]],
    history_rows: Sequence[Dict[str, Any]],
    hybrid_rows: Sequence[Dict[str, Any]],
    confidence: float,
    clarification: str,
) -> str:
    sections = [
        f"【用户故障描述】\n{query}",
        f"【当前证据支持度（非故障概率）】\n{confidence:.2f}",
    ]

    if function_rows:
        lines = []
        for row in function_rows[:6]:
            lines.append(
                f"- {row.get('function_id', '')} {row.get('function_name', '')}；"
                f"系统/模块：{row.get('system_name', '')}/{row.get('module_name', '')}；"
                f"说明：{row.get('description', '') or row.get('failure_symptoms', '')}"
            )
        sections.append("【Function Locator 候选】\n" + "\n".join(lines))

    if hmc_rows:
        lines = []
        for row in hmc_rows[:8]:
            code = row.get("hmc_code") or f"{row.get('lcn', '')}{row.get('hamc', '')}{row.get('hsc', '')}"
            lines.append(
                f"- HMC {code}：{row.get('fault_desc', '')}；维修：{row.get('repair_method', '')}；"
                f"匹配方式：{row.get('match_type', '结构化查询')}"
            )
        sections.append("【HMC】\n" + "\n".join(lines))

    if fmeca_rows:
        lines = []
        for row in fmeca_rows[:6]:
            lines.append(
                f"- {row.get('code', '')} {row.get('fault_mode', '')}；原因：{row.get('fault_cause', '')}；"
                f"局部影响：{row.get('local_impact', '')}；最终影响：{row.get('final_impact', '')}；"
                f"严酷度：{row.get('severity', '')}；检测：{row.get('detect_method', '')}"
            )
        sections.append("【FMECA】\n" + "\n".join(lines))

    for label, rows in (("FTA", fta_rows), ("RBD", rbd_rows)):
        if not rows:
            continue
        lines = []
        for row in rows[:4]:
            graph = row.get("graph") or {}
            summary = summarize_graph_structure(
                graph,
                diagram_type=label.casefold(),
                title=str(row.get("image_title") or row.get("title") or ""),
            )
            lines.append(f"- {row.get('image_title', '')}：{summary}")
        sections.append(f"【{label}】\n" + "\n".join(lines))

    if mfl_rows:
        lines = []
        for row in mfl_rows[:6]:
            lines.append(
                f"- {row.get('mfl_code', '')}：模块故障 {row.get('module_fault', '')}；"
                f"子电路 {row.get('subcircuit_fault', '')}；API {row.get('api_name', '')} {row.get('api_method', '')}；"
                f"用法：{row.get('api_usage', '')}"
            )
        sections.append("【MFL】\n" + "\n".join(lines))

    if history_rows:
        lines = []
        for row in history_rows[:5]:
            lines.append(
                f"- {row.get('case_id', '')}：现象 {row.get('fault_symptom', '')}；"
                f"根因 {row.get('root_cause', '')}；处置 {row.get('final_solution', '')}；"
                f"关联 HMC {row.get('hmc_code', '')}"
            )
        sections.append("【故障历史】\n" + "\n".join(lines))

    if hybrid_rows:
        lines = []
        for row in hybrid_rows[:10]:
            lines.append(
                f"- [{row.get('kb_id') or row.get('doc_type_kwd') or row.get('section', '')}] "
                f"{row.get('title', '')}：{(row.get('content') or '')[:320]}"
            )
        sections.append("【跨七类文档混合召回】\n" + "\n".join(lines))

    if clarification:
        sections.append("【仍需确认】\n" + clarification)
    return "\n\n".join(sections)


def _candidate_faults(
    hmc_rows: Sequence[Dict[str, Any]],
    fmeca_rows: Sequence[Dict[str, Any]],
    hybrid_rows: Sequence[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "score": 0.0,
            "evidence": [],
            "supporting_evidence_ids": [],
            "exact_hmc_match": False,
            "structured_hmc_match": False,
            "authoritative_support": False,
        }
    )
    search_scored: set[Tuple[str, str]] = set()

    def attach(candidate: Dict[str, Any], *, doc_type: str, record_id: str = "", chunk_id: str = "") -> None:
        for item in evidence:
            if item.get("doc_type") != doc_type:
                continue
            if record_id and str(item.get("record_id") or "") != record_id:
                continue
            if chunk_id and str(item.get("chunk_id") or "") != chunk_id:
                continue
            # 维修动作不反向支撑故障候选；OpenSearch 命中仍只是线索。
            if item.get("claim_type") == "maintenance_action":
                continue
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in candidate["supporting_evidence_ids"]:
                candidate["supporting_evidence_ids"].append(evidence_id)

    for row in hmc_rows:
        name = str(row.get("fault_desc") or row.get("hmc_code") or "").strip()
        if not name:
            continue
        candidate = candidates[name]
        match_type = str(row.get("match_type") or "")
        exact_match = match_type == "exact_code"
        component_match = match_type == "exact_components"
        candidate["score"] += 0.55 if exact_match else (0.45 if component_match else 0.3)
        candidate["exact_hmc_match"] = candidate["exact_hmc_match"] or exact_match
        candidate["structured_hmc_match"] = (
            candidate["structured_hmc_match"] or match_type in _STRUCTURED_HMC_MATCH_TYPES
        )
        candidate["authoritative_support"] = True
        candidate["evidence"].append({"source": "HMC", "id": row.get("hmc_code")})
        attach(candidate, doc_type="hmc", record_id=_record_id(row, "hmc_code", "id"))
    for row in fmeca_rows:
        name = str(row.get("fault_mode") or row.get("code") or "").strip()
        if not name:
            continue
        candidate = candidates[name]
        candidate["score"] += 0.25
        candidate["authoritative_support"] = True
        candidate["evidence"].append({"source": "FMECA", "id": row.get("code")})
        attach(candidate, doc_type="fmeca", record_id=_record_id(row, "code", "id"))
    for row in hybrid_rows[:12]:
        doc_type = _search_doc_type(row)
        if doc_type not in {"hmc", "fmeca"}:
            continue
        name = str(row.get("title") or "").strip()
        if not name:
            continue
        raw_score = row.get("score", 0)
        try:
            score = min(float(raw_score), 1.0)
        except Exception:
            score = 0.0
        candidate = candidates[name]
        search_key = (doc_type, name.casefold())
        # Multiple chunks with the same title are duplicate retrieval clues, not
        # independent corroboration.  Score the clue once while retaining a few
        # source IDs for auditability.
        if search_key not in search_scored:
            candidate["score"] += 0.1 + 0.15 * max(score, 0)
            search_scored.add(search_key)
        candidate["evidence"].append(
            {"source": doc_type, "id": row.get("chunk_id")}
        )
        attach(
            candidate,
            doc_type=doc_type,
            chunk_id=str(row.get("chunk_id") or row.get("id") or ""),
        )

    ranked = []
    for name, value in candidates.items():
        ranked.append(
            {
                "fault": name,
                "score": round(min(value["score"], 0.99), 3),
                "evidence": value["evidence"][:6],
                "supporting_evidence_ids": value["supporting_evidence_ids"][:8],
                "exact_hmc_match": bool(value["exact_hmc_match"]),
                "structured_hmc_match": bool(value["structured_hmc_match"]),
                "authoritative_support": bool(value["authoritative_support"]),
            }
        )
    ranked.sort(
        key=lambda item: (
            bool(item.get("exact_hmc_match")),
            bool(item.get("structured_hmc_match")),
            bool(item.get("authoritative_support")),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )
    return ranked[:8]


def _graph_seed_bindings(
    *,
    function_rows: Sequence[Dict[str, Any]] = (),
    hmc_rows: Sequence[Dict[str, Any]] = (),
    fmeca_rows: Sequence[Dict[str, Any]] = (),
    fta_rows: Sequence[Dict[str, Any]] = (),
    rbd_rows: Sequence[Dict[str, Any]] = (),
    mfl_rows: Sequence[Dict[str, Any]] = (),
    history_rows: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Bind authoritative row IDs to graph logical IDs without fuzzy merging.

    A PostgreSQL row always owns its business identifier, but the *query to
    row* binding is exact only for a complete HMC match. Rows found from user
    wording remain candidate anchors even though their graph ID is stable. This
    distinction lets graph traversal expand useful candidates without silently
    promoting semantic similarity into entity identity.
    """

    by_logical_id: Dict[str, Dict[str, Any]] = {}

    def add(
        entity_kind: str,
        business_id: Any,
        *,
        source_tool: str,
        row: Dict[str, Any],
        binding_status: str = "candidate",
        match_type: str = "semantic_candidate",
    ) -> None:
        value = str(business_id or "").strip()
        if not value:
            return
        try:
            logical_id = business_node_id(entity_kind, value)
        except Exception:
            return
        item = {
            "logical_id": logical_id,
            "entity_kind": entity_kind,
            "business_id": value,
            "binding_status": "exact" if binding_status == "exact" else "candidate",
            "match_type": match_type,
            "source_tool": source_tool,
            "doc_id": str(row.get("doc_id") or ""),
            "source_ref": str(row.get("source_ref") or ""),
        }
        existing = by_logical_id.get(logical_id)
        # An exact code binding dominates a semantic occurrence of the same ID.
        if existing is None or (
            existing.get("binding_status") != "exact"
            and item["binding_status"] == "exact"
        ):
            by_logical_id[logical_id] = item

    def add_diagram_occurrence(
        diagram_type: str,
        *,
        source_tool: str,
        row: Dict[str, Any],
    ) -> None:
        doc_id = str(row.get("doc_id") or "").strip()
        revision_id = str(row.get("revision_id") or "").strip()
        diagram_id = str(row.get("diagram_id") or "").strip()
        if not doc_id or not revision_id or not diagram_id:
            return
        occurrence_kind = f"{diagram_type}_diagram_occurrence"
        logical_id = document_text_node_id(
            doc_id,
            occurrence_kind,
            f"{revision_id}:{diagram_id}",
        )
        by_logical_id[logical_id] = {
            "logical_id": logical_id,
            "entity_kind": occurrence_kind,
            "business_id": "",
            "binding_status": "candidate",
            "match_type": "semantic_candidate",
            "source_tool": source_tool,
            "doc_id": doc_id,
            "source_ref": str(row.get("source_ref") or ""),
            "revision_id": revision_id,
            "diagram_id": diagram_id,
        }

    for row in function_rows:
        add("function", row.get("function_id"), source_tool="function_locator", row=row)
        add("component", row.get("module_id"), source_tool="function_locator", row=row)
        add("component", row.get("component_id"), source_tool="function_locator", row=row)

    for row in hmc_rows:
        match_type = str(row.get("match_type") or "keyword")
        add(
            "hmc_code",
            row.get("hmc_code"),
            source_tool="hmc_query",
            row=row,
            binding_status="exact" if match_type == "exact_code" else "candidate",
            match_type=match_type,
        )

    for row in fmeca_rows:
        add("fmeca_record", row.get("code"), source_tool="fmeca_query", row=row)
    for row in mfl_rows:
        add("mfl_record", row.get("mfl_code"), source_tool="mfl_query", row=row)
    for row in history_rows:
        add(
            "fault_history_case",
            row.get("case_id"),
            source_tool="fault_history_query",
            row=row,
        )

    # Only formal current diagrams may seed formal multi-hop traversal.
    for row in fta_rows:
        if not _is_formal_graph_row(row):
            continue
        add_diagram_occurrence("fta", source_tool="fta_query", row=row)
        diagram_key = row.get("diagram_key") or row.get("business_id")
        if _FTA_DIAGRAM_BUSINESS_ID_RE.fullmatch(str(diagram_key or "").strip()):
            add("fta_diagram", diagram_key, source_tool="fta_query", row=row)
        graph = row.get("graph") or row.get("canonical_graph") or {}
        for vertex in (graph.get("vertices") or [])[:8] if isinstance(graph, dict) else []:
            if not isinstance(vertex, dict):
                continue
            props = vertex.get("properties") if isinstance(vertex.get("properties"), dict) else {}
            event_id = (
                vertex.get("business_id") or vertex.get("event_id")
                or props.get("business_id") or props.get("event_id")
            )
            if _FTA_EVENT_BUSINESS_ID_RE.fullmatch(str(event_id or "").strip()):
                add("fta_event", event_id, source_tool="fta_query", row=row)

    for row in rbd_rows:
        if not _is_formal_graph_row(row):
            continue
        add_diagram_occurrence("rbd", source_tool="rbd_query", row=row)
        diagram_key = row.get("diagram_key") or row.get("business_id")
        if not _RBD_DIAGRAM_BUSINESS_ID_RE.fullmatch(str(diagram_key or "").strip()):
            continue
        add(
            "rbd_diagram",
            diagram_key,
            source_tool="rbd_query",
            row=row,
        )

    values = list(by_logical_id.values())
    exact = sorted(
        (item for item in values if item.get("binding_status") == "exact"),
        key=lambda item: (item.get("source_tool", ""), item.get("logical_id", "")),
    )
    selected = exact[:_GRAPH_SEED_LIMIT]
    if len(selected) >= _GRAPH_SEED_LIMIT:
        return selected

    # Round-robin candidate anchors across tools. One large FTA graph or one
    # wide FMECA hit set must not consume the entire traversal seed budget and
    # hide otherwise available HMC/history/module bridges.
    source_order = (
        "function_locator", "hmc_query", "fmeca_query", "fta_query",
        "rbd_query", "mfl_query", "fault_history_query",
    )
    grouped = {
        tool: sorted(
            (
                item for item in values
                if item.get("binding_status") != "exact"
                and item.get("source_tool") == tool
            ),
            key=lambda item: item.get("logical_id", ""),
        )
        for tool in source_order
    }
    position = 0
    while len(selected) < _GRAPH_SEED_LIMIT:
        added = False
        for tool in source_order:
            candidates = grouped[tool]
            if position < len(candidates):
                selected.append(candidates[position])
                added = True
                if len(selected) >= _GRAPH_SEED_LIMIT:
                    break
        if not added:
            break
        position += 1
    return selected


def _graph_request(
    bindings: Sequence[Dict[str, Any]], allowed_tools: Iterable[str]
) -> Dict[str, Any]:
    if "graph_traversal" not in set(allowed_tools or ()) or not bindings:
        return {}
    return {
        "mode": "subgraph",
        "direction": "both",
        "kb_ids": ["icp_structured"],
        "seed_ids": list(
            dict.fromkeys(
                str(item.get("logical_id") or "")
                for item in bindings
                if item.get("logical_id")
            )
        )[:_GRAPH_SEED_LIMIT],
        "relations": [],
        "max_depth": 3,
        "max_nodes": 80,
        "max_edges": 160,
        "max_paths": 20,
        # HugeGraph is only a projection. Every returned edge must still exist
        # as a current PostgreSQL assertion when the traversal is executed.
        "require_relation_reauthorization": True,
    }


async def icp_diagnosis_node(state) -> Dict[str, Any]:
    """按策略白名单执行只读工具，并生成统一 Evidence/ToolResult。"""
    scope_route = str(state.get("scope_route") or "")
    if scope_route not in {"", "unclassified", "knowledge_base"}:
        # Graph routing is the primary boundary. Keep a second execution-side
        # guard so a manually invoked node cannot turn a direct-response route
        # into a broad knowledge-base query.
        return {
            "diagnostic_context": "",
            "candidate_faults": [],
            "diagnostic_support": 0.0,
            "diagnostic_confidence": 0.0,
            "clarification_question": str(
                state.get("clarification_question") or ""
            ),
            "domain_results": [],
            "vector_results": [],
            "tool_calls": [],
            "tool_results": [],
            "evidence_registry": [],
            "search_locator_hits": [],
            "graph_seed_bindings": [],
            "graph_traversal_request": {},
        }
    query = str(state.get("rewritten_query") or state.get("query") or "").strip()
    raw_user_query = str(state.get("query") or query).strip()
    terms = _unique_terms(state)
    intent = state.get("intent", "symptom_diagnosis")
    answer_mode = answer_mode_for_intent(intent)
    diagnostic_narrowing = answer_mode in {"fault_diagnosis", "exact_code_lookup"}
    limit = max(4, int(state.get("sql_limit", 8) or 8))
    retrieval_top_k = max(8, int(state.get("retrieval_top_k", 12) or 12))
    allowed_tools_present = "allowed_tools" in state
    allowed_tools = set(state.get("allowed_tools") or [])
    authorization_ceiling = set(
        state.get("authorized_tools") or state.get("allowed_tools") or []
    )
    structured_hydration_tools = set(
        state.get("structured_hydration_tools") or []
    ) & authorization_ceiling
    target_intents = {
        str(item.get("intent") or "")
        for item in (state.get("query_targets") or [])
        if isinstance(item, Mapping) and str(item.get("intent") or "")
    }
    if intent:
        target_intents.add(str(intent))
    explicit_business_codes = extract_business_codes(
        [
            raw_user_query,
            (state.get("query_frame") or {}).get("entities")
            if isinstance(state.get("query_frame"), Mapping)
            else [],
        ]
    )
    exact_record_identity_lookup = (
        "entity_profile" in target_intents
        and "record_identity" in set(required_answer_fields(state))
        and bool(explicit_business_codes)
    )
    architecture_query = "architecture_query" in target_intents
    function_knowledge_intents = {
        "architecture_query", "entity_profile", "function_location",
    }
    function_knowledge_query = bool(target_intents & function_knowledge_intents)
    complete_collection_request = closed_inventory_requested(state)
    # A complete-collection question asks what exists in one authoritative
    # scope; it is not a request to narrow those members to a single diagnosis.
    diagnostic_narrowing = diagnostic_narrowing and not complete_collection_request
    # 只有单一类实体/架构知识目标可以走专用的提前返回。若同一问题
    # 还包含影响、原因、维修等目标，必须继续执行其余已授权查询；
    # 例如“这个模块重要吗”需要同时取回职责和有证据的失效影响。
    function_knowledge_only = (
        function_knowledge_query
        and not complete_collection_request
        and not (target_intents - function_knowledge_intents)
    )
    policy_hybrid_doc_types: List[str] = []
    # 只对旧版/手工构造且缺少该字段的状态做兼容降级。显式空集合
    # 代表策略决定不执行直接查询，不能被再次扩张为固定工具套餐。
    if not allowed_tools_present:
        fallback_policy = resolve_tool_policy(
            intent=intent,
            proposed_plan=state.get("tools_plan") or [],
            query=query,
            type_keywords=state.get("type_keywords") or [],
            query_targets=state.get("query_targets") or [],
            raw_user_query=raw_user_query,
            coverage_scope=(state.get("query_frame") or {}).get(
                "coverage_scope", "unknown"
            ),
            collection_member_kind=(state.get("query_frame") or {}).get(
                "collection_member_kind", "unknown"
            ),
            relation_expansion_required=bool(
                isinstance(state.get("relation_expansion"), Mapping)
                and state["relation_expansion"].get("needed") is True
            ),
        )
        allowed_tools = set(fallback_policy["allowed_tools"])
        policy_hybrid_doc_types = list(fallback_policy["hybrid_doc_types"])
    # ``entity_profile`` describes the user's information need, not a storage
    # family.  FMECA/MFL/history/HMC business IDs use the same intent but must
    # reach their own PostgreSQL query instead of being swallowed by the
    # Function Locator fast path.
    structured_record_tools = {
        "hmc_query",
        "fmeca_query",
        "fta_query",
        "rbd_query",
        "mfl_query",
        "fault_history_query",
    }
    function_knowledge_only = function_knowledge_only and not bool(
        allowed_tools & structured_record_tools
    )
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    function_rows: List[Dict[str, Any]] = []
    function_hybrid: List[Dict[str, Any]] = []
    function_retrieval_info: Dict[str, Any] = {}
    function_search_errors: List[str] = []
    function_fallbacks: List[str] = []
    function_scope_warnings: List[str] = []
    function_call: Optional[Dict[str, Any]] = None
    structured_entity_binding: Dict[str, Any] = {
        "bound": False,
        "ambiguous": False,
        "business_ids": [],
        "terms": [],
        "row_indexes": [],
        "basis": "",
    }
    terms_by_tool = {
        tool: build_tool_terms(state, tool, terms)
        for tool in (
            "function_locator", "hmc_query", "fmeca_query", "fta_query",
            "rbd_query", "mfl_query", "fault_history_query",
        )
    }

    if "function_locator" in allowed_tools:
        started = time.time()
        function_errors: List[str] = []
        function_limit = max(limit, 12) if function_knowledge_query else limit
        try:
            function_rows = icp_repository.search_functions(
                terms_by_tool["function_locator"], limit=function_limit
            ) or []
        except Exception as exc:
            logger.warning("Function Locator query failed: %s", exc)
            function_errors.append(str(exc))
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
        # Component records live in a separate authoritative table. Query it
        # directly for an entity-profile request, or as a fallback when the
        # function table did not bind an object. Broad architecture queries
        # otherwise expand from their matched functions through explicit IDs.
        should_query_components = (
            "entity_profile" in target_intents
            or complete_collection_request
            or not function_rows
            # FMECA and RBD frequently store a module abbreviation/ID while
            # the user names a natural-language branch.  Read the authoritative
            # component identity table even when a broad function row happened
            # to match first, then let the conservative binder decide.
            or bool(allowed_tools & {"fmeca_query", "rbd_query"})
        )
        try:
            component_rows = (
                icp_repository.search_components(
                    terms_by_tool["function_locator"], limit=function_limit
                ) or []
            ) if should_query_components else []
            component_rows = [
                {**dict(row), "content_type": "function_component"}
                for row in component_rows
            ]
            if component_rows:
                # Direct component records are the most useful result for an
                # entity definition and remain useful seeds for later impact
                # or dependency expansion. De-duplicate by the explicit ID.
                merged_rows: List[Dict[str, Any]] = []
                seen_rows: set[Tuple[str, str]] = set()
                for row in [*component_rows, *function_rows]:
                    content_type = str(row.get("content_type") or "function_record")
                    business_id = str(
                        row.get("component_id")
                        or row.get("function_id")
                        or row.get("interface_id")
                        or row.get("id")
                        or ""
                    )
                    identity = (content_type, business_id)
                    if not business_id or identity in seen_rows:
                        continue
                    seen_rows.add(identity)
                    merged_rows.append(dict(row))
                function_rows = merged_rows
        except Exception as exc:
            logger.warning("Function component query failed: %s", exc)
            function_errors.append(f"组件查询失败: {exc}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
        if exact_record_identity_lookup:
            function_rows = _exact_identity_rows(
                function_rows,
                tool="function_locator",
                business_codes=explicit_business_codes,
            )
        if function_rows and not exact_record_identity_lookup:
            function_rows, structured_entity_binding = (
                _apply_structured_entity_binding(
                    state,
                    function_rows,
                    terms_by_tool,
                )
            )
        # 非架构诊断会在后面的 hybrid_search 工具统一召回，避免同一查询
        # 被 Function Locator 和混合检索各执行一次。架构路径会提前返回，
        # 因而在这里执行一次受限于 function_locator 的检索。
        if function_knowledge_only and "hybrid_search" in allowed_tools:
            raw_function_hybrid, function_retrieval_info = _hybrid_search(
                state,
                ("function_locator",),
                top_k=retrieval_top_k if architecture_query else min(limit, 8),
            )
            function_hybrid, function_scope_warnings = _filter_search_scope(
                raw_function_hybrid,
                ("function_locator",),
            )
            if function_scope_warnings:
                function_retrieval_info["scope_warnings"] = function_scope_warnings
            function_locators = _search_locator_handoffs(
                function_hybrid,
                limit=max(function_limit, 12),
                limit_per_type=max(function_limit, 12),
            )
            hydrated_functions = _hydrate_record_search_hits(
                function_locators,
                limit_per_type=max(function_limit, 12),
                errors={"function_locator": function_errors},
            )["function_locator"]
            existing_function_rows = {
                (
                    str(row.get("content_type") or "function_record"),
                    str(row.get("id") or ""),
                )
                for row in function_rows
                if row.get("id") is not None
            }
            hydrated_function_additions = [
                row
                for row in hydrated_functions
                if (
                    str(row.get("content_type") or "function_record"),
                    str(row.get("id") or ""),
                ) not in existing_function_rows
            ]
            # These rows were selected by semantic relevance and then
            # re-authorized by exact storage identity. Put them before broad
            # lexical PG matches so the bounded architecture context cannot
            # discard the useful records when the literal query filled its
            # result limit with generic modules.
            function_rows = [*hydrated_function_additions, *function_rows]
            if hydrated_function_additions:
                function_retrieval_info["postgres_hydrated"] = len(
                    hydrated_function_additions
                )
        elif "hybrid_search" not in allowed_tools:
            function_retrieval_info = {
                "skipped": "hybrid_search 不在执行白名单中，仅查询 PostgreSQL 功能事实"
            }

        # A semantic hit is commonly one leaf function.  Architecture/data-flow
        # answers also need its explicitly related parents, sibling functions,
        # implementing components and adjacent interfaces.  Expand only from a
        # few highest-priority PostgreSQL rows and let the repository follow
        # stored IDs; names and vector similarity never create relationships.
        if architecture_query and function_rows:
            architecture_anchors = function_rows[:4]
            try:
                expanded_function_rows = icp_repository.expand_function_context(
                    architecture_anchors,
                    limit=max(24, function_limit * 2),
                )
            except Exception as exc:
                logger.warning("Function Locator relationship expansion failed: %s", exc)
                function_errors.append(f"架构关系扩展失败: {exc}")
                try:
                    postgres_client.conn.rollback()
                except Exception:
                    pass
                expanded_function_rows = []
            if expanded_function_rows:
                merged_function_rows: List[Dict[str, Any]] = []
                seen_function_rows: set[Tuple[str, str]] = set()
                for row in [
                    *architecture_anchors,
                    *expanded_function_rows,
                    *function_rows[len(architecture_anchors):],
                ]:
                    content_type = str(row.get("content_type") or "function_record")
                    business_id = str(
                        row.get("function_id")
                        or row.get("component_id")
                        or row.get("interface_id")
                        or row.get("id")
                        or ""
                    )
                    identity = (content_type, business_id)
                    if not business_id or identity in seen_function_rows:
                        continue
                    seen_function_rows.add(identity)
                    merged_function_rows.append(dict(row))
                function_rows = merged_function_rows
                function_retrieval_info["postgres_relation_expanded"] = len(
                    expanded_function_rows
                )
        if (
            architecture_query
            and closed_inventory_requested(state)
            and function_rows
        ):
            try:
                inventory_rows = icp_repository.expand_architecture_inventory(
                    function_rows,
                    limit=16,
                )
            except Exception as exc:
                logger.warning("Whole-system architecture inventory expansion failed: %s", exc)
                function_errors.append(f"完整子系统清单查询失败: {exc}")
                try:
                    postgres_client.conn.rollback()
                except Exception:
                    pass
                inventory_rows = []
            if inventory_rows:
                # Closed-list rows lead the evidence stream so the system and
                # every direct subsystem are available before leaf functions.
                # Their metadata also lets the answer layer verify N-of-N
                # coverage after model editing and citation repair.
                merged_function_rows = []
                seen_inventory_rows: set[Tuple[str, str]] = set()
                for row in [*inventory_rows, *function_rows]:
                    content_type = str(row.get("content_type") or "function_record")
                    business_id = str(
                        row.get("function_id")
                        or row.get("component_id")
                        or row.get("interface_id")
                        or row.get("id")
                        or ""
                    )
                    identity = (content_type, business_id)
                    if not business_id or identity in seen_inventory_rows:
                        continue
                    seen_inventory_rows.add(identity)
                    merged_function_rows.append(dict(row))
                function_rows = merged_function_rows
                function_retrieval_info["postgres_inventory_count"] = len(
                    inventory_rows
                )
        function_rows = _enrich_function_sources(function_rows, function_errors)
        function_call = _tool_call(
            "function_locator",
            len(function_rows),
            started,
            _preview_rows(
                function_rows,
                ("function_id", "function_name"),
                ("function_name", "module_name", "description"),
            ),
            intent=intent,
        )
        function_call["postgres_authoritative_count"] = len(function_rows)
        if structured_entity_binding.get("bound"):
            function_call["structured_entity_binding"] = {
                "business_ids": list(
                    structured_entity_binding.get("business_ids") or []
                ),
                "basis": structured_entity_binding.get("basis") or "",
                "query_terms": list(
                    structured_entity_binding.get("terms") or []
                ),
            }
        tool_calls.append(function_call)
        function_evidence = _direct_evidence("function_locator", function_rows)
        verified_function_text = _verified_function_text_evidence(
            function_hybrid,
            limit=2,
            errors=function_errors,
        )
        function_search_evidence = _search_evidence(function_hybrid)
        evidence.extend(function_evidence)
        evidence.extend(verified_function_text)
        evidence.extend(function_search_evidence)
        function_search_errors, function_fallbacks = _retrieval_diagnostics(function_retrieval_info)
        if function_search_errors:
            # B1 降级计数：检索管线出错（含投影失败）时只进 stats，不静默。
            try:
                from backend.agent.tracing import record_degradation

                record_degradation(
                    "retrieval_projection",
                    detail="; ".join(function_search_errors)[:200],
                )
            except Exception:
                pass
        tool_results.append(
            make_tool_result(
                tool="function_locator",
                status=_tool_status(function_rows, function_errors),
                params={"terms": terms_by_tool["function_locator"], "limit": function_limit},
                data={"postgres_count": len(function_rows)},
                evidence_ids=[
                    item["evidence_id"]
                    for item in [*function_evidence, *verified_function_text]
                ],
                backend=("postgres+artifact_store" if verified_function_text else "postgres"),
                errors=function_errors,
            )
        )

    collection_rows: List[Dict[str, Any]] = []
    collection_errors: List[str] = []
    collection_resolution: Dict[str, Any] = {
        "requested": False,
        "closed": False,
        "member_kind": "",
        "subject": "",
        "reason": "",
    }
    # The planner may describe a system-level collection as “功能域” or
    # “模块”.  The authoritative inventory resolver is the source of truth
    # for the hierarchy level it can actually prove (for a system root, its
    # direct ``subsystem`` members).  Keep a repaired frame so the answer
    # layer cannot mix system, subsystem and leaf-function rows into one
    # falsely-counted list.
    architecture_collection_frame: Optional[Dict[str, Any]] = None
    collection_member_kind = str(
        (
            state.get("query_frame") or {}
        ).get("collection_member_kind") or ""
    ) if isinstance(state.get("query_frame"), Mapping) else ""
    collection_subject = str(
        (
            state.get("query_frame") or {}
        ).get("collection_subject") or ""
    ).strip() if isinstance(state.get("query_frame"), Mapping) else ""
    if complete_collection_request and collection_member_kind in {"failure_mode", "cause"}:
        collection_resolution.update(
            {
                "requested": True,
                "member_kind": "failure_mode",
                "requested_member_kind": collection_member_kind,
                "subject": collection_subject,
            }
        )
        try:
            scope_candidates = icp_repository.list_collection_scope_candidates() or []
        except Exception as exc:
            logger.warning("Complete-collection identity catalogue failed: %s", exc)
            scope_candidates = []
            collection_errors.append(f"完整集合实体目录查询失败: {exc}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
        component_ids, function_ids, seed_doc_id = _closed_collection_scope_seeds(
            scope_candidates,
            state,
        )
        if not seed_doc_id or not (component_ids or function_ids):
            collection_resolution["reason"] = (
                "未能将集合主体绑定到当前 Function Locator 的明确业务ID"
            )
        else:
            try:
                collection_rows = icp_repository.expand_fmeca_fault_mode_collection(
                    seed_doc_id,
                    component_ids=component_ids,
                    function_ids=function_ids,
                    # The answer evidence ledger is intentionally bounded to
                    # 25 records. A larger result must remain open rather than
                    # claiming completeness while silently hiding members.
                    limit=25,
                ) or []
            except Exception as exc:
                logger.warning("FMECA closed-collection expansion failed: %s", exc)
                direct_error = f"完整故障模式集合查询失败: {exc}"
                collection_errors.append(direct_error)
                collection_resolution["reason"] = direct_error
                try:
                    postgres_client.conn.rollback()
                except Exception:
                    pass
            if collection_rows:
                member_ids = [
                    str(row.get("collection_member_id") or "").strip()
                    for row in collection_rows
                ]
                parent_ids = {
                    str(row.get("collection_parent_id") or "").strip()
                    for row in collection_rows
                }
                member_kinds = {
                    str(row.get("collection_member_kind") or "").strip()
                    for row in collection_rows
                }
                declared_counts: set[int] = set()
                for row in collection_rows:
                    try:
                        declared_counts.add(int(row.get("collection_member_count") or 0))
                    except (TypeError, ValueError):
                        declared_counts.add(0)
                all_closed = bool(
                    all(
                        row.get("collection_closed") is True
                        and str(row.get("collection_role") or "") == "member"
                        for row in collection_rows
                    )
                    and all(member_ids)
                    and len(set(member_ids)) == len(member_ids)
                    and len(parent_ids) == 1
                    and "" not in parent_ids
                    and member_kinds == {"failure_mode"}
                    and declared_counts == {len(collection_rows)}
                )
                missing_cause_mode_ids = [
                    member_id
                    for member_id, row in zip(member_ids, collection_rows)
                    if not str(row.get("fault_cause") or "").strip()
                ]
                cause_coverage_complete = not missing_cause_mode_ids
                collection_resolution["cause_present_count"] = (
                    len(collection_rows) - len(missing_cause_mode_ids)
                )
                collection_resolution["missing_cause_mode_ids"] = missing_cause_mode_ids
                collection_resolution["cause_coverage_complete"] = cause_coverage_complete
                effective_closed = bool(
                    all_closed
                    and (
                        collection_member_kind != "cause"
                        or cause_coverage_complete
                    )
                )
                collection_resolution["closed"] = effective_closed
                declared_total = max(declared_counts) if declared_counts else 0
                collection_resolution["member_count"] = declared_total or len(collection_rows)
                collection_resolution["parent_id"] = str(
                    collection_rows[0].get("collection_parent_id") or ""
                )
                collection_resolution["parent_name"] = str(
                    collection_rows[0].get("collection_parent_name") or ""
                )
                if not all_closed:
                    collection_resolution["reason"] = "权威查询返回的集合未通过闭合校验"
                elif not effective_closed:
                    collection_resolution["reason"] = (
                        "故障模式清单完整，但部分记录未填写故障原因，"
                        "不能声明原因映射完整"
                    )
            elif not collection_resolution["reason"]:
                collection_resolution["reason"] = "权威结构化资料未返回可闭合的集合"

    if architecture_query and complete_collection_request:
        inventory_members = [
            row
            for row in function_rows
            if row.get("collection_closed") is True
            and str(row.get("collection_role") or "") == "member"
            and str(row.get("collection_member_kind") or "").strip()
        ]
        if inventory_members:
            first_inventory = inventory_members[0]
            proven_kind = str(
                first_inventory.get("collection_member_kind") or ""
            ).strip()
            parent_id = str(
                first_inventory.get("collection_parent_id") or ""
            ).strip()
            parent_name = str(
                first_inventory.get("collection_parent_name") or ""
            ).strip()
            declared_counts = {
                int(row.get("collection_member_count") or 0)
                for row in inventory_members
                if str(row.get("collection_member_count") or "").strip()
            }
            declared_count = max(declared_counts) if declared_counts else 0
            collection_resolution.update(
                {
                    "requested": True,
                    "closed": True,
                    "member_kind": proven_kind,
                    "requested_member_kind": collection_member_kind,
                    "subject": collection_subject,
                    "parent_id": parent_id,
                    "parent_name": parent_name,
                    "member_count": declared_count or len(inventory_members),
                }
            )
            architecture_collection_frame = dict(
                state.get("query_frame")
                if isinstance(state.get("query_frame"), Mapping)
                else {}
            )
            architecture_collection_frame.update(
                {
                    "collection_subject": parent_id or parent_name,
                    "collection_member_kind": proven_kind,
                }
            )
            required_fields = list(
                architecture_collection_frame.get("required_answer_fields") or []
            )
            if "closed_collection_members" not in required_fields:
                required_fields.append("closed_collection_members")
            architecture_collection_frame["required_answer_fields"] = required_fields

    if function_knowledge_only:
        # PostgreSQL owns Function Locator facts. The following graph node may
        # expand their stable IDs, but every edge is re-authorized against PG;
        # HugeGraph itself never becomes a second authority source.
        if function_call is not None:
            function_call["graph_handoff"] = {
                "enabled": "graph_traversal" in allowed_tools,
                "basis": "仅把 PostgreSQL 返回的明确业务ID交给关系查询",
            }
        domain_results = _architecture_domain_results(function_rows, function_hybrid)
        context = _format_architecture_context(query, domain_results)
        if "hybrid_search" in allowed_tools:
            hybrid_call = _tool_call(
                "hybrid_search",
                len(function_hybrid),
                time.time(),
                _preview_rows(function_hybrid, ("title",), ("content",)),
                doc_types=["function_locator"],
                top_k=retrieval_top_k,
            )
            hybrid_call["retrieval_info"] = function_retrieval_info
            tool_calls.append(hybrid_call)
            tool_results.append(
                make_tool_result(
                    tool="hybrid_search",
                    status=_tool_status(
                        function_hybrid,
                        function_search_errors,
                        partial=bool(function_fallbacks or function_scope_warnings),
                    ),
                    params={"doc_types": ["function_locator"], "top_k": retrieval_top_k},
                    data={"count": len(function_hybrid)},
                    evidence_ids=[item["evidence_id"] for item in _search_evidence(function_hybrid)],
                    backend="opensearch",
                    fallbacks=function_fallbacks,
                    warnings=function_scope_warnings,
                    errors=function_search_errors,
                )
            )
        _attach_tool_call_evidence(tool_calls, tool_results, evidence)
        seed_bindings = _graph_seed_bindings(function_rows=function_rows)
        return {
            "diagnostic_context": context,
            # 架构知识问答不制造“候选故障/诊断置信度/故障追问”。
            "candidate_faults": [],
            "diagnostic_support": 0.0,
            "diagnostic_confidence": 0.0,
            "clarification_question": "",
            "domain_results": domain_results,
            "vector_results": domain_results,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "evidence_registry": evidence,
            "search_locator_hits": [],
            "graph_seed_bindings": seed_bindings,
            "graph_traversal_request": _graph_request(seed_bindings, allowed_tools),
            "collection_resolution": collection_resolution,
            **(
                {"query_frame": architecture_collection_frame}
                if architecture_collection_frame is not None
                else {}
            ),
        }

    direct_rows: Dict[str, List[Dict[str, Any]]] = {
        "hmc_query": [], "fmeca_query": [], "fta_query": [], "rbd_query": [],
        "mfl_query": [], "fault_history_query": [],
    }
    direct_errors: Dict[str, List[str]] = {name: [] for name in direct_rows}
    direct_errors["fmeca_query"].extend(collection_errors)

    def run_direct(tool: str, call, *, match_query: Optional[str] = None) -> None:
        started = time.time()
        try:
            direct_rows[tool] = list(call())
            if exact_record_identity_lookup and tool in _IDENTITY_FIELDS_BY_TOOL:
                direct_rows[tool] = _exact_identity_rows(
                    direct_rows[tool],
                    tool=tool,
                    business_codes=explicit_business_codes,
                )
        except Exception as exc:
            direct_errors[tool].append(str(exc))
            logger.warning("%s failed: %s", tool, exc)
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
        rows = direct_rows[tool]
        preview_keys = {
            "hmc_query": (("hmc_code", "hsc"), ("fault_desc", "repair_method")),
            "fmeca_query": (("code", "fault_mode"), ("fault_mode", "fault_cause", "final_impact")),
            "fta_query": (("title", "image_title", "diagram_id"), ("title", "image_title")),
            "rbd_query": (("title", "image_title", "diagram_id"), ("title", "image_title")),
            "mfl_query": (("mfl_code", "api_name"), ("module_fault", "subcircuit_fault", "api_usage")),
            "fault_history_query": (("case_id",), ("fault_symptom", "root_cause", "final_solution")),
        }[tool]
        params = {"terms": terms_by_tool[tool], "limit": limit}
        if tool == "hmc_query":
            parsed_hmc = _extract_hmc(match_query or raw_user_query)
            params["match_type"] = (
                "exact_code"
                if parsed_hmc.get("hamc")
                else ("exact_components" if parsed_hmc else "keyword")
            )
        tool_calls.append(_tool_call(tool, len(rows), started, _preview_rows(rows, *preview_keys), **params))

    if "hmc_query" in allowed_tools:
        conversation_hmc_queries = _conversation_hmc_queries(state)
        run_direct(
            "hmc_query",
            (
                lambda: _query_hmc_from_conversation_anchors(
                    conversation_hmc_queries,
                    terms_by_tool["hmc_query"],
                    limit,
                    direct_errors["hmc_query"],
                    include_repair_method=answer_mode == "maintenance_guidance",
                )
                if conversation_hmc_queries
                else _query_hmc(
                    raw_user_query,
                    terms_by_tool["hmc_query"],
                    limit,
                    direct_errors["hmc_query"],
                    include_repair_method=answer_mode == "maintenance_guidance",
                )
            ),
            match_query=conversation_hmc_queries[0] if conversation_hmc_queries else None,
        )
        if conversation_hmc_queries and tool_calls:
            tool_calls[-1]["conversation_anchor_codes"] = conversation_hmc_queries
    if "fmeca_query" in allowed_tools:
        conversation_fmeca_codes = _conversation_fmeca_codes(state)
        if collection_resolution.get("requested") and collection_rows:
            # A prior FMECA citation can seed the collection subject, but is
            # never itself treated as the complete set. Only the repository's
            # explicit-ID expansion may mark the response closed.
            fmeca_call = lambda: collection_rows
        elif conversation_fmeca_codes:
            fmeca_call = lambda: _query_fmeca_by_codes(
                conversation_fmeca_codes,
                max(limit, len(conversation_fmeca_codes)),
                direct_errors["fmeca_query"],
                match_type="exact_conversation_reference",
            )
        else:
            fmeca_call = lambda: _query_fmeca(
                terms_by_tool["fmeca_query"],
                limit,
                direct_errors["fmeca_query"],
                entity_terms=structured_entity_binding.get("terms") or [],
            )
        run_direct("fmeca_query", fmeca_call)
        if collection_resolution.get("requested") and tool_calls:
            tool_calls[-1]["collection_resolution"] = dict(collection_resolution)
        if conversation_fmeca_codes and tool_calls:
            tool_calls[-1]["conversation_anchor_codes"] = conversation_fmeca_codes
    fmeca_rbd_identity_terms: List[str] = []
    runtime_topology_dependency = state.get("topology_dependency")
    frame_topology_dependency = (
        (state.get("query_frame") or {}).get("topology_dependency")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    topology_dependency = (
        runtime_topology_dependency
        if isinstance(runtime_topology_dependency, Mapping)
        and runtime_topology_dependency.get("required") is True
        else frame_topology_dependency
    )
    if (
        "rbd_query" in allowed_tools
        and isinstance(topology_dependency, Mapping)
        and topology_dependency.get("required") is True
    ):
        fmeca_rbd_identity_terms = _shared_fmeca_identity_terms(
            direct_rows["fmeca_query"],
            structured_business_ids=[
                *(structured_entity_binding.get("business_ids") or []),
                *extract_business_codes(
                    structured_entity_binding.get("terms") or []
                ),
                *(topology_dependency.get("canonical_business_ids") or []),
            ],
        )
        if fmeca_rbd_identity_terms:
            terms_by_tool["rbd_query"] = _prepend_query_terms(
                terms_by_tool["rbd_query"], fmeca_rbd_identity_terms
            )
    if "fta_query" in allowed_tools:
        run_direct(
            "fta_query",
            lambda: _query_diagrams(
                "fta",
                terms_by_tool["fta_query"],
                max(4, limit // 2),
                direct_errors["fta_query"],
            ),
        )
    if "rbd_query" in allowed_tools:
        requested_source_tools = {
            str(value or "").strip()
            for value in (
                state.get("requested_source_tools")
                or (state.get("query_frame") or {}).get("requested_source_tools")
                or []
            )
            if str(value or "").strip()
        }

        def rbd_call():
            rows = _query_diagrams(
                "rbd",
                terms_by_tool["rbd_query"],
                max(4, limit // 2),
                direct_errors["rbd_query"],
                entity_terms=_prepend_query_terms(
                    structured_entity_binding.get("terms") or [],
                    fmeca_rbd_identity_terms,
                ),
            )
            # If the user expressly asked us to consult RBD and Function
            # Locator produced a durable object ID, a broad RBD hit is not
            # enough. Keep only graphs that persist the same ID on a block.
            # Returning no row is preferable to answering with a different
            # system model and calling it evidence for the requested object.
            if (
                "rbd_query" in requested_source_tools
                and structured_entity_binding.get("bound")
                and structured_entity_binding.get("business_ids")
            ):
                return _filter_diagrams_by_explicit_object_ids(
                    rows,
                    structured_entity_binding.get("business_ids") or [],
                )
            return rows

        run_direct(
            "rbd_query",
            rbd_call,
        )
        if fmeca_rbd_identity_terms and tool_calls:
            tool_calls[-1]["structured_fmeca_handoff"] = {
                "basis": "多条 PostgreSQL FMECA 记录共享的显式产品/功能代码",
                "terms": fmeca_rbd_identity_terms,
            }
    if "mfl_query" in allowed_tools:
        run_direct(
            "mfl_query",
            lambda: _query_mfl(
                terms_by_tool["mfl_query"], limit, direct_errors["mfl_query"]
            ),
        )
    if "fault_history_query" in allowed_tools:
        run_direct(
            "fault_history_query",
            lambda: icp_repository.search_history(
                terms_by_tool["fault_history_query"], limit=limit
            ),
        )

    hybrid_rows: List[Dict[str, Any]] = []
    hybrid_retrieval_info: Dict[str, Any] = {}
    hybrid_call: Optional[Dict[str, Any]] = None
    if "hybrid_search" in allowed_tools:
        requested_doc_types = list(state.get("hybrid_doc_types") or policy_hybrid_doc_types)
        if not requested_doc_types:
            tool_to_type = {
                "function_locator": "function_locator", "hmc_query": "hmc", "fmeca_query": "fmeca",
                "fta_query": "fta", "rbd_query": "rbd", "mfl_query": "mfl",
                "fault_history_query": "fault_history",
            }
            requested_doc_types = [
                doc_type for tool, doc_type in tool_to_type.items() if tool in allowed_tools
            ]
        hybrid_scope_warnings = [
            f"忽略策略范围外文档类型: {value}"
            for value in requested_doc_types
            if str(value or "").strip() not in ICP_DOC_TYPES
        ]
        hybrid_doc_types = list(
            dict.fromkeys(
                str(value).strip()
                for value in requested_doc_types
                if str(value or "").strip() in ICP_DOC_TYPES
            )
        )
        started = time.time()
        if hybrid_doc_types:
            raw_hybrid_rows, hybrid_retrieval_info = _hybrid_search(
                state, hybrid_doc_types, top_k=retrieval_top_k
            )
            hybrid_rows, returned_scope_warnings = _filter_search_scope(
                raw_hybrid_rows,
                hybrid_doc_types,
            )
            hybrid_scope_warnings.extend(returned_scope_warnings)
        else:
            hybrid_retrieval_info = {
                "fallback": "策略未允许任何七类文档范围，未执行 OpenSearch 混合召回"
            }
        if hybrid_scope_warnings:
            hybrid_retrieval_info["scope_warnings"] = hybrid_scope_warnings
        hybrid_call = _tool_call(
            "hybrid_search",
            len(hybrid_rows),
            started,
            _preview_rows(hybrid_rows, ("title",), ("content",)),
            doc_types=hybrid_doc_types,
            top_k=retrieval_top_k,
        )
        hybrid_call["retrieval_info"] = hybrid_retrieval_info
        tool_calls.append(hybrid_call)
        hybrid_evidence = _search_evidence(hybrid_rows)
        hybrid_errors, hybrid_fallbacks = _retrieval_diagnostics(hybrid_retrieval_info)
        verified_hybrid_text = _verified_function_text_evidence(
            hybrid_rows,
            limit=2,
            errors=hybrid_errors,
        )
        evidence.extend(verified_hybrid_text)
        evidence.extend(hybrid_evidence)
        tool_results.append(
            make_tool_result(
                tool="hybrid_search",
                status=(
                    "blocked"
                    if not hybrid_doc_types
                    else _tool_status(
                        hybrid_rows,
                        hybrid_errors,
                        partial=bool(hybrid_fallbacks or hybrid_scope_warnings),
                    )
                ),
                params={"doc_types": hybrid_doc_types, "top_k": retrieval_top_k},
                data={"count": len(hybrid_rows), "retrieval_info": hybrid_retrieval_info},
                evidence_ids=[
                    item["evidence_id"]
                    for item in [*verified_hybrid_text, *hybrid_evidence]
                ],
                backend=("opensearch+artifact_store" if verified_hybrid_text else "opensearch"),
                fallbacks=hybrid_fallbacks,
                warnings=hybrid_scope_warnings,
                errors=hybrid_errors,
            )
        )

    # Semantic search discovers the relevant graph; PostgreSQL re-authorizes
    # its current active revision. Search status fields are never trusted.
    # A comparison request may name four or more diagrams in one document
    # (for example four RBD chains).  Keep the same per-type budget as the
    # PostgreSQL hydration step; the old default of three silently discarded
    # the fourth exact locator before deterministic calculation.
    # Keep a sufficient semantic locator window for a multi-branch question.
    # The previous ``limit // 2`` default meant an ordinary SQL limit of 8
    # handed only four locators to PostgreSQL and a later fixed Top-2 hydration
    # could erase one entire branch.  All later presentation and answer gates
    # remain bounded independently.
    locator_limit_per_type = max(6, min(limit, 8))
    locator_handoffs = _search_locator_handoffs(
        [*(state.get("search_locator_hits") or []), *hybrid_rows],
        limit=max(24, locator_limit_per_type * len(ICP_DOC_TYPES)),
        limit_per_type=locator_limit_per_type,
    )
    hydrated_graphs = _hydrate_diagram_search_hits(
        locator_handoffs,
        allowed_tools=allowed_tools,
        limit_per_type=locator_limit_per_type,
        errors=direct_errors,
    )
    graph_preview_keys = {
        "fta_query": (("title", "image_title", "diagram_id"), ("title", "image_title")),
        "rbd_query": (("title", "image_title", "diagram_id"), ("title", "image_title")),
    }
    for tool, hydrated_rows in hydrated_graphs.items():
        existing_ids = {
            str(row.get("revision_id") or "")
            for row in direct_rows[tool]
            if row.get("revision_id")
        }
        additions = [
            row
            for row in hydrated_rows
            if str(row.get("revision_id") or "") not in existing_ids
        ]
        if not additions:
            continue
        direct_rows[tool].extend(additions)
        for call in tool_calls:
            if call.get("tool") != tool:
                continue
            call["count"] = len(direct_rows[tool])
            call["semantic_hydrated_count"] = len(additions)
            call["results_preview"] = _preview_rows(
                direct_rows[tool],
                *graph_preview_keys[tool],
            )
            break

    # The strongest bounded row hits in each structured family are re-read by
    # the exact PostgreSQL storage identity carried by the index.  Paired
    # HMC/FMECA/MFL modes are retained for explicit comparisons; history stays
    # at one row.  This validates semantic recall without fuzzy entity merging
    # or trusting indexed payload text.
    hydrated_records = _hydrate_record_search_hits(
        locator_handoffs,
        limit_per_type=1,
        limits_by_type=_DIAGNOSTIC_RECORD_HYDRATION_LIMITS,
        errors=direct_errors,
    )
    hydrated_record_counts: Dict[str, int] = {}
    hydrated_function_rows = hydrated_records.pop("function_locator", [])
    existing_function_ids = {
        (
            str(row.get("content_type") or "function_record"),
            str(row.get("id") or ""),
        )
        for row in function_rows
        if row.get("id") is not None
    }
    function_additions = [
        row
        for row in hydrated_function_rows
        if (
            str(row.get("content_type") or "function_record"),
            str(row.get("id") or ""),
        ) not in existing_function_ids
    ]
    if function_additions:
        function_additions = _enrich_function_sources(
            function_additions,
            direct_errors.get("function_locator"),
        )
        function_rows.extend(function_additions)
        function_evidence = _direct_evidence("function_locator", function_additions)
        evidence.extend(function_evidence)
        hydrated_record_counts["function_locator"] = len(function_additions)
        if function_call is not None:
            function_call["count"] = len(function_rows)
            function_call["semantic_hydrated_count"] = len(function_additions)
        for result in tool_results:
            if result.get("tool") != "function_locator":
                continue
            result["data"] = {
                **(result.get("data") or {}),
                "postgres_count": len(function_rows),
                "semantic_hydrated_count": len(function_additions),
            }
            result["evidence_ids"] = list(
                dict.fromkeys(
                    [*(result.get("evidence_ids") or []), *[item["evidence_id"] for item in function_evidence]]
                )
            )
            break
    if (
        function_rows
        and not structured_entity_binding.get("bound")
        and isinstance(topology_dependency, Mapping)
        and topology_dependency.get("required") is True
    ):
        # The broad literal Function Locator query may miss the object while
        # hybrid retrieval later discovers one row.  That hit has now been
        # re-read by its PostgreSQL storage identity, so it is safe to perform
        # the same conservative structured binding after hydration.  This
        # enriches only the durable answer contract; it neither reruns an RBD
        # query nor expands the planner's tool permissions.
        post_hydration_binding = bind_structured_function_entity(
            [
                topology_dependency.get("subject") or "",
                (state.get("query_frame") or {}).get("entities")
                if isinstance(state.get("query_frame"), Mapping)
                else [],
                (state.get("query_frame") or {}).get("normalized_terms")
                if isinstance(state.get("query_frame"), Mapping)
                else [],
                state.get("rewritten_query") or "",
                state.get("query") or "",
            ],
            function_rows,
        )
        if (
            post_hydration_binding.get("bound")
            and _binding_can_drive_cross_document_expansion(
                state, post_hydration_binding
            )
        ):
            structured_entity_binding = post_hydration_binding
            binding_summary = {
                "business_ids": list(
                    structured_entity_binding.get("business_ids") or []
                ),
                "basis": structured_entity_binding.get("basis") or "",
                "query_terms": list(
                    structured_entity_binding.get("terms") or []
                ),
                "stage": "postgres_hydration",
            }
            if hybrid_call is not None:
                hybrid_call["structured_entity_binding"] = binding_summary
            elif function_call is not None:
                function_call["structured_entity_binding"] = binding_summary
            if (
                "fmeca_query" in (allowed_tools | structured_hydration_tools)
                and not (
                    collection_resolution.get("requested") and collection_rows
                )
            ):
                exact_fmeca_rows = _query_fmeca_by_business_ids(
                    structured_entity_binding.get("business_ids") or [],
                    max(limit, 8),
                    direct_errors["fmeca_query"],
                )
                existing_fmeca_ids = {
                    str(row.get("id") or row.get("code") or "")
                    for row in direct_rows["fmeca_query"]
                }
                exact_fmeca_additions = [
                    row
                    for row in exact_fmeca_rows
                    if str(row.get("id") or row.get("code") or "")
                    not in existing_fmeca_ids
                ]
                if exact_fmeca_additions:
                    direct_rows["fmeca_query"].extend(exact_fmeca_additions)
                    if hybrid_call is not None:
                        hybrid_call["structured_identity_hydrated"] = {
                            "tool": "fmeca_query",
                            "count": len(exact_fmeca_additions),
                            "business_ids": list(
                                structured_entity_binding.get("business_ids") or []
                            ),
                        }
                    for call in tool_calls:
                        if call.get("tool") != "fmeca_query":
                            continue
                        call["count"] = len(direct_rows["fmeca_query"])
                        call["structured_identity_hydrated_count"] = len(
                            exact_fmeca_additions
                        )
                        call["structured_identity_terms"] = list(
                            structured_entity_binding.get("business_ids") or []
                        )
                        call["results_preview"] = _preview_rows(
                            direct_rows["fmeca_query"],
                            ("code", "fault_mode"),
                            ("fault_mode", "fault_cause", "final_impact"),
                        )
                        break
    for tool, hydrated_rows in hydrated_records.items():
        if (
            collection_resolution.get("requested")
            and collection_rows
            and tool == "fmeca_query"
        ):
            # Semantic Top-K may help discover a scope, but it cannot add
            # members to a PostgreSQL-closed collection.
            continue
        existing_by_id = {
            str(row.get("id") or ""): row
            for row in direct_rows[tool]
            if row.get("id") is not None
        }
        additions: List[Dict[str, Any]] = []
        semantic_merges = 0
        for row in hydrated_rows:
            existing = existing_by_id.get(str(row.get("id") or ""))
            if existing is None:
                additions.append(row)
                continue
            # A literal PG lookup and semantic-id hydration can resolve to the
            # same history row.  Preserve the semantic handoff on the existing
            # object instead of dropping it during deduplication; otherwise its
            # explicit business-code references would never be authorized for
            # bounded cross-document hydration.
            if tool == "fault_history_query" and row.get("match_type") == "semantic_id_hydration":
                existing["match_type"] = "semantic_id_hydration"
                existing["retrieval_match_score"] = row.get("retrieval_match_score")
                existing["retrieval_source_chunk_id"] = row.get(
                    "retrieval_source_chunk_id"
                )
                semantic_merges += 1
        if not additions and not semantic_merges:
            continue
        direct_rows[tool].extend(additions)
        hydrated_record_counts[tool] = len(additions) + semantic_merges
        for call in tool_calls:
            if call.get("tool") != tool:
                continue
            call["count"] = len(direct_rows[tool])
            call["semantic_hydrated_count"] = (
                int(call.get("semantic_hydrated_count") or 0)
                + len(additions)
                + semantic_merges
            )
            break
    if hybrid_call is not None and hydrated_record_counts:
        hybrid_call["postgres_hydrated"] = dict(hydrated_record_counts)

    # A semantically recalled history row may carry explicit HMC/FMECA/MFL
    # business IDs.  The row above has already been re-read by its exact
    # PostgreSQL storage ID, so those stored references can safely authorize a
    # bounded cross-document lookup.  This is intentionally not fuzzy entity
    # resolution: only exact codes from ``semantic_id_hydration`` rows are
    # used, and every linked result is read back from an active PG document.
    history_reference_context = _semantic_history_reference_context(
        direct_rows["fault_history_query"]
    )
    history_linked_counts: Dict[str, int] = {}
    history_linked_specs = (
        ("hmc_query", "hmc", "hmc_code"),
        ("fmeca_query", "fmeca", "code"),
        ("mfl_query", "mfl", "mfl_code"),
    )
    for tool, ref_type, code_column in history_linked_specs:
        if (
            collection_resolution.get("requested")
            and collection_rows
            and tool == "fmeca_query"
        ):
            continue
        reference_context = history_reference_context.get(ref_type) or {}
        if tool not in allowed_tools or not reference_context:
            continue
        if ref_type == "fmeca":
            linked_rows = _query_fmeca_by_codes(
                list(reference_context),
                max(limit * 4, 32),
                direct_errors[tool],
                match_type="exact_history_reference",
                reference_context=reference_context,
            )
        else:
            linked_rows = _query_history_linked_rows(
                ref_type,
                reference_context,
                max(limit * 4, 32),
                direct_errors[tool],
            )
        existing_by_id = {
            str(row.get("id")): row
            for row in direct_rows[tool]
            if row.get("id") is not None
        }
        existing_by_fallback = {
            (str(row.get("doc_id") or ""), str(row.get(code_column) or "")): row
            for row in direct_rows[tool]
        }
        linked_additions: List[Dict[str, Any]] = []
        linked_matches = 0
        for linked_row in linked_rows:
            existing = None
            if linked_row.get("id") is not None:
                existing = existing_by_id.get(str(linked_row.get("id")))
            if existing is None:
                existing = existing_by_fallback.get(
                    (
                        str(linked_row.get("doc_id") or ""),
                        str(linked_row.get(code_column) or ""),
                    )
                )
            if existing is None:
                linked_additions.append(linked_row)
                continue
            linked_matches += 1
            if _retrieval_score(linked_row.get("retrieval_match_score")) > _retrieval_score(
                existing.get("retrieval_match_score")
            ):
                existing["retrieval_match_score"] = linked_row.get(
                    "retrieval_match_score"
                )
            existing["history_reference_match_type"] = "exact_history_reference"
            existing["reference_source"] = "fault_history"
            existing["linked_history_record_ids"] = list(
                linked_row.get("linked_history_record_ids") or []
            )
            existing["linked_history_case_ids"] = list(
                linked_row.get("linked_history_case_ids") or []
            )
        if not linked_additions and not linked_matches:
            continue
        direct_rows[tool].extend(linked_additions)
        # Preserve deterministic stable ordering while ensuring that a record
        # linked by the strongest relevant history hit is not hidden behind an
        # unrelated weaker semantic row.
        direct_rows[tool].sort(
            key=lambda row: _retrieval_score(row.get("retrieval_match_score")),
            reverse=True,
        )
        history_linked_counts[tool] = len(linked_additions) + linked_matches
        for call in tool_calls:
            if call.get("tool") != tool:
                continue
            call["count"] = len(direct_rows[tool])
            call["history_linked_count"] = len(linked_additions) + linked_matches
            call["history_linked_codes"] = list(reference_context)
            preview_keys = {
                "hmc_query": (("hmc_code", "hsc"), ("fault_desc", "repair_method")),
                "fmeca_query": (("code", "fault_mode"), ("fault_mode", "fault_cause", "final_impact")),
                "mfl_query": (("mfl_code", "api_name"), ("module_fault", "subcircuit_fault", "api_usage")),
            }[tool]
            call["results_preview"] = _preview_rows(direct_rows[tool], *preview_keys)
            break
    if hybrid_call is not None and history_linked_counts:
        hybrid_call["history_reference_hydrated"] = dict(history_linked_counts)

    # A formally validated FTA carries explicit failure-mode IDs.  Once a
    # semantic locator has been re-hydrated, use those IDs to fetch FMECA truth
    # rows exactly.  Re-running a broad lexical FMECA query would throw away the
    # very cross-document identity that the FTA supplied.
    if (
        "fmeca_query" in allowed_tools
        and direct_rows["fta_query"]
        and not (
            collection_resolution.get("requested") and collection_rows
        )
    ):
        linked_codes = _explicit_failure_mode_ids(direct_rows["fta_query"])
        linked_rows = _query_fmeca_by_codes(
            linked_codes,
            max(limit * 4, 32),
            direct_errors["fmeca_query"],
        )
        existing_codes = {
            str(row.get("code") or "") for row in direct_rows["fmeca_query"]
        }
        linked_additions = [
            row for row in linked_rows if str(row.get("code") or "") not in existing_codes
        ]
        if linked_additions:
            direct_rows["fmeca_query"].extend(linked_additions)
            for call in tool_calls:
                if call.get("tool") != "fmeca_query":
                    continue
                call["count"] = len(direct_rows["fmeca_query"])
                call["fta_linked_count"] = len(linked_additions)
                call["results_preview"] = _preview_rows(
                    direct_rows["fmeca_query"],
                    ("code", "fault_mode"),
                    ("fault_mode", "fault_cause", "final_impact"),
                )
                break

    # 首轮结构化图查询可能因纯词法错配而空手而归：用户说“B路稳压偏低”，而
    # 权威行存储的是“B路稳压供电/B路输出欠压”，SQL 词法召回无法命中。语义
    # 召回后按存储身份水合出的显式业务 ID（绑定对象 FUNC-121/MOD-PWR-PSUB，
    # 或 FMECA 回读后共享的 PSU-B/FUNC-121 集群）现在可以授权一次有界的
    # 图查询重试。这仍是显式代码驱动的查询补全，不是名称相似度合并；每条
    # 结果都来自 PostgreSQL 当前活动版本。
    if (
        isinstance(topology_dependency, Mapping)
        and topology_dependency.get("required") is True
    ):
        post_hydration_identity_terms = _shared_fmeca_identity_terms(
            direct_rows["fmeca_query"],
            structured_business_ids=[
                *(structured_entity_binding.get("business_ids") or []),
                *(topology_dependency.get("canonical_business_ids") or []),
            ],
        )
        if not post_hydration_identity_terms:
            post_hydration_identity_terms = [
                str(value or "").strip()
                for value in (structured_entity_binding.get("terms") or [])
                if str(value or "").strip()
            ] or list(fmeca_rbd_identity_terms)
        for diagram_tool, diagram_type_name in (
            ("rbd_query", "rbd"),
            ("fta_query", "fta"),
        ):
            if (
                diagram_tool not in (allowed_tools | structured_hydration_tools)
                or direct_rows[diagram_tool]
                or not post_hydration_identity_terms
            ):
                continue
            # 自适应补查批次里该工具没有在本批执行；若首轮已拿到同类型图，
            # 不重复查询，避免同一 revision 生成两份证据。
            already_confirmed = any(
                str(row.get("diagram_type") or "") == diagram_type_name
                for row in (state.get("confirmed_graphs") or [])
                if isinstance(row, Mapping)
            )
            if already_confirmed:
                continue
            retry_terms = _prepend_query_terms(
                terms_by_tool[diagram_tool],
                post_hydration_identity_terms,
            )
            backfilled_rows = _query_diagrams(
                diagram_type_name,
                retry_terms,
                max(4, limit // 2),
                direct_errors[diagram_tool],
                entity_terms=post_hydration_identity_terms,
            )
            if not backfilled_rows:
                continue
            direct_rows[diagram_tool].extend(backfilled_rows)
            fmeca_rbd_identity_terms = (
                fmeca_rbd_identity_terms or list(post_hydration_identity_terms)
            )
            retry_note = {
                "terms": list(post_hydration_identity_terms),
                "count": len(backfilled_rows),
            }
            updated_call = False
            for call in tool_calls:
                if call.get("tool") != diagram_tool:
                    continue
                call["count"] = len(direct_rows[diagram_tool])
                call["structured_identity_retry"] = dict(retry_note)
                call["results_preview"] = _preview_rows(
                    direct_rows[diagram_tool],
                    ("title", "image_title", "diagram_id"),
                    ("title", "image_title"),
                )
                updated_call = True
                break
            if not updated_call:
                tool_calls.append(
                    _tool_call(
                        diagram_tool,
                        len(direct_rows[diagram_tool]),
                        time.time(),
                        _preview_rows(
                            direct_rows[diagram_tool],
                            ("title", "image_title", "diagram_id"),
                            ("title", "image_title"),
                        ),
                        terms=list(retry_terms),
                        limit=limit,
                        structured_identity_retry=dict(retry_note),
                    )
                )

    # Attach applicability/version and produce evidence only after semantic
    # locators have been hydrated into PostgreSQL truth rows.
    for tool, rows in list(direct_rows.items()):
        direct_rows[tool] = _enrich_function_sources(rows, direct_errors[tool])
        current_evidence = _direct_evidence(tool, direct_rows[tool])
        evidence.extend(current_evidence)
        if tool in allowed_tools:
            semantic_hydrated_count = sum(
                row.get("match_type")
                in {"semantic_locator_hydration", "semantic_id_hydration"}
                for row in direct_rows[tool]
            )
            tool_results.append(
                make_tool_result(
                    tool=tool,
                    status=_tool_status(direct_rows[tool], direct_errors[tool]),
                    params={"terms": terms_by_tool[tool], "limit": limit},
                    data={
                        "count": len(direct_rows[tool]),
                        "semantic_hydrated_count": semantic_hydrated_count,
                    },
                    evidence_ids=[item["evidence_id"] for item in current_evidence],
                    backend="postgres",
                    errors=direct_errors[tool],
                )
            )

    hmc_rows = direct_rows["hmc_query"]
    fmeca_rows = direct_rows["fmeca_query"]
    fta_rows = direct_rows["fta_query"]
    rbd_rows = direct_rows["rbd_query"]
    mfl_rows = direct_rows["mfl_query"]
    history_rows = direct_rows["fault_history_query"]
    history_hybrid = [
        row for row in hybrid_rows
        if _search_doc_type(row) == "fault_history"
    ]

    candidates = _candidate_faults(hmc_rows, fmeca_rows, hybrid_rows, evidence)
    exact_hmc = any(row.get("match_type") == "exact_code" for row in hmc_rows)
    component_hmc = any(row.get("match_type") == "exact_components" for row in hmc_rows)
    source_count = sum(
        bool(rows)
        for rows in (
            function_rows or function_hybrid,
            hmc_rows,
            fmeca_rows,
            fta_rows,
            rbd_rows,
            mfl_rows,
            history_rows or history_hybrid,
        )
    )
    if exact_hmc:
        support = 0.95
    elif component_hmc:
        # LCN+HSC is an authoritative structured match, but it is weaker than
        # a complete triplet and may legitimately return multiple HAMC rows.
        support = 0.84 if len(hmc_rows) == 1 else 0.72
    elif candidates and source_count >= 4:
        support = 0.82
    elif candidates and source_count >= 2:
        support = 0.68
    elif hybrid_rows:
        support = 0.48
    else:
        support = 0.2

    clarification = ""
    # Candidate narrowing questions belong only to actual diagnosis/code lookup.
    # Knowledge explanations such as “电容老化会有什么影响” may still use the
    # same evidence pipeline, but candidate scores are internal retrieval aids and
    # must not turn a complete question into a generic diagnostic questionnaire.
    if diagnostic_narrowing:
        if component_hmc and len(hmc_rows) > 1:
            clarification = (
                "LCN 与 HSC 在 HMC 权威记录中命中多条不同 HAMC。请补充 HAMC，"
                "或结合原始告警文字和故障阶段区分这些候选。"
            )
        elif support < 0.65:
            clarification = (
                "当前信息不足以唯一定位故障。请补充：①故障发生阶段（上电/初始化/运行中）；"
                "②原始告警文字或 HMC；③是否伴随重启、过温、通信中断或显示异常；"
                "④重新上电后是否恢复。"
            )
        elif (
            len(candidates) == 1
            and not candidates[0].get("structured_hmc_match")
        ):
            clarification = (
                "当前只有一个非精确关键词候选，仍不足以收敛诊断。请补充原始告警或完整 HMC、"
                "故障发生阶段、可复现条件及伴随现象。"
            )
        elif len(candidates) >= 2 and candidates[0]["score"] - candidates[1]["score"] < 0.12:
            clarification = (
                "前两项候选接近，请补充故障发生阶段、可复现条件以及是否存在明确告警码，"
                "以区分相邻故障模式。"
            )

    context = _format_domain_context(
        query=query,
        function_rows=function_rows,
        hmc_rows=hmc_rows,
        fmeca_rows=fmeca_rows,
        fta_rows=fta_rows,
        rbd_rows=rbd_rows,
        mfl_rows=mfl_rows,
        history_rows=history_rows,
        hybrid_rows=hybrid_rows,
        confidence=support,
        clarification=clarification,
    )

    domain_results: List[Dict[str, Any]] = []
    for row in hybrid_rows[:20]:
        domain_results.append(
            {
                "source_type": row.get("kb_id") or row.get("section") or "document",
                "doc_id": row.get("doc_id", ""),
                "doc_name": row.get("doc_name", ""),
                "chunk_id": row.get("chunk_id", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "score": row.get("score", 0),
            }
        )
    for row in hmc_rows[:8]:
        domain_results.append(
            {
                "source_type": "hmc",
                "doc_id": row.get("doc_id", ""),
                "title": row.get("hmc_code", "HMC"),
                "content": f"{row.get('fault_desc', '')}\n维修方法：{row.get('repair_method', '')}",
                "score": (
                    1.0
                    if row.get("match_type") == "exact_code"
                    else (0.9 if row.get("match_type") == "exact_components" else 0.75)
                ),
            }
        )

    if diagnostic_narrowing:
        tool_calls.append(
            _tool_call(
                "fault_reasoning",
                len(candidates),
                time.time(),
                [
                    {
                        "title": c["fault"],
                        "snippet": f"排序启发式分数（非概率） {c['score']}",
                        "evidence_ids": list(c.get("supporting_evidence_ids") or [])[:4],
                    }
                    for c in candidates
                ],
                evidence_support=round(support, 3),
                clarification_required=bool(clarification),
            )
        )
    tool_results.append(
        make_tool_result(
            tool="evidence_synthesis",
            status="ok" if candidates else "empty",
            params={"candidate_limit": 8},
            data={"candidate_count": len(candidates), "diagnostic_support": round(support, 3)},
            evidence_ids=[item["evidence_id"] for item in evidence],
            backend="local_deterministic",
        )
    )
    _attach_tool_call_evidence(tool_calls, tool_results, evidence)

    seed_bindings = _graph_seed_bindings(
        function_rows=function_rows,
        hmc_rows=hmc_rows,
        fmeca_rows=fmeca_rows,
        fta_rows=fta_rows,
        rbd_rows=rbd_rows,
        mfl_rows=mfl_rows,
        history_rows=history_rows,
    )
    enriched_topology_dependency = _enriched_topology_dependency(
        topology_dependency,
        structured_entity_binding,
        fmeca_rbd_identity_terms,
    )

    return {
        "diagnostic_context": context,
        "candidate_faults": candidates,
        "diagnostic_support": round(support, 3),
        # 旧前端字段仅兼容展示；语义已经改为“证据支持度”，不是故障概率。
        "diagnostic_confidence": round(support, 3),
        "clarification_question": clarification,
        "domain_results": domain_results,
        # 兼容旧 Answer/引用逻辑：把跨源结果同时写入 vector_results。
        "vector_results": domain_results,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "evidence_registry": evidence,
        "search_locator_hits": locator_handoffs,
        "graph_seed_bindings": seed_bindings,
        "graph_traversal_request": _graph_request(seed_bindings, allowed_tools),
        "collection_resolution": collection_resolution,
        "topology_dependency": enriched_topology_dependency,
        **(
            {"query_frame": architecture_collection_frame}
            if architecture_collection_frame is not None
            else {}
        ),
        # Keep the legacy state key for graph-node compatibility, but never put
        # provisional material into the deterministic-calculation handoff.
        "confirmed_graphs": [
            row for row in [*fta_rows, *rbd_rows] if _is_formal_graph_row(row)
        ],
    }
