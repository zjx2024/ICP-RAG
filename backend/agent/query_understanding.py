"""Planner-provided query slots and retrieval-query helpers.

Semantic understanding belongs to the planning model and to embedding search.
This module deliberately contains no retrieval synonym dictionary and no
intent classifier.  Its small stage/cadence phrase maps are provenance guards:
they only verify that a planner slot agrees with the user's quoted wording.
They do not expand retrieval or infer a slot when the planner omitted one.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from backend.agent.answer_contract import normalize_answer_fields


_BUSINESS_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z]{2,}[A-Za-z0-9]*(?:[._/-][A-Za-z0-9*]+)+"
    r"|\d{4,}(?:\*{2}[A-Za-z0-9]+|[._/-][A-Za-z0-9*]+)+"
    r"|\d{6,}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_EXPLICIT_BRANCH_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9一二三四五六七八九十])\s*"
    r"(?:路|通道|支路|分支)"
)
_TECHNICAL_BRANCH_RE = re.compile(r"(?:^|[-_/])([A-Z0-9])$", re.IGNORECASE)

# These maps are intentionally narrow and deterministic.  Stage and cadence
# become hard relevance filters later in the pipeline, so a verbatim basis is
# accepted only when it contains wording assigned to the planner's selected
# enum.  This prevents a model from citing “运行中” while selecting “高负载”,
# without turning this validator into another intent or retrieval classifier.
_OPERATING_STAGE_ENUM_BY_PHRASE: Dict[str, str] = {
    "上电": "上电",
    "加电": "上电",
    "通电": "上电",
    "开机": "上电",
    "初始化": "初始化",
    "启动阶段": "初始化",
    "启动时": "初始化",
    "自检阶段": "初始化",
    "自检时": "初始化",
    "运行中": "运行中",
    "运行时": "运行中",
    "运行过程": "运行中",
    "正常运行": "运行中",
    "正常工作时": "运行中",
    "工作中": "运行中",
    "高负载": "高负载",
    "满负荷": "高负载",
    "大负载": "高负载",
    "峰值负载": "高负载",
    "持续运行后": "持续运行后",
    "运行一段时间后": "持续运行后",
    "工作一段时间后": "持续运行后",
    "长时间运行后": "持续运行后",
    "长时间工作后": "持续运行后",
    "用久后": "持续运行后",
    "久用后": "持续运行后",
}

_TEMPORAL_PATTERN_ENUM_BY_PHRASE: Dict[str, str] = {
    "反复": "重复发生",
    "重复": "重复发生",
    "循环": "重复发生",
    "多次": "重复发生",
    "一再": "重复发生",
    "间歇": "间歇发生",
    "偶发": "间歇发生",
    "时有时无": "间歇发生",
    "不定期": "间歇发生",
    "偶尔": "间歇发生",
    # Cadence needs an occurrence predicate. Bare “持续/连续” may only
    # describe operating duration (for example “持续运行后”) and must not
    # silently become a hard “持续发生” filter.
    "持续发生": "持续发生",
    "持续出现": "持续发生",
    "一直发生": "持续发生",
    "一直存在": "持续发生",
    "始终发生": "持续发生",
    "连续发生": "持续发生",
    "连续出现": "持续发生",
    "首次": "首次发生",
    "第一次": "首次发生",
    "初次": "首次发生",
}


def normalize_search_term(value: Any) -> str:
    """Return a comparison-only Unicode-normalized string.

    This is intentionally lexical. It does not remove question phrases or
    claim that two different domain expressions are synonyms.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _iter_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip()
        if text:
            yield text
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_text(item)


def extract_business_codes(value: Any, *, limit: int = 12) -> List[str]:
    """Extract explicit engineering-code tokens without assigning a type.

    A code such as ``FM-SYN-002`` is safe to parse lexically, but its prefix
    is not used here to decide whether it belongs to FMECA, MFL, history or an
    architecture table.  Callers can therefore probe authoritative identity
    columns and let stored data determine the record family.
    """

    result: List[str] = []
    seen: set[str] = set()
    for text in _iter_text(value):
        for match in _BUSINESS_CODE_RE.finditer(text):
            code = unicodedata.normalize("NFKC", match.group(0)).strip()
            identity = code.casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            result.append(code)
            if len(result) >= max(0, int(limit)):
                return result
    return result


def _unique_text(values: Iterable[Any], *, limit: int) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _iter_text(value):
            identity = normalize_search_term(text)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            result.append(text)
            if len(result) >= limit:
                return result
    return result


def _generic_tokens(value: Any) -> List[str]:
    """Extract technical tokens without performing semantic classification."""

    result: List[str] = []
    for text in _iter_text(value):
        result.extend(
            re.findall(
                r"[A-Za-z][A-Za-z0-9_./-]{1,40}|\d{3,}|[\u4e00-\u9fff]{2,32}",
                text,
            )
        )
    return result


def extract_explicit_branch_markers(value: Any) -> List[str]:
    """Extract literal branch designators without assigning domain meaning.

    ``A路`` and an identity label ending in ``-A`` share the marker ``a``.
    This is a lexical constraint only: it cannot create an entity identity and
    is useful only together with a non-trivial overlap against authoritative
    Function Locator identity fields.
    """

    result: List[str] = []
    seen: set[str] = set()
    for text in _iter_text(value):
        normalized = unicodedata.normalize("NFKC", text)
        for match in _EXPLICIT_BRANCH_RE.finditer(normalized):
            marker = match.group(1).casefold()
            if marker not in seen:
                seen.add(marker)
                result.append(marker)
        for token in re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+", normalized):
            match = _TECHNICAL_BRANCH_RE.search(token)
            if match:
                marker = match.group(1).casefold()
                if marker not in seen:
                    seen.add(marker)
                    result.append(marker)
    return result


def _structured_text_values(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        return []
    return list(_iter_text(value))


def _function_identity_labels(
    row: Mapping[str, Any],
) -> List[Tuple[str, str, str]]:
    """Return ``(business_id, label, field)`` from stored identity columns."""

    result: List[Tuple[str, str, str]] = []

    def add(business_id: Any, label: Any, field: str) -> None:
        identity = str(business_id or "").strip()
        if not identity:
            return
        for text in _structured_text_values(label):
            if text.strip():
                result.append((identity, text.strip(), field))

    component_id = row.get("component_id")
    add(component_id, component_id, "component_id")
    add(component_id, row.get("name"), "name")
    add(component_id, row.get("abbreviation"), "abbreviation")

    module_id = row.get("module_id")
    add(module_id, module_id, "module_id")
    add(module_id, row.get("module_name"), "module_name")

    function_id = row.get("function_id")
    add(function_id, function_id, "function_id")
    add(function_id, row.get("function_name"), "function_name")
    add(function_id, row.get("aliases"), "aliases")

    interface_id = row.get("interface_id")
    add(interface_id, interface_id, "interface_id")
    add(interface_id, row.get("signal_name"), "signal_name")
    return result


def _longest_common_text(left: str, right: str) -> str:
    match = SequenceMatcher(None, left, right).find_longest_match(
        0, len(left), 0, len(right)
    )
    return left[match.a : match.a + match.size]


def bind_structured_function_entity(
    values: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 16,
) -> Dict[str, Any]:
    """Bind a natural-language object to one stored Function Locator ID.

    The binding uses only exact codes and lexical evidence from structured
    identity/name/abbreviation/alias fields.  Descriptions, failure text,
    embeddings and vector scores are deliberately excluded.  A short generic
    overlap such as ``电源`` cannot bind by itself; an explicit branch marker
    plus a longer identity overlap (for example ``A路电源``) can.

    The result is a query handoff, not a cross-document entity merge.  Later
    document families still establish identity only through their own stored
    business IDs and PostgreSQL-authorized relations.
    """

    inputs = _unique_text([values], limit=24)
    normalized_inputs = [
        (text, normalize_search_term(text), set(extract_explicit_branch_markers(text)))
        for text in inputs
        if normalize_search_term(text)
    ]
    explicit_codes = {
        normalize_search_term(code) for code in extract_business_codes(inputs)
    }
    if not normalized_inputs or not rows:
        return {
            "bound": False,
            "ambiguous": False,
            "business_ids": [],
            "terms": [],
            "row_indexes": [],
            "basis": "",
        }

    scored: Dict[str, Dict[str, Any]] = {}
    labels_by_id: Dict[str, List[str]] = {}
    row_ids: List[set[str]] = []
    for index, raw_row in enumerate(rows):
        identities = _function_identity_labels(raw_row)
        current_ids = {identity for identity, _, _ in identities}
        row_ids.append(current_ids)
        for business_id, label, field in identities:
            normalized_label = normalize_search_term(label)
            if not normalized_label:
                continue
            labels_by_id.setdefault(business_id, []).append(label)
            label_markers = set(extract_explicit_branch_markers(label))
            best = (0, "", "")
            for _, normalized_input, input_markers in normalized_inputs:
                score = 0
                basis = ""
                common = _longest_common_text(normalized_input, normalized_label)
                if (
                    normalized_label in explicit_codes
                    and field in {"component_id", "module_id", "function_id", "interface_id"}
                ):
                    score, basis = 10000, "exact_business_id"
                elif normalized_input == normalized_label and len(normalized_label) >= 2:
                    score, basis = 8000, "exact_identity_label"
                elif input_markers & label_markers and len(common) >= 3:
                    # A branch/channel marker comes directly from the user
                    # expression (for example “A路”); it is a materially
                    # stronger identity discriminator than a generic label
                    # containment such as “供电模块” → “保持供电模块”.
                    score = 9000 + len(common) ** 2
                    basis = "explicit_branch_identity_overlap"
                elif (
                    min(len(normalized_input), len(normalized_label)) >= 4
                    and (
                        normalized_input in normalized_label
                        or normalized_label in normalized_input
                    )
                ):
                    score = 4000 + min(len(normalized_input), len(normalized_label)) ** 2
                    basis = "identity_label_containment"
                elif len(common) >= 5:
                    score = 1000 + len(common) ** 2
                    basis = "specific_identity_overlap"
                if score > best[0]:
                    best = (score, basis, common)
            if best[0] <= 0:
                continue
            candidate = scored.setdefault(
                business_id,
                {
                    "score": 0,
                    "basis": "",
                    "fragments": set(),
                    "row_indexes": set(),
                    "identity_kinds": set(),
                },
            )
            candidate["row_indexes"].add(index)
            if field in {"component_id", "name", "abbreviation"}:
                candidate["identity_kinds"].add("component")
            elif field in {"module_id", "module_name"}:
                candidate["identity_kinds"].add("module")
            elif field in {"function_id", "function_name", "aliases"}:
                candidate["identity_kinds"].add("function")
            elif field in {"interface_id", "signal_name"}:
                candidate["identity_kinds"].add("interface")
            if best[0] > candidate["score"]:
                candidate["score"] = best[0]
                candidate["basis"] = best[1]
            if len(best[2]) >= 3:
                candidate["fragments"].add(best[2])

    ranked = sorted(
        scored.items(), key=lambda item: (-int(item[1]["score"]), item[0].casefold())
    )
    if not ranked:
        return {
            "bound": False,
            "ambiguous": False,
            "business_ids": [],
            "terms": [],
            "row_indexes": [],
            "basis": "",
        }
    business_id, winner = ranked[0]
    if len(ranked) > 1 and winner["score"] - ranked[1][1]["score"] < 100:
        close_candidates = [
            item
            for item in ranked
            if winner["score"] - item[1]["score"] < 100
        ]
        exact_tie = all(
            candidate["score"] == winner["score"]
            for _, candidate in close_candidates
        )
        common_rows = set.intersection(
            *(
                set(candidate["row_indexes"])
                for _, candidate in close_candidates
            )
        )
        single_authoritative_row = (
            exact_tie
            and len(common_rows) == 1
            and all(
                set(candidate["row_indexes"]) == common_rows
                for _, candidate in close_candidates
            )
        )
        candidate_kinds = {
            identity: set(candidate.get("identity_kinds") or [])
            for identity, candidate in close_candidates
        }
        structural_chain = (
            single_authoritative_row
            and all(
                kinds <= {"component", "module", "function"}
                for kinds in candidate_kinds.values()
            )
            and any(
                kinds & {"component", "module"}
                for kinds in candidate_kinds.values()
            )
            and any("function" in kinds for kinds in candidate_kinds.values())
        )
        if not structural_chain:
            return {
                "bound": False,
                "ambiguous": True,
                "business_ids": [],
                "terms": [],
                "row_indexes": [],
                "basis": "ambiguous_structured_identity",
            }

        def chain_priority(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
            kinds = set(item[1].get("identity_kinds") or [])
            if "component" in kinds:
                priority = 0
            elif "module" in kinds:
                priority = 1
            else:
                priority = 2
            return priority, item[0].casefold()

        business_id, winner = min(close_candidates, key=chain_priority)
        winner = {**winner, "basis": "same_row_module_function_chain"}

    matching_indexes = {
        index for index, identities in enumerate(row_ids) if business_id in identities
    }
    # Every ID on the same authoritative Function Locator row is an explicit
    # structural chain.  Preserve those IDs in the durable handoff as well as
    # in retrieval terms.  This lets an exact FUNC query reach its implementing
    # module, and a module query reach its linked function, without using names
    # or vector similarity as cross-document identity.
    linked_business_ids: List[str] = [business_id]
    linked_identity_terms: List[str] = []
    for index in sorted(matching_indexes):
        row = rows[index]
        for field in ("component_id", "module_id", "function_id", "interface_id"):
            value = str(row.get(field) or "").strip()
            if value and normalize_search_term(value) not in {
                normalize_search_term(existing) for existing in linked_business_ids
            }:
                linked_business_ids.append(value)
        for linked_id, label, _ in _function_identity_labels(row):
            if normalize_search_term(linked_id) in {
                normalize_search_term(value) for value in linked_business_ids
            }:
                linked_identity_terms.append(label)
    ordered_terms: List[Any] = [
        linked_business_ids,
        labels_by_id.get(business_id, []),
        linked_identity_terms,
        sorted(winner["fragments"], key=lambda value: (-len(value), value)),
    ]
    return {
        "bound": True,
        "ambiguous": False,
        "business_ids": linked_business_ids,
        "terms": _unique_text(ordered_terms, limit=max(int(limit), 0)),
        "row_indexes": sorted(matching_indexes),
        "basis": winner["basis"],
    }


def normalize_tool_queries(value: Any) -> Dict[str, List[str]]:
    """Accept planner tool queries as strings or lists and bound their size."""

    if not isinstance(value, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for raw_tool, raw_queries in value.items():
        tool = str(raw_tool or "").strip()
        if not tool:
            continue
        queries = _unique_text([raw_queries], limit=6)
        if queries:
            result[tool] = queries
    return result


def build_query_frame(
    *,
    query: Any,
    rewritten_query: Any = "",
    entity_keywords: Any = None,
    query_targets: Any = None,
    planner_frame: Any = None,
) -> Dict[str, Any]:
    """Validate the semantic frame emitted by the planning model.

    Missing slots stay missing/unknown. In particular, this function never
    infers an operating stage, temporal pattern, symptom alias or intent from
    raw text.
    """

    source = dict(planner_frame) if isinstance(planner_frame, dict) else {}

    def evidence_backed_slot(
        value: Any,
        basis: Any,
        enum_by_phrase: Mapping[str, str],
    ) -> Tuple[str, str]:
        """Keep planner constraints only when tied to the user's own words.

        Stage and occurrence cadence are relevance filters, so an invented or
        over-interpreted slot can hide an otherwise exact PostgreSQL record.
        The model still performs the semantic classification; this validator
        requires both a meaningful verbatim basis from the current turn and a
        deterministic phrase mapping that agrees with the selected enum.
        """

        normalized_value = str(value or "").strip()
        allowed = set(enum_by_phrase.values())
        if normalized_value not in allowed or normalized_value == "未知":
            return "未知", ""
        rendered_basis = str(basis or "").strip()
        normalized_basis = normalize_search_term(rendered_basis)
        normalized_query = normalize_search_term(query)
        if (
            len(normalized_basis) < 2
            or not normalized_query
            or normalized_basis not in normalized_query
        ):
            return "未知", ""
        mapped_values = {
            mapped_value
            for phrase, mapped_value in enum_by_phrase.items()
            if normalize_search_term(phrase) in normalized_basis
        }
        if normalized_value not in mapped_values:
            return "未知", ""
        return normalized_value, rendered_basis

    operating_stage, operating_stage_basis = evidence_backed_slot(
        source.get("operating_stage"),
        source.get("operating_stage_basis"),
        _OPERATING_STAGE_ENUM_BY_PHRASE,
    )
    temporal_pattern, temporal_pattern_basis = evidence_backed_slot(
        source.get("temporal_pattern"),
        source.get("temporal_pattern_basis"),
        _TEMPORAL_PATTERN_ENUM_BY_PHRASE,
    )
    target_intents: List[str] = []
    for target in query_targets or []:
        intent = target.get("intent") if isinstance(target, dict) else target
        intent = str(intent or "").strip()
        if intent and intent not in target_intents:
            target_intents.append(intent)

    entities = _unique_text(
        [entity_keywords or [], source.get("entities") or []],
        limit=16,
    )
    symptoms = _unique_text([source.get("symptoms") or []], limit=16)
    normalized_terms = _unique_text(
        [source.get("normalized_terms") or [], entities, symptoms],
        limit=32,
    )
    coverage_scope = str(source.get("coverage_scope") or "").strip()
    if coverage_scope not in {
        "complete_collection",
        "focused_detail",
        "relation_path",
        "unknown",
    }:
        coverage_scope = "unknown"
    collection_subject = _unique_text([source.get("collection_subject")], limit=1)
    collection_member_kind = str(
        source.get("collection_member_kind") or ""
    ).strip()
    if collection_member_kind not in {
        "failure_mode",
        "cause",
        "component",
        "module",
        "subsystem",
        "record",
        "unknown",
    }:
        collection_member_kind = "unknown"
    requested_facets = _unique_text(
        [source.get("requested_facets") or []],
        limit=16,
    )
    required_answer_fields = normalize_answer_fields(
        source.get("required_answer_fields"),
        requested_facets=requested_facets,
    )
    raw_closure = source.get("diagnostic_closure")
    if isinstance(raw_closure, Mapping):
        closure_reason = " ".join(
            str(raw_closure.get("reason") or "").split()
        )[:360]
        closure_required = bool(
            raw_closure.get("required") is True and closure_reason
        )
    else:
        closure_required = False
        closure_reason = ""
    return {
        "entities": entities,
        "symptoms": symptoms,
        "operating_stage": operating_stage,
        "operating_stage_basis": operating_stage_basis,
        "temporal_pattern": temporal_pattern,
        "temporal_pattern_basis": temporal_pattern_basis,
        # This generic slot is authored by the planning model. Retrieval uses
        # it to distinguish an explicitly requested complete collection from a
        # focused detail or relation path; no raw-text keyword classifier is
        # introduced here as a fallback.
        "coverage_scope": coverage_scope,
        # A collection contract is model-authored and later backed only by a
        # PostgreSQL closed-set resolver. It never turns a semantic Top-K hit
        # into a purportedly complete inventory by itself.
        "collection_subject": collection_subject[0] if collection_subject else "",
        "collection_member_kind": collection_member_kind,
        "requested_facets": requested_facets,
        # This typed contract is authored by the semantic planner and merely
        # validated here.  Later nodes use it for tool stopping and output
        # completeness; they never re-classify the raw question by keywords.
        "required_answer_fields": required_answer_fields,
        # This semantic flag comes from the target classifier.  It separates
        # an observed/current abnormality needing bounded evidence closure
        # from a detached one-field knowledge explanation; no raw-text rule is
        # introduced here.
        "diagnostic_closure": {
            "required": closure_required,
            "reason": closure_reason if closure_required else "",
        },
        "constraints": _unique_text([source.get("constraints") or []], limit=12),
        "query_targets": target_intents,
        "normalized_terms": normalized_terms,
        # Retain both wordings for auditability, not as inferred slots or facts.
        "original_query": str(query or "").strip(),
        "rewritten_query": str(rewritten_query or "").strip(),
    }


def closed_inventory_requested(state: Dict[str, Any]) -> bool:
    """Whether the planner explicitly requested a complete collection.

    A complete-collection answer has a closed-list obligation: once an
    authoritative query has returned every direct member, the answer must not
    silently present a partial subset as the whole. Keeping this decision in
    the semantic query frame avoids inferring user intent from hard-coded
    phrases.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), dict) else {}
    return str(frame.get("coverage_scope") or "") == "complete_collection"


def build_tool_terms(
    state: Dict[str, Any],
    tool: str,
    base_terms: Iterable[Any] = (),
    *,
    limit: int = 24,
) -> List[str]:
    """Build lexical PostgreSQL terms from planner-authored tool queries.

    Tool-specific queries take priority. The generic frame and original query
    are bounded fallbacks. Semantic synonym recall remains OpenSearch/BGE's
    responsibility.
    """

    frame = state.get("query_frame") if isinstance(state.get("query_frame"), dict) else {}
    tool_queries = normalize_tool_queries(state.get("tool_queries"))
    authored = tool_queries.get(tool, [])
    ordered: List[Any] = [
        authored,
        frame.get("entities") or [],
        frame.get("symptoms") or [],
        frame.get("operating_stage") or "",
        frame.get("temporal_pattern") or "",
        frame.get("constraints") or [],
        frame.get("normalized_terms") or [],
        list(base_terms),
        state.get("rewritten_query") or "",
        state.get("query") or "",
    ]
    # Keep complete planner phrases and their generic technical tokens. This
    # is tokenization only; it introduces no aliases.
    return _unique_text([ordered, _generic_tokens(authored)], limit=max(int(limit), 0))


def build_hybrid_query(state: Dict[str, Any]) -> str:
    """Return the planner-authored semantic query used for BM25+BGE recall."""

    tool_queries = normalize_tool_queries(state.get("tool_queries"))
    values = tool_queries.get("hybrid_search", [])
    if not values:
        values = [
            str(state.get("rewritten_query") or state.get("query") or "").strip()
        ]
    return "；".join(_unique_text([values], limit=4))


def diagram_match_score(text: Any, terms: Iterable[Any]) -> Tuple[int, List[str]]:
    """Score lexical overlap after semantic recall or planner normalization."""

    haystack = normalize_search_term(text)
    matched: List[str] = []
    seen: set[str] = set()
    for value in terms:
        raw = str(value or "").strip()
        term = normalize_search_term(raw)
        if (len(term) < 2 and not re.fullmatch(r"[a-z0-9]", term)) or term in seen:
            continue
        seen.add(term)
        if term in haystack:
            matched.append(raw)
    return sum(len(normalize_search_term(value)) ** 2 for value in matched), matched
