"""Pure stable graph identities shared by retrieval and calculation."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Mapping


def _graph_identity_values(row: Mapping[str, Any]) -> set[str]:
    """Return stable identifiers exposed by a graph revision or source row.

    The graph repository commonly returns a revision UUID, while FTA/RBD
    source evidence uses the business key (for example ``FTA-SYN-005``).
    Both identify the same model and must bind to one another for deterministic
    calculation evidence to pass the provenance gate.  Human-readable titles
    are deliberately excluded: they are useful recall text, but are not a
    model identity and may be shared by several documents/configurations.
    """

    values: set[str] = set()
    layers: List[Mapping[str, Any]] = [row]
    for container_key in ("metadata", "document_metadata", "answer_fields"):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            layers.append(container)
            answer_fields = container.get("answer_fields")
            if isinstance(answer_fields, Mapping):
                layers.append(answer_fields)
    for layer in layers:
        for key in ("revision_id", "diagram_key", "diagram_id", "record_id"):
            value = str(layer.get(key) or "").strip()
            if value:
                values.add(value.casefold())
    return values


def _graph_scope_values(row: Mapping[str, Any], scope_name: str) -> set[str]:
    """Return stable document/configuration qualifiers when they are present."""

    aliases = {
        "doc_id": ("doc_id",),
        "configuration": (
            "configuration",
            "applicable_configuration",
            "configuration_id",
            "config_id",
            "config",
        ),
    }[scope_name]
    layers: List[Mapping[str, Any]] = [row]
    for key in ("metadata", "document_metadata", "scope"):
        value = row.get(key)
        if isinstance(value, Mapping):
            layers.append(value)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("answer_fields", "scope"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                layers.append(value)
    result: set[str] = set()
    for layer in layers:
        for key in aliases:
            value = str(layer.get(key) or "").strip()
            if value:
                result.add(value.casefold())
    return result


def _stable_graph_identity_matches(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Bind two graph records only through stable IDs and compatible scope."""

    if not (_graph_identity_values(left) & _graph_identity_values(right)):
        return False
    for scope_name in ("doc_id", "configuration"):
        left_scope = _graph_scope_values(left, scope_name)
        right_scope = _graph_scope_values(right, scope_name)
        if left_scope and right_scope and not (left_scope & right_scope):
            return False
    return True


def _requested_graph_selectors(
    state: Mapping[str, Any], kind: str
) -> List[Dict[str, set[str]]]:
    """Extract stable, scoped graph selectors from the request and anchors.

    A title can still be present in retrieval terms, but it never enters a
    selector.  Selected citations contribute only record/revision/diagram IDs;
    document and configuration scope are retained when available so a reused
    business key cannot silently cross an applicability boundary.
    """

    prefix = kind.upper()
    selectors: List[Dict[str, set[str]]] = []
    # Calculation target identity may come only from the user's raw turn or a
    # server-offered citation selected for this turn.  Planner entities and
    # normalized terms are recall hints and must not mint a graph selector.
    query_texts: List[str] = [str(state.get("query") or "")]

    requested_scope = state.get("retrieval_scope")
    requested_scope = requested_scope if isinstance(requested_scope, Mapping) else {}
    global_doc_ids = {
        str(requested_scope.get("doc_id") or "").strip().casefold()
    } - {""}
    global_configurations = {
        str(
            requested_scope.get("configuration")
            or requested_scope.get("applicable_configuration")
            or ""
        ).strip().casefold()
    } - {""}

    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(prefix)}(?:[-_:])[A-Za-z0-9\u4e00-\u9fff]"
        rf"[A-Za-z0-9\u4e00-\u9fff._:-]*",
        re.IGNORECASE,
    )
    labelled_pattern = re.compile(
        r"\b(revision_id|diagram_id|diagram_key|record_id)\s*[:=]\s*"
        r"([A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff._:/-]*)",
        re.IGNORECASE,
    )
    for text in query_texts:
        labelled_spans: List[tuple[int, int]] = []
        for labelled in labelled_pattern.finditer(text):
            field = labelled.group(1).casefold()
            selectors.append(
                {
                    "query_ids": set(),
                    "revision_ids": (
                        {labelled.group(2).casefold()}
                        if field == "revision_id"
                        else set()
                    ),
                    "diagram_ids": (
                        {labelled.group(2).casefold()}
                        if field == "diagram_id"
                        else set()
                    ),
                    "diagram_keys": (
                        {labelled.group(2).casefold()}
                        if field == "diagram_key"
                        else set()
                    ),
                    "record_ids": (
                        {labelled.group(2).casefold()}
                        if field == "record_id"
                        else set()
                    ),
                    "doc_ids": set(global_doc_ids),
                    "configurations": set(global_configurations),
                }
            )
            labelled_spans.append(labelled.span())
        for match in pattern.finditer(text):
            if any(start <= match.start() and match.end() <= end for start, end in labelled_spans):
                continue
            selectors.append(
                {
                    "query_ids": {match.group(0).casefold()},
                    "revision_ids": set(),
                    "diagram_ids": set(),
                    "diagram_keys": set(),
                    "record_ids": set(),
                    "doc_ids": set(global_doc_ids),
                    "configurations": set(global_configurations),
                }
            )

    for anchor in state.get("selected_conversation_anchors") or []:
        if not isinstance(anchor, Mapping):
            continue
        if str(anchor.get("doc_type") or "").strip().casefold() not in {"", kind.casefold()}:
            continue
        selector: Dict[str, set[str]] = {
            "query_ids": set(),
            "revision_ids": set(),
            "diagram_ids": set(),
            "diagram_keys": set(),
            "record_ids": set(),
            "doc_ids": _graph_scope_values(anchor, "doc_id") or set(global_doc_ids),
            "configurations": (
                _graph_scope_values(anchor, "configuration")
                or set(global_configurations)
            ),
        }
        record_id = str(anchor.get("record_id") or "").strip()
        if record_id:
            selector["record_ids"].add(record_id.casefold())
        fields = anchor.get("answer_fields")
        if isinstance(fields, Mapping):
            for key, bucket in (
                ("revision_id", "revision_ids"),
                ("diagram_id", "diagram_ids"),
                ("diagram_key", "diagram_keys"),
                ("record_id", "record_ids"),
            ):
                value = str(fields.get(key) or "").strip()
                if value:
                    selector[bucket].add(value.casefold())
        if any(
            selector[key]
            for key in (
                "revision_ids",
                "diagram_ids",
                "diagram_keys",
                "record_ids",
            )
        ):
            selectors.append(selector)
    return selectors


def _requested_graph_keys(state: Mapping[str, Any], kind: str) -> set[str]:
    """Return the stable graph IDs requested or cited in this turn."""

    return {
        value
        for selector in _requested_graph_selectors(state, kind)
        for key in (
            "query_ids",
            "revision_ids",
            "diagram_ids",
            "diagram_keys",
            "record_ids",
        )
        for value in selector[key]
    }


def _selector_match_score(
    row: Mapping[str, Any], selector: Mapping[str, set[str]]
) -> int:
    """Rank exact stable-ID matches; return zero for unverified scope."""

    requested_doc_ids = selector.get("doc_ids") or set()
    requested_configurations = selector.get("configurations") or set()
    row_doc_ids = _graph_scope_values(row, "doc_id")
    row_configurations = _graph_scope_values(row, "configuration")
    if requested_doc_ids and not (row_doc_ids & requested_doc_ids):
        return 0
    if requested_configurations and not (
        row_configurations & requested_configurations
    ):
        return 0

    def direct_values(field: str) -> set[str]:
        result: set[str] = set()
        layers: List[Mapping[str, Any]] = [row]
        for container_key in ("metadata", "document_metadata", "answer_fields"):
            container = row.get(container_key)
            if not isinstance(container, Mapping):
                continue
            layers.append(container)
            answer_fields = container.get("answer_fields")
            if isinstance(answer_fields, Mapping):
                layers.append(answer_fields)
        for layer in layers:
            value = str(layer.get(field) or "").strip()
            if value:
                result.add(value.casefold())
        return result

    # A revision ID is the strongest selector, followed by the stable diagram
    # UUID and business key.  ``record_id`` is cross-compared because evidence
    # records historically stored either a revision ID or a diagram key there.
    if direct_values("revision_id") & (selector.get("revision_ids") or set()):
        return 500
    if direct_values("diagram_id") & (selector.get("diagram_ids") or set()):
        return 400
    if direct_values("diagram_key") & (selector.get("diagram_keys") or set()):
        return 300
    row_ids = _graph_identity_values(row)
    if row_ids & (selector.get("record_ids") or set()):
        return 200
    if row_ids & (selector.get("query_ids") or set()):
        return 100
    return 0
