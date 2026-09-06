"""Read-only formal FTA/RBD views with shared authority revalidation.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.nodes.answer_payload import _evidence_answer_fields
from backend.agent.nodes.answer_payload import _first_structured_value
from backend.agent.nodes.answer_payload import _flatten_target_values
from backend.agent.nodes.answer_payload import _json_object
from backend.agent.nodes.answer_payload import _normalize_business_id
from backend.agent.nodes.answer_request import _topology_dependency
from backend.agent.rbd_contract import rbd_model_descriptor
from backend.agent.rbd_contract import validate_rbd_authority
from backend.agent.rbd_contract import validated_rbd_single_failure_relations
from typing import Any
from typing import List
from typing import Mapping
import json


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


_RBD_FORMAL_AUTHORITIES = {"confirmed_relation", "validated_relation"}


_RBD_NON_FORMAL_REVIEW_STATUSES = {
    "pending",
    "provisional",
    "quarantined",
    "failed",
    "rejected",
}


_RBD_CONNECTION_LABELS = {
    "series": "串联",
    "parallel": "并联",
    "redundant": "冗余",
    "bypass": "旁路",
}


def _rbd_structure_is_formal(item: Mapping[str, Any]) -> bool:
    """Reject a non-formal RBD revision even if its authority is inconsistent.

    Evidence gate normally keeps these states consistent.  The answer layer
    still fails closed so a stale or synthetic ``validated_relation`` cannot
    promote a provisional visual extraction into an operational statement.
    ``not_required`` remains accepted for legacy, already-authorized evidence.
    """

    if (
        str(item.get("doc_type") or "") != "rbd"
        or str(item.get("claim_type") or "") != "rbd_structure"
        or str(item.get("authority") or "") not in _RBD_FORMAL_AUTHORITIES
    ):
        return False
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    graph = _json_object(item.get("content"))
    graph_metadata = (
        graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
    )
    statuses = {
        str(value or "").strip().casefold()
        for value in (
            item.get("review_status"),
            metadata.get("validation_status"),
            graph_metadata.get("validation_status"),
        )
        if str(value or "").strip()
    }
    return not bool(statuses & _RBD_NON_FORMAL_REVIEW_STATUSES)


def _rbd_target_boundary_clauses(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> List[str]:
    """Return one explicit formal relation bound to the exact queried object."""

    dependency = _topology_dependency(state)
    if not dependency:
        return []
    models = _rbd_authority_models(item, state)
    relations = validated_rbd_single_failure_relations(
        item,
        binding=state.get("rbd_query_binding"),
        raw_user_query=state.get("query"),
        selected_anchors=state.get("selected_conversation_anchors") or [],
        target_business_ids=dependency.get("canonical_business_ids") or [],
        models=models,
    )
    return [str(relation["statement"]) for relation in relations]


def _rbd_authority_models(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> List[Mapping[str, Any]]:
    models = [
        value
        for value in (state.get("evidence_registry") or [])
        if isinstance(value, Mapping)
        and str(value.get("doc_type") or "") == "rbd"
        and str(value.get("claim_type") or "") == "rbd_structure"
    ]
    evidence_id = str(item.get("evidence_id") or "")
    if not any(
        value is item
        or (
            evidence_id
            and str(value.get("evidence_id") or "") == evidence_id
        )
        for value in models
    ):
        models.append(item)
    return models


def _rbd_item_authority_is_valid(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    models = _rbd_authority_models(item, state)
    validation = validate_rbd_authority(
        state.get("rbd_query_binding"),
        models=models,
        raw_user_query=state.get("query"),
        selected_anchors=state.get("selected_conversation_anchors") or [],
    )
    if not validation.get("valid"):
        return False
    item_revision = _normalize_business_id(
        rbd_model_descriptor(item).get("revision_id")
    )
    return bool(item_revision and item_revision in set(validation.get("revision_ids") or []))


def _rbd_matches_topology_dependency(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    if str(item.get("doc_type") or "") != "rbd":
        return True
    # The target-group boundary applies only to source RBD structure.  A
    # deterministic calculator result has already been authorized against its
    # selected formal graph and must not be reinterpreted as a graph payload;
    # it does not carry branch vertices or single-failure clauses to match.
    claim_type = str(item.get("claim_type") or "")
    if claim_type == "rbd_calculation":
        return True
    if claim_type != "rbd_structure":
        return False
    return _rbd_item_authority_is_valid(item, state)


def _rbd_structure_answer_fields_for_state(
    item: Mapping[str, Any], state: Mapping[str, Any]
) -> Mapping[str, Any]:
    fields = dict(_rbd_structure_answer_fields(item))
    if str(item.get("doc_type") or "") != "rbd":
        return fields
    if not _rbd_item_authority_is_valid(item, state):
        return {}
    fields.pop("single_failure_boundary", None)
    if not _topology_dependency(state):
        return fields
    clauses = _rbd_target_boundary_clauses(item, state)
    if not clauses:
        return {}
    fields["single_failure_boundary"] = clauses[0] if len(clauses) == 1 else clauses
    return fields


def _rbd_structure_answer_fields(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project only explicit, approved RBD JSON fields into the answer contract.

    RBD query evidence predates ``metadata.answer_fields`` and therefore keeps
    the versioned graph in ``content``.  This adapter deliberately recognizes a
    small whitelist instead of asking the LLM to interpret the raw graph.  It
    does not enumerate paths, calculate reliability, or infer fault tolerance
    from topology.
    """

    fields = dict(_evidence_answer_fields(item))
    # Old rows may contain an untyped prose field with this name.  It is not a
    # formal topology relation and must never cross the answer boundary.
    fields.pop("single_failure_boundary", None)
    fields.pop("single_failure_relations", None)
    if (
        str(item.get("doc_type") or "") != "rbd"
        or str(item.get("claim_type") or "") != "rbd_structure"
    ):
        return fields
    if not _rbd_structure_is_formal(item):
        return {}
    graph = _json_object(item.get("content"))
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
    containers = (fields, metadata, graph)

    aliases = {
        "success_criterion": ("success_criterion", "success_criteria"),
        "structure_description": (
            "structure",
            "structure_description",
            "topology_description",
        ),
        "structure_expression": ("structure_expression", "topology_expression"),
        "path_structure": ("paths", "success_paths", "series_paths"),
        "redundancy_structure": (
            "redundancy_structure",
            "parallel_branches",
            "redundant_paths",
            "redundancy",
        ),
        "missing_parameters": (
            "missing_parameters",
            "missing_reliability_blocks",
            "parameter_gaps",
            "missing_inputs",
        ),
    }
    for target, keys in aliases.items():
        value = _first_structured_value(containers, keys)
        if value not in (None, "", [], {}):
            fields[target] = value

    connection_types: List[str] = []
    for raw_edge in graph.get("edges") or []:
        if not isinstance(raw_edge, Mapping):
            continue
        properties = raw_edge.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        raw_type = str(properties.get("connection_type") or "").strip().casefold()
        label = _RBD_CONNECTION_LABELS.get(raw_type)
        if label and label not in connection_types:
            connection_types.append(label)
    if connection_types:
        fields.setdefault("connection_types", connection_types)
        redundant_types = [
            value for value in connection_types if value in {"并联", "冗余", "旁路"}
        ]
        if redundant_types:
            fields.setdefault("redundancy_structure", redundant_types)

    # Some source tables explicitly mark which columns are missing.  Preserve
    # those declarations, but do not infer a gap merely because a property is
    # absent from a graph vertex.
    declared_missing: List[str] = []
    parameters = metadata.get("parameters")
    if isinstance(parameters, (list, tuple)):
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                continue
            name = str(
                parameter.get("module_name")
                or parameter.get("name")
                or parameter.get("module_id")
                or "参数项"
            ).strip()
            raw_missing = parameter.get("missing_fields") or parameter.get("missing_parameters")
            values = raw_missing if isinstance(raw_missing, (list, tuple, set)) else [raw_missing]
            for value in values:
                label = str(value or "").strip()
                rendered = f"{name}：{label}" if name and label else label
                if rendered and rendered not in declared_missing:
                    declared_missing.append(rendered)
    if declared_missing:
        existing = _flatten_target_values(fields.get("missing_parameters"))
        fields["missing_parameters"] = list(dict.fromkeys([*existing, *declared_missing]))

    return fields
