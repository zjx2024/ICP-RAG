"""PostgreSQL authority checks for HugeGraph relationship traversal.

HugeGraph is only a rebuildable projection.  Every edge returned to the
diagnosis agent is therefore checked again against the current PostgreSQL
document and relation state.  Missing data, stale revisions and backend errors
all fail closed.

Two projection forms are supported:

* ``icp_structured`` edges must have a current ``icp_relation_assertions`` row
  for every document claimed by the physical HugeGraph edge;
* legacy FTA/RBD revision projections must still be the active, published,
  formally validated revision, and the exact edge must remain in that
  revision's canonical graph.

The callback deliberately does not use names, embeddings or fuzzy matching.
Those mechanisms may find traversal seeds, but they cannot authorize a graph
relationship.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from backend.icp.diagram_models import content_hash
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import FORMAL_VALIDATION_STATUSES
from backend.icp.entity_resolution import (
    clean_text,
    entity_resolution_repository,
)
from backend.storage.postgres_client import postgres_client


# Only a completely rebuilt document may authorize a diagnostic path.  During
# ``indexed``/``kg_pending`` the PostgreSQL assertions and HugeGraph projection
# can legitimately belong to different rebuild generations.
READY_DOCUMENT_STATUSES = frozenset({"kg_done"})
ENGINEERING_KB_IDS = frozenset({"fta", "rbd"})
STRUCTURED_KB_ID = "icp_structured"


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _strings(value: Any) -> list[str]:
    return sorted({clean_text(item) for item in _list(value) if clean_text(item)})


def _casefold(value: Any) -> str:
    return clean_text(value).casefold()


def _authorization_key(request: Mapping[str, Any]) -> str:
    return clean_text(
        request.get("authorization_key")
        or request.get("physical_edge_id")
        or request.get("logical_edge_id")
    )


def _revision_ids_from_assertion(assertion: Mapping[str, Any]) -> list[str]:
    """Read deterministic revision IDs from assertion provenance only.

    A HugeGraph request is untrusted projection data.  It may corroborate the
    asserted revision, but it can never select which revision PostgreSQL loads.
    """

    provenance = _mapping(assertion.get("provenance"))
    values: set[str] = set()
    for candidate in (
        provenance.get("revision_id"),
        _mapping(provenance.get("source_properties")).get("revision_id"),
        _mapping(provenance.get("source_properties")).get("diagram_revision_id"),
    ):
        value = clean_text(candidate)
        if value:
            values.add(value)
    for qualifier in _list(assertion.get("occurrence_qualifiers")):
        item = _mapping(qualifier)
        for key in ("revision_id", "diagram_revision_id"):
            value = clean_text(item.get(key))
            if value:
                values.add(value)
    return sorted(values)


def _revision_id_from_assertion(assertion: Mapping[str, Any]) -> str:
    values = _revision_ids_from_assertion(assertion)
    return values[0] if len(values) == 1 else ""


def _relation_assertion_matches(
    request: Mapping[str, Any], assertion: Mapping[str, Any], *, doc_id: str
) -> bool:
    if _casefold(assertion.get("assertion_status")) != "asserted":
        return False

    by_doc = _mapping(request.get("logical_edge_ids_by_doc"))
    requested_logical_ids = _strings(by_doc.get(doc_id))
    if not requested_logical_ids and len(_strings(request.get("doc_ids"))) == 1:
        requested_logical_ids = _strings(
            request.get("logical_edge_ids") or request.get("logical_edge_id")
        )
    asserted_logical_ids = set(_strings(assertion.get("logical_edge_ids")))
    if not requested_logical_ids or not set(requested_logical_ids).issubset(
        asserted_logical_ids
    ):
        return False

    provenance = _mapping(assertion.get("provenance"))
    source_relation = _casefold(provenance.get("source_relation"))
    canonical_relation = _casefold(assertion.get("relation_type"))
    requested_relation = _casefold(request.get("relation"))
    if not requested_relation or requested_relation not in {
        source_relation,
        canonical_relation,
    }:
        return False

    # Canonical physical endpoints bind all logical occurrences in the merged
    # SINGLE edge to this one PG assertion.  A representative logical endpoint
    # would be unsafe because canonical vertices may carry several occurrences.
    expected_source_physical = "icp:ce:" + clean_text(
        assertion.get("source_entity_id")
    )
    expected_target_physical = "icp:ce:" + clean_text(
        assertion.get("target_entity_id")
    )
    requested_source_physical = clean_text(request.get("source_physical_id"))
    requested_target_physical = clean_text(request.get("target_physical_id"))
    return bool(
        expected_source_physical != "icp:ce:"
        and expected_target_physical != "icp:ce:"
        and requested_source_physical == expected_source_physical
        and requested_target_physical == expected_target_physical
    )


def _canonical_edge(revision: Mapping[str, Any], logical_edge_id: str) -> Dict[str, Any]:
    graph = _mapping(revision.get("canonical_graph"))
    for raw in _list(graph.get("edges")):
        edge = _mapping(raw)
        if not edge:
            continue
        edge_id = clean_text(edge.get("id")) or content_hash(edge)[:24]
        if edge_id == logical_edge_id:
            return edge
    return {}


def _engineering_edge_matches(
    request: Mapping[str, Any], revision: Mapping[str, Any]
) -> bool:
    logical_edge_id = clean_text(request.get("logical_edge_id"))
    edge = _canonical_edge(revision, logical_edge_id)
    if not edge:
        return False
    return bool(
        _casefold(edge.get("label")) == _casefold(request.get("relation"))
        and clean_text(edge.get("source"))
        == clean_text(request.get("source_logical_id"))
        and clean_text(edge.get("target"))
        == clean_text(request.get("target_logical_id"))
    )


class GraphRelationAuthority:
    """Callable PostgreSQL relation reauthorizer used by graph traversal."""

    def __init__(
        self,
        *,
        entity_repository: Any = entity_resolution_repository,
        document_loader: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
        revision_loader: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self.entity_repository = entity_repository
        self.document_loader = document_loader or postgres_client.get_icp_document
        self.revision_loader = revision_loader or diagram_repository.get_revision

    @staticmethod
    def _document_ready(document: Mapping[str, Any], doc_id: str) -> bool:
        return bool(
            clean_text(document.get("doc_id")) == doc_id
            and _casefold(document.get("status")) in READY_DOCUMENT_STATUSES
        )

    @staticmethod
    def _revision_formal(
        revision: Mapping[str, Any],
        *,
        revision_id: str,
        doc_id: str,
        diagram_type: str,
    ) -> bool:
        return bool(
            clean_text(revision.get("revision_id")) == revision_id
            and clean_text(revision.get("doc_id")) == doc_id
            and _casefold(revision.get("diagram_type")) == diagram_type
            and revision.get("is_active") is True
            and _casefold(revision.get("projection_status")) == "published"
            and _casefold(revision.get("validation_status"))
            in FORMAL_VALIDATION_STATUSES
        )

    def _load_documents(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Mapping[str, Any]]:
        doc_ids = sorted(
            {
                doc_id
                for request in requests
                for doc_id in _strings(request.get("doc_ids"))
            }
        )
        result: Dict[str, Mapping[str, Any]] = {}
        for doc_id in doc_ids:
            document = self.document_loader(doc_id)
            if document is not None:
                result[doc_id] = _mapping(document)
        return result

    def _authorize_structured(
        self,
        request: Mapping[str, Any],
        documents: Mapping[str, Mapping[str, Any]],
        revisions: Dict[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        doc_ids = _strings(request.get("doc_ids"))
        if not doc_ids:
            return {"authorized": False}
        logical_ids_by_doc = _mapping(request.get("logical_edge_ids_by_doc"))
        if logical_ids_by_doc:
            mapped_doc_ids = {
                clean_text(key) for key in logical_ids_by_doc if clean_text(key)
            }
            mapped_logical_ids = {
                value
                for key in mapped_doc_ids
                for value in _strings(logical_ids_by_doc.get(key))
            }
            claimed_logical_ids = set(
                _strings(
                    request.get("logical_edge_ids")
                    or request.get("logical_edge_id")
                )
            )
            if (
                mapped_doc_ids != set(doc_ids)
                or not mapped_logical_ids
                or mapped_logical_ids != claimed_logical_ids
            ):
                return {"authorized": False}

        assertions: list[Mapping[str, Any]] = []
        source_refs: set[str] = set()
        validation_statuses: set[str] = set()
        assertion_revision_ids: set[str] = set()
        for doc_id in doc_ids:
            document = documents.get(doc_id) or {}
            if not self._document_ready(document, doc_id):
                return {"authorized": False}
            logical_ids_by_doc = _mapping(request.get("logical_edge_ids_by_doc"))
            requested_logical_ids = _strings(logical_ids_by_doc.get(doc_id))
            if not requested_logical_ids and len(doc_ids) == 1:
                requested_logical_ids = _strings(
                    request.get("logical_edge_ids")
                    or request.get("logical_edge_id")
                )
            if not requested_logical_ids:
                return {"authorized": False}
            assertion = self.entity_repository.find_relation_by_logical_edge_id(
                doc_id=doc_id,
                logical_edge_id=requested_logical_ids[0],
                formal_only=True,
            )
            if not assertion or not _relation_assertion_matches(
                request, assertion, doc_id=doc_id
            ):
                return {"authorized": False}
            assertion_id = clean_text(assertion.get("assertion_id"))
            if not assertion_id:
                return {"authorized": False}
            # Every marker owned by this document must resolve back to the same
            # canonical assertion.  Independent assertions may not be stitched
            # together merely because HugeGraph coalesced their physical edge.
            for logical_edge_id in requested_logical_ids[1:]:
                corroborating = (
                    self.entity_repository.find_relation_by_logical_edge_id(
                        doc_id=doc_id,
                        logical_edge_id=logical_edge_id,
                        formal_only=True,
                    )
                    or {}
                )
                if clean_text(corroborating.get("assertion_id")) != assertion_id:
                    return {"authorized": False}
            assertions.append(assertion)
            source_refs.update(_strings(assertion.get("source_ref")))

            asserted_doc_revision_ids = _revision_ids_from_assertion(assertion)
            revision_ids_by_doc = _mapping(request.get("revision_ids_by_doc"))
            requested_doc_revision_ids = _strings(revision_ids_by_doc.get(doc_id))
            requested_revision_id = clean_text(request.get("revision_id"))
            # A supplied projection marker must match PG provenance exactly.
            # It is not accepted as a fallback when provenance is absent.
            if requested_revision_id and requested_revision_id not in set(
                asserted_doc_revision_ids
            ):
                return {"authorized": False}
            if requested_doc_revision_ids and not set(
                requested_doc_revision_ids
            ).issubset(set(asserted_doc_revision_ids)):
                return {"authorized": False}
            assertion_revision_ids.update(asserted_doc_revision_ids)

            doc_type = _casefold(document.get("doc_type"))
            if doc_type in ENGINEERING_KB_IDS:
                if not asserted_doc_revision_ids:
                    return {"authorized": False}
                for revision_id in asserted_doc_revision_ids:
                    if revision_id not in revisions:
                        revisions[revision_id] = _mapping(
                            self.revision_loader(revision_id)
                        )
                    revision = revisions[revision_id]
                    if not self._revision_formal(
                        revision,
                        revision_id=revision_id,
                        doc_id=doc_id,
                        diagram_type=doc_type,
                    ):
                        return {"authorized": False}
                    validation_statuses.add(
                        _casefold(revision.get("validation_status"))
                    )

        status = (
            "manual_confirmed"
            if validation_statuses == {"manual_confirmed"}
            else "auto_validated"
        )
        return {
            "authorized": True,
            "validation_status": status,
            "projection_status": "published",
            "is_active": True,
            "doc_ids": doc_ids,
            "source_refs": sorted(source_refs),
            "assertion_ids": sorted(
                clean_text(item.get("assertion_id"))
                for item in assertions
                if clean_text(item.get("assertion_id"))
            ),
            "revision_ids": sorted(assertion_revision_ids),
            "revision_id": (
                sorted(assertion_revision_ids)[0]
                if len(assertion_revision_ids) == 1
                else ""
            ),
        }

    def _authorize_engineering(
        self,
        request: Mapping[str, Any],
        documents: Mapping[str, Mapping[str, Any]],
        revisions: Dict[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        kb_id = _casefold(request.get("kb_id"))
        doc_ids = _strings(request.get("doc_ids"))
        revision_id = clean_text(request.get("revision_id"))
        if kb_id not in ENGINEERING_KB_IDS or len(doc_ids) != 1 or not revision_id:
            return {"authorized": False}
        doc_id = doc_ids[0]
        document = documents.get(doc_id) or {}
        if not self._document_ready(document, doc_id):
            return {"authorized": False}
        if _casefold(document.get("doc_type")) != kb_id:
            return {"authorized": False}

        if revision_id not in revisions:
            revisions[revision_id] = _mapping(self.revision_loader(revision_id))
        revision = revisions[revision_id]
        if not self._revision_formal(
            revision,
            revision_id=revision_id,
            doc_id=doc_id,
            diagram_type=kb_id,
        ) or not _engineering_edge_matches(request, revision):
            return {"authorized": False}

        edge = _canonical_edge(revision, clean_text(request.get("logical_edge_id")))
        properties = _mapping(edge.get("properties"))
        source_refs = _strings(
            properties.get("source_refs") or properties.get("source_ref")
        )
        return {
            "authorized": True,
            "validation_status": _casefold(revision.get("validation_status")),
            "projection_status": "published",
            "is_active": True,
            "doc_ids": [doc_id],
            "source_refs": source_refs,
            "revision_id": revision_id,
        }

    def __call__(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Mapping[str, Any]]:
        """Authorize a batch; malformed individual requests are denied.

        Storage exceptions are intentionally not swallowed.  The traversal
        layer catches them and returns no partial graph result.
        """

        normalized = [item for item in requests if isinstance(item, Mapping)]
        documents = self._load_documents(normalized)
        revisions: Dict[str, Mapping[str, Any]] = {}
        decisions: Dict[str, Mapping[str, Any]] = {}
        for request in normalized:
            key = _authorization_key(request)
            if not key:
                continue
            kb_id = _casefold(request.get("kb_id"))
            if kb_id == STRUCTURED_KB_ID:
                decision = self._authorize_structured(request, documents, revisions)
            elif kb_id in ENGINEERING_KB_IDS:
                decision = self._authorize_engineering(request, documents, revisions)
            else:
                decision = {"authorized": False}
            decisions[key] = decision
        return decisions


graph_relation_authority = GraphRelationAuthority()


def reauthorize_graph_relations(
    requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Process-wide callback for API and LangGraph integration."""

    return graph_relation_authority(requests)


__all__ = [
    "ENGINEERING_KB_IDS",
    "GraphRelationAuthority",
    "READY_DOCUMENT_STATUSES",
    "STRUCTURED_KB_ID",
    "graph_relation_authority",
    "reauthorize_graph_relations",
]
