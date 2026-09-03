#!/usr/bin/env python3
"""Rebuild ICP canonical entities, relation assertions and graph projections.

PostgreSQL rows and approved diagram revisions are the authority.  HugeGraph,
the entity registry and semantic candidates are rebuildable projections.  The
script never promotes a semantic candidate to a formal identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DOC_TYPE_ORDER = (
    "function_locator",
    "hmc",
    "fmeca",
    "mfl",
    "fta",
    "rbd",
    "fault_history",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def load_documents(connection: Any, doc_ids: Sequence[str]) -> List[Dict[str, Any]]:
    filters = ""
    params: List[Any] = []
    normalized_ids = sorted({_text(value) for value in doc_ids if _text(value)})
    if normalized_ids:
        filters = "AND doc_id = ANY(%s)"
        params.append(normalized_ids)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT doc_id, doc_type, file_name, status, metadata
            FROM icp_documents
            WHERE status IN ('indexed','kg_pending','kg_done')
              {filters}
            ORDER BY CASE doc_type
                WHEN 'function_locator' THEN 1
                WHEN 'hmc' THEN 2
                WHEN 'fmeca' THEN 3
                WHEN 'mfl' THEN 4
                WHEN 'fta' THEN 5
                WHEN 'rbd' THEN 6
                WHEN 'fault_history' THEN 7
                ELSE 99 END,
                created_at, doc_id
            """,
            tuple(params),
        )
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
    return [
        dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))
        for row in rows
        if _text((row.get("doc_type") if isinstance(row, Mapping) else row[1]))
        in DOC_TYPE_ORDER
    ]


def rebuild(
    documents: Sequence[Mapping[str, Any]],
    *,
    project_hugegraph: bool,
    discover_candidates: bool,
    embedding_model_id: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    from backend.icp.entity_candidates import entity_candidate_discovery_service
    from backend.icp.entity_resolution import entity_resolution_repository
    from backend.icp.structured_projection import (
        build_document_graph,
        project_structured_document,
    )

    entity_resolution_repository.init_tables()
    projection_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for document in documents:
        doc_id = _text(document.get("doc_id"))
        doc_type = _text(document.get("doc_type")).casefold()
        try:
            graph = build_document_graph(doc_id, doc_type)
            if dry_run:
                result = {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "vertices": len(graph.get("vertices") or []),
                    "edges": len(graph.get("edges") or []),
                    "status": "dry_run",
                }
            elif project_hugegraph:
                # The projection pipeline also synchronizes PostgreSQL entity
                # authority before publishing the HugeGraph snapshot.
                result = project_structured_document(doc_id, doc_type)
            else:
                metadata = document.get("metadata") or {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                result = entity_resolution_repository.sync_structured_graph(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    graph=graph,
                    document_metadata=(
                        metadata if isinstance(metadata, Mapping) else {}
                    ),
                )
            projection_results.append(dict(result))
        except Exception as exc:
            errors.append(
                {"doc_id": doc_id, "doc_type": doc_type, "error": str(exc)}
            )

    candidate_results: List[Dict[str, Any]] = []
    if discover_candidates and not dry_run and not errors:
        for document in documents:
            doc_id = _text(document.get("doc_id"))
            try:
                candidate_results.append(
                    entity_candidate_discovery_service.discover_for_document(
                        doc_id,
                        embedding_model_id=embedding_model_id or None,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "doc_id": doc_id,
                        "doc_type": _text(document.get("doc_type")),
                        "error": f"candidate discovery failed: {exc}",
                    }
                )

    return {
        "status": "completed" if not errors else "failed",
        "document_count": len(documents),
        "projection_mode": (
            "dry_run"
            if dry_run
            else "postgres_and_hugegraph"
            if project_hugegraph
            else "postgres_registry_only"
        ),
        "candidate_discovery": bool(discover_candidates and not dry_run),
        "projection_results": projection_results,
        "candidate_results": candidate_results,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument(
        "--project-hugegraph",
        action="store_true",
        help="also replace the rebuildable HugeGraph structured projection",
    )
    parser.add_argument(
        "--discover-candidates",
        action="store_true",
        help="use the configured BGE embedding model for candidate-only matches",
    )
    parser.add_argument("--embedding-model-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _compact_cli_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep operational output readable without printing thousands of ID pairs."""

    compact = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    for projection in compact.get("projection_results") or []:
        if not isinstance(projection, dict):
            continue
        snapshots = [projection]
        nested = projection.get("entity_resolution")
        if isinstance(nested, dict):
            snapshots.append(nested)
        for snapshot in snapshots:
            logical_ids = snapshot.pop("logical_entity_ids", None)
            if isinstance(logical_ids, dict):
                snapshot["logical_entity_id_count"] = len(logical_ids)
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from backend.storage.postgres_client import postgres_client

    documents = load_documents(postgres_client.conn, args.doc_id)
    result = rebuild(
        documents,
        project_hugegraph=args.project_hugegraph,
        discover_candidates=args.discover_candidates,
        embedding_model_id=args.embedding_model_id,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            _compact_cli_result(result),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
