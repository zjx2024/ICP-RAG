"""ICP 七类文档图谱投影服务（HugeGraph）。

通用知识库时代的 GraphExtractor 抽取、跨文档实体合并与社区报告生成已随旧链路
退役；本模块只保留 FTA/RBD/架构图经类型校验后的版本化投影。
"""
import asyncio
import copy
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from backend.core.task_manager import task_manager
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.postgres_client import postgres_client
from backend.icp.diagram_models import content_hash
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import (
    VALIDATION_QUARANTINED,
    revision_is_projectable,
)


# ===================== ICP 图类（fta/rbd）图谱生成 =====================

_ICP_VERTEX_PROPERTIES = frozenset(
    {
        "name",
        "description",
        "summary",
        "weight",
        "level",
        "probability",
        "keywords",
    }
)
_ICP_EDGE_PROPERTIES = frozenset(
    {
        "description",
        "weight",
        "probability",
        "gate_type",
        "gate_group",
        "threshold",
        "keywords",
    }
)
_PROJECTABLE_REVISION_STATES = frozenset(
    {"not_projected", "publishing", "failed", "stale"}
)


def select_latest_publishable_revisions(
    repository,
    *,
    doc_id: str,
    doc_type: str,
    requested_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Select each diagram's latest machine- or human-validated revision.

    Explicit IDs are fail-closed: an older, unknown, unvalidated, or otherwise
    unpublishable revision rejects the whole request instead of being silently
    ignored. With no explicit IDs, quarantined diagrams are left isolated while
    other latest revisions may be published independently.
    """

    requested = {
        str(value).strip() for value in (requested_ids or []) if str(value).strip()
    }
    latest: List[Dict] = []
    for diagram in repository.list_diagrams_by_document(doc_id):
        if str(diagram.get("diagram_type") or doc_type) != doc_type:
            continue
        revisions = repository.list_revisions(str(diagram["diagram_id"]))
        if not revisions:
            continue
        item = dict(revisions[0])
        item.setdefault("diagram_id", diagram["diagram_id"])
        item.setdefault("doc_id", doc_id)
        item.setdefault("diagram_type", doc_type)
        latest.append(item)

    latest_by_id = {str(item.get("revision_id")): item for item in latest}
    if requested:
        invalid = sorted(requested - set(latest_by_id))
        if invalid:
            raise ValueError(
                "只能发布每张工程图的最新修订；以下修订不是当前最新版本: "
                + ", ".join(invalid)
            )
        candidates = [item for item in latest if str(item.get("revision_id")) in requested]
    else:
        candidates = latest

    selected: List[Dict] = []
    invalid_states: List[str] = []
    for item in candidates:
        revision_id = str(item.get("revision_id") or "")
        if not revision_is_projectable(item):
            if requested:
                invalid_states.append(f"{revision_id}: 未确认或未通过自动校验")
            continue
        projection_status = str(item.get("projection_status") or "")
        if projection_status == "published":
            # An explicitly retried active revision is a read-back-only
            # idempotent request.  Normal bulk publication skips it.
            if requested and item.get("is_active"):
                selected.append(item)
            elif requested:
                invalid_states.append(f"{revision_id}: published 但非 active")
            continue
        if projection_status not in _PROJECTABLE_REVISION_STATES:
            if requested:
                invalid_states.append(f"{revision_id}: {projection_status or '未知状态'}")
            continue
        selected.append(item)

    if invalid_states:
        raise ValueError("图修订不可发布: " + "; ".join(invalid_states))
    return selected


def select_latest_confirmed_revisions(
    repository,
    *,
    doc_id: str,
    doc_type: str,
    requested_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Backward-compatible name for the publishable-revision selector."""

    return select_latest_publishable_revisions(
        repository,
        doc_id=doc_id,
        doc_type=doc_type,
        requested_ids=requested_ids,
    )


def projection_item_revision_id(item: Dict) -> str:
    """Read the revision marker using only existing HugeGraph schema fields."""

    properties = item.get("properties") or {}
    if not isinstance(properties, dict):
        return ""
    revision_id = str(properties.get("tenant_id") or "").strip()
    if revision_id:
        return revision_id
    source_ids = properties.get("source_id") or []
    if isinstance(source_ids, str):
        source_ids = [part for part in source_ids.split(",") if part]
    for value in source_ids if isinstance(source_ids, list) else []:
        text = str(value)
        if text.startswith("revision:"):
            return text.split(":", 1)[1]
    return ""


def _revision_to_hugegraph(revision: Dict) -> tuple[List[Dict], List[Dict]]:
    """Convert one validated canonical revision using deterministic identities.

    The existing HugeGraph schema is intentionally not expanded here.  Full
    engineering attributes remain authoritative in PostgreSQL; this projection
    keeps only schema-supported search/topology properties plus revision markers
    encoded in ``project_id``/``tenant_id``/``source_id``.
    """
    graph = revision.get("canonical_graph") or {}
    revision_id = str(revision.get("revision_id") or "")
    diagram_id = str(revision.get("diagram_id") or "")
    doc_id = str(revision.get("doc_id") or "")
    doc_type = str(revision.get("diagram_type") or "")
    canonical_hash = str(revision.get("canonical_hash") or "")
    created_at = str(revision.get("created_at") or "")
    validation_status = str(revision.get("validation_status") or "")
    validation_tier = str(revision.get("validation_tier") or "")
    source_markers = [
        doc_id,
        f"diagram:{diagram_id}",
        f"revision:{revision_id}",
        f"canonical:{canonical_hash}",
        f"validation:{validation_status or 'legacy_manual'}",
        f"validation_tier:{validation_tier or 'human_review'}",
    ]
    vertices: List[Dict] = []
    id_map: Dict[str, str] = {}
    for node in graph.get("vertices") or []:
        if not isinstance(node, dict) or not node.get("id"):
            continue
        source_id = str(node["id"])
        vertex_id = f"icp:{revision_id}:v:{source_id}"
        id_map[source_id] = vertex_id
        raw_properties = copy.deepcopy(node.get("properties") or {})
        properties = {
            key: value
            for key, value in raw_properties.items()
            if key in _ICP_VERTEX_PROPERTIES and value is not None
        }
        properties.update(
            {
                "name": str(properties.get("name") or source_id),
                "description": str(properties.get("description") or ""),
                "doc_id": [doc_id],
                "kb_id": doc_type,
                "project_id": diagram_id,
                "tenant_id": revision_id,
                "source_id": source_markers,
                "entity_type": str(node.get("label") or "EngineeringNode"),
                "created_at": created_at,
            }
        )
        vertices.append({"id": vertex_id, "label": "Entity", "properties": properties})

    edges: List[Dict] = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source = id_map.get(str(edge.get("source") or ""))
        target = id_map.get(str(edge.get("target") or ""))
        if not source or not target:
            continue
        source_edge_id = str(edge.get("id") or content_hash(edge)[:24])
        raw_properties = copy.deepcopy(edge.get("properties") or {})
        properties = {
            key: value
            for key, value in raw_properties.items()
            if key in _ICP_EDGE_PROPERTIES and value is not None
        }
        if not properties.get("description"):
            connection_type = raw_properties.get("connection_type")
            properties["description"] = str(connection_type or edge.get("label") or "")
        properties.update(
            {
                "weight": properties.get("weight", 1),
                "doc_id": [doc_id],
                "kb_id": doc_type,
                "project_id": diagram_id,
                "tenant_id": revision_id,
                "source_id": source_markers,
                "created_at": created_at,
            }
        )
        if isinstance(properties.get("gate_type"), dict):
            properties["gate_type"] = properties["gate_type"].get("value", "direct")
        edges.append(
            {
                "id": f"icp:{revision_id}:e:{source_edge_id}",
                "label": str(edge.get("label") or "relation"),
                "outV": source,
                "inV": target,
                "outVLabel": "Entity",
                "inVLabel": "Entity",
                "properties": properties,
            }
        )
    return vertices, edges


def _projected_revision_count(items: List[Dict], revision_id: str) -> int:
    return sum(1 for item in items if projection_item_revision_id(item) == revision_id)


def _remove_revision_projection(client, items: List[Dict], revision_id: str, *, edges: bool) -> None:
    """Remove a partial inactive revision namespace before an idempotent retry."""

    delete = client.delete_edge if edges else client.delete_vertex
    kind = "edge" if edges else "vertex"
    for item in items:
        if projection_item_revision_id(item) != revision_id:
            continue
        item_id = str(item.get("id") or "")
        if not item_id or not delete(item_id):
            raise RuntimeError(f"无法清理重试中的 HugeGraph {kind}: {item_id or '<missing>'}")


async def generate_icp_kg_background(
    task_id: str,
    doc_id: str,
    doc_type: str,
    file_name: str,
    *,
    revision_ids: Optional[List[str]] = None,
    structured_result: Optional[Dict] = None,
):
    """Publish validated revisions and activate only verified projections."""
    running_projection: Optional[tuple[str, str]] = None
    try:
        loop = asyncio.get_event_loop()
        await task_manager.update_status(task_id, "generating_kg")
        await task_manager.update_progress(task_id, 10, "[START] 开始发布已自动校验的工程图修订")

        if getattr(hugegraph_client, "_use_mock", False):
            raise Exception("HugeGraph 服务不可用；拒绝将内存 Mock 视为真实投影成功")

        ids = list(dict.fromkeys(str(value) for value in (revision_ids or []) if str(value)))
        if not ids:
            raise Exception("未指定可发布图修订")

        revisions: List[Dict] = []
        for revision_id in ids:
            revision = await loop.run_in_executor(None, diagram_repository.get_revision, revision_id)
            if not revision:
                raise Exception(f"图修订不存在: {revision_id}")
            if revision.get("doc_id") != doc_id or revision.get("diagram_type") != doc_type:
                raise Exception(f"图修订不属于当前文档: {revision_id}")
            if not revision_is_projectable(revision):
                raise Exception(f"图修订未达到可发布校验等级: {revision_id}")
            revisions.append(revision)

        total_vertices = 0
        total_edges = 0
        published_ids: List[str] = []
        for index, revision in enumerate(revisions, start=1):
            revision_id = str(revision["revision_id"])
            vertices, edges = _revision_to_hugegraph(revision)
            if not vertices:
                raise Exception(f"已校验图修订没有有效节点: {revision_id}")
            progress = 15 + int((index - 1) / max(len(revisions), 1) * 75)
            await task_manager.update_progress(
                task_id,
                progress,
                f"[REVISION {index}/{len(revisions)}] 发布 {revision.get('title') or revision_id}: "
                f"{len(vertices)} 节点 / {len(edges)} 边",
            )
            idempotency_key = (
                f"hugegraph:{revision_id}:{revision.get('canonical_hash')}:projector-v1"
            )
            run = await loop.run_in_executor(
                None,
                partial(
                    diagram_repository.start_projection,
                    revision_id,
                    idempotency_key=idempotency_key,
                    projector_version="projector-v1",
                    expected_vertex_count=len(vertices),
                    expected_edge_count=len(edges),
                ),
            )
            # Once PG has transitioned a non-active revision/run to
            # publishing/running, every later failure—including the first
            # HugeGraph read—must close that run through fail_projection.
            if not run.get("already_active"):
                running_projection = (revision_id, str(run["run_id"]))
            stored_vertices = await loop.run_in_executor(
                None, hugegraph_client.get_vertices_by_kb_with_details, doc_type
            )
            stored_edges = await loop.run_in_executor(
                None, hugegraph_client.get_edges_by_kb_with_details, doc_type
            )
            if run.get("already_active"):
                actual_vertices = _projected_revision_count(stored_vertices, revision_id)
                actual_edges = _projected_revision_count(stored_edges, revision_id)
                if actual_vertices != len(vertices) or actual_edges != len(edges):
                    verification_error = (
                        f"已发布图修订的 HugeGraph 读回数量不一致: {revision_id} "
                        f"({actual_vertices}/{len(vertices)} 节点, {actual_edges}/{len(edges)} 边)"
                    )
                    # A succeeded run is not sufficient proof once its physical
                    # data drifts.  Close the authoritative active gate before
                    # surfacing the failure; a retry can rebuild the same
                    # immutable revision with the same idempotency key.
                    await loop.run_in_executor(
                        None,
                        partial(
                            diagram_repository.invalidate_active_projection,
                            revision_id,
                            run_id=str(run["run_id"]),
                            error=verification_error,
                        ),
                    )
                    raise Exception(verification_error)
                published_ids.append(revision_id)
                total_vertices += actual_vertices
                total_edges += actual_edges
                continue

            existing_vertices = _projected_revision_count(stored_vertices, revision_id)
            existing_edges = _projected_revision_count(stored_edges, revision_id)
            if existing_vertices or existing_edges:
                # A crashed/failed attempt may have left a partial namespace.
                # Edges are removed before vertices; other revisions are never
                # touched because physical IDs and markers are revision-scoped.
                await loop.run_in_executor(
                    None,
                    partial(
                        _remove_revision_projection,
                        hugegraph_client,
                        stored_edges,
                        revision_id,
                        edges=True,
                    ),
                )
                await loop.run_in_executor(
                    None,
                    partial(
                        _remove_revision_projection,
                        hugegraph_client,
                        stored_vertices,
                        revision_id,
                        edges=False,
                    ),
                )

            ok = await loop.run_in_executor(None, hugegraph_client.batch_insert, vertices, edges)
            if not ok:
                raise Exception(f"HugeGraph 写入失败: {revision_id}")

            stored_vertices = await loop.run_in_executor(
                None, hugegraph_client.get_vertices_by_kb_with_details, doc_type
            )
            stored_edges = await loop.run_in_executor(
                None, hugegraph_client.get_edges_by_kb_with_details, doc_type
            )
            actual_vertices = _projected_revision_count(stored_vertices, revision_id)
            actual_edges = _projected_revision_count(stored_edges, revision_id)
            if actual_vertices != len(vertices) or actual_edges != len(edges):
                raise Exception(
                    f"HugeGraph 写后读回数量不一致: {revision_id} "
                    f"({actual_vertices}/{len(vertices)} 节点, {actual_edges}/{len(edges)} 边)"
                )
            manifest = {
                "canonical_hash": revision.get("canonical_hash"),
                "revision_id": revision_id,
                "vertex_ids": sorted(str(item.get("id")) for item in vertices),
                "edge_ids": sorted(str(item.get("id")) for item in edges),
            }
            manifest["manifest_hash"] = content_hash(manifest)
            await loop.run_in_executor(
                None,
                partial(
                    diagram_repository.activate_projection,
                    revision_id,
                    run_id=str(run["run_id"]),
                    manifest=manifest,
                    actual_vertex_count=actual_vertices,
                    actual_edge_count=actual_edges,
                ),
            )
            running_projection = None
            published_ids.append(revision_id)
            total_vertices += actual_vertices
            total_edges += actual_edges

        # The common ``icp_structured`` graph is the cross-document traversal
        # surface.  Its first pass runs before validation/publication and thus
        # cannot contain a newly activated FTA/RBD revision.  Rebuild it only
        # after every revision activation has passed HugeGraph read-back so a
        # provisional or half-written topology can never leak into formal
        # multi-hop queries.
        # This lower-level publisher is also unit-tested/reused independently.
        # The seven-type pipeline always supplies the pre-publication result;
        # its presence identifies the complete workflow which owns the common
        # projection refresh.
        if structured_result is not None:
            await task_manager.update_progress(
                task_id,
                92,
                "[SYNC] 正在更新正式工程图的跨文档关系",
            )
            from backend.icp.structured_projection import project_structured_document

            structured_result = await loop.run_in_executor(
                None, project_structured_document, doc_id, doc_type
            )

        lifecycle = _diagram_lifecycle_summary(doc_id, doc_type)
        final_doc_status = "kg_done" if lifecycle["terminal"] else "kg_pending"
        _record_icp_graph_terminal_state(
            doc_id,
            final_doc_status,
            lifecycle=lifecycle,
        )

        await task_manager.update_progress(
            task_id,
            100,
            f"[FINISH] 已验证并激活 {len(published_ids)} 个图修订："
            f"{total_vertices} 节点 / {total_edges} 边",
        )
        await task_manager.update_status(task_id, "completed")
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "revision_ids": published_ids,
            "vertices": total_vertices,
            "edges": total_edges,
            "status": final_doc_status,
            "diagram_lifecycle": lifecycle,
            "structured_projection": structured_result or {},
            "message": f"知识图谱投影完成：{total_vertices} 节点 / {total_edges} 边",
        })

    except asyncio.CancelledError:
        if running_projection:
            revision_id, run_id = running_projection
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(
                        diagram_repository.fail_projection,
                        revision_id,
                        run_id=run_id,
                        error="projection task cancelled",
                    ),
                )
            except Exception:
                pass
        await task_manager.update_progress(task_id, 0, "[CANCEL] 知识图谱生成已被用户取消", log_type="cancel")
        await task_manager.update_status(task_id, "cancelled")
        _record_icp_graph_terminal_state(
            doc_id,
            "kg_pending",
            error="projection task cancelled",
        )

    except Exception as e:
        import traceback
        if running_projection:
            revision_id, run_id = running_projection
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(
                        diagram_repository.fail_projection,
                        revision_id,
                        run_id=run_id,
                        error=str(e),
                    ),
                )
            except Exception:
                pass
        error_detail = f"[ERROR] {str(e)}\n\n{traceback.format_exc()}"
        await task_manager.set_error(task_id, error_detail)
        await task_manager.update_progress(task_id, 0, f"[ERROR] 知识图谱生成失败: {str(e)}", log_type="error")
        await task_manager.update_status(task_id, "failed")
        _record_icp_graph_terminal_state(
            doc_id,
            "kg_pending",
            error=str(e),
        )


_ICP_DIAGRAM_TYPES = frozenset({"function_locator", "fta", "rbd"})


def _record_icp_graph_terminal_state(
    doc_id: str,
    status: str,
    *,
    lifecycle: Optional[Mapping[str, Any]] = None,
    error: str = "",
) -> None:
    """Keep the document status and its asynchronous parse summary aligned."""

    postgres_client.update_icp_document_status(doc_id, status)
    graph_state: Dict[str, Any] = {"status": status}
    if lifecycle is not None:
        graph_state["diagram_lifecycle"] = dict(lifecycle)
    if error:
        graph_state["error"] = str(error)[:1000]
    postgres_client.patch_icp_document_result(
        doc_id,
        {
            "kg_pending": status != "kg_done",
            "graph_projection": graph_state,
        },
    )


def _diagram_lifecycle_summary(doc_id: str, doc_type: str) -> Dict[str, Any]:
    """Classify every discovered diagram without inferring missing state.

    A diagram master is written before visual extraction.  Consequently, no
    revision can mean a model/response parsing failure just as easily as an
    image without explicit topology.  Until the parser persists a positive
    terminal outcome, absence of a revision must remain fail-closed pending.
    Published active revisions and quarantined revisions are terminal.
    """

    summary: Dict[str, Any] = {
        "total": 0,
        "published": 0,
        "quarantined": 0,
        # Retained for response-shape compatibility.  This count may only be
        # advanced by a future explicit, persisted no-structure outcome.
        "no_explicit_structure": 0,
        "unresolved_without_revision": 0,
        "pending": 0,
        "terminal": True,
    }
    for diagram in diagram_repository.list_diagrams_by_document(doc_id):
        if str(diagram.get("diagram_type") or "") != doc_type:
            continue
        summary["total"] += 1
        revisions = diagram_repository.list_revisions(str(diagram["diagram_id"]))
        if not revisions:
            summary["unresolved_without_revision"] += 1
            summary["pending"] += 1
            summary["terminal"] = False
            continue
        latest = revisions[0]
        if (
            revision_is_projectable(latest)
            and latest.get("projection_status") == "published"
            and latest.get("is_active")
        ):
            summary["published"] += 1
            continue
        if str(latest.get("validation_status") or "") == VALIDATION_QUARANTINED:
            summary["quarantined"] += 1
            continue
        summary["pending"] += 1
        summary["terminal"] = False
    return summary


def _latest_diagrams_are_terminal(doc_id: str, doc_type: str) -> bool:
    """Return true when each discovered diagram is published or quarantined.

    A document with no diagram masters is a terminal, relation-sparse result:
    the common Document vertex and PostgreSQL explicit relations are complete.
    A discovered master without a revision is unresolved and remains pending;
    missing engineering topology is never fabricated to fill it.
    """

    return bool(_diagram_lifecycle_summary(doc_id, doc_type)["terminal"])


async def generate_icp_graph_background(
    task_id: str,
    doc_id: str,
    doc_type: str,
    file_name: str,
    *,
    requested_revision_ids: Optional[List[str]] = None,
):
    """Project one of the seven ICP document types without a human gate.

    All types first publish their common Document vertex and deterministic
    PostgreSQL relationships.  Architecture/FTA/RBD revisions then pass the
    versioned automatic validator.  Publishable revisions are written with
    read-back verification; invalid revisions remain quarantined and cannot be
    used as formal diagnosis or calculation evidence.
    """

    loop = asyncio.get_event_loop()
    try:
        await task_manager.update_status(task_id, "generating_kg")
        await task_manager.update_progress(
            task_id, 5, "[START] 开始自动生成显式关系图投影"
        )
        from backend.icp.structured_projection import project_structured_document

        structured_result = await loop.run_in_executor(
            None, project_structured_document, doc_id, doc_type
        )
        await task_manager.update_progress(
            task_id,
            25,
            "[SUCCESS] PostgreSQL 显式关系已投影："
            f"{structured_result.get('vertices', 0)} 节点 / "
            f"{structured_result.get('edges', 0)} 边",
        )

        if doc_type not in _ICP_DIAGRAM_TYPES:
            _record_icp_graph_terminal_state(doc_id, "kg_done")
            await task_manager.update_progress(
                task_id, 100, "[FINISH] 结构化显式关系图投影完成"
            )
            await task_manager.update_status(task_id, "completed")
            await task_manager.set_result(
                task_id,
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "status": "kg_done",
                    "structured_projection": structured_result,
                    "revision_ids": [],
                    "vertices": structured_result.get("vertices", 0),
                    "edges": structured_result.get("edges", 0),
                    "message": "结构化显式关系已自动投影到 HugeGraph",
                },
            )
            return

        await task_manager.update_progress(
            task_id, 30, "[INFO] 开始执行工程图类型校验与证据分级"
        )
        validation = await loop.run_in_executor(
            None,
            partial(
                diagram_repository.auto_validate_document,
                doc_id,
                diagram_type=doc_type,
            ),
        )
        if validation.get("errors"):
            reasons = "; ".join(
                str(item.get("reason") or "自动校验失败")
                for item in validation["errors"]
            )
            raise RuntimeError(f"部分工程图自动校验执行失败: {reasons}")

        selected = select_latest_publishable_revisions(
            diagram_repository,
            doc_id=doc_id,
            doc_type=doc_type,
            requested_ids=requested_revision_ids,
        )
        revision_ids = [str(item["revision_id"]) for item in selected]
        if revision_ids:
            await generate_icp_kg_background(
                task_id,
                doc_id,
                doc_type,
                file_name,
                revision_ids=revision_ids,
                structured_result=structured_result,
            )
            return

        if not _latest_diagrams_are_terminal(doc_id, doc_type):
            raise RuntimeError("工程图尚未形成可发布或可安全隔离的自动校验结果")

        quarantined = list(validation.get("quarantined_revision_ids") or [])
        lifecycle = _diagram_lifecycle_summary(doc_id, doc_type)
        _record_icp_graph_terminal_state(
            doc_id,
            "kg_done",
            lifecycle=lifecycle,
        )
        await task_manager.update_progress(
            task_id,
            100,
            "[FINISH] 自动图处理完成"
            + (f"，{len(quarantined)} 个候选已隔离" if quarantined else ""),
        )
        await task_manager.update_status(task_id, "completed")
        await task_manager.set_result(
            task_id,
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "status": "kg_done",
                "structured_projection": structured_result,
                "automatic_validation": validation,
                "revision_ids": [],
                "vertices": structured_result.get("vertices", 0),
                "edges": structured_result.get("edges", 0),
                "message": "自动图处理完成；未凭空补造缺失关系",
            },
        )
    except asyncio.CancelledError:
        await task_manager.update_progress(
            task_id, 0, "[CANCEL] 自动图处理已取消", log_type="cancel"
        )
        await task_manager.update_status(task_id, "cancelled")
        _record_icp_graph_terminal_state(
            doc_id,
            "kg_pending",
            error="projection task cancelled",
        )
    except Exception as exc:
        import traceback

        await task_manager.set_error(
            task_id, f"[ERROR] {exc}\n\n{traceback.format_exc()}"
        )
        await task_manager.update_progress(
            task_id, 0, f"[ERROR] 自动图处理失败: {exc}", log_type="error"
        )
        await task_manager.update_status(task_id, "failed")
        _record_icp_graph_terminal_state(
            doc_id,
            "kg_pending",
            error=str(exc),
        )
