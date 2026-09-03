#!/usr/bin/env python3
"""Rebuild and audit the OpenSearch search projection from PostgreSQL.

PostgreSQL and the original artifacts are the sources of truth.  This module
therefore deliberately has no application imports at module-import time: DB,
embedding and OpenSearch dependencies are resolved only after ``main`` has
validated the destructive flags, or are injected by unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


# When invoked exactly as documented (``python scripts/reindex_opensearch.py``),
# Python puts ``scripts/`` rather than the repository root on ``sys.path``.
# Resolve the sibling ``backend`` package without relying on PYTHONPATH or the
# caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_QUERIES: Tuple[Tuple[str, str], ...] = (
    (
        "document_chunks",
        """
        SELECT chunk_id, doc_id, section, title, content, content_type,
               chunk_index, selected_for_kg, created_at
        FROM document_chunks
        ORDER BY doc_id, chunk_index, chunk_id
        """,
    ),
    (
        "fault_hmc",
        """
        SELECT id, doc_id, variant_id, hmc_code, lcn, hamc, hsc, fault_desc,
               repair_method, source_ref, created_at, updated_at
        FROM fault_hmc
        ORDER BY doc_id, id
        """,
    ),
    (
        "fmea",
        """
        SELECT id, doc_id, code, product_flag, func_management, fault_mode,
               fault_cause, mission_phase, local_impact, final_impact,
               severity, severity_raw, severity_normalized,
               probability, probability_raw, probability_level,
               detect_method, design_improve,
               compensate_action, created_at, updated_at
        FROM fmea
        ORDER BY doc_id, id
        """,
    ),
    (
        "mfl",
        """
        SELECT id, doc_id, mfl_code, module_fault, sur_fault,
               subcircuit_fault, api_reportable, api_name, api_method,
               api_usage, platform_test, platform_test_raw,
               platform_test_status, maintenance_action,
               created_at, updated_at
        FROM mfl
        ORDER BY doc_id, id
        """,
    ),
    (
        "icp_functions",
        """
        SELECT id, doc_id, function_id, system_name, subsystem_name,
               module_name, function_name, description, aliases,
               failure_symptoms, source_ref, created_at, updated_at
        FROM icp_functions
        ORDER BY doc_id, id
        """,
    ),
    (
        "fault_history_cases",
        """
        SELECT c.id, c.doc_id, c.case_id, c.occurred_at, c.personnel,
               c.device_type, c.fault_symptom, c.related_module,
               c.diagnosis_method, c.root_cause, c.final_solution,
               c.preventive_action, c.hmc_code, c.fmeca_code, c.fta_ref,
               c.mfl_code, c.source_ref, c.created_at, c.updated_at,
               COALESCE(
                   json_agg(s.content ORDER BY s.step_no)
                       FILTER (WHERE s.id IS NOT NULL),
                   '[]'::json
               ) AS steps
        FROM fault_history_cases c
        LEFT JOIN fault_history_steps s ON s.case_pk = c.id
        GROUP BY c.id
        ORDER BY c.doc_id, c.id
        """,
    ),
    (
        "document_fault_trees",
        """
        SELECT id, doc_id, kb_id, chunk_id, image_id, image_title, graph,
               created_at
        FROM document_fault_trees
        ORDER BY doc_id, id
        """,
    ),
    (
        "community_reports",
        """
        SELECT projection_id, report_id, community_id, kb_id, doc_id,
               doc_ids, entities, report, stale_kwd, stale_reason, stale_at,
               created_at, updated_at
        FROM community_reports
        ORDER BY kb_id, projection_id
        """,
    ),
)


ICP_DOCUMENTS_QUERY = """
    SELECT doc_id, doc_type, file_name, metadata, created_at, updated_at
    FROM icp_documents
    ORDER BY doc_id
"""


LEGACY_DOCUMENTS_QUERY = """
    SELECT doc_id, kb_id, file_name, status, created_at, updated_at
    FROM documents
    WHERE COALESCE(is_deleted, FALSE) = FALSE
    ORDER BY doc_id
"""


RowReader = Callable[[Any, str, int], Iterable[Mapping[str, Any]]]
MetadataTagger = Callable[[Dict[str, Any]], Dict[str, Any]]
Projection = Tuple[str, Dict[str, Any], str]


def iter_query(connection: Any, query: str, batch_size: int) -> Iterator[Dict[str, Any]]:
    """Read rows in bounded batches without requiring a specific cursor factory."""

    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                if isinstance(row, Mapping):
                    yield dict(row)
                else:
                    yield dict(zip(columns, row))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PostgreSQL JSON 字段格式无效: {exc}") from exc
    return value


def _json_dict(value: Any) -> Dict[str, Any]:
    decoded = _json_value(value, {})
    if not isinstance(decoded, Mapping):
        raise ValueError("PostgreSQL JSON 对象字段不是 object")
    return dict(decoded)


def _json_list(value: Any) -> List[Any]:
    decoded = _json_value(value, [])
    if not isinstance(decoded, list):
        raise ValueError("PostgreSQL JSON 数组字段不是 array")
    return decoded


def _serializable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _lines(*values: Any) -> str:
    return "\n".join(_text(value) for value in values if _text(value))


def _load_document_contexts(
    connection: Any,
    batch_size: int,
    row_reader: RowReader,
    metadata_tagger: MetadataTagger,
) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for row in row_reader(connection, ICP_DOCUMENTS_QUERY, batch_size):
        doc_id = _text(row.get("doc_id"))
        if not doc_id:
            raise ValueError("icp_documents 存在空 doc_id")
        metadata = _json_dict(row.get("metadata"))
        contexts[doc_id] = {
            "doc_id": doc_id,
            "doc_type": _text(row.get("doc_type")),
            "kb_id": _text(row.get("doc_type")),
            "file_name": _text(row.get("file_name")),
            "created_at": _serializable(row.get("created_at")),
            "updated_at": _serializable(row.get("updated_at")),
            "metadata_tags": _serializable(metadata_tagger(metadata)),
        }

    for row in row_reader(connection, LEGACY_DOCUMENTS_QUERY, batch_size):
        doc_id = _text(row.get("doc_id"))
        if not doc_id:
            raise ValueError("documents 存在空 doc_id")
        # An ICP document is more specific if malformed legacy data reuses an ID.
        contexts.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "doc_type": "",
                "kb_id": _text(row.get("kb_id")),
                "file_name": _text(row.get("file_name")),
                "status": _text(row.get("status")),
                "created_at": _serializable(row.get("created_at")),
                "updated_at": _serializable(row.get("updated_at")),
                "metadata_tags": {},
            },
        )
    return contexts


def _base_document(
    contexts: Mapping[str, Mapping[str, Any]],
    row: Mapping[str, Any],
    projection_id: str,
    *,
    title: Any,
    section: Any,
    content: str,
    content_type: str,
    doc_type: str = "",
    kb_id: str = "",
) -> Dict[str, Any]:
    doc_id = _text(row.get("doc_id"))
    context = contexts.get(doc_id, {})
    resolved_doc_type = doc_type or _text(context.get("doc_type"))
    resolved_kb = kb_id or _text(context.get("kb_id")) or resolved_doc_type
    document: Dict[str, Any] = {
        **dict(context.get("metadata_tags") or {}),
        "chunk_id": projection_id,
        "doc_id": doc_id,
        "doc_name": _text(context.get("file_name")),
        "title": _text(title),
        "section": _text(section),
        "content": content,
        "kb_id": resolved_kb,
        "content_type": content_type,
        "doc_type_kwd": resolved_doc_type,
        "removed_kwd": "N",
        "available_int": 1,
    }
    created_at = row.get("created_at") or context.get("created_at")
    updated_at = row.get("updated_at") or context.get("updated_at") or created_at
    if created_at is not None:
        document["created_at"] = _serializable(created_at)
    if updated_at is not None:
        document["updated_at"] = _serializable(updated_at)
    return document


def _document_chunk_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    content = _text(row.get("content"))
    if not content:
        return []
    projection_id = _text(row.get("chunk_id"))
    if not projection_id:
        raise ValueError("document_chunks 存在空 chunk_id")
    document = _base_document(
        contexts,
        row,
        projection_id,
        title=row.get("title"),
        section=row.get("section"),
        content=content,
        content_type=_text(row.get("content_type")) or "text",
    )
    document["chunk_index"] = int(row.get("chunk_index") or 0)
    document["selected_for_kg"] = bool(row.get("selected_for_kg"))
    return [(projection_id, document, content)]


def _hmc_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    row_id = _text(row.get("id"))
    doc_id = _text(row.get("doc_id"))
    code = _text(row.get("hmc_code"))
    variant_id = _text(row.get("variant_id"))
    if not variant_id:
        # Compatibility for a rebuild started before the one-time PG migration
        # completed.  It is still source/business-stable and never falls back
        # to the surrogate sequence id.
        variant_id = hashlib.sha256(
            _lines(
                doc_id,
                code,
                row.get("lcn"),
                row.get("hamc"),
                row.get("hsc"),
                row.get("source_ref"),
                row.get("fault_desc"),
            ).encode("utf-8")
        ).hexdigest()[:32]
    common_tags = {
        "business_id_kwd": code,
        "variant_id_kwd": variant_id,
        "hmc_code_kwd": code,
        "lcn_kwd": _text(row.get("lcn")),
        "hamc_kwd": _text(row.get("hamc")),
        "hsc_kwd": _text(row.get("hsc")),
        "pg_id_kwd": row_id,
    }
    diagnostic = _lines(
        f"HMC: {code}",
        f"LCN: {_text(row.get('lcn'))}；HAMC: {_text(row.get('hamc'))}；HSC: {_text(row.get('hsc'))}",
        f"故障现象: {_text(row.get('fault_desc'))}",
    )
    maintenance = _lines(
        f"HMC: {code}",
        f"故障现象: {_text(row.get('fault_desc'))}",
        f"故障分析程序与维修方法: {_text(row.get('repair_method'))}",
    )
    projections: List[Projection] = []
    for suffix, content_type, content in (
        ("diagnostic", "hmc_diagnostic", diagnostic),
        ("maintenance", "hmc_maintenance", maintenance),
    ):
        projection_id = f"{doc_id}_hmc_{variant_id}_{suffix}"
        document = _base_document(
            contexts,
            row,
            projection_id,
            title=code,
            section="hmc",
            content=content,
            content_type=content_type,
            doc_type="hmc",
            kb_id="hmc",
        )
        document.update(common_tags)
        projections.append((projection_id, document, content))
    return projections


def _fmea_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    row_id = _text(row.get("id"))
    code = _text(row.get("code"))
    content = _lines(
        f"FMECA代码: {code}",
        f"产品/功能: {_text(row.get('product_flag'))} / {_text(row.get('func_management'))}",
        f"故障模式: {_text(row.get('fault_mode'))}",
        f"故障原因: {_text(row.get('fault_cause'))}",
        f"任务阶段与工作方式: {_text(row.get('mission_phase'))}",
        f"局部影响: {_text(row.get('local_impact'))}",
        f"最终影响: {_text(row.get('final_impact'))}",
        f"严酷度: {_text(row.get('severity_raw') or row.get('severity'))}",
        f"故障概率: {_text(row.get('probability_raw') or row.get('probability_level') or row.get('probability'))}",
        f"检测方法: {_text(row.get('detect_method'))}",
        f"设计改进: {_text(row.get('design_improve'))}",
        f"使用补偿措施: {_text(row.get('compensate_action'))}",
    )
    projection_id = f"{_text(row.get('doc_id'))}_row_{row_id}_fmeca"
    document = _base_document(
        contexts,
        row,
        projection_id,
        title=code,
        section="fmeca",
        content=content,
        content_type="fmeca_row",
        doc_type="fmeca",
        kb_id="fmeca",
    )
    document.update(
        {
            "business_id_kwd": code,
            "pg_id_kwd": row_id,
            "severity_kwd": _text(row.get("severity_normalized") or row.get("severity")),
            "product_kwd": _text(row.get("product_flag")),
        }
    )
    if row.get("probability") is not None:
        document["probability_flt"] = float(row["probability"])
    return [(projection_id, document, content)]


def _mfl_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    row_id = _text(row.get("id"))
    code = _text(row.get("mfl_code"))
    api_reportable = row.get("api_reportable")
    platform_test = row.get("platform_test")
    api_reportable_text = (
        "未知" if api_reportable is None else ("是" if api_reportable else "否")
    )
    platform_status = _text(row.get("platform_test_status")) or (
        "未知" if platform_test is None else ("是" if platform_test else "否")
    )
    content = _lines(
        f"MFL: {code}",
        f"模块故障模式: {_text(row.get('module_fault'))}",
        f"SUR板故障模式: {_text(row.get('sur_fault'))}",
        f"功能子电路级故障模式: {_text(row.get('subcircuit_fault'))}",
        f"API可直接上报: {api_reportable_text}",
        f"API名称: {_text(row.get('api_name'))}",
        f"API调用方法: {_text(row.get('api_method'))}",
        f"API使用方法: {_text(row.get('api_usage'))}",
        f"平台级摸底实验: {platform_status}",
        f"维修处置: {_text(row.get('maintenance_action'))}",
    )
    projection_id = f"{_text(row.get('doc_id'))}_row_{row_id}_mfl"
    document = _base_document(
        contexts,
        row,
        projection_id,
        title=code,
        section="mfl",
        content=content,
        content_type="mfl_row",
        doc_type="mfl",
        kb_id="mfl",
    )
    document.update(
        {
            "business_id_kwd": code,
            "pg_id_kwd": row_id,
            "api_reportable_kwd": (
                "unknown" if api_reportable is None else ("true" if api_reportable else "false")
            ),
            "platform_test_kwd": (
                "unknown" if platform_test is None else ("true" if platform_test else "false")
            ),
            "platform_test_status_kwd": platform_status,
        }
    )
    return [(projection_id, document, content)]


def _function_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    aliases = [_text(value) for value in _json_list(row.get("aliases")) if _text(value)]
    function_id = _text(row.get("function_id"))
    function_name = _text(row.get("function_name"))
    content = _lines(
        f"功能ID: {function_id}",
        f"功能名称: {function_name}",
        f"系统: {_text(row.get('system_name'))}",
        f"子系统: {_text(row.get('subsystem_name'))}",
        f"模块: {_text(row.get('module_name'))}",
        f"功能描述: {_text(row.get('description'))}",
        f"别名: {', '.join(aliases)}",
        f"功能失效表现: {_text(row.get('failure_symptoms'))}",
    )
    projection_id = f"{_text(row.get('doc_id'))}_function_{_text(row.get('id'))}"
    document = _base_document(
        contexts,
        row,
        projection_id,
        title=function_name,
        section="function_locator",
        content=content,
        content_type="function_record",
        doc_type="function_locator",
        kb_id="function_locator",
    )
    document.update(
        {
            "business_id_kwd": function_id,
            "system_kwd": _text(row.get("system_name")),
            "module_kwd": _text(row.get("module_name")),
            "source_ref": _text(row.get("source_ref")),
        }
    )
    return [(projection_id, document, content)]


def _history_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    steps = [_text(value) for value in _json_list(row.get("steps")) if _text(value)]
    case_id = _text(row.get("case_id"))
    content = _lines(
        f"故障历史编号: {case_id}",
        f"发生时间: {_text(row.get('occurred_at'))}",
        f"排故人员: {_text(row.get('personnel'))}",
        f"设备类型: {_text(row.get('device_type'))}",
        f"关联模块: {_text(row.get('related_module'))}",
        f"故障现象: {_text(row.get('fault_symptom'))}",
        f"排故方法: {_text(row.get('diagnosis_method'))}",
        f"根本原因: {_text(row.get('root_cause'))}",
        f"处理步骤: {'；'.join(steps)}",
        f"最终方案: {_text(row.get('final_solution'))}",
        f"预防措施: {_text(row.get('preventive_action'))}",
        (
            f"关联 HMC: {_text(row.get('hmc_code'))}；"
            f"FMECA: {_text(row.get('fmeca_code'))}；"
            f"FTA: {_text(row.get('fta_ref'))}；MFL: {_text(row.get('mfl_code'))}"
        ),
    )
    projection_id = f"{_text(row.get('doc_id'))}_history_{_text(row.get('id'))}"
    document = _base_document(
        contexts,
        row,
        projection_id,
        title=case_id,
        section="fault_history",
        content=content,
        content_type="fault_history_case",
        doc_type="fault_history",
        kb_id="fault_history",
    )
    document.update(
        {
            "business_id_kwd": case_id,
            "hmc_code_kwd": _text(row.get("hmc_code")),
            "fmeca_code_kwd": _text(row.get("fmeca_code")),
            "mfl_code_kwd": _text(row.get("mfl_code")),
            "source_ref": _text(row.get("source_ref")),
        }
    )
    return [(projection_id, document, content)]


def _vertex_name(vertex: Mapping[str, Any]) -> str:
    properties = vertex.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    return _text(properties.get("name")) or _text(vertex.get("id"))


def _fault_tree_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    graph = _json_dict(row.get("graph"))
    vertices = graph.get("vertices") or []
    edges = graph.get("edges") or []
    if not isinstance(vertices, list) or not isinstance(edges, list):
        raise ValueError(f"document_fault_trees {row.get('id')} 的 vertices/edges 无效")
    metadata = graph.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    doc_type = _text(graph.get("diagram_type")) or _text(row.get("kb_id"))
    image_id = (
        _text(row.get("image_id"))
        or _text(row.get("chunk_id"))
        or f"{_text(row.get('doc_id'))}_graph_{_text(row.get('id'))}"
    )
    title = _text(row.get("image_title"))
    names = {
        _text(vertex.get("id")) or f"v{index}": _vertex_name(vertex)
        for index, vertex in enumerate(vertices)
        if isinstance(vertex, Mapping)
    }
    whole_content = _lines(
        f"{doc_type.upper()}图: {title}",
        f"图意: {_text(metadata.get('description'))}",
        f"顶事件/根节点: {_text(graph.get('root_fault_id'))}",
        f"关键节点: {', '.join(name for name in names.values() if name)[:1200]}",
    )
    whole = _base_document(
        contexts,
        row,
        image_id,
        title=title,
        section=doc_type,
        content=whole_content,
        content_type=f"{doc_type}_diagram" if doc_type else "engineering_diagram",
        doc_type=doc_type,
        kb_id=_text(row.get("kb_id")) or doc_type,
    )
    whole.update(
        {
            "chapter_kwd": _text(row.get("chunk_id")),
            "image_id": image_id,
            "image_url": _text(metadata.get("image_url")),
        }
    )
    if metadata.get("confidence") is not None:
        whole["diagram_confidence"] = _serializable(metadata.get("confidence"))
    projections: List[Projection] = [(image_id, whole, whole_content)]

    parent_names: Dict[str, List[str]] = {}
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        parent_names.setdefault(target, []).append(names.get(source, source))

    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Mapping):
            raise ValueError(f"document_fault_trees {row.get('id')} 含非对象节点")
        vertex_id = _text(vertex.get("id")) or f"v{index}"
        properties = vertex.get("properties") or {}
        if not isinstance(properties, Mapping):
            properties = {}
        name = names.get(vertex_id, vertex_id)
        content = _lines(
            f"{doc_type.upper()}节点: {name}",
            f"类型: {_text(vertex.get('label'))}",
            f"描述: {_text(properties.get('description'))}",
            f"上游节点: {', '.join(parent_names.get(vertex_id, []))}",
        )
        projection_id = f"{image_id}_node_{index}"
        document = _base_document(
            contexts,
            row,
            projection_id,
            title=name,
            section=doc_type,
            content=content,
            content_type=f"{doc_type}_node" if doc_type else "diagram_node",
            doc_type=doc_type,
            kb_id=_text(row.get("kb_id")) or doc_type,
        )
        document.update(
            {
                "business_id_kwd": name,
                "node_type_kwd": _text(vertex.get("label")),
                "parent_kwd": ",".join(parent_names.get(vertex_id, [])),
                "chapter_kwd": _text(row.get("chunk_id")),
            }
        )
        projections.append((projection_id, document, content))

    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise ValueError(f"document_fault_trees {row.get('id')} 含非对象边")
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        source_name = names.get(source, source)
        target_name = names.get(target, target)
        properties = edge.get("properties") or {}
        if not isinstance(properties, Mapping):
            properties = {}
        content = _lines(
            f"{doc_type.upper()}关系: {source_name} → {target_name}",
            f"关系类型: {_text(edge.get('label'))}",
            f"逻辑/连接属性: {json.dumps(_serializable(properties), ensure_ascii=False)}",
        )
        projection_id = f"{image_id}_edge_{index}"
        document = _base_document(
            contexts,
            row,
            projection_id,
            title=f"{source_name} → {target_name}",
            section=doc_type,
            content=content,
            content_type=f"{doc_type}_edge" if doc_type else "diagram_edge",
            doc_type=doc_type,
            kb_id=_text(row.get("kb_id")) or doc_type,
        )
        document.update(
            {
                "business_id_kwd": source_name,
                "parent_kwd": source_name,
                "target_kwd": target_name,
                "chapter_kwd": _text(row.get("chunk_id")),
            }
        )
        projections.append((projection_id, document, content))
    return projections


def _community_report_projections(
    row: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]]
) -> Iterable[Projection]:
    report = _json_dict(row.get("report"))
    doc_ids = [_text(value) for value in _json_list(row.get("doc_ids")) if _text(value)]
    doc_id = _text(row.get("doc_id")) or (doc_ids[0] if doc_ids else "")
    source_row = dict(row)
    source_row["doc_id"] = doc_id
    entities = [_text(value) for value in _json_list(row.get("entities")) if _text(value)]
    projection_id = _text(row.get("projection_id"))
    if not projection_id:
        raise ValueError("community_reports 存在空 projection_id")
    findings = report.get("findings") or []
    content = _lines(
        f"社区报告: {_text(report.get('title'))}",
        f"摘要: {_text(report.get('summary'))}",
        f"实体: {', '.join(entities)}",
        f"发现: {json.dumps(_serializable(findings), ensure_ascii=False)}",
        f"评级说明: {_text(report.get('rating_explanation'))}",
    )
    document = _base_document(
        contexts,
        source_row,
        projection_id,
        title=report.get("title"),
        section="community_report",
        content=content,
        content_type="community_report",
        kb_id=_text(row.get("kb_id")),
    )
    document.update(
        {
            "projection_id": projection_id,
            "report_id": _text(row.get("report_id")),
            "community_id": _text(row.get("community_id")),
            "doc_ids": doc_ids,
            "entities": entities,
            "report": _serializable(report),
            "knowledge_graph_kwd": "community_report",
            "stale_kwd": _text(row.get("stale_kwd")),
            "stale_reason": _text(row.get("stale_reason")),
        }
    )
    if row.get("stale_at") is not None:
        document["stale_at"] = _serializable(row.get("stale_at"))
    return [(projection_id, document, content or "社区报告")]


SOURCE_BUILDERS: Dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]]], Iterable[Projection]],
] = {
    "document_chunks": _document_chunk_projections,
    "fault_hmc": _hmc_projections,
    "fmea": _fmea_projections,
    "mfl": _mfl_projections,
    "icp_functions": _function_projections,
    "fault_history_cases": _history_projections,
    "document_fault_trees": _fault_tree_projections,
    "community_reports": _community_report_projections,
}


def run_reindex(
    connection: Any,
    embedding_service: Any,
    search_store: Any,
    metadata_tagger: MetadataTagger,
    *,
    dry_run: bool = False,
    batch_size: int = 32,
    reset: bool = False,
    yes: bool = False,
    row_reader: RowReader | None = None,
) -> Dict[str, Any]:
    """Build every supported projection and verify per-source index counts."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if reset and not yes:
        raise ValueError("--reset 会删除当前 OpenSearch 投影，必须同时提供 --yes")
    row_reader = row_reader or iter_query

    contexts = _load_document_contexts(
        connection, batch_size, row_reader, metadata_tagger
    )
    source_counts = {source: 0 for source, _ in SOURCE_QUERIES}
    indexed_counts = {source: 0 for source, _ in SOURCE_QUERIES}
    reset_deleted = 0

    if not dry_run:
        search_store.ensure_ready()
        if reset:
            reset_deleted = int(search_store.delete_by_filter(None))

    pending: List[Tuple[str, str, Dict[str, Any], str]] = []

    def flush() -> None:
        if not pending:
            return
        texts = [item[3] for item in pending]
        vectors = embedding_service.embed_many(texts)
        if not isinstance(vectors, list) or len(vectors) != len(pending):
            raise RuntimeError(
                f"Embedding 批量响应数量错误：期望 {len(pending)}，实际 "
                f"{len(vectors) if isinstance(vectors, list) else '非列表'}"
            )
        for (source, projection_id, document, _), vector in zip(pending, vectors):
            search_store.upsert(projection_id, document, vector=vector)
            indexed_counts[source] += 1
        pending.clear()

    for source, query in SOURCE_QUERIES:
        builder = SOURCE_BUILDERS[source]
        for row in row_reader(connection, query, batch_size):
            for projection_id, document, embedding_text in builder(row, contexts):
                if not projection_id:
                    raise ValueError(f"{source} 生成了空 projection id")
                if not _text(embedding_text):
                    raise ValueError(f"{source}/{projection_id} 生成了空 embedding 文本")
                document["projection_source_kwd"] = source
                document = _serializable(document)
                source_counts[source] += 1
                if not dry_run:
                    pending.append((source, projection_id, document, embedding_text))
                    if len(pending) >= batch_size:
                        flush()
    if not dry_run:
        flush()

    audited_counts: Dict[str, int] = {}
    index_total = None
    if not dry_run:
        for source, _ in SOURCE_QUERIES:
            audited_counts[source] = int(
                search_store.count({"projection_source_kwd": source})
            )
        mismatches = {
            source: {"expected": source_counts[source], "actual": audited_counts[source]}
            for source, _ in SOURCE_QUERIES
            if source_counts[source] != audited_counts[source]
        }
        if mismatches:
            raise RuntimeError(
                "OpenSearch 投影审计不一致；请使用 --reset --yes 完整重建: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        index_total = int(search_store.count())

    return {
        "mode": "dry-run" if dry_run else "reindex",
        "reset": bool(reset and not dry_run),
        "reset_deleted": reset_deleted,
        "sources": source_counts,
        "indexed": indexed_counts if not dry_run else {},
        "audited": audited_counts,
        "total": sum(source_counts.values()),
        "index_total": index_total,
    }


def _load_runtime_dependencies() -> Tuple[Any, Any, Any, MetadataTagger]:
    # All application imports are intentionally local to keep ``--help`` and
    # importing this module free of DB/network initialization.
    import psycopg2

    from backend.config.config import config
    from backend.core.document_metadata import metadata_search_tags
    from backend.core.embedding import embedding_service
    from backend.storage.search_store import search_store

    # A projection rebuild must never import the application's singleton
    # PostgresClient: constructing that object also runs schema migrations and
    # can wait behind locks held by the live application.  Use one explicit,
    # read-only repeatable snapshot and close it in ``main`` instead.
    connection = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=config.POSTGRES_DB,
        connect_timeout=10,
        application_name="icp-rag-opensearch-reindex",
    )
    connection.set_session(
        isolation_level="REPEATABLE READ",
        readonly=True,
        autocommit=False,
    )
    return connection, embedding_service, search_store, metadata_search_tags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 PostgreSQL 真源重建并审计 OpenSearch 全文+向量投影"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只扫描并统计，不连接或写入 OpenSearch/Embedding"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="PostgreSQL 读取与 Embedding 批大小（默认 32）"
    )
    parser.add_argument(
        "--reset", action="store_true", help="写入前删除当前 OpenSearch alias 下的全部投影"
    )
    parser.add_argument(
        "--yes", action="store_true", help="确认 --reset 的破坏性操作"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    dependency_loader: Callable[[], Tuple[Any, Any, Any, MetadataTagger]] = _load_runtime_dependencies,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    if args.reset and not args.yes:
        parser.error("--reset 必须同时提供 --yes")
    connection = None
    try:
        connection, embedder, store, tagger = dependency_loader()
        report = run_reindex(
            connection,
            embedder,
            store,
            tagger,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            reset=args.reset,
            yes=args.yes,
        )
    except Exception as exc:
        print(f"OpenSearch 重建失败: {exc}", file=sys.stderr)
        return 1
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
