"""文档删除级联处理的纯函数辅助模块。

本模块刻意不实例化任何存储客户端（检索/PG/HugeGraph/ArtifactStore），所需客户端由调用方
注入，便于在单测中直接导入而不会触发服务连接。
"""

import json
from datetime import datetime

# 删除流程步骤顺序，None 表示未开始，done 表示全部完成
DELETE_STEP_ORDER = [None, "edges_done", "vertices_done", "media_done",
                     "chunks_done", "logs_done", "es_done", "file_done", "pg_record_done", "done"]


def step_index(step):
    """返回 step 在 DELETE_STEP_ORDER 中的序号，未知/None 返回 0。"""
    try:
        return DELETE_STEP_ORDER.index(step)
    except (ValueError, TypeError):
        return 0


def parse_json_parts(raw):
    """安全解析 HugeGraph 中以 JSON 字符串存储的 *_parts 字段（description_parts 等）。"""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def handle_community_reports(search_store, postgres_client, doc_id, kb_id, fully_removed_entities):
    """删除文档后分级处理社区报告真源与检索投影：

    - 单文档社区报告：物理删除
    - 多文档且无实体被彻底删除：标 content stale（仅描述被裁剪，正文引用过时）
    - 多文档且有实体被彻底删除：标 topology stale（社区结构已变，需重建）

    stale 报告由后续异步重建任务刷新正文，检索阶段据此降权。
    """
    community_reports = postgres_client.get_community_reports(kb_id=kb_id, doc_id=doc_id)
    fully_removed_set = {str(n).upper() for n in fully_removed_entities if n}
    now = datetime.now().isoformat()
    deleted = 0
    updated = 0

    for report in community_reports:
        projection_id = report.get("projection_id") or report.get("es_doc_id", "")
        if not projection_id:
            continue

        report_doc_ids = report.get("doc_ids", [])
        if isinstance(report_doc_ids, str):
            report_doc_ids = [report_doc_ids]

        report_entities = set(report.get("entities", []))
        # 该社区中"被彻底删除"的实体（大小写不敏感匹配）
        gone_in_community = {e for e in report_entities if str(e).upper() in fully_removed_set}

        if len(report_doc_ids) <= 1:
            # PG 是权威源，先更新 PG，再更新可重建投影。
            postgres_client.delete_community_report(projection_id)
            try:
                search_store.delete_by_id(projection_id)
            except Exception:
                # 投影删除失败时恢复真源，保留可重试锦标，
                # 避免只留下一条无法从 PG 定位的孤儿投影。
                postgres_client.upsert_community_report(projection_id, report)
                raise
            deleted += 1
        else:
            new_doc_ids = [d for d in report_doc_ids if d != doc_id]
            if gone_in_community:
                stale = "topology"
                reason = "doc_deleted:entities_gone:" + ",".join(sorted(gone_in_community))[:200]
            else:
                stale = "content"
                reason = "doc_deleted:entities_trimmed"
            updates = {
                "doc_ids": new_doc_ids,
                "doc_id": new_doc_ids[0] if new_doc_ids else "",
                "stale_kwd": stale,
                "stale_reason": reason,
                "stale_at": now,
            }
            postgres_client.mark_community_report_stale(
                projection_id, new_doc_ids, stale, reason, now
            )
            projection = {
                key: value
                for key, value in report.items()
                if key not in {"es_doc_id", "updated_at"}
            }
            projection.update(updates)
            projection.setdefault("knowledge_graph_kwd", "community_report")
            try:
                search_store.upsert(projection_id, projection)
            except Exception:
                postgres_client.upsert_community_report(projection_id, report)
                raise
            updated += 1

    return {"deleted": deleted, "updated": updated}
