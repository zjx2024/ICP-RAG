"""ICP 文档后台解析任务：按 doc_type 路由到七类专用解析器。

旧通用知识库解析链路（parse_document_background / convert_fault_tree_to_hugegraph）
已随通用链路退役；任务进度、状态与结果落库逻辑保持不变。
"""
import os
import asyncio
from typing import Dict, Optional

from backend.core.task_manager import task_manager
from backend.storage.artifact_store import artifact_store, storage_key_from
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store


async def parse_icp_document_background(task_id: str, doc_id: str, model_config: Optional[Dict] = None):
    """按 icp_documents.doc_type 路由到对应解析器（ExcelRow / FaultTree / Text），
    解析器负责 PG + 检索投影入库，本函数负责进度/状态/结果落库。"""
    loop = asyncio.get_event_loop()

    try:
        await task_manager.update_status(task_id, "parsing")
        await task_manager.update_progress(task_id, 5, "[START] 开始解析文档")

        document = postgres_client.get_icp_document(doc_id)
        if not document:
            raise Exception("文档不存在")
        doc_type = document["doc_type"]
        file_name = document["file_name"]
        postgres_client.update_icp_document_status(doc_id, "parsing")

        await task_manager.update_progress(task_id, 8, f"[INFO] 加载文件: {file_name}")

        file_ext = os.path.splitext(file_name)[1].lower()
        storage_key = storage_key_from(document)
        with artifact_store.materialize(storage_key, suffix=file_ext) as temp_path:
            await task_manager.update_progress(task_id, 12, "[INFO] ArtifactStore 原件已就绪")
            await task_manager.check_cancelled(task_id)

            # doc_type 路由解析器
            from backend.core.parsers import ParseContext, ParserRegistry
            from backend.core.document_metadata import (
                extract_document_metadata,
                metadata_search_tags,
            )
            parser = ParserRegistry.get(doc_type)
            document_metadata = extract_document_metadata(temp_path)
            postgres_client.update_icp_document_metadata(doc_id, document_metadata)
            ctx = ParseContext(
                doc_id=doc_id,
                doc_type=doc_type,
                file_name=file_name,
                task_id=task_id,
                model_config=model_config,
                temp_path=temp_path,
                kb_id=doc_type,
                storage_key=storage_key,
                common_tags=metadata_search_tags(document_metadata),
            )

            # 解析器内部进度回调：只追加日志，沿用当前进度值（避免 None 传入）
            prog = {"value": 18}

            def progress_cb(msg: str):
                asyncio.run_coroutine_threadsafe(
                    task_manager.update_progress(task_id, prog["value"], f"[{doc_type}] {msg}"), loop).result()

            ctx.progress_cb = progress_cb

            await task_manager.update_progress(task_id, 15, f"[INFO] 解析器: {type(parser).__name__}（doc_type={doc_type}）")

            errors = await loop.run_in_executor(None, parser.validate, temp_path)
            if errors:
                raise Exception("格式校验失败: " + "; ".join(errors))

            projection_reset_error = ""
            try:
                await loop.run_in_executor(
                    None,
                    search_store.delete_by_filter,
                    {"doc_id": doc_id},
                )
            except Exception as exc:
                # PostgreSQL facts remain authoritative.  Continue parsing and
                # make the required projection rebuild visible in last_result.
                projection_reset_error = str(exc)[:500]
                await task_manager.update_progress(
                    task_id,
                    18,
                    f"[WARN] 旧检索投影清理失败，将标记待重建: {projection_reset_error}",
                )

            await task_manager.update_progress(task_id, 18, "[INFO] 开始解析...")
            prog["value"] = 30
            result = await loop.run_in_executor(None, parser.parse, temp_path, ctx)

        await task_manager.check_cancelled(task_id)

        # PostgreSQL 与检索投影完成后，七类文档都进入可重建的 HugeGraph
        # 自动投影阶段。不存在显式关系时也只写 Document 来源节点，绝不
        # 通过 LLM 或相似度补造关系。
        stats = dict(result.stats)
        if stats.get("projection_failed"):
            # PostgreSQL facts are valid, but the rebuildable search replica is
            # incomplete. Keep the document queryable through authoritative
            # tools while making the repair requirement explicit to operators.
            stats["projection_rebuild_required"] = True
        if projection_reset_error:
            stats["projection_rebuild_required"] = True
            stats.setdefault("projection_warnings", []).append(projection_reset_error)
        stats["kg_pending"] = True
        new_status = "kg_pending"
        postgres_client.update_icp_document_status(doc_id, new_status)
        postgres_client.update_icp_document_result(doc_id, {**stats, "kg_pending": True})

        graph_task_id = None
        try:
            graph_task_id = await task_manager.create_task(
                doc_id=doc_id,
                kb_id=doc_type,
                file_name=file_name,
                task_type="generate_kg",
            )
            if not postgres_client.transition_icp_document_status(
                doc_id, ("kg_pending",), "generating_kg"
            ):
                raise RuntimeError("文档状态已变化，自动图任务未启动")
            from backend.services.kg_service import generate_icp_graph_background

            asyncio.create_task(
                generate_icp_graph_background(
                    graph_task_id,
                    doc_id,
                    doc_type,
                    file_name,
                )
            )
            new_status = "generating_kg"
            stats["graph_task_id"] = graph_task_id
        except Exception as graph_schedule_error:
            postgres_client.update_icp_document_status(doc_id, "kg_pending")
            stats.setdefault("graph_warnings", []).append(
                f"自动图任务未启动，可安全重试: {str(graph_schedule_error)[:500]}"
            )
            await task_manager.update_progress(
                task_id,
                95,
                f"[WARN] 自动图任务未启动，可稍后重试: {str(graph_schedule_error)[:200]}",
            )
            postgres_client.update_icp_document_result(doc_id, stats)

        await task_manager.update_progress(task_id, 100, f"[FINISH] {result.message}")
        await task_manager.update_status(task_id, "completed")
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "status": new_status,
            "stats": stats,
            "graph_task_id": graph_task_id,
            "message": result.message,
        })

    except asyncio.CancelledError:
        await task_manager.update_progress(task_id, 0, "[CANCEL] 文档解析已被用户取消")
        await task_manager.update_status(task_id, "cancelled")
        postgres_client.update_icp_document_status(doc_id, "failed")

    except Exception as e:
        import traceback
        error_detail = f"[ERROR] {str(e)}\n\n{traceback.format_exc()}"
        await task_manager.set_error(task_id, error_detail)
        await task_manager.update_progress(task_id, 0, f"[ERROR] 解析失败: {str(e)}")
        await task_manager.update_status(task_id, "failed")
        postgres_client.update_icp_document_status(doc_id, "failed")
        try:
            # Never leave partially rebuilt search documents visible after a
            # PostgreSQL/parser failure.  They are projections and can be
            # regenerated from the raw artifact and authority tables.
            await loop.run_in_executor(
                None,
                search_store.delete_by_filter,
                {"doc_id": doc_id},
            )
        except Exception:
            pass
