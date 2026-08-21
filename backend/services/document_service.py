"""文档服务：后台解析任务（从 documents.py 路由抽取）。

保持原实现行为完全一致（含 llm_semaphore 并发信号量、任务进度上报）。
"""
import os
import json
import asyncio
import concurrent.futures
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException

from backend.core.docx_parser import DocumentParser
from backend.core.embedding import embedding_service
from backend.core.llm_service import llm_service
from backend.core.task_manager import task_manager
from backend.storage.es_client import es_client
from backend.storage.minio_client import minio_client
from backend.storage.postgres_client import postgres_client

llm_semaphore = asyncio.Semaphore(3)

def convert_fault_tree_to_hugegraph(fault_tree_graph: dict, doc_id: str, kb_id: str, image_index: int):
    vertices = []
    edges = []
    
    if not fault_tree_graph:
        return vertices, edges
    
    graph_vertices = fault_tree_graph.get("vertices", [])
    graph_edges = fault_tree_graph.get("edges", [])
    
    for idx, v in enumerate(graph_vertices):
        vertex_id = f"{doc_id}_ft_{image_index}_v_{idx}"
        label = v.get("label", "Fault")
        
        properties = v.get("properties", {})
        properties["doc_id"] = doc_id
        properties["kb_id"] = kb_id
        properties["source_type"] = "fault_tree"
        properties["source_id"] = [doc_id]
        properties["created_at"] = datetime.now().isoformat()
        
        entities_type = v.get("label", "Fault")
        properties["entity_type"] = entities_type
        
        vertices.append({
            "id": vertex_id,
            "label": "Entity",
            "properties": properties
        })
    
    vertex_id_map = {}
    for idx, v in enumerate(graph_vertices):
        original_id = v.get("id", f"v_{idx}")
        vertex_id_map[original_id] = f"{doc_id}_ft_{image_index}_v_{idx}"
    
    for idx, e in enumerate(graph_edges):
            source_orig = e.get("source", "")
            target_orig = e.get("target", "")
            
            if source_orig not in vertex_id_map or target_orig not in vertex_id_map:
                continue
            
            edge_id = f"{doc_id}_ft_{image_index}_e_{idx}"
            label = e.get("label", "causes")
            
            properties = e.get("properties", {})
            if isinstance(properties.get("gate_type"), dict):
                properties["gate_type"] = properties["gate_type"].get("value", "direct")
            properties["doc_id"] = doc_id
            properties["kb_id"] = kb_id
            properties["source_type"] = "fault_tree"
            properties["source_id"] = [doc_id]
            properties["created_at"] = datetime.now().isoformat()
            
            edges.append({
                "id": edge_id,
                "label": label,
                "outV": vertex_id_map[source_orig],
                "inV": vertex_id_map[target_orig],
                "outVLabel": "Entity",
                "inVLabel": "Entity",
                "properties": properties
            })
    
    return vertices, edges

async def parse_document_background(task_id: str, kb_id: str, doc_id: str, model_config: Optional[Dict] = None):
    loop = asyncio.get_event_loop()
    
    try:
        await task_manager.update_status(task_id, "parsing")
        await task_manager.update_progress(task_id, 5, "[START] 开始解析文档")
        
        document = postgres_client.get_document(doc_id)
        if not document:
            raise Exception("文档不存在")
        
        file_name = document["file_name"]
        
        await task_manager.update_progress(task_id, 8, f"[INFO] 加载文件: {file_name}")
        
        try:
            file_data = minio_client.get_file(document["minio_path"])
            file_ext = os.path.splitext(file_name)[1].lower()
            temp_path = f"/tmp/parse_{doc_id}{file_ext}"
            with open(temp_path, "wb") as f:
                f.write(file_data)
            await task_manager.update_progress(task_id, 12, "[INFO] 文件下载完成")
        except Exception:
            temp_path = f"/tmp/{doc_id}_{file_name}"
            await task_manager.update_progress(task_id, 12, "[WARNING] 使用本地文件路径")
        
        await task_manager.check_cancelled(task_id)
        
        await task_manager.update_progress(task_id, 15, "[INFO] 初始化文档解析器")
        
        def parse_document_sync():
            parser = DocumentParser(temp_path, doc_id)
            return parser.process()
        
        await task_manager.update_progress(task_id, 17, "[INFO] 开始解析文档内容...")
        result = await loop.run_in_executor(None, parse_document_sync)
        
        if "error" in result:
            raise Exception(result["error"])
        
        blocks = result["blocks"]
        total_blocks = len(blocks)
        
        await task_manager.update_progress(task_id, 20, f"[SUCCESS] 文档解析完成，发现 {total_blocks} 个分块")
        
        total_chunks = 0
        index_name = es_client.get_main_index_name("default")
        es_client.create_index(index_name)
        
        fault_tree_count = 0
        
        block_counter = 0
        for block_key, block_info in blocks.items():
            block_counter += 1
            progress = 20 + int((block_counter / total_blocks) * 60)
            
            await task_manager.update_progress(task_id, progress, f"[STEP {block_counter}/{total_blocks}] 处理分块: {block_info['title']}")
            
            await task_manager.check_cancelled(task_id)
            
            chunk_id = f"{doc_id}_{block_key}"
            
            text_content = block_info["text"]
            enhanced_content_parts = [text_content] if text_content else []
            
            image_count = len(block_info["images"])
            table_count = len(block_info["tables"])
            
            if image_count > 0:
                await task_manager.update_progress(task_id, progress, f"  ├─ 发现 {image_count} 张图片，开始分析...")
            
            for img_idx, img in enumerate(block_info["images"]):
                image_local_path = img.get("local_path", "")
                if image_local_path and os.path.exists(image_local_path):
                    await task_manager.update_progress(task_id, progress, f"  │  └─ 分析图片 {img_idx + 1}/{image_count}: {img.get('title', '未命名图片')}")
                    
                    try:
                        async with llm_semaphore:
                            await task_manager.update_progress(task_id, progress, f"  │     └─ 调用 LLM 分析图片内容...")
                            img_analysis = await loop.run_in_executor(None, llm_service.analyze_image,
                                image_local_path, img.get("title", ""), block_key, text_content, model_config)
                        
                        img["llm_description"] = img_analysis.get("description", "")
                        img["image_type"] = img_analysis.get("image_type", "normal")
                        enhanced_content_parts.append(f"\n[图片分析] {img_analysis.get('description', '')}")
                        
                        if img_analysis.get("image_type") == "fault_tree" and img_analysis.get("fault_tree_graph"):
                            await task_manager.update_progress(task_id, progress, f"  │     └─ 检测到故障树，保存结构化数据到 PostgreSQL...")
                            try:
                                image_id = f"{chunk_id}_img_{img.get('index', img_idx)}"
                                fault_tree_id = str(uuid4())
                                await loop.run_in_executor(None, postgres_client.insert_document_fault_tree,
                                    fault_tree_id, doc_id, kb_id, chunk_id, image_id,
                                    img.get("title", ""), json.dumps(img_analysis["fault_tree_graph"], ensure_ascii=False))
                                fault_tree_count += 1
                                await task_manager.update_progress(task_id, progress, f"  │        └─ 已保存故障树结构化数据（手动生成知识图谱时入库）")
                            except Exception as e:
                                await task_manager.update_progress(task_id, progress, f"  │        └─ 故障树数据保存失败: {str(e)[:100]}")
                    except Exception as e:
                        await task_manager.update_progress(task_id, progress, f"  │     └─ 图片分析失败: {str(e)[:100]}")
                        img["llm_description"] = f"[图片分析失败: {str(e)[:100]}]"
                        img["image_type"] = "normal"
            
            if table_count > 0:
                await task_manager.update_progress(task_id, progress, f"  └─ 发现 {table_count} 个表格，开始分析...")
            
            for tbl in block_info["tables"]:
                table_markdown = tbl.get("markdown", "")
                if table_markdown:
                    await task_manager.update_progress(task_id, progress, f"     └─ 分析表格: {tbl.get('title', '未命名表格')}")
                    
                    try:
                        async with llm_semaphore:
                            await task_manager.update_progress(task_id, progress, f"        └─ 调用 LLM 分析表格内容...")
                            tbl_analysis = await loop.run_in_executor(None, llm_service.analyze_table,
                                table_markdown, tbl.get("title", ""), block_key, text_content, model_config)
                        
                        tbl["llm_description"] = tbl_analysis
                        enhanced_content_parts.append(f"\n[表格分析] {tbl_analysis}")
                    except Exception as e:
                        await task_manager.update_progress(task_id, progress, f"        └─ 表格分析失败: {str(e)[:100]}")
                        tbl["llm_description"] = f"[表格分析失败: {str(e)[:100]}]"
            
            enhanced_text_content = "\n".join(enhanced_content_parts)
            
            await task_manager.update_progress(task_id, progress, f"  └─ 保存分块数据到 PostgreSQL")
            try:
                await loop.run_in_executor(None, postgres_client.insert_chunk,
                    chunk_id, doc_id, block_key, block_info["title"], enhanced_text_content,
                    "text", total_chunks)
            except Exception as e:
                await task_manager.update_progress(task_id, progress, f"  └─ 分块保存失败: {str(e)[:100]}")
            
            if enhanced_text_content:
                try:
                    await task_manager.update_progress(task_id, progress, f"     └─ 生成文本向量嵌入...")
                    vector = await loop.run_in_executor(None, embedding_service.embed, enhanced_text_content)
                    doc = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "doc_name": file_name,
                        "title": block_info.get("title", ""),
                        "section": block_key,
                        "content": enhanced_text_content,
                        "content_vector": vector,
                        "kb_id": kb_id,
                        "content_type": "text",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    await task_manager.update_progress(task_id, progress, f"        └─ 保存向量到 Elasticsearch")
                    await loop.run_in_executor(None, partial(es_client.client.index, index=index_name, id=chunk_id, body=doc))
                except Exception as e:
                    await task_manager.update_progress(task_id, progress, f"        └─ 向量保存失败: {str(e)[:100]}")
            
            for img in block_info["images"]:
                image_id = f"{chunk_id}_img_{img['index']}"
                image_type = img.get("image_type", "normal")
                llm_description = img.get("llm_description", "")
                image_url = img.get("image_url", "")
                
                await task_manager.update_progress(task_id, progress, f"  ├─ 保存图片数据: {img.get('title', '图片')}")
                try:
                    await loop.run_in_executor(None, postgres_client.insert_image,
                        image_id, doc_id, chunk_id, img.get("title", ""),
                        llm_description, img.get("minio_path", ""), image_type, image_url)
                    
                    img_content = f"图片: {img.get('title', '')}\n{llm_description}"
                    img_vector = await loop.run_in_executor(None, embedding_service.embed, img_content)
                    img_doc = {
                        "chunk_id": image_id,
                        "doc_id": doc_id,
                        "doc_name": file_name,
                        "title": img.get("title", ""),
                        "section": block_key,
                        "content": img_content,
                        "content_vector": img_vector,
                        "kb_id": kb_id,
                        "content_type": "image",
                        "image_url": image_url,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    await loop.run_in_executor(None, partial(es_client.client.index, index=index_name, id=image_id, body=img_doc))
                except Exception as e:
                    await task_manager.update_progress(task_id, progress, f"  ├─ 图片数据保存失败: {str(e)[:100]}")
            
            for tbl in block_info["tables"]:
                table_id = f"{chunk_id}_tbl_{tbl['index']}"
                llm_description = tbl.get("llm_description", "")
                
                await task_manager.update_progress(task_id, progress, f"  └─ 保存表格数据: {tbl.get('title', '表格')}")
                try:
                    await loop.run_in_executor(None, postgres_client.insert_table,
                        table_id, doc_id, chunk_id, tbl.get("title", ""),
                        llm_description, json.dumps(tbl.get("meta_data", {}), ensure_ascii=False))
                    
                    tbl_content = f"表格: {tbl.get('title', '')}\n{llm_description}"
                    tbl_vector = await loop.run_in_executor(None, embedding_service.embed, tbl_content)
                    tbl_doc = {
                        "chunk_id": table_id,
                        "doc_id": doc_id,
                        "doc_name": file_name,
                        "title": tbl.get("title", ""),
                        "section": block_key,
                        "content": tbl_content,
                        "content_vector": tbl_vector,
                        "kb_id": kb_id,
                        "content_type": "table",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    await loop.run_in_executor(None, partial(es_client.client.index, index=index_name, id=table_id, body=tbl_doc))
                except Exception as e:
                    await task_manager.update_progress(task_id, progress, f"  └─ 表格数据保存失败: {str(e)[:100]}")
            
            total_chunks += 1
            await task_manager.update_progress(task_id, progress, f"[SUCCESS] 分块 {block_counter}/{total_blocks} 处理完成")
        
        await task_manager.check_cancelled(task_id)
        
        await task_manager.update_progress(task_id, 90, "[INFO] 更新文档状态")
        
        postgres_client.update_document_status(doc_id, "indexed")
        postgres_client.update_document_chunks(doc_id, total_chunks)
        
        await task_manager.update_progress(task_id, 100, "[FINISH] 文档解析完成")
        await task_manager.update_status(task_id, "completed")
        
        message = f"文档解析完成，共生成 {total_chunks} 个分块"
        if fault_tree_count:
            message += f"，检测到 {fault_tree_count} 张故障树图，结构化数据已保存（手动生成知识图谱时入库）"
        
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "total_chunks": total_chunks,
            "status": "indexed",
            "message": message
        })
    
    except asyncio.CancelledError:
        await task_manager.update_progress(task_id, 0, "[CANCEL] 文档解析已被用户取消")
        await task_manager.update_status(task_id, "cancelled")
        postgres_client.update_document_status(doc_id, "failed")
    
    except Exception as e:
        import traceback
        error_detail = f"[ERROR] {str(e)}\n\n{traceback.format_exc()}"
        await task_manager.set_error(task_id, error_detail)
        await task_manager.update_progress(task_id, 0, f"[ERROR] 解析失败: {str(e)}")
        await task_manager.update_status(task_id, "failed")
        postgres_client.update_document_status(doc_id, "failed")


# ===================== ICP 类型驱动解析编排（doc_type 路由） =====================

async def parse_icp_document_background(task_id: str, doc_id: str, model_config: Optional[Dict] = None):
    """按 icp_documents.doc_type 路由到对应解析器（ExcelRow / FaultTree / Text），
    解析器负责 PG + ES 入库，本函数负责进度/状态/结果落库。"""
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

        try:
            file_data = minio_client.get_file(document["minio_path"])
            file_ext = os.path.splitext(file_name)[1].lower()
            temp_path = f"/tmp/icp_parse_{doc_id}{file_ext}"
            with open(temp_path, "wb") as f:
                f.write(file_data)
            await task_manager.update_progress(task_id, 12, "[INFO] 文件下载完成")
        except Exception:
            temp_path = f"/tmp/icp_parse_{doc_id}"
            await task_manager.update_progress(task_id, 12, "[WARNING] 使用本地文件路径")

        await task_manager.check_cancelled(task_id)

        # doc_type 路由解析器
        from backend.core.parsers import ParseContext, ParserRegistry
        parser = ParserRegistry.get(doc_type)
        ctx = ParseContext(
            doc_id=doc_id,
            doc_type=doc_type,
            file_name=file_name,
            task_id=task_id,
            model_config=model_config,
            temp_path=temp_path,
            kb_id=doc_type,
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

        await task_manager.update_progress(task_id, 18, "[INFO] 开始解析...")
        prog["value"] = 30
        result = await loop.run_in_executor(None, parser.parse, temp_path, ctx)

        await task_manager.check_cancelled(task_id)

        # 状态更新：图类含三元组 → kg_pending，其余 indexed
        stats = result.stats
        new_status = "kg_pending" if stats.get("kg_pending") else "indexed"
        postgres_client.update_icp_document_status(doc_id, new_status)
        postgres_client.update_icp_document_result(doc_id, {**stats, "kg_pending": stats.get("kg_pending", False)})

        await task_manager.update_progress(task_id, 100, f"[FINISH] {result.message}")
        await task_manager.update_status(task_id, "completed")
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "status": new_status,
            "stats": stats,
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

