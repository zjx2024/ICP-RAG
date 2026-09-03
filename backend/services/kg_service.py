"""知识图谱服务：图谱生成后台任务（从 documents.py 路由抽取）。

保持原实现行为完全一致（含 GraphExtractor 抽取、跨文档实体合并、社区报告生成）。
"""
import asyncio
import copy
import json
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import networkx as nx

from backend.core.task_manager import task_manager
from backend.graphrag.graph_extractor import GraphExtractor, doc_id_to_str
from backend.services.document_service import convert_fault_tree_to_hugegraph
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store
from backend.api.routers._deletion_helpers import parse_json_parts
from backend.icp.diagram_models import content_hash
from backend.icp.diagram_repository import diagram_repository
from backend.icp.diagram_validation import (
    VALIDATION_FAILED,
    VALIDATION_QUARANTINED,
    revision_is_projectable,
)

async def generate_kg_background(task_id: str, kb_id: str, doc_id: str, selected_chunk_ids: List[str], model_config: Optional[Dict] = None, mode: str = "enhanced", entity_resolution: bool = True, with_community: bool = True):
    try:
        loop = asyncio.get_running_loop()

        # 兜底：入口已做预检查，正常不会进入此分支；这里仍按 llm 类型取默认，避免误用 embedding 模型
        if not model_config:
            model_config = postgres_client.get_default_model_by_type("llm")
        
        await task_manager.update_status(task_id, "generating_kg")
        
        await task_manager.update_progress(task_id, 5, "[START] 开始生成知识图谱", log_type="start")
        
        if model_config:
            model_name = model_config.get("name", model_config.get("model_name", "Unknown"))
            model_url = model_config.get("api_url", model_config.get("base_url", "Unknown"))
            await task_manager.update_progress(task_id, 6, 
                f"[INFO] 使用模型: {model_name} (地址: {model_url})", log_type="model")
        
        document = postgres_client.get_document(doc_id)
        file_name = document["file_name"]
        
        await task_manager.update_progress(task_id, 8, f"[INFO] 获取文档数据: {file_name}", log_type="document")
        
        await task_manager.update_progress(task_id, 10, "[INFO] 加载文档分块数据", log_type="chunks")
        
        all_chunks = await loop.run_in_executor(None, postgres_client.get_chunks_by_doc, doc_id)
        
        await task_manager.update_progress(task_id, 12, 
            f"[INFO] 文档共 {len(all_chunks)} 个分块，选中 {len(selected_chunk_ids)} 个", log_type="chunks")
        
        selected_chunks_info = []
        for chunk_id in selected_chunk_ids:
            chunk_info = next((c for c in all_chunks if c["chunk_id"] == chunk_id), None)
            if chunk_info and chunk_info.get("content"):
                selected_chunks_info.append({
                    "chunk_id": chunk_id,
                    "title": chunk_info.get("title", ""),
                    "chunk_index": chunk_info.get("chunk_index", 0),
                    "content": chunk_info["content"]
                })
        
        if not selected_chunks_info:
            raise Exception("选中的分块中没有内容")
        
        await task_manager.update_progress(task_id, 15, 
            f"[INFO] 准备处理 {len(selected_chunks_info)} 个分块", log_type="chunks")
        
        await task_manager.update_progress(task_id, 20, "[INFO] 开始逐个分块提取实体和关系...", log_type="extract")
        
        await task_manager.check_cancelled(task_id)
        
        graph_extractor = GraphExtractor()
        
        all_entities = {}
        all_relations = {}
        
        total_chunks = len(selected_chunks_info)
        for idx, chunk_info in enumerate(selected_chunks_info):
            chunk_title = chunk_info.get("title", f"分块 {chunk_info.get('chunk_index', idx + 1)}")
            chunk_content_preview = chunk_info["content"][:100] + "..." if len(chunk_info["content"]) > 100 else chunk_info["content"]
            chunk_content_length = len(chunk_info["content"])
            
            progress = 20 + int((idx + 1) * 55 / total_chunks)
            
            await task_manager.update_progress(task_id, progress, 
                f"[STEP] 正在处理分块 {idx + 1}/{total_chunks}: {chunk_title}", 
                log_type="chunk_process",
                chunk_id=chunk_info["chunk_id"],
                chunk_index=chunk_info.get("chunk_index", idx + 1),
                chunk_title=chunk_title)
            
            await task_manager.check_cancelled(task_id)
            
            await task_manager.update_progress(task_id, progress,
                f"[INFO]   分块ID: {chunk_info['chunk_id']}",
                log_type="chunk_detail")
            
            await task_manager.update_progress(task_id, progress,
                f"[INFO]   内容长度: {chunk_content_length} 字符",
                log_type="chunk_detail")
            
            await task_manager.update_progress(task_id, progress,
                f"[INFO]   内容预览: {chunk_content_preview}",
                log_type="chunk_content")
            
            chunk_result = await loop.run_in_executor(None,
                graph_extractor.extract_entities_and_relations, 
                f"{doc_id}_{idx}", [chunk_info["content"]], model_config, mode, False, entity_resolution)
            
            chunk_entities, chunk_relations, _ = chunk_result
            
            for en_name in chunk_entities:
                chunk_entities[en_name]["doc_id"] = [doc_id]
            
            await task_manager.update_progress(task_id, progress,
                f"[RESULT] 提取完成: {len(chunk_entities)} 个实体，{len(chunk_relations)} 条关系",
                log_type="chunk_result",
                chunk_id=chunk_info["chunk_id"],
                chunk_index=chunk_info.get("chunk_index", idx + 1),
                chunk_title=chunk_title,
                entity_count=len(chunk_entities),
                relation_count=len(chunk_relations),
                entities=list(chunk_entities.keys())[:20] if chunk_entities else None)
            
            if chunk_entities:
                entity_names = list(chunk_entities.keys())[:10]
                entity_str = ", ".join(entity_names)
                if len(chunk_entities) > 10:
                    entity_str += f" ... (共{len(chunk_entities)}个)"
                await task_manager.update_progress(task_id, progress,
                    f"[INFO]   实体列表: {entity_str}",
                    log_type="chunk_entities")
            
            if chunk_relations:
                rel_descs = []
                for (src, tgt), rel in list(chunk_relations.items())[:5]:
                    label = rel.get("label", "related_to")
                    rel_descs.append(f"{src} -{label}-> {tgt}")
                rel_str = ", ".join(rel_descs)
                if len(chunk_relations) > 5:
                    rel_str += f" ... (共{len(chunk_relations)}条)"
                await task_manager.update_progress(task_id, progress,
                    f"[INFO]   关系列表: {rel_str}",
                    log_type="chunk_relations")
            
            for en_name, en_info in chunk_entities.items():
                if en_name in all_entities:
                    all_entities[en_name]["description"] = graph_extractor._handle_entity_relation_summary(
                        en_name, 
                        all_entities[en_name]["description"] + graph_extractor.GRAPH_FIELD_SEP + en_info["description"],
                        model_config
                    )
                    all_entities[en_name]["source_id"] = sorted(set(
                        all_entities[en_name]["source_id"] + en_info["source_id"]
                    ))
                    existing_doc_ids = all_entities[en_name].get("doc_id", [])
                    new_doc_ids = en_info.get("doc_id", [])
                    all_entities[en_name]["doc_id"] = list(set(existing_doc_ids + new_doc_ids))
                else:
                    all_entities[en_name] = en_info
            
            for (src, tgt), rel_info in chunk_relations.items():
                key = tuple(sorted([src, tgt]))
                if key in all_relations:
                    all_relations[key]["description"] += graph_extractor.GRAPH_FIELD_SEP + rel_info["description"]
                    all_relations[key]["weight"] += rel_info["weight"]
                    all_relations[key]["keywords"] = list(set(
                        all_relations[key]["keywords"] + rel_info.get("keywords", [])
                    ))
                    all_relations[key]["source_id"] = sorted(set(
                        all_relations[key]["source_id"] + rel_info.get("source_id", [])
                    ))
                else:
                    all_relations[key] = rel_info
        
        await task_manager.update_progress(task_id, 78, 
            f"[INFO] 合并完成，共 {len(all_entities)} 个实体，{len(all_relations)} 条关系", 
            log_type="merge",
            entity_count=len(all_entities),
            relation_count=len(all_relations))
        
        await task_manager.check_cancelled(task_id)
        
        await task_manager.update_progress(task_id, 80, "[INFO] 执行实体消歧和合并...", log_type="disambiguation")
        
        graph = nx.Graph()
        for en_nm, en_info in all_entities.items():
            graph.add_node(en_nm, **en_info)
        
        for (src, tgt), rel_info in all_relations.items():
            graph.add_edge(src, tgt, **rel_info)
        
        resolution_result = await loop.run_in_executor(None, graph_extractor._resolve_entities, graph)
        if resolution_result and resolution_result.resolution_pairs:
            await task_manager.update_progress(task_id, 82, 
                f"[INFO] 发现 {len(resolution_result.resolution_pairs)} 对相似实体，进行合并...", 
                log_type="disambiguation")
            
            connect_graph = nx.Graph()
            connect_graph.add_edges_from(resolution_result.resolution_pairs)
            
            for sub_connect_graph in nx.connected_components(connect_graph):
                merging_nodes = list(sub_connect_graph)
                await task_manager.update_progress(task_id, 83, 
                    f"[INFO]   合并实体: {', '.join(merging_nodes)}",
                    log_type="merge_detail")
                await loop.run_in_executor(None, graph_extractor._merge_graph_nodes, graph, merging_nodes, model_config)
            
            pr = nx.pagerank(graph)
            for node_name, pagerank in pr.items():
                graph.nodes[node_name]["pagerank"] = pagerank
            
            all_entities = {}
            for node_name in graph.nodes():
                all_entities[node_name] = dict(graph.nodes[node_name])
            
            all_relations = {}
            for src, tgt in graph.edges():
                edge_data = dict(graph.get_edge_data(src, tgt))
                key = tuple(sorted([src, tgt]))
                all_relations[key] = edge_data
            
            await task_manager.update_progress(task_id, 85, 
                f"[INFO] 消歧后: {len(all_entities)} 个实体，{len(all_relations)} 条关系",
                log_type="disambiguation",
                entity_count=len(all_entities),
                relation_count=len(all_relations))
        else:
            await task_manager.update_progress(task_id, 85, "[INFO] 未发现相似实体，跳过消歧", log_type="disambiguation")
        
        await task_manager.check_cancelled(task_id)
        
        await task_manager.update_progress(task_id, 86, "[INFO] 开始跨文档实体合并...", log_type="cross_doc_merge")
        
        await task_manager.update_progress(task_id, 87, "[INFO] 从 HugeGraph 获取同知识库已有实体和关系...", log_type="cross_doc_merge")
        
        existing_vertices = await loop.run_in_executor(None, 
            hugegraph_client.get_vertices_by_kb_with_details, kb_id)
        existing_edges = await loop.run_in_executor(None, 
            hugegraph_client.get_edges_by_kb_with_details, kb_id)
        
        await task_manager.update_progress(task_id, 88, 
            f"[INFO] 获取到 {len(existing_vertices)} 个已有实体，{len(existing_edges)} 条已有关系",
            log_type="cross_doc_merge")
        
        name_to_vertex_id = {}
        vid_to_name = {}
        existing_graph = nx.Graph()
        
        for vertex in existing_vertices:
            props = vertex.get("properties", {})
            name = props.get("name", "").upper()
            vid = vertex.get("id")
            
            if name and vid:
                name_to_vertex_id[name] = vid
                vid_to_name[vid] = name
                raw_doc_id = props.get("doc_id", [])
                if isinstance(raw_doc_id, str):
                    raw_doc_id = raw_doc_id.split(",") if raw_doc_id else []
                    
                existing_graph.add_node(name,
                    entity_type=props.get("entity_type", "category"),
                    description=props.get("description", ""),
                    source_id=props.get("source_id", []),
                    pagerank=props.get("pagerank", 0),
                    doc_id=raw_doc_id,
                    description_parts=parse_json_parts(props.get("description_parts")),
                    keywords_parts=parse_json_parts(props.get("keywords_parts")),
                    _hugegraph_id=vid,
                    _is_existing=True
                )
        
        for edge in existing_edges:
            props = edge.get("properties", {})
            outV = edge.get("outV")
            inV = edge.get("inV")
            
            out_name = vid_to_name.get(outV)
            in_name = vid_to_name.get(inV)
            
            if out_name and in_name:
                existing_graph.add_edge(out_name, in_name,
                    description=props.get("description", ""),
                    label=props.get("relation_type", "related_to"),
                    keywords=props.get("keywords", []),
                    weight=props.get("weight", 1),
                    source_id=props.get("source_id", []),
                    description_parts=parse_json_parts(props.get("description_parts")),
                    keywords_parts=parse_json_parts(props.get("keywords_parts")),
                    weight_parts=parse_json_parts(props.get("weight_parts")),
                    _hugegraph_id=edge.get("id"),
                    _is_existing=True
                )
        
        await task_manager.update_progress(task_id, 88.5, 
            f"[INFO] 构建现有图谱完成: {existing_graph.number_of_nodes()} 个节点，{existing_graph.number_of_edges()} 条边",
            log_type="cross_doc_merge")
        
        new_entity_names = set(all_entities.keys())
        
        combined_graph = nx.Graph()
        
        for en_nm, en_info in all_entities.items():
            combined_graph.add_node(en_nm, **en_info, _is_existing=False)
        
        for (src, tgt), rel_info in all_relations.items():
            combined_graph.add_edge(src, tgt, **rel_info, _is_existing=False)
        
        for node in existing_graph.nodes():
            if node not in combined_graph.nodes():
                combined_graph.add_node(node, **existing_graph.nodes[node])
            else:
                # 同名实体已存在于新文档的实体中，合并 doc_id 以正确追踪
                # 该实体属于哪些文档（社区报告的 doc_ids 依赖此字段）
                existing_doc_id = existing_graph.nodes[node].get("doc_id", [])
                combined_doc_id = combined_graph.nodes[node].get("doc_id", [])
                if isinstance(existing_doc_id, str):
                    existing_doc_id = [existing_doc_id]
                if isinstance(combined_doc_id, str):
                    combined_doc_id = [combined_doc_id]
                merged_doc_id = list(set(combined_doc_id + existing_doc_id))
                combined_graph.nodes[node]["doc_id"] = merged_doc_id
        
        for src, tgt in existing_graph.edges():
            if not combined_graph.has_edge(src, tgt):
                edge_data = dict(existing_graph.get_edge_data(src, tgt))
                combined_graph.add_edge(src, tgt, **edge_data)
        
        await task_manager.update_progress(task_id, 89, 
            f"[INFO] 合并后总图: {combined_graph.number_of_nodes()} 个节点，{combined_graph.number_of_edges()} 条边",
            log_type="cross_doc_merge")
        
        resolution_result = await loop.run_in_executor(None, 
            graph_extractor._resolve_entities, combined_graph, new_entity_names)
        
        if resolution_result.resolution_pairs:
            await task_manager.update_progress(task_id, 90, 
                f"[INFO] 跨文档消歧发现 {len(resolution_result.resolution_pairs)} 对相似实体，进行合并...",
                log_type="cross_doc_merge")
            
            for merge_detail in resolution_result.change.merge_details:
                base_name = merge_detail["base_entity"]["name"]
                base_type = merge_detail["base_entity"]["entity_type"]
                merged_names = [e["name"] for e in merge_detail["merged_entities"]]
                
                await task_manager.update_progress(task_id, 90.5,
                    f"[MERGE] 主实体: {base_name} ({base_type})",
                    log_type="merge_detail")
                
                for merged_entity in merge_detail["merged_entities"]:
                    merged_name = merged_entity["name"]
                    merged_type = merged_entity["entity_type"]
                    merged_desc = merged_entity["description"]
                    
                    await task_manager.update_progress(task_id, 90.6,
                        f"[MERGE]   <- 合并: {merged_name} ({merged_type})",
                        log_type="merge_detail")
                    if merged_desc:
                        await task_manager.update_progress(task_id, 90.7,
                            f"[MERGE]       描述: {merged_desc}",
                            log_type="merge_detail")
            
            connect_graph = nx.Graph()
            connect_graph.add_edges_from(resolution_result.resolution_pairs)
            
            for sub_connect_graph in nx.connected_components(connect_graph):
                merging_nodes = list(sub_connect_graph)
                await task_manager.update_progress(task_id, 91,
                    f"[MERGE] 执行合并: {', '.join(merging_nodes)}",
                    log_type="merge_detail")
                await loop.run_in_executor(None, 
                    graph_extractor._merge_graph_nodes, 
                    combined_graph, merging_nodes, model_config, resolution_result.change)
            
            pr = nx.pagerank(combined_graph)
            for node_name, pagerank in pr.items():
                combined_graph.nodes[node_name]["pagerank"] = pagerank
            
            await task_manager.update_progress(task_id, 92,
                f"[INFO] 跨文档合并后: {combined_graph.number_of_nodes()} 个实体，{combined_graph.number_of_edges()} 条关系",
                log_type="cross_doc_merge")
            
            await task_manager.update_progress(task_id, 92.5,
                f"[INFO] 合并统计: {len(resolution_result.change.removed_nodes)} 个实体被删除，{len(resolution_result.change.added_updated_nodes)} 个实体被更新",
                log_type="cross_doc_merge")
            
            if resolution_result.change.removed_nodes:
                await task_manager.update_progress(task_id, 92.6,
                    f"[INFO] 被删除的实体: {', '.join(sorted(resolution_result.change.removed_nodes))}",
                    log_type="merge_detail")
            
            if resolution_result.change.added_updated_nodes:
                await task_manager.update_progress(task_id, 92.7,
                    f"[INFO] 被更新的实体: {', '.join(sorted(resolution_result.change.added_updated_nodes))}",
                    log_type="merge_detail")
        else:
            await task_manager.update_progress(task_id, 92, "[INFO] 未发现跨文档相似实体，跳过消歧", log_type="cross_doc_merge")
        
        merged_entities = {}
        merged_relations = {}
        merged_to_base = {}
        
        for node_name in combined_graph.nodes():
            node_data = dict(combined_graph.nodes[node_name])
            merged_entities[node_name] = node_data
            if "_hugegraph_id" in node_data:
                merged_to_base[node_name] = node_data["_hugegraph_id"]
        
        for src, tgt in combined_graph.edges():
            edge_data = dict(combined_graph.get_edge_data(src, tgt))
            key = tuple(sorted([src, tgt]))
            merged_relations[key] = edge_data
        
        updated_vertices = []
        inserted_vertices = []
        deleted_vertices = []
        
        for en_name, en_info in merged_entities.items():
            hugegraph_id = en_info.get("_hugegraph_id")
            is_existing = en_info.get("_is_existing", False)
            
            if hugegraph_id and is_existing:
                entity_source_ids = en_info.get("source_id", [])
                if isinstance(entity_source_ids, str):
                    entity_source_ids = [entity_source_ids]
                
                involved_in_current_doc = any(
                    str(s).startswith(f"{doc_id}_") or s == doc_id
                    for s in entity_source_ids
                )
                
                if involved_in_current_doc:
                    existing_doc_ids = en_info.get("doc_id", [])
                    if isinstance(existing_doc_ids, str):
                        existing_doc_ids = [existing_doc_ids]
                    new_doc_ids = list(set(existing_doc_ids + [doc_id]))
                    
                    props_to_update = {
                        "description": en_info.get("description", ""),
                        "source_id": en_info.get("source_id", []),
                        "doc_id": doc_id_to_str(new_doc_ids, doc_id),
                        "description_parts": json.dumps(en_info.get("description_parts", []), ensure_ascii=False),
                        "keywords_parts": json.dumps(en_info.get("keywords_parts", []), ensure_ascii=False),
                    }
                    if "pagerank" in en_info:
                        props_to_update["pagerank"] = en_info["pagerank"]
                    
                    updated_vertices.append({
                        "id": hugegraph_id,
                        "name": en_name,
                        "properties": props_to_update
                    })
            else:
                inserted_vertices.append(en_name)
        
        for en_name in name_to_vertex_id.keys():
            if en_name not in merged_entities:
                deleted_vertices.append({
                    "id": name_to_vertex_id[en_name],
                    "name": en_name
                })
        
        updated_edges = []
        inserted_edges = []
        deleted_edges = []
        
        for (src, tgt), rel_info in merged_relations.items():
            hugegraph_id = rel_info.get("_hugegraph_id")
            is_existing = rel_info.get("_is_existing", False)
            
            if hugegraph_id and is_existing:
                edge_source_ids = rel_info.get("source_id", [])
                if isinstance(edge_source_ids, str):
                    edge_source_ids = [edge_source_ids]
                
                involved_in_current_doc = any(
                    str(s).startswith(f"{doc_id}_") or s == doc_id
                    for s in edge_source_ids
                )
                
                if involved_in_current_doc:
                    existing_doc_ids = rel_info.get("doc_id", [])
                    if isinstance(existing_doc_ids, str):
                        existing_doc_ids = [existing_doc_ids]
                    new_doc_ids = list(set(existing_doc_ids + [doc_id]))
                    
                    props_to_update = {
                        "description": rel_info.get("description", ""),
                        "weight": rel_info.get("weight", 1),
                        "source_id": rel_info.get("source_id", []),
                        "doc_id": doc_id_to_str(new_doc_ids, doc_id),
                        "description_parts": json.dumps(rel_info.get("description_parts", []), ensure_ascii=False),
                        "keywords_parts": json.dumps(rel_info.get("keywords_parts", []), ensure_ascii=False),
                        "weight_parts": json.dumps(rel_info.get("weight_parts", []), ensure_ascii=False),
                    }
                    if "keywords" in rel_info:
                        props_to_update["keywords"] = rel_info["keywords"]
                    
                    updated_edges.append({
                        "id": hugegraph_id,
                        "source": src,
                        "target": tgt,
                        "properties": props_to_update
                    })
            else:
                inserted_edges.append((src, tgt))
        
        await task_manager.update_progress(task_id, 93,
            f"[INFO] 准备执行: {len(updated_vertices)} 个实体更新，{len(inserted_vertices)} 个实体插入，{len(deleted_vertices)} 个实体删除",
            log_type="cross_doc_merge")
        
        if updated_vertices:
            await task_manager.update_progress(task_id, 93.5,
                f"[UPDATE] 即将更新的实体: {', '.join([v['name'] for v in updated_vertices])}",
                log_type="merge_detail")
            
            for vertex_update in updated_vertices:
                await loop.run_in_executor(None,
                    hugegraph_client.update_vertex_properties,
                    vertex_update["id"], vertex_update["properties"])
                await task_manager.update_progress(task_id, 93.6,
                    f"[UPDATE] 已更新实体: {vertex_update['name']} (ID: {vertex_update['id'][:20]}...)",
                    log_type="merge_detail")
        
        if updated_edges:
            await task_manager.update_progress(task_id, 93.7,
                f"[UPDATE] 即将更新的关系: {len(updated_edges)} 条",
                log_type="merge_detail")
            
            for edge_update in updated_edges:
                await loop.run_in_executor(None,
                    hugegraph_client.update_edge_properties,
                    edge_update["id"], edge_update["properties"])
                await task_manager.update_progress(task_id, 93.8,
                    f"[UPDATE] 已更新关系: {edge_update['source']} -> {edge_update['target']}",
                    log_type="merge_detail")
        
        if deleted_vertices:
            await task_manager.update_progress(task_id, 93.9,
                f"[DELETE] 即将删除的实体: {', '.join([v['name'] for v in deleted_vertices])}",
                log_type="merge_detail")
            
            for vid_info in deleted_vertices:
                await loop.run_in_executor(None, hugegraph_client.delete_vertex, vid_info["id"])
                await task_manager.update_progress(task_id, 94.0,
                    f"[DELETE] 已删除实体: {vid_info['name']} (ID: {vid_info['id'][:20]}...)",
                    log_type="merge_detail")
        
        await task_manager.update_progress(task_id, 94, "[INFO] 转换新实体为 HugeGraph 格式...", log_type="convert")
        
        new_entities = {name: info for name, info in merged_entities.items() 
                       if not info.get("_is_existing", False)}
        
        new_relations = {}
        for (src, tgt), rel_info in merged_relations.items():
            if not rel_info.get("_is_existing", False):
                key = tuple(sorted([src, tgt]))
                new_relations[key] = rel_info
        
        vertices, edges = await loop.run_in_executor(None,
            graph_extractor.convert_to_graph_format, new_entities, new_relations, doc_id, kb_id)
        
        for vertex in vertices:
            vertex["properties"]["created_at"] = datetime.now().isoformat()
        for edge in edges:
            edge["properties"]["created_at"] = datetime.now().isoformat()
        
        await task_manager.update_progress(task_id, 95, 
            f"[INFO] 准备插入 {len(vertices)} 个新实体和 {len(edges)} 条新关系到 HugeGraph",
            log_type="insert",
            entity_count=len(vertices),
            relation_count=len(edges))
        
        if vertices:
            await task_manager.update_progress(task_id, 95.5,
                f"[INSERT] 即将插入的实体: {', '.join([v['properties']['name'] for v in vertices])}",
                log_type="merge_detail")
        
        await task_manager.check_cancelled(task_id)
        
        await task_manager.update_progress(task_id, 96, "[INFO] 正在批量插入数据到 HugeGraph...", log_type="insert")

        insert_ok = await loop.run_in_executor(None, hugegraph_client.batch_insert, vertices, edges)

        if not insert_ok:
            await task_manager.update_progress(task_id, 97, "[ERROR] HugeGraph 数据插入失败，请检查日志", log_type="error")
            raise Exception("HugeGraph batch_insert 返回 False，数据插入失败")

        await task_manager.update_progress(task_id, 97, "[SUCCESS] 知识图谱数据保存完成", log_type="success")
        
        # 将解析阶段保存的故障树结构化数据入库 HugeGraph（零额外图片 LLM 调用）
        # 仅入库选中分块对应的故障树：若该故障树图片所在分块未被选中，则不写入 HugeGraph
        fault_trees = await loop.run_in_executor(None, postgres_client.get_document_fault_trees, doc_id)
        if fault_trees:
            await task_manager.update_progress(task_id, 97.1,
                f"[FAULT_TREE] 从 PostgreSQL 读取到 {len(fault_trees)} 张故障树结构化数据",
                log_type="info")
            selected_chunk_set = set(selected_chunk_ids)
            skipped_fault_trees = [ft for ft in fault_trees if ft.get("chunk_id") not in selected_chunk_set]
            fault_trees = [ft for ft in fault_trees if ft.get("chunk_id") in selected_chunk_set]
            if skipped_fault_trees:
                await task_manager.update_progress(task_id, 97.1,
                    f"[FAULT_TREE] 跳过 {len(skipped_fault_trees)} 张未选中分块的故障树（不写入 HugeGraph）",
                    log_type="info")
            ft_vertices = []
            ft_edges = []
            for ft_idx, ft in enumerate(fault_trees):
                try:
                    ft_graph = ft.get("graph")
                    if isinstance(ft_graph, str):
                        ft_graph = json.loads(ft_graph)
                    v, e = convert_fault_tree_to_hugegraph(ft_graph or {}, doc_id, kb_id, ft_idx)
                    ft_vertices.extend(v)
                    ft_edges.extend(e)
                except Exception as ex:
                    await task_manager.update_progress(task_id, 97.2,
                        f"[WARNING] 故障树数据转换失败: {str(ex)[:100]}", log_type="warning")
            if fault_trees:
                await task_manager.update_progress(task_id, 97.2,
                    f"[FAULT_TREE] 将 {len(fault_trees)} 张选中分块的故障树图写入 HugeGraph: {len(ft_vertices)} 个实体, {len(ft_edges)} 条关系",
                    log_type="insert")
            if ft_vertices:
                ft_ok = await loop.run_in_executor(None, hugegraph_client.batch_insert, ft_vertices, ft_edges)
                if not ft_ok:
                    await task_manager.update_progress(task_id, 97.4,
                        "[ERROR] 故障树数据写入 HugeGraph 失败", log_type="error")
                else:
                    await task_manager.update_progress(task_id, 97.5,
                        "[SUCCESS] 故障树数据写入 HugeGraph 完成", log_type="success")
            elif not skipped_fault_trees:
                await task_manager.update_progress(task_id, 97.2,
                    "[FAULT_TREE] 选中分块中无故障树结构化数据（解析阶段未检测到）", log_type="info")
        else:
            await task_manager.update_progress(task_id, 97.2,
                "[FAULT_TREE] 本文档无故障树结构化数据（解析阶段未检测到）", log_type="info")
        
        await task_manager.check_cancelled(task_id)
        
        community_reports = {}
        await task_manager.update_progress(task_id, 97.1, "[COMMUNITY] ========== 社区报告生成阶段 ==========", log_type="community")
        
        graph_nodes = combined_graph.number_of_nodes()
        graph_edges = combined_graph.number_of_edges()
        await task_manager.update_progress(task_id, 97.2, 
            f"[COMMUNITY] 当前图谱统计: {graph_nodes} 个节点, {graph_edges} 条边", 
            log_type="community")
        
        if with_community:
            await task_manager.update_progress(task_id, 97.3, "[COMMUNITY] 社区报告生成: 已启用 ✓", log_type="community")
            await task_manager.update_progress(task_id, 97.4, "[COMMUNITY] 开始执行社区检测算法 (Leiden)...", log_type="community")
            
            community_logs = []
            
            def community_log_callback(msg, log_type="community"):
                community_logs.append((msg, log_type))
                print(f"[COMMUNITY LOG COLLECTED] {msg[:50]}...")
            
            community_reports = await loop.run_in_executor(None,
                graph_extractor._detect_communities_and_generate_reports, 
                combined_graph, merged_entities, merged_relations, community_log_callback)
            
            await task_manager.update_progress(task_id, 97.5, 
                f"[COMMUNITY] 社区检测完成，收集到 {len(community_logs)} 条日志", 
                log_type="community")
            
            for msg, log_type in community_logs:
                await task_manager.update_progress(task_id, 97.6, f"[COMMUNITY] {msg}", log_type=log_type)
            
            await task_manager.update_progress(task_id, 97.7, 
                f"[COMMUNITY] 社区报告生成结果: {len(community_reports)} 个报告", 
                log_type="community")
        else:
            await task_manager.update_progress(task_id, 97.3, "[COMMUNITY] 社区报告生成: 已禁用 ✗", log_type="community")
            await task_manager.update_progress(task_id, 97.4, "[COMMUNITY] 用户未开启社区报告生成功能，跳过此阶段", log_type="community")
        
        # 社区报告以 PostgreSQL 为真源：每次重建先清理 PG 旧版，
        # 再清理可重建的 OpenSearch 投影。投影失败会使任务失败，
        # 避免在前端展示一个“已同步”的假成功状态。
        if with_community:
            previous_reports = await loop.run_in_executor(
                None, postgres_client.get_community_reports, kb_id
            )
            await loop.run_in_executor(None, postgres_client.delete_community_reports_by_kb, kb_id)
            try:
                await loop.run_in_executor(
                    None,
                    partial(
                        search_store.delete_by_filter,
                        {"kb_id": kb_id, "knowledge_graph_kwd": "community_report"},
                    ),
                )
            except Exception:
                # 删除投影失败时恢复 PG 旧版，下次任务仍能完整重试。
                for previous in previous_reports:
                    await loop.run_in_executor(
                        None,
                        postgres_client.upsert_community_report,
                        previous["projection_id"],
                        previous,
                    )
                raise
            await task_manager.update_progress(
                task_id, 98.1, "[INFO] 已清除旧的社区报告真源与检索投影", log_type="community"
            )

        if community_reports:
            await task_manager.update_progress(task_id, 98, 
                f"[INFO] 检测到 {len(community_reports)} 个社区，生成报告并索引...", 
                log_type="community",
                community_count=len(community_reports))
            
            for report_id, report_data in community_reports.items():
                report_doc_ids = report_data.get("doc_ids", [doc_id])
                if not report_doc_ids:
                    report_doc_ids = [doc_id]
                if doc_id not in report_doc_ids:
                    report_doc_ids.append(doc_id)
                    
                # 投影 id 必须带 kb_id 前缀，否则不同知识库社区报告
                # 使用相同的 community_{level}_{community_id} id 会互相覆盖
                projection_id = f"{kb_id}_{report_id}"
                    
                report_doc = {
                    "report_id": report_id,
                    "community_id": report_data["community_id"],
                    "entities": report_data["entities"],
                    "report": report_data["report"],
                    "doc_id": report_doc_ids[0] if report_doc_ids else doc_id,
                    "doc_ids": report_doc_ids,
                    "kb_id": kb_id,
                    "knowledge_graph_kwd": "community_report",
                    "projection_id": projection_id,
                    "created_at": datetime.now().isoformat()
                }

                # 真源成功后再刷新检索投影。
                await loop.run_in_executor(
                    None, postgres_client.upsert_community_report, projection_id, report_doc
                )
                await loop.run_in_executor(
                    None, search_store.upsert, projection_id, report_doc
                )
                
                report_title = report_data["report"].get("title", "未命名社区")
                await task_manager.update_progress(task_id, 98.5,
                    f"[INFO]   已索引社区报告: {report_title}",
                    log_type="community")
        else:
            await task_manager.update_progress(task_id, 98, 
                "[COMMUNITY] ⚠️ 未生成任何社区报告，可能原因:", 
                log_type="warning")
            await task_manager.update_progress(task_id, 98.1, 
                "[COMMUNITY]   - 图谱节点数不足（需要至少3个节点才能检测社区）", 
                log_type="warning")
            await task_manager.update_progress(task_id, 98.2, 
                "[COMMUNITY]   - 未形成有效的社区结构（每个社区需要至少2个节点）", 
                log_type="warning")
            await task_manager.update_progress(task_id, 98.3, 
                "[COMMUNITY]   - LLM调用失败或返回格式错误", 
                log_type="warning")
        
        await task_manager.update_progress(task_id, 99, "[INFO] 更新文档状态", log_type="finish")
        
        postgres_client.update_document_status(doc_id, "completed")
        
        await task_manager.update_progress(task_id, 99, 
            f"[INFO] 跨文档合并总结:",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.1, 
            f"[INFO]   - 原有实体: {len(existing_vertices)}",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.2, 
            f"[INFO]   - 新增实体: {len(inserted_vertices)}",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.3, 
            f"[INFO]   - 更新实体: {len(updated_vertices)}",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.4, 
            f"[INFO]   - 删除实体: {len(deleted_vertices)}",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.5, 
            f"[INFO]   - 最终实体总数: {combined_graph.number_of_nodes()}",
            log_type="finish")
        await task_manager.update_progress(task_id, 99.6, 
            f"[INFO]   - 最终关系总数: {combined_graph.number_of_edges()}",
            log_type="finish")
        
        await task_manager.update_progress(task_id, 100, 
            f"[FINISH] 知识图谱生成完成！跨文档合并后共 {combined_graph.number_of_nodes()} 个实体，{combined_graph.number_of_edges()} 条关系",
            log_type="finish",
            entity_count=combined_graph.number_of_nodes(),
            relation_count=combined_graph.number_of_edges())
        await task_manager.update_status(task_id, "completed")
        
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "entities_count": combined_graph.number_of_nodes(),
            "relations_count": combined_graph.number_of_edges(),
            "community_reports_count": len(community_reports) if community_reports else 0,
            "updated_vertices": len(updated_vertices),
            "inserted_vertices": len(inserted_vertices),
            "deleted_vertices": len(deleted_vertices),
            "updated_edges": len(updated_edges),
            "inserted_edges": len(inserted_edges),
            "deleted_edges": len(deleted_edges),
            "merge_details": resolution_result.change.merge_details if resolution_result else [],
            "status": "completed",
            "message": f"知识图谱生成完成，跨文档合并后共 {combined_graph.number_of_nodes()} 个实体，{combined_graph.number_of_edges()} 条关系，生成了 {len(community_reports) if community_reports else 0} 份社区报告"
        })
    
    except asyncio.CancelledError:
        await task_manager.update_progress(task_id, 0, "[CANCEL] 知识图谱生成已被用户取消", log_type="cancel")
        await task_manager.update_status(task_id, "cancelled")
        postgres_client.update_document_status(doc_id, "indexed")
    
    except Exception as e:
        import traceback
        error_detail = f"[ERROR] {str(e)}\n\n{traceback.format_exc()}"
        await task_manager.set_error(task_id, error_detail)
        await task_manager.update_progress(task_id, 0, f"[ERROR] 知识图谱生成失败: {str(e)}", log_type="error")
        await task_manager.update_status(task_id, "failed")
        postgres_client.update_document_status(doc_id, "indexed")


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

        diagrams = diagram_repository.list_diagrams_by_document(doc_id)
        all_latest_are_terminal = True
        for diagram in diagrams:
            revisions_for_diagram = diagram_repository.list_revisions(
                str(diagram["diagram_id"])
            )
            if not revisions_for_diagram:
                all_latest_are_terminal = False
                break
            latest = revisions_for_diagram[0]
            is_active_projection = (
                revision_is_projectable(latest)
                and latest.get("projection_status") == "published"
                and latest.get("is_active")
            )
            validation_status = str(latest.get("validation_status") or "")
            is_safely_isolated = validation_status == VALIDATION_QUARANTINED
            if not (is_active_projection or is_safely_isolated):
                all_latest_are_terminal = False
                break
        final_doc_status = "kg_done" if all_latest_are_terminal else "kg_pending"
        postgres_client.update_icp_document_status(doc_id, final_doc_status)

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
        postgres_client.update_icp_document_status(doc_id, "kg_pending")

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
        postgres_client.update_icp_document_status(doc_id, "kg_pending")


_ICP_DIAGRAM_TYPES = frozenset({"function_locator", "fta", "rbd"})


def _latest_diagram_states(doc_id: str, doc_type: str) -> List[Dict]:
    latest: List[Dict] = []
    for diagram in diagram_repository.list_diagrams_by_document(doc_id):
        if str(diagram.get("diagram_type") or "") != doc_type:
            continue
        revisions = diagram_repository.list_revisions(str(diagram["diagram_id"]))
        if revisions:
            latest.append(dict(revisions[0]))
    return latest


def _latest_diagrams_are_terminal(doc_id: str, doc_type: str) -> bool:
    """Return true when each discovered diagram is published or quarantined.

    No diagram is a terminal, warning-only outcome: the common Document vertex
    and the other explicit structured relations are still a complete rebuildable
    projection.  Missing engineering topology is never fabricated to fill it.
    """

    for revision in _latest_diagram_states(doc_id, doc_type):
        if (
            revision_is_projectable(revision)
            and revision.get("projection_status") == "published"
            and revision.get("is_active")
        ):
            continue
        if str(revision.get("validation_status") or "") == VALIDATION_QUARANTINED:
            continue
        return False
    return True


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
            postgres_client.update_icp_document_status(doc_id, "kg_done")
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
        postgres_client.update_icp_document_status(doc_id, "kg_done")
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
        postgres_client.update_icp_document_status(doc_id, "kg_pending")
    except Exception as exc:
        import traceback

        await task_manager.set_error(
            task_id, f"[ERROR] {exc}\n\n{traceback.format_exc()}"
        )
        await task_manager.update_progress(
            task_id, 0, f"[ERROR] 自动图处理失败: {exc}", log_type="error"
        )
        await task_manager.update_status(task_id, "failed")
        postgres_client.update_icp_document_status(doc_id, "kg_pending")
