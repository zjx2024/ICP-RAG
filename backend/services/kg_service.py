"""知识图谱服务：图谱生成后台任务（从 documents.py 路由抽取）。

保持原实现行为完全一致（含 GraphExtractor 抽取、跨文档实体合并、社区报告生成）。
"""
import asyncio
import json
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import networkx as nx

from backend.core.task_manager import task_manager
from backend.graphrag.graph_extractor import GraphExtractor, doc_id_to_str
from backend.services.document_service import convert_fault_tree_to_hugegraph
from backend.storage.es_client import es_client
from backend.storage.hugegraph_client import hugegraph_client
from backend.storage.postgres_client import postgres_client
from backend.api.routers._deletion_helpers import parse_json_parts

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
        
        if community_reports:
            await task_manager.update_progress(task_id, 98, 
                f"[INFO] 检测到 {len(community_reports)} 个社区，生成报告并索引...", 
                log_type="community",
                community_count=len(community_reports))
            
            index_name = es_client.get_main_index_name("default")
            
            try:
                await loop.run_in_executor(None,
                    partial(es_client.client.delete_by_query, 
                        index=index_name, 
                        body={"query": {"bool": {"must": [
                            {"term": {"kb_id": kb_id}},
                            {"term": {"knowledge_graph_kwd": "community_report"}}
                        ]}}},
                        refresh=True))
                await task_manager.update_progress(task_id, 98.1,
                    "[INFO] 已清除旧的社区报告",
                    log_type="community")
            except Exception as e:
                await task_manager.update_progress(task_id, 98.1,
                    f"[WARNING] 清除旧社区报告失败: {str(e)}",
                    log_type="warning")
            
            for report_id, report_data in community_reports.items():
                report_doc_ids = report_data.get("doc_ids", [doc_id])
                if not report_doc_ids:
                    report_doc_ids = [doc_id]
                if doc_id not in report_doc_ids:
                    report_doc_ids.append(doc_id)
                    
                # ES 文档 id 必须带 kb_id 前缀，否则不同知识库社区报告
                # 使用相同的 community_{level}_{community_id} id 会互相覆盖
                es_doc_id = f"{kb_id}_{report_id}"
                    
                report_doc = {
                    "report_id": report_id,
                    "community_id": report_data["community_id"],
                    "entities": report_data["entities"],
                    "report": report_data["report"],
                    "doc_id": report_doc_ids[0] if report_doc_ids else doc_id,
                    "doc_ids": report_doc_ids,
                    "kb_id": kb_id,
                    "knowledge_graph_kwd": "community_report",
                    "created_at": datetime.now().isoformat()
                }
                
                await loop.run_in_executor(None, 
                    partial(es_client.client.index, index=index_name, id=es_doc_id, body=report_doc))
                
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

async def generate_icp_kg_background(task_id: str, doc_id: str, doc_type: str, file_name: str,
                                     model_config: Optional[Dict] = None):
    """两阶段流程阶段二：用户人工确认三元组后，将 document_fault_trees 中的图数据写入 HugeGraph。"""
    try:
        loop = asyncio.get_event_loop()
        await task_manager.update_status(task_id, "generating_kg")
        await task_manager.update_progress(task_id, 10, "[START] 开始生成知识图谱")

        fault_trees = await loop.run_in_executor(None, postgres_client.get_document_fault_trees, doc_id)
        if not fault_trees:
            raise Exception("该文档没有已确认的故障树/框图三元组数据，请先解析")

        await task_manager.update_progress(task_id, 30, f"[INFO] 读取到 {len(fault_trees)} 张图的结构化数据")

        ft_vertices = []
        ft_edges = []
        for ft_idx, ft in enumerate(fault_trees):
            try:
                ft_graph = ft.get("graph")
                if isinstance(ft_graph, str):
                    ft_graph = json.loads(ft_graph)
                v, e = convert_fault_tree_to_hugegraph(ft_graph or {}, doc_id, doc_type, ft_idx)
                ft_vertices.extend(v)
                ft_edges.extend(e)
            except Exception as ex:
                await task_manager.update_progress(task_id, 30 + int((ft_idx + 1) / len(fault_trees) * 30),
                    f"[WARNING] 第 {ft_idx + 1} 张图转换失败: {str(ex)[:100]}", log_type="warning")

        if not ft_vertices:
            raise Exception("未能从三元组转换出有效节点（图数据为空）")

        await task_manager.update_progress(task_id, 70,
            f"[INFO] 转换完成: {len(ft_vertices)} 节点 / {len(ft_edges)} 边，写入 HugeGraph...")
        ok = await loop.run_in_executor(None, hugegraph_client.batch_insert, ft_vertices, ft_edges)
        if not ok:
            raise Exception("HugeGraph 写入失败")

        postgres_client.update_icp_document_status(doc_id, "kg_done")

        await task_manager.update_progress(task_id, 100,
            f"[FINISH] 知识图谱生成完成: {len(ft_vertices)} 节点 / {len(ft_edges)} 边")
        await task_manager.update_status(task_id, "completed")
        await task_manager.set_result(task_id, {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "vertices": len(ft_vertices),
            "edges": len(ft_edges),
            "status": "kg_done",
            "message": f"知识图谱生成完成：{len(ft_vertices)} 节点 / {len(ft_edges)} 边",
        })

    except asyncio.CancelledError:
        await task_manager.update_progress(task_id, 0, "[CANCEL] 知识图谱生成已被用户取消", log_type="cancel")
        await task_manager.update_status(task_id, "cancelled")
        postgres_client.update_icp_document_status(doc_id, "kg_pending")

    except Exception as e:
        import traceback
        error_detail = f"[ERROR] {str(e)}\n\n{traceback.format_exc()}"
        await task_manager.set_error(task_id, error_detail)
        await task_manager.update_progress(task_id, 0, f"[ERROR] 知识图谱生成失败: {str(e)}", log_type="error")
        await task_manager.update_status(task_id, "failed")
        postgres_client.update_icp_document_status(doc_id, "kg_pending")
