from fastapi import APIRouter, Query, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from backend.core.retrieval import knowledge_retriever, RetrievalMode
from backend.storage.postgres_client import postgres_client

router = APIRouter(prefix="/knowledge-bases", tags=["retrieval"])

@router.get("/retrieval-modes")
async def get_retrieval_modes():
    return {
        "modes": [
            {
                "mode": RetrievalMode.POSTGRES_FULLTEXT,
                "name": "PostgreSQL全文检索",
                "description": "使用PostgreSQL的LIKE查询进行关键词匹配，适用于精确关键词搜索"
            }, 
            {
                "mode": RetrievalMode.ES_VECTOR,
                "name": "OpenSearch向量检索",
                "description": "使用语义向量进行相似度匹配，适用于语义理解和模糊查询"
            },
            {
                "mode": RetrievalMode.ES_KEYWORD,
                "name": "OpenSearch关键词检索",
                "description": "使用BM25算法进行关键词匹配，适用于精确匹配场景"
            },
            {
                "mode": RetrievalMode.ES_HYBRID,
                "name": "OpenSearch混合检索",
                "description": "结合关键词检索和向量检索，可调节权重比例"
            },
            {
                "mode": RetrievalMode.KG_ENTITY,
                "name": "知识图谱实体检索",
                "description": "直接从知识图谱中检索相关实体，适用于实体识别和关联查询"
            },
            {
                "mode": RetrievalMode.KG_GRAPH,
                "name": "知识图谱图遍历",
                "description": "通过图遍历发现实体之间的关系路径，结合关系权重和关键词进行排序，适用于关系推理"
            },
            {
                "mode": RetrievalMode.GRAPH_ENHANCED,
                "name": "图增强检索",
                "description": "先用知识图谱发现相关实体，利用关系权重和关键词增强查询，再进行向量检索"
            },
            {
                "mode": RetrievalMode.KG_RELATION_WEIGHTED,
                "name": "关系权重检索",
                "description": "基于增强模式的关系权重和关键词进行检索，优先返回高权重且关键词匹配的关系路径"
            },
            {
                "mode": RetrievalMode.HYBRID_ALL,
                "name": "全源融合检索",
                "description": "融合PostgreSQL、OpenSearch向量、OpenSearch关键词三种检索结果，使用RRF算法重新排序"
            },
            {
                "mode": RetrievalMode.COMMUNITY_REPORT,
                "name": "社区报告检索",
                "description": "从社区报告中检索相关内容，适用于获取文档的结构化摘要和关键发现"
            },
            {
                "mode": RetrievalMode.RAG_FUSION,
                "name": "RAG融合检索（推荐）",
                "description": "融合文档分块向量、知识图谱实体和社区报告，提供全面的检索结果"
            }
        ],
        "description": "多种检索方式可根据不同场景选择使用"
    }

@router.get("/{kb_id}/retrieval-modes")
async def get_retrieval_modes_by_kb(kb_id: str):
    """带知识库 ID 的检索模式列表（前端统一走此路径）"""
    return await get_retrieval_modes()

class SearchRequest(BaseModel):
    query: str
    mode: str = "es_hybrid"
    top_k: int = 10
    keyword_weight: float = 0.5
    vector_weight: float = 0.5
    depth: int = 2
    min_weight: int = 1
    chunks_count: Optional[int] = None
    entities_count: Optional[int] = None
    reports_count: Optional[int] = None

class SearchResponse(BaseModel):
    kb_id: str
    query: str
    mode: str
    results: List[Dict]
    kg_context: Optional[List[Dict]] = None
    community_reports: Optional[List[Dict]] = None
    total_results: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_reports: int = 0

class ChunkSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    section: str
    doc_name: str
    content_type: str = "text"
    chunk_index: int = 0
    retrieval_mode: str
    score: float = 0.0
    sources: Optional[List[str]] = None
    related_entities: Optional[List[str]] = None

class EntitySearchResult(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    weight: int = 0
    pagerank: float = 0.0
    doc_id: str = ""
    retrieval_mode: str
    score: float = 0.0

class GraphSearchResult(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    relation: str
    source_entity: str
    retrieval_mode: str
    score: float = 0.0

class KBStatsResponse(BaseModel):
    kb_id: str
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relations: int

@router.post("/{kb_id}/search", response_model=SearchResponse)
async def search_knowledge_base(
    kb_id: str,
    request: SearchRequest = Body(...)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        result = knowledge_retriever.search(
            kb_id=kb_id,
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            keyword_weight=request.keyword_weight,
            vector_weight=request.vector_weight,
            depth=request.depth,
            min_weight=request.min_weight,
            chunks_count=request.chunks_count,
            entities_count=request.entities_count,
            reports_count=request.reports_count
        )
        
        if isinstance(result, dict) and "chunks" in result:
            total = result.get("total_chunks", len(result["chunks"]))
            total += result.get("total_entities", 0)
            total += result.get("total_reports", 0)
            return {
                "kb_id": kb_id,
                "query": request.query,
                "mode": request.mode,
                "results": result["chunks"],
                "kg_context": result.get("kg_context"),
                "community_reports": result.get("community_reports"),
                "total_results": total,
                "total_chunks": result.get("total_chunks", len(result["chunks"])),
                "total_entities": result.get("total_entities", 0),
                "total_reports": result.get("total_reports", 0)
            }
        else:
            return {
                "kb_id": kb_id,
                "query": request.query,
                "mode": request.mode,
                "results": result,
                "total_results": len(result)
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/{kb_id}/search/postgres-fulltext")
async def search_postgres_fulltext(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_postgres_fulltext(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "postgres_fulltext",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/es-vector")
async def search_es_vector(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_es_vector(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "es_vector",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/es-keyword")
async def search_es_keyword(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_es_keyword(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "es_keyword",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/es-hybrid")
async def search_es_hybrid(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    keyword_weight: float = Query(0.5, ge=0.0, le=1.0),
    vector_weight: float = Query(0.5, ge=0.0, le=1.0)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_es_hybrid(kb_id, query, top_k, keyword_weight, vector_weight)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "es_hybrid",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/kg-entity")
async def search_kg_entity(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(20, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_kg_entity(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "kg_entity",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/kg-graph")
async def search_kg_graph(
    kb_id: str,
    query: str = Query(...),
    depth: int = Query(2, ge=1, le=5),
    top_k: int = Query(20, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_kg_graph(kb_id, query, depth, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "kg_graph",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/graph-enhanced")
async def search_graph_enhanced(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_graph_enhanced(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "graph_enhanced",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/hybrid-all")
async def search_hybrid_all(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    chunks_count: int = Query(None, ge=0, le=100),
    entities_count: int = Query(None, ge=0, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    result = knowledge_retriever.search_hybrid_all(
        kb_id, query, top_k,
        chunks_count=chunks_count,
        entities_count=entities_count
    )
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "hybrid_all",
        "results": result.get("chunks", []),
        "kg_context": result.get("kg_context", []),
        "total_chunks": result.get("total_chunks", 0),
        "total_entities": result.get("total_entities", 0),
        "total_results": result.get("total_chunks", 0) + result.get("total_entities", 0)
    }

@router.get("/{kb_id}/stats", response_model=KBStatsResponse)
async def get_knowledge_base_stats(kb_id: str):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    stats = knowledge_retriever.get_knowledge_base_stats(kb_id)
    return stats

@router.get("/{kb_id}/entities/top")
async def get_top_entities(
    kb_id: str,
    limit: int = Query(20, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from backend.storage.hugegraph_client import hugegraph_client
    entities = hugegraph_client.get_top_entities_by_pagerank(kb_id, limit)
    return {
        "kb_id": kb_id,
        "entities": entities,
        "total": len(entities)
    }

@router.get("/{kb_id}/entities/type")
async def get_entities_by_type(
    kb_id: str,
    entity_type: str = Query(...),
    limit: int = Query(100, ge=1, le=500)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from backend.storage.hugegraph_client import hugegraph_client
    entities = hugegraph_client.search_by_entity_type(kb_id, entity_type, limit)
    return {
        "kb_id": kb_id,
        "entity_type": entity_type,
        "entities": entities,
        "total": len(entities)
    }

@router.get("/{kb_id}/entities/{entity_name}/relations")
async def get_entity_relations(
    kb_id: str,
    entity_name: str,
    limit: int = Query(50, ge=1, le=200)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from backend.storage.hugegraph_client import hugegraph_client
    relations = hugegraph_client.get_entity_relations(kb_id, entity_name, limit)
    return {
        "kb_id": kb_id,
        "entity_name": entity_name,
        "relations": relations,
        "total": len(relations)
    }

@router.get("/{kb_id}/entities/path")
async def get_shortest_path(
    kb_id: str,
    source: str = Query(...),
    target: str = Query(...)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from backend.storage.hugegraph_client import hugegraph_client
    paths = hugegraph_client.get_shortest_path(kb_id, source, target)
    return {
        "kb_id": kb_id,
        "source": source,
        "target": target,
        "paths": paths,
        "total_paths": len(paths)
    }

@router.get("/{kb_id}/search/kg-relation-weighted")
async def search_kg_relation_weighted(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(20, ge=1, le=100),
    min_weight: int = Query(1, ge=1, le=10),
    depth: int = Query(2, ge=1, le=5)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_kg_relation_weighted(kb_id, query, top_k, min_weight, depth)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "kg_relation_weighted",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/community-report")
async def search_community_report(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    results = knowledge_retriever.search_community_reports(kb_id, query, top_k)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "community_report",
        "results": results,
        "total_results": len(results)
    }

@router.get("/{kb_id}/search/rag-fusion")
async def search_rag_fusion(
    kb_id: str,
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    chunks_count: int = Query(None, ge=0, le=100),
    entities_count: int = Query(None, ge=0, le=100),
    reports_count: int = Query(None, ge=0, le=100)
):
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    result = knowledge_retriever.search_rag_fusion(
        kb_id, query, top_k,
        chunks_count=chunks_count,
        entities_count=entities_count,
        reports_count=reports_count
    )
    total = result.get("total_chunks", 0) + result.get("total_entities", 0) + result.get("total_reports", 0)
    return {
        "kb_id": kb_id,
        "query": query,
        "mode": "rag_fusion",
        "chunks": result.get("chunks", []),
        "entities": result.get("entities", []),
        "community_reports": result.get("community_reports", []),
        "total_chunks": result.get("total_chunks", 0),
        "total_entities": result.get("total_entities", 0),
        "total_reports": result.get("total_reports", 0),
        "total_results": total
    }
