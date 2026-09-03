import logging
from typing import List, Dict, Optional, Any

from backend.core.retrieval_pipeline import RetrievalOutcome, build_retrieval_pipeline
from backend.storage.postgres_client import postgres_client
from backend.storage.hugegraph_client import hugegraph_client
from backend.core.embedding import embedding_service

logger = logging.getLogger(__name__)

class RetrievalMode:
    POSTGRES_FULLTEXT = "postgres_fulltext"
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    # Historical mode strings remain API-compatible aliases.
    ES_VECTOR = "es_vector"
    ES_KEYWORD = "es_keyword"
    ES_HYBRID = "es_hybrid"
    KG_ENTITY = "kg_entity"
    KG_GRAPH = "kg_graph"
    HYBRID_ALL = "hybrid_all"
    GRAPH_ENHANCED = "graph_enhanced"
    KG_RELATION_WEIGHTED = "kg_relation_weighted"
    COMMUNITY_REPORT = "community_report"
    RAG_FUSION = "rag_fusion"

class KnowledgeRetriever:
    def __init__(self):
        self.rrf_k = 60
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = build_retrieval_pipeline()
        return self._pipeline

    @staticmethod
    def _filters(kb_id: str = "", kb_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        values = [value for value in (kb_ids or []) if value]
        if values:
            return {"kb_id": values}
        return {"kb_id": kb_id} if kb_id else None

    @staticmethod
    def _search_hit_to_result(hit: Any, retrieval_mode: str) -> Dict[str, Any]:
        if isinstance(hit, dict):
            document = dict(hit.get("document") or hit)
            hit_id = hit.get("id") or document.get("chunk_id") or document.get("id") or ""
            score = hit.get("score", document.get("score", 0.0))
        else:
            document = dict(getattr(hit, "document", {}) or {})
            hit_id = getattr(hit, "id", "") or document.get("chunk_id") or document.get("id") or ""
            score = getattr(hit, "score", document.get("score", 0.0))
        document.setdefault("chunk_id", hit_id)
        document["score"] = score
        document["retrieval_mode"] = retrieval_mode
        return document
    
    def search_postgres_fulltext(self, kb_id: str, query: str, top_k: int = 10) -> List[Dict]:
        try:
            with postgres_client.conn.cursor() as cur:
                cur.execute("""
                    SELECT dc.chunk_id, dc.doc_id, dc.title, dc.content, dc.section,
                           d.file_name as doc_name, dc.chunk_index
                    FROM document_chunks dc
                    JOIN documents d ON dc.doc_id = d.doc_id
                    WHERE d.kb_id = %s AND dc.content_type = 'text'
                    AND (dc.content ILIKE %s OR dc.title ILIKE %s OR dc.section ILIKE %s)
                    ORDER BY dc.chunk_index
                    LIMIT %s
                """, (kb_id, f"%{query}%", f"%{query}%", f"%{query}%", top_k))
                
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "title": row[2],
                        "content": row[3][:500] if row[3] else "",
                        "section": row[4],
                        "doc_name": row[5],
                        "chunk_index": row[6],
                        "retrieval_mode": "postgres_fulltext",
                        "score": 1.0
                    })
                return results
        except Exception as e:
            print(f"PostgreSQL fulltext search failed: {e}")
            return []
    
    def search_vector(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        *,
        embedding_model_id: Optional[str] = None,
        kb_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Backend-neutral dense recall; keyword fallback keeps retrieval usable."""
        try:
            query_vector = embedding_service.embed(query, model_id=embedding_model_id)
            if not query_vector or all(v == 0 for v in query_vector):
                logger.warning("Embedding 返回零向量，回退到关键词搜索")
                return self.search_keyword(kb_id, query, top_k, kb_ids=kb_ids)
            from backend.storage.search_store import search_store

            hits = search_store.vector_search(
                query_vector,
                filters=self._filters(kb_id, kb_ids),
                top_k=top_k,
            )
            return [self._search_hit_to_result(hit, "vector") for hit in hits]
        except Exception as e:
            logger.warning("Vector search failed, falling back to keyword search: %s", e)
            return self.search_keyword(kb_id, query, top_k, kb_ids=kb_ids)

    def search_keyword(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        *,
        kb_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Backend-neutral keyword recall."""
        try:
            from backend.storage.search_store import search_store

            hits = search_store.keyword_search(
                query,
                filters=self._filters(kb_id, kb_ids),
                top_k=top_k,
            )
            return [self._search_hit_to_result(hit, "keyword") for hit in hits]
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
            return []

    def search_hybrid_with_info(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        embedding_model_id: Optional[str] = None,
        kb_ids: Optional[List[str]] = None,
        agent_id: str = "",
        rerank_model_id: str = "",
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalOutcome:
        """Unified backend-neutral retrieval including full-candidate rerank."""
        return self.pipeline.retrieve(
            query=query,
            top_k=top_k,
            filters=filters if filters is not None else self._filters(kb_id, kb_ids),
            embedding_model_id=embedding_model_id,
            agent_id=agent_id,
            rerank_model_id=rerank_model_id,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )

    def search_hybrid(self, kb_id: str, query: str, top_k: int = 10, **kwargs) -> List[Dict]:
        return self.search_hybrid_with_info(kb_id, query, top_k, **kwargs).results

    # Compatibility aliases: public API paths and mode strings remain stable,
    # while the implementation no longer depends on Elasticsearch directly.
    def search_es_vector(self, kb_id: str, query: str, top_k: int = 10) -> List[Dict]:
        return [dict(item, retrieval_mode="es_vector") for item in self.search_vector(kb_id, query, top_k)]

    def search_es_keyword(self, kb_id: str, query: str, top_k: int = 10) -> List[Dict]:
        return [dict(item, retrieval_mode="es_keyword") for item in self.search_keyword(kb_id, query, top_k)]

    def search_es_hybrid(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        embedding_model_id: str = None,
        kb_ids: List[str] = None,
        agent_id: str = "",
        rerank_model_id: str = "",
    ) -> List[Dict]:
        results = self.search_hybrid(
            kb_id,
            query,
            top_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            embedding_model_id=embedding_model_id,
            kb_ids=kb_ids,
            agent_id=agent_id,
            rerank_model_id=rerank_model_id,
        )
        return [dict(item, retrieval_mode="es_hybrid") for item in results]
    
    def search_kg_entity(self, kb_id: str, query: str, top_k: int = 20) -> List[Dict]:
        try:
            entities = hugegraph_client.search_vertices_by_name(kb_id, query, limit=top_k)
            
            results = []
            for entity in entities:
                props = entity.get("properties", {})
                results.append({
                    "entity_id": entity.get("id", ""),
                    "entity_name": props.get("name", ""),
                    "entity_type": props.get("entity_type", ""),
                    "description": props.get("description", "")[:500],
                    "weight": props.get("weight", 0),
                    "pagerank": props.get("pagerank", 0),
                    "doc_id": props.get("doc_id", ""),
                    "retrieval_mode": "kg_entity",
                    "score": float(props.get("pagerank", 0) or 0) + float(props.get("weight", 0) or 0) / 100
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"KG entity search failed: {e}")
            return []
    
    def search_kg_graph(self, kb_id: str, query: str, depth: int = 2, top_k: int = 20) -> List[Dict]:
        try:
            entities = hugegraph_client.search_vertices_by_name(kb_id, query, limit=10)
            
            all_results = []
            seen_entities = set()
            query_lower = query.lower()
            
            for entity in entities:
                entity_id = entity.get("id", "")
                entity_name = entity.get("properties", {}).get("name", "")
                
                if entity_name in seen_entities:
                    continue
                seen_entities.add(entity_name)
                
                neighbors = hugegraph_client.get_neighbors(entity_id, depth=depth, limit=top_k * 2)
                
                for neighbor in neighbors.get("neighbors", []):
                    neighbor_name = neighbor.get("name", "")
                    if neighbor_name not in seen_entities:
                        seen_entities.add(neighbor_name)
                        
                        relation_weight = neighbor.get("weight", 1)
                        relation_keywords = neighbor.get("keywords", [])
                        distance = neighbor.get("distance", 1)
                        relation_type = neighbor.get("relation_type", neighbor.get("relation", ""))
                        
                        keyword_match_score = 0
                        if relation_keywords:
                            for kw in relation_keywords:
                                if query_lower in str(kw).lower():
                                    keyword_match_score += 1
                        
                        distance_factor = 1.0 / distance
                        weight_factor = relation_weight / 10.0
                        keyword_factor = keyword_match_score * 0.5
                        
                        score = distance_factor + weight_factor + keyword_factor
                        
                        all_results.append({
                            "entity_id": neighbor.get("id", ""),
                            "entity_name": neighbor_name,
                            "entity_type": neighbor.get("label", ""),
                            "relation": neighbor.get("relation", ""),
                            "relation_type": relation_type,
                            "source_entity": entity_name,
                            "relation_weight": relation_weight,
                            "relation_keywords": relation_keywords,
                            "distance": distance,
                            "retrieval_mode": "kg_graph",
                            "score": score
                        })
            
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]
        except Exception as e:
            print(f"KG graph search failed: {e}")
            return []
    
    def search_graph_enhanced(self, kb_id: str, query: str, top_k: int = 10) -> List[Dict]:
        try:
            entities = hugegraph_client.search_vertices_by_name(kb_id, query, limit=10)
            
            enhanced_entities = []
            enhanced_keywords = []
            query_lower = query.lower()
            
            for entity in entities:
                entity_id = entity.get("id", "")
                entity_name = entity.get("properties", {}).get("name", "")
                
                if entity_name:
                    neighbors = hugegraph_client.get_neighbors(entity_id, depth=1, limit=20)
                    
                    for neighbor in neighbors.get("neighbors", []):
                        relation_weight = neighbor.get("weight", 1)
                        relation_keywords = neighbor.get("keywords", [])
                        
                        if relation_weight >= 2:
                            enhanced_entities.append(neighbor.get("name", ""))
                        
                        if relation_keywords:
                            for kw in relation_keywords:
                                kw_str = str(kw).strip().lower()
                                if kw_str and kw_str not in enhanced_keywords:
                                    if query_lower in kw_str or kw_str in query_lower:
                                        enhanced_keywords.append(str(kw).strip())
            
            entity_names = [e.get("properties", {}).get("name", "") for e in entities if e.get("properties", {}).get("name")]
            entity_names.extend(enhanced_entities)
            entity_names = list(set(entity_names))
            
            related_chunk_ids = hugegraph_client.get_related_chunks_by_entities(kb_id, entity_names)
            
            if not related_chunk_ids:
                return self.search_hybrid(kb_id, query, top_k)
            
            enhanced_query_parts = [query]
            enhanced_query_parts.extend(entity_names)
            enhanced_query_parts.extend(enhanced_keywords)
            enhanced_query = " ".join(enhanced_query_parts)

            outcome = self.search_hybrid_with_info(
                kb_id,
                enhanced_query,
                top_k,
                filters={"kb_id": kb_id, "chunk_id": related_chunk_ids[:50]},
            )
            return [
                dict(
                    item,
                    retrieval_mode="graph_enhanced",
                    related_entities=entity_names,
                    enhanced_keywords=enhanced_keywords,
                )
                for item in outcome.results
            ]
        except Exception as e:
            print(f"Graph enhanced search failed: {e}")
            return self.search_hybrid(kb_id, query, top_k)
    
    def search_kg_relation_weighted(self, kb_id: str, query: str, top_k: int = 20, 
                                     min_weight: int = 1, depth: int = 2) -> List[Dict]:
        try:
            entities = hugegraph_client.search_vertices_by_name(kb_id, query, limit=10)
            
            all_results = []
            seen_pairs = set()
            query_lower = query.lower()
            
            for entity in entities:
                entity_id = entity.get("id", "")
                entity_name = entity.get("properties", {}).get("name", "")
                
                neighbors = hugegraph_client.get_neighbors(entity_id, depth=depth, limit=top_k * 3)
                
                for neighbor in neighbors.get("neighbors", []):
                    neighbor_name = neighbor.get("name", "")
                    relation_weight = neighbor.get("weight", 1)
                    relation_keywords = neighbor.get("keywords", [])
                    distance = neighbor.get("distance", 1)
                    relation_type = neighbor.get("relation_type", neighbor.get("relation", ""))
                    
                    if relation_weight < min_weight:
                        continue
                    
                    pair_key = tuple(sorted([entity_name, neighbor_name]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    
                    keyword_match_score = 0
                    keyword_matches = []
                    if relation_keywords:
                        for kw in relation_keywords:
                            kw_str = str(kw).strip().lower()
                            if query_lower in kw_str or kw_str in query_lower:
                                keyword_match_score += 1
                                keyword_matches.append(str(kw).strip())
                    
                    distance_factor = 1.0 / distance
                    weight_factor = relation_weight / 5.0
                    keyword_factor = keyword_match_score * 0.8
                    
                    score = distance_factor + weight_factor + keyword_factor
                    
                    all_results.append({
                        "source_entity": entity_name,
                        "target_entity": neighbor_name,
                        "target_entity_id": neighbor.get("id", ""),
                        "target_entity_type": neighbor.get("label", ""),
                        "relation_type": relation_type,
                        "relation_weight": relation_weight,
                        "relation_keywords": relation_keywords,
                        "matching_keywords": keyword_matches,
                        "distance": distance,
                        "retrieval_mode": "kg_relation_weighted",
                        "score": score
                    })
            
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            return all_results[:top_k]
        except Exception as e:
            print(f"KG relation weighted search failed: {e}")
            return []
    
    def _apply_rrf(self, results_list: List[List[Dict]]) -> List[Dict]:
        scores = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                chunk_id = result.get("chunk_id", "")
                if not chunk_id:
                    chunk_id = result.get("entity_id", "")
                
                if chunk_id:
                    if chunk_id not in scores:
                        scores[chunk_id] = {"score": 0.0, "sources": [], "data": {}}
                    
                    rrf_score = 1.0 / (self.rrf_k + rank + 1)
                    scores[chunk_id]["score"] += rrf_score
                    scores[chunk_id]["sources"].append(result.get("retrieval_mode", ""))
                    
                    if not scores[chunk_id]["data"]:
                        scores[chunk_id]["data"] = result
        
        final_results = []
        for chunk_id, info in scores.items():
            result = info["data"]
            result["score"] = info["score"]
            result["retrieval_mode"] = "hybrid_all"
            result["sources"] = info["sources"]
            final_results.append(result)
        
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results
    
    def search_hybrid_all(self, kb_id: str, query: str, top_k: int = 10,
                        chunks_count: int = None, entities_count: int = None) -> Dict:
        try:
            pg_results = self.search_postgres_fulltext(kb_id, query, top_k * 2)
            vector_results = self.search_vector(kb_id, query, top_k * 2)
            keyword_results = self.search_keyword(kb_id, query, top_k * 2)
            kg_entity_results = self.search_kg_entity(kb_id, query, top_k * 2)
            
            results_list = [pg_results, vector_results, keyword_results, kg_entity_results]
            
            final_results = self._apply_rrf(results_list)
            
            chunks_available = len(final_results)
            entities_available = len(kg_entity_results)
            
            use_manual = any(c is not None for c in [chunks_count, entities_count])
            
            if use_manual:
                return self._search_hybrid_all_manual(
                    final_results, kg_entity_results,
                    chunks_available, entities_available,
                    chunks_count, entities_count
                )
            
            kg_context = []
            entity_limit = min(5, entities_available)
            for entity in kg_entity_results[:entity_limit]:
                kg_context.append({
                    "type": "entity",
                    "name": entity.get("entity_name", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "description": entity.get("description", "")
                })
            
            return {
                "chunks": final_results[:top_k],
                "kg_context": kg_context,
                "total_chunks": min(top_k, chunks_available),
                "total_entities": entity_limit
            }
        except Exception as e:
            print(f"Hybrid all search failed: {e}")
            return {"chunks": [], "kg_context": [], "total_chunks": 0, "total_entities": 0}
    
    def _search_hybrid_all_manual(self, final_results, kg_entity_results,
                                   chunks_available, entities_available,
                                   chunks_count, entities_count):
        try:
            c_count = 0
            e_count = 0
            
            if chunks_count is not None and chunks_available > 0:
                c_count = min(chunks_count, chunks_available)
            
            if entities_count is not None and entities_available > 0:
                e_count = min(entities_count, entities_available)
            
            final_chunks = final_results[:c_count]
            
            kg_context = []
            for entity in kg_entity_results[:e_count]:
                kg_context.append({
                    "type": "entity",
                    "name": entity.get("entity_name", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "description": entity.get("description", "")
                })
            
            return {
                "chunks": final_chunks,
                "kg_context": kg_context,
                "total_chunks": len(final_chunks),
                "total_entities": len(kg_context)
            }
        except Exception as e:
            print(f"Hybrid all manual search failed: {e}")
            return {"chunks": [], "kg_context": [], "total_chunks": 0, "total_entities": 0}
    
    def search(self, kb_id: str, query: str, mode: str = "es_hybrid", 
               top_k: int = 10, **kwargs) -> Any:
        modes_map = {
            RetrievalMode.POSTGRES_FULLTEXT: lambda: self.search_postgres_fulltext(kb_id, query, top_k),
            RetrievalMode.VECTOR: lambda: self.search_vector(
                kb_id,
                query,
                top_k,
                embedding_model_id=kwargs.get("embedding_model_id"),
                kb_ids=kwargs.get("kb_ids"),
            ),
            RetrievalMode.KEYWORD: lambda: self.search_keyword(
                kb_id, query, top_k, kb_ids=kwargs.get("kb_ids")
            ),
            RetrievalMode.HYBRID: lambda: self.search_hybrid(
                kb_id,
                query,
                top_k,
                keyword_weight=kwargs.get("keyword_weight", 0.5),
                vector_weight=kwargs.get("vector_weight", 0.5),
                embedding_model_id=kwargs.get("embedding_model_id"),
                kb_ids=kwargs.get("kb_ids"),
                agent_id=kwargs.get("agent_id", ""),
                rerank_model_id=kwargs.get("rerank_model_id", ""),
            ),
            RetrievalMode.ES_VECTOR: lambda: self.search_es_vector(kb_id, query, top_k),
            RetrievalMode.ES_KEYWORD: lambda: self.search_es_keyword(kb_id, query, top_k),
            RetrievalMode.ES_HYBRID: lambda: self.search_es_hybrid(
                kb_id, query, top_k, 
                kwargs.get('keyword_weight', 0.5), 
                kwargs.get('vector_weight', 0.5)
            ),
            RetrievalMode.KG_ENTITY: lambda: self.search_kg_entity(kb_id, query, top_k),
            RetrievalMode.KG_GRAPH: lambda: self.search_kg_graph(
                kb_id, query, 
                kwargs.get('depth', 2), 
                top_k
            ),
            RetrievalMode.HYBRID_ALL: lambda: self.search_hybrid_all(
                kb_id, query, top_k,
                chunks_count=kwargs.get('chunks_count'),
                entities_count=kwargs.get('entities_count')
            ),
            RetrievalMode.GRAPH_ENHANCED: lambda: self.search_graph_enhanced(kb_id, query, top_k),
            RetrievalMode.KG_RELATION_WEIGHTED: lambda: self.search_kg_relation_weighted(
                kb_id, query, top_k,
                kwargs.get('min_weight', 1),
                kwargs.get('depth', 2)
            ),
            RetrievalMode.COMMUNITY_REPORT: lambda: self.search_community_reports(kb_id, query, top_k),
            RetrievalMode.RAG_FUSION: lambda: self.search_rag_fusion(
                kb_id, query, top_k,
                chunks_count=kwargs.get('chunks_count'),
                entities_count=kwargs.get('entities_count'),
                reports_count=kwargs.get('reports_count')
            )
        }
        
        if mode in modes_map:
            return modes_map[mode]()
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
    
    def search_community_reports(self, kb_id: str, query: str, top_k: int = 5) -> List[Dict]:
        try:
            from backend.storage.search_store import search_store

            hits = search_store.search_community_reports(kb_id=kb_id, query=query, top_k=top_k)

            results = []
            for hit in hits:
                if isinstance(hit, dict):
                    source = dict(hit.get("document") or hit)
                    hit_score = hit.get("score", source.get("score", 0.0))
                else:
                    source = dict(getattr(hit, "document", {}) or {})
                    hit_score = getattr(hit, "score", source.get("score", 0.0))
                report = source.get("report", {})
                stale = source.get("stale_kwd", "N")
                # SearchStore owns stale filtering/weighting; do not apply the
                # content-stale penalty a second time in the retrieval layer.
                score = float(hit_score or 0.0)

                results.append({
                    "report_id": source.get("report_id", ""),
                    "community_id": source.get("community_id", ""),
                    "title": report.get("title", ""),
                    "summary": report.get("summary", ""),
                    "rating": report.get("rating", 0),
                    "rating_explanation": report.get("rating_explanation", ""),
                    "findings": report.get("findings", []),
                    "entities": source.get("entities", []),
                    "doc_id": source.get("doc_id", ""),
                    "doc_ids": source.get("doc_ids", []),
                    "created_at": source.get("created_at", ""),
                    "stale_kwd": stale,
                    "retrieval_mode": "community_report",
                    "score": score
                })

            return results
        except Exception as e:
            print(f"Community report search failed: {e}")
            return []
    
    def search_rag_fusion(self, kb_id: str, query: str, top_k: int = 10,
                          chunks_count: int = None, entities_count: int = None, 
                          reports_count: int = None) -> Dict:
        try:
            retrieval_outcome = self.search_hybrid_with_info(kb_id, query, top_k)
            chunk_results = retrieval_outcome.results
            retrieval_info = retrieval_outcome.info()
            kg_entity_results = self.search_kg_entity(kb_id, query, top_k * 2)
            community_report_results = self.search_community_reports(kb_id, query, top_k * 2)
            
            for r in chunk_results:
                r["data_type"] = "chunk"
            for r in kg_entity_results:
                r["data_type"] = "entity"
            for r in community_report_results:
                r["data_type"] = "community_report"
            
            chunks_available = len(chunk_results)
            entities_available = len(kg_entity_results)
            reports_available = len(community_report_results)
            
            if chunks_available == 0 and entities_available == 0 and reports_available == 0:
                return {
                    "chunks": [],
                    "entities": [],
                    "community_reports": [],
                    "total_chunks": 0,
                    "total_entities": 0,
                    "total_reports": 0,
                    "retrieval_info": retrieval_info,
                }
            
            use_manual = any(c is not None for c in [chunks_count, entities_count, reports_count])
            
            if use_manual:
                result = self._search_rag_fusion_manual(
                    chunk_results, kg_entity_results, community_report_results,
                    chunks_available, entities_available, reports_available,
                    chunks_count, entities_count, reports_count
                )
                result["retrieval_info"] = retrieval_info
                return result
            
            total_avail = chunks_available + entities_available + reports_available
            
            if total_avail <= top_k:
                return {
                    "chunks": chunk_results,
                    "entities": kg_entity_results,
                    "community_reports": community_report_results,
                    "total_chunks": chunks_available,
                    "total_entities": entities_available,
                    "total_reports": reports_available,
                    "retrieval_info": retrieval_info,
                }
            
            has_chunks = chunks_available > 0
            has_entities = entities_available > 0
            has_reports = reports_available > 0
            type_count = sum([has_chunks, has_entities, has_reports])
            
            if type_count == 1:
                if has_chunks:
                    final_chunks = chunk_results[:top_k]
                    return {"chunks": final_chunks, "entities": [], "community_reports": [],
                            "total_chunks": len(final_chunks), "total_entities": 0, "total_reports": 0,
                            "retrieval_info": retrieval_info}
                elif has_entities:
                    final_entities = kg_entity_results[:top_k]
                    return {"chunks": [], "entities": final_entities, "community_reports": [],
                            "total_chunks": 0, "total_entities": len(final_entities), "total_reports": 0,
                            "retrieval_info": retrieval_info}
                else:
                    final_reports = community_report_results[:top_k]
                    return {"chunks": [], "entities": [], "community_reports": final_reports,
                            "total_chunks": 0, "total_entities": 0, "total_reports": len(final_reports),
                            "retrieval_info": retrieval_info}
            
            base_weights = {"chunks": 0.5, "entities": 0.3, "reports": 0.2}
            active_types = {}
            if has_chunks: active_types["chunks"] = base_weights["chunks"]
            if has_entities: active_types["entities"] = base_weights["entities"]
            if has_reports: active_types["reports"] = base_weights["reports"]
            
            total_weight = sum(active_types.values())
            weights = {k: v / total_weight for k, v in active_types.items()}
            
            small_type_threshold = 5
            
            chunks_quota = 0
            entities_quota = 0
            reports_quota = 0
            
            if has_chunks:
                chunks_quota = min(chunks_available, max(1, round(top_k * weights["chunks"])))
                if chunks_available <= small_type_threshold:
                    chunks_quota = chunks_available
            if has_entities:
                entities_quota = min(entities_available, max(1, round(top_k * weights["entities"])))
                if entities_available <= small_type_threshold:
                    entities_quota = entities_available
            if has_reports:
                reports_quota = min(reports_available, max(1, round(top_k * weights["reports"])))
                if reports_available <= small_type_threshold:
                    reports_quota = reports_available
            
            total_quota = chunks_quota + entities_quota + reports_quota
            
            if total_quota > top_k:
                excess = total_quota - top_k
                while excess > 0:
                    reducible = []
                    if has_chunks and chunks_quota > 1 and chunks_available > small_type_threshold:
                        reducible.append(("chunks", chunks_quota))
                    if has_entities and entities_quota > 1 and entities_available > small_type_threshold:
                        reducible.append(("entities", entities_quota))
                    if has_reports and reports_quota > 1 and reports_available > small_type_threshold:
                        reducible.append(("reports", reports_quota))
                    if not reducible:
                        break
                    reducible.sort(key=lambda x: x[1], reverse=True)
                    for name, _ in reducible:
                        if excess <= 0:
                            break
                        if name == "chunks" and chunks_quota > 1:
                            chunks_quota -= 1
                            excess -= 1
                        elif name == "entities" and entities_quota > 1:
                            entities_quota -= 1
                            excess -= 1
                        elif name == "reports" and reports_quota > 1:
                            reports_quota -= 1
                            excess -= 1
            
            chunks_count = min(chunks_quota, chunks_available) if has_chunks else 0
            entities_count = min(entities_quota, entities_available) if has_entities else 0
            reports_count = min(reports_quota, reports_available) if has_reports else 0
            
            remaining = top_k - chunks_count - entities_count - reports_count
            
            unused_chunks = chunks_available - chunks_count if has_chunks else 0
            unused_entities = entities_available - entities_count if has_entities else 0
            unused_reports = reports_available - reports_count if has_reports else 0
            
            redist_order = []
            if unused_reports > 0:
                redist_order.append(("reports", unused_reports))
            if unused_entities > 0:
                redist_order.append(("entities", unused_entities))
            if unused_chunks > 0:
                redist_order.append(("chunks", unused_chunks))
            
            idx = 0
            while remaining > 0 and redist_order:
                type_name, avail = redist_order[idx % len(redist_order)]
                if type_name == "reports" and unused_reports > 0:
                    reports_count += 1
                    unused_reports -= 1
                    remaining -= 1
                elif type_name == "entities" and unused_entities > 0:
                    entities_count += 1
                    unused_entities -= 1
                    remaining -= 1
                elif type_name == "chunks" and unused_chunks > 0:
                    chunks_count += 1
                    unused_chunks -= 1
                    remaining -= 1
                idx += 1
                if idx > 1000:
                    break
            
            final_chunks = chunk_results[:chunks_count]
            final_entities = kg_entity_results[:entities_count]
            final_reports = community_report_results[:reports_count]
            
            return {
                "chunks": final_chunks,
                "entities": final_entities,
                "community_reports": final_reports,
                "total_chunks": len(final_chunks),
                "total_entities": len(final_entities),
                "total_reports": len(final_reports),
                "retrieval_info": retrieval_info,
            }
        except Exception as e:
            print(f"RAG fusion search failed: {e}")
            return {
                "chunks": [],
                "entities": [],
                "community_reports": [],
                "total_chunks": 0,
                "total_entities": 0,
                "total_reports": 0,
                "retrieval_info": {
                    "reranked": False,
                    "fallback": f"RAG Fusion 检索失败: {e}",
                },
            }
    
    def _search_rag_fusion_manual(self, chunk_results, kg_entity_results,
                                   community_report_results, chunks_available, 
                                   entities_available, reports_available,
                                   chunks_count, entities_count, reports_count):
        try:
            has_chunks = chunks_available > 0
            has_entities = entities_available > 0
            has_reports = reports_available > 0
            
            c_count = 0
            e_count = 0
            r_count = 0
            
            if has_chunks:
                if chunks_count is not None:
                    c_count = min(chunks_count, chunks_available)
            if has_entities:
                if entities_count is not None:
                    e_count = min(entities_count, entities_available)
            if has_reports:
                if reports_count is not None:
                    r_count = min(reports_count, reports_available)
            
            final_chunks = chunk_results[:c_count]
            final_entities = kg_entity_results[:e_count]
            final_reports = community_report_results[:r_count]
            
            return {
                "chunks": final_chunks,
                "entities": final_entities,
                "community_reports": final_reports,
                "total_chunks": len(final_chunks),
                "total_entities": len(final_entities),
                "total_reports": len(final_reports)
            }
        except Exception as e:
            print(f"RAG fusion manual search failed: {e}")
            return {
                "chunks": [],
                "entities": [],
                "community_reports": [],
                "total_chunks": 0,
                "total_entities": 0,
                "total_reports": 0
            }
    
    def get_knowledge_base_stats(self, kb_id: str) -> Dict:
        try:
            documents = postgres_client.get_documents_by_kb(kb_id)
            total_docs = len(documents)
            
            total_chunks = 0
            for doc in documents:
                total_chunks += doc.get("total_chunks", 0)
            
            kg_vertices = hugegraph_client.get_vertices_by_kb(kb_id)
            kg_edges = hugegraph_client.get_edges_by_kb(kb_id)
            
            return {
                "kb_id": kb_id,
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_entities": len(kg_vertices),
                "total_relations": len(kg_edges)
            }
        except Exception as e:
            print(f"Get KB stats failed: {e}")
            return {
                "kb_id": kb_id,
                "total_documents": 0,
                "total_chunks": 0,
                "total_entities": 0,
                "total_relations": 0
            }

knowledge_retriever = KnowledgeRetriever()
