from elasticsearch import Elasticsearch, exceptions
from elasticsearch.exceptions import ApiError
from typing import Dict, List, Optional, Any, Tuple
from backend.config.config import config
import json
import time

class ESClient:
    def __init__(self):
        self.client = None
        self._connect()
    
    def _connect(self):
        if self.client is None:
            try:
                client_kwargs = {
                    "verify_certs": False,
                    "ssl_show_warn": False,
                    "request_timeout": 60,
                    "max_retries": 5,
                    "retry_on_timeout": True
                }
                if config.ES_USERNAME and config.ES_PASSWORD:
                    client_kwargs["basic_auth"] = (config.ES_USERNAME, config.ES_PASSWORD)
                
                self.client = Elasticsearch(
                    f"{config.ES_SCHEME}://{config.ES_HOST}:{config.ES_PORT}",
                    **client_kwargs
                )
                
                info = self.client.info()
                print(f"ES cluster info: {info.get('cluster_name')}, version: {info.get('version', {}).get('number')}")
                ping_result = True
                print(f"ES connection successful")
                
                self._ensure_index_template_exists()
                self._ensure_icp_mapping()
                print("ES connection established successfully")
            except Exception as e:
                print(f"ES connection failed: {e}, using mock client")
                self.client = MockESClient()
    
    def _ensure_index_template_exists(self):
        try:
            template_name = "knowledge_graph_template"
            # 始终写入完整模板（幂等覆盖）：保证新索引具备 ICP 类型字段与中文分析器
            # （存量旧模板可能缺少 ICP keyword 字段，导致新建索引时这些字段被动态映射为 text）
            template = {
                "index_patterns": ["knowledge_graph_*"],
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "1s",
                    "analysis": {
                        "analyzer": {
                            "chinese_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
                            },
                            "chinese_search_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "project_id": {"type": "keyword"},
                        "kb_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "doc_name": {"type": "text", "analyzer": "chinese_analyzer", "search_analyzer": "chinese_search_analyzer"},
                        "content": {"type": "text", "analyzer": "chinese_analyzer", "search_analyzer": "chinese_search_analyzer"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": config.EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "knowledge_graph_kwd": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "entity_kwd": {"type": "keyword"},
                        "entity_type_kwd": {"type": "keyword"},
                        "from_entity_kwd": {"type": "keyword"},
                        "to_entity_kwd": {"type": "keyword"},
                        "weight_int": {"type": "integer"},
                        "rank_flt": {"type": "float"},
                        "entities_kwd": {"type": "keyword"},
                        "removed_kwd": {"type": "keyword"},
                        "available_int": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "stale_kwd": {"type": "keyword"},
                        "stale_reason": {"type": "text"},
                        "stale_at": {"type": "date"},
                        "section": {"type": "text", "analyzer": "chinese_analyzer", "search_analyzer": "chinese_search_analyzer"},
                        "title": {"type": "text", "analyzer": "chinese_analyzer", "search_analyzer": "chinese_search_analyzer"},
                        "summary": {"type": "text", "analyzer": "chinese_analyzer", "search_analyzer": "chinese_search_analyzer"},
                        "content_type": {"type": "keyword"},
                        "chapter": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "table_count": {"type": "integer"},
                        "image_count": {"type": "integer"},
                        "language": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "author": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "doc_type_kwd": {"type": "keyword"},
                        "business_id_kwd": {"type": "keyword"},
                        "pg_id_kwd": {"type": "keyword"},
                        "hmc_code_kwd": {"type": "keyword"},
                        "lcn_kwd": {"type": "keyword"},
                        "hamc_kwd": {"type": "keyword"},
                        "hsc_kwd": {"type": "keyword"},
                        "severity_kwd": {"type": "keyword"},
                        "product_kwd": {"type": "keyword"},
                        "api_reportable_kwd": {"type": "keyword"},
                        "platform_test_kwd": {"type": "keyword"},
                        "parent_kwd": {"type": "keyword"},
                        "node_type_kwd": {"type": "keyword"},
                        "probability_flt": {"type": "float"}
                    }
                }
            }
            if hasattr(self.client.indices, 'put_template'):
                self.client.indices.put_template(name=template_name, body=template)
            else:
                index_name = "knowledge_graph_default"
                if not self.client.indices.exists(index=index_name):
                    self.client.indices.create(index=index_name, body=template)
            print("Index template created successfully")
        except Exception as e:
            print(f"Failed to create index template: {e}")
    
    def get_main_index_name(self, tenant_id: str = "default") -> str:
        return f"knowledge_graph_{tenant_id}"
    
    def create_project_alias(self, tenant_id: str, project_id: str):
        alias_name = f"kg_{tenant_id}_{project_id}"
        index_name = self.get_main_index_name(tenant_id)
        
        self._ensure_index_exists(index_name)
        
        try:
            if self.client.indices.exists_alias(name=alias_name):
                return
            
            self.client.indices.put_alias(
                index=index_name,
                name=alias_name,
                body={
                    "filter": {
                        "term": {"project_id": project_id}
                    }
                }
            )
        except ApiError as e:
            print(f"Failed to create project alias: {e}")
    
    def create_kb_alias(self, tenant_id: str, project_id: str, kb_id: str):
        alias_name = f"kg_{tenant_id}_{project_id}_{kb_id}"
        index_name = self.get_main_index_name(tenant_id)
        
        self._ensure_index_exists(index_name)
        
        try:
            if self.client.indices.exists_alias(name=alias_name):
                return
            
            self.client.indices.put_alias(
                index=index_name,
                name=alias_name,
                body={
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"project_id": project_id}},
                                {"term": {"kb_id": kb_id}}
                            ]
                        }
                    }
                }
            )
        except ApiError as e:
            print(f"Failed to create kb alias: {e}")
    
    def _ensure_index_exists(self, index_name: str):
        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name)
        except Exception as e:
            print(f"Failed to create index: {e}")
    
    def create_index(self, index_name: str):
        self._ensure_index_exists(index_name)
    
    def index_document(self, tenant_id: str, project_id: str, kb_id: str, doc_id: str, 
                      chunk_id: str, content: str, content_vector: List[float], 
                      content_type: str = "text", **kwargs):
        index_name = self.get_main_index_name(tenant_id)
        self._ensure_index_exists(index_name)
        
        doc = {
            "project_id": project_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "content": content,
            "content_vector": content_vector,
            "content_type": content_type,
            "knowledge_graph_kwd": "chunk",
            "available_int": 0,
            "removed_kwd": "N",
            "created_at": kwargs.get("created_at"),
            "updated_at": kwargs.get("updated_at"),
            "title": kwargs.get("title", ""),
            "section": kwargs.get("section", "")
        }
        
        doc.update(kwargs)
        
        try:
            self.client.index(index=index_name, id=chunk_id, body=doc)
        except ApiError as e:
            print(f"Failed to index document: {e}")
    
    def search_by_vector(self, tenant_id: str, kb_id: str, query_vector: List[float], 
                         k: int = 10, filters: Dict = None) -> List[Dict]:
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"removed_kwd": "N"}}
                    ]
                }
            },
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10
            },
            "size": k,
            "_source": ["content", "title", "section", "chapter", "doc_id", "doc_name", "content_type"]
        }
        
        if filters:
            for key, value in filters.items():
                query_body["query"]["bool"]["must"].append({"term": {key: value}})
        
        try:
            result = self.client.search(index=index_name, body=query_body)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except ApiError as e:
            print(f"Vector search failed: {e}")
            return []
    
    def search_by_keyword(self, tenant_id: str, kb_id: str, keyword: str, 
                          size: int = 20) -> List[Dict]:
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"removed_kwd": "N"}},
                        {
                            "multi_match": {
                                "query": keyword,
                                "fields": ["content^3", "title^2", "section", "chapter"],
                                "analyzer": "chinese_analyzer",
                                "search_analyzer": "chinese_search_analyzer"
                            }
                        }
                    ]
                }
            },
            "size": size,
            "_source": ["content", "title", "section", "chapter", "doc_id", "doc_name", "content_type"]
        }
        
        try:
            result = self.client.search(index=index_name, body=query_body)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except ApiError as e:
            print(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, tenant_id: str, kb_id: str, query: str, 
                      query_vector: List[float] = None, k: int = 10, 
                      keyword_weight: float = 0.5, vector_weight: float = 0.5) -> List[Dict]:
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"removed_kwd": "N"}}
                    ],
                    "should": []
                }
            },
            "size": k,
            "_source": ["content", "title", "section", "chapter", "doc_id", "doc_name", "content_type"]
        }
        
        if query:
            query_body["query"]["bool"]["should"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "title^2", "section", "chapter"],
                    "analyzer": "chinese_analyzer",
                    "search_analyzer": "chinese_search_analyzer",
                    "boost": keyword_weight
                }
            })
        
        if query_vector:
            query_body["knn"] = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10,
                "boost": vector_weight
            }
        
        try:
            result = self.client.search(index=index_name, body=query_body)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except ApiError as e:
            print(f"Hybrid search failed: {e}")
            return []
    
    def get_by_doc_id(self, tenant_id: str, kb_id: str, doc_id: str) -> List[Dict]:
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"doc_id": doc_id}},
                        {"term": {"removed_kwd": "N"}}
                    ]
                }
            },
            "size": 1000,
            "_source": ["chunk_id", "content", "title", "section", "content_type", "doc_name"]
        }
        
        try:
            result = self.client.search(index=index_name, body=query_body)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except ApiError as e:
            print(f"Get by doc_id failed: {e}")
            return []
    
    def delete_by_doc_id(self, tenant_id: str, kb_id: str, doc_id: str):
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"doc_id": doc_id}}
                    ]
                }
            }
        }
        
        try:
            self.client.delete_by_query(index=index_name, body=query_body)
        except ApiError as e:
            print(f"Delete by doc_id failed: {e}")
    
    def delete_by_kb_id(self, tenant_id: str, kb_id: str):
        index_name = self.get_main_index_name(tenant_id)
        
        query_body = {
            "query": {"term": {"kb_id": kb_id}}
        }
        
        try:
            self.client.delete_by_query(index=index_name, body=query_body)
            alias_name = f"kg_{tenant_id}_default_{kb_id}"
            if self.client.indices.exists_alias(name=alias_name):
                self.client.indices.delete_alias(index=index_name, name=alias_name)
        except ApiError as e:
            print(f"Delete by kb_id failed: {e}")
    
    def _ensure_icp_mapping(self):
        """对存量索引增量添加 ICP 类型字段（只增不改，幂等）。"""
        props = {
            "doc_type_kwd": {"type": "keyword"},
            "business_id_kwd": {"type": "keyword"},
            "pg_id_kwd": {"type": "keyword"},
            "hmc_code_kwd": {"type": "keyword"},
            "lcn_kwd": {"type": "keyword"},
            "hamc_kwd": {"type": "keyword"},
            "hsc_kwd": {"type": "keyword"},
            "severity_kwd": {"type": "keyword"},
            "product_kwd": {"type": "keyword"},
            "api_reportable_kwd": {"type": "keyword"},
            "platform_test_kwd": {"type": "keyword"},
            "parent_kwd": {"type": "keyword"},
            "node_type_kwd": {"type": "keyword"},
            "probability_flt": {"type": "float"}
        }
        try:
            self.client.indices.put_mapping(
                index="knowledge_graph_*",
                body={"properties": props}
            )
        except Exception as e:
            print(f"Failed to add ICP mapping: {e}")

    def search_structured(self, tenant_id: str, doc_type: str, query: str,
                          query_vector: List[float] = None, k: int = 10,
                          filters: Dict = None) -> List[Dict]:
        """按 doc_type 过滤的混合检索（替代 kb_id 过滤）。filters 为 {keyword字段: 值}。"""
        index_name = self.get_main_index_name(tenant_id)

        must = [
            {"term": {"doc_type_kwd": doc_type}},
            {"term": {"removed_kwd": "N"}}
        ]
        should = []
        if query:
            should.append({
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "title^2", "section", "chapter"],
                    "analyzer": "chinese_analyzer",
                    "boost": 0.5
                }
            })
        if filters:
            for key, value in filters.items():
                if value in (None, "", False):
                    continue
                if isinstance(value, bool):
                    value = "true" if value else "false"
                must.append({"term": {key: value}})

        query_body = {
            "query": {"bool": {"must": must, "should": should}},
            "size": k,
            "_source": ["content", "title", "section", "chapter", "doc_id", "doc_name",
                        "content_type", "doc_type_kwd", "business_id_kwd", "pg_id_kwd",
                        "hmc_code_kwd", "lcn_kwd", "hamc_kwd", "hsc_kwd", "severity_kwd",
                        "api_reportable_kwd", "platform_test_kwd", "parent_kwd", "node_type_kwd"]
        }
        if query_vector:
            query_body["knn"] = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10,
                "boost": 0.5
            }

        try:
            result = self.client.search(index=index_name, body=query_body)
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except ApiError as e:
            print(f"Structured search failed: {e}")
            return []

    def get_doc_vectors(self, tenant_id: str, doc_id: str, page: int = 1, size: int = 50) -> Dict:
        """某文档的向量化记录（查看页用），分页返回 content + chunk_id。"""
        index_name = self.get_main_index_name(tenant_id)
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"removed_kwd": "N"}}
                    ]
                }
            },
            "sort": [{"updated_at": "desc"}],
            "from": (page - 1) * size,
            "size": size,
            "_source": ["chunk_id", "content", "title", "section", "content_type",
                        "doc_type_kwd", "business_id_kwd", "parent_kwd", "node_type_kwd"]
        }
        try:
            result = self.client.search(index=index_name, body=query_body)
            hits = result["hits"]["hits"]
            return {
                "total": result["hits"]["total"]["value"] if isinstance(result["hits"].get("total"), dict) else result["hits"].get("total", 0),
                "page": page,
                "size": size,
                "vectors": [hit["_source"] for hit in hits]
            }
        except ApiError as e:
            print(f"Get doc vectors failed: {e}")
            return {"total": 0, "page": page, "size": size, "vectors": []}

    def delete_vectors_by_doc(self, tenant_id: str, doc_id: str):
        """按 doc_id 删除全部向量记录（不依赖 kb_id）。"""
        index_name = self.get_main_index_name(tenant_id)
        query_body = {"query": {"term": {"doc_id": doc_id}}}
        try:
            self.client.delete_by_query(index=index_name, body=query_body)
        except ApiError as e:
            print(f"Delete vectors by doc failed: {e}")

    def get_stats(self, tenant_id: str) -> Dict:
        index_name = self.get_main_index_name(tenant_id)
        
        try:
            stats = self.client.indices.stats(index=index_name)
            return {
                "documents": stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"]
            }
        except ApiError as e:
            print(f"Get stats failed: {e}")
            return {"documents": 0, "size_bytes": 0}

    def search_community_reports(self, index_name: str, kb_id: str, doc_id: str = None) -> List[Dict]:
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "community_report"}}
                    ]
                }
            },
            "size": 100,
            "_source": ["report_id", "community_id", "entities", "report", "doc_id", "doc_ids",
                        "kb_id", "created_at", "stale_kwd", "stale_reason", "stale_at"]
        }
        
        if doc_id:
            # doc_ids 在 ES 中被动态映射为 text 类型（带 .keyword 子字段），
            # text 字段会对 UUID 按连字符分词，导致 terms 精确匹配失败。
            # 这里使用 doc_ids.keyword 子字段做精确匹配。
            query_body["query"]["bool"]["must"].append({"terms": {"doc_ids.keyword": [doc_id]}})

        try:
            result = self.client.search(index=index_name, body=query_body)
            reports = []
            for hit in result["hits"]["hits"]:
                report = hit["_source"]
                report["es_doc_id"] = hit["_id"]
                reports.append(report)
            return reports
        except ApiError as e:
            print(f"Community reports search failed: {e}")
            return []

class MockESIndices:
    def __init__(self, data):
        self.data = data
    
    def exists(self, index):
        return index in self.data
    
    def create(self, index, body=None):
        if index not in self.data:
            self.data[index] = {}
    
    def delete(self, index):
        if index in self.data:
            del self.data[index]
    
    def exists_template(self, name):
        return True
    
    def put_template(self, name, body):
        pass
    
    def exists_alias(self, name):
        return name in self.data.get("aliases", {})
    
    def put_alias(self, index, name, body):
        if "aliases" not in self.data:
            self.data["aliases"] = {}
        self.data["aliases"][name] = {"index": index, "filter": body.get("filter", {})}
    
    def delete_alias(self, index, name):
        if "aliases" in self.data and name in self.data["aliases"]:
            del self.data["aliases"][name]
    
    def stats(self, index):
        count = 0
        if index in self.data:
            count = len(self.data[index])
        return {"_all": {"primaries": {"docs": {"count": count}, "store": {"size_in_bytes": count * 1000}}}}

class MockESClient:
    def __init__(self):
        self.data = {}
        self.indices = MockESIndices(self.data)
    
    def ping(self):
        return True
    
    def index(self, index, id, body):
        if index not in self.data:
            self.data[index] = {}
        self.data[index][id] = body
    
    def _get_field_value(self, doc, field_name):
        if field_name.endswith(".keyword"):
            base_field = field_name[:-8]
            return doc.get(base_field, doc.get(field_name))
        if "." in field_name:
            parts = field_name.split(".")
            value = doc
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
                if value is None:
                    return None
            return value
        return doc.get(field_name)
    
    def _compute_cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def search(self, index, body=None, **kwargs):
        if index not in self.data:
            return {"hits": {"hits": []}}
        
        hits = []
        docs = list(self.data[index].items())
        
        if body:
            query = body.get("query", {})
            bool_query = query.get("bool", {})
            must_clauses = bool_query.get("must", [])
            should_clauses = bool_query.get("should", [])
            knn_query = body.get("knn")
            
            filtered_docs = []
            
            for doc_id, doc in docs:
                match = True
                for clause in must_clauses:
                    if "term" in clause:
                        for key, value in clause["term"].items():
                            field_value = self._get_field_value(doc, key)
                            if field_value != value:
                                match = False
                                break
                    elif "terms" in clause:
                        for key, values in clause["terms"].items():
                            field_value = self._get_field_value(doc, key)
                            if isinstance(field_value, list):
                                if not any(v in field_value for v in values):
                                    match = False
                                    break
                            else:
                                if field_value not in values:
                                    match = False
                                    break
                    elif "multi_match" in clause:
                        keyword = clause["multi_match"]["query"]
                        fields = clause["multi_match"]["fields"]
                        boost = clause.get("boost", 1.0)
                        found = False
                        for field in fields:
                            field_name = field.replace("^", "")
                            field_value = str(self._get_field_value(doc, field_name))
                            if keyword.lower() in field_value.lower():
                                found = True
                                break
                        if not found:
                            match = False
                if match:
                    filtered_docs.append((doc_id, doc))
            
            if knn_query:
                query_vector = knn_query.get("query_vector", [])
                field_name = knn_query.get("field", "content_vector")
                k = knn_query.get("k", 10)
                boost = knn_query.get("boost", 1.0)
                
                scored_docs = []
                for doc_id, doc in filtered_docs:
                    doc_vector = doc.get(field_name, [])
                    similarity = self._compute_cosine_similarity(query_vector, doc_vector)
                    if similarity > 0:
                        scored_docs.append((doc_id, doc, similarity * boost))
                
                scored_docs.sort(key=lambda x: x[2], reverse=True)
                hits = [{"_source": doc, "_score": score} for doc_id, doc, score in scored_docs[:k]]
            elif should_clauses:
                scored_docs = []
                for doc_id, doc in filtered_docs:
                    score = 0.0
                    for clause in should_clauses:
                        if "multi_match" in clause:
                            keyword = clause["multi_match"]["query"]
                            fields = clause["multi_match"]["fields"]
                            boost = clause.get("boost", 1.0)
                            for field in fields:
                                field_name = field.replace("^", "").replace("3", "").replace("2", "")
                                field_value = str(self._get_field_value(doc, field_name))
                                if keyword.lower() in field_value.lower():
                                    score += boost
                    if score > 0:
                        scored_docs.append((doc_id, doc, score))
                
                scored_docs.sort(key=lambda x: x[2], reverse=True)
                size = body.get("size", 10)
                hits = [{"_id": doc_id, "_source": doc, "_score": score} for doc_id, doc, score in scored_docs[:size]]
            else:
                size = body.get("size", 10)
                hits = [{"_id": doc_id, "_source": doc, "_score": 1.0} for doc_id, doc in filtered_docs[:size]]
        else:
            size = body.get("size", 10) if body else 10
            hits = [{"_id": doc_id, "_source": doc, "_score": 1.0} for doc_id, doc in docs[:size]]
        
        return {"hits": {"hits": hits}}
    
    def delete(self, index, id):
        if index in self.data and id in self.data[index]:
            del self.data[index][id]
    
    def update(self, index, id, body):
        if index in self.data and id in self.data[index]:
            doc = self.data[index][id]
            doc_data = body.get("doc", {})
            for key, value in doc_data.items():
                doc[key] = value
            self.data[index][id] = doc
            return {"result": "updated"}
        return {"result": "not_found"}
    
    def delete_by_query(self, index, body=None, query=None):
        if index not in self.data:
            return
        
        actual_query = query or body
        if actual_query:
            if isinstance(actual_query, dict) and "query" in actual_query:
                query_dict = actual_query.get("query", {})
            else:
                query_dict = actual_query
            
            if "bool" in query_dict:
                bool_query = query_dict.get("bool", {})
                must_clauses = bool_query.get("must", [])
                
                keys_to_delete = []
                for doc_id, doc in self.data[index].items():
                    match = True
                    for clause in must_clauses:
                        if "term" in clause:
                            for key, value in clause["term"].items():
                                field_value = self._get_field_value(doc, key)
                                if field_value != value:
                                    match = False
                                    break
                        elif "terms" in clause:
                            for key, values in clause["terms"].items():
                                field_value = self._get_field_value(doc, key)
                                if isinstance(field_value, list):
                                    if not any(v in field_value for v in values):
                                        match = False
                                        break
                                else:
                                    if field_value not in values:
                                        match = False
                                        break
                    if match:
                        keys_to_delete.append(doc_id)
                
                for key in keys_to_delete:
                    del self.data[index][key]
            
            elif "term" in query_dict:
                keys_to_delete = []
                for doc_id, doc in self.data[index].items():
                    match = True
                    for key, value in query_dict["term"].items():
                        field_value = self._get_field_value(doc, key)
                        if field_value != value:
                            match = False
                            break
                    if match:
                        keys_to_delete.append(doc_id)
                
                for key in keys_to_delete:
                    del self.data[index][key]
    
    def scroll(self, scroll_id, scroll):
        return {"hits": {"hits": []}}
    
    def clear_scroll(self, scroll_id):
        pass
    
    def search_community_reports(self, index_name: str, kb_id: str, doc_id: str = None) -> List[Dict]:
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "community_report"}}
                    ]
                }
            },
            "size": 100
        }
        
        if doc_id:
            query_body["query"]["bool"]["must"].append({"terms": {"doc_ids.keyword": [doc_id]}})

        try:
            result = self.search(index_name, body=query_body)
            reports = []
            for hit in result["hits"]["hits"]:
                report = hit["_source"]
                report["es_doc_id"] = hit["_id"]
                reports.append(report)
            return reports
        except Exception as e:
            print(f"Mock community reports search failed: {e}")
            return []

es_client = ESClient()