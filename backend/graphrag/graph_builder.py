import uuid
import json
import networkx as nx
from difflib import SequenceMatcher
from typing import Dict, Tuple, List, Optional
from backend.storage.es_client import es_client
from backend.core.embedding import embedding_service
from backend.config.config import config

class GraphBuilder:
    def __init__(self):
        self.es_client = es_client
        self.embedding_service = embedding_service
        self._similarity_threshold = 0.85
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        return SequenceMatcher(None, name1, name2).ratio()
    
    def _resolve_entity_name(self, entity_name: str, existing_nodes: List[str]) -> Optional[str]:
        for node in existing_nodes:
            similarity = self._calculate_similarity(entity_name, node)
            if similarity >= self._similarity_threshold:
                return node
        return None
    
    def build_subgraph(self, doc_id: str, entities: Dict, relations: Dict, kb_id: str = "", 
                       project_id: str = "", tenant_id: str = "default") -> nx.Graph:
        subgraph = nx.Graph()
        subgraph.graph["source_id"] = [doc_id]
        subgraph.graph["kb_id"] = kb_id
        subgraph.graph["project_id"] = project_id
        subgraph.graph["tenant_id"] = tenant_id
        
        for entity_name, attrs in entities.items():
            node_attrs = {
                "entity_type": attrs.get("entity_type", "Entity"),
                "description": attrs.get("description", ""),
                "source_id": attrs.get("source_id", [doc_id]),
                "kb_id": kb_id,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "doc_id": [doc_id],
                "weight": attrs.get("weight", 1),
                "probability": attrs.get("probability", 0.0),
                "level": attrs.get("level", 0),
                "created_at": attrs.get("created_at", ""),
                "keywords": attrs.get("keywords", []),
                "pagerank": attrs.get("pagerank", 0.0),
                "communities": attrs.get("communities", "")
            }
            subgraph.add_node(entity_name, **node_attrs)
        
        for (src, tgt), attrs in relations.items():
            if src in subgraph.nodes and tgt in subgraph.nodes:
                edge_attrs = {
                    "description": attrs.get("description", ""),
                    "keywords": attrs.get("keywords", []),
                    "weight": attrs.get("weight", 1),
                    "source_id": attrs.get("source_id", [doc_id]),
                    "kb_id": kb_id,
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                    "doc_id": [doc_id],
                    "probability": attrs.get("probability", 0.0),
                    "gate_type": attrs.get("gate_type", ""),
                    "gate_group": attrs.get("gate_group", ""),
                    "threshold": attrs.get("threshold", 0),
                    "created_at": attrs.get("created_at", "")
                }
                subgraph.add_edge(src, tgt, **edge_attrs)
        
        return subgraph
    
    def _merge_node_properties(self, old_attrs: Dict, new_attrs: Dict) -> Dict:
        merged = {}
        
        merged["entity_type"] = old_attrs.get("entity_type", new_attrs.get("entity_type", "Entity"))
        
        old_desc = old_attrs.get("description", "")
        new_desc = new_attrs.get("description", "")
        merged["description"] = self._merge_descriptions(old_desc, new_desc)
        
        old_source_id = old_attrs.get("source_id", [])
        new_source_id = new_attrs.get("source_id", [])
        merged["source_id"] = list(set(old_source_id + new_source_id))
        
        old_weight = old_attrs.get("weight", 0)
        new_weight = new_attrs.get("weight", 0)
        merged["weight"] = old_weight + new_weight
        
        old_prob = old_attrs.get("probability", 0.0)
        new_prob = new_attrs.get("probability", 0.0)
        merged["probability"] = max(old_prob, new_prob)
        
        old_level = old_attrs.get("level", 999)
        new_level = new_attrs.get("level", 999)
        merged["level"] = min(old_level, new_level)
        
        old_keywords = old_attrs.get("keywords", [])
        new_keywords = new_attrs.get("keywords", [])
        merged["keywords"] = list(set(old_keywords + new_keywords))
        
        merged["kb_id"] = old_attrs.get("kb_id", new_attrs.get("kb_id", ""))
        merged["project_id"] = old_attrs.get("project_id", new_attrs.get("project_id", ""))
        merged["tenant_id"] = old_attrs.get("tenant_id", new_attrs.get("tenant_id", "default"))
        merged["doc_id"] = list(set(old_attrs.get("doc_id", []) + new_attrs.get("doc_id", [])))
        
        merged["pagerank"] = old_attrs.get("pagerank", 0.0)
        merged["communities"] = old_attrs.get("communities", "")
        merged["created_at"] = old_attrs.get("created_at", "")
        
        return merged
    
    def _merge_edge_properties(self, old_attrs: Dict, new_attrs: Dict) -> Dict:
        merged = {}
        
        old_desc = old_attrs.get("description", "")
        new_desc = new_attrs.get("description", "")
        merged["description"] = self._merge_descriptions(old_desc, new_desc)
        
        old_weight = old_attrs.get("weight", 0)
        new_weight = new_attrs.get("weight", 0)
        merged["weight"] = old_weight + new_weight
        
        old_keywords = old_attrs.get("keywords", [])
        new_keywords = new_attrs.get("keywords", [])
        merged["keywords"] = list(set(old_keywords + new_keywords))
        
        old_source_id = old_attrs.get("source_id", [])
        new_source_id = new_attrs.get("source_id", [])
        merged["source_id"] = list(set(old_source_id + new_source_id))
        
        old_prob = old_attrs.get("probability", 0.0)
        new_prob = new_attrs.get("probability", 0.0)
        merged["probability"] = max(old_prob, new_prob)
        
        merged["gate_type"] = old_attrs.get("gate_type", new_attrs.get("gate_type", ""))
        merged["gate_group"] = old_attrs.get("gate_group", new_attrs.get("gate_group", ""))
        merged["threshold"] = min(old_attrs.get("threshold", 999), new_attrs.get("threshold", 999))
        
        merged["kb_id"] = old_attrs.get("kb_id", new_attrs.get("kb_id", ""))
        merged["project_id"] = old_attrs.get("project_id", new_attrs.get("project_id", ""))
        merged["tenant_id"] = old_attrs.get("tenant_id", new_attrs.get("tenant_id", "default"))
        merged["doc_id"] = list(set(old_attrs.get("doc_id", []) + new_attrs.get("doc_id", [])))
        merged["created_at"] = old_attrs.get("created_at", "")
        
        return merged
    
    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        if not desc1:
            return desc2
        if not desc2:
            return desc1
        if desc2 in desc1:
            return desc1
        if desc1 in desc2:
            return desc2
        return desc1 + "<SEP>" + desc2
    
    def merge_graphs(self, old_graph: nx.Graph, subgraph: nx.Graph) -> nx.Graph:
        if old_graph is None:
            return subgraph
        
        existing_nodes = list(old_graph.nodes())
        
        for node_name, attrs in subgraph.nodes(data=True):
            resolved_name = self._resolve_entity_name(node_name, existing_nodes)
            
            if resolved_name and resolved_name != node_name:
                if old_graph.has_node(resolved_name):
                    merged_attrs = self._merge_node_properties(
                        old_graph.nodes[resolved_name], attrs
                    )
                    old_graph.nodes[resolved_name].update(merged_attrs)
                
                for src, tgt, edge_attrs in list(subgraph.edges(node_name, data=True)):
                    new_src = resolved_name if src == node_name else src
                    new_tgt = resolved_name if tgt == node_name else tgt
                    
                    if new_src in old_graph.nodes and new_tgt in old_graph.nodes:
                        existing_edge = old_graph.get_edge_data(new_src, new_tgt)
                        if existing_edge:
                            merged_edge_attrs = self._merge_edge_properties(existing_edge, edge_attrs)
                            old_graph[new_src][new_tgt].update(merged_edge_attrs)
                        else:
                            old_graph.add_edge(new_src, new_tgt, **edge_attrs)
            
            elif not old_graph.has_node(node_name):
                old_graph.add_node(node_name, **attrs)
            else:
                merged_attrs = self._merge_node_properties(old_graph.nodes[node_name], attrs)
                old_graph.nodes[node_name].update(merged_attrs)
        
        for src, tgt, attrs in subgraph.edges(data=True):
            if src in old_graph.nodes and tgt in old_graph.nodes:
                existing_edge = old_graph.get_edge_data(src, tgt)
                if existing_edge is None:
                    old_graph.add_edge(src, tgt, **attrs)
                else:
                    merged_edge_attrs = self._merge_edge_properties(existing_edge, attrs)
                    old_graph[src][tgt].update(merged_edge_attrs)
        
        if "source_id" not in old_graph.graph:
            old_graph.graph["source_id"] = []
        old_graph.graph["source_id"] = list(
            set(old_graph.graph["source_id"] + subgraph.graph.get("source_id", []))
        )
        
        old_graph.graph["kb_id"] = old_graph.graph.get("kb_id", subgraph.graph.get("kb_id", ""))
        old_graph.graph["project_id"] = old_graph.graph.get("project_id", subgraph.graph.get("project_id", ""))
        old_graph.graph["tenant_id"] = old_graph.graph.get("tenant_id", subgraph.graph.get("tenant_id", "default"))
        
        try:
            pr = nx.pagerank(old_graph)
            for node_name, pagerank in pr.items():
                old_graph.nodes[node_name]["pagerank"] = pagerank
        except Exception as e:
            print(f"Pagerank calculation failed: {e}")
        
        return old_graph
    
    def save_to_index(self, tenant_id: str, kb_id: str, graph: nx.Graph, entities: Dict, relations: Dict):
        try:
            index_name = self.es_client.get_main_index_name(tenant_id)
            
            for node_name, attrs in graph.nodes(data=True):
                content = f"{node_name} - {attrs.get('description', '')}"
                try:
                    embedding = self.embedding_service.embed(content)
                except Exception:
                    embedding = [0.0] * config.EMBEDDING_DIM
                
                chunk_id = str(uuid.uuid4())
                chunk = {
                    "project_id": attrs.get("project_id", ""),
                    "kb_id": kb_id,
                    "doc_id": attrs.get("doc_id", ""),
                    "chunk_id": chunk_id,
                    "doc_name": "",
                    "content": content,
                    "content_vector": embedding,
                    "content_type": "entity",
                    "knowledge_graph_kwd": "entity",
                    "entity_kwd": node_name,
                    "entity_type_kwd": attrs.get("entity_type", "Entity"),
                    "source_id": attrs.get("source_id", []),
                    "weight_int": attrs.get("weight", 1),
                    "rank_flt": attrs.get("pagerank", 0),
                    "available_int": 0,
                    "removed_kwd": "N",
                    "title": node_name,
                    "section": attrs.get("description", "")[:100]
                }
                self.es_client.index_document(tenant_id, attrs.get("project_id", ""), kb_id, 
                                             attrs.get("doc_id", ""), chunk_id, content, embedding,
                                             content_type="entity", **chunk)
            
            for src, tgt, attrs in graph.edges(data=True):
                content = f"{src}->{tgt}: {attrs.get('description', '')}"
                try:
                    embedding = self.embedding_service.embed(content)
                except Exception:
                    embedding = [0.0] * config.EMBEDDING_DIM
                
                chunk_id = str(uuid.uuid4())
                chunk = {
                    "project_id": attrs.get("project_id", ""),
                    "kb_id": kb_id,
                    "doc_id": attrs.get("doc_id", ""),
                    "chunk_id": chunk_id,
                    "doc_name": "",
                    "content": content,
                    "content_vector": embedding,
                    "content_type": "relation",
                    "knowledge_graph_kwd": "relation",
                    "from_entity_kwd": src,
                    "to_entity_kwd": tgt,
                    "source_id": attrs.get("source_id", []),
                    "weight_int": attrs.get("weight", 1),
                    "available_int": 0,
                    "removed_kwd": "N",
                    "title": f"{src}-{tgt}",
                    "section": attrs.get("description", "")[:100]
                }
                self.es_client.index_document(tenant_id, attrs.get("project_id", ""), kb_id, 
                                             attrs.get("doc_id", ""), chunk_id, content, embedding,
                                             content_type="relation", **chunk)
            
            print(f"Graph saved successfully: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        except Exception as e:
            print(f"Failed to save graph to index: {e}")
    
    def delete_graph_by_doc_id(self, tenant_id: str, kb_id: str, doc_id: str):
        try:
            self.es_client.delete_by_doc_id(tenant_id, kb_id, doc_id)
            print(f"Deleted graph data for doc_id: {doc_id}")
        except Exception as e:
            print(f"Failed to delete graph by doc_id: {e}")
    
    def convert_to_hugegraph_format(self, graph: nx.Graph) -> Tuple[List[Dict], List[Dict]]:
        vertices = []
        edges = []
        
        for node_name, attrs in graph.nodes(data=True):
            vertex = {
                "id": node_name,
                "label": "Entity",
                "properties": {
                    "name": node_name,
                    "entity_type": attrs.get("entity_type", "Entity"),
                    "description": attrs.get("description", ""),
                    "source_id": attrs.get("source_id", []),
                    "kb_id": attrs.get("kb_id", ""),
                    "project_id": attrs.get("project_id", ""),
                    "tenant_id": attrs.get("tenant_id", "default"),
                    "doc_id": attrs.get("doc_id", []),
                    "weight": attrs.get("weight", 1),
                    "probability": attrs.get("probability", 0.0),
                    "level": attrs.get("level", 0),
                    "keywords": attrs.get("keywords", []),
                    "pagerank": attrs.get("pagerank", 0.0),
                    "created_at": attrs.get("created_at", "")
                }
            }
            vertices.append(vertex)
        
        edge_id_counter = 0
        for src, tgt, attrs in graph.edges(data=True):
            edge_id = f"edge_{src}_{tgt}_{edge_id_counter}"
            edge_id_counter += 1
            
            edge = {
                "id": edge_id,
                "label": attrs.get("relation_type", "relation"),
                "outV": src,
                "inV": tgt,
                "outVLabel": "Entity",
                "inVLabel": "Entity",
                "properties": {
                    "description": attrs.get("description", ""),
                    "weight": attrs.get("weight", 1),
                    "keywords": attrs.get("keywords", []),
                    "source_id": attrs.get("source_id", []),
                    "kb_id": attrs.get("kb_id", ""),
                    "project_id": attrs.get("project_id", ""),
                    "tenant_id": attrs.get("tenant_id", "default"),
                    "doc_id": ",".join(attrs.get("doc_id", [])) if isinstance(attrs.get("doc_id"), list) else attrs.get("doc_id", ""),
                    "probability": attrs.get("probability", 0.0),
                    "gate_type": attrs.get("gate_type", ""),
                    "gate_group": attrs.get("gate_group", ""),
                    "threshold": attrs.get("threshold", 0),
                    "created_at": attrs.get("created_at", "")
                }
            }
            edges.append(edge)
        
        return vertices, edges