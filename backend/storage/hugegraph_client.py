import os
import json
import time
import subprocess
import requests
from collections import deque
from typing import Dict, List, Optional, Any, Tuple, Mapping
from backend.config.config import config
import logging

logger = logging.getLogger(__name__)


PROPERTY_KEY_DEFINITIONS = (
    ("name", "TEXT", "SINGLE"),
    ("entity_type", "TEXT", "SINGLE"),
    ("description", "TEXT", "SINGLE"),
    ("doc_id", "TEXT", "LIST"),
    ("kb_id", "TEXT", "SINGLE"),
    ("project_id", "TEXT", "SINGLE"),
    ("tenant_id", "TEXT", "SINGLE"),
    ("source", "TEXT", "SINGLE"),
    ("source_id", "TEXT", "LIST"),
    ("created_at", "TEXT", "SINGLE"),
    ("relation_type", "TEXT", "SINGLE"),
    ("keywords", "TEXT", "LIST"),
    ("weight", "INT", "SINGLE"),
    ("level", "INT", "SINGLE"),
    ("probability", "DOUBLE", "SINGLE"),
    ("role_in_fault_tree", "TEXT", "SINGLE"),
    ("gate_type", "TEXT", "SINGLE"),
    ("gate_group", "TEXT", "SINGLE"),
    ("threshold", "INT", "SINGLE"),
    ("action_type", "TEXT", "SINGLE"),
    ("pagerank", "DOUBLE", "SINGLE"),
    ("communities", "TEXT", "LIST"),
    ("similarity_score", "DOUBLE", "SINGLE"),
    ("confidence", "DOUBLE", "SINGLE"),
    ("summary", "TEXT", "SINGLE"),
    ("original_id", "TEXT", "SINGLE"),
    ("description_parts", "TEXT", "SINGLE"),
    ("keywords_parts", "TEXT", "SINGLE"),
    ("weight_parts", "TEXT", "SINGLE"),
    # structured-projector v5 physical identity fields.  Both are TEXT so the
    # existing Entity schema can append them without rewriting stored vertices.
    ("canonical_entity_id", "TEXT", "SINGLE"),
    ("identity_scope", "TEXT", "SINGLE"),
)

ENTITY_VERTEX_PROPERTIES = (
    "name", "entity_type", "description", "summary", "source_id", "pagerank",
    "communities", "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
    "weight", "level", "probability", "keywords", "description_parts",
    "keywords_parts", "canonical_entity_id", "identity_scope",
)
ENTITY_VERTEX_NULLABLE_KEYS = (
    "summary", "pagerank", "communities", "project_id", "tenant_id", "weight",
    "level", "probability", "keywords", "description_parts", "keywords_parts",
    "canonical_entity_id", "identity_scope",
)

DYNAMIC_EDGE_PROPERTIES = (
    "description", "weight", "keywords", "source_id", "project_id", "doc_id",
    "kb_id", "tenant_id", "created_at", "probability", "gate_type",
    "gate_group", "threshold", "description_parts", "keywords_parts",
    "weight_parts",
)
DYNAMIC_EDGE_NULLABLE_KEYS = (
    "keywords", "probability", "gate_type", "gate_group", "threshold",
    "description_parts", "keywords_parts", "weight_parts",
)

# The legacy shortest-path HTTP route is intentionally bounded independently
# of caller input.  HugeGraph is a projection and may grow over time; a public
# lookup must never deserialize an unbounded graph or enumerate unbounded
# paths in the application process.
_SHORTEST_PATH_MAX_VERTICES = 10_000
_SHORTEST_PATH_MAX_EDGES = 20_000
_SHORTEST_PATH_MAX_HOPS = 32
_PROJECTION_READ_PAGE_SIZE = 500

class HugeGraphUnavailable(RuntimeError):
    """HugeGraph 服务不可用 / 响应异常（区别于「查询结果为空」）。

    读写方法不再用空列表、None 或 False 把服务故障伪装成空结果：真空结果
    （确实没有数据、顶点确实不存在）继续返回空容器；服务/网络/解析故障统一
    抛此异常并沿调用链向上传递，由路由（503 / error 字段）或任务层（失败
    记录）显式降级，供 tracing 与监控区分“真空”与“故障”。
    """


class HugeGraphClient:
    def __init__(self):
        self.host = config.HUGEGRAPH_HOST
        self.port = config.HUGEGRAPH_PORT
        self.graph_name = config.HUGEGRAPH_GRAPH_NAME
        self.graphspace = config.HUGEGRAPH_GRAPHSPACE
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
        self._mock_client = MockHugeGraph()
        self._use_mock = not self._is_hugegraph_running()
        if not self._use_mock:
            self._ensure_schema_exists()
    
    def _is_hugegraph_running(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/graphs", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_schema_url(self, endpoint: str) -> str:
        return f"{self.base_url}/graphs/{self.graph_name}{endpoint}"
    
    def _get_graph_url(self, endpoint: str) -> str:
        return f"{self.base_url}/graphs/{self.graph_name}/graph{endpoint}"

    def _gremlin_aliases(self) -> Dict[str, str]:
        """Return the configured server-side graph/traversal bindings.

        HugeGraph 1.7 qualifies globals with the graphspace (the pinned image
        exposes ``DEFAULT-hugegraph``).  Leaving ``HUGEGRAPH_GRAPHSPACE`` empty
        deliberately selects the legacy unqualified binding names.
        """

        graphspace = str(getattr(self, "graphspace", "DEFAULT") or "").strip()
        graph_name = str(self.graph_name).strip()
        binding = f"{graphspace}-{graph_name}" if graphspace else graph_name
        return {"graph": binding, "g": f"__g_{binding}"}

    @staticmethod
    def _compact_error(exc: Exception) -> str:
        """Return a bounded, single-line capability error for health output."""

        return " ".join(str(exc).split())[:240]

    @staticmethod
    def _raise_unavailable(response, context: str) -> None:
        """2xx/204 之外一律视为图谱服务故障，抛 HugeGraphUnavailable。

        真空结果由调用方通过 200 + 空集合表达，因此这里不区分 4xx/5xx：
        响应本身失败就是故障，而不是“没有数据”。
        """
        if response.status_code in (200, 201, 204):
            return
        body = getattr(response, "text", "") or ""
        raise HugeGraphUnavailable(
            f"{context} failed: status={response.status_code}, body={body[:200]}"
        )
    
    def _ensure_schema_exists(self):
        logger.info("Initializing HugeGraph schema...")
        try:
            self._create_property_keys()
            logger.info("PropertyKeys created successfully")
        except Exception as e:
            logger.error(f"Warning: Failed to create some property keys: {e}")
        
        try:
            self._create_vertex_labels()
            logger.info("VertexLabels created successfully")
        except Exception as e:
            logger.error(f"Warning: Failed to create some vertex labels: {e}")
        
        try:
            self._create_edge_labels()
            logger.info("EdgeLabels created successfully")
        except Exception as e:
            logger.error(f"Warning: Failed to create some edge labels: {e}")
        
        try:
            self._create_indexes()
            logger.info("Indexes created successfully")
        except Exception as e:
            logger.error(f"Warning: Failed to create some indexes: {e}")
        
        logger.info("Schema initialization completed (existing items skipped)")
    
    def _create_property_keys(self):
        for name, data_type, cardinality in PROPERTY_KEY_DEFINITIONS:
            try:
                response = self.session.get(self._get_schema_url(f"/schema/propertykeys/{name}"))
                if response.status_code == 404:
                    data = {
                        "name": name,
                        "data_type": data_type,
                        "cardinality": cardinality
                    }
                    self.session.post(self._get_schema_url("/schema/propertykeys"), json=data)
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to create property key {name}: {e}")
                pass
    
    def _create_vertex_labels(self):
        vertex_labels_def = {
            "Fault": {
                "properties": ["name", "entity_type", "description", "level", "probability", 
                              "role_in_fault_tree", "source_id", "pagerank", "communities",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["level", "probability", "role_in_fault_tree", "pagerank", "communities", "project_id", "tenant_id"]
            },
            "Cause": {
                "properties": ["name", "entity_type", "description", "level", "probability", 
                              "source_id", "pagerank", "communities",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["level", "probability", "pagerank", "communities", "project_id", "tenant_id"]
            },
            "Action": {
                "properties": ["name", "entity_type", "description", "action_type", 
                              "source_id", "pagerank", "communities",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["action_type", "pagerank", "communities", "project_id", "tenant_id"]
            },
            "Check": {
                "properties": ["name", "entity_type", "description", 
                              "source_id", "pagerank", "communities",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "communities", "project_id", "tenant_id"]
            },
            "Entity": {
                "properties": list(ENTITY_VERTEX_PROPERTIES),
                "nullable_keys": list(ENTITY_VERTEX_NULLABLE_KEYS),
            },
            "Organization": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            },
            "Person": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            },
            "Geo": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            },
            "Event": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            },
            "Category": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            },
            "Component": {
                "properties": ["name", "entity_type", "description", "source_id", "pagerank",
                              "project_id", "doc_id", "kb_id", "tenant_id", "created_at"],
                "nullable_keys": ["pagerank", "project_id", "tenant_id"]
            }
        }
        
        for label, definition in vertex_labels_def.items():
            self._create_vertex_label_if_not_exists(label, definition["properties"], definition["nullable_keys"])
    
    def _create_vertex_label_if_not_exists(self, label: str, properties: List[str], nullable_keys: List[str] = None):
        try:
            response = self.session.get(self._get_schema_url(f"/schema/vertexlabels/{label}"))
            if response.status_code == 404:
                data = {
                    "name": label,
                    "id_strategy": "CUSTOMIZE_STRING",
                    "properties": properties,
                    "nullable_keys": nullable_keys or []
                }
                self.session.post(self._get_schema_url("/schema/vertexlabels"), json=data)
                time.sleep(0.1)
            else:
                existing_label = response.json()
                existing_props = set(existing_label.get("properties", []))
                new_props = set(properties)
                missing_props = new_props - existing_props
                
                if missing_props:
                    logger.info(f"Updating vertex label '{label}' - adding missing properties: {missing_props}")
                    
                    append_data = {
                        "name": label,
                        "properties": list(missing_props),
                        "nullable_keys": [k for k in (nullable_keys or []) if k in missing_props]
                    }
                    append_response = self.session.put(
                        self._get_schema_url(f"/schema/vertexlabels/{label}?action=append"), 
                        json=append_data
                    )
                    if append_response.status_code == 200:
                        logger.info(f"Successfully appended properties to vertex label '{label}'")
                    else:
                        logger.error(f"Failed to append properties to vertex label '{label}': {append_response.text}")
        except Exception as e:
            logger.error(f"Failed to create vertex label {label}: {e}")
            pass
    
    def _create_edge_label_if_not_exists(self, label: str, source_label: List[str], 
                                         target_label: List[str], properties: List[str], 
                                         nullable_keys: List[str] = None):
        try:
            response = self.session.get(self._get_schema_url(f"/schema/edgelabels/{label}"))
            if response.status_code == 404:
                data = {
                    "name": label,
                    "frequency": "SINGLE",
                    "source_label": source_label,
                    "target_label": target_label,
                    "properties": properties,
                    "nullable_keys": nullable_keys or []
                }
                response = self.session.post(self._get_schema_url("/schema/edgelabels"), json=data)
                if response.status_code != 201:
                    logger.error(f"Failed to create edge label {label}: {response.text}")
                time.sleep(0.1)
            else:
                existing_label = response.json()
                has_source_label = existing_label.get("source_label") is not None
                has_target_label = existing_label.get("target_label") is not None

                if not has_source_label or not has_target_label:
                    logger.warning(f"Warning: Edge label '{label}' exists but missing source/target label configuration.")
                    logger.info(f"         This may cause edge insertion failures. Consider deleting and recreating.")
                    logger.info(f"         Current config: source_label={existing_label.get('source_label')}, target_label={existing_label.get('target_label')}")

                # 已存在的边标签追加缺失的 properties（镜像 vertex label 的 append 逻辑）
                existing_props = set(existing_label.get("properties", []))
                new_props = set(properties)
                missing_props = new_props - existing_props
                if missing_props:
                    logger.info(f"Updating edge label '{label}' - adding missing properties: {missing_props}")
                    append_data = {
                        "name": label,
                        "properties": list(missing_props),
                        "nullable_keys": [k for k in (nullable_keys or []) if k in missing_props]
                    }
                    append_response = self.session.put(
                        self._get_schema_url(f"/schema/edgelabels/{label}?action=append"),
                        json=append_data
                    )
                    if append_response.status_code == 200:
                        logger.info(f"Successfully appended properties to edge label '{label}'")
                    else:
                        logger.error(f"Failed to append properties to edge label '{label}': {append_response.text}")
        except Exception as e:
            logger.error(f"Failed to create edge label {label}: {e}")
            pass
    
    def _update_edge_label_if_needed(self, label: str, source_label: List[str], 
                                     target_label: List[str], properties: List[str], 
                                     nullable_keys: List[str] = None):
        try:
            response = self.session.get(self._get_schema_url(f"/schema/edgelabels/{label}"))
            if response.status_code == 200:
                existing_label = response.json()
                has_source_label = existing_label.get("source_label") is not None
                has_target_label = existing_label.get("target_label") is not None
                
                if not has_source_label or not has_target_label:
                    # Deleting an edge label also deletes its edges in
                    # HugeGraph.  Startup schema migration must never perform
                    # that destructive repair.  Property additions are handled
                    # independently by the append-only path below; endpoint
                    # metadata requires an explicit offline migration.
                    logger.info(
                        f"Warning: Edge label '{label}' lacks source/target "
                        "metadata; refusing destructive automatic recreation"
                    )
                    return False
                return True
        except Exception as e:
            logger.error(f"Failed to update edge label {label}: {e}")
            return False
        return True
    
    def _create_edge_labels(self):
        edge_labels_def = {
            "causes": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "gate_type", "gate_group",
                               "probability", "threshold", "keywords", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "keywords_parts", "weight_parts"],
                "nullable_keys": ["gate_type", "gate_group", "probability", "threshold", "keywords",
                                  "description_parts", "keywords_parts", "weight_parts"]
            },
            "has_solution": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "keywords", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "keywords_parts", "weight_parts"],
                "nullable_keys": ["keywords", "description_parts", "keywords_parts", "weight_parts"]
            },
            "skips_check": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "keywords", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "keywords_parts", "weight_parts"],
                "nullable_keys": ["keywords", "description_parts", "keywords_parts", "weight_parts"]
            },
            "relation": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "keywords", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at", "relation_type",
                               "description_parts", "keywords_parts", "weight_parts"],
                "nullable_keys": ["keywords", "description_parts", "keywords_parts", "weight_parts"]
            },
            "same_as": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["similarity_score", "confidence", "description"],
                "nullable_keys": ["description"]
            },
            "related_to": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "keywords", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "keywords_parts", "weight_parts"],
                "nullable_keys": ["keywords", "description_parts", "keywords_parts", "weight_parts"]
            },
            "belongs_to": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "weight_parts"],
                "nullable_keys": ["description_parts", "weight_parts"]
            },
            "has_component": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "weight_parts"],
                "nullable_keys": ["description_parts", "weight_parts"]
            },
            "part_of": {
                "source_label": "Entity",
                "target_label": "Entity",
                "properties": ["description", "weight", "source_id",
                               "project_id", "doc_id", "kb_id", "tenant_id", "created_at",
                               "description_parts", "weight_parts"],
                "nullable_keys": ["description_parts", "weight_parts"]
            }
        }
        
        for label, definition in edge_labels_def.items():
            self._update_edge_label_if_needed(label, definition["source_label"], 
                                              definition["target_label"], 
                                              definition["properties"], 
                                              definition["nullable_keys"])
            self._create_edge_label_if_not_exists(label, definition["source_label"], 
                                                  definition["target_label"], 
                                                  definition["properties"], 
                                                  definition["nullable_keys"])
    
    def create_edge_label_dynamic(self, label: str):
        default_props = list(DYNAMIC_EDGE_PROPERTIES)
        default_nullable = list(DYNAMIC_EDGE_NULLABLE_KEYS)
        
        try:
            response = self.session.get(self._get_schema_url(f"/schema/edgelabels/{label}"))
            if response.status_code == 404:
                data = {
                    "name": label,
                    "frequency": "SINGLE",
                    "source_label": "Entity",
                    "target_label": "Entity",
                    "properties": default_props,
                    "nullable_keys": default_nullable
                }
                response = self.session.post(self._get_schema_url("/schema/edgelabels"), json=data)
                if response.status_code == 201:
                    logger.info(f"Created dynamic edge label '{label}'")
                    return True
                else:
                    logger.error(f"Failed to create dynamic edge label '{label}': {response.text}")
                    return False
            else:
                existing_label = response.json()
                existing_props = set(existing_label.get("properties", []))
                missing_props = set(default_props) - existing_props
                
                if missing_props:
                    logger.info(f"Updating edge label '{label}' - adding missing properties: {missing_props}")
                    
                    append_data = {
                        "name": label,
                        "properties": list(missing_props),
                        "nullable_keys": [k for k in default_nullable if k in missing_props]
                    }
                    append_response = self.session.put(
                        self._get_schema_url(f"/schema/edgelabels/{label}?action=append"),
                        json=append_data
                    )
                    if append_response.status_code == 200:
                        logger.info(f"Successfully appended properties to edge label '{label}'")
                        return True
                    else:
                        logger.error(f"Failed to append properties to edge label '{label}': {append_response.text}")
                        return False
                else:
                    logger.info(f"Edge label '{label}' already exists with all required properties")
                    return True
        except Exception as e:
            logger.error(f"Failed to create dynamic edge label '{label}': {e}")
            return False
    
    def validate_vertices(self, vertices: List[Dict]) -> Tuple[List[Dict], List[str]]:
        valid_vertices = []
        errors = []
        
        for i, vertex in enumerate(vertices):
            try:
                if not vertex.get("id"):
                    errors.append(f"Vertex {i}: missing 'id' field")
                    continue
                if not vertex.get("label"):
                    errors.append(f"Vertex {i}: missing 'label' field")
                    continue
                
                props = vertex.get("properties", {})
                for key, value in props.items():
                    if isinstance(value, list):
                        props[key] = value
                
                valid_vertices.append(vertex)
            except Exception as e:
                errors.append(f"Vertex {i}: validation error - {e}")
        
        return valid_vertices, errors
    
    def validate_edges(self, edges: List[Dict]) -> Tuple[List[Dict], List[str]]:
        valid_edges = []
        errors = []
        
        for i, edge in enumerate(edges):
            try:
                if not edge.get("id"):
                    errors.append(f"Edge {i}: missing 'id' field")
                    continue
                if not edge.get("label"):
                    errors.append(f"Edge {i}: missing 'label' field")
                    continue
                if not edge.get("outV"):
                    errors.append(f"Edge {i}: missing 'outV' field")
                    continue
                if not edge.get("inV"):
                    errors.append(f"Edge {i}: missing 'inV' field")
                    continue
                
                props = edge.get("properties", {})
                for key, value in props.items():
                    if isinstance(value, list):
                        props[key] = value
                
                if not edge.get("outVLabel"):
                    edge["outVLabel"] = "Entity"
                if not edge.get("inVLabel"):
                    edge["inVLabel"] = "Entity"
                
                valid_edges.append(edge)
            except Exception as e:
                errors.append(f"Edge {i}: validation error - {e}")
        
        return valid_edges, errors
    
    def ensure_edge_labels_for_edges(self, edges: List[Dict]):
        edge_labels = set(edge.get("label") for edge in edges if edge.get("label"))
        
        for label in edge_labels:
            self.create_edge_label_dynamic(label)
    
    def _create_indexes(self):
        indexes = [
            {"name": "entity_kb_index", "base_type": "VERTEX_LABEL", "base_value": "Entity", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True},
            {"name": "entity_name_index", "base_type": "VERTEX_LABEL", "base_value": "Entity", 
             "index_type": "SECONDARY", "fields": ["name"], "rebuild": True},
            {"name": "fault_kb_index", "base_type": "VERTEX_LABEL", "base_value": "Fault", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True},
            {"name": "cause_kb_index", "base_type": "VERTEX_LABEL", "base_value": "Cause", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True},
            {"name": "action_kb_index", "base_type": "VERTEX_LABEL", "base_value": "Action", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True},
            {"name": "causes_kb_index", "base_type": "EDGE_LABEL", "base_value": "causes", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True},
            {"name": "relation_kb_index", "base_type": "EDGE_LABEL", "base_value": "relation", 
             "index_type": "SECONDARY", "fields": ["kb_id"], "rebuild": True}
        ]
        
        for index_def in indexes:
            try:
                response = self.session.get(self._get_schema_url(f"/schema/indexlabels/{index_def['name']}"))
                if response.status_code == 404:
                    self.session.post(self._get_schema_url("/schema/indexlabels"), json=index_def)
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to create index {index_def['name']}: {e}")
                pass
    
    def batch_insert(self, vertices: List[Dict], edges: List[Dict], max_retries: int = 3):
        if self._use_mock:
            return self._mock_client.batch_insert(vertices, edges)
        
        try:
            logger.info("=== Data Validation ===")
            vertices, vertex_errors = self.validate_vertices(vertices)
            edges, edge_errors = self.validate_edges(edges)
            
            if vertex_errors:
                logger.error(f"Vertex validation errors: {len(vertex_errors)}")
                for err in vertex_errors[:5]:
                    logger.warning(f"  - {err}")
            
            if edge_errors:
                logger.error(f"Edge validation errors: {len(edge_errors)}")
                for err in edge_errors[:5]:
                    logger.warning(f"  - {err}")
            
            logger.info("=== Ensuring Edge Labels ===")
            self.ensure_edge_labels_for_edges(edges)
            
            batch_size = 100
            inserted_vertices = 0
            vertex_id_map = {}
            
            for i in range(0, len(vertices), batch_size):
                batch = vertices[i:i + batch_size]
                success = False
                retries = 0
                
                while not success and retries < max_retries:
                    try:
                        response = self.session.post(self._get_graph_url("/vertices/batch"), json=batch)
                        if response.status_code in [200, 201]:
                            result = response.json()
                            if isinstance(result, list):
                                for j, vid in enumerate(result):
                                    original_id = batch[j].get("id")
                                    if original_id:
                                        vertex_id_map[original_id] = vid
                                inserted_vertices += len(result)
                            else:
                                inserted_vertices += result.get("success", 0)
                            logger.info(f"Inserted {len(result) if isinstance(result, list) else result.get('success', 0)}/{len(batch)} vertices in batch")
                            success = True
                        else:
                            error_msg = response.text[:500] if len(response.text) > 500 else response.text
                            logger.error(f"Insert vertices batch failed (attempt {retries+1}): {error_msg}")
                            retries += 1
                            if retries < max_retries:
                                time.sleep(2 ** retries)
                    except Exception as ve:
                        logger.info(f"Vertex batch insert exception (attempt {retries+1}): {ve}")
                        retries += 1
                        if retries < max_retries:
                            time.sleep(2 ** retries)
            
            if vertex_id_map:
                for edge in edges:
                    orig_outv = edge.get("outV")
                    orig_inv = edge.get("inV")
                    if orig_outv in vertex_id_map:
                        edge["outV"] = vertex_id_map[orig_outv]
                    if orig_inv in vertex_id_map:
                        edge["inV"] = vertex_id_map[orig_inv]
            
            inserted_edges = 0
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                success = False
                retries = 0
                
                while not success and retries < max_retries:
                    try:
                        response = self.session.post(self._get_graph_url("/edges/batch"), json=batch)
                        if response.status_code in [200, 201]:
                            result = response.json()
                            if isinstance(result, list):
                                inserted_edges += len(result)
                            else:
                                inserted_edges += result.get("success", 0)
                            logger.info(f"Inserted {len(result) if isinstance(result, list) else result.get('success', 0)}/{len(batch)} edges in batch")
                            success = True
                        else:
                            error_msg = response.text[:500] if len(response.text) > 500 else response.text
                            logger.error(f"Insert edges batch failed (attempt {retries+1}): HTTP {response.status_code} - {error_msg}")
                            retries += 1
                            if retries < max_retries:
                                time.sleep(2 ** retries)
                    except Exception as ee:
                        logger.info(f"Edge batch insert exception (attempt {retries+1}): {ee}")
                        retries += 1
                        if retries < max_retries:
                            time.sleep(2 ** retries)
            
            logger.info(f"Batch insert complete: {inserted_vertices}/{len(vertices)} vertices, {inserted_edges}/{len(edges)} edges")
            
            if inserted_vertices != len(vertices) or inserted_edges != len(edges):
                logger.warning(f"Warning: Not all data was inserted successfully")
            
            return inserted_vertices == len(vertices) and inserted_edges == len(edges)
        except Exception as e:
            logger.info(f"Batch insert exception: {e}")
            return False
    
    def get_vertices_by_kb(self, kb_id: str, label: str = None, limit: int = 10000) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_vertices_by_kb(kb_id, label)
        
        try:
            all_vertices = []
            offset = 0
            while True:
                params = {"limit": limit, "offset": offset}
                if label:
                    params["label"] = label
                params["properties"] = json.dumps({"kb_id": kb_id})
                response = self.session.get(
                    self._get_graph_url("/vertices"),
                    params=params
                )
                if response.status_code != 200:
                    self._raise_unavailable(response, "Get vertices by kb")
                vertices = response.json().get("vertices", [])
                all_vertices.extend(vertices)
                if len(vertices) < limit:
                    break
                offset += limit
            return all_vertices
        except Exception as e:
            if isinstance(e, HugeGraphUnavailable):
                raise
            logger.error(f"Get vertices by kb failed: {e}")
            raise HugeGraphUnavailable(f"Get vertices by kb failed: {e}") from e

    def get_vertex_by_id(self, vertex_id: str) -> Optional[Dict]:
        """Read one CUSTOMIZE_STRING vertex without relying on a secondary index.

        HugeGraph's offset pagination can briefly miss an item while a LIST or
        secondary-indexed property is being updated.  Projection verification
        uses this direct lookup only for expected IDs absent from the bulk scan.
        """

        if self._use_mock:
            vertex = self._mock_client.vertices.get(vertex_id)
            return dict(vertex) if vertex else None
        try:
            import urllib.parse

            encoded_vid = urllib.parse.quote(self._quote_vid(vertex_id), safe="")
            response = self.session.get(
                self._get_graph_url(f"/vertices/{encoded_vid}")
            )
            if response.status_code == 404:
                # 顶点确实不存在（真空），不是服务故障
                return None
            if response.status_code != 200:
                self._raise_unavailable(response, "Get vertex by id")
            value = response.json()
            return value if isinstance(value, dict) else None
        except Exception as exc:
            if isinstance(exc, HugeGraphUnavailable):
                raise
            logger.error(f"Get vertex by id failed: {exc}")
            raise HugeGraphUnavailable(f"Get vertex by id failed: {exc}") from exc
    
    def get_edges_by_kb(self, kb_id: str, label: str = None, limit: int = 10000) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_edges_by_kb(kb_id, label)
        
        try:
            # HugeGraph 的 ``properties={kb_id: ...}`` 查询依赖每个 edge label
            # 都存在对应索引。动态领域关系（例如 has_fault_mode、series_to）
            # 没有逐标签索引时，服务会返回空列表，即使边已经成功写入。这里按
            # label（若指定）分页读取，再在客户端严格过滤 kb_id，确保投影读回、
            # 回滚和删除不会把“缺索引”误判成“没有边”。
            all_edges = []
            offset = 0
            while True:
                params = {"limit": limit, "offset": offset}
                if label:
                    params["label"] = label
                response = self.session.get(
                    self._get_graph_url("/edges"),
                    params=params
                )
                if response.status_code != 200:
                    self._raise_unavailable(response, "Get edges by kb")
                edges = response.json().get("edges", [])
                all_edges.extend(edges)
                if len(edges) < limit:
                    break
                offset += limit
            return [
                edge
                for edge in all_edges
                if (edge.get("properties") or {}).get("kb_id") == kb_id
            ]
        except Exception as e:
            if isinstance(e, HugeGraphUnavailable):
                raise
            logger.error(f"Get edges by kb failed: {e}")
            raise HugeGraphUnavailable(f"Get edges by kb failed: {e}") from e
    
    def get_graph_data(self) -> Dict:
        if self._use_mock:
            return self._mock_client.get_graph_data()
        
        try:
            all_vertices = []
            offset = 0
            limit = 1000
            while True:
                response = self.session.get(
                    self._get_graph_url("/vertices"),
                    params={"limit": limit, "offset": offset}
                )
                if response.status_code != 200:
                    self._raise_unavailable(response, "Get graph data vertices")
                vertices = response.json().get("vertices", [])
                all_vertices.extend(vertices)
                if len(vertices) < limit:
                    break
                offset += limit
            
            all_edges = []
            offset = 0
            while True:
                response = self.session.get(
                    self._get_graph_url("/edges"),
                    params={"limit": limit, "offset": offset}
                )
                if response.status_code != 200:
                    self._raise_unavailable(response, "Get graph data edges")
                edges = response.json().get("edges", [])
                all_edges.extend(edges)
                if len(edges) < limit:
                    break
                offset += limit
            
            return {"vertices": all_vertices, "edges": all_edges}
        except Exception as e:
            if isinstance(e, HugeGraphUnavailable):
                raise
            logger.error(f"Get graph data failed: {e}")
            raise HugeGraphUnavailable(f"Get graph data failed: {e}") from e
    
    def delete_by_kb_id(self, kb_id: str):
        if self._use_mock:
            return self._mock_client.delete_by_kb_id(kb_id)
        
        try:
            import urllib.parse
            
            edges = self.get_edges_by_kb(kb_id)
            for edge in edges:
                eid = edge.get("id")
                if eid:
                    encoded_eid = urllib.parse.quote(eid, safe='')
                    self.session.delete(
                        self._get_graph_url(f"/edges/{encoded_eid}")
                    )
            
            vertices = self.get_vertices_by_kb(kb_id)
            for vertex in vertices:
                vid = vertex.get("id")
                if vid:
                    encoded_vid = urllib.parse.quote(self._quote_vid(vid), safe='')
                    self.session.delete(
                        self._get_graph_url(f"/vertices/{encoded_vid}")
                    )
            
            logger.info(f"Deleted all data for kb_id: {kb_id}")
            return True
        except Exception as e:
            logger.error(f"Delete by kb_id failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def _quote_vid(vid: str) -> str:
        """HugeGraph CUSTOMIZE_STRING 顶点 ID 必须用双引号包裹以标识为字符串类型。

        裸字符串 ID 会被 HugeGraph 拒绝并返回 400:
        'The vertex id must be formatted as Number/String/UUID'。
        用于 query parameter 时直接返回带引号字符串；用于 URL path 时需再
        urllib.parse.quote 编码（双引号会被编码为 %22）。
        """
        if not isinstance(vid, str):
            vid = str(vid)
        # 避免重复包裹
        if vid.startswith('"') and vid.endswith('"'):
            return vid
        return f'"{vid}"'

    def _parse_response(self, response):
        if response.status_code == 204:
            return {}
        if response.status_code not in (200, 201):
            self._raise_unavailable(response, "HugeGraph response")
        content = response.content
        try:
            import gzip
            content = gzip.decompress(content)
        except Exception:
            # 响应未压缩或已由 requests 自动解压，按原文解析
            pass
        try:
            return json.loads(content)
        except Exception as e:
            raise HugeGraphUnavailable(f"HugeGraph response parse failed: {e}") from e

    @staticmethod
    def _trim_parts_by_doc(props: dict, doc_id: str) -> dict:
        """裁剪 *_parts 中属于 doc_id 的片段，并重算派生字段 description/keywords/weight。

        返回需要更新的属性 dict（仅含被裁剪/重算的字段）。props 为 HugeGraph 读回的
        原始属性，parts 字段可能是 JSON 字符串或 list。source_id 中属于该文档的
        chunk_key（形如 {doc_id}_{idx}）也一并裁剪。
        """
        def _parse(raw):
            if not raw:
                return []
            if isinstance(raw, list):
                return raw
            try:
                p = json.loads(raw)
                return p if isinstance(p, list) else []
            except Exception:
                return []

        updated = {}

        if "description_parts" in props or props.get("description_parts"):
            parts = _parse(props.get("description_parts"))
            new_parts = [p for p in parts if p.get("doc_id") != doc_id]
            updated["description_parts"] = json.dumps(new_parts, ensure_ascii=False)
            # 重算 description 缓存（删除路径不触发 LLM 摘要，保留全部剩余片段）
            updated["description"] = GRAPH_FIELD_SEP.join(p.get("text", "") for p in new_parts)

        if "keywords_parts" in props or props.get("keywords_parts"):
            kparts = _parse(props.get("keywords_parts"))
            new_kparts = [p for p in kparts if p.get("doc_id") != doc_id]
            updated["keywords_parts"] = json.dumps(new_kparts, ensure_ascii=False)
            updated["keywords"] = sorted({k for p in new_kparts for k in p.get("keywords", [])})

        if "weight_parts" in props or props.get("weight_parts"):
            wparts = _parse(props.get("weight_parts"))
            new_wparts = [p for p in wparts if p.get("doc_id") != doc_id]
            updated["weight_parts"] = json.dumps(new_wparts, ensure_ascii=False)
            # 只有存在 weight_parts 时才重算 weight，避免旧顶点被误归零
            if wparts:
                updated["weight"] = sum(p.get("weight", 0) for p in new_wparts)

        # source_id 裁剪掉该文档的 chunk_key
        # HugeGraph LIST 字段在 PUT 时需用逗号分隔字符串（与本项目 doc_id 序列化约定一致）
        src = props.get("source_id", [])
        if isinstance(src, str):
            src = [s for s in src.split(",") if s]
        if src:
            new_src = [
                s
                for s in src
                if not (
                    str(s) == doc_id
                    or str(s).startswith(f"{doc_id}_")
                    or str(s) == f"structured-doc:{doc_id}"
                    or str(s).startswith(f"projection:{doc_id}:")
                )
            ]
            updated["source_id"] = new_src

        return updated

    @staticmethod
    def _matches_doc(props: dict, doc_id: str) -> bool:
        """判断顶点/边是否属于指定 doc_id。

        匹配策略（按优先级）：
        1) doc_id 字段匹配（严格相等，或逗号分隔列表包含该 doc_id）
        2) source_id 兜底：任一 source_id 以 "{doc_id}_" 开头（chunk_key 命名约定）
        """
        # 1) doc_id 字段匹配
        raw = props.get("doc_id", "")
        doc_id_list = []
        if isinstance(raw, list):
            doc_id_list = [str(d) for d in raw if d]
        elif isinstance(raw, str):
            if raw:
                doc_id_list = [d for d in raw.split(",") if d]
        if doc_id in doc_id_list:
            return True

        # 2) source_id 兜底
        src = props.get("source_id", [])
        if isinstance(src, str):
            src_list = [s for s in src.split(",") if s]
        elif isinstance(src, list):
            src_list = [str(s) for s in src if s]
        else:
            src_list = []
        for s in src_list:
            if s == doc_id or s.startswith(f"{doc_id}_"):
                return True
        return False

    @staticmethod
    def _remove_doc_id_from_props(props: dict, doc_id: str) -> list:
        """从 props 中解析 doc_id 列表并移除 doc_id，返回剩余列表。"""
        raw = props.get("doc_id", "")
        doc_id_list = []
        if isinstance(raw, list):
            doc_id_list = [str(d) for d in raw if d]
        elif isinstance(raw, str):
            if raw:
                doc_id_list = [d for d in raw.split(",") if d]
        return [d for d in doc_id_list if d != doc_id]

    def delete_edges_by_doc_id(self, doc_id: str):
        if self._use_mock:
            return self._mock_client.delete_edges_by_doc_id(doc_id)

        import urllib.parse
        deleted_count = 0
        updated_count = 0
        matched_count = 0
        errors = 0

        all_edges = []
        offset = 0
        limit = 1000

        try:
            while True:
                response = self.session.get(
                    self._get_graph_url("/edges"),
                    params={"limit": limit, "offset": offset}
                )
                result = self._parse_response(response)
                edges = result.get("edges", [])

                if not edges:
                    break

                all_edges.extend(edges)
                if len(edges) < limit:
                    break
                offset += limit
        except Exception as e:
            logger.error(f"[KG-DELETE] Load edges failed: {e}")
            return False

        for edge in all_edges:
            try:
                eid = edge.get("id")
                props = edge.get("properties", {})

                if not self._matches_doc(props, doc_id):
                    continue

                doc_id_list = self._remove_doc_id_from_props(props, doc_id)
                matched_count += 1
                # 判断原始 doc_id 列表长度决定分支；若原列表为空（通过 source_id 匹配），
                # 保守地按 DELETE 分支处理
                raw_doc_id = props.get("doc_id", "")
                was_in_doc_id_field = False
                if isinstance(raw_doc_id, list):
                    was_in_doc_id_field = True
                elif isinstance(raw_doc_id, str) and raw_doc_id:
                    was_in_doc_id_field = True
                is_shared = was_in_doc_id_field and len(doc_id_list) > 0

                branch = "DELETE" if not is_shared else (
                    "DELETE_EMPTY" if not doc_id_list else "UPDATE"
                )
                logger.info(f"[KG-DELETE] Edge {eid}: matched via {'doc_id' if was_in_doc_id_field else 'source_id'}, "
                      f"remaining_docs={doc_id_list}, branch={branch}")

                if branch in ("DELETE", "DELETE_EMPTY"):
                    encoded_eid = urllib.parse.quote(eid, safe='')
                    delete_response = self.session.delete(
                        self._get_graph_url(f"/edges/{encoded_eid}")
                    )
                    logger.info(f"[KG-DELETE] Edge {eid} DELETE response: status={delete_response.status_code}, body={delete_response.text[:200]}")
                    if delete_response.status_code == 204:
                        deleted_count += 1
                        logger.info(f"[KG-DELETE] Deleted edge: {eid}")
                    else:
                        errors += 1
                else:
                    new_doc_id_list = doc_id_list

                    updated_props = {"doc_id": new_doc_id_list}
                    # 同步裁剪 description_parts / keywords_parts / weight_parts
                    trimmed = self._trim_parts_by_doc(props, doc_id)
                    if trimmed:
                        updated_props.update(trimmed)
                        logger.info(f"[KG-DELETE] Edge {eid} PUT payload: keys={list(updated_props.keys())}, doc_id={updated_props.get('doc_id')}, source_id={updated_props.get('source_id','<not_set>')}")
                    else:
                        logger.info(f"[KG-DELETE] Edge {eid} PUT (no parts): doc_id={updated_props.get('doc_id')}")
                    replacement = dict(props)
                    replacement.update(updated_props)
                    update_ok = self.replace_edge_properties(eid, replacement)
                    logger.info(f"[KG-DELETE] Edge {eid} exact batch UPDATE: success={update_ok}")
                    if update_ok:
                        updated_count += 1
                    else:
                        errors += 1
            except Exception as e:
                errors += 1
                import traceback
                traceback.print_exc()
                logger.error(f"[KG-DELETE] Edge iteration error for {eid if 'eid' in locals() else '?'}: {e}")

        logger.error(f"[KG-DELETE] Delete edges result: matched={matched_count}, deleted={deleted_count}, updated={updated_count}, errors={errors} for doc_id: {doc_id}")
        return errors == 0
    
    def _get_all_edges_for_vertex(self, vertex_id: str, vertex_label: str) -> List[Dict]:
        try:
            all_edges = []
            offset = 0
            limit = 100
            while True:
                response = self.session.get(
                    self._get_graph_url("/edges"),
                    params={
                        "vertex_id": self._quote_vid(vertex_id),
                        "vertex_label": vertex_label,
                        "direction": "BOTH",
                        "limit": limit,
                        "offset": offset
                    }
                )
                result = self._parse_response(response)
                edges = result.get("edges", [])
                if not edges:
                    break
                all_edges.extend(edges)
                if len(edges) < limit:
                    break
                offset += limit
            return all_edges
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Get edges for vertex {vertex_id} failed: {e}")
            return []

    def _delete_all_edges_for_vertex(self, vertex_id: str, vertex_label: str) -> int:
        import urllib.parse
        deleted = 0
        try:
            edges = self._get_all_edges_for_vertex(vertex_id, vertex_label)
            for edge in edges:
                eid = edge.get("id")
                if eid:
                    encoded_eid = urllib.parse.quote(eid, safe='')
                    resp = self.session.delete(
                        self._get_graph_url(f"/edges/{encoded_eid}")
                    )
                    if resp.status_code == 204:
                        deleted += 1
                    else:
                        logger.error(f"Failed to delete edge {eid} for vertex {vertex_id}, status={resp.status_code}")
        except Exception as e:
            logger.error(f"Delete edges for vertex {vertex_id} failed: {e}")
        return deleted

    def delete_vertices_by_doc_id(self, doc_id: str):
        if self._use_mock:
            return self._mock_client.delete_vertices_by_doc_id(doc_id)

        import urllib.parse
        deleted_count = 0
        updated_count = 0
        matched_count = 0
        errors = 0
        removed_names = []  # 被物理删除的实体名，供社区报告分级使用
        error_details = []  # 前 10 条错误详情，供删除日志展示

        logger.info(f"[KG-DELETE] === Delete vertices by doc_id={doc_id} ===")

        all_vertices = []
        offset = 0
        limit = 1000

        try:
            while True:
                response = self.session.get(
                    self._get_graph_url("/vertices"),
                    params={"limit": limit, "offset": offset}
                )
                result = self._parse_response(response)
                vertices = result.get("vertices", [])

                if not vertices:
                    break

                all_vertices.extend(vertices)
                if len(vertices) < limit:
                    break
                offset += limit
        except Exception as e:
            logger.error(f"[KG-DELETE] Load vertices failed: {e}")
            return {"success": False, "deleted": 0, "updated": 0, "matched": 0, "removed_names": []}

        logger.info(f"[KG-DELETE] Loaded {len(all_vertices)} vertices total, scanning for doc_id={doc_id}")

        for vertex in all_vertices:
            try:
                vid = vertex.get("id")
                props = vertex.get("properties", {}) or {}
                vertex_label = vertex.get("label", "Entity")
                vname = props.get("name", "")

                raw_doc_id = props.get("doc_id", "")
                # 判断是否匹配
                if not self._matches_doc(props, doc_id):
                    continue

                # 计算剩余 doc_id（用于决定 DELETE 还是 UPDATE 分支）
                remaining_docs = self._remove_doc_id_from_props(props, doc_id)
                # 判断原始 doc_id 列表长度决定分支；若原列表为空（通过 source_id 匹配），
                # 保守地按 DELETE 分支处理
                was_in_doc_id_field = False
                if isinstance(raw_doc_id, list) and raw_doc_id:
                    was_in_doc_id_field = True
                elif isinstance(raw_doc_id, str) and raw_doc_id:
                    was_in_doc_id_field = True
                is_shared = was_in_doc_id_field and len(remaining_docs) > 0

                matched_count += 1
                branch = "DELETE" if not is_shared else (
                    "DELETE_EMPTY" if not remaining_docs else "UPDATE"
                )
                matched_via = "doc_id" if was_in_doc_id_field else "source_id"
                logger.info(f"[KG-DELETE] Vertex {vid} (name={vname}) matched via {matched_via}, "
                      f"branch={branch}, remaining_docs={remaining_docs}")

                if branch in ("DELETE", "DELETE_EMPTY"):
                    edges_deleted = self._delete_all_edges_for_vertex(vid, vertex_label)
                    logger.info(f"[KG-DELETE] Vertex {vid}: deleted {edges_deleted} edges before vertex deletion")

                    encoded_vid = urllib.parse.quote(self._quote_vid(vid), safe='')
                    delete_response = self.session.delete(
                        self._get_graph_url(f"/vertices/{encoded_vid}")
                    )
                    logger.info(f"[KG-DELETE] Vertex {vid} DELETE response: status={delete_response.status_code}, "
                          f"body={delete_response.text[:300]}")
                    if delete_response.status_code == 204:
                        deleted_count += 1
                        removed_names.append(vname)
                        logger.info(f"[KG-DELETE] DELETED vertex: {vid} (name={vname})")
                    else:
                        errors += 1
                        err_msg = f"DELETE vertex {vid} (name={vname}) failed: HTTP {delete_response.status_code}, body={delete_response.text[:200]}"
                        logger.warning(f"[KG-DELETE] {err_msg}")
                        if len(error_details) < 10:
                            error_details.append(err_msg)
                else:
                    replacement = dict(props)
                    replacement["doc_id"] = remaining_docs
                    trimmed = self._trim_parts_by_doc(props, doc_id)
                    if trimmed:
                        if "description" in trimmed and not trimmed["description"]:
                            logger.info(f"[KG-DELETE] Vertex {vid}: description_parts all removed, "
                                  f"keeping original description={props.get('description','')!r}")
                            trimmed.pop("description", None)
                        if "keywords" in trimmed and not trimmed["keywords"]:
                            trimmed.pop("keywords", None)
                        replacement.update(trimmed)
                    vertex_updated = self.replace_vertex_properties(vid, replacement)
                    logger.info(f"[KG-DELETE] Vertex {vid} exact batch UPDATE: success={vertex_updated}")
                    if vertex_updated:
                        updated_count += 1
                    else:
                        errors += 1
                        err_msg = f"UPDATE shared vertex {vid} (name={vname}) failed"
                        logger.warning(f"[KG-DELETE] {err_msg}")
                        if len(error_details) < 10:
                            error_details.append(err_msg)
            except Exception as e:
                errors += 1
                import traceback
                traceback.print_exc()
                err_msg = f"Vertex iteration error for {vid if 'vid' in locals() else '?'}: {e}"
                logger.warning(f"[KG-DELETE] {err_msg}")
                if len(error_details) < 10:
                    error_details.append(err_msg)

        logger.info(f"[KG-DELETE] Delete vertices result for doc_id={doc_id}: "
              f"matched={matched_count}, deleted={deleted_count}, "
              f"updated={updated_count}, errors={errors}")
        if error_details:
            logger.error(f"[KG-DELETE] First {len(error_details)} error details:")
            for i, e in enumerate(error_details):
                logger.error(f"[KG-DELETE]   error[{i}]: {e}")
        return {
            "success": errors == 0,
            "deleted": deleted_count,
            "updated": updated_count,
            "matched": matched_count,
            "errors": errors,
            "error_details": error_details,
            "removed_names": [n for n in removed_names if n],
        }
    
    def get_graph_stats(self) -> Dict:
        if self._use_mock:
            return self._mock_client.get_graph_stats()
        
        try:
            response = self.session.get(self._get_graph_url("/stats"))
            if response.status_code != 200:
                self._raise_unavailable(response, "Get graph stats")
            return response.json()
        except Exception as e:
            if isinstance(e, HugeGraphUnavailable):
                raise
            logger.error(f"Get graph stats failed: {e}")
            raise HugeGraphUnavailable(f"Get graph stats failed: {e}") from e
    
    def get_vertices_by_kb_with_details(self, kb_id: str) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_vertices_by_kb_with_details(kb_id)

        # Keep one pagination implementation.  HugeGraph 1.7 returns the last
        # partial page again for offsets beyond the result set.  The former
        # implementation only stopped on an empty page, so that partial page
        # was repeated and projection cleanup tried to delete the same vertex
        # more than once.  ``get_vertices_by_kb`` correctly stops when a page
        # is shorter than ``limit``.
        return self.get_vertices_by_kb(kb_id)
    
    def get_edges_by_kb_with_details(self, kb_id: str) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_edges_by_kb_with_details(kb_id)

        return self.get_edges_by_kb(kb_id)

    def replace_vertex_properties(self, vertex_id: str, properties: Dict) -> bool:
        """Exactly override vertex properties through HugeGraph's batch API.

        The single-vertex ``action=append`` endpoint appends LIST values and
        therefore duplicates ``doc_id``/``source_id`` on an idempotent retry.
        HugeGraph's batch update supports ``OVERRIDE`` for both scalar and
        collection properties, which is the operation required by projection
        contribution merge and rollback.
        """

        if self._use_mock:
            vertex = self._mock_client.vertices.get(vertex_id)
            if vertex is None:
                return False
            vertex["properties"] = dict(properties)
            return True

        import urllib.parse
        try:
            encoded_vid = urllib.parse.quote(self._quote_vid(vertex_id), safe="")
            response = self.session.get(self._get_graph_url(f"/vertices/{encoded_vid}"))
            if response.status_code != 200:
                return False
            existing = response.json()
            payload = {
                "vertices": [
                    {
                        "id": vertex_id,
                        "label": existing.get("label", "Entity"),
                        "properties": dict(properties),
                    }
                ],
                "update_strategies": {
                    str(key): "OVERRIDE" for key in properties
                },
            }
            updated = self.session.put(
                self._get_graph_url("/vertices/batch"), json=payload
            )
            return updated.status_code == 200
        except Exception as exc:
            logger.error(f"Replace vertex properties failed: {exc}")
            return False

    def replace_edge_properties(self, edge_id: str, properties: Dict) -> bool:
        """Exactly override edge properties without assuming a client edge ID."""

        if self._use_mock:
            edge = self._mock_client.edges.get(edge_id)
            if edge is None:
                return False
            edge["properties"] = dict(properties)
            return True

        import urllib.parse
        try:
            encoded_eid = urllib.parse.quote(str(edge_id), safe="")
            response = self.session.get(self._get_graph_url(f"/edges/{encoded_eid}"))
            if response.status_code != 200:
                return False
            existing = response.json()
            payload = {
                "edges": [
                    {
                        "id": existing.get("id", edge_id),
                        "label": existing.get("label", "relation"),
                        "outV": existing.get("outV"),
                        "inV": existing.get("inV"),
                        "outVLabel": existing.get("outVLabel", "Entity"),
                        "inVLabel": existing.get("inVLabel", "Entity"),
                        "properties": dict(properties),
                    }
                ],
                "update_strategies": {
                    str(key): "OVERRIDE" for key in properties
                },
                "check_vertex": True,
            }
            updated = self.session.put(
                self._get_graph_url("/edges/batch"), json=payload
            )
            return updated.status_code == 200
        except Exception as exc:
            logger.error(f"Replace edge properties failed: {exc}")
            return False
    
    def update_vertex_properties(self, vertex_id: str, properties: Dict) -> bool:
        if self._use_mock:
            return self._mock_client.update_vertex_properties(vertex_id, properties)

        import urllib.parse
        try:
            encoded_vid = urllib.parse.quote(self._quote_vid(vertex_id), safe='')
            response = self.session.get(self._get_graph_url(f"/vertices/{encoded_vid}"))
            if response.status_code != 200:
                return False
            
            existing_vertex = response.json()
            existing_props = existing_vertex.get("properties", {})
            
            for key, value in properties.items():
                if key == "source_id":
                    existing_sources = existing_props.get("source_id", [])
                    if isinstance(existing_sources, str):
                        existing_sources = existing_sources.split(",")
                    if isinstance(value, list):
                        new_sources = sorted(set(existing_sources + value))
                    else:
                        new_sources = sorted(set(existing_sources + [value]))
                    existing_props["source_id"] = new_sources
                elif key == "description":
                    if "description_parts" in properties:
                        # parts 是事实来源，description 已由 _rederive_cached_fields 重算，直接覆盖
                        existing_props["description"] = value
                    else:
                        existing_desc = existing_props.get("description", "")
                        if existing_desc:
                            existing_props["description"] = existing_desc + GRAPH_FIELD_SEP + value
                        else:
                            existing_props["description"] = value
                elif key == "doc_id":
                    existing_docs = existing_props.get("doc_id", [])
                    if isinstance(existing_docs, str):
                        existing_docs = [d for d in existing_docs.split(",") if d]
                    elif isinstance(existing_docs, list):
                        existing_docs = [d for d in existing_docs if d]
                    else:
                        existing_docs = []
                    if isinstance(value, list):
                        incoming_docs = [str(v) for v in value if v]
                    elif isinstance(value, str):
                        incoming_docs = [d for d in value.split(",") if d]
                    else:
                        incoming_docs = [str(value)] if value else []
                    new_docs = sorted(set(existing_docs + incoming_docs))
                    existing_props["doc_id"] = new_docs
                else:
                    existing_props[key] = value

            return self.replace_vertex_properties(vertex_id, existing_props)
        except Exception as e:
            logger.error(f"Update vertex properties failed: {e}")
            return False
    
    def update_edge_properties(self, edge_id: str, properties: Dict) -> bool:
        if self._use_mock:
            return self._mock_client.update_edge_properties(edge_id, properties)
        
        try:
            import urllib.parse
            encoded_eid = urllib.parse.quote(str(edge_id), safe="")
            response = self.session.get(self._get_graph_url(f"/edges/{encoded_eid}"))
            if response.status_code != 200:
                return False
            
            existing_edge = response.json()
            existing_props = existing_edge.get("properties", {})
            
            for key, value in properties.items():
                if key == "source_id":
                    existing_sources = existing_props.get("source_id", [])
                    if isinstance(existing_sources, str):
                        existing_sources = existing_sources.split(",")
                    if isinstance(value, list):
                        new_sources = sorted(set(existing_sources + value))
                    else:
                        new_sources = sorted(set(existing_sources + [value]))
                    existing_props["source_id"] = new_sources
                elif key == "description":
                    if "description_parts" in properties:
                        # parts 是事实来源，description 已重算，直接覆盖
                        existing_props["description"] = value
                    else:
                        existing_desc = existing_props.get("description", "")
                        if existing_desc:
                            existing_props["description"] = existing_desc + GRAPH_FIELD_SEP + value
                        else:
                            existing_props["description"] = value
                elif key == "weight":
                    if "weight_parts" in properties:
                        # weight_parts 是事实来源，weight 已重算，直接覆盖
                        existing_props["weight"] = value
                    else:
                        existing_weight = existing_props.get("weight", 0)
                        existing_props["weight"] = existing_weight + value
                elif key == "keywords":
                    existing_keywords = existing_props.get("keywords", [])
                    if isinstance(existing_keywords, str):
                        existing_keywords = existing_keywords.split(",")
                    if isinstance(value, list):
                        new_keywords = sorted(set(existing_keywords + value))
                    else:
                        new_keywords = sorted(set(existing_keywords + [value]))
                    existing_props["keywords"] = new_keywords
                elif key == "doc_id":
                    existing_docs = existing_props.get("doc_id", [])
                    if isinstance(existing_docs, str):
                        existing_docs = [d for d in existing_docs.split(",") if d]
                    elif isinstance(existing_docs, list):
                        existing_docs = [d for d in existing_docs if d]
                    else:
                        existing_docs = []
                    if isinstance(value, list):
                        incoming_docs = [str(v) for v in value if v]
                    elif isinstance(value, str):
                        incoming_docs = [d for d in value.split(",") if d]
                    else:
                        incoming_docs = [str(value)] if value else []
                    new_docs = sorted(set(existing_docs + incoming_docs))
                    existing_props["doc_id"] = new_docs
                else:
                    existing_props[key] = value

            return self.replace_edge_properties(edge_id, existing_props)
        except Exception as e:
            logger.error(f"Update edge properties failed: {e}")
            return False
    
    def delete_vertex(self, vertex_id: str) -> bool:
        if self._use_mock:
            return self._mock_client.delete_vertex(vertex_id)

        import urllib.parse
        try:
            encoded_vid = urllib.parse.quote(self._quote_vid(vertex_id), safe='')
            response = self.session.delete(self._get_graph_url(f"/vertices/{encoded_vid}"))
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Delete vertex failed: {e}")
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        if self._use_mock:
            return self._mock_client.delete_edge(edge_id)
        
        try:
            response = self.session.delete(self._get_graph_url(f"/edges/{edge_id}"))
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Delete edge failed: {e}")
            return False
    
    def _execute_gremlin(
        self, gremlin_query: str, *, timeout: float
    ) -> List[Any]:
        """Execute one Gremlin request using HugeGraph's synchronous protocol.

        HugeGraph exposes the graph and traversal source through configured
        graphspace-qualified globals (or legacy unqualified globals when the
        graphspace setting is empty).  ``session`` belongs to a different
        TinkerPop request mode and must not be sent to this REST endpoint.
        Successful REST responses use ``status.code``; older compatible
        deployments may additionally expose ``success``.
        """

        response = self.session.post(
            f"{self.base_url}/gremlin",
            json={
                "gremlin": gremlin_query,
                "bindings": {},
                "language": "gremlin-groovy",
                "aliases": self._gremlin_aliases(),
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            self._raise_unavailable(response, "Gremlin query")

        result = response.json()
        if not isinstance(result, Mapping):
            raise HugeGraphUnavailable("Gremlin query returned a non-object response")
        status = result.get("status")
        status_code = status.get("code") if isinstance(status, Mapping) else None
        legacy_success = result.get("success")
        if status_code is not None:
            try:
                succeeded = int(status_code) == 200
            except (TypeError, ValueError):
                succeeded = False
        else:
            succeeded = legacy_success is True
        if not succeeded:
            raise HugeGraphUnavailable(
                f"Gremlin query returned an unsuccessful status: {str(result)[:200]}"
            )

        result_body = result.get("result")
        data = result_body.get("data", []) if isinstance(result_body, Mapping) else []
        if data is None:
            return []
        if not isinstance(data, list):
            raise HugeGraphUnavailable("Gremlin query result.data is not a list")
        return data

    def execute_gremlin(self, gremlin_query: str) -> List[Any]:
        if self._use_mock:
            return self._mock_client.execute_gremlin(gremlin_query)

        try:
            return self._execute_gremlin(gremlin_query, timeout=30)
        except Exception as e:
            if isinstance(e, HugeGraphUnavailable):
                raise
            logger.error(f"Gremlin query failed: {e}")
            raise HugeGraphUnavailable(f"Gremlin query failed: {e}") from e

    def capability_status(self, *, timeout: float = 3.0) -> Dict[str, Any]:
        """Probe capabilities actually used by ICP graph traversal.

        Formal ICP traversal reads the bounded graph projection through the
        vertices and edges REST collections, then traverses it in Python.  A
        healthy ``/graphs`` metadata endpoint alone does not prove those reads
        work.  Native Gremlin is probed separately because it is retained only
        for legacy callers and is not a readiness dependency.
        """

        if self._use_mock:
            reason = "client is using the in-memory fallback"
            return {
                "ok": False,
                "rest_projection_read": {
                    "ok": False,
                    "required": True,
                    "error": reason,
                },
                "gremlin_query": {
                    "ok": False,
                    "required": False,
                    "error": reason,
                },
            }

        collection_errors: List[str] = []
        for endpoint, response_key in (
            ("/vertices", "vertices"),
            ("/edges", "edges"),
        ):
            try:
                response = self.session.get(
                    self._get_graph_url(endpoint),
                    params={"limit": 1},
                    timeout=timeout,
                )
                self._raise_unavailable(
                    response, f"HugeGraph {response_key} capability probe"
                )
                payload = response.json()
                if not isinstance(payload, Mapping) or not isinstance(
                    payload.get(response_key), list
                ):
                    raise HugeGraphUnavailable(
                        f"HugeGraph {response_key} capability probe returned "
                        "an invalid collection response"
                    )
            except Exception as exc:
                collection_errors.append(
                    f"{response_key}: {self._compact_error(exc)}"
                )

        rest_ok = not collection_errors
        rest_status: Dict[str, Any] = {"ok": rest_ok, "required": True}
        if collection_errors:
            rest_status["error"] = "; ".join(collection_errors)[:500]

        try:
            # This is deliberately read-only and returns zero without scanning
            # or serializing application vertices.
            self._execute_gremlin("g.V().limit(0).count()", timeout=timeout)
            gremlin_status: Dict[str, Any] = {"ok": True, "required": False}
        except Exception as exc:
            gremlin_status = {
                "ok": False,
                "required": False,
                "error": self._compact_error(exc),
            }

        return {
            # Only the projection collection reads are a hard dependency of
            # the current LangGraph tool.  Gremlin remains visible as degraded
            # capability without incorrectly forcing /readyz to 503.
            "ok": rest_ok,
            "rest_projection_read": rest_status,
            "gremlin_query": gremlin_status,
        }

    def _read_projection_collection_bounded(
        self,
        endpoint: str,
        response_key: str,
        *,
        max_items: int,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Read a REST projection collection with a hard item ceiling.

        HugeGraph's offset pagination stops advancing beyond an implementation
        limit, while its opaque ``page`` cursor is designed for complete graph
        scans.  Cursor paging cannot be combined with property filters, so the
        bounded global collection is filtered by the caller after readback.
        """

        if max_items <= 0:
            raise ValueError("max_items must be positive")
        collected: Dict[str, Dict[str, Any]] = {}
        page_token = ""
        seen_tokens = set()
        try:
            while len(collected) <= max_items:
                remaining_probe = max_items + 1 - len(collected)
                page_limit = min(_PROJECTION_READ_PAGE_SIZE, remaining_probe)
                query = {"limit": page_limit, "page": page_token}
                response = self.session.get(
                    self._get_graph_url(endpoint),
                    params=query,
                    timeout=timeout,
                )
                self._raise_unavailable(
                    response, f"HugeGraph bounded {response_key} read"
                )
                payload = response.json()
                page = payload.get(response_key) if isinstance(payload, Mapping) else None
                if not isinstance(page, list):
                    raise HugeGraphUnavailable(
                        f"HugeGraph bounded {response_key} read returned "
                        "an invalid collection response"
                    )
                for item in page:
                    if not isinstance(item, Mapping) or not str(item.get("id") or ""):
                        raise HugeGraphUnavailable(
                            f"HugeGraph bounded {response_key} read returned "
                            "an item without a stable id"
                        )
                    item_id = str(item["id"])
                    if item_id not in collected:
                        collected[item_id] = dict(item)
                        if len(collected) > max_items:
                            raise HugeGraphUnavailable(
                                f"HugeGraph {response_key} exceeds bounded path limit "
                                f"({max_items})"
                            )
                next_token = (
                    str(payload.get("page") or "")
                    if isinstance(payload, Mapping)
                    else ""
                )
                if not next_token:
                    break
                if next_token == page_token or next_token in seen_tokens:
                    raise HugeGraphUnavailable(
                        f"HugeGraph {response_key} page cursor did not advance"
                    )
                seen_tokens.add(next_token)
                page_token = next_token
        except HugeGraphUnavailable:
            raise
        except Exception as exc:
            raise HugeGraphUnavailable(
                f"HugeGraph bounded {response_key} read failed: {exc}"
            ) from exc
        return list(collected.values())

    @staticmethod
    def _property_contains(properties: Mapping[str, Any], key: str, value: str) -> bool:
        raw = properties.get(key)
        if isinstance(raw, str):
            candidates = [item for item in raw.split(",") if item]
        elif isinstance(raw, (list, tuple, set, frozenset)):
            candidates = [str(item) for item in raw]
        else:
            candidates = [str(raw)] if raw not in (None, "") else []
        return value in candidates

    @classmethod
    def _shortest_path_from_projection(
        cls,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        *,
        kb_id: str,
        source_name: str,
        target_name: str,
        max_hops: int = _SHORTEST_PATH_MAX_HOPS,
    ) -> List[Dict[str, Any]]:
        """Return one deterministic undirected shortest path from REST data."""

        vertex_map: Dict[str, Dict[str, Any]] = {}
        for vertex in vertices:
            properties = vertex.get("properties") or {}
            if not isinstance(properties, Mapping) or not cls._property_contains(
                properties, "kb_id", kb_id
            ):
                continue
            vertex_id = str(vertex.get("id") or "")
            if vertex_id:
                vertex_map[vertex_id] = vertex

        source_ids = sorted(
            vertex_id
            for vertex_id, vertex in vertex_map.items()
            if str((vertex.get("properties") or {}).get("name") or "")
            == source_name
        )
        target_ids = sorted(
            vertex_id
            for vertex_id, vertex in vertex_map.items()
            if str((vertex.get("properties") or {}).get("name") or "")
            == target_name
        )
        if not source_ids or not target_ids:
            return []

        # The old endpoint selected one exact-name vertex.  Sorting makes that
        # behavior deterministic when legacy data contains duplicate names.
        source_id = source_ids[0]
        target_id = target_ids[0]
        edge_map: Dict[str, Dict[str, Any]] = {}
        adjacency: Dict[str, List[Tuple[str, str]]] = {
            vertex_id: [] for vertex_id in vertex_map
        }
        for edge in edges:
            properties = edge.get("properties") or {}
            if not isinstance(properties, Mapping) or not cls._property_contains(
                properties, "kb_id", kb_id
            ):
                continue
            edge_id = str(edge.get("id") or "")
            out_v = str(edge.get("outV") or edge.get("source") or "")
            in_v = str(edge.get("inV") or edge.get("target") or "")
            if not edge_id or out_v not in vertex_map or in_v not in vertex_map:
                continue
            edge_map[edge_id] = edge
            adjacency[out_v].append((in_v, edge_id))
            adjacency[in_v].append((out_v, edge_id))
        for neighbors in adjacency.values():
            neighbors.sort()

        parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            source_id: (None, None)
        }
        depths = {source_id: 0}
        queue = deque([source_id])
        while queue and target_id not in parents:
            current = queue.popleft()
            if depths[current] >= max_hops:
                continue
            for neighbor, edge_id in adjacency.get(current, []):
                if neighbor in parents:
                    continue
                parents[neighbor] = (current, edge_id)
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)

        if target_id not in parents:
            return []

        vertex_ids: List[str] = []
        edge_ids: List[str] = []
        current: Optional[str] = target_id
        while current is not None:
            vertex_ids.append(current)
            parent, edge_id = parents[current]
            if edge_id is not None:
                edge_ids.append(edge_id)
            current = parent
        vertex_ids.reverse()
        edge_ids.reverse()

        path: List[Dict[str, Any]] = []
        for index, vertex_id in enumerate(vertex_ids):
            vertex = vertex_map[vertex_id]
            properties = vertex.get("properties") or {}
            item = {
                "id": vertex_id,
                "label": vertex.get("label"),
                "name": properties.get("name"),
            }
            if index:
                edge = edge_map[edge_ids[index - 1]]
                edge_properties = edge.get("properties") or {}
                item["relation"] = (
                    edge.get("label")
                    or edge_properties.get("relation_type")
                    or ""
                )
            path.append(item)
        return [path]
    
    def search_vertices_by_name(self, kb_id: str, name: str, limit: int = 20) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.search_vertices_by_name(kb_id, name, limit)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            filtered = []
            for v in all_vertices:
                v_name = v.get('properties', {}).get('name', '')
                if name.lower() in v_name.lower():
                    filtered.append(v)
                if len(filtered) >= limit:
                    break
            return filtered
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Search vertices by name failed: {e}")
            return []
    
    def search_vertices_by_keyword(self, kb_id: str, keyword: str, limit: int = 20) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.search_vertices_by_keyword(kb_id, keyword, limit)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            filtered = []
            for v in all_vertices:
                keywords = v.get('properties', {}).get('keywords', [])
                if isinstance(keywords, str):
                    keywords = keywords.split(',')
                if isinstance(keywords, list):
                    for kw in keywords:
                        if keyword.lower() in str(kw).lower():
                            filtered.append(v)
                            break
                if len(filtered) >= limit:
                    break
            return filtered
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Search vertices by keyword failed: {e}")
            return []
    
    def get_neighbors(self, vertex_id: str, depth: int = 1, limit: int = 50) -> Dict:
        if self._use_mock:
            return self._mock_client.get_neighbors(vertex_id, depth, limit)
        
        try:
            kb_id = vertex_id.split("_")[0] if "_" in vertex_id else ""
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            all_edges = self.get_edges_by_kb_with_details(kb_id)
            
            vertex_map = {v.get("id"): v for v in all_vertices}
            edge_map = {}
            for e in all_edges:
                key1 = (e.get("outV"), e.get("inV"))
                key2 = (e.get("inV"), e.get("outV"))
                edge_map[key1] = e
                edge_map[key2] = e
            
            result = {
                "vertex_id": vertex_id,
                "neighbors": [],
                "paths": []
            }
            
            visited = set([vertex_id])
            current_level = [vertex_id]
            
            for level in range(depth):
                next_level = []
                for v_id in current_level:
                    for e in all_edges:
                        out_v = e.get("outV")
                        in_v = e.get("inV")
                        edge_label = e.get("label", "")
                        edge_props = e.get("properties", {})
                        edge_weight = edge_props.get("weight", 1)
                        edge_keywords = edge_props.get("keywords", [])
                        if isinstance(edge_keywords, str):
                            edge_keywords = edge_keywords.split(",")
                        
                        if out_v == v_id and in_v not in visited:
                            visited.add(in_v)
                            next_level.append(in_v)
                            if in_v in vertex_map:
                                v = vertex_map[in_v]
                                props = v.get("properties", {})
                                neighbor_info = {
                                    "id": v.get("id"),
                                    "label": v.get("label"),
                                    "name": props.get("name"),
                                    "relation": edge_label,
                                    "weight": edge_weight,
                                    "keywords": edge_keywords,
                                    "relation_type": edge_props.get("relation_type", edge_label),
                                    "distance": level + 1
                                }
                                if neighbor_info["id"] not in [n["id"] for n in result["neighbors"]]:
                                    result["neighbors"].append(neighbor_info)
                        
                        elif in_v == v_id and out_v not in visited:
                            visited.add(out_v)
                            next_level.append(out_v)
                            if out_v in vertex_map:
                                v = vertex_map[out_v]
                                props = v.get("properties", {})
                                neighbor_info = {
                                    "id": v.get("id"),
                                    "label": v.get("label"),
                                    "name": props.get("name"),
                                    "relation": edge_label,
                                    "weight": edge_weight,
                                    "keywords": edge_keywords,
                                    "relation_type": edge_props.get("relation_type", edge_label),
                                    "distance": level + 1
                                }
                                if neighbor_info["id"] not in [n["id"] for n in result["neighbors"]]:
                                    result["neighbors"].append(neighbor_info)
                
                current_level = next_level
                if len(result["neighbors"]) >= limit:
                    break
            
            result["neighbors"] = result["neighbors"][:limit]
            return result
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Get neighbors failed: {e}")
            return {"vertex_id": vertex_id, "neighbors": [], "paths": []}
    
    def get_entity_relations(self, kb_id: str, entity_name: str, limit: int = 50) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_entity_relations(kb_id, entity_name, limit)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            all_edges = self.get_edges_by_kb_with_details(kb_id)
            
            entity_id = None
            for v in all_vertices:
                if v.get('properties', {}).get('name') == entity_name:
                    entity_id = v.get('id')
                    break
            
            if not entity_id:
                return []
            
            relations = []
            related_ids = set()
            for e in all_edges:
                out_v = e.get('outV')
                in_v = e.get('inV')
                if out_v == entity_id and in_v not in related_ids:
                    related_ids.add(in_v)
                elif in_v == entity_id and out_v not in related_ids:
                    related_ids.add(out_v)
            
            for v_id in related_ids:
                for v in all_vertices:
                    if v.get('id') == v_id:
                        props = v.get('properties', {})
                        relations.append({
                            "id": v.get("id"),
                            "label": v.get("label"),
                            "name": props.get("name"),
                            "entity_type": props.get("entity_type"),
                            "description": props.get("description")
                        })
                        break
                if len(relations) >= limit:
                    break
            
            return relations
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Get entity relations failed: {e}")
            return []
    
    def get_related_chunks_by_entities(self, kb_id: str, entity_names: List[str]) -> List[str]:
        if not entity_names:
            return []
        
        if self._use_mock:
            return self._mock_client.get_related_chunks_by_entities(kb_id, entity_names)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            source_ids = set()
            
            for v in all_vertices:
                name = v.get('properties', {}).get('name', '')
                if name in entity_names:
                    source_id = v.get('properties', {}).get('source_id', '')
                    if source_id:
                        if isinstance(source_id, str):
                            source_ids.update(source_id.split(","))
                        elif isinstance(source_id, list):
                            source_ids.update(source_id)
            
            return list(source_ids)
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Get related chunks by entities failed: {e}")
            return []
    
    def search_by_entity_type(self, kb_id: str, entity_type: str, limit: int = 100) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.search_by_entity_type(kb_id, entity_type, limit)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            entities = []
            for v in all_vertices:
                props = v.get("properties", {})
                if props.get("entity_type") == entity_type:
                    entities.append({
                        "id": v.get("id"),
                        "label": v.get("label"),
                        "name": props.get("name"),
                        "entity_type": props.get("entity_type"),
                        "description": props.get("description"),
                        "weight": props.get("weight"),
                        "pagerank": props.get("pagerank")
                    })
                if len(entities) >= limit:
                    break
            return entities
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Search by entity type failed: {e}")
            return []
    
    def get_top_entities_by_pagerank(self, kb_id: str, limit: int = 20) -> List[Dict]:
        if self._use_mock:
            return self._mock_client.get_top_entities_by_pagerank(kb_id, limit)
        
        try:
            all_vertices = self.get_vertices_by_kb_with_details(kb_id)
            entities = []
            for v in all_vertices:
                props = v.get("properties", {})
                entities.append({
                    "id": v.get("id"),
                    "label": v.get("label"),
                    "name": props.get("name"),
                    "entity_type": props.get("entity_type"),
                    "description": props.get("description"),
                    "pagerank": props.get("pagerank"),
                    "weight": props.get("weight")
                })
            
            entities.sort(key=lambda x: x.get("pagerank", 0) or 0, reverse=True)
            return entities[:limit]
        except HugeGraphUnavailable:
            raise
        except Exception as e:
            logger.error(f"Get top entities by pagerank failed: {e}")
            return []
    
    def get_shortest_path(self, kb_id: str, source_name: str, target_name: str) -> List[Dict]:
        if self._use_mock:
            vertices = self._mock_client.get_vertices_by_kb_with_details(kb_id)
            edges = self._mock_client.get_edges_by_kb_with_details(kb_id)
            if len(vertices) > _SHORTEST_PATH_MAX_VERTICES:
                raise HugeGraphUnavailable("mock vertices exceed bounded path limit")
            if len(edges) > _SHORTEST_PATH_MAX_EDGES:
                raise HugeGraphUnavailable("mock edges exceed bounded path limit")
            return self._shortest_path_from_projection(
                vertices,
                edges,
                kb_id=kb_id,
                source_name=str(source_name),
                target_name=str(target_name),
            )

        # User-provided names remain ordinary Python strings throughout this
        # path.  They are never interpolated into Gremlin/Groovy source.
        vertices = self._read_projection_collection_bounded(
            "/vertices",
            "vertices",
            max_items=_SHORTEST_PATH_MAX_VERTICES,
        )
        available_names = {
            str(properties.get("name") or "")
            for vertex in vertices
            if isinstance((properties := vertex.get("properties") or {}), Mapping)
            and self._property_contains(properties, "kb_id", str(kb_id))
        }
        if (
            str(source_name) not in available_names
            or str(target_name) not in available_names
        ):
            return []
        edges = self._read_projection_collection_bounded(
            "/edges",
            "edges",
            max_items=_SHORTEST_PATH_MAX_EDGES,
        )
        return self._shortest_path_from_projection(
            vertices,
            edges,
            kb_id=str(kb_id),
            source_name=str(source_name),
            target_name=str(target_name),
        )

GRAPH_FIELD_SEP = "<SEP>"

class MockHugeGraph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
    
    def batch_insert(self, vertices: List[Dict], edges: List[Dict]) -> bool:
        for v in vertices:
            vid = v.get("id")
            if vid:
                self.vertices[vid] = v
        
        for e in edges:
            eid = e.get("id")
            if eid:
                self.edges[eid] = e
        
        return True
    
    def get_graph_data(self) -> Dict:
        return {"vertices": list(self.vertices.values()), "edges": list(self.edges.values())}
    
    def get_vertices_by_kb(self, kb_id: str, label: str = None) -> List[Dict]:
        result = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                if label is None or v.get("label") == label:
                    result.append(v)
        return result
    
    def get_edges_by_kb(self, kb_id: str, label: str = None) -> List[Dict]:
        result = []
        for e in self.edges.values():
            if e.get("properties", {}).get("kb_id") == kb_id:
                if label is None or e.get("label") == label:
                    result.append(e)
        return result
    
    def delete_by_kb_id(self, kb_id: str):
        self.vertices = {k: v for k, v in self.vertices.items() 
                         if v.get("properties", {}).get("kb_id") != kb_id}
        self.edges = {k: v for k, v in self.edges.items() 
                      if v.get("properties", {}).get("kb_id") != kb_id}
        return True
    
    def _matches_doc(self, source_id, doc_id_list, doc_id):
        if isinstance(source_id, str):
            source_id = source_id.split(",")
        if not isinstance(source_id, list):
            source_id = []
        
        matches = any(s.startswith(f"{doc_id}_") or s == doc_id for s in source_id)
        if not matches:
            matches = doc_id in doc_id_list
        return matches

    def delete_edges_by_doc_id(self, doc_id: str):
        edges_to_remove = []
        edges_to_update = []
        
        for eid, edge in self.edges.items():
            props = edge.get("properties", {})

            if not HugeGraphClient._matches_doc(props, doc_id):
                continue

            remaining = HugeGraphClient._remove_doc_id_from_props(props, doc_id)
            # 只要还有其他 doc_id 存在即为共享
            raw_doc_id = props.get("doc_id", "")
            was_in_field = (isinstance(raw_doc_id, list) and raw_doc_id) or (isinstance(raw_doc_id, str) and raw_doc_id)
            is_shared = was_in_field and len(remaining) > 0

            if not is_shared:
                edges_to_remove.append(eid)
            else:
                edges_to_update.append({
                    "eid": eid,
                    "doc_id": remaining,
                    "props": props,
                })

        for eid in edges_to_remove:
            del self.edges[eid]

        for update in edges_to_update:
            eid = update["eid"]
            if eid in self.edges:
                self.edges[eid]["properties"]["doc_id"] = update["doc_id"]
                # 同步裁剪 parts 并重算 description/keywords/weight
                trimmed = HugeGraphClient._trim_parts_by_doc(update["props"], doc_id)
                self.edges[eid]["properties"].update(trimmed)

        return True

    def delete_vertices_by_doc_id(self, doc_id: str):
        vertices_to_remove = []
        vertices_to_update = []
        removed_names = []

        for vid, vertex in self.vertices.items():
            props = vertex.get("properties", {})

            if not HugeGraphClient._matches_doc(props, doc_id):
                continue

            remaining = HugeGraphClient._remove_doc_id_from_props(props, doc_id)
            raw_doc_id = props.get("doc_id", "")
            was_in_field = (isinstance(raw_doc_id, list) and raw_doc_id) or (isinstance(raw_doc_id, str) and raw_doc_id)
            is_shared = was_in_field and len(remaining) > 0

            if not is_shared:
                vertices_to_remove.append(vid)
                if props.get("name"):
                    removed_names.append(props.get("name"))
            else:
                if not remaining:
                    vertices_to_remove.append(vid)
                    if props.get("name"):
                        removed_names.append(props.get("name"))
                else:
                    vertices_to_update.append({
                        "vid": vid,
                        "doc_id": remaining,
                        "props": props,
                    })

        for vid in vertices_to_remove:
            edges_to_remove = []
            for eid, edge in self.edges.items():
                if edge.get("outV") == vid or edge.get("inV") == vid:
                    edges_to_remove.append(eid)
            for eid in edges_to_remove:
                del self.edges[eid]
            del self.vertices[vid]

        for update in vertices_to_update:
            vid = update["vid"]
            if vid in self.vertices:
                self.vertices[vid]["properties"]["doc_id"] = update["doc_id"]
                # 同步裁剪 parts 并重算 description/keywords
                trimmed = HugeGraphClient._trim_parts_by_doc(update["props"], doc_id)
                self.vertices[vid]["properties"].update(trimmed)

        return {
            "success": True,
            "deleted": len(vertices_to_remove),
            "updated": len(vertices_to_update),
            "matched": len(vertices_to_remove) + len(vertices_to_update),
            "removed_names": removed_names,
        }
    
    def get_graph_stats(self) -> Dict:
        return {
            "vertices": len(self.vertices),
            "edges": len(self.edges)
        }
    
    def get_vertices_by_kb_with_details(self, kb_id: str) -> List[Dict]:
        result = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                result.append(v)
        return result
    
    def get_edges_by_kb_with_details(self, kb_id: str) -> List[Dict]:
        result = []
        for e in self.edges.values():
            if e.get("properties", {}).get("kb_id") == kb_id:
                result.append(e)
        return result
    
    def update_vertex_properties(self, vertex_id: str, properties: Dict) -> bool:
        if vertex_id not in self.vertices:
            return False
        
        existing_props = self.vertices[vertex_id].get("properties", {})
        
        for key, value in properties.items():
            if key == "source_id":
                existing_sources = existing_props.get("source_id", [])
                if isinstance(existing_sources, str):
                    existing_sources = existing_sources.split(",")
                if isinstance(value, list):
                    new_sources = sorted(set(existing_sources + value))
                else:
                    new_sources = sorted(set(existing_sources + [value]))
                existing_props["source_id"] = new_sources
            elif key == "description":
                existing_desc = existing_props.get("description", "")
                if existing_desc:
                    existing_props["description"] = existing_desc + GRAPH_FIELD_SEP + value
                else:
                    existing_props["description"] = value
            elif key == "doc_id":
                existing_docs = existing_props.get("doc_id", [])
                if isinstance(existing_docs, str):
                    existing_docs = existing_docs.split(",")
                if isinstance(value, list):
                    new_docs = sorted(set(existing_docs + value))
                else:
                    new_docs = sorted(set(existing_docs + [value]))
                existing_props["doc_id"] = new_docs
            else:
                existing_props[key] = value
        
        return True
    
    def update_edge_properties(self, edge_id: str, properties: Dict) -> bool:
        if edge_id not in self.edges:
            return False
        
        existing_props = self.edges[edge_id].get("properties", {})
        
        for key, value in properties.items():
            if key == "source_id":
                existing_sources = existing_props.get("source_id", [])
                if isinstance(existing_sources, str):
                    existing_sources = existing_sources.split(",")
                if isinstance(value, list):
                    new_sources = sorted(set(existing_sources + value))
                else:
                    new_sources = sorted(set(existing_sources + [value]))
                existing_props["source_id"] = new_sources
            elif key == "description":
                existing_desc = existing_props.get("description", "")
                if existing_desc:
                    existing_props["description"] = existing_desc + GRAPH_FIELD_SEP + value
                else:
                    existing_props["description"] = value
            elif key == "weight":
                existing_weight = existing_props.get("weight", 0)
                existing_props["weight"] = existing_weight + value
            elif key == "keywords":
                existing_keywords = existing_props.get("keywords", [])
                if isinstance(existing_keywords, str):
                    existing_keywords = existing_keywords.split(",")
                if isinstance(value, list):
                    new_keywords = sorted(set(existing_keywords + value))
                else:
                    new_keywords = sorted(set(existing_keywords + [value]))
                existing_props["keywords"] = new_keywords
            elif key == "doc_id":
                existing_docs = existing_props.get("doc_id", [])
                if isinstance(existing_docs, str):
                    existing_docs = existing_docs.split(",")
                if isinstance(value, list):
                    new_docs = sorted(set(existing_docs + value))
                else:
                    new_docs = sorted(set(existing_docs + [value]))
                existing_props["doc_id"] = new_docs
            else:
                existing_props[key] = value
        
        return True
    
    def delete_vertex(self, vertex_id: str) -> bool:
        if vertex_id in self.vertices:
            del self.vertices[vertex_id]
            edges_to_remove = []
            for eid, edge in self.edges.items():
                if edge.get("outV") == vertex_id or edge.get("inV") == vertex_id:
                    edges_to_remove.append(eid)
            for eid in edges_to_remove:
                del self.edges[eid]
            return True
        return False
    
    def delete_edge(self, edge_id: str) -> bool:
        if edge_id in self.edges:
            del self.edges[edge_id]
            return True
        return False
    
    def execute_gremlin(self, gremlin_query: str) -> List[Dict]:
        if "textContains" in gremlin_query and "has('name'" in gremlin_query:
            import re
            match = re.search(r"textContains\('([^']+)'\)", gremlin_query)
            if match:
                keyword = match.group(1)
                result = []
                for v in self.vertices.values():
                    name = v.get("properties", {}).get("name", "")
                    if keyword.lower() in str(name).lower():
                        result.append(v)
                        if len(result) >= 20:
                            break
                return result
        
        elif "has('keywords'" in gremlin_query:
            import re
            match = re.search(r"textContains\('([^']+)'\)", gremlin_query)
            if match:
                keyword = match.group(1)
                result = []
                for v in self.vertices.values():
                    keywords = v.get("properties", {}).get("keywords", "")
                    if keyword.lower() in str(keywords).lower():
                        result.append(v)
                        if len(result) >= 20:
                            break
                return result
        
        elif "within" in gremlin_query and "values('source_id')" in gremlin_query:
            import re
            match = re.search(r"within\('([^']+)'\)", gremlin_query)
            if match:
                entity_names = match.group(1).split("', '")
                source_ids = set()
                for v in self.vertices.values():
                    name = v.get("properties", {}).get("name", "")
                    if name in entity_names:
                        source_id = v.get("properties", {}).get("source_id", "")
                        if source_id:
                            source_ids.update(str(source_id).split(","))
                return list(source_ids)
        
        elif "has('entity_type'" in gremlin_query:
            import re
            match = re.search(r"has\('entity_type', '([^']+)'\)", gremlin_query)
            if match:
                entity_type = match.group(1)
                result = []
                for v in self.vertices.values():
                    etype = v.get("properties", {}).get("entity_type", "")
                    if etype == entity_type:
                        result.append(v)
                        if len(result) >= 100:
                            break
                return result
        
        return []
    
    def search_vertices_by_name(self, kb_id: str, name: str, limit: int = 20) -> List[Dict]:
        result = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                v_name = v.get("properties", {}).get("name", "")
                if name.lower() in str(v_name).lower():
                    result.append(v)
                    if len(result) >= limit:
                        break
        return result
    
    def search_vertices_by_keyword(self, kb_id: str, keyword: str, limit: int = 20) -> List[Dict]:
        result = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                keywords = v.get("properties", {}).get("keywords", "")
                if keyword.lower() in str(keywords).lower():
                    result.append(v)
                    if len(result) >= limit:
                        break
        return result
    
    def get_neighbors(self, vertex_id: str, depth: int = 1, limit: int = 50) -> Dict:
        result = {
            "vertex_id": vertex_id,
            "neighbors": [],
            "paths": []
        }
        
        if vertex_id not in self.vertices:
            return result
        
        visited = {vertex_id}
        current_level = [vertex_id]
        
        for _ in range(depth):
            next_level = []
            for vid in current_level:
                for eid, edge in self.edges.items():
                    if edge.get("outV") == vid and edge.get("inV") not in visited:
                        visited.add(edge.get("inV"))
                        next_level.append(edge.get("inV"))
                        neighbor = self.vertices.get(edge.get("inV"))
                        if neighbor:
                            props = neighbor.get("properties", {})
                            result["neighbors"].append({
                                "id": edge.get("inV"),
                                "label": neighbor.get("label"),
                                "name": props.get("name"),
                                "relation": edge.get("label")
                            })
                    elif edge.get("inV") == vid and edge.get("outV") not in visited:
                        visited.add(edge.get("outV"))
                        next_level.append(edge.get("outV"))
                        neighbor = self.vertices.get(edge.get("outV"))
                        if neighbor:
                            props = neighbor.get("properties", {})
                            result["neighbors"].append({
                                "id": edge.get("outV"),
                                "label": neighbor.get("label"),
                                "name": props.get("name"),
                                "relation": edge.get("label")
                            })
            current_level = next_level
            if len(result["neighbors"]) >= limit:
                break
        
        return result
    
    def get_entity_relations(self, kb_id: str, entity_name: str, limit: int = 50) -> List[Dict]:
        relations = []
        target_vid = None
        
        for vid, v in self.vertices.items():
            if v.get("properties", {}).get("kb_id") == kb_id:
                if v.get("properties", {}).get("name") == entity_name:
                    target_vid = vid
                    break
        
        if not target_vid:
            return relations
        
        for eid, edge in self.edges.items():
            if edge.get("outV") == target_vid or edge.get("inV") == target_vid:
                other_vid = edge.get("inV") if edge.get("outV") == target_vid else edge.get("outV")
                other_v = self.vertices.get(other_vid)
                if other_v:
                    props = other_v.get("properties", {})
                    relations.append({
                        "id": other_vid,
                        "label": other_v.get("label"),
                        "name": props.get("name"),
                        "entity_type": props.get("entity_type"),
                        "description": props.get("description")
                    })
                    if len(relations) >= limit:
                        break
        
        return relations
    
    def get_related_chunks_by_entities(self, kb_id: str, entity_names: List[str]) -> List[str]:
        source_ids = set()
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                name = v.get("properties", {}).get("name", "")
                if name in entity_names:
                    source_id = v.get("properties", {}).get("source_id", "")
                    if isinstance(source_id, list):
                        source_ids.update(source_id)
                    elif isinstance(source_id, str):
                        source_ids.update(source_id.split(","))
        return list(source_ids)
    
    def search_by_entity_type(self, kb_id: str, entity_type: str, limit: int = 100) -> List[Dict]:
        entities = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                if v.get("properties", {}).get("entity_type") == entity_type:
                    props = v.get("properties", {})
                    entities.append({
                        "id": v.get("id"),
                        "label": v.get("label"),
                        "name": props.get("name"),
                        "entity_type": props.get("entity_type"),
                        "description": props.get("description"),
                        "weight": props.get("weight"),
                        "pagerank": props.get("pagerank")
                    })
                    if len(entities) >= limit:
                        break
        return entities
    
    def get_top_entities_by_pagerank(self, kb_id: str, limit: int = 20) -> List[Dict]:
        entities = []
        for v in self.vertices.values():
            if v.get("properties", {}).get("kb_id") == kb_id:
                props = v.get("properties", {})
                entities.append({
                    "id": v.get("id"),
                    "label": v.get("label"),
                    "name": props.get("name"),
                    "entity_type": props.get("entity_type"),
                    "description": props.get("description"),
                    "pagerank": props.get("pagerank", 0),
                    "weight": props.get("weight")
                })
        
        entities.sort(key=lambda x: x.get("pagerank", 0), reverse=True)
        return entities[:limit]
    
    def get_shortest_path(self, kb_id: str, source_name: str, target_name: str) -> List[Dict]:
        source_vid = None
        target_vid = None
        
        for vid, v in self.vertices.items():
            if v.get("properties", {}).get("kb_id") == kb_id:
                if v.get("properties", {}).get("name") == source_name:
                    source_vid = vid
                if v.get("properties", {}).get("name") == target_name:
                    target_vid = vid
        
        if not source_vid or not target_vid:
            return []
        
        from collections import deque
        queue = deque([(source_vid, [])])
        visited = {source_vid}
        
        while queue:
            current_vid, path = queue.popleft()
            
            if current_vid == target_vid:
                path_info = []
                for vid in path + [target_vid]:
                    v = self.vertices.get(vid)
                    if v:
                        props = v.get("properties", {})
                        path_info.append({
                            "id": vid,
                            "label": v.get("label"),
                            "name": props.get("name")
                        })
                return [path_info]
            
            for eid, edge in self.edges.items():
                if edge.get("outV") == current_vid and edge.get("inV") not in visited:
                    visited.add(edge.get("inV"))
                    queue.append((edge.get("inV"), path + [current_vid]))
                elif edge.get("inV") == current_vid and edge.get("outV") not in visited:
                    visited.add(edge.get("outV"))
                    queue.append((edge.get("outV"), path + [current_vid]))
        
        return []

hugegraph_client = HugeGraphClient()
