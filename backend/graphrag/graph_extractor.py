import re
import json
import uuid
import logging
import datetime
import itertools
import editdistance
import html
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Any, Set, Optional, Callable, cast
from backend.core.llm_service import llm_service

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from graspologic.partition import hierarchical_leiden
    from graspologic.utils import largest_connected_component
    USE_GRASPOLOGIC = True
except ImportError:
    USE_GRASPOLOGIC = False
    logger.warning("graspologic not available, falling back to networkx community detection")


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    fixed_graph = nx.Graph()
    sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))
    
    if not graph.is_directed():
        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                source, target = target, source
            return source, target, edge_data
        edges = [_sort_source_target(edge) for edge in edges]
    
    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"
    
    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
    fixed_graph.add_edges_from(edges)
    return fixed_graph


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}
    return nx.relabel_nodes(graph, node_mapping)


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    if USE_GRASPOLOGIC:
        graph = cast(nx.Graph, largest_connected_component(graph))
    else:
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(components[0]).copy() if components else graph
    graph = normalize_node_names(graph)
    return _stabilize_graph(graph)


def _compute_leiden_communities(
    graph: nx.Graph,
    max_cluster_size: int = 12,
    use_lcc: bool = False,
    seed: int = 0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    results: dict[int, dict[str, int]] = {}
    if nx.is_empty(graph):
        return results
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    if USE_GRASPOLOGIC:
        components = list(nx.connected_components(graph))
        if len(components) == 1:
            community_mapping = hierarchical_leiden(
                graph, max_cluster_size=max_cluster_size, random_seed=seed
            )
            for partition in community_mapping:
                results[partition.level] = results.get(partition.level, {})
                results[partition.level][partition.node] = partition.cluster
        else:
            global_cluster_offset = 0
            for comp_nodes in components:
                subgraph = graph.subgraph(comp_nodes).copy()
                if nx.is_empty(subgraph):
                    continue
                sub_results = {}
                community_mapping = hierarchical_leiden(
                    subgraph, max_cluster_size=max_cluster_size, random_seed=seed
                )
                for partition in community_mapping:
                    sub_results[partition.level] = sub_results.get(partition.level, {})
                    sub_results[partition.level][partition.node] = partition.cluster

                for level, mapping in sub_results.items():
                    if level not in results:
                        results[level] = {}
                    for node, cluster_id in mapping.items():
                        results[level][node] = cluster_id + global_cluster_offset

                max_cluster_id = 0
                for mapping in sub_results.values():
                    if mapping:
                        max_cluster_id = max(max_cluster_id, max(mapping.values()))
                global_cluster_offset += max_cluster_id + 1
    else:
        communities = list(nx.community.greedy_modularity_communities(graph))
        results[0] = {}
        for community_id, community in enumerate(communities):
            for node in community:
                results[0][node] = community_id

    return results


def run_leiden(graph: nx.Graph, args: dict[str, Any] = {}) -> dict[int, dict[str, dict]]:
    max_cluster_size = args.get("max_cluster_size", 12)
    use_lcc = args.get("use_lcc", False)
    nodes = set(graph.nodes())
    if not nodes:
        return {}

    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])

    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    results_by_level: dict[int, dict[str, dict]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            if node_id not in nodes:
                logger.warning(f"Node {node_id} not found in the graph.")
                continue
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = {"weight": 0, "nodes": []}
            result[community_id]["nodes"].append(node_id)
            result[community_id]["weight"] += graph.nodes[node_id].get("rank", 0) * graph.nodes[node_id].get("weight", 1)
        
        weights = [comm["weight"] for _, comm in result.items()]
        if not weights:
            continue
        max_weight = max(weights)
        if max_weight == 0:
            continue
        for _, comm in result.items():
            comm["weight"] /= max_weight

    return results_by_level


def _compute_adaptive_max_cluster_size(graph: nx.Graph) -> int:
    n_nodes = graph.number_of_nodes()
    if n_nodes <= 10:
        return 4
    elif n_nodes <= 20:
        return 6
    elif n_nodes <= 40:
        return 8
    elif n_nodes <= 80:
        return 10
    else:
        return 12


def add_community_info2graph(graph: nx.Graph, nodes: list[str], community_title: str):
    for n in nodes:
        if "communities" not in graph.nodes[n]:
            graph.nodes[n]["communities"] = []
        graph.nodes[n]["communities"].append(community_title)
        graph.nodes[n]["communities"] = list(set(graph.nodes[n]["communities"]))


def dict_has_keys_with_types(d: dict, required: list[tuple[str, type]]) -> bool:
    for key, expected_type in required:
        if key not in d:
            return False
        if not isinstance(d[key], expected_type):
            return False
    return True

GRAPH_FIELD_SEP = "<SEP>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category", "fault", "cause", "action", "component"]

COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format(in language of 'Text' content):
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:

-Entities-

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

-Relationships-

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:

-Entities-
{entity_df}

-Relationships-
{relation_df}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format(in language of 'Text' content):
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Output:"""


@dataclass
class GraphChange:
    removed_nodes: Set[str] = field(default_factory=set)
    added_updated_nodes: Set[str] = field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = field(default_factory=set)
    added_updated_edges: Set[Tuple[str, str]] = field(default_factory=set)
    merge_details: List[Dict] = field(default_factory=list)


@dataclass
class EntityResolutionResult:
    graph: nx.Graph
    change: GraphChange
    resolution_pairs: Set[Tuple[str, str]]

ENTITY_RESOLUTION_PROMPT = """
-Goal-
Please answer the following Question as required

-Steps-
1. Identify each line of questioning as required

2. Return output in English as a single list of each line answer in steps 1. Use **##** as the list delimiter.

######################
-Examples-
######################
Example 1:

Question:
When determining whether two Products are the same, you should only focus on critical properties and overlook noisy factors. 

Demonstration 1: name of Product A is : "computer", name of Product B is :"phone"  No, Product A and Product B are different products.
Question 1: name of Product A is : "television", name of Product B is :"TV"  
Question 2: name of Product A is : "cup", name of Product B is :"mug"  
Question 3: name of Product A is : "soccer", name of Product B is :"football"  
Question 4: name of Product A is : "pen", name of Product B is  :"eraser"  

Use domain knowledge of Products to help understand the text and answer the above 4 questions in the format: For Question i, Yes, Product A and Product B are the same product. or  No, Product A and Product B are different products. For Question i+1, (repeat the above procedures)
################
Output:
(For question <|>1<|>, &&no&&, Product A and Product B are different products.)##
(For question <|>2<|>, &&no&&, Product A and Product B are different products.)##
(For question <|>3<|>, &&yes&&, Product A and Product B are the same product.)##
(For question <|>4<|>, &&no&&, Product A and Product B are different products.)##
#############################

Example 2:

Question:
When determining whether two toponym are the same, you should only focus on critical properties and overlook noisy factors. 

Demonstration 1: name of toponym A is : "nanjing", name of toponym B is :"nanjing city"  No, toponym A and toponym B are same toponym.
Question 1: name of toponym A is : "Chicago", name of toponym B is :"ChiTown"  
Question 2: name of toponym A is : "Shanghai", name of toponym B is :"Zhengzhou"  
Question 3: name of toponym A is : "Beijing", name of toponym B is :"Peking"
Question 4: name of toponym A is : "Los Angeles", name of toponym B is :"Cleveland" 

Use domain knowledge of toponym to help understand the text and answer the above 4 questions in the format: For Question i, Yes, toponym A and toponym B are the same toponym. or  No, toponym A and toponym B are different toponym. For Question i+1, (repeat the above procedures)
################
Output:
(For question <|>1<|>, &&yes&&, toponym A and toponym B are same toponym.)##
(For question <|>2<|>, &&no&&, toponym A and toponym B are different toponym.)##
(For question <|>3<|>, &&yes&&, toponym A and toponym B are the same toponym.)##
(For question <|>4<|>, &&no&&, toponym A and toponym B are different toponym.)##
#############################

-Real Data-
######################
Question:{input_text}
######################
Output:
"""

SUMMARIZE_DESCRIPTIONS_PROMPT = """---Goal---
Summarize the descriptions of an entity or relationship into a concise and informative summary.

---Entity/Relationship Name---
{entity_name}

---Descriptions---
{description_list}

---Language---
{language}

---Output---
Provide a concise summary that captures the key information from all descriptions."""

ENTITY_EXTRACTION_PROMPT = """---Goal---
Given a text document and a list of entity types, identify all entities of those types 
from the text and all relationships among the identified entities.

---Entity Types---
organization, person, geo, event, category, fault, cause, action, component

---Steps---
1. Identify all entities. Format each entity as:
("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all relationships. Format each relationship as:
("relationship"<|><source_entity_name><|><target_entity_name><|><relationship_description><|><keywords><|><weight>)

3. Use ## as the list delimiter.
4. For relationships, use the exact entity names identified in step 1, NOT entity IDs.

---Text---
{input_text}

---Output---
"""


def clean_str(input: Any) -> str:
    if not isinstance(input, str):
        return input
    import html
    result = html.unescape(input.strip())
    return re.sub(r"[\"\\x00-\\x1f\\x7f-\\x9f]", "", result)


def handle_single_entity_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1]).upper()
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2]).upper()
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1]).upper()
    target = clean_str(record_attributes[2]).upper()
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = float(record_attributes[-1]) if re.match(r"^[-+]?[0-9]*\.?[0-9]+$", record_attributes[-1]) else 1.0
    pair = sorted([source, target])
    return dict(
        src_id=pair[0],
        tgt_id=pair[1],
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def flat_uniq_list(arr: list[dict], key: str) -> list:
    res = []
    for a in arr:
        val = a[key]
        if isinstance(val, list):
            res.extend(val)
        else:
            res.append(val)
    return list(set(res))


def get_from_to(node1: str, node2: str) -> tuple[str, str]:
    if node1 < node2:
        return (node1, node2)
    else:
        return (node2, node1)


def doc_id_to_str(doc_id_field, default_doc_id: str = "") -> str:
    """将 doc_id 字段序列化为 HugeGraph 可接受的字符串。

    HugeGraph schema 中 doc_id 为 TEXT SINGLE（单值字符串），而内部流程
    （提取/合并/跨文档消歧）为方便聚合将其保留为 list。本函数在写入
    HugeGraph 的边界处把 list 用逗号拼接成字符串，读回时由各处已有的
    `split(",")` 逻辑还原。空值回落到 default_doc_id。
    """
    if isinstance(doc_id_field, list):
        joined = ",".join(str(d) for d in doc_id_field if d)
        return joined or default_doc_id
    if doc_id_field:
        return str(doc_id_field)
    return default_doc_id


class GraphExtractor:
    GRAPH_FIELD_SEP = GRAPH_FIELD_SEP
    
    def __init__(self):
        self.entity_types = DEFAULT_ENTITY_TYPES

    def _has_digit_in_2gram_diff(self, a: str, b: str) -> bool:
        def to_2gram_set(s: str) -> set[str]:
            return {s[i:i+2] for i in range(len(s) - 1)}

        set_a = to_2gram_set(a)
        set_b = to_2gram_set(b)
        diff = set_a ^ set_b
        return any(any(c.isdigit() for c in pair) for pair in diff)

    def _is_similarity(self, a: str, b: str) -> bool:
        if self._has_digit_in_2gram_diff(a, b):
            return False

        a_en = a.replace(" ", "").isalpha() and not any(ch.isalpha() and ch.islower() for ch in a)
        b_en = b.replace(" ", "").isalpha() and not any(ch.isalpha() and ch.islower() for ch in b)
        
        if a_en and b_en:
            if editdistance.eval(a, b) <= min(len(a), len(b)) // 2:
                return True
            return False

        a_chars = set(a)
        b_chars = set(b)
        max_l = max(len(a_chars), len(b_chars))
        if max_l < 4:
            return len(a_chars & b_chars) > 1
        return len(a_chars & b_chars) * 1.0 / max_l >= 0.8

    def _call_llm_for_resolution(self, prompt: str) -> str:
        try:
            return llm_service.generate_text(prompt)
        except Exception as e:
            logger.error(f"LLM resolution call failed: {e}")
            return ""

    def _process_resolution_results(self, records_length: int, results: str) -> list[tuple[int, str]]:
        ans_list = []
        records = [r.strip() for r in results.split("##")]
        for record in records:
            match_int = re.search(r"<\|>(\d+)<\|>", record)
            res_int = int(str(match_int.group(1) if match_int else '0'))
            if res_int > records_length:
                continue

            match_bool = re.search(r"&&([a-zA-Z]+)&&", record)
            res_bool = str(match_bool.group(1) if match_bool else '')

            if res_int and res_bool:
                if res_bool.lower() == 'yes':
                    ans_list.append((res_int, "yes"))

        return ans_list

    def _resolve_entities(self, graph: nx.Graph, subgraph_nodes: Optional[Set[str]] = None) -> EntityResolutionResult:
        nodes = sorted(graph.nodes())
        if len(nodes) < 2:
            return EntityResolutionResult(
                graph=graph,
                change=GraphChange(),
                resolution_pairs=set()
            )

        entity_types = sorted(set(graph.nodes[node].get('entity_type', 'category').upper() for node in nodes))
        node_clusters = {entity_type: [] for entity_type in entity_types}

        for node in nodes:
            node_clusters[graph.nodes[node].get('entity_type', 'category').upper()].append(node)

        candidate_resolution = {entity_type: [] for entity_type in entity_types}
        for k, v in node_clusters.items():
            if subgraph_nodes:
                candidate_resolution[k] = [
                    (a, b) for a, b in itertools.combinations(v, 2) 
                    if self._is_similarity(a, b) and (a in subgraph_nodes or b in subgraph_nodes)
                ]
            else:
                candidate_resolution[k] = [
                    (a, b) for a, b in itertools.combinations(v, 2) 
                    if self._is_similarity(a, b)
                ]

        num_candidates = sum([len(candidates) for _, candidates in candidate_resolution.items()])
        logger.info(f"Identified {num_candidates} candidate pairs for resolution")

        resolution_result = set()

        for entity_type, candidates in candidate_resolution.items():
            if not candidates:
                continue

            pair_txt = [f'When determining whether two {entity_type}s are the same, you should only focus on critical properties and overlook noisy factors.\n']
            for index, candidate in enumerate(candidates):
                pair_txt.append(f'Question {index + 1}: name of{entity_type} A is {candidate[0]} ,name of{entity_type} B is {candidate[1]}')

            sent = 'question above' if len(pair_txt) == 1 else f'above {len(pair_txt)} questions'
            pair_txt.append(f'\nUse domain knowledge of {entity_type}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {entity_type} A and {entity_type} B are the same {entity_type}./No, {entity_type} A and {entity_type} B are different {entity_type}s. For Question i+1, (repeat the above procedures)')

            pair_prompt = '\n'.join(pair_txt)
            variables = {"input_text": pair_prompt}
            text = ENTITY_RESOLUTION_PROMPT.format(**variables)

            logger.info(f"Created resolution prompt {len(text)} bytes for {len(candidates)} entity pairs")

            response = self._call_llm_for_resolution(text)
            logger.debug(f"Resolution response: {response[:500]}")

            results = self._process_resolution_results(len(candidates), response)
            for result_i in results:
                resolution_result.add(candidates[result_i[0] - 1])

        logger.info(f"Resolved {len(resolution_result)} pairs to merge")
        
        change = GraphChange()
        
        if resolution_result:
            connect_graph = nx.Graph()
            connect_graph.add_edges_from(resolution_result)
            
            for sub_connect_graph in nx.connected_components(connect_graph):
                merging_nodes = list(sub_connect_graph)
                if len(merging_nodes) > 1:
                    base_node = merging_nodes[0]
                    merged_nodes = merging_nodes[1:]
                    
                    merge_detail = {
                        "base_entity": {
                            "name": base_node,
                            "entity_type": graph.nodes[base_node].get('entity_type', 'category'),
                            "description": graph.nodes[base_node].get('description', '')[:100] + '...' if len(graph.nodes[base_node].get('description', '')) > 100 else graph.nodes[base_node].get('description', '')
                        },
                        "merged_entities": []
                    }
                    
                    for merged_node in merged_nodes:
                        merge_detail["merged_entities"].append({
                            "name": merged_node,
                            "entity_type": graph.nodes[merged_node].get('entity_type', 'category'),
                            "description": graph.nodes[merged_node].get('description', '')[:100] + '...' if len(graph.nodes[merged_node].get('description', '')) > 100 else graph.nodes[merged_node].get('description', '')
                        })
                    
                    change.merge_details.append(merge_detail)
                    change.removed_nodes.update(merged_nodes)
                    change.added_updated_nodes.add(base_node)

        return EntityResolutionResult(
            graph=graph,
            change=change,
            resolution_pairs=resolution_result
        )

    def _rederive_cached_fields(self, name: str, attrs: dict, model_config: Optional[Dict] = None) -> None:
        """根据 *_parts 重算派生缓存字段 description / keywords / weight。

        parts 是事实来源，派生字段供检索/展示直接使用。删除文档裁剪 parts 后调用本方法刷新缓存。
        """
        parts = attrs.get("description_parts", [])
        desc = GRAPH_FIELD_SEP.join(p.get("text", "") for p in parts)
        if len(parts) > 12:
            desc = self._handle_entity_relation_summary(name, desc, model_config)
        attrs["description"] = desc

        kparts = attrs.get("keywords_parts", [])
        attrs["keywords"] = sorted({k for p in kparts for k in p.get("keywords", [])})

        if "weight_parts" in attrs:
            wparts = attrs.get("weight_parts", [])
            attrs["weight"] = sum(p.get("weight", 0) for p in wparts)

    def _merge_graph_nodes(self, graph: nx.Graph, nodes: list[str], model_config: Optional[Dict] = None, change: Optional[GraphChange] = None) -> None:
        if len(nodes) <= 1:
            return

        nodes_set = set(nodes)
        base_node = nodes[0]
        node0_attrs = graph.nodes[base_node]
        node0_neighbors = set(graph.neighbors(base_node))

        for merged_node in nodes[1:]:
            node1_attrs = graph.nodes[merged_node]
            # 描述改为合并 parts（保留每段来源 doc_id），不再破坏性拼接 description
            node0_attrs["source_id"] = sorted(set(node0_attrs.get("source_id", []) + node1_attrs.get("source_id", [])))
            node0_attrs["description_parts"] = node0_attrs.get("description_parts", []) + node1_attrs.get("description_parts", [])
            node0_attrs["keywords_parts"] = node0_attrs.get("keywords_parts", []) + node1_attrs.get("keywords_parts", [])

            existing_doc_ids = node0_attrs.get("doc_id", [])
            if isinstance(existing_doc_ids, str):
                existing_doc_ids = [existing_doc_ids]
            other_doc_ids = node1_attrs.get("doc_id", [])
            if isinstance(other_doc_ids, str):
                other_doc_ids = [other_doc_ids]
            node0_attrs["doc_id"] = list(set(existing_doc_ids + other_doc_ids))

            for neighbor in graph.neighbors(merged_node):
                if neighbor not in nodes_set:
                    edge1_attrs = graph.get_edge_data(merged_node, neighbor)
                    if neighbor in node0_neighbors:
                        edge0_attrs = graph.get_edge_data(base_node, neighbor)
                        # 边同样合并 parts
                        edge0_attrs["description_parts"] = edge0_attrs.get("description_parts", []) + edge1_attrs.get("description_parts", [])
                        edge0_attrs["keywords_parts"] = edge0_attrs.get("keywords_parts", []) + edge1_attrs.get("keywords_parts", [])
                        edge0_attrs["weight_parts"] = edge0_attrs.get("weight_parts", []) + edge1_attrs.get("weight_parts", [])
                        for attr in ["keywords", "source_id"]:
                            edge0_attrs[attr] = sorted(set(edge0_attrs.get(attr, []) + edge1_attrs.get(attr, [])))
                        self._rederive_cached_fields(f"{base_node} -> {neighbor}", edge0_attrs, model_config)
                        graph.add_edge(base_node, neighbor, **edge0_attrs)
                        if change:
                            change.added_updated_edges.add(get_from_to(base_node, neighbor))
                    else:
                        graph.add_edge(base_node, neighbor, **edge1_attrs)
                        if change:
                            change.added_updated_edges.add(get_from_to(base_node, neighbor))

            graph.remove_node(merged_node)
            if change:
                change.removed_nodes.add(merged_node)

        # 合并完成后根据 parts 重算 base_node 的派生字段
        self._rederive_cached_fields(base_node, node0_attrs, model_config)
        graph.nodes[base_node].update(node0_attrs)
        if change:
            change.added_updated_nodes.add(base_node)

    def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str, model_config: Optional[Dict] = None) -> str:
        summary_max_tokens = 512
        use_description = description[:summary_max_tokens]
        description_list = use_description.split(GRAPH_FIELD_SEP)

        if len(description_list) <= 12:
            return use_description

        try:
            prompt_template = SUMMARIZE_DESCRIPTIONS_PROMPT
            context_base = dict(
                entity_name=entity_or_relation_name,
                description_list=description_list,
                language="Chinese"
            )
            use_prompt = prompt_template.format(**context_base)
            logger.info(f"Trigger summary for: {entity_or_relation_name}")

            response = llm_service.generate_text(use_prompt, system_prompt="你是一个文本摘要专家。", model_config=model_config)
            return response.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return use_description

    def extract_entities_and_relations(self, doc_id: str, chunks: List[str], model_config: Optional[Dict] = None, mode: str = "enhanced", with_community: bool = True, entity_resolution: bool = True) -> Tuple[Dict, Dict, Dict]:
        entities = {}
        relations = {}
        community_reports = {}
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_key = f"{doc_id}_{chunk_idx}"
            if mode == "classic":
                llm_result = llm_service.extract_entities_and_relations_classic(chunk, model_config)
            else:
                llm_result = llm_service.extract_entities_and_relations(chunk, model_config)
            
            if isinstance(llm_result, dict) and "entities" in llm_result:
                for entity in llm_result.get("entities", []):
                    entity_name = entity.get("name", "").upper()
                    entity_type = entity.get("type", "category").upper()
                    entity_desc = entity.get("description", "")
                    
                    if entity_name and entity_type.lower() in [t.lower() for t in self.entity_types]:
                        maybe_nodes[entity_name].append({
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "description": entity_desc,
                            "source_id": chunk_key
                        })

                for relation in llm_result.get("relations", []):
                    source = relation.get("source", "").upper()
                    target = relation.get("target", "").upper()
                    rel_desc = relation.get("description", "")
                    rel_label = relation.get("label", "related_to")
                    keywords = relation.get("keywords", "")
                    weight = relation.get("weight", 1.0)

                    if source and target:
                        pair = sorted([source, target])
                        maybe_edges[(pair[0], pair[1])].append({
                            "src_id": pair[0],
                            "tgt_id": pair[1],
                            "weight": float(weight) if isinstance(weight, (int, float)) else 1.0,
                            "description": rel_desc,
                            "keywords": keywords,
                            "source_id": chunk_key,
                            "label": rel_label
                        })
            
            elif isinstance(llm_result, str):
                records = split_string_by_multi_markers(llm_result, ["##"])
                for record in records:
                    record_attributes = split_string_by_multi_markers(record, ["<|>"])
                    
                    if_entities = handle_single_entity_extraction(record_attributes, chunk_key)
                    if if_entities is not None and if_entities.get("entity_type", "unknown").lower() in [t.lower() for t in self.entity_types]:
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        continue

                    if_relation = handle_single_relationship_extraction(record_attributes, chunk_key)
                    if if_relation is not None:
                        maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)

        logger.info(f"Raw extraction done: {len(maybe_nodes)} nodes, {len(maybe_edges)} edges")

        for en_nm, ents in maybe_nodes.items():
            entity_type = sorted(
                Counter([dp["entity_type"] for dp in ents]).items(),
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in ents])))
            already_source_ids = flat_uniq_list(ents, "source_id")
            description = self._handle_entity_relation_summary(en_nm, description, model_config)

            entities[en_nm] = {
                "entity_type": entity_type,
                "description": description,
                "source_id": already_source_ids,
                "doc_id": [doc_id],
                # 可追溯结构：记录该文档对此实体描述的贡献，删除文档时精确裁剪
                "description_parts": [{"doc_id": doc_id, "text": description}],
                "keywords_parts": [{"doc_id": doc_id, "keywords": []}],
            }

        for (src, tgt), rels in maybe_edges.items():
            if src not in entities or tgt not in entities:
                continue
            
            weight = sum([edge["weight"] for edge in rels])
            description = GRAPH_FIELD_SEP.join(sorted(set([edge["description"] for edge in rels])))
            description = self._handle_entity_relation_summary(f"{src} -> {tgt}", description, model_config)
            
            keywords_list = []
            for edge in rels:
                kw = edge.get("keywords", "")
                if kw:
                    if isinstance(kw, list):
                        keywords_list.extend(kw)
                    else:
                        keywords_list.append(kw)
            keywords = list(set(keywords_list))
            source_id = flat_uniq_list(rels, "source_id")
            label = rels[0].get("label", "related_to")

            key = (src, tgt)
            if key not in relations:
                relations[key] = {
                    "description": description,
                    "label": label,
                    "keywords": keywords,
                    "weight": weight,
                    "source_id": source_id,
                    "description_parts": [{"doc_id": doc_id, "text": description}],
                    "keywords_parts": [{"doc_id": doc_id, "keywords": list(keywords)}],
                    "weight_parts": [{"doc_id": doc_id, "weight": weight}],
                }
            else:
                relations[key]["description"] += GRAPH_FIELD_SEP + description
                relations[key]["weight"] += weight
                relations[key]["keywords"] = list(set(relations[key]["keywords"] + keywords))
                relations[key]["source_id"] = list(set(relations[key]["source_id"] + source_id))
                relations[key].setdefault("description_parts", []).append({"doc_id": doc_id, "text": description})
                relations[key].setdefault("keywords_parts", []).append({"doc_id": doc_id, "keywords": list(keywords)})
                relations[key].setdefault("weight_parts", []).append({"doc_id": doc_id, "weight": weight})

        logger.info(f"After initial merging: {len(entities)} entities, {len(relations)} relations")

        graph = nx.Graph()
        for en_nm, en_info in entities.items():
            graph.add_node(en_nm, **en_info)
        
        for (src, tgt), rel_info in relations.items():
            graph.add_edge(src, tgt, **rel_info)

        if entity_resolution:
            resolution_result = self._resolve_entities(graph)
            if resolution_result.resolution_pairs:
                connect_graph = nx.Graph()
                connect_graph.add_edges_from(resolution_result.resolution_pairs)
                
                for sub_connect_graph in nx.connected_components(connect_graph):
                    merging_nodes = list(sub_connect_graph)
                    self._merge_graph_nodes(graph, merging_nodes, model_config, resolution_result.change)

                pr = nx.pagerank(graph)
                for node_name, pagerank in pr.items():
                    graph.nodes[node_name]["pagerank"] = pagerank

                entities = {}
                for node_name in graph.nodes():
                    entities[node_name] = dict(graph.nodes[node_name])
                
                relations = {}
                for src, tgt in graph.edges():
                    edge_data = dict(graph.get_edge_data(src, tgt))
                    key = get_from_to(src, tgt)
                    relations[key] = edge_data

                logger.info(f"After entity resolution: {len(entities)} entities, {len(relations)} relations")

        if with_community and len(entities) >= 3:
            community_reports = self._detect_communities_and_generate_reports(graph, entities, relations)
            logger.info(f"Community reports generated: {len(community_reports)}")

        if not entities:
            entities["DEFAULT_ENTITY"] = {
                "entity_type": "category",
                "description": "默认实体",
                "source_id": [doc_id]
            }

        return entities, relations, community_reports

    def _detect_communities_and_generate_reports(self, graph: nx.Graph, entities: Dict, relations: Dict, log_callback=None) -> Dict:
        community_reports = {}
        
        def log(msg, log_type="community"):
            print(f"[COMMUNITY DEBUG] {msg}")
            if log_callback:
                try:
                    log_callback(msg, log_type)
                except Exception as e:
                    print(f"[COMMUNITY DEBUG] Failed to call log callback: {e}")
        
        try:
            log("═" * 50)
            log("【阶段1】初始化社区检测")
            log("═" * 50)
            
            total_nodes = graph.number_of_nodes()
            total_edges = graph.number_of_edges()
            num_connected_components = nx.number_connected_components(graph)
            graph_density = nx.density(graph)
            
            log(f"图基本统计:")
            log(f"  - 总节点数: {total_nodes}")
            log(f"  - 总边数: {total_edges}")
            log(f"  - 连通分量数: {num_connected_components}")
            log(f"  - 图密度: {graph_density:.4f}")
            log(f"  - 实体字典大小: {len(entities)}")
            log(f"  - 关系字典大小: {len(relations)}")
            
            if total_nodes < 3:
                log(f"【跳过】节点数({total_nodes}) < 最小阈值(3)，无法进行社区检测")
                logger.info("Graph too small for community detection")
                return community_reports
            
            adaptive_max_cluster_size = _compute_adaptive_max_cluster_size(graph)
            log("【阶段2】执行Leiden社区检测算法")
            log(f"算法参数: max_cluster_size={adaptive_max_cluster_size} (自适应, 总节点={total_nodes})")
            log(f"算法参数: use_lcc=False (保留所有连通分量)")
            
            communities_result = run_leiden(graph, {
                "max_cluster_size": adaptive_max_cluster_size,
                "use_lcc": False
            })
            
            if not communities_result:
                log("【跳过】Leiden算法未检测到任何社区")
                return community_reports
            
            log(f"检测结果: 共发现 {len(communities_result)} 个层级")
            
            for level, level_communities in communities_result.items():
                log(f"  - 层级 {level}: {len(level_communities)} 个社区")
            
            all_communities = []
            for level, level_communities in communities_result.items():
                for comm_id, comm_data in level_communities.items():
                    all_communities.append({
                        "level": level,
                        "community_id": comm_id,
                        "nodes": comm_data["nodes"],
                        "weight": comm_data["weight"]
                    })
            
            all_communities.sort(key=lambda x: x["weight"], reverse=True)
            
            valid_communities = [comm for comm in all_communities if len(comm["nodes"]) >= 2]
            invalid_communities = [comm for comm in all_communities if len(comm["nodes"]) < 2]
            
            log(f"社区过滤:")
            log(f"  - 有效社区(>=2个节点): {len(valid_communities)} 个")
            log(f"  - 无效社区(<2个节点): {len(invalid_communities)} 个")
            
            if not valid_communities:
                log("【跳过】没有找到有效社区(>=2个节点)，无法生成社区报告")
                return community_reports
            
            log("【阶段3】逐个处理有效社区")
            log(f"开始为 {len(valid_communities)} 个社区生成报告...")
            
            for idx, comm_info in enumerate(valid_communities, 1):
                community_id = comm_info["community_id"]
                community_nodes = comm_info["nodes"]
                community_weight = comm_info["weight"]
                level = comm_info["level"]
                
                log("─" * 50)
                log(f"处理社区 [{idx}/{len(valid_communities)}] ID={community_id} (层级={level}, 权重={community_weight:.4f})")
                log("─" * 50)
                log(f"社区节点列表: {community_nodes}")
                
                subgraph = graph.subgraph(community_nodes)
                
                entity_records = []
                for idx2, node in enumerate(community_nodes):
                    entity_info = entities.get(node, {})
                    entity_records.append({
                        "id": idx2,
                        "entity": node,
                        "description": entity_info.get("description", "")[:200]
                    })
                
                relation_records = []
                rel_idx = 0
                max_relations = 10000
                for src, tgt in subgraph.edges():
                    if rel_idx >= max_relations:
                        log(f"  警告: 关系数量超过 {max_relations} 限制，已截断")
                        break
                    rel_info = relations.get(get_from_to(src, tgt), {})
                    relation_records.append({
                        "id": rel_idx,
                        "source": src,
                        "target": tgt,
                        "description": rel_info.get("description", "")[:200]
                    })
                    rel_idx += 1
                
                log(f"社区内容:")
                log(f"  - 实体数量: {len(entity_records)}")
                log(f"  - 关系数量: {len(relation_records)}")
                log(f"  - 社区权重: {community_weight:.4f}")
                
                for er in entity_records:
                    desc = er['description'] if er['description'] else '(无描述)'
                    log(f"    * [{er['id']}] {er['entity']}: {desc[:100]}")
                
                for rr in relation_records[:10]:
                    desc = rr['description'] if rr['description'] else '(无描述)'
                    log(f"    → [{rr['id']}] {rr['source']} -- {desc[:50]} -- {rr['target']}")
                if len(relation_records) > 10:
                    log(f"    ... (还有 {len(relation_records) - 10} 条关系)")
                
                entity_df = pd.DataFrame(entity_records)
                relation_df = pd.DataFrame(relation_records)
                
                entity_df_str = entity_df.to_csv(index=False)
                relation_df_str = relation_df.to_csv(index=False)
                
                prompt = COMMUNITY_REPORT_PROMPT.format(
                    entity_df=entity_df_str,
                    relation_df=relation_df_str
                )
                
                log("【阶段4】调用LLM生成社区报告")
                log(f"Prompt长度: {len(prompt)} 字符")
                log(f"Prompt预览: {prompt[:200]}...")
                
                try:
                    log(f"正在调用LLM服务...")
                    report_result = llm_service.generate_community_report(prompt)
                    log(f"LLM响应完成")
                    log(f"响应长度: {len(report_result)} 字符")
                    log(f"响应预览: {report_result[:300]}...")
                    
                    try:
                        response = report_result.strip()
                        response = re.sub(r'^[^{]*', '', response)
                        response = re.sub(r'[^}]*$', '', response)
                        response = re.sub(r'{{', '{', response)
                        response = re.sub(r'}}', '}', response)
                        
                        report_data = json.loads(response)
                        log(f"JSON解析成功")
                        
                        required_fields = [
                            ("title", str),
                            ("summary", str),
                            ("rating", (int, float)),
                            ("rating_explanation", str),
                            ("findings", list)
                        ]
                        
                        if not dict_has_keys_with_types(report_data, required_fields):
                            log(f"【错误】报告数据缺少必要字段或类型错误", log_type="error")
                            log(f"报告数据: {json.dumps(report_data, ensure_ascii=False)[:500]}", log_type="error")
                            continue
                        
                        for finding in report_data.get("findings", []):
                            if not dict_has_keys_with_types(finding, [("summary", str), ("explanation", str)]):
                                log(f"【错误】发现项缺少必要字段", log_type="error")
                                continue
                        
                        report_title = report_data.get('title', '未命名社区')
                        report_summary = report_data.get('summary', '')[:100]
                        report_rating = report_data.get('rating', 0)
                        report_rating_explanation = report_data.get('rating_explanation', '')
                        report_findings = report_data.get('findings', [])
                        
                        log(f"报告内容:")
                        log(f"  - 标题: {report_title}")
                        log(f"  - 摘要: {report_summary}...")
                        log(f"  - 影响评级: {report_rating}")
                        log(f"  - 评级解释: {report_rating_explanation[:50]}...")
                        log(f"  - 发现数量: {len(report_findings)}")
                        log(f"  - 社区权重: {community_weight:.4f}")
                        
                        report_id = f"community_{level}_{community_id}"
                        
                        community_doc_ids = set()
                        for node in community_nodes:
                            entity_info = entities.get(node, {})
                            node_doc_ids = entity_info.get("doc_id", [])
                            if isinstance(node_doc_ids, str):
                                node_doc_ids = [node_doc_ids]
                            if node_doc_ids:
                                community_doc_ids.update(node_doc_ids)
                        
                        community_reports[report_id] = {
                            "community_id": community_id,
                            "level": level,
                            "entities": community_nodes,
                            "weight": community_weight,
                            "report": report_data,
                            "doc_ids": list(community_doc_ids),
                            # 记录每个实体当时的来源文档集合，删除文档时据此判断
                            # "实体是否被彻底删除"以分级处理社区报告
                            "entity_doc_map": {
                                node: (lambda dl: dl if isinstance(dl, list) else ([dl] if dl else []))(
                                    entities.get(node, {}).get("doc_id", [])
                                )
                                for node in community_nodes
                            },
                        }
                        
                        add_community_info2graph(graph, community_nodes, report_title)
                        
                        log(f"【成功】社区报告已生成: {report_id}")
                        
                    except json.JSONDecodeError as je:
                        log(f"【错误】JSON解析失败: {je}", log_type="error")
                        log(f"清理后响应: {response[:500]}", log_type="error")
                        logger.error(f"Failed to parse LLM response for community {community_id}: {je}")
                        continue
                        
                except Exception as e:
                    log(f"【错误】调用LLM失败 (社区 {community_id}): {str(e)}", log_type="error")
                    logger.error(f"Failed to generate community report for {community_id}: {e}")
                    continue

        except Exception as e:
            log(f"【错误】社区检测整体失败: {str(e)}", log_type="error")
            log(f"错误类型: {type(e).__name__}", log_type="error")
            import traceback
            log(f"堆栈信息: {traceback.format_exc()}", log_type="error")
            logger.error(f"Community detection failed: {e}")

        log("═" * 50)
        log(f"【阶段5】社区报告生成完成")
        log(f"  - 总社区数: {len(valid_communities) if 'valid_communities' in locals() else 0}")
        log(f"  - 成功生成报告: {len(community_reports)}")
        log(f"  - 失败/跳过: {(len(valid_communities) if 'valid_communities' in locals() else 0) - len(community_reports)}")
        log("═" * 50)
        
        return community_reports

    def convert_to_graph_format(self, entities: Dict, relations: Dict, doc_id: str, kb_id: str) -> Tuple[List[Dict], List[Dict]]:
        vertices = []
        edges = []
        created_at = datetime.datetime.now().isoformat()

        # 从边权重累加计算每个实体的 weight（实体权重 = 所有关联边的权重之和）
        entity_weights = {}
        for (src, tgt), rel_info in relations.items():
            w = rel_info.get("weight", 1)
            entity_weights[src] = entity_weights.get(src, 0) + w
            entity_weights[tgt] = entity_weights.get(tgt, 0) + w

        entity_id_map = {}
        for idx, (entity_name, entity_info) in enumerate(entities.items()):
            vertex_id = f"{kb_id}_{doc_id}_{idx}"
            entity_id_map[entity_name] = vertex_id

            vertex = {
                "id": vertex_id,
                "label": "Entity",
                "properties": {
                    "name": entity_name,
                    "entity_type": entity_info["entity_type"],
                    "description": entity_info["description"],
                    "doc_id": doc_id_to_str(entity_info.get("doc_id", [doc_id]), doc_id),
                    "kb_id": kb_id,
                    "project_id": "",
                    "tenant_id": "",
                    "source_id": entity_info["source_id"],
                    "created_at": created_at,
                    "weight": int(entity_info.get("weight", entity_weights.get(entity_name, 0))),
                    "description_parts": json.dumps(entity_info.get("description_parts", []), ensure_ascii=False),
                    "keywords_parts": json.dumps(entity_info.get("keywords_parts", []), ensure_ascii=False),
                }
            }
            if "pagerank" in entity_info:
                vertex["properties"]["pagerank"] = entity_info["pagerank"]
            vertices.append(vertex)

        edge_idx = 0
        
        for (src, tgt), rel_info in relations.items():
            if src in entity_id_map and tgt in entity_id_map:
                edge_id = f"{kb_id}_{doc_id}_edge_{edge_idx}"
                edge_idx += 1

                original_rel_label = rel_info.get("label", "related_to")

                edge = {
                    "id": edge_id,
                    "label": "relation",
                    "outV": entity_id_map[src],
                    "outVLabel": "Entity",
                    "inV": entity_id_map[tgt],
                    "inVLabel": "Entity",
                    "properties": {
                        "description": rel_info.get("description", ""),
                        "keywords": rel_info.get("keywords", []),
                        "weight": int(rel_info.get("weight", 1)),
                        "doc_id": doc_id_to_str([doc_id], doc_id),
                        "kb_id": kb_id,
                        "project_id": "",
                        "tenant_id": "",
                        "source_id": rel_info.get("source_id", []),
                        "created_at": created_at,
                        "relation_type": original_rel_label,
                        "description_parts": json.dumps(rel_info.get("description_parts", []), ensure_ascii=False),
                        "keywords_parts": json.dumps(rel_info.get("keywords_parts", []), ensure_ascii=False),
                        "weight_parts": json.dumps(rel_info.get("weight_parts", []), ensure_ascii=False),
                    }
                }
                edges.append(edge)

        logger.info(f"Converted to graph format: {len(vertices)} vertices, {len(edges)} edges")
        return vertices, edges
