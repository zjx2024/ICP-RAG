"""FTA / RBD 确定性计算。

LLM 只负责把图像转为节点/边；最小割集、顶事件概率和 RBD 可靠度由本模块计算，
避免让语言模型直接猜公式或数值。计算假设：基本事件/功能块相互独立。
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


class ReliabilityModelError(ValueError):
    pass


def _props(item: Dict[str, Any]) -> Dict[str, Any]:
    value = item.get("properties") or {}
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(str(value).replace("%", "").strip()) / (100.0 if "%" in str(value) else 1.0)
    except Exception:
        return None


def _name(item: Dict[str, Any]) -> str:
    props = _props(item)
    return str(props.get("name") or item.get("name") or item.get("id") or "")


def _node_probability(node: Dict[str, Any], mission_time: Optional[float]) -> Optional[float]:
    props = _props(node)
    for key in ("probability", "failure_probability", "unreliability", "q"):
        value = _as_float(props.get(key))
        if value is not None:
            return min(max(value, 0.0), 1.0)
    failure_rate = None
    for key in ("failure_rate", "lambda", "λ"):
        failure_rate = _as_float(props.get(key))
        if failure_rate is not None:
            break
    time_value = mission_time
    if time_value is None:
        time_value = _as_float(props.get("mission_time"))
    if failure_rate is not None and time_value is not None:
        return 1.0 - math.exp(-failure_rate * time_value)
    return None


def _block_reliability(node: Dict[str, Any], mission_time: Optional[float]) -> Optional[float]:
    props = _props(node)
    for key in ("reliability", "success_probability", "r"):
        value = _as_float(props.get(key))
        if value is not None:
            return min(max(value, 0.0), 1.0)
    failure_probability = _node_probability(node, mission_time)
    if failure_probability is not None:
        return 1.0 - failure_probability
    return None


def _graph(graph: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    vertices = {
        str(item.get("id")): item
        for item in graph.get("vertices", []) or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    children: Dict[str, List[Dict[str, Any]]] = {node_id: [] for node_id in vertices}
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in vertices and target in vertices:
            children.setdefault(source, []).append(edge)
    return vertices, children


def _gate_type(node: Dict[str, Any], outgoing_edges: Sequence[Dict[str, Any]]) -> str:
    props = _props(node)
    candidates: List[Any] = [props.get("gate_type"), props.get("type"), props.get("name")]
    candidates.extend(_props(edge).get("gate_type") for edge in outgoing_edges)
    for value in candidates:
        text = str(value or "").strip().upper()
        # XOR contains OR lexically but is not an inclusive OR gate.
        if "XOR" in text:
            return "XOR"
        if "AND" in text or "与" == text:
            return "AND"
        if "OR" in text or "或" == text:
            return "OR"
    return "OR"


def _fta_root(graph: Dict[str, Any], vertices: Dict[str, Dict[str, Any]], children: Dict[str, List[Dict[str, Any]]]) -> str:
    root = str(graph.get("root_fault_id") or graph.get("root_id") or "")
    if root in vertices:
        return root
    incoming = {str(edge.get("target")) for edge in graph.get("edges", []) or [] if isinstance(edge, dict)}
    candidates = [node_id for node_id in vertices if node_id not in incoming]
    top = [node_id for node_id in candidates if str(vertices[node_id].get("label", "")).lower() in {"topevent", "top_event"}]
    if top:
        return top[0]
    if candidates:
        return candidates[0]
    if vertices:
        return next(iter(vertices))
    raise ReliabilityModelError("FTA 图没有节点")


def calculate_fta(graph: Dict[str, Any], mission_time: Optional[float] = None) -> Dict[str, Any]:
    """计算 FTA 最小割集与顶事件概率。缺少基本事件概率时仍返回割集。"""
    vertices, children = _graph(graph)
    root = _fta_root(graph, vertices, children)
    # The current cut-set representation contains positive events only.
    # For XOR, A alone is not a sufficient cut when B also occurs: signed
    # literals are needed. Do not silently reuse the coherent OR algorithm or
    # choose a multi-input XOR convention on the user's behalf.
    pending = [root]
    checked: Set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in checked:
            continue
        checked.add(node_id)
        outgoing = children.get(node_id, [])
        if outgoing and _gate_type(vertices[node_id], outgoing) == "XOR":
            raise ReliabilityModelError(
                "当前 FTA 算法不支持 XOR 的非单调割集及多输入约定；"
                "可查询显式结构，但不能按 OR 替代计算"
            )
        pending.extend(str(edge["target"]) for edge in outgoing)
    visiting: Set[str] = set()
    probability_cache: Dict[str, Optional[float]] = {}
    cut_cache: Dict[str, List[frozenset[str]]] = {}
    missing: Set[str] = set()

    def reachable_nodes() -> Set[str]:
        reachable: Set[str] = set()
        pending = [root]
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(
                str(edge.get("target"))
                for edge in children.get(node_id, [])
                if str(edge.get("target") or "") in vertices
            )
        return reachable

    def dependency_conflicts() -> List[str]:
        """Detect shared events that make recursive subtree products invalid.

        The recursive probability formula is exact only when sibling subtrees do
        not share a basic/intermediate event.  Real FTA diagrams may reconverge on
        the same vertex, or repeat the same named basic event as separate visual
        nodes.  Until a bounded exact Boolean solver is available, those models
        must fail closed for numeric probability while retaining their topology
        and minimal cut sets.
        """
        reachable = reachable_nodes()
        incoming: Dict[str, int] = {node_id: 0 for node_id in reachable}
        leaves_by_name: Dict[str, List[str]] = {}
        for source in reachable:
            for edge in children.get(source, []):
                target = str(edge.get("target") or "")
                if target in reachable:
                    incoming[target] = incoming.get(target, 0) + 1
            if not child_ids(source):
                event_name = (_name(vertices[source]) or source).strip().casefold()
                leaves_by_name.setdefault(event_name, []).append(source)

        conflicts = [
            f"共享节点:{_name(vertices[node_id]) or node_id}"
            for node_id, count in incoming.items()
            if node_id != root and count > 1
        ]
        conflicts.extend(
            f"重复基本事件:{_name(vertices[node_ids[0]]) or name}"
            for name, node_ids in leaves_by_name.items()
            if name and len(node_ids) > 1
        )
        return list(dict.fromkeys(conflicts))

    def child_ids(node_id: str) -> List[str]:
        return [str(edge.get("target")) for edge in children.get(node_id, [])]

    def cuts(node_id: str) -> List[frozenset[str]]:
        if node_id in cut_cache:
            return cut_cache[node_id]
        if node_id in visiting:
            raise ReliabilityModelError("FTA 图存在环路")
        visiting.add(node_id)
        node = vertices[node_id]
        ids = child_ids(node_id)
        if not ids:
            result = [frozenset({_name(node) or node_id})]
        else:
            gate = _gate_type(node, children.get(node_id, []))
            child_cuts = [cuts(child) for child in ids]
            if gate == "AND":
                result = []
                for combination in itertools.product(*child_cuts):
                    merged: Set[str] = set()
                    for item in combination:
                        merged.update(item)
                    result.append(frozenset(merged))
            else:  # OR/XOR 的最小割集均为各分支割集，概率公式另行处理。
                result = [item for group in child_cuts for item in group]

        # 移除重复和非最小超集。
        unique = sorted(set(result), key=lambda value: (len(value), sorted(value)))
        minimal: List[frozenset[str]] = []
        for candidate in unique:
            if not any(existing.issubset(candidate) for existing in minimal):
                minimal.append(candidate)
        visiting.remove(node_id)
        cut_cache[node_id] = minimal
        return minimal

    def probability(node_id: str) -> Optional[float]:
        if node_id in probability_cache:
            return probability_cache[node_id]
        node = vertices[node_id]
        ids = child_ids(node_id)
        if not ids:
            value = _node_probability(node, mission_time)
            if value is None:
                missing.add(_name(node) or node_id)
            probability_cache[node_id] = value
            return value

        values = [probability(child) for child in ids]
        if any(value is None for value in values):
            probability_cache[node_id] = None
            return None
        numeric = [float(value) for value in values if value is not None]
        gate = _gate_type(node, children.get(node_id, []))
        if gate == "AND":
            result = math.prod(numeric)
        elif gate == "XOR":
            result = sum(
                value * math.prod(1.0 - other for index, other in enumerate(numeric) if index != current)
                for current, value in enumerate(numeric)
            )
        else:
            result = 1.0 - math.prod(1.0 - value for value in numeric)
        probability_cache[node_id] = min(max(result, 0.0), 1.0)
        return probability_cache[node_id]

    minimal_cut_sets = [sorted(item) for item in cuts(root)]
    shared_dependencies = dependency_conflicts()
    if shared_dependencies:
        # Do not publish a numerically plausible but mathematically invalid value.
        # Still inspect leaves so missing input parameters remain visible.
        for node_id in reachable_nodes():
            if child_ids(node_id):
                continue
            if _node_probability(vertices[node_id], mission_time) is None:
                missing.add(_name(vertices[node_id]) or node_id)
        top_probability = None
        probability_block_reason = (
            "FTA 存在共享/重复基本事件，递归子树独立公式不适用；"
            "在精确布尔概率求解前禁止给出顶事件概率"
        )
    else:
        top_probability = probability(root)
        probability_block_reason = ""
    return {
        "model": "FTA",
        "root_id": root,
        "top_event": _name(vertices[root]),
        "minimal_cut_sets": minimal_cut_sets,
        "minimal_cut_set_count": len(minimal_cut_sets),
        "top_event_probability": top_probability,
        "mission_time": mission_time,
        "missing_probability_nodes": sorted(missing),
        "shared_dependency_nodes": shared_dependencies,
        "probability_block_reason": probability_block_reason,
        "assumption": "基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算",
    }


def _rbd_terminals(vertices: Dict[str, Dict[str, Any]], graph: Dict[str, Any]) -> Tuple[str, str]:
    start = ""
    end = ""
    for node_id, node in vertices.items():
        label = str(node.get("label", "")).lower()
        node_name = _name(node).strip().lower()
        if label == "start" or node_name == "start" or node_name == "开始":
            start = node_id
        if label == "end" or node_name == "end" or node_name == "结束":
            end = node_id
    metadata = graph.get("metadata") or {}
    start = start or str(metadata.get("start_id") or "")
    end = end or str(metadata.get("end_id") or "")
    if not start or not end or start not in vertices or end not in vertices:
        raise ReliabilityModelError("RBD 缺少可识别的 START/END 节点")
    return start, end


def _simple_paths(
    children: Dict[str, List[Dict[str, Any]]],
    start: str,
    end: str,
    max_paths: int = 12,
) -> Tuple[List[List[str]], bool]:
    paths: List[List[str]] = []
    search_limit = max(max_paths, 1) + 1

    def walk(node_id: str, path: List[str], visited: Set[str]) -> None:
        if len(paths) >= search_limit:
            return
        if node_id == end:
            paths.append(list(path))
            return
        for edge in children.get(node_id, []):
            target = str(edge.get("target") or "")
            if not target or target in visited:
                continue
            walk(target, path + [target], visited | {target})

    walk(start, [start], {start})
    truncated = len(paths) > max_paths
    return paths[:max_paths], truncated


def calculate_rbd(graph: Dict[str, Any], mission_time: Optional[float] = None, max_paths: int = 12) -> Dict[str, Any]:
    """枚举成功路径并使用容斥原理计算 RBD 系统可靠度。"""
    vertices, children = _graph(graph)
    start, end = _rbd_terminals(vertices, graph)
    paths, path_limit_reached = _simple_paths(children, start, end, max_paths=max_paths)
    if not paths:
        raise ReliabilityModelError("START 到 END 之间不存在成功路径")

    block_reliability: Dict[str, float] = {}
    missing: List[str] = []
    path_blocks: List[Set[str]] = []
    for path in paths:
        blocks = {
            node_id for node_id in path
            if node_id not in (start, end)
            and str(vertices[node_id].get("label", "")).lower() not in {"start", "end"}
        }
        path_blocks.append(blocks)
        for node_id in blocks:
            if node_id in block_reliability:
                continue
            value = _block_reliability(vertices[node_id], mission_time)
            if value is None:
                missing.append(_name(vertices[node_id]) or node_id)
            else:
                block_reliability[node_id] = value

    system_reliability = None
    if not missing and not path_limit_reached:
        result = 0.0
        # 路径成功事件的并集：对路径集合做容斥，交集对应所有涉及块同时成功。
        for subset_size in range(1, len(path_blocks) + 1):
            sign = 1.0 if subset_size % 2 == 1 else -1.0
            for subset in itertools.combinations(path_blocks, subset_size):
                required_blocks: Set[str] = set().union(*subset)
                event_probability = math.prod(block_reliability[node_id] for node_id in required_blocks)
                result += sign * event_probability
        system_reliability = min(max(result, 0.0), 1.0)

    named_paths = [
        [_name(vertices[node_id]) for node_id in path if node_id not in (start, end)]
        for path in paths
    ]
    return {
        "model": "RBD",
        "start_id": start,
        "end_id": end,
        "success_paths": named_paths,
        "success_path_count": len(paths),
        "success_path_count_complete": not path_limit_reached,
        "system_reliability": system_reliability,
        "system_unreliability": None if system_reliability is None else 1.0 - system_reliability,
        "mission_time": mission_time,
        "missing_reliability_blocks": sorted(set(missing)),
        "path_limit": max_paths,
        "path_limit_reached": path_limit_reached,
        "calculation_limit_reason": (
            f"成功路径超过安全枚举上限 {max_paths}，禁止基于截断路径给出系统可靠度"
            if path_limit_reached
            else ""
        ),
        "assumption": "功能块相互独立；成功路径并集使用容斥原理计算",
    }
