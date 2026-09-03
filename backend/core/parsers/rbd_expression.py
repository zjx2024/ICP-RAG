"""Deterministic parser for explicit RBD structure expressions.

Engineering documents often state topology as ``SERIES(A, PARALLEL(B, C), D)``
next to the picture.  This module treats that expression as a traceable source
candidate; it does not infer topology from prose and never calls an LLM.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class RbdExpressionError(ValueError):
    pass


_OPERATORS = {"SERIES", "PARALLEL", "REDUNDANT", "BYPASS"}


@dataclass(frozen=True)
class RbdExpr:
    kind: str
    value: str = ""
    children: Tuple["RbdExpr", ...] = ()

    def canonical(self) -> str:
        if self.kind == "BLOCK":
            return self.value
        return f"{self.kind}({','.join(child.canonical() for child in self.children)})"


def _normalize_source(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("（", "(").replace("）", ")").replace("，", ",")
    # Accept a paragraph such as "结构表达式：SERIES(...)" without treating the
    # human-readable prefix as a block.
    marker = re.search(r"\b(?:SERIES|PARALLEL|REDUNDANT|BYPASS)\s*\(", text, flags=re.IGNORECASE)
    if marker:
        text = text[marker.start():]
    return text.strip().rstrip("。；;")


def _tokenize(value: str) -> List[str]:
    source = _normalize_source(value)
    if not source:
        raise RbdExpressionError("RBD 结构表达式为空")
    tokens: List[str] = []
    pos = 0
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        if source[pos] in "(),":
            tokens.append(source[pos])
            pos += 1
            continue
        end = pos
        while end < len(source) and source[end] not in "()," and not source[end].isspace():
            end += 1
        token = source[pos:end].strip()
        if not token:
            raise RbdExpressionError(f"无法解析 RBD 表达式位置 {pos}")
        tokens.append(token)
        pos = end
    return tokens


def parse_rbd_expression(value: str) -> RbdExpr:
    tokens = _tokenize(value)
    index = 0

    def parse_one() -> RbdExpr:
        nonlocal index
        if index >= len(tokens) or tokens[index] in {"(", ")", ","}:
            raise RbdExpressionError(f"RBD 表达式缺少模块或操作符（token={index}）")
        name = tokens[index].strip()
        index += 1
        if index < len(tokens) and tokens[index] == "(":
            operator = name.upper()
            if operator not in _OPERATORS:
                raise RbdExpressionError(f"不支持的 RBD 操作符: {name}")
            index += 1
            children: List[RbdExpr] = []
            while True:
                if index >= len(tokens):
                    raise RbdExpressionError("RBD 表达式缺少右括号")
                if tokens[index] == ")":
                    index += 1
                    break
                children.append(parse_one())
                if index >= len(tokens):
                    raise RbdExpressionError("RBD 表达式缺少右括号")
                if tokens[index] == ",":
                    index += 1
                    continue
                if tokens[index] == ")":
                    index += 1
                    break
                raise RbdExpressionError(f"RBD 表达式中存在意外 token: {tokens[index]}")
            if not children:
                raise RbdExpressionError(f"{operator} 至少需要一个子项")
            if operator in {"PARALLEL", "REDUNDANT", "BYPASS"} and len(children) < 2:
                raise RbdExpressionError(f"{operator} 至少需要两个分支")
            return RbdExpr(kind=operator, children=tuple(children))
        return RbdExpr(kind="BLOCK", value=name)

    result = parse_one()
    if index != len(tokens):
        raise RbdExpressionError(f"RBD 表达式尾部存在未解析内容: {' '.join(tokens[index:])}")
    return result


def _norm_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]+", "", text)


def _parameter_lookup(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        for key in ("module_id", "abbreviation", "short_name", "module_name", "name"):
            normalized = _norm_key(item.get(key))
            if normalized:
                lookup.setdefault(normalized, item)
    return lookup


def _as_number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def rbd_expression_to_graph(
    expression: str | RbdExpr,
    parameters: Sequence[Mapping[str, Any]] = (),
    *,
    title: str = "",
    source_ref: str = "",
) -> Dict[str, Any]:
    """Build a stable START→END directed graph from an explicit expression.

    The compiler connects sets of entry/exit blocks directly, avoiding virtual
    junction nodes that a reliability calculator could accidentally count as
    components.
    """
    root = parse_rbd_expression(expression) if isinstance(expression, str) else expression
    parameter_by_name = _parameter_lookup(parameters)
    vertices: List[Dict[str, Any]] = [
        {"id": "START", "label": "Start", "properties": {"name": "START"}},
        {"id": "END", "label": "End", "properties": {"name": "END"}},
    ]
    edges: List[Dict[str, Any]] = []
    occurrences: Dict[str, int] = {}
    edge_keys = set()

    def add_edge(source: str, target: str, connection_type: str, branch_id: str = "") -> None:
        identity = (source, target, connection_type, branch_id)
        if identity in edge_keys:
            return
        edge_keys.add(identity)
        digest = hashlib.sha256("|".join(identity).encode("utf-8")).hexdigest()[:16]
        props: Dict[str, Any] = {"connection_type": connection_type}
        if branch_id:
            props["branch_id"] = branch_id
        edges.append({
            "id": f"edge:{digest}",
            "label": "success_path",
            "source": source,
            "target": target,
            "properties": props,
        })

    def build(node: RbdExpr, path: str) -> Tuple[set[str], set[str]]:
        if node.kind == "BLOCK":
            normalized = _norm_key(node.value) or "BLOCK"
            occurrence = occurrences.get(normalized, 0) + 1
            occurrences[normalized] = occurrence
            node_id = f"block:{normalized}" if occurrence == 1 else f"block:{normalized}:{occurrence}"
            parameter = parameter_by_name.get(normalized, {})
            properties: Dict[str, Any] = {
                "name": node.value,
                "module_id": parameter.get("module_id", ""),
                "module_name": parameter.get("module_name") or parameter.get("name") or "",
                "source_ref": parameter.get("source_ref") or source_ref,
            }
            failure_rate = _as_number(
                parameter.get("failure_rate", parameter.get("lambda", parameter.get("λ")))
            )
            reliability = _as_number(parameter.get("reliability"))
            repair_rate = _as_number(
                parameter.get("repair_rate", parameter.get("mu", parameter.get("μ")))
            )
            if failure_rate is not None:
                properties["failure_rate"] = failure_rate
            if reliability is not None:
                properties["reliability"] = reliability
            if repair_rate is not None:
                properties["repair_rate"] = repair_rate
            vertices.append({"id": node_id, "label": "ReliabilityBlock", "properties": properties})
            return {node_id}, {node_id}

        child_parts = [build(child, f"{path}.{idx}") for idx, child in enumerate(node.children, start=1)]
        if node.kind == "SERIES":
            for idx in range(len(child_parts) - 1):
                for source in child_parts[idx][1]:
                    for target in child_parts[idx + 1][0]:
                        add_edge(source, target, "series")
            return child_parts[0][0], child_parts[-1][1]

        # PARALLEL/REDUNDANT/BYPASS share path topology; the exact engineering
        # semantics remains on the connecting edges for review and explanation.
        connection = node.kind.lower()
        entries: set[str] = set()
        exits: set[str] = set()
        for idx, (child_entries, child_exits) in enumerate(child_parts, start=1):
            entries.update(child_entries)
            exits.update(child_exits)
            branch_id = f"{path}:branch:{idx}"
            for entry in child_entries:
                # Self metadata is represented on the first edge from the parent
                # boundary later; no synthetic graph node is introduced here.
                for vertex in vertices:
                    if vertex["id"] == entry:
                        vertex.setdefault("properties", {})["branch_id"] = branch_id
                        vertex["properties"]["connection_type"] = connection
                        break
        return entries, exits

    entries, exits = build(root, "root")
    top_connection = root.kind.lower() if root.kind != "BLOCK" else "series"
    for idx, entry in enumerate(sorted(entries), start=1):
        add_edge("START", entry, top_connection, f"root:entry:{idx}" if len(entries) > 1 else "")
    for idx, exit_id in enumerate(sorted(exits), start=1):
        add_edge(exit_id, "END", top_connection, f"root:exit:{idx}" if len(exits) > 1 else "")

    vertices.sort(key=lambda item: str(item.get("id")))
    edges.sort(key=lambda item: str(item.get("id")))
    canonical_expression = root.canonical()
    return {
        "diagram_type": "rbd",
        "root_fault_id": "",
        "metadata": {
            "title": title,
            "structure_expression": canonical_expression,
            "source_ref": source_ref,
            "start_id": "START",
            "end_id": "END",
            "topology_source": "structured_expression",
        },
        "vertices": vertices,
        "edges": edges,
    }
