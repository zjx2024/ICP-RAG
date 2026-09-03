"""Deterministic graph projections for structured ICP PostgreSQL rows.

The functions in this module are deliberately storage- and model-independent.
They only project relationships that are explicitly present in PostgreSQL
columns.  Exact business identifiers are stable across documents; free text is
deduplicated only inside one document and one field kind.  Name similarity,
embeddings, and LLM judgements never participate in identity resolution.

The returned graph uses the canonical ``vertices``/``edges`` shape already
used by ICP diagram code, but it is not written to HugeGraph here.  Callers are
expected to retain PostgreSQL as the authority and treat this as a rebuildable
projection.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


SUPPORTED_DOC_TYPES = frozenset(
    {"function_locator", "hmc", "fmeca", "mfl", "fault_history"}
)
_BUILDER_DOC_TYPES = frozenset((*SUPPORTED_DOC_TYPES, "fta", "rbd"))

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_FUNC_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])FUNC-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MODULE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])MOD-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_FTA_DIAGRAM_ID_RE = re.compile(
    r"FTA-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.IGNORECASE,
)
_FTA_EVENT_ID_RE = re.compile(
    r"(?:EVT|FTA)-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.IGNORECASE,
)
_RBD_DIAGRAM_ID_RE = re.compile(
    r"RBD-[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
    re.IGNORECASE,
)
_MISSING = object()


class StructuredGraphError(ValueError):
    """Raised when structured rows cannot be projected without mixing scope."""


def clean_text(value: Any) -> str:
    """Normalize presentation whitespace while preserving codes and zeroes."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _SPACE_RE.sub(" ", text.replace("\r", " ").replace("\n", " ")).strip()


def _token(value: Any) -> str:
    token = _TOKEN_RE.sub("_", clean_text(value).casefold()).strip("_")
    return token or "item"


def _digest(*parts: Any) -> str:
    content = "\x1f".join(clean_text(part) for part in parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def business_node_id(entity_kind: str, business_id: Any) -> str:
    """Return a cross-document ID for one explicit domain identifier.

    Case and leading zeroes are intentionally significant.  Only NFKC and
    whitespace normalization are applied, avoiding accidental semantic merges.
    """

    kind = _token(entity_kind)
    identity = clean_text(business_id)
    if not identity:
        raise StructuredGraphError("business_id must not be empty")
    return f"sg:b:{kind}:{_digest(kind, identity)}"


def document_text_node_id(doc_id: str, value_kind: str, text: Any) -> str:
    """Return an ID whose text identity cannot cross a document boundary."""

    document = clean_text(doc_id)
    value = clean_text(text)
    if not document or not value:
        raise StructuredGraphError("doc_id and text must not be empty")
    kind = _token(value_kind)
    return f"sg:t:{kind}:{_digest(document, kind, value)}"


def _edge_id(
    doc_id: str,
    relation: str,
    source: str,
    target: str,
    qualifier: Any = "",
) -> str:
    relation_token = _token(relation)
    return (
        f"sg:e:{relation_token}:"
        f"{_digest(doc_id, relation_token, source, target, qualifier)}"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_json_safe(item) for item in value]
        return sorted(values, key=_canonical_json)
    return clean_text(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique(values: Iterable[Any]) -> List[Any]:
    by_value = {_canonical_json(value): _json_safe(value) for value in values}
    return [by_value[key] for key in sorted(by_value)]


def _string_values(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return list(
        dict.fromkeys(text for item in values if (text := clean_text(item)))
    )


def _source_refs(row: Mapping[str, Any]) -> List[str]:
    return _string_values(row.get("source_refs") or row.get("source_ref"))


def _explicit_ids(value: Any, pattern: re.Pattern[str]) -> List[str]:
    """Extract only syntactically explicit business IDs from declared fields.

    This helper is intentionally not a name/alias resolver.  It is used only
    on fields whose schema already declares that they contain cross-document
    references, and it preserves the exact spelling from the source.
    """

    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    matches: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            # Canonical diagram reconciliation may retain candidate details
            # beside one selected scalar.  Only the explicit selected value is
            # a field assertion; IDs buried in audit metadata are not.
            item = item.get("value")
        text = clean_text(item)
        if not text:
            continue
        matches.extend(match.group(0) for match in pattern.finditer(text))
    return list(dict.fromkeys(matches))


def _declared_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    return clean_text(value)


def _json_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


class _GraphBuilder:
    def __init__(self, doc_type: str, doc_id: str) -> None:
        dtype = clean_text(doc_type).casefold()
        document = clean_text(doc_id)
        if dtype not in _BUILDER_DOC_TYPES:
            raise StructuredGraphError(f"unsupported doc_type: {doc_type}")
        if not document:
            raise StructuredGraphError("doc_id must not be empty")
        self.doc_type = dtype
        self.doc_id = document
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self._warnings: Dict[str, Dict[str, Any]] = {}

    def warning(
        self,
        code: str,
        message: str,
        *,
        record_id: str = "",
        field: str = "",
    ) -> None:
        warning = {
            "code": clean_text(code),
            "message": clean_text(message),
            "record_id": clean_text(record_id),
            "field": clean_text(field),
        }
        self._warnings[_canonical_json(warning)] = warning

    def checked_rows(
        self, rows: Iterable[Mapping[str, Any]], *, group: str
    ) -> List[Mapping[str, Any]]:
        checked: List[Mapping[str, Any]] = []
        for index, row in enumerate(rows or (), start=1):
            if not isinstance(row, Mapping):
                raise StructuredGraphError(f"{group} row {index} must be a mapping")
            row_doc_id = clean_text(row.get("doc_id"))
            if row_doc_id and row_doc_id != self.doc_id:
                raise StructuredGraphError(
                    f"{group} row {index} belongs to doc_id={row_doc_id}, "
                    f"expected {self.doc_id}"
                )
            checked.append(row)
        return checked

    def _merge_properties(
        self,
        current: MutableMapping[str, Any],
        incoming: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> None:
        for key, raw_value in incoming.items():
            value = _json_safe(raw_value)
            if value in (None, "", [], {}):
                continue
            existing = current.get(key, _MISSING)
            if existing is _MISSING or existing in (None, "", [], {}):
                current[key] = value
                continue
            if _canonical_json(existing) == _canonical_json(value):
                continue
            if key in {
                "doc_ids",
                "source_doc_types",
                "source_refs",
                "variant_ids",
                "occurrence_qualifiers",
            }:
                left = existing if isinstance(existing, list) else [existing]
                right = value if isinstance(value, list) else [value]
                current[key] = _unique([*left, *right])
                continue

            # An exact business ID may legitimately occur in several source
            # rows or document versions.  Preserve every exact value without
            # choosing one semantically.  The lexicographically first value is
            # kept for scalar-schema compatibility and alternatives are audited.
            alternatives_key = f"{key}_alternatives"
            prior = current.get(alternatives_key, [])
            alternatives = prior if isinstance(prior, list) else [prior]
            alternatives = _unique([existing, value, *alternatives])
            current[key] = alternatives[0]
            current[alternatives_key] = alternatives
            self.warning(
                "explicit_id_property_conflict",
                f"同一明确标识的字段 {key} 存在多个原始值，已全部保留且未做语义选择",
                record_id=owner_id,
                field=key,
            )

    def add_business_node(
        self,
        entity_kind: str,
        label: str,
        business_id: Any,
        properties: Mapping[str, Any] | None = None,
        *,
        source_refs: Sequence[str] = (),
    ) -> str:
        identity = clean_text(business_id)
        node_id = business_node_id(entity_kind, identity)
        base_properties: Dict[str, Any] = {
            "business_id": identity,
            "identity_scope": "business_id",
            # fault_history reference nodes intentionally share an ID with the
            # target HMC/FMECA/MFL node.  Source types therefore form a set
            # instead of a scalar that would conflict when projections merge.
            "source_doc_types": [self.doc_type],
            "doc_ids": [self.doc_id],
        }
        if source_refs:
            base_properties["source_refs"] = list(source_refs)
        self._merge_properties(
            base_properties, properties or {}, owner_id=node_id
        )
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "id": node_id,
                "label": clean_text(label) or "StructuredEntity",
                "properties": base_properties,
            }
        else:
            if clean_text(existing.get("label")) != clean_text(label):
                self.warning(
                    "explicit_id_label_conflict",
                    "同一明确标识具有不同实体类型，未自动拆分或重命名",
                    record_id=node_id,
                    field="label",
                )
            self._merge_properties(
                existing["properties"], base_properties, owner_id=node_id
            )
        return node_id

    def add_text_node(
        self,
        value_kind: str,
        text: Any,
        *,
        source_refs: Sequence[str] = (),
    ) -> str:
        value = clean_text(text)
        node_id = document_text_node_id(self.doc_id, value_kind, value)
        properties: Dict[str, Any] = {
            "name": value,
            "text": value,
            "value_kind": _token(value_kind),
            "identity_scope": "document_text",
            "doc_type": self.doc_type,
            "doc_ids": [self.doc_id],
        }
        if source_refs:
            properties["source_refs"] = list(source_refs)
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "id": node_id,
                "label": "TextValue",
                "properties": properties,
            }
        else:
            self._merge_properties(
                existing["properties"], properties, owner_id=node_id
            )
        return node_id

    def add_document_node(
        self,
        entity_kind: str,
        label: str,
        occurrence_id: Any,
        properties: Mapping[str, Any] | None = None,
        *,
        source_refs: Sequence[str] = (),
    ) -> str:
        """Create a document-scoped structural occurrence node.

        Ordered history steps are occurrences, not free-text identities.  A
        separate occurrence node preserves repeated step text and sequence
        without relying on parallel edges, which HugeGraph ``SINGLE`` edge
        labels intentionally collapse.
        """

        identity = clean_text(occurrence_id)
        node_id = document_text_node_id(self.doc_id, entity_kind, identity)
        node_properties: Dict[str, Any] = {
            "occurrence_id": identity,
            "identity_scope": "document_occurrence",
            "doc_type": self.doc_type,
            "doc_ids": [self.doc_id],
        }
        if source_refs:
            node_properties["source_refs"] = list(source_refs)
        self._merge_properties(
            node_properties, properties or {}, owner_id=node_id
        )
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "id": node_id,
                "label": clean_text(label) or "StructuredOccurrence",
                "properties": node_properties,
            }
        else:
            self._merge_properties(
                existing["properties"], node_properties, owner_id=node_id
            )
        return node_id

    def add_edge(
        self,
        relation: str,
        source: str,
        target: str,
        *,
        qualifier: Any = "",
        properties: Mapping[str, Any] | None = None,
        source_refs: Sequence[str] = (),
    ) -> str:
        # HugeGraph ``SINGLE`` labels permit only one edge for a given
        # (label, source, target).  Use that same tuple as logical identity and
        # retain row/field occurrences as provenance instead of fabricating
        # parallel edges that the storage engine would collapse.
        edge_id = _edge_id(self.doc_id, relation, source, target)
        edge_properties: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "explicit": True,
        }
        if qualifier not in (None, "", [], {}):
            edge_properties["occurrence_qualifiers"] = [_json_safe(qualifier)]
        if source_refs:
            edge_properties["source_refs"] = list(source_refs)
        self._merge_properties(
            edge_properties, properties or {}, owner_id=edge_id
        )
        existing = self.edges.get(edge_id)
        if existing is None:
            self.edges[edge_id] = {
                "id": edge_id,
                "label": _token(relation),
                "source": source,
                "target": target,
                "properties": edge_properties,
            }
        else:
            self._merge_properties(
                existing["properties"], edge_properties, owner_id=edge_id
            )
        return edge_id

    def link_text_fields(
        self,
        source_id: str,
        row: Mapping[str, Any],
        field_relations: Mapping[str, str],
        *,
        record_key: str,
        source_refs: Sequence[str],
    ) -> None:
        for field, relation in field_relations.items():
            for position, value in enumerate(_string_values(row.get(field)), start=1):
                target = self.add_text_node(
                    f"{self.doc_type}_{field}", value, source_refs=source_refs
                )
                self.add_edge(
                    relation,
                    source_id,
                    target,
                    qualifier=f"{record_key}:{field}:{position}",
                    properties={"field_name": field, "record_key": record_key},
                    source_refs=source_refs,
                )

    def result(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_type": "structured_postgres_projection",
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "vertices": [self.nodes[key] for key in sorted(self.nodes)],
            "edges": [self.edges[key] for key in sorted(self.edges)],
            "warnings": [self._warnings[key] for key in sorted(self._warnings)],
        }


def build_function_locator_graph(
    doc_id: str,
    *,
    components: Iterable[Mapping[str, Any]] = (),
    functions: Iterable[Mapping[str, Any]] = (),
    interfaces: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Project only explicit component/function/interface identifiers."""

    builder = _GraphBuilder("function_locator", doc_id)
    component_rows = builder.checked_rows(components, group="components")
    function_rows = builder.checked_rows(functions, group="functions")
    interface_rows = builder.checked_rows(interfaces, group="interfaces")

    for row in sorted(
        component_rows, key=lambda value: clean_text(value.get("component_id"))
    ):
        component_id = clean_text(row.get("component_id"))
        if not component_id:
            builder.warning(
                "missing_business_id",
                "component 行缺少 component_id，未创建图节点",
                field="component_id",
            )
            continue
        refs = _source_refs(row)
        node_id = builder.add_business_node(
            "component",
            "Component",
            component_id,
            {
                "name": clean_text(row.get("name")),
                "component_type": clean_text(row.get("component_type")).casefold(),
                "abbreviation": clean_text(row.get("abbreviation")),
                "description": clean_text(row.get("description")),
            },
            source_refs=refs,
        )
        parent_id = clean_text(row.get("parent_component_id"))
        if parent_id:
            parent = builder.add_business_node(
                "component", "Component", parent_id
            )
            builder.add_edge(
                "contains_component",
                parent,
                node_id,
                qualifier=component_id,
                properties={"child_component_id": component_id},
                source_refs=refs,
            )

    for row in sorted(
        function_rows, key=lambda value: clean_text(value.get("function_id"))
    ):
        function_id = clean_text(row.get("function_id"))
        if not function_id:
            builder.warning(
                "missing_business_id",
                "function 行缺少 function_id，未创建图节点",
                field="function_id",
            )
            continue
        refs = _source_refs(row)
        function_node = builder.add_business_node(
            "function",
            "Function",
            function_id,
            {
                "name": clean_text(row.get("function_name")),
                "description": clean_text(row.get("description")),
                "aliases": _string_values(row.get("aliases")),
                "failure_symptoms": clean_text(row.get("failure_symptoms")),
                # Names remain attributes.  They are never used to resolve an
                # edge to a component without an explicit component ID.
                "system_name": clean_text(row.get("system_name")),
                "subsystem_name": clean_text(row.get("subsystem_name")),
                "module_name": clean_text(row.get("module_name")),
            },
            source_refs=refs,
        )
        module_id = clean_text(row.get("module_id"))
        if module_id:
            module_node = builder.add_business_node(
                "component", "Component", module_id
            )
            builder.add_edge(
                "provides_function",
                module_node,
                function_node,
                qualifier=function_id,
                properties={"module_id": module_id, "function_id": function_id},
                source_refs=refs,
            )
        parent_function_id = clean_text(row.get("parent_function_id"))
        if parent_function_id:
            parent_function = builder.add_business_node(
                "function", "Function", parent_function_id
            )
            builder.add_edge(
                "contains_function",
                parent_function,
                function_node,
                qualifier=function_id,
                properties={"child_function_id": function_id},
                source_refs=refs,
            )
        builder.link_text_fields(
            function_node,
            row,
            {"inputs": "has_input", "outputs": "has_output"},
            record_key=function_id,
            source_refs=refs,
        )

    for row in sorted(
        interface_rows, key=lambda value: clean_text(value.get("interface_id"))
    ):
        interface_id = clean_text(row.get("interface_id"))
        if not interface_id:
            builder.warning(
                "missing_business_id",
                "interface 行缺少 interface_id，未创建图节点",
                field="interface_id",
            )
            continue
        refs = _source_refs(row)
        interface_node = builder.add_business_node(
            "interface",
            "Interface",
            interface_id,
            {
                "name": clean_text(row.get("signal_name")) or interface_id,
                "relation_type": clean_text(row.get("relation_type")),
                "direction": clean_text(row.get("direction")),
                "protocol": clean_text(row.get("protocol")),
                "description": clean_text(row.get("description")),
            },
            source_refs=refs,
        )
        source_component_id = clean_text(row.get("source_component_id"))
        if source_component_id:
            source_component = builder.add_business_node(
                "component", "Component", source_component_id
            )
            builder.add_edge(
                "interface_source",
                source_component,
                interface_node,
                qualifier=interface_id,
                properties={"interface_id": interface_id},
                source_refs=refs,
            )
        target_component_id = clean_text(row.get("target_component_id"))
        if target_component_id:
            target_component = builder.add_business_node(
                "component", "Component", target_component_id
            )
            builder.add_edge(
                "interface_target",
                interface_node,
                target_component,
                qualifier=interface_id,
                properties={"interface_id": interface_id},
                source_refs=refs,
            )

    return builder.result()


def build_hmc_graph(
    doc_id: str, rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    builder = _GraphBuilder("hmc", doc_id)
    checked = builder.checked_rows(rows, group="hmc")
    for row in sorted(
        checked,
        key=lambda value: (
            clean_text(value.get("hmc_code")),
            clean_text(value.get("variant_id")),
        ),
    ):
        hmc_code = clean_text(row.get("hmc_code"))
        if not hmc_code:
            builder.warning(
                "missing_business_id",
                "HMC 行缺少 hmc_code，未创建图节点",
                field="hmc_code",
            )
            continue
        refs = _source_refs(row)
        variant_id = clean_text(row.get("variant_id")) or _digest(
            builder.doc_id,
            hmc_code,
            row.get("fault_desc"),
            row.get("repair_method"),
        )
        code_node = builder.add_business_node(
            "hmc_code",
            "HMCCode",
            hmc_code,
            {
                "name": hmc_code,
                "lcn": clean_text(row.get("lcn")),
                "hamc": clean_text(row.get("hamc")),
                "hsc": clean_text(row.get("hsc")),
                "variant_ids": [variant_id],
            },
            source_refs=refs,
        )
        builder.link_text_fields(
            code_node,
            row,
            {
                "fault_desc": "describes_fault",
                "repair_method": "has_repair_method",
            },
            record_key=variant_id,
            source_refs=refs,
        )
    return builder.result()


def build_fmeca_graph(
    doc_id: str, rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    builder = _GraphBuilder("fmeca", doc_id)
    checked = builder.checked_rows(rows, group="fmeca")
    field_relations = {
        "product_flag": "has_product_flag",
        "func_management": "has_function_management",
        "fault_mode": "has_fault_mode",
        "fault_cause": "has_fault_cause",
        "mission_phase": "applies_in_mission_phase",
        "local_impact": "has_local_impact",
        "final_impact": "has_final_impact",
        "detect_method": "has_detection_method",
        "design_improve": "has_design_improvement",
        "compensate_action": "has_compensation_action",
    }
    for row in sorted(checked, key=lambda value: clean_text(value.get("code"))):
        code = clean_text(row.get("code"))
        if not code:
            builder.warning(
                "missing_business_id",
                "FMECA 行缺少 code，未创建图节点",
                field="code",
            )
            continue
        refs = _source_refs(row)
        record = builder.add_business_node(
            "fmeca_record",
            "FMECARecord",
            code,
            {
                "name": code,
                "severity": clean_text(row.get("severity")),
                "severity_raw": clean_text(row.get("severity_raw")),
                "severity_normalized": clean_text(row.get("severity_normalized")),
                "probability": _json_safe(row.get("probability")),
                "probability_raw": clean_text(row.get("probability_raw")),
                "probability_level": clean_text(row.get("probability_level")),
            },
            source_refs=refs,
        )
        builder.link_text_fields(
            record,
            row,
            field_relations,
            record_key=code,
            source_refs=refs,
        )
        # ``product_flag`` is still retained as source text above.  Only an
        # explicit FUNC-* token additionally creates a cross-document bridge;
        # product names and abbreviations never become Function identities.
        for function_id in _explicit_ids(row.get("product_flag"), _FUNC_ID_RE):
            function = builder.add_business_node(
                "function",
                "Function",
                function_id,
                {"name": function_id},
                source_refs=refs,
            )
            builder.add_edge(
                "has_product_flag",
                record,
                function,
                qualifier=f"{code}:product_flag:{function_id}",
                properties={
                    "field_name": "product_flag",
                    "record_key": code,
                    "reference_id": function_id,
                    "reference_kind": "explicit_function_id",
                },
                source_refs=refs,
            )
    return builder.result()


def build_mfl_graph(
    doc_id: str, rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    builder = _GraphBuilder("mfl", doc_id)
    checked = builder.checked_rows(rows, group="mfl")
    field_relations = {
        "module_fault": "has_module_fault",
        "sur_fault": "has_sur_fault",
        "subcircuit_fault": "has_subcircuit_fault",
        "maintenance_action": "has_maintenance_action",
    }
    for row in sorted(
        checked, key=lambda value: clean_text(value.get("mfl_code"))
    ):
        code = clean_text(row.get("mfl_code"))
        if not code:
            builder.warning(
                "missing_business_id",
                "MFL 行缺少 mfl_code，未创建图节点",
                field="mfl_code",
            )
            continue
        refs = _source_refs(row)
        record = builder.add_business_node(
            "mfl_record",
            "MFLRecord",
            code,
            {
                "name": code,
                "api_reportable": row.get("api_reportable"),
                "api_reportable_raw": clean_text(row.get("api_reportable_raw")),
                "platform_test": row.get("platform_test"),
                "platform_test_raw": clean_text(row.get("platform_test_raw")),
                "platform_test_status": clean_text(row.get("platform_test_status")),
            },
            source_refs=refs,
        )
        builder.link_text_fields(
            record,
            row,
            field_relations,
            record_key=code,
            source_refs=refs,
        )
        api_name = clean_text(row.get("api_name"))
        if api_name:
            api_node = builder.add_business_node(
                "api", "API", api_name, {"name": api_name}, source_refs=refs
            )
            builder.add_edge(
                "reports_via_api",
                record,
                api_node,
                qualifier=code,
                properties={"mfl_code": code},
                source_refs=refs,
            )
            builder.link_text_fields(
                api_node,
                row,
                {"api_method": "has_api_method", "api_usage": "has_api_usage"},
                record_key=code,
                source_refs=refs,
            )
    return builder.result()


def build_fault_history_graph(
    doc_id: str, rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    builder = _GraphBuilder("fault_history", doc_id)
    checked = builder.checked_rows(rows, group="fault_history")
    text_relations = {
        "fault_symptom": "has_fault_symptom",
        "related_module": "mentions_module",
        "related_function": "mentions_function",
        "diagnosis_method": "has_diagnosis_method",
        "root_cause": "has_recorded_root_cause",
        "mission_phase": "occurred_in_mission_phase",
        "final_solution": "has_final_solution",
        "verification_result": "has_verification_result",
        "preventive_action": "has_preventive_action",
    }
    for row in sorted(
        checked, key=lambda value: clean_text(value.get("case_id"))
    ):
        case_id = clean_text(row.get("case_id"))
        if not case_id:
            builder.warning(
                "missing_business_id",
                "故障历史行缺少 case_id，未创建图节点",
                field="case_id",
            )
            continue
        refs = _source_refs(row)
        case = builder.add_business_node(
            "fault_history_case",
            "FaultHistoryCase",
            case_id,
            {
                "name": case_id,
                "occurred_at": clean_text(row.get("occurred_at")),
                "personnel": clean_text(row.get("personnel")),
                "device_type": clean_text(row.get("device_type")),
                "document_links": _json_safe(row.get("document_links") or {}),
            },
            source_refs=refs,
        )
        builder.link_text_fields(
            case,
            row,
            text_relations,
            record_key=case_id,
            source_refs=refs,
        )
        links = _json_mapping(row.get("document_links"))

        # The parser keeps a backward-compatible scalar in each legacy column
        # and the complete normalized list in ``document_links``.  Build from
        # both and deduplicate before adding edges so later references are not
        # silently discarded.
        reference_groups = (
            (
                "hmc_code",
                "hmc_codes",
                "hmc_code",
                "HMCCode",
                "references_hmc",
            ),
            (
                "fmeca_code",
                "fmeca_refs",
                "fmeca_record",
                "FMECARecord",
                "references_fmeca",
            ),
            (
                "mfl_code",
                "mfl_refs",
                "mfl_record",
                "MFLRecord",
                "references_mfl",
            ),
            (
                "rbd_ref",
                "rbd_refs",
                "rbd_diagram",
                "RBDDiagramRef",
                "references_rbd",
            ),
        )
        for (
            scalar_field,
            links_field,
            entity_kind,
            label,
            relation,
        ) in reference_groups:
            business_ids = list(
                dict.fromkeys(
                    [
                        *_string_values(row.get(scalar_field)),
                        *_string_values(links.get(links_field)),
                    ]
                )
            )
            for business_id in business_ids:
                target = builder.add_business_node(
                    entity_kind,
                    label,
                    business_id,
                    {"name": business_id},
                    source_refs=refs,
                )
                builder.add_edge(
                    relation,
                    case,
                    target,
                    qualifier=f"{case_id}:{links_field}:{business_id}",
                    properties={
                        "field_name": links_field,
                        "record_key": case_id,
                        "reference_id": business_id,
                    },
                    source_refs=refs,
                )

        # FTA document IDs and FTA event IDs have different identities even
        # when both live in the same declared list.  They share only the
        # ``references_fta`` relation from this case.
        fta_refs = list(
            dict.fromkeys(
                [
                    *_string_values(row.get("fta_ref")),
                    *_string_values(links.get("fta_refs")),
                ]
            )
        )
        for business_id in fta_refs:
            is_event = business_id.upper().startswith("EVT-")
            target = builder.add_business_node(
                "fta_event" if is_event else "fta_diagram",
                "FTAEvent" if is_event else "FTADiagramRef",
                business_id,
                {"name": business_id},
                source_refs=refs,
            )
            builder.add_edge(
                "references_fta",
                case,
                target,
                qualifier=f"{case_id}:fta_refs:{business_id}",
                properties={
                    "field_name": "fta_refs",
                    "record_key": case_id,
                    "reference_id": business_id,
                    "reference_kind": "event" if is_event else "diagram",
                },
                source_refs=refs,
            )

        function_ids = list(
            dict.fromkeys(
                [
                    *_string_values(links.get("function_refs")),
                    *_explicit_ids(row.get("related_function"), _FUNC_ID_RE),
                ]
            )
        )
        for function_id in function_ids:
            # A document_links value remains an exact reference only when it
            # still has the declared FUNC-* syntax.
            if not _FUNC_ID_RE.fullmatch(function_id):
                continue
            target = builder.add_business_node(
                "function",
                "Function",
                function_id,
                {"name": function_id},
                source_refs=refs,
            )
            builder.add_edge(
                "mentions_function",
                case,
                target,
                qualifier=f"{case_id}:function_refs:{function_id}",
                properties={
                    "field_name": "function_refs",
                    "record_key": case_id,
                    "reference_id": function_id,
                    "reference_kind": "explicit_function_id",
                },
                source_refs=refs,
            )

        for module_id in _explicit_ids(row.get("related_module"), _MODULE_ID_RE):
            target = builder.add_business_node(
                "component",
                "Component",
                module_id,
                {"name": module_id, "component_type": "module"},
                source_refs=refs,
            )
            builder.add_edge(
                "mentions_module",
                case,
                target,
                qualifier=f"{case_id}:related_module:{module_id}",
                properties={
                    "field_name": "related_module",
                    "record_key": case_id,
                    "reference_id": module_id,
                    "reference_kind": "explicit_module_id",
                },
                source_refs=refs,
            )

        raw_steps = row.get("steps") or ()
        if isinstance(raw_steps, (str, Mapping)):
            raw_steps = [raw_steps]
        previous_step_node = ""
        for position, step in enumerate(raw_steps, start=1):
            if isinstance(step, Mapping):
                content = clean_text(step.get("content"))
                raw_step_no = step.get("step_no", position)
                step_refs = _source_refs(step) or refs
            else:
                content = clean_text(step)
                raw_step_no = position
                step_refs = refs
            if not content:
                continue
            try:
                step_no = int(raw_step_no)
            except (TypeError, ValueError):
                step_no = position
            step_text_node = builder.add_text_node(
                "fault_history_step", content, source_refs=step_refs
            )
            step_node = builder.add_document_node(
                "fault_history_step_occurrence",
                "FaultHistoryStep",
                f"{case_id}:{step_no}:{position}",
                {
                    "name": f"{case_id} 步骤 {step_no}",
                    "case_id": case_id,
                    "step_no": step_no,
                    "position": position,
                },
                source_refs=step_refs,
            )
            builder.add_edge(
                "has_diagnosis_step",
                case,
                step_node,
                qualifier=f"{case_id}:step:{step_no}:{position}",
                properties={"step_no": step_no, "position": position},
                source_refs=step_refs,
            )
            builder.add_edge(
                "has_step_content",
                step_node,
                step_text_node,
                qualifier=f"{case_id}:step:{step_no}:{position}:content",
                properties={"step_no": step_no, "position": position},
                source_refs=step_refs,
            )
            if previous_step_node:
                builder.add_edge(
                    "next_step",
                    previous_step_node,
                    step_node,
                    qualifier=f"{case_id}:step:{step_no}:{position}:next",
                    properties={"to_step_no": step_no, "to_position": position},
                    source_refs=step_refs,
                )
            previous_step_node = step_node
    return builder.result()


def _diagram_revision_properties(revision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "diagram_id": clean_text(revision.get("diagram_id")),
        "diagram_key": clean_text(revision.get("diagram_key")),
        "revision_id": clean_text(revision.get("revision_id")),
        "revision_no": revision.get("revision_no"),
        "canonical_hash": clean_text(revision.get("canonical_hash")),
        "review_status": clean_text(revision.get("review_status")),
        "validation_status": clean_text(revision.get("validation_status")),
        "validation_tier": clean_text(revision.get("validation_tier")),
        "projection_status": clean_text(revision.get("projection_status")),
        "is_active": bool(revision.get("is_active")),
        "formal_claims_allowed": True,
    }


def _diagram_source_refs(
    revision: Mapping[str, Any],
    graph: Mapping[str, Any],
    item: Mapping[str, Any] | None = None,
) -> List[str]:
    metadata = graph.get("metadata") or {}
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    properties = (item or {}).get("properties") or {}
    properties = dict(properties) if isinstance(properties, Mapping) else {}
    refs = [
        *_string_values(revision.get("source_refs") or revision.get("source_ref")),
        *_string_values(metadata.get("source_refs") or metadata.get("source_ref")),
        *_string_values(properties.get("source_refs") or properties.get("source_ref")),
    ]
    return list(dict.fromkeys(refs))


def _canonical_graph(revision: Mapping[str, Any]) -> Dict[str, Any]:
    value = revision.get("canonical_graph") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise StructuredGraphError("diagram canonical_graph is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise StructuredGraphError("diagram canonical_graph must be a mapping")
    return dict(value)


def _formal_revisions(
    builder: _GraphBuilder,
    revisions: Iterable[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    # Lazy import keeps the base table builders independent from diagram model
    # initialization and avoids making a visual candidate authoritative by
    # merely passing it to this function.
    from backend.icp.diagram_validation import revision_allows_formal_claims

    accepted: List[Mapping[str, Any]] = []
    for index, revision in enumerate(revisions or (), start=1):
        if not isinstance(revision, Mapping):
            raise StructuredGraphError(f"diagram revision {index} must be a mapping")
        revision_id = clean_text(revision.get("revision_id"))
        row_doc_id = clean_text(revision.get("doc_id"))
        row_type = clean_text(revision.get("diagram_type")).casefold()
        if row_doc_id and row_doc_id != builder.doc_id:
            raise StructuredGraphError(
                f"diagram revision {revision_id or index} belongs to doc_id={row_doc_id}, "
                f"expected {builder.doc_id}"
            )
        if row_type and row_type != builder.doc_type:
            raise StructuredGraphError(
                f"diagram revision {revision_id or index} has type={row_type}, "
                f"expected {builder.doc_type}"
            )
        reasons: List[str] = []
        if not revision_id:
            reasons.append("missing_revision_id")
        if revision.get("is_active") is not True:
            reasons.append("not_active")
        if clean_text(revision.get("projection_status")) != "published":
            reasons.append("not_published")
        if not revision_allows_formal_claims(revision):
            reasons.append("formal_claims_not_allowed")
        if reasons:
            builder.warning(
                "diagram_revision_excluded",
                "工程图修订未进入正式公共图：" + ",".join(reasons),
                record_id=revision_id or str(index),
            )
            continue
        accepted.append(revision)
    return sorted(
        accepted,
        key=lambda value: (
            clean_text(value.get("diagram_id")),
            int(value.get("revision_no") or 0),
            clean_text(value.get("revision_id")),
        ),
    )


def _safe_occurrence_properties(properties: Any) -> Dict[str, Any]:
    """Keep raw diagram fields without turning an occurrence into identity."""

    values = dict(properties) if isinstance(properties, Mapping) else {}
    business_id = clean_text(values.pop("business_id", ""))
    if business_id:
        values["source_business_id"] = business_id
    values.pop("doc_ids", None)
    values.pop("source_doc_types", None)
    values.pop("identity_scope", None)
    return values


def _diagram_business_ids(
    revision: Mapping[str, Any],
    graph: Mapping[str, Any],
    pattern: re.Pattern[str],
    *,
    diagram_type: str,
) -> List[str]:
    metadata = graph.get("metadata") or {}
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    declared = [
        metadata.get("business_id"),
        metadata.get("business_key"),
        metadata.get("model_id"),
        metadata.get("diagram_id"),
        metadata.get("tree_id") if diagram_type == "fta" else metadata.get("rbd_id"),
        revision.get("diagram_key"),
    ]
    result: List[str] = []
    for value in declared:
        text = clean_text(value)
        if text and pattern.fullmatch(text) and text not in result:
            result.append(text)
    return result


def _add_function_bridges(
    builder: _GraphBuilder,
    source_node: str,
    values: Iterable[Any],
    *,
    record_key: str,
    revision_properties: Mapping[str, Any],
    source_refs: Sequence[str],
) -> None:
    function_ids: List[str] = []
    for value in values:
        function_ids.extend(_explicit_ids(value, _FUNC_ID_RE))
    for function_id in dict.fromkeys(function_ids):
        function = builder.add_business_node(
            "function",
            "Function",
            function_id,
            {"name": function_id},
            source_refs=source_refs,
        )
        builder.add_edge(
            "implements_function",
            source_node,
            function,
            qualifier=f"{record_key}:function:{function_id}",
            properties={
                **dict(revision_properties),
                "record_key": record_key,
                "reference_id": function_id,
                "reference_kind": "explicit_function_id",
            },
            source_refs=source_refs,
        )


def _build_fta_revision(builder: _GraphBuilder, revision: Mapping[str, Any]) -> None:
    graph = _canonical_graph(revision)
    revision_id = clean_text(revision.get("revision_id"))
    diagram_id = clean_text(revision.get("diagram_id"))
    common = _diagram_revision_properties(revision)
    graph_refs = _diagram_source_refs(revision, graph)
    diagram_occurrence = builder.add_document_node(
        "fta_diagram_occurrence",
        "FTADiagramRef",
        f"{revision_id}:{diagram_id}",
        {
            **common,
            "name": clean_text(revision.get("title"))
            or clean_text(revision.get("diagram_key"))
            or diagram_id,
            "source_diagram_id": diagram_id,
        },
        source_refs=graph_refs,
    )
    for business_id in _diagram_business_ids(
        revision, graph, _FTA_DIAGRAM_ID_RE, diagram_type="fta"
    ):
        target = builder.add_business_node(
            "fta_diagram",
            "FTADiagramRef",
            business_id,
            {"name": business_id},
            source_refs=graph_refs,
        )
        builder.add_edge(
            "references_fta",
            diagram_occurrence,
            target,
            qualifier=f"{revision_id}:diagram:{business_id}",
            properties={
                **common,
                "record_key": diagram_id,
                "reference_id": business_id,
                "reference_kind": "diagram",
            },
            source_refs=graph_refs,
        )

    metadata = graph.get("metadata") or {}
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    _add_function_bridges(
        builder,
        diagram_occurrence,
        (
            metadata.get("linked_function"),
            metadata.get("function_id"),
            metadata.get("function_ids"),
        ),
        record_key=diagram_id,
        revision_properties=common,
        source_refs=graph_refs,
    )

    node_map: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    root_id = clean_text(graph.get("root_fault_id") or graph.get("root_id"))
    for index, item in enumerate(graph.get("vertices") or (), start=1):
        if not isinstance(item, Mapping):
            raise StructuredGraphError(f"FTA revision {revision_id} vertex {index} is invalid")
        source_node_id = clean_text(item.get("id"))
        if not source_node_id:
            raise StructuredGraphError(f"FTA revision {revision_id} vertex {index} has no id")
        source_label = clean_text(item.get("label"))
        is_gate = source_label == "Gate"
        label = "FTAGate" if is_gate else "FTAEvent"
        item_refs = _diagram_source_refs(revision, graph, item)
        raw_properties = _safe_occurrence_properties(item.get("properties"))
        occurrence = builder.add_document_node(
            "fta_gate_occurrence" if is_gate else "fta_event_occurrence",
            label,
            f"{revision_id}:{source_node_id}",
            {
                **raw_properties,
                **common,
                "name": clean_text(raw_properties.get("name")) or source_node_id,
                "source_node_id": source_node_id,
                "event_type": source_label if not is_gate else "",
            },
            source_refs=item_refs,
        )
        node_map[source_node_id] = occurrence
        labels[source_node_id] = "gate" if is_gate else "event"

        if not is_gate:
            explicit_event_ids: List[str] = []
            for value in (
                raw_properties.get("event_id"),
                raw_properties.get("source_business_id"),
            ):
                text = _declared_text(value)
                if text and _FTA_EVENT_ID_RE.fullmatch(text):
                    explicit_event_ids.append(text)
            for event_id in dict.fromkeys(explicit_event_ids):
                event_ref = builder.add_business_node(
                    "fta_event",
                    "FTAEvent",
                    event_id,
                    {"name": event_id},
                    source_refs=item_refs,
                )
                builder.add_edge(
                    "references_fta",
                    occurrence,
                    event_ref,
                    qualifier=f"{revision_id}:{source_node_id}:event:{event_id}",
                    properties={
                        **common,
                        "record_key": source_node_id,
                        "reference_id": event_id,
                        "reference_kind": "event",
                    },
                    source_refs=item_refs,
                )

            failure_mode_id = _declared_text(raw_properties.get("failure_mode_id"))
            if failure_mode_id:
                failure_mode = builder.add_business_node(
                    "fmeca_record",
                    "FMECARecord",
                    failure_mode_id,
                    {"name": failure_mode_id},
                    source_refs=item_refs,
                )
                builder.add_edge(
                    "represents_fault_mode",
                    occurrence,
                    failure_mode,
                    qualifier=f"{revision_id}:{source_node_id}:{failure_mode_id}",
                    properties={
                        **common,
                        "record_key": source_node_id,
                        "reference_id": failure_mode_id,
                    },
                    source_refs=item_refs,
                )
            _add_function_bridges(
                builder,
                occurrence,
                (
                    raw_properties.get("function_id"),
                    raw_properties.get("linked_function"),
                    raw_properties.get("function_ids"),
                ),
                record_key=source_node_id,
                revision_properties=common,
                source_refs=item_refs,
            )

    if root_id and root_id in node_map:
        builder.add_edge(
            "references_fta",
            node_map[root_id],
            diagram_occurrence,
            qualifier=f"{revision_id}:root:{root_id}",
            properties={**common, "record_key": root_id, "reference_kind": "diagram_occurrence"},
            source_refs=graph_refs,
        )

    for index, item in enumerate(graph.get("edges") or (), start=1):
        if not isinstance(item, Mapping):
            raise StructuredGraphError(f"FTA revision {revision_id} edge {index} is invalid")
        source_key = clean_text(item.get("source") or item.get("outV"))
        target_key = clean_text(item.get("target") or item.get("inV"))
        source = node_map.get(source_key)
        target = node_map.get(target_key)
        edge_key = clean_text(item.get("id")) or f"edge-{index}"
        if not source or not target:
            builder.warning(
                "diagram_edge_unknown_endpoint",
                "FTA 正式修订存在无法定位端点的边，未投影该边",
                record_id=f"{revision_id}:{edge_key}",
            )
            continue
        edge_refs = _diagram_source_refs(revision, graph, item)
        raw_properties = _safe_occurrence_properties(item.get("properties"))
        edge_properties = {
            **raw_properties,
            **common,
            "record_key": edge_key,
            "source_edge_id": edge_key,
            "source_relation": clean_text(item.get("label")),
        }
        source_kind = labels.get(source_key)
        target_kind = labels.get(target_key)
        if source_kind == "event" and target_kind == "gate":
            builder.add_edge(
                "decomposed_by",
                source,
                target,
                qualifier=f"{revision_id}:{edge_key}",
                properties=edge_properties,
                source_refs=edge_refs,
            )
        elif source_kind == "gate" and target_kind == "event":
            builder.add_edge(
                "has_input_event",
                source,
                target,
                qualifier=f"{revision_id}:{edge_key}",
                properties=edge_properties,
                source_refs=edge_refs,
            )
        elif source_kind == "event" and target_kind == "event":
            # Some deterministic source tables encode a one-input gate as one
            # direct event edge.  Preserve that occurrence without merging the
            # events or guessing OR/AND semantics when the field is absent.
            gate_type = _declared_text(raw_properties.get("gate_type")) or "DIRECT"
            gate = builder.add_document_node(
                "fta_implicit_gate_occurrence",
                "FTAGate",
                f"{revision_id}:{edge_key}:implicit-gate",
                {
                    **common,
                    "name": f"{gate_type} 关系",
                    "gate_type": gate_type,
                    "source_edge_id": edge_key,
                    "implicit_from_direct_edge": True,
                },
                source_refs=edge_refs,
            )
            builder.add_edge(
                "decomposed_by",
                source,
                gate,
                qualifier=f"{revision_id}:{edge_key}:out",
                properties=edge_properties,
                source_refs=edge_refs,
            )
            builder.add_edge(
                "has_input_event",
                gate,
                target,
                qualifier=f"{revision_id}:{edge_key}:in",
                properties=edge_properties,
                source_refs=edge_refs,
            )
        else:
            builder.warning(
                "unsupported_fta_topology_edge",
                "FTA 正式修订包含非事件-逻辑门结构，未投影该边",
                record_id=f"{revision_id}:{edge_key}",
            )


def _build_rbd_revision(builder: _GraphBuilder, revision: Mapping[str, Any]) -> None:
    graph = _canonical_graph(revision)
    revision_id = clean_text(revision.get("revision_id"))
    diagram_id = clean_text(revision.get("diagram_id"))
    common = _diagram_revision_properties(revision)
    graph_refs = _diagram_source_refs(revision, graph)
    diagram_occurrence = builder.add_document_node(
        "rbd_diagram_occurrence",
        "RBDDiagramRef",
        f"{revision_id}:{diagram_id}",
        {
            **common,
            "name": clean_text(revision.get("title"))
            or clean_text(revision.get("diagram_key"))
            or diagram_id,
            "source_diagram_id": diagram_id,
        },
        source_refs=graph_refs,
    )
    for business_id in _diagram_business_ids(
        revision, graph, _RBD_DIAGRAM_ID_RE, diagram_type="rbd"
    ):
        target = builder.add_business_node(
            "rbd_diagram",
            "RBDDiagramRef",
            business_id,
            {"name": business_id},
            source_refs=graph_refs,
        )
        builder.add_edge(
            "references_rbd",
            diagram_occurrence,
            target,
            qualifier=f"{revision_id}:diagram:{business_id}",
            properties={
                **common,
                "record_key": diagram_id,
                "reference_id": business_id,
                "reference_kind": "diagram",
            },
            source_refs=graph_refs,
        )

    node_map: Dict[str, str] = {}
    source_labels: Dict[str, str] = {}
    start_ids: List[str] = []
    for index, item in enumerate(graph.get("vertices") or (), start=1):
        if not isinstance(item, Mapping):
            raise StructuredGraphError(f"RBD revision {revision_id} vertex {index} is invalid")
        source_node_id = clean_text(item.get("id"))
        if not source_node_id:
            raise StructuredGraphError(f"RBD revision {revision_id} vertex {index} has no id")
        source_label = clean_text(item.get("label"))
        item_refs = _diagram_source_refs(revision, graph, item)
        raw_properties = _safe_occurrence_properties(item.get("properties"))
        occurrence_label = "RBDBlock" if source_label == "ReliabilityBlock" else "RBDJunction"
        occurrence = builder.add_document_node(
            "rbd_block_occurrence" if occurrence_label == "RBDBlock" else "rbd_junction_occurrence",
            occurrence_label,
            f"{revision_id}:{source_node_id}",
            {
                **raw_properties,
                **common,
                "name": clean_text(raw_properties.get("name")) or source_node_id,
                "source_node_id": source_node_id,
                "block_type": source_label,
            },
            source_refs=item_refs,
        )
        node_map[source_node_id] = occurrence
        source_labels[source_node_id] = source_label
        if source_label == "Start":
            start_ids.append(source_node_id)

        module_id = _declared_text(raw_properties.get("module_id"))
        if module_id:
            module = builder.add_business_node(
                "component",
                "Component",
                module_id,
                {"name": module_id, "component_type": "module"},
                source_refs=item_refs,
            )
            builder.add_edge(
                "represents_module",
                occurrence,
                module,
                qualifier=f"{revision_id}:{source_node_id}:{module_id}",
                properties={
                    **common,
                    "record_key": source_node_id,
                    "reference_id": module_id,
                },
                source_refs=item_refs,
            )
        _add_function_bridges(
            builder,
            occurrence,
            (
                raw_properties.get("function_id"),
                raw_properties.get("function_ids"),
                raw_properties.get("linked_function"),
            ),
            record_key=source_node_id,
            revision_properties=common,
            source_refs=item_refs,
        )

    for start_id in start_ids:
        builder.add_edge(
            "references_rbd",
            node_map[start_id],
            diagram_occurrence,
            qualifier=f"{revision_id}:start:{start_id}",
            properties={**common, "record_key": start_id, "reference_kind": "diagram_occurrence"},
            source_refs=graph_refs,
        )

    for index, item in enumerate(graph.get("edges") or (), start=1):
        if not isinstance(item, Mapping):
            raise StructuredGraphError(f"RBD revision {revision_id} edge {index} is invalid")
        source_key = clean_text(item.get("source") or item.get("outV"))
        target_key = clean_text(item.get("target") or item.get("inV"))
        source = node_map.get(source_key)
        target = node_map.get(target_key)
        edge_key = clean_text(item.get("id")) or f"edge-{index}"
        if not source or not target:
            builder.warning(
                "diagram_edge_unknown_endpoint",
                "RBD 正式修订存在无法定位端点的边，未投影该边",
                record_id=f"{revision_id}:{edge_key}",
            )
            continue
        edge_refs = _diagram_source_refs(revision, graph, item)
        raw_properties = _safe_occurrence_properties(item.get("properties"))
        builder.add_edge(
            "next_block",
            source,
            target,
            qualifier=f"{revision_id}:{edge_key}",
            properties={
                **raw_properties,
                **common,
                "record_key": edge_key,
                "source_edge_id": edge_key,
                "source_relation": clean_text(item.get("label")),
                "source_block_type": source_labels.get(source_key, ""),
                "target_block_type": source_labels.get(target_key, ""),
            },
            source_refs=edge_refs,
        )


def build_formal_diagram_graph(
    doc_type: str,
    doc_id: str,
    revisions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the formal, cross-document view of active FTA/RBD revisions.

    Provisional and quarantined revisions remain available in their dedicated
    revision projection for browsing, but this common graph is the only one
    intended for cross-document traversal and therefore fails closed.
    """

    dtype = clean_text(doc_type).casefold()
    if dtype not in {"fta", "rbd"}:
        raise StructuredGraphError(f"unsupported diagram doc_type: {doc_type}")
    builder = _GraphBuilder(dtype, doc_id)
    for revision in _formal_revisions(builder, revisions):
        if dtype == "fta":
            _build_fta_revision(builder, revision)
        else:
            _build_rbd_revision(builder, revision)
    return builder.result()


def build_structured_graph(
    doc_type: str,
    doc_id: str,
    rows: Iterable[Mapping[str, Any]] | Mapping[str, Iterable[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Dispatch PostgreSQL rows using the project's five structured types.

    ``function_locator`` requires a mapping with ``components``, ``functions``,
    and ``interfaces`` sequences.  Other document types accept one row
    sequence.  This explicit shape prevents accidental name-based joins.
    """

    dtype = clean_text(doc_type).casefold()
    if dtype == "function_locator":
        if not isinstance(rows, Mapping):
            raise StructuredGraphError(
                "function_locator rows must map components/functions/interfaces"
            )
        return build_function_locator_graph(
            doc_id,
            components=rows.get("components") or (),
            functions=rows.get("functions") or (),
            interfaces=rows.get("interfaces") or (),
        )
    if isinstance(rows, Mapping):
        raise StructuredGraphError(f"{dtype} rows must be a sequence of mappings")
    builders = {
        "hmc": build_hmc_graph,
        "fmeca": build_fmeca_graph,
        "mfl": build_mfl_graph,
        "fault_history": build_fault_history_graph,
    }
    builder = builders.get(dtype)
    if builder is None:
        raise StructuredGraphError(f"unsupported doc_type: {doc_type}")
    return builder(doc_id, rows)


__all__ = [
    "SUPPORTED_DOC_TYPES",
    "StructuredGraphError",
    "business_node_id",
    "build_fault_history_graph",
    "build_fmeca_graph",
    "build_formal_diagram_graph",
    "build_function_locator_graph",
    "build_hmc_graph",
    "build_mfl_graph",
    "build_structured_graph",
    "clean_text",
    "document_text_node_id",
]
