"""ICP cross-document entity identity and relation assertion authority.

PostgreSQL owns four deliberately separate concepts:

* canonical entities: stable business identities, or document-scoped text
  identities when no business identifier exists;
* mentions: what an individual source field actually said;
* resolution candidates: similarity-based suggestions that never resolve a
  mention by themselves;
* relation assertions: source-backed edges between canonical entities.

The separation is a safety boundary.  A business identifier or an explicit
source reference may resolve a mention across documents.  Lexical, embedding,
context, and graph-structure similarity may only create a candidate.  This
module therefore has no API that promotes a similarity candidate to a formal
resolution.

The connection is injectable and otherwise resolved lazily from the existing
``postgres_client``.  Unit tests do not need a running PostgreSQL instance.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
import uuid
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


CORE_ENTITY_TYPES = frozenset(
    {
        "System",
        "Subsystem",
        "Module",
        "Component",
        "Function",
        "Interface",
        "HMCCode",
        "FMECARecord",
        "MFLRecord",
        "FaultMode",
        "Symptom",
        "FaultCause",
        "FaultEffect",
        "DetectionMethod",
        "MaintenanceAction",
        "API",
        "FTAEvent",
        "FTAGate",
        "FTADiagram",
        "RBDBlock",
        "RBDJunction",
        "RBDDiagram",
        "FaultHistoryCase",
        "DiagnosticStep",
        "Document",
        "SourceFragment",
    }
)

CORE_RELATION_TYPES = frozenset(
    {
        "contains",
        "implements",
        "exposes",
        "connects_to",
        "input_from",
        "output_to",
        "has_fault_mode",
        "manifests_as",
        "caused_by",
        "results_in",
        "affects",
        "detected_by",
        "repaired_by",
        "describes",
        "recommends",
        "analyzes",
        "locates",
        "references_api",
        "observed",
        "confirmed_cause",
        "resolved_by",
        "references_hmc",
        "references_fmeca",
        "references_fta",
        "references_rbd",
        "references_mfl",
        "has_diagnosis_step",
        "has_step_content",
        "next_step",
        "decomposed_by",
        "has_input_event",
        "represents_fault_mode",
        "next_block",
        "represents_module",
        "implements_function",
        "supported_by",
        "extracted_from",
    }
)

FORMAL_RESOLUTION_METHODS = frozenset({"business_id", "explicit_reference"})
RELATION_ASSERTION_METHODS = frozenset(
    {
        "explicit_field",
        "explicit_code_reference",
        "deterministic_structure",
        "ordered_step",
        "auto_validated_diagram",
        "manual_confirmed_diagram",
    }
)
SIMILARITY_METHODS = frozenset(
    {"lexical", "embedding", "hybrid_similarity", "contextual_similarity"}
)

_JSON_FIELDS = frozenset(
    {
        "identity_scope",
        "logical_ids",
        "attributes",
        "scope",
        "resolution_evidence",
        "candidate_metadata",
        "occurrence_qualifiers",
        "logical_edge_ids",
        "provenance",
    }
)
_SCORE_WEIGHTS = {
    "lexical_score": 0.15,
    "embedding_score": 0.45,
    "context_score": 0.25,
    "structure_score": 0.15,
}

# Adapter from the existing deterministic ``build_document_graph`` contract to
# canonical ontology names.  Unknown textual values remain SourceFragment;
# they are never promoted to a more specific type by semantic guessing.
_STRUCTURED_LABEL_TYPES = {
    "System": "System",
    "Subsystem": "Subsystem",
    "Module": "Module",
    "Component": "Component",
    "Function": "Function",
    "Interface": "Interface",
    "HMCCode": "HMCCode",
    "FMECARecord": "FMECARecord",
    "MFLRecord": "MFLRecord",
    "API": "API",
    "FaultHistoryCase": "FaultHistoryCase",
    "FaultHistoryStep": "DiagnosticStep",
    "FTAEvent": "FTAEvent",
    "FTAGate": "FTAGate",
    "FTADiagramRef": "FTADiagram",
    "RBDBlock": "RBDBlock",
    "RBDJunction": "RBDJunction",
    "RBDDiagramRef": "RBDDiagram",
    "Document": "Document",
    "SourceFragment": "SourceFragment",
}

_TEXT_KIND_TYPES = {
    "function_locator_inputs": "Interface",
    "function_locator_outputs": "Interface",
    "hmc_fault_desc": "Symptom",
    "hmc_repair_method": "MaintenanceAction",
    "fmeca_func_management": "Function",
    "fmeca_fault_mode": "FaultMode",
    "fmeca_fault_cause": "FaultCause",
    "fmeca_local_impact": "FaultEffect",
    "fmeca_final_impact": "FaultEffect",
    "fmeca_detect_method": "DetectionMethod",
    "fmeca_design_improve": "MaintenanceAction",
    "fmeca_compensate_action": "MaintenanceAction",
    "mfl_module_fault": "FaultMode",
    "mfl_sur_fault": "FaultMode",
    "mfl_subcircuit_fault": "FaultMode",
    "mfl_maintenance_action": "MaintenanceAction",
    "fault_history_fault_symptom": "Symptom",
    "fault_history_related_module": "Module",
    "fault_history_related_function": "Function",
    "fault_history_diagnosis_method": "DetectionMethod",
    "fault_history_root_cause": "FaultCause",
    "fault_history_final_solution": "MaintenanceAction",
    "fault_history_preventive_action": "MaintenanceAction",
}

_STRUCTURED_RELATION_TYPES = {
    "contains_component": "contains",
    "contains_function": "contains",
    "provides_function": "implements",
    "has_input": "input_from",
    "has_output": "output_to",
    "interface_source": "exposes",
    "interface_target": "connects_to",
    "describes_fault": "describes",
    "has_repair_method": "recommends",
    "has_product_flag": "describes",
    "has_function_management": "analyzes",
    "has_fault_mode": "analyzes",
    "has_fault_cause": "analyzes",
    "applies_in_mission_phase": "describes",
    "has_local_impact": "analyzes",
    "has_final_impact": "analyzes",
    "has_detection_method": "analyzes",
    "has_design_improvement": "recommends",
    "has_compensation_action": "recommends",
    "has_module_fault": "describes",
    "has_sur_fault": "describes",
    "has_subcircuit_fault": "describes",
    "has_maintenance_action": "recommends",
    "reports_via_api": "references_api",
    "has_api_method": "describes",
    "has_api_usage": "describes",
    "has_fault_symptom": "observed",
    "mentions_module": "locates",
    "mentions_function": "describes",
    "has_diagnosis_method": "describes",
    "has_recorded_root_cause": "confirmed_cause",
    "occurred_in_mission_phase": "describes",
    "has_final_solution": "resolved_by",
    "has_verification_result": "describes",
    "has_preventive_action": "recommends",
    "references_hmc": "references_hmc",
    "references_fmeca": "references_fmeca",
    "references_fta": "references_fta",
    "references_rbd": "references_rbd",
    "references_mfl": "references_mfl",
    "has_diagnosis_step": "has_diagnosis_step",
    "has_step_content": "has_step_content",
    "next_step": "next_step",
    "decomposed_by": "decomposed_by",
    "has_input_event": "has_input_event",
    "represents_fault_mode": "represents_fault_mode",
    "next_block": "next_block",
    "represents_module": "represents_module",
    "implements_function": "implements_function",
    "extracted_from": "extracted_from",
}


class EntityResolutionError(RuntimeError):
    """Base persistence or invariant error."""


class EntityNotFoundError(EntityResolutionError):
    pass


class UnsafeResolutionError(EntityResolutionError):
    """Raised when a caller tries to turn similarity into identity."""


def clean_text(value: Any) -> str:
    """Normalize Unicode and whitespace without changing domain punctuation."""

    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def normalize_business_id(value: Any) -> str:
    """Apply only NFKC/whitespace cleanup to an explicit business code."""

    # Case, punctuation, internal spacing, wildcards, and leading zeroes remain
    # significant.  Type-specific parsers (for example HMC) own code assembly
    # before this identity layer is called.
    return clean_text(value)


def normalize_scope(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a JSON-stable applicability scope used by identity keys."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                clean_text(key): normalize(val)
                for key, val in sorted(item.items(), key=lambda pair: clean_text(pair[0]))
                if clean_text(key) and val is not None and normalize(val) not in ("", [], {})
            }
        if isinstance(item, (list, tuple, set, frozenset)):
            normalized = [normalize(entry) for entry in item]
            return [entry for entry in normalized if entry not in ("", [], {})]
        if isinstance(item, str):
            return clean_text(item)
        return item

    normalized = normalize(dict(value or {}))
    return normalized if isinstance(normalized, dict) else {}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _structured_entity_type(vertex: Mapping[str, Any]) -> str:
    label = clean_text(vertex.get("label"))
    properties = vertex.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    if label == "Component":
        component_type = clean_text(properties.get("component_type")).casefold()
        return {
            "system": "System",
            "subsystem": "Subsystem",
            "module": "Module",
            "component": "Component",
        }.get(component_type, "Component")
    if label == "TextValue":
        return _TEXT_KIND_TYPES.get(
            clean_text(properties.get("value_kind")).casefold(), "SourceFragment"
        )
    return _STRUCTURED_LABEL_TYPES.get(label, "SourceFragment")


def _identity_scope_from_metadata(
    document_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select applicability fields; file names and timestamps are not identity."""

    metadata = dict(document_metadata or {})
    aliases = {
        "system_code": ("system_code", "system_id", "applicable_system"),
        "configuration": ("configuration", "config", "applicable_configuration"),
        "model": ("model", "aircraft_model", "applicable_model"),
    }
    selected: Dict[str, Any] = {}
    for canonical, keys in aliases.items():
        for key in keys:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                selected[canonical] = value
                break
    return normalize_scope(selected)


def business_entity_key(
    entity_type: str,
    business_id: str,
    scope: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the canonical key for an exact, scoped business identifier."""

    dtype = clean_text(entity_type)
    code = normalize_business_id(business_id)
    if not dtype or not code:
        raise ValueError("entity_type and business_id are required")
    return _digest(
        {"entity_type": dtype, "business_id": code, "scope": normalize_scope(scope)}
    )


def document_entity_key(
    doc_id: str,
    entity_type: str,
    field_type: str,
    raw_text: str,
) -> str:
    """Build a non-global identity for text lacking a business identifier.

    ``doc_id + field_type + raw_text`` is intentionally part of the key.  Two
    similar descriptions in different documents remain different entities.
    """

    document_id = clean_text(doc_id)
    dtype = clean_text(entity_type)
    field = clean_text(field_type)
    text = clean_text(raw_text)
    if not document_id or not dtype or not field or not text:
        raise ValueError("doc_id, entity_type, field_type and raw_text are required")
    return _digest(
        {
            "doc_id": document_id,
            "entity_type": dtype,
            "field_type": field,
            "raw_text": text,
        }
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Pure cosine score for precomputed vectors; no model call is performed."""

    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal length")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(math.isfinite(value) for value in [*left_values, *right_values]):
        raise ValueError("embedding vectors must contain only finite values")
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding vectors must not be zero vectors")
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _connection_locked(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize access to the repository's shared DB-API connection.

    The project-wide PostgreSQL client exposes one connection.  DB-API
    connections and their transaction state must not be interleaved by two
    document jobs, so every repository entry point that can touch the
    connection holds the same re-entrant lock for its complete operation.
    Re-entrancy is required because higher-level sync methods call the
    lower-level upsert/query methods.
    """

    @wraps(method)
    def locked(self: "EntityResolutionRepository", *args: Any, **kwargs: Any) -> Any:
        with self._connection_lock:
            return method(self, *args, **kwargs)

    return locked


class EntityResolutionRepository:
    """Transaction-oriented repository for safe entity resolution."""

    def __init__(
        self,
        connection: Any = None,
        *,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._connection = connection
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._initialized = False
        self._init_lock = threading.Lock()
        self._transaction_state = threading.local()
        self._connection_lock = threading.RLock()

    @property
    def connection(self):
        if self._connection is None:
            from backend.storage.postgres_client import postgres_client

            self._connection = postgres_client.conn
        return self._connection

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @staticmethod
    def _column_name(description_item: Any) -> str:
        if hasattr(description_item, "name"):
            return str(description_item.name)
        return str(description_item[0])

    @classmethod
    def _row_to_dict(cls, cursor: Any, row: Any) -> Dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, Mapping):
            result = dict(row)
        else:
            description = getattr(cursor, "description", None) or []
            result = {
                cls._column_name(item): item_value
                for item, item_value in zip(description, row)
            }
        for key in _JSON_FIELDS:
            value = result.get(key)
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except (TypeError, ValueError):
                    pass
        return result

    @classmethod
    def _rows_to_dicts(cls, cursor: Any, rows: Iterable[Any]) -> List[Dict[str, Any]]:
        return [cls._row_to_dict(cursor, row) for row in rows]

    def _commit(self) -> None:
        if int(getattr(self._transaction_state, "depth", 0) or 0) == 0:
            self.connection.commit()

    def _rollback(self) -> None:
        if int(getattr(self._transaction_state, "depth", 0) or 0) > 0:
            return
        try:
            self.connection.rollback()
        except Exception:
            pass

    def _run_atomically(self, operation: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Run a document snapshot with one outer commit/rollback boundary."""

        depth = int(getattr(self._transaction_state, "depth", 0) or 0)
        self._transaction_state.depth = depth + 1
        try:
            result = operation()
        except Exception:
            self._transaction_state.depth = depth
            if depth == 0:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            raise
        self._transaction_state.depth = depth
        if depth == 0:
            try:
                self.connection.commit()
            except Exception:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
                raise
        return result

    @staticmethod
    def _validate_entity_type(entity_type: str) -> str:
        value = clean_text(entity_type)
        if value not in CORE_ENTITY_TYPES:
            raise ValueError(f"unsupported entity_type: {entity_type}")
        return value

    @staticmethod
    def _validate_relation_type(relation_type: str) -> str:
        value = clean_text(relation_type).casefold()
        if value not in CORE_RELATION_TYPES:
            raise ValueError(f"unsupported relation_type: {relation_type}")
        return value

    @staticmethod
    def _validate_score(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        score = float(value)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return score

    @classmethod
    def calculate_candidate_score(
        cls,
        *,
        lexical_score: Optional[float] = None,
        embedding_score: Optional[float] = None,
        context_score: Optional[float] = None,
        structure_score: Optional[float] = None,
    ) -> float:
        """Combine supplied discovery scores; this is not an identity verdict."""

        supplied = {
            "lexical_score": cls._validate_score(lexical_score, "lexical_score"),
            "embedding_score": cls._validate_score(embedding_score, "embedding_score"),
            "context_score": cls._validate_score(context_score, "context_score"),
            "structure_score": cls._validate_score(structure_score, "structure_score"),
        }
        denominator = sum(
            _SCORE_WEIGHTS[key] for key, value in supplied.items() if value is not None
        )
        if denominator <= 0:
            raise ValueError("at least one similarity score is required")
        score = sum(
            _SCORE_WEIGHTS[key] * float(value)
            for key, value in supplied.items()
            if value is not None
        ) / denominator
        return round(score, 6)

    @_connection_locked
    def init_tables(self) -> None:
        """Create the entity authority schema idempotently."""

        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            statements = [
                """
                CREATE TABLE IF NOT EXISTS icp_canonical_entities (
                    entity_id          VARCHAR(96) PRIMARY KEY,
                    entity_type        VARCHAR(48) NOT NULL,
                    canonical_key      CHAR(64) NOT NULL,
                    canonical_name     TEXT NOT NULL,
                    business_id        VARCHAR(300),
                    identity_method    VARCHAR(32) NOT NULL
                                       CHECK (identity_method IN ('business_id','document_scoped')),
                    identity_scope     JSONB NOT NULL DEFAULT '{}'::jsonb,
                    logical_ids         JSONB NOT NULL DEFAULT '[]'::jsonb,
                    attributes         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    origin_doc_id       VARCHAR(36)
                                        REFERENCES icp_documents(doc_id) ON DELETE SET NULL,
                    lifecycle_status   VARCHAR(20) NOT NULL DEFAULT 'active'
                                       CHECK (lifecycle_status IN ('active','quarantined','retired')),
                    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (entity_type, canonical_key),
                    CONSTRAINT ck_icp_entity_identity CHECK (
                        identity_method <> 'business_id'
                        OR business_id IS NOT NULL
                    )
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_entity_mentions (
                    mention_id          VARCHAR(96) PRIMARY KEY,
                    mention_key         CHAR(64) NOT NULL,
                    doc_id              VARCHAR(36) NOT NULL
                                        REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
                    record_id           VARCHAR(300),
                    source_ref          TEXT,
                    field_type          VARCHAR(100) NOT NULL,
                    entity_type         VARCHAR(48) NOT NULL,
                    raw_text            TEXT NOT NULL,
                    normalized_text     TEXT NOT NULL,
                    business_id         VARCHAR(300),
                    scope               JSONB NOT NULL DEFAULT '{}'::jsonb,
                    attributes          JSONB NOT NULL DEFAULT '{}'::jsonb,
                    resolved_entity_id  VARCHAR(96)
                                        REFERENCES icp_canonical_entities(entity_id) ON DELETE SET NULL,
                    resolution_status   VARCHAR(40) NOT NULL DEFAULT 'unresolved'
                                        CHECK (resolution_status IN (
                                            'unresolved','document_scoped',
                                            'resolved_business_id',
                                            'resolved_explicit_reference',
                                            'quarantined'
                                        )),
                    resolution_method   VARCHAR(40),
                    resolution_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
                    resolved_at         TIMESTAMP,
                    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (doc_id, mention_key)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_entity_resolution_candidates (
                    candidate_id        VARCHAR(96) PRIMARY KEY,
                    mention_id          VARCHAR(96) NOT NULL
                                        REFERENCES icp_entity_mentions(mention_id) ON DELETE CASCADE,
                    candidate_entity_id VARCHAR(96) NOT NULL
                                        REFERENCES icp_canonical_entities(entity_id) ON DELETE CASCADE,
                    resolution_method   VARCHAR(40) NOT NULL
                                        CHECK (resolution_method IN (
                                            'lexical','embedding','hybrid_similarity',
                                            'contextual_similarity'
                                        )),
                    lexical_score       DOUBLE PRECISION CHECK (lexical_score BETWEEN 0 AND 1),
                    embedding_score     DOUBLE PRECISION CHECK (embedding_score BETWEEN 0 AND 1),
                    context_score       DOUBLE PRECISION CHECK (context_score BETWEEN 0 AND 1),
                    structure_score     DOUBLE PRECISION CHECK (structure_score BETWEEN 0 AND 1),
                    combined_score      DOUBLE PRECISION NOT NULL
                                        CHECK (combined_score BETWEEN 0 AND 1),
                    candidate_status    VARCHAR(24) NOT NULL DEFAULT 'candidate_semantic'
                                        CHECK (candidate_status IN (
                                            'candidate_semantic','rejected','quarantined'
                                        )),
                    model_version       VARCHAR(200) NOT NULL DEFAULT '',
                    candidate_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (
                        mention_id, candidate_entity_id,
                        resolution_method, model_version
                    )
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_relation_assertions (
                    assertion_id        VARCHAR(96) PRIMARY KEY,
                    doc_id              VARCHAR(36) NOT NULL
                                        REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
                    relation_type       VARCHAR(64) NOT NULL,
                    source_entity_id    VARCHAR(96) NOT NULL
                                        REFERENCES icp_canonical_entities(entity_id) ON DELETE CASCADE,
                    target_entity_id    VARCHAR(96) NOT NULL
                                        REFERENCES icp_canonical_entities(entity_id) ON DELETE CASCADE,
                    assertion_method    VARCHAR(40) NOT NULL
                                        CHECK (assertion_method IN (
                                            'explicit_field','explicit_code_reference',
                                            'deterministic_structure','ordered_step',
                                            'auto_validated_diagram','manual_confirmed_diagram'
                                        )),
                    assertion_status    VARCHAR(24) NOT NULL DEFAULT 'asserted'
                                        CHECK (assertion_status IN ('asserted','quarantined')),
                    record_id           VARCHAR(300),
                    source_ref          TEXT,
                    evidence_text       TEXT,
                    logical_edge_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
                    occurrence_qualifiers JSONB NOT NULL DEFAULT '[]'::jsonb,
                    provenance          JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (source_entity_id <> target_entity_id),
                    UNIQUE (doc_id, relation_type, source_entity_id, target_entity_id)
                )
                """,
                """
                CREATE OR REPLACE FUNCTION icp_purge_orphan_entity_after_delete()
                RETURNS TRIGGER
                LANGUAGE plpgsql
                AS $$
                DECLARE
                    affected_entity_ids VARCHAR(96)[];
                BEGIN
                    IF TG_TABLE_NAME = 'icp_entity_mentions' THEN
                        affected_entity_ids := ARRAY[OLD.resolved_entity_id];
                    ELSE
                        affected_entity_ids := ARRAY[
                            OLD.source_entity_id, OLD.target_entity_id
                        ];
                    END IF;

                    DELETE FROM icp_canonical_entities AS entity
                    WHERE entity.entity_id = ANY(affected_entity_ids)
                      AND NOT EXISTS (
                          SELECT 1 FROM icp_entity_mentions AS mention
                          WHERE mention.resolved_entity_id=entity.entity_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM icp_relation_assertions AS assertion
                          WHERE assertion.source_entity_id=entity.entity_id
                             OR assertion.target_entity_id=entity.entity_id
                      );
                    RETURN OLD;
                END
                $$
                """,
                "DROP TRIGGER IF EXISTS trg_icp_purge_entity_after_mention_delete ON icp_entity_mentions",
                """
                CREATE TRIGGER trg_icp_purge_entity_after_mention_delete
                AFTER DELETE ON icp_entity_mentions
                FOR EACH ROW EXECUTE FUNCTION icp_purge_orphan_entity_after_delete()
                """,
                "DROP TRIGGER IF EXISTS trg_icp_purge_entity_after_assertion_delete ON icp_relation_assertions",
                """
                CREATE TRIGGER trg_icp_purge_entity_after_assertion_delete
                AFTER DELETE ON icp_relation_assertions
                FOR EACH ROW EXECUTE FUNCTION icp_purge_orphan_entity_after_delete()
                """,
                "ALTER TABLE icp_canonical_entities ADD COLUMN IF NOT EXISTS logical_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
                "ALTER TABLE icp_relation_assertions ADD COLUMN IF NOT EXISTS logical_edge_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
                """
                DO $$
                DECLARE unsafe_constraint RECORD;
                BEGIN
                    FOR unsafe_constraint IN
                        SELECT conname
                        FROM pg_constraint
                        WHERE conrelid='icp_canonical_entities'::regclass
                          AND contype='c'
                          AND pg_get_constraintdef(oid) ILIKE '%origin_doc_id%'
                          AND pg_get_constraintdef(oid) ILIKE '%identity_method%'
                    LOOP
                        EXECUTE format(
                            'ALTER TABLE icp_canonical_entities DROP CONSTRAINT %I',
                            unsafe_constraint.conname
                        );
                    END LOOP;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid='icp_canonical_entities'::regclass
                          AND conname='ck_icp_entity_identity'
                    ) THEN
                        ALTER TABLE icp_canonical_entities
                        ADD CONSTRAINT ck_icp_entity_identity CHECK (
                            identity_method <> 'business_id'
                            OR business_id IS NOT NULL
                        );
                    END IF;
                END $$
                """,
                "CREATE INDEX IF NOT EXISTS idx_icp_entities_business ON icp_canonical_entities(entity_type, business_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_entities_name ON icp_canonical_entities(entity_type, canonical_name)",
                "CREATE INDEX IF NOT EXISTS idx_icp_entities_logical_ids ON icp_canonical_entities USING GIN(logical_ids)",
                "CREATE INDEX IF NOT EXISTS idx_icp_mentions_doc ON icp_entity_mentions(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_mentions_business ON icp_entity_mentions(entity_type, business_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_mentions_resolution ON icp_entity_mentions(resolution_status)",
                "CREATE INDEX IF NOT EXISTS idx_icp_candidates_mention_score ON icp_entity_resolution_candidates(mention_id, combined_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_icp_relations_source ON icp_relation_assertions(source_entity_id, relation_type)",
                "CREATE INDEX IF NOT EXISTS idx_icp_relations_target ON icp_relation_assertions(target_entity_id, relation_type)",
                "CREATE INDEX IF NOT EXISTS idx_icp_relations_doc ON icp_relation_assertions(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_relations_logical_edges ON icp_relation_assertions USING GIN(logical_edge_ids)",
            ]
            try:
                with self.connection.cursor() as cursor:
                    for statement in statements:
                        cursor.execute(statement)
                self._commit()
                self._initialized = True
            except Exception:
                self._rollback()
                raise

    @_connection_locked
    def upsert_business_entity(
        self,
        *,
        entity_type: str,
        business_id: str,
        canonical_name: str = "",
        scope: Optional[Mapping[str, Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        origin_doc_id: Optional[str] = None,
        logical_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create/reuse a canonical entity based only on an exact business ID."""

        self.init_tables()
        dtype = self._validate_entity_type(entity_type)
        code = normalize_business_id(business_id)
        normalized_scope = normalize_scope(scope)
        if not code:
            raise ValueError("business_id is required")
        canonical_key = business_entity_key(dtype, code, normalized_scope)
        entity_id = f"ent-{canonical_key[:32]}"
        name = clean_text(canonical_name) or code
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_canonical_entities
                        (entity_id, entity_type, canonical_key, canonical_name,
                         business_id, identity_method, identity_scope, logical_ids,
                         attributes, origin_doc_id)
                    VALUES (%s,%s,%s,%s,%s,'business_id',%s::jsonb,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT (entity_type, canonical_key) DO UPDATE SET
                        canonical_name = CASE
                            WHEN EXCLUDED.canonical_name <> ''
                            THEN EXCLUDED.canonical_name
                            ELSE icp_canonical_entities.canonical_name
                        END,
                        logical_ids = (
                            SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
                            FROM jsonb_array_elements(
                                icp_canonical_entities.logical_ids || EXCLUDED.logical_ids
                            ) AS items(item)
                        ),
                        attributes = icp_canonical_entities.attributes || EXCLUDED.attributes,
                        origin_doc_id = COALESCE(
                            icp_canonical_entities.origin_doc_id,
                            EXCLUDED.origin_doc_id
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        entity_id,
                        dtype,
                        canonical_key,
                        name,
                        code,
                        self._json(normalized_scope),
                        self._json([clean_text(logical_id)] if clean_text(logical_id) else []),
                        self._json(dict(attributes or {})),
                        clean_text(origin_doc_id) or None,
                    ),
                )
                result = self._row_to_dict(cursor, cursor.fetchone())
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def upsert_document_scoped_entity(
        self,
        *,
        doc_id: str,
        entity_type: str,
        field_type: str,
        raw_text: str,
        canonical_name: str = "",
        scope: Optional[Mapping[str, Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        logical_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist uncoded text without merging it across documents."""

        self.init_tables()
        document_id = clean_text(doc_id)
        dtype = self._validate_entity_type(entity_type)
        field = clean_text(field_type)
        text = clean_text(raw_text)
        canonical_key = document_entity_key(document_id, dtype, field, text)
        entity_id = f"ent-{canonical_key[:32]}"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_canonical_entities
                        (entity_id, entity_type, canonical_key, canonical_name,
                         business_id, identity_method, identity_scope, logical_ids,
                         attributes, origin_doc_id)
                    VALUES (%s,%s,%s,%s,NULL,'document_scoped',%s::jsonb,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT (entity_type, canonical_key) DO UPDATE SET
                        logical_ids = (
                            SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
                            FROM jsonb_array_elements(
                                icp_canonical_entities.logical_ids || EXCLUDED.logical_ids
                            ) AS items(item)
                        ),
                        attributes = icp_canonical_entities.attributes || EXCLUDED.attributes,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        entity_id,
                        dtype,
                        canonical_key,
                        clean_text(canonical_name) or text,
                        self._json(normalize_scope(scope)),
                        self._json([clean_text(logical_id)] if clean_text(logical_id) else []),
                        self._json(dict(attributes or {})),
                        document_id,
                    ),
                )
                result = self._row_to_dict(cursor, cursor.fetchone())
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def upsert_mention(
        self,
        *,
        doc_id: str,
        entity_type: str,
        field_type: str,
        raw_text: str,
        record_id: Optional[str] = None,
        source_ref: Optional[str] = None,
        business_id: Optional[str] = None,
        scope: Optional[Mapping[str, Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a source mention without making an identity decision."""

        self.init_tables()
        document_id = clean_text(doc_id)
        dtype = self._validate_entity_type(entity_type)
        field = clean_text(field_type)
        text = clean_text(raw_text)
        if not document_id or not field or not text:
            raise ValueError("doc_id, field_type and raw_text are required")
        code = normalize_business_id(business_id) or None
        normalized_scope = normalize_scope(scope)
        mention_key = _digest(
            {
                "doc_id": document_id,
                "record_id": clean_text(record_id),
                "source_ref": clean_text(source_ref),
                "field_type": field,
                "entity_type": dtype,
                "raw_text": text,
            }
        )
        mention_id = self._id_factory()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_entity_mentions
                        (mention_id, mention_key, doc_id, record_id, source_ref,
                         field_type, entity_type, raw_text, normalized_text,
                         business_id, scope, attributes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    ON CONFLICT (doc_id, mention_key) DO UPDATE SET
                        record_id = COALESCE(EXCLUDED.record_id, icp_entity_mentions.record_id),
                        source_ref = COALESCE(EXCLUDED.source_ref, icp_entity_mentions.source_ref),
                        business_id = COALESCE(EXCLUDED.business_id, icp_entity_mentions.business_id),
                        scope = EXCLUDED.scope,
                        attributes = icp_entity_mentions.attributes || EXCLUDED.attributes,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        mention_id,
                        mention_key,
                        document_id,
                        clean_text(record_id) or None,
                        clean_text(source_ref) or None,
                        field,
                        dtype,
                        text,
                        clean_text(text).casefold(),
                        code,
                        self._json(normalized_scope),
                        self._json(dict(attributes or {})),
                    ),
                )
                result = self._row_to_dict(cursor, cursor.fetchone())
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def _resolve_mention(
        self,
        *,
        mention_id: str,
        entity_id: str,
        method: str,
        reference_value: str = "",
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if method not in FORMAL_RESOLUTION_METHODS:
            raise UnsafeResolutionError(
                "formal resolution requires an exact business ID or explicit reference"
            )
        status = (
            "resolved_business_id"
            if method == "business_id"
            else "resolved_explicit_reference"
        )
        guard_params: List[Any] = []
        if method == "business_id":
            identity_guard = (
                "AND mention.business_id = entity.business_id "
                "AND mention.scope = entity.identity_scope"
            )
        else:
            reference = normalize_business_id(reference_value)
            if not reference:
                raise ValueError("reference_value is required for explicit resolution")
            identity_guard = (
                "AND entity.business_id = %s "
                "AND mention.scope = entity.identity_scope"
            )
            guard_params.append(reference)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE icp_entity_mentions AS mention SET
                        resolved_entity_id = entity.entity_id,
                        resolution_status = %s,
                        resolution_method = %s,
                        resolution_evidence = %s::jsonb,
                        resolved_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    FROM icp_canonical_entities AS entity
                    WHERE mention.mention_id = %s
                      AND entity.entity_id = %s
                      AND entity.entity_type = mention.entity_type
                      AND entity.lifecycle_status = 'active'
                      {identity_guard}
                    RETURNING mention.*
                    """,
                    tuple(
                        [
                        status,
                        method,
                        self._json(dict(evidence or {})),
                        clean_text(mention_id),
                        clean_text(entity_id),
                        *guard_params,
                        ]
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise EntityNotFoundError(
                        "mention/entity not found, inactive, or entity types differ"
                    )
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def resolve_mention_by_business_id(
        self,
        *,
        mention_id: str,
        entity_type: str,
        business_id: str,
        canonical_name: str = "",
        scope: Optional[Mapping[str, Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        origin_doc_id: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve using an exact normalized code; never by approximate text."""

        entity = self.upsert_business_entity(
            entity_type=entity_type,
            business_id=business_id,
            canonical_name=canonical_name,
            scope=scope,
            attributes=attributes,
            origin_doc_id=origin_doc_id,
        )
        return self._resolve_mention(
            mention_id=mention_id,
            entity_id=str(entity.get("entity_id") or ""),
            method="business_id",
            evidence={
                "business_id": normalize_business_id(business_id),
                **dict(evidence or {}),
            },
        )

    @_connection_locked
    def resolve_mention_by_explicit_reference(
        self,
        *,
        mention_id: str,
        entity_id: str,
        reference_value: str,
        source_ref: str = "",
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve when the source explicitly names another entity/code."""

        reference = clean_text(reference_value)
        if not reference:
            raise ValueError("reference_value is required")
        return self._resolve_mention(
            mention_id=mention_id,
            entity_id=entity_id,
            method="explicit_reference",
            reference_value=reference,
            evidence={
                "reference_value": reference,
                "source_ref": clean_text(source_ref),
                **dict(evidence or {}),
            },
        )

    @_connection_locked
    def resolve_mention_document_scoped(
        self,
        *,
        mention_id: str,
        entity_id: str,
    ) -> Dict[str, Any]:
        """Link uncoded text only to an entity owned by the same document."""

        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE icp_entity_mentions AS mention SET
                        resolved_entity_id = entity.entity_id,
                        resolution_status = 'document_scoped',
                        resolution_method = 'document_scoped',
                        resolution_evidence = jsonb_build_object(
                            'doc_id', mention.doc_id,
                            'field_type', mention.field_type,
                            'raw_text', mention.raw_text
                        ),
                        resolved_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    FROM icp_canonical_entities AS entity
                    WHERE mention.mention_id = %s
                      AND entity.entity_id = %s
                      AND entity.entity_type = mention.entity_type
                      AND entity.identity_method = 'document_scoped'
                      AND entity.origin_doc_id = mention.doc_id
                    RETURNING mention.*
                    """,
                    (clean_text(mention_id), clean_text(entity_id)),
                )
                row = cursor.fetchone()
                if row is None:
                    raise EntityNotFoundError(
                        "document-scoped mention/entity mismatch or record not found"
                    )
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def add_similarity_candidate(
        self,
        *,
        mention_id: str,
        candidate_entity_id: str,
        resolution_method: str = "hybrid_similarity",
        lexical_score: Optional[float] = None,
        embedding_score: Optional[float] = None,
        mention_embedding: Optional[Sequence[float]] = None,
        candidate_embedding: Optional[Sequence[float]] = None,
        context_score: Optional[float] = None,
        structure_score: Optional[float] = None,
        combined_score: Optional[float] = None,
        model_version: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a similarity candidate without modifying mention resolution."""

        self.init_tables()
        method = clean_text(resolution_method).casefold()
        if method not in SIMILARITY_METHODS:
            raise ValueError(f"unsupported similarity method: {resolution_method}")
        if (mention_embedding is None) != (candidate_embedding is None):
            raise ValueError("both embedding vectors must be supplied together")
        if embedding_score is None and mention_embedding is not None:
            # Candidate scores use [0, 1].  Negative cosine means no semantic
            # support; it is clamped to zero rather than treated as identity.
            embedding_score = max(
                0.0, cosine_similarity(mention_embedding, candidate_embedding or ())
            )
        scores = {
            "lexical_score": self._validate_score(lexical_score, "lexical_score"),
            "embedding_score": self._validate_score(embedding_score, "embedding_score"),
            "context_score": self._validate_score(context_score, "context_score"),
            "structure_score": self._validate_score(structure_score, "structure_score"),
        }
        calculated = self.calculate_candidate_score(**scores)
        if combined_score is not None:
            supplied_combined = self._validate_score(combined_score, "combined_score")
            assert supplied_combined is not None
            calculated = supplied_combined
        candidate_id = self._id_factory()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_entity_resolution_candidates
                        (candidate_id, mention_id, candidate_entity_id,
                         resolution_method, lexical_score, embedding_score,
                         context_score, structure_score, combined_score,
                         candidate_status, model_version, candidate_metadata)
                    SELECT %s, mention.mention_id, entity.entity_id,
                           %s,%s,%s,%s,%s,%s,'candidate_semantic',%s,%s::jsonb
                    FROM icp_entity_mentions AS mention
                    JOIN icp_canonical_entities AS entity
                      ON entity.entity_id = %s
                     AND entity.entity_type = mention.entity_type
                     AND entity.lifecycle_status = 'active'
                    WHERE mention.mention_id = %s
                    ON CONFLICT (
                        mention_id, candidate_entity_id,
                        resolution_method, model_version
                    ) DO UPDATE SET
                        lexical_score = EXCLUDED.lexical_score,
                        embedding_score = EXCLUDED.embedding_score,
                        context_score = EXCLUDED.context_score,
                        structure_score = EXCLUDED.structure_score,
                        combined_score = EXCLUDED.combined_score,
                        candidate_status = 'candidate_semantic',
                        candidate_metadata = EXCLUDED.candidate_metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        candidate_id,
                        method,
                        scores["lexical_score"],
                        scores["embedding_score"],
                        scores["context_score"],
                        scores["structure_score"],
                        calculated,
                        clean_text(model_version),
                        self._json(dict(metadata or {})),
                        clean_text(candidate_entity_id),
                        clean_text(mention_id),
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise EntityNotFoundError(
                        "mention/entity not found, inactive, or entity types differ"
                    )
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def list_similarity_candidates(
        self,
        mention_id: str,
        *,
        include_rejected: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        self.init_tables()
        if limit <= 0 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        with self.connection.cursor() as cursor:
            status_clause = "" if include_rejected else "AND c.candidate_status='candidate_semantic'"
            cursor.execute(
                f"""
                SELECT c.*, e.entity_type, e.canonical_name, e.business_id,
                       e.identity_scope
                FROM icp_entity_resolution_candidates c
                JOIN icp_canonical_entities e
                  ON e.entity_id=c.candidate_entity_id
                WHERE c.mention_id=%s {status_clause}
                ORDER BY c.combined_score DESC, c.candidate_id
                LIMIT %s
                """,
                (clean_text(mention_id), limit),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def list_candidate_source_mentions(
        self,
        *,
        doc_id: Optional[str] = None,
        entity_types: Optional[Sequence[str]] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Return document-scoped mentions eligible for semantic discovery.

        Resolved business identifiers are deliberately excluded: an exact code
        already has a formal identity and must not be reconsidered by a model.
        """

        self.init_tables()
        if limit <= 0 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        params: List[Any] = []
        document_clause = ""
        document_id = clean_text(doc_id)
        if document_id:
            document_clause = "AND mention.doc_id=%s"
            params.append(document_id)
        type_clause = ""
        if entity_types:
            normalized_types = sorted(
                {self._validate_entity_type(value) for value in entity_types}
            )
            type_clause = "AND mention.entity_type = ANY(%s)"
            params.append(normalized_types)
        params.append(limit)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT mention.*,
                       entity.canonical_name AS resolved_entity_name,
                       entity.identity_scope AS entity_identity_scope,
                       entity.identity_method AS entity_identity_method,
                       entity.origin_doc_id AS entity_origin_doc_id
                FROM icp_entity_mentions mention
                JOIN icp_canonical_entities entity
                  ON entity.entity_id=mention.resolved_entity_id
                WHERE mention.resolution_status='document_scoped'
                  AND entity.identity_method='document_scoped'
                  AND entity.lifecycle_status='active'
                  {document_clause}
                  {type_clause}
                ORDER BY mention.doc_id, mention.entity_type, mention.mention_id
                LIMIT %s
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def list_candidate_target_entities(
        self,
        *,
        entity_type: str,
        exclude_doc_id: str,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Return same-type entities from other documents for comparison."""

        self.init_tables()
        if limit <= 0 or limit > 20000:
            raise ValueError("limit must be between 1 and 20000")
        dtype = self._validate_entity_type(entity_type)
        document_id = clean_text(exclude_doc_id)
        if not document_id:
            raise ValueError("exclude_doc_id is required")
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT entity.*
                FROM icp_canonical_entities entity
                WHERE entity.entity_type=%s
                  AND entity.lifecycle_status='active'
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM icp_entity_mentions mention
                          WHERE mention.resolved_entity_id=entity.entity_id
                            AND mention.doc_id <> %s
                            AND mention.resolution_status IN (
                                'document_scoped',
                                'resolved_business_id',
                                'resolved_explicit_reference'
                            )
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM icp_relation_assertions assertion
                          WHERE assertion.doc_id <> %s
                            AND assertion.assertion_status='asserted'
                            AND (
                                assertion.source_entity_id=entity.entity_id
                                OR assertion.target_entity_id=entity.entity_id
                            )
                      )
                  )
                ORDER BY entity.identity_method, entity.canonical_key
                LIMIT %s
                """,
                (dtype, document_id, document_id, limit),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def purge_orphan_entities(self) -> int:
        """Remove canonical rows no longer backed by a mention or assertion.

        This is an idempotent repair hook for callers that delete documents in
        bulk or upgrade from a schema created before the delete triggers were
        installed.  It deliberately ignores semantic-candidate rows: a search
        candidate is not authority and must not keep an otherwise orphaned
        entity alive.
        """

        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM icp_canonical_entities AS entity
                    WHERE NOT EXISTS (
                        SELECT 1 FROM icp_entity_mentions AS mention
                        WHERE mention.resolved_entity_id=entity.entity_id
                    )
                      AND NOT EXISTS (
                        SELECT 1 FROM icp_relation_assertions AS assertion
                        WHERE assertion.source_entity_id=entity.entity_id
                           OR assertion.target_entity_id=entity.entity_id
                    )
                    """
                )
                deleted_count = max(0, int(getattr(cursor, "rowcount", 0)))
            self._commit()
            return deleted_count
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def clear_similarity_candidates(
        self,
        mention_ids: Sequence[str],
        *,
        resolution_method: str = "hybrid_similarity",
        model_version: str = "",
    ) -> int:
        """Delete a rebuildable candidate snapshot before recomputing it."""

        self.init_tables()
        ids = sorted({clean_text(value) for value in mention_ids if clean_text(value)})
        if not ids:
            return 0
        method = clean_text(resolution_method).casefold()
        if method not in SIMILARITY_METHODS:
            raise ValueError(f"unsupported similarity method: {resolution_method}")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM icp_entity_resolution_candidates
                    WHERE mention_id = ANY(%s)
                      AND resolution_method=%s
                      AND model_version=%s
                    """,
                    (ids, method, clean_text(model_version)),
                )
                deleted = max(0, int(getattr(cursor, "rowcount", 0)))
            self._commit()
            return deleted
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def upsert_relation_assertion(
        self,
        *,
        doc_id: str,
        relation_type: str,
        source_entity_id: str,
        target_entity_id: str,
        assertion_method: str,
        record_id: Optional[str] = None,
        source_ref: Optional[str] = None,
        evidence_text: Optional[str] = None,
        logical_edge_id: Optional[str] = None,
        occurrence_qualifiers: Optional[Sequence[Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        assertion_status: str = "asserted",
    ) -> Dict[str, Any]:
        """Persist an explicit/deterministic edge with complete provenance.

        Similarity methods are rejected.  Repeated occurrences of the same
        canonical edge in one document are accumulated without creating
        duplicate graph edges.
        """

        self.init_tables()
        document_id = clean_text(doc_id)
        relation = self._validate_relation_type(relation_type)
        method = clean_text(assertion_method).casefold()
        status = clean_text(assertion_status).casefold()
        source_id = clean_text(source_entity_id)
        target_id = clean_text(target_entity_id)
        if not document_id or not source_id or not target_id:
            raise ValueError("doc_id, source_entity_id and target_entity_id are required")
        if source_id == target_id:
            raise ValueError("self-referential relation assertions are not allowed")
        if method not in RELATION_ASSERTION_METHODS:
            if method in SIMILARITY_METHODS or "similar" in method:
                raise UnsafeResolutionError(
                    "similarity may discover candidates but cannot assert a formal relation"
                )
            raise ValueError(f"unsupported assertion_method: {assertion_method}")
        if status not in {"asserted", "quarantined"}:
            raise ValueError(f"unsupported assertion_status: {assertion_status}")
        qualifiers = [
            dict(item) if isinstance(item, Mapping) else {"value": item}
            for item in (occurrence_qualifiers or [])
            if item not in (None, "", [], {})
        ]
        assertion_id = self._id_factory()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_relation_assertions
                        (assertion_id, doc_id, relation_type, source_entity_id,
                         target_entity_id, assertion_method, assertion_status,
                         record_id, source_ref, evidence_text,
                         logical_edge_ids, occurrence_qualifiers, provenance)
                    SELECT %s,%s,%s,source.entity_id,target.entity_id,%s,%s,
                           %s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb
                    FROM icp_canonical_entities source
                    CROSS JOIN icp_canonical_entities target
                    WHERE source.entity_id=%s AND target.entity_id=%s
                      AND source.lifecycle_status='active'
                      AND target.lifecycle_status='active'
                    ON CONFLICT (
                        doc_id, relation_type, source_entity_id, target_entity_id
                    ) DO UPDATE SET
                        assertion_method = EXCLUDED.assertion_method,
                        assertion_status = EXCLUDED.assertion_status,
                        record_id = COALESCE(
                            EXCLUDED.record_id, icp_relation_assertions.record_id
                        ),
                        source_ref = COALESCE(
                            EXCLUDED.source_ref, icp_relation_assertions.source_ref
                        ),
                        evidence_text = COALESCE(
                            EXCLUDED.evidence_text,
                            icp_relation_assertions.evidence_text
                        ),
                        logical_edge_ids = (
                            SELECT COALESCE(jsonb_agg(DISTINCT item), '[]'::jsonb)
                            FROM jsonb_array_elements(
                                icp_relation_assertions.logical_edge_ids
                                || EXCLUDED.logical_edge_ids
                            ) AS items(item)
                        ),
                        occurrence_qualifiers = (
                            SELECT COALESCE(jsonb_agg(DISTINCT occurrence), '[]'::jsonb)
                            FROM jsonb_array_elements(
                                icp_relation_assertions.occurrence_qualifiers
                                || EXCLUDED.occurrence_qualifiers
                            ) AS occurrences(occurrence)
                        ),
                        provenance = icp_relation_assertions.provenance || EXCLUDED.provenance,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        assertion_id,
                        document_id,
                        relation,
                        method,
                        status,
                        clean_text(record_id) or None,
                        clean_text(source_ref) or None,
                        clean_text(evidence_text) or None,
                        self._json(
                            [clean_text(logical_edge_id)]
                            if clean_text(logical_edge_id)
                            else []
                        ),
                        self._json(qualifiers),
                        self._json(dict(provenance or {})),
                        source_id,
                        target_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise EntityNotFoundError("relation endpoint entity not found or inactive")
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @_connection_locked
    def list_relation_assertions(
        self,
        entity_ids: Sequence[str],
        *,
        direction: str = "both",
        relation_types: Optional[Sequence[str]] = None,
        include_quarantined: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Read one-hop assertions for a future bounded graph traversal tool."""

        self.init_tables()
        ids = sorted({clean_text(value) for value in entity_ids if clean_text(value)})
        if not ids:
            return []
        normalized_direction = clean_text(direction).casefold()
        if normalized_direction not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in, or both")
        if limit <= 0 or limit > 5000:
            raise ValueError("limit must be between 1 and 5000")
        if normalized_direction == "out":
            endpoint_clause = "r.source_entity_id = ANY(%s)"
        elif normalized_direction == "in":
            endpoint_clause = "r.target_entity_id = ANY(%s)"
        else:
            endpoint_clause = (
                "(r.source_entity_id = ANY(%s) OR r.target_entity_id = ANY(%s))"
            )
        params: List[Any] = [ids]
        if normalized_direction == "both":
            params.append(ids)
        relation_clause = ""
        if relation_types:
            normalized_relations = sorted(
                {self._validate_relation_type(value) for value in relation_types}
            )
            relation_clause = "AND r.relation_type = ANY(%s)"
            params.append(normalized_relations)
        status_clause = "" if include_quarantined else "AND r.assertion_status='asserted'"
        params.append(limit)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT r.*,
                       source.entity_type AS source_entity_type,
                       source.canonical_name AS source_entity_name,
                       source.business_id AS source_business_id,
                       target.entity_type AS target_entity_type,
                       target.canonical_name AS target_entity_name,
                       target.business_id AS target_business_id
                FROM icp_relation_assertions r
                JOIN icp_canonical_entities source
                  ON source.entity_id=r.source_entity_id
                JOIN icp_canonical_entities target
                  ON target.entity_id=r.target_entity_id
                WHERE {endpoint_clause} {relation_clause} {status_clause}
                ORDER BY r.relation_type, r.assertion_id
                LIMIT %s
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def find_entities_by_logical_id(
        self,
        logical_id: str,
        *,
        entity_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Resolve a deterministic graph logical ID back to PG authority.

        A list is returned because one logical code can be isolated by
        applicability scope (for example two aircraft configurations).
        """

        self.init_tables()
        logical = clean_text(logical_id)
        if not logical:
            return []
        params: List[Any] = [logical]
        type_clause = ""
        if entity_type:
            type_clause = "AND entity_type=%s"
            params.append(self._validate_entity_type(entity_type))
        status_clause = "AND lifecycle_status='active'" if active_only else ""
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT * FROM icp_canonical_entities
                WHERE logical_ids ? %s {type_clause} {status_clause}
                ORDER BY entity_type, canonical_key
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM icp_canonical_entities WHERE entity_id=%s",
                (clean_text(entity_id),),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None

    @_connection_locked
    def list_entity_mentions(
        self,
        entity_id: str,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        formal_only: bool = True,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return the exact source records supporting one canonical entity."""

        self.init_tables()
        if limit <= 0 or limit > 2000:
            raise ValueError("limit must be between 1 and 2000")
        status_clause = (
            "AND resolution_status IN ("
            "'resolved_business_id','resolved_explicit_reference','document_scoped'"
            ")"
            if formal_only
            else ""
        )
        params: List[Any] = [clean_text(entity_id)]
        document_clause = ""
        if doc_ids:
            documents = sorted(
                {clean_text(value) for value in doc_ids if clean_text(value)}
            )
            if documents:
                document_clause = "AND doc_id = ANY(%s)"
                params.append(documents)
        params.append(limit)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT * FROM icp_entity_mentions
                WHERE resolved_entity_id=%s {status_clause} {document_clause}
                ORDER BY doc_id, source_ref, mention_id
                LIMIT %s
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def get_relation_assertion(
        self,
        *,
        doc_id: str,
        relation_type: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read the canonical relation identity tuple required by projection."""

        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM icp_relation_assertions
                WHERE doc_id=%s AND relation_type=%s
                  AND source_entity_id=%s AND target_entity_id=%s
                """,
                (
                    clean_text(doc_id),
                    self._validate_relation_type(relation_type),
                    clean_text(source_entity_id),
                    clean_text(target_entity_id),
                ),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None

    @_connection_locked
    def find_relations_by_logical_tuple(
        self,
        *,
        doc_id: str,
        relation_type: str,
        source_logical_id: str,
        target_logical_id: str,
        formal_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Resolve an input graph tuple to one or more scoped PG assertions."""

        self.init_tables()
        status_clause = "AND relation.assertion_status='asserted'" if formal_only else ""
        source_relation = clean_text(relation_type).casefold()
        canonical_relation = _STRUCTURED_RELATION_TYPES.get(
            source_relation, source_relation
        )
        canonical_relation = self._validate_relation_type(canonical_relation)
        source_relation_clause = ""
        params: List[Any] = [
            clean_text(doc_id),
            canonical_relation,
            clean_text(source_logical_id),
            clean_text(target_logical_id),
        ]
        if source_relation != canonical_relation:
            source_relation_clause = "AND relation.provenance->>'source_relation'=%s"
            params.append(source_relation)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT relation.*,
                       source.entity_type AS source_entity_type,
                       source.canonical_name AS source_entity_name,
                       target.entity_type AS target_entity_type,
                       target.canonical_name AS target_entity_name
                FROM icp_relation_assertions relation
                JOIN icp_canonical_entities source
                  ON source.entity_id=relation.source_entity_id
                JOIN icp_canonical_entities target
                  ON target.entity_id=relation.target_entity_id
                WHERE relation.doc_id=%s AND relation.relation_type=%s
                  AND source.logical_ids ? %s
                  AND target.logical_ids ? %s
                  {source_relation_clause}
                  {status_clause}
                ORDER BY relation.assertion_id
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    @_connection_locked
    def find_relation_by_logical_edge_id(
        self,
        *,
        doc_id: str,
        logical_edge_id: str,
        formal_only: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return the PG assertion that backs one projection logical edge."""

        self.init_tables()
        status_clause = "AND assertion_status='asserted'" if formal_only else ""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT * FROM icp_relation_assertions
                    WHERE doc_id=%s AND logical_edge_ids ? %s
                      {status_clause}
                    """,
                    (clean_text(doc_id), clean_text(logical_edge_id)),
                )
                row = cursor.fetchone()
                result = self._row_to_dict(cursor, row) if row else None
            # Reauthorization is read-only but psycopg is not autocommit.  End
            # the transaction so repeated graph queries do not retain relation
            # locks while the process waits for the next HTTP request.
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def _sync_graph_impl(
        self,
        *,
        doc_id: str,
        doc_type: str,
        graph: Mapping[str, Any],
        document_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronize one ``build_document_graph`` result into the authority.

        The adapter persists vertices first, then source-backed edges.  Only a
        graph vertex carrying ``business_id`` is globally resolved; all other
        text is document-scoped.  Unknown edge labels are reported and skipped
        instead of being invented at runtime.  The previous document snapshot
        is removed inside the same outer transaction before the replacement is
        written.  A failed replacement therefore rolls the deletion back, while
        a successful retry cannot retain removed fields, mentions or edges.
        """

        document_id = clean_text(doc_id)
        dtype = clean_text(doc_type).casefold()
        if not document_id or not dtype:
            raise ValueError("doc_id and doc_type are required")
        if not isinstance(graph, Mapping):
            raise ValueError("graph must be a mapping")
        graph_doc_id = clean_text(graph.get("doc_id"))
        graph_doc_type = clean_text(graph.get("doc_type")).casefold()
        if graph_doc_id and graph_doc_id != document_id:
            raise ValueError("graph doc_id does not match sync doc_id")
        if graph_doc_type and graph_doc_type != dtype:
            raise ValueError("graph doc_type does not match sync doc_type")

        vertices = graph.get("vertices") or []
        edges = graph.get("edges") or []
        if not isinstance(vertices, Sequence) or isinstance(vertices, (str, bytes)):
            raise ValueError("graph vertices must be a sequence")
        if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
            raise ValueError("graph edges must be a sequence")

        affected_entity_ids: set[str] = set()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT entity_id
                FROM (
                    SELECT resolved_entity_id AS entity_id
                    FROM icp_entity_mentions
                    WHERE doc_id=%s
                    UNION
                    SELECT source_entity_id AS entity_id
                    FROM icp_relation_assertions
                    WHERE doc_id=%s
                    UNION
                    SELECT target_entity_id AS entity_id
                    FROM icp_relation_assertions
                    WHERE doc_id=%s
                ) AS affected
                WHERE entity_id IS NOT NULL
                """,
                (document_id, document_id, document_id),
            )
            for row in cursor.fetchall():
                entity_id = clean_text(
                    row.get("entity_id") if isinstance(row, Mapping) else row[0]
                )
                if entity_id:
                    affected_entity_ids.add(entity_id)

            # Relations are removed first so a mention-delete trigger can
            # immediately collect entities that disappear from this snapshot.
            cursor.execute(
                "DELETE FROM icp_relation_assertions WHERE doc_id=%s",
                (document_id,),
            )
            stale_assertion_count = max(
                0, int(getattr(cursor, "rowcount", 0) or 0)
            )
            cursor.execute(
                "DELETE FROM icp_entity_mentions WHERE doc_id=%s",
                (document_id,),
            )
            stale_mention_count = max(0, int(getattr(cursor, "rowcount", 0) or 0))

        metadata = dict(document_metadata or {})
        identity_scope = _identity_scope_from_metadata(metadata)
        # Missing applicability must reduce recall, never silently broaden it.
        # A document guard keeps an otherwise identical local code isolated
        # until a complete system + configuration scope is available.
        scope_guarded = not {
            "system_code",
            "configuration",
        }.issubset(identity_scope)
        if scope_guarded:
            identity_scope = normalize_scope(
                {**identity_scope, "unresolved_scope_doc_id": document_id}
            )
        logical_to_entity: Dict[str, Dict[str, Any]] = {}
        mention_count = 0
        for index, raw_vertex in enumerate(vertices, start=1):
            if not isinstance(raw_vertex, Mapping):
                raise ValueError(f"graph vertex {index} must be a mapping")
            logical_id = clean_text(raw_vertex.get("id"))
            if not logical_id:
                raise ValueError(f"graph vertex {index} has no logical id")
            if logical_id in logical_to_entity:
                raise ValueError(f"duplicate graph logical id: {logical_id}")
            raw_properties = raw_vertex.get("properties") or {}
            properties = (
                dict(raw_properties) if isinstance(raw_properties, Mapping) else {}
            )
            entity_type = _structured_entity_type(raw_vertex)
            field_type = (
                clean_text(properties.get("value_kind"))
                or clean_text(raw_vertex.get("label"))
                or "source_fragment"
            )
            name = clean_text(
                properties.get("name")
                or properties.get("text")
                or properties.get("occurrence_id")
                or properties.get("business_id")
                or logical_id
            )
            business_id = normalize_business_id(properties.get("business_id"))
            entity_attributes = {
                "structured_label": clean_text(raw_vertex.get("label")),
                "structured_properties": properties,
                "source_doc_type": dtype,
            }
            entity_scope = identity_scope
            if business_id:
                entity_scope = {} if entity_type == "Document" else identity_scope
                entity = self.upsert_business_entity(
                    entity_type=entity_type,
                    business_id=business_id,
                    canonical_name=name,
                    scope=entity_scope,
                    attributes=entity_attributes,
                    origin_doc_id=document_id,
                    logical_id=logical_id,
                )
            else:
                entity = self.upsert_document_scoped_entity(
                    doc_id=document_id,
                    entity_type=entity_type,
                    field_type=field_type,
                    raw_text=name,
                    canonical_name=name,
                    scope=identity_scope,
                    attributes=entity_attributes,
                    logical_id=logical_id,
                )
            logical_to_entity[logical_id] = entity

            source_refs = properties.get("source_refs") or []
            if isinstance(source_refs, str):
                source_refs = [source_refs]
            source_ref = clean_text(source_refs[0]) if source_refs else ""
            mention = self.upsert_mention(
                doc_id=document_id,
                entity_type=entity_type,
                field_type=field_type,
                raw_text=name,
                record_id=clean_text(
                    properties.get("record_key")
                    or properties.get("occurrence_id")
                    or business_id
                )
                or None,
                source_ref=source_ref or None,
                business_id=business_id or None,
                scope=entity_scope,
                attributes={"logical_id": logical_id, "doc_type": dtype},
            )
            mention_id = clean_text(mention.get("mention_id"))
            entity_id = clean_text(entity.get("entity_id"))
            if not mention_id or not entity_id:
                raise EntityResolutionError("upsert did not return mention/entity identity")
            affected_entity_ids.add(entity_id)
            if business_id:
                self._resolve_mention(
                    mention_id=mention_id,
                    entity_id=entity_id,
                    method="business_id",
                    evidence={
                        "business_id": business_id,
                        "logical_id": logical_id,
                        "source": "structured_graph",
                    },
                )
            else:
                self.resolve_mention_document_scoped(
                    mention_id=mention_id, entity_id=entity_id
                )
            mention_count += 1

        assertion_ids: List[str] = []
        skipped_edges: List[Dict[str, str]] = []
        for index, raw_edge in enumerate(edges, start=1):
            if not isinstance(raw_edge, Mapping):
                raise ValueError(f"graph edge {index} must be a mapping")
            logical_edge_id = clean_text(raw_edge.get("id")) or f"edge-{index}"
            raw_relation = clean_text(raw_edge.get("label")).casefold()
            relation = _STRUCTURED_RELATION_TYPES.get(raw_relation)
            if not relation:
                skipped_edges.append(
                    {
                        "edge_id": logical_edge_id,
                        "relation": raw_relation,
                        "reason": "unsupported_relation_type",
                    }
                )
                continue
            source_logical_id = clean_text(
                raw_edge.get("source") or raw_edge.get("outV")
            )
            target_logical_id = clean_text(
                raw_edge.get("target") or raw_edge.get("inV")
            )
            source_entity = logical_to_entity.get(source_logical_id)
            target_entity = logical_to_entity.get(target_logical_id)
            if not source_entity or not target_entity:
                raise ValueError(
                    f"graph edge {logical_edge_id} has an unknown endpoint"
                )
            raw_properties = raw_edge.get("properties") or {}
            properties = (
                dict(raw_properties) if isinstance(raw_properties, Mapping) else {}
            )
            source_refs = properties.get("source_refs") or []
            if isinstance(source_refs, str):
                source_refs = [source_refs]
            method = "deterministic_structure"
            if raw_relation.startswith("references_"):
                method = "explicit_code_reference"
            elif raw_relation == "next_step":
                method = "ordered_step"
            elif raw_relation not in {
                "contains_component",
                "contains_function",
                "provides_function",
                "interface_source",
                "interface_target",
                "decomposed_by",
                "has_input_event",
                "next_block",
            }:
                method = "explicit_field"
            revision_id = clean_text(
                properties.get("revision_id")
                or properties.get("diagram_revision_id")
            )
            validation_status = clean_text(
                properties.get("validation_status")
            ).casefold()
            if dtype in {"fta", "rbd"} and raw_relation != "extracted_from" and revision_id:
                if validation_status == "auto_validated":
                    method = "auto_validated_diagram"
                elif validation_status == "manual_confirmed":
                    method = "manual_confirmed_diagram"
            raw_qualifiers = properties.get("occurrence_qualifiers") or []
            if isinstance(raw_qualifiers, (str, bytes, Mapping)):
                raw_qualifiers = [raw_qualifiers]
            assertion = self.upsert_relation_assertion(
                doc_id=document_id,
                relation_type=relation,
                source_entity_id=clean_text(source_entity.get("entity_id")),
                target_entity_id=clean_text(target_entity.get("entity_id")),
                assertion_method=method,
                record_id=clean_text(properties.get("record_key")) or None,
                source_ref=clean_text(source_refs[0]) if source_refs else None,
                evidence_text=clean_text(properties.get("evidence_text")) or None,
                logical_edge_id=logical_edge_id,
                occurrence_qualifiers=[
                    *raw_qualifiers,
                    {
                        "logical_edge_id": logical_edge_id,
                        "source_relation": raw_relation,
                    },
                ],
                provenance={
                    "logical_edge_id": logical_edge_id,
                    "source_logical_id": source_logical_id,
                    "target_logical_id": target_logical_id,
                    "source_relation": raw_relation,
                    "source_properties": properties,
                    "doc_type": dtype,
                    "document_metadata": metadata,
                },
            )
            assertion_id = clean_text(assertion.get("assertion_id"))
            if not assertion_id:
                raise EntityResolutionError("upsert did not return assertion identity")
            assertion_ids.append(assertion_id)

        # Canonical logical IDs are a projection of the surviving mentions,
        # not an append-only cache.  Recompute every touched entity so removed
        # source fields and document revisions cannot leave stale identifiers.
        garbage_collected_entity_count = 0
        if affected_entity_ids:
            with self.connection.cursor() as cursor:
                entity_ids = sorted(affected_entity_ids)
                cursor.execute(
                    """
                    UPDATE icp_canonical_entities AS entity
                    SET logical_ids = COALESCE(
                            (
                                SELECT jsonb_agg(
                                    DISTINCT to_jsonb(
                                        mention.attributes->>'logical_id'
                                    )
                                )
                                FROM icp_entity_mentions AS mention
                                WHERE mention.resolved_entity_id=entity.entity_id
                                  AND COALESCE(
                                      mention.attributes->>'logical_id', ''
                                  ) <> ''
                            ),
                            '[]'::jsonb
                        ),
                        origin_doc_id = (
                            SELECT MIN(mention.doc_id)
                            FROM icp_entity_mentions AS mention
                            WHERE mention.resolved_entity_id=entity.entity_id
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE entity.entity_id = ANY(%s)
                    """,
                    (entity_ids,),
                )
                cursor.execute(
                    """
                    DELETE FROM icp_canonical_entities AS entity
                    WHERE entity.entity_id = ANY(%s)
                      AND NOT EXISTS (
                          SELECT 1 FROM icp_entity_mentions AS mention
                          WHERE mention.resolved_entity_id=entity.entity_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM icp_relation_assertions AS assertion
                          WHERE assertion.source_entity_id=entity.entity_id
                             OR assertion.target_entity_id=entity.entity_id
                      )
                    """,
                    (entity_ids,),
                )
                garbage_collected_entity_count = max(
                    0, int(getattr(cursor, "rowcount", 0) or 0)
                )

        return {
            "doc_id": document_id,
            "doc_type": dtype,
            "identity_scope": identity_scope,
            "identity_scope_guarded": scope_guarded,
            "entity_count": len(logical_to_entity),
            "mention_count": mention_count,
            "assertion_count": len(assertion_ids),
            "stale_assertion_count": stale_assertion_count,
            "stale_mention_count": stale_mention_count,
            "garbage_collected_entity_count": garbage_collected_entity_count,
            "skipped_edges": skipped_edges,
            "logical_entity_ids": {
                logical_id: clean_text(entity.get("entity_id"))
                for logical_id, entity in sorted(logical_to_entity.items())
            },
        }

    @_connection_locked
    def sync_graph(
        self,
        *,
        doc_id: str,
        doc_type: str,
        graph: Mapping[str, Any],
        document_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically replace one document's entity/relation authority snapshot."""

        # Schema creation owns its own commit and must finish before opening the
        # document replacement transaction.
        self.init_tables()
        return self._run_atomically(
            lambda: self._sync_graph_impl(
                doc_id=doc_id,
                doc_type=doc_type,
                graph=graph,
                document_metadata=document_metadata,
            )
        )

    @_connection_locked
    def sync_structured_graph(
        self,
        *,
        doc_id: str,
        doc_type: str,
        graph: Mapping[str, Any],
        document_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Named integration alias for ``structured_projection`` callers."""

        return self.sync_graph(
            doc_id=doc_id,
            doc_type=doc_type,
            graph=graph,
            document_metadata=document_metadata,
        )


# Importing this module has no database side effect; the existing connection is
# resolved only when a repository method is called.
entity_resolution_repository = EntityResolutionRepository()


__all__ = [
    "CORE_ENTITY_TYPES",
    "CORE_RELATION_TYPES",
    "FORMAL_RESOLUTION_METHODS",
    "RELATION_ASSERTION_METHODS",
    "SIMILARITY_METHODS",
    "EntityNotFoundError",
    "EntityResolutionError",
    "EntityResolutionRepository",
    "UnsafeResolutionError",
    "business_entity_key",
    "clean_text",
    "cosine_similarity",
    "document_entity_key",
    "entity_resolution_repository",
    "normalize_business_id",
    "normalize_scope",
]
