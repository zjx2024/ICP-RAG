"""PostgreSQL authority for reviewable ICP engineering diagrams.

This repository intentionally has no dependency on OpenSearch or HugeGraph.
Those systems are rebuildable projections; PostgreSQL owns candidate evidence,
canonical revisions, explicit review state, and the active-revision pointer.

The connection is injectable and resolved lazily, which keeps pure unit tests
independent from a running PostgreSQL service.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.icp.diagram_models import (
    DIAGRAM_TYPES,
    PROJECTION_FAILED,
    PROJECTION_NOT_PROJECTED,
    PROJECTION_PUBLISHED,
    PROJECTION_PUBLISHING,
    PROJECTION_STALE,
    REVIEW_CONFIRMED,
    REVIEW_CONFLICTED,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    SOURCE_KINDS,
    canonical_graph_hash,
    clean_text,
    content_hash,
    detect_graph_conflicts,
    merge_candidates,
    normalize_graph,
)
from backend.icp.diagram_validation import (
    POLICY_VERSION,
    VALIDATION_MANUAL_CONFIRMED,
    VALIDATION_PENDING,
    VALIDATION_QUARANTINED,
    evaluate_automatic_validation,
    revision_is_projectable,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEWABLE_DIAGRAM_TYPES = frozenset({"fta", "rbd"})
_REVIEWABLE_DOCUMENT_STATUSES = frozenset({"kg_pending", "kg_done"})


class DiagramRepositoryError(RuntimeError):
    """Base error for diagram persistence and state operations."""


class DiagramNotFoundError(DiagramRepositoryError):
    pass


class DiagramStateError(DiagramRepositoryError):
    pass


class DiagramConcurrencyError(DiagramRepositoryError):
    pass


class DiagramRepository:
    """Small, transaction-oriented repository over a DB-API connection."""

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

    @property
    def connection(self):
        if self._connection is None:
            # Lazy import prevents model-only unit tests from initializing the
            # legacy PostgreSQL client and its fallback database.
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
                cls._column_name(item): value
                for item, value in zip(description, row)
            }
        for key in (
            "scope",
            "canonical_graph",
            "conflicts",
            "validation_report",
            "source_locator",
            "raw_graph",
            "normalized_graph",
            "manifest",
        ):
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
        self.connection.commit()

    def _rollback(self) -> None:
        try:
            self.connection.rollback()
        except Exception:
            pass

    def init_tables(self) -> None:
        """Create the graph-review schema idempotently.

        No legacy table is rewritten.  ``document_fault_trees`` can therefore
        remain as a compatibility source while callers migrate to this schema.
        """

        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            statements = [
                """
                CREATE TABLE IF NOT EXISTS icp_diagrams (
                    diagram_id          VARCHAR(36) PRIMARY KEY,
                    doc_id              VARCHAR(36) NOT NULL
                                        REFERENCES icp_documents(doc_id) ON DELETE CASCADE,
                    diagram_type        VARCHAR(32) NOT NULL
                                        CHECK (diagram_type IN ('function_locator','fta','rbd')),
                    diagram_key         VARCHAR(300) NOT NULL,
                    title               VARCHAR(512),
                    source_section      VARCHAR(512),
                    scope               JSONB NOT NULL DEFAULT '{}'::jsonb,
                    lifecycle_status    VARCHAR(20) NOT NULL DEFAULT 'active',
                    archived_at         TIMESTAMP,
                    archive_reason      TEXT,
                    active_revision_id  VARCHAR(36),
                    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (doc_id, diagram_key)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_diagram_revisions (
                    revision_id       VARCHAR(36) PRIMARY KEY,
                    diagram_id        VARCHAR(36) NOT NULL
                                      REFERENCES icp_diagrams(diagram_id) ON DELETE CASCADE,
                    revision_no       INTEGER NOT NULL CHECK (revision_no > 0),
                    schema_version    INTEGER NOT NULL DEFAULT 1 CHECK (schema_version > 0),
                    canonical_graph   JSONB NOT NULL,
                    canonical_hash    CHAR(64) NOT NULL,
                    review_status     VARCHAR(20) NOT NULL DEFAULT 'pending'
                                      CHECK (review_status IN ('pending','conflicted','confirmed','rejected')),
                    projection_status VARCHAR(20) NOT NULL DEFAULT 'not_projected'
                                      CHECK (projection_status IN ('not_projected','publishing','published','failed','stale')),
                    conflicts         JSONB NOT NULL DEFAULT '[]'::jsonb,
                    validation_report JSONB NOT NULL DEFAULT '{}'::jsonb,
                    validation_status VARCHAR(24) NOT NULL DEFAULT 'pending'
                                      CHECK (validation_status IN ('pending','auto_validated','provisional','quarantined','failed','manual_confirmed')),
                    validation_tier   VARCHAR(40) NOT NULL DEFAULT 'none',
                    validation_method VARCHAR(40) NOT NULL DEFAULT 'automatic',
                    validator_version VARCHAR(100),
                    validated_at      TIMESTAMP,
                    quarantine_reason TEXT,
                    conflict_count    INTEGER NOT NULL DEFAULT 0 CHECK (conflict_count >= 0),
                    reviewed_by       VARCHAR(200),
                    reviewed_at       TIMESTAMP,
                    review_note       TEXT,
                    lock_version      INTEGER NOT NULL DEFAULT 1 CHECK (lock_version > 0),
                    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    published_at      TIMESTAMP,
                    UNIQUE (diagram_id, revision_no),
                    UNIQUE (diagram_id, revision_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_diagram_candidates (
                    candidate_id       VARCHAR(36) PRIMARY KEY,
                    diagram_id         VARCHAR(36) NOT NULL
                                       REFERENCES icp_diagrams(diagram_id) ON DELETE CASCADE,
                    source_kind        VARCHAR(40) NOT NULL
                                       CHECK (source_kind IN ('image_vision','table_nodes','table_edges','structured_expression','manual','legacy')),
                    artifact_ref_id    VARCHAR(36),
                    source_locator     JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_fingerprint CHAR(64) NOT NULL,
                    extractor_name     VARCHAR(200) NOT NULL DEFAULT 'unknown',
                    extractor_version  VARCHAR(100) NOT NULL DEFAULT '1',
                    model_name         VARCHAR(200),
                    prompt_version     VARCHAR(100),
                    raw_graph          JSONB NOT NULL,
                    normalized_graph   JSONB NOT NULL,
                    candidate_hash     CHAR(64) NOT NULL,
                    extraction_status  VARCHAR(20) NOT NULL DEFAULT 'succeeded'
                                       CHECK (extraction_status IN ('succeeded','failed','skipped')),
                    merge_status       VARCHAR(20) NOT NULL DEFAULT 'included'
                                       CHECK (merge_status IN ('included','duplicate','conflict','isolated','rejected')),
                    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (
                        diagram_id, source_kind, source_fingerprint,
                        extractor_version, candidate_hash
                    )
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_diagram_revision_candidates (
                    revision_id  VARCHAR(36) NOT NULL
                                 REFERENCES icp_diagram_revisions(revision_id) ON DELETE CASCADE,
                    candidate_id VARCHAR(36) NOT NULL
                                 REFERENCES icp_diagram_candidates(candidate_id) ON DELETE RESTRICT,
                    merge_role   VARCHAR(20) NOT NULL DEFAULT 'included'
                                 CHECK (merge_role IN ('included','duplicate','conflict','isolated','rejected')),
                    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (revision_id, candidate_id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS icp_graph_projection_runs (
                    run_id                VARCHAR(36) PRIMARY KEY,
                    revision_id           VARCHAR(36) NOT NULL
                                          REFERENCES icp_diagram_revisions(revision_id) ON DELETE CASCADE,
                    idempotency_key        VARCHAR(200) NOT NULL UNIQUE,
                    canonical_hash         CHAR(64) NOT NULL,
                    projector_version      VARCHAR(100) NOT NULL,
                    status                 VARCHAR(20) NOT NULL DEFAULT 'running'
                                           CHECK (status IN ('running','succeeded','failed')),
                    expected_vertex_count  INTEGER NOT NULL DEFAULT 0,
                    expected_edge_count    INTEGER NOT NULL DEFAULT 0,
                    actual_vertex_count    INTEGER,
                    actual_edge_count      INTEGER,
                    manifest               JSONB NOT NULL DEFAULT '{}'::jsonb,
                    attempt_count          INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count > 0),
                    error                  TEXT,
                    started_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at           TIMESTAMP,
                    updated_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_icp_diagrams_doc ON icp_diagrams(doc_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_diagrams_type ON icp_diagrams(diagram_type)",
                "ALTER TABLE icp_diagrams ADD COLUMN IF NOT EXISTS lifecycle_status VARCHAR(20) NOT NULL DEFAULT 'active'",
                "ALTER TABLE icp_diagrams ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP",
                "ALTER TABLE icp_diagrams ADD COLUMN IF NOT EXISTS archive_reason TEXT",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS validation_status VARCHAR(24) NOT NULL DEFAULT 'pending'",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS validation_tier VARCHAR(40) NOT NULL DEFAULT 'none'",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS validation_method VARCHAR(40) NOT NULL DEFAULT 'automatic'",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS validator_version VARCHAR(100)",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP",
                "ALTER TABLE icp_diagram_revisions ADD COLUMN IF NOT EXISTS quarantine_reason TEXT",
                # ``isolated`` keeps a visual candidate available as source
                # evidence without merging it into a deterministic canonical
                # topology.  Older local databases predate that state, so the
                # candidate-table guard needs the same idempotent migration as
                # the revision/candidate association below.
                "ALTER TABLE icp_diagram_candidates DROP CONSTRAINT IF EXISTS icp_diagram_candidates_merge_status_check",
                "ALTER TABLE icp_diagram_candidates ADD CONSTRAINT icp_diagram_candidates_merge_status_check CHECK (merge_status IN ('included','duplicate','conflict','isolated','rejected'))",
                "ALTER TABLE icp_diagram_revision_candidates DROP CONSTRAINT IF EXISTS icp_diagram_revision_candidates_merge_role_check",
                "ALTER TABLE icp_diagram_revision_candidates ADD CONSTRAINT icp_diagram_revision_candidates_merge_role_check CHECK (merge_role IN ('included','duplicate','conflict','isolated','rejected'))",
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='ck_icp_revisions_validation_status'
                          AND conrelid='icp_diagram_revisions'::regclass
                    ) THEN
                        ALTER TABLE icp_diagram_revisions
                        ADD CONSTRAINT ck_icp_revisions_validation_status
                        CHECK (validation_status IN (
                            'pending','auto_validated','provisional',
                            'quarantined','failed','manual_confirmed'
                        ));
                    END IF;
                END $$
                """,
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='ck_icp_diagrams_lifecycle_status'
                          AND conrelid='icp_diagrams'::regclass
                    ) THEN
                        ALTER TABLE icp_diagrams
                        ADD CONSTRAINT ck_icp_diagrams_lifecycle_status
                        CHECK (lifecycle_status IN ('active','archived'));
                    END IF;
                END $$
                """,
                # Early development builds briefly used this uniqueness rule.
                # Evidence revisions with the same topology must coexist when
                # their candidate sets differ, so remove that generated
                # constraint idempotently before creating the lookup index.
                "ALTER TABLE icp_diagram_revisions DROP CONSTRAINT IF EXISTS icp_diagram_revisions_diagram_id_canonical_hash_key",
                "CREATE INDEX IF NOT EXISTS idx_icp_revisions_diagram ON icp_diagram_revisions(diagram_id, revision_no DESC)",
                "CREATE INDEX IF NOT EXISTS idx_icp_revisions_hash ON icp_diagram_revisions(diagram_id, canonical_hash)",
                "CREATE INDEX IF NOT EXISTS idx_icp_revisions_review ON icp_diagram_revisions(review_status)",
                "CREATE INDEX IF NOT EXISTS idx_icp_revisions_validation ON icp_diagram_revisions(validation_status)",
                "CREATE INDEX IF NOT EXISTS idx_icp_candidates_diagram ON icp_diagram_candidates(diagram_id)",
                "CREATE INDEX IF NOT EXISTS idx_icp_projection_revision ON icp_graph_projection_runs(revision_id)",
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_icp_revision_one_published
                ON icp_diagram_revisions(diagram_id)
                WHERE projection_status = 'published'
                """,
                # The active pointer is nullable during extraction/review.  The
                # application additionally joins on diagram_id when reading it.
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'fk_icp_diagrams_active_revision'
                    ) THEN
                        ALTER TABLE icp_diagrams
                        ADD CONSTRAINT fk_icp_diagrams_active_revision
                        FOREIGN KEY (active_revision_id)
                        REFERENCES icp_diagram_revisions(revision_id)
                        ON DELETE SET NULL
                        DEFERRABLE INITIALLY DEFERRED;
                    END IF;
                END $$
                """,
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

    @staticmethod
    def _validate_diagram_type(diagram_type: str) -> str:
        value = clean_text(diagram_type).casefold()
        if value not in DIAGRAM_TYPES:
            raise ValueError(f"unsupported diagram_type: {diagram_type}")
        return value

    @staticmethod
    def _validate_sha256(value: str, field_name: str) -> str:
        normalized = clean_text(value).casefold()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        return normalized

    def upsert_diagram(
        self,
        *,
        doc_id: str,
        diagram_type: str,
        diagram_key: str,
        title: str = "",
        source_section: str = "",
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.init_tables()
        dtype = self._validate_diagram_type(diagram_type)
        key = clean_text(diagram_key)
        if not clean_text(doc_id) or not key:
            raise ValueError("doc_id and diagram_key are required")
        diagram_id = self._id_factory()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO icp_diagrams
                        (diagram_id, doc_id, diagram_type, diagram_key, title, source_section, scope)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (doc_id, diagram_key) DO UPDATE SET
                        title = CASE WHEN EXCLUDED.title <> '' THEN EXCLUDED.title ELSE icp_diagrams.title END,
                        source_section = CASE WHEN EXCLUDED.source_section <> '' THEN EXCLUDED.source_section ELSE icp_diagrams.source_section END,
                        scope = CASE WHEN EXCLUDED.scope <> '{}'::jsonb THEN EXCLUDED.scope ELSE icp_diagrams.scope END,
                        lifecycle_status = 'active', archived_at = NULL, archive_reason = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE icp_diagrams.diagram_type = EXCLUDED.diagram_type
                    RETURNING *
                    """,
                    (
                        diagram_id,
                        clean_text(doc_id),
                        dtype,
                        key,
                        clean_text(title),
                        clean_text(source_section),
                        self._json(dict(scope or {})),
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise DiagramStateError(
                        "diagram_key already exists with a different diagram_type"
                    )
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def get_diagram(self, diagram_id: str) -> Optional[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM icp_diagrams WHERE diagram_id=%s", (diagram_id,))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None

    def list_diagrams_by_document(
        self, doc_id: str, *, include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            lifecycle_clause = "" if include_archived else "AND d.lifecycle_status='active'"
            cursor.execute(
                f"""
                SELECT d.*, r.review_status AS active_review_status,
                       r.projection_status AS active_projection_status,
                       r.canonical_hash AS active_canonical_hash
                FROM icp_diagrams d
                LEFT JOIN icp_diagram_revisions r
                  ON r.revision_id=d.active_revision_id AND r.diagram_id=d.diagram_id
                WHERE d.doc_id=%s {lifecycle_clause}
                ORDER BY d.created_at, d.diagram_id
                """,
                (doc_id,),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def archive_missing_diagrams(
        self,
        *,
        doc_id: str,
        present_diagram_ids: Sequence[str],
        reason: str = "missing_from_successful_reparse",
        allow_empty: bool = False,
    ) -> Dict[str, Any]:
        """Converge one successful parse to its complete diagram snapshot.

        Call this only after the whole document parse and candidate persistence
        have succeeded.  Missing masters are archived, their published active
        revision is marked stale, and the active pointer is cleared atomically.
        An empty snapshot requires ``allow_empty=True`` so parser bugs cannot
        silently retire every diagram.
        """

        self.init_tables()
        document_id = clean_text(doc_id)
        present = sorted(
            {clean_text(value) for value in present_diagram_ids if clean_text(value)}
        )
        if not document_id:
            raise ValueError("doc_id is required")
        if not present and not allow_empty:
            raise ValueError("empty diagram snapshot requires allow_empty=True")
        archive_reason = clean_text(reason) or "missing_from_successful_reparse"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT doc_id FROM icp_documents WHERE doc_id=%s FOR UPDATE",
                    (document_id,),
                )
                if not cursor.fetchone():
                    raise DiagramNotFoundError(f"document not found: {document_id}")
                cursor.execute(
                    "SELECT diagram_id FROM icp_diagrams WHERE doc_id=%s FOR UPDATE",
                    (document_id,),
                )
                rows = cursor.fetchall()
                existing = {
                    str(row["diagram_id"] if isinstance(row, Mapping) else row[0])
                    for row in rows
                }
                foreign_or_unknown = sorted(set(present) - existing)
                if foreign_or_unknown:
                    raise DiagramStateError(
                        "present diagram ids do not belong to document: "
                        + ", ".join(foreign_or_unknown)
                    )
                missing = sorted(existing - set(present))
                stale_revision_ids: List[str] = []
                archived_ids: List[str] = []
                reactivated_ids: List[str] = []
                if missing:
                    cursor.execute(
                        """
                        UPDATE icp_diagram_revisions r SET
                            projection_status='stale', updated_at=CURRENT_TIMESTAMP,
                            lock_version=lock_version+1
                        FROM icp_diagrams d
                        WHERE d.doc_id=%s AND d.diagram_id = ANY(%s)
                          AND d.active_revision_id=r.revision_id
                          AND r.diagram_id=d.diagram_id
                          AND r.projection_status='published'
                        RETURNING r.revision_id
                        """,
                        (document_id, missing),
                    )
                    stale_revision_ids = [
                        str(row["revision_id"] if isinstance(row, Mapping) else row[0])
                        for row in cursor.fetchall()
                    ]
                    cursor.execute(
                        """
                        UPDATE icp_diagrams SET
                            lifecycle_status='archived', archived_at=CURRENT_TIMESTAMP,
                            archive_reason=%s, active_revision_id=NULL,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE doc_id=%s AND diagram_id = ANY(%s)
                          AND lifecycle_status <> 'archived'
                        RETURNING diagram_id
                        """,
                        (archive_reason, document_id, missing),
                    )
                    archived_ids = [
                        str(row["diagram_id"] if isinstance(row, Mapping) else row[0])
                        for row in cursor.fetchall()
                    ]
                if present:
                    cursor.execute(
                        """
                        UPDATE icp_diagrams SET
                            lifecycle_status='active', archived_at=NULL,
                            archive_reason=NULL, updated_at=CURRENT_TIMESTAMP
                        WHERE doc_id=%s AND diagram_id = ANY(%s)
                          AND lifecycle_status='archived'
                        RETURNING diagram_id
                        """,
                        (document_id, present),
                    )
                    reactivated_ids = [
                        str(row["diagram_id"] if isinstance(row, Mapping) else row[0])
                        for row in cursor.fetchall()
                    ]
            self._commit()
            return {
                "doc_id": document_id,
                "present_diagram_ids": present,
                "archived_diagram_ids": sorted(archived_ids),
                "reactivated_diagram_ids": sorted(reactivated_ids),
                "stale_revision_ids": sorted(stale_revision_ids),
            }
        except Exception:
            self._rollback()
            raise

    def upsert_candidate(
        self,
        *,
        diagram_id: str,
        source_kind: str,
        source_fingerprint: str,
        raw_graph: Mapping[str, Any],
        artifact_ref_id: Optional[str] = None,
        source_locator: Optional[Mapping[str, Any]] = None,
        extractor_name: str = "unknown",
        extractor_version: str = "1",
        model_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        extraction_status: str = "succeeded",
    ) -> Dict[str, Any]:
        self.init_tables()
        kind = clean_text(source_kind)
        if kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source_kind: {source_kind}")
        fingerprint = self._validate_sha256(source_fingerprint, "source_fingerprint")
        if extraction_status not in {"succeeded", "failed", "skipped"}:
            raise ValueError(f"unsupported extraction_status: {extraction_status}")
        candidate_id = self._id_factory()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT diagram_type,lifecycle_status FROM icp_diagrams WHERE diagram_id=%s FOR SHARE",
                    (diagram_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramNotFoundError(f"diagram not found: {diagram_id}")
                diagram_type = row[0] if not isinstance(row, Mapping) else row["diagram_type"]
                lifecycle_status = (
                    row[1] if not isinstance(row, Mapping) else row["lifecycle_status"]
                )
                if lifecycle_status != "active":
                    raise DiagramStateError("cannot add a candidate to an archived diagram")
                normalized = normalize_graph(
                    raw_graph,
                    diagram_id,
                    diagram_type=str(diagram_type),
                    source_ref={
                        "source_kind": kind,
                        "artifact_ref_id": artifact_ref_id or "",
                        "source_fingerprint": fingerprint,
                        "source_locator": dict(source_locator or {}),
                    },
                )
                candidate_hash = canonical_graph_hash(normalized)
                cursor.execute(
                    """
                    INSERT INTO icp_diagram_candidates
                        (candidate_id, diagram_id, source_kind, artifact_ref_id,
                         source_locator, source_fingerprint, extractor_name,
                         extractor_version, model_name, prompt_version, raw_graph,
                         normalized_graph, candidate_hash, extraction_status, merge_status)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,'included')
                    ON CONFLICT (
                        diagram_id, source_kind, source_fingerprint,
                        extractor_version, candidate_hash
                    ) DO UPDATE SET
                        artifact_ref_id = COALESCE(EXCLUDED.artifact_ref_id, icp_diagram_candidates.artifact_ref_id),
                        source_locator = EXCLUDED.source_locator,
                        model_name = COALESCE(EXCLUDED.model_name, icp_diagram_candidates.model_name),
                        prompt_version = COALESCE(EXCLUDED.prompt_version, icp_diagram_candidates.prompt_version),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (
                        candidate_id,
                        diagram_id,
                        kind,
                        artifact_ref_id,
                        self._json(dict(source_locator or {})),
                        fingerprint,
                        clean_text(extractor_name) or "unknown",
                        clean_text(extractor_version) or "1",
                        clean_text(model_name) or None,
                        clean_text(prompt_version) or None,
                        self._json(dict(raw_graph)),
                        self._json(normalized),
                        candidate_hash,
                        extraction_status,
                    ),
                )
                result = self._row_to_dict(cursor, cursor.fetchone())
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def list_candidates(self, diagram_id: str) -> List[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM icp_diagram_candidates WHERE diagram_id=%s ORDER BY created_at,candidate_id",
                (diagram_id,),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def _load_candidates_for_update(
        self, cursor: Any, diagram_id: str, candidate_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        ids = sorted({clean_text(value) for value in candidate_ids if clean_text(value)})
        if not ids:
            raise ValueError("at least one candidate_id is required")
        cursor.execute(
            """
            SELECT * FROM icp_diagram_candidates
            WHERE diagram_id=%s AND candidate_id = ANY(%s)
            ORDER BY candidate_id
            FOR SHARE
            """,
            (diagram_id, ids),
        )
        rows = self._rows_to_dicts(cursor, cursor.fetchall())
        found = {str(item.get("candidate_id")) for item in rows}
        missing = sorted(set(ids) - found)
        if missing:
            raise DiagramNotFoundError(f"candidate(s) not found in diagram: {missing}")
        return rows

    @staticmethod
    def _merge_role(
        candidate_id: str,
        merge_result: Any,
        *,
        isolated_candidate_ids: Sequence[str] = (),
    ) -> str:
        if candidate_id in set(isolated_candidate_ids):
            return "isolated"
        if candidate_id in merge_result.duplicate_candidate_ids:
            return "duplicate"
        if any(candidate_id in conflict.candidate_ids for conflict in merge_result.conflicts):
            return "conflict"
        return "included"

    def create_or_reuse_revision(
        self,
        *,
        diagram_id: str,
        candidate_ids: Sequence[str],
        isolated_candidate_ids: Sequence[str] = (),
        schema_version: int = 1,
    ) -> Dict[str, Any]:
        """Build a pending/conflicted revision, or reuse an identical latest draft.

        This method never writes ``review_status='confirmed'``.  An existing
        latest confirmed revision can be returned only for the exact same
        candidate set, but its review state is neither created nor changed by
        automatic extraction.
        """

        self.init_tables()
        if schema_version <= 0:
            raise ValueError("schema_version must be positive")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM icp_diagrams WHERE diagram_id=%s FOR UPDATE",
                    (diagram_id,),
                )
                diagram_row = cursor.fetchone()
                if not diagram_row:
                    raise DiagramNotFoundError(f"diagram not found: {diagram_id}")
                diagram = self._row_to_dict(cursor, diagram_row)
                canonical_ids = sorted(
                    {clean_text(value) for value in candidate_ids if clean_text(value)}
                )
                isolated_ids = sorted(
                    {
                        clean_text(value)
                        for value in isolated_candidate_ids
                        if clean_text(value) and clean_text(value) not in canonical_ids
                    }
                )
                candidates = self._load_candidates_for_update(cursor, diagram_id, canonical_ids)
                isolated_candidates = (
                    self._load_candidates_for_update(cursor, diagram_id, isolated_ids)
                    if isolated_ids
                    else []
                )
                all_candidates = [*candidates, *isolated_candidates]
                merge_result = merge_candidates(
                    diagram_id,
                    str(diagram["diagram_type"]),
                    candidates,
                )
                if merge_result.review_status not in {REVIEW_PENDING, REVIEW_CONFLICTED}:
                    raise AssertionError("automatic merge attempted to create a confirmed revision")

                # Hash equality alone is not enough for reuse: a new source or
                # extractor may reproduce the same topology while adding
                # evidence/conflicts that have never been reviewed.  Only the
                # exact same candidate set is an idempotent retry.
                requested_candidate_ids = sorted(
                    str(item["candidate_id"]) for item in all_candidates
                )
                cursor.execute(
                    """
                    SELECT r.* FROM icp_diagram_revisions r
                    WHERE r.diagram_id=%s AND r.canonical_hash=%s
                      AND r.revision_no=(
                            SELECT MAX(latest.revision_no)
                            FROM icp_diagram_revisions latest
                            WHERE latest.diagram_id=r.diagram_id
                          )
                      AND COALESCE(
                            ARRAY(
                                SELECT rc.candidate_id
                                FROM icp_diagram_revision_candidates rc
                                WHERE rc.revision_id=r.revision_id
                                ORDER BY rc.candidate_id
                            ),
                            ARRAY[]::VARCHAR[]
                          ) = %s::VARCHAR[]
                    ORDER BY r.revision_no DESC
                    LIMIT 1
                    """,
                    (
                        diagram_id,
                        merge_result.canonical_hash,
                        requested_candidate_ids,
                    ),
                )
                existing = cursor.fetchone()
                reused = existing is not None
                if reused:
                    revision = self._row_to_dict(cursor, existing)
                    revision_id = str(revision["revision_id"])
                else:
                    cursor.execute(
                        "SELECT COALESCE(MAX(revision_no),0)+1 FROM icp_diagram_revisions WHERE diagram_id=%s",
                        (diagram_id,),
                    )
                    next_row = cursor.fetchone()
                    revision_no = int(next_row[0] if not isinstance(next_row, Mapping) else next_row["revision_no"])
                    revision_id = self._id_factory()
                    conflicts = [item.as_dict() for item in merge_result.conflicts]
                    validation_report = {
                        "valid": merge_result.conflict_count == 0,
                        "candidate_count": len(candidates),
                        "isolated_candidate_ids": isolated_ids,
                        "duplicate_candidate_ids": list(merge_result.duplicate_candidate_ids),
                        "candidate_hashes": dict(merge_result.candidate_hashes),
                    }
                    cursor.execute(
                        """
                        INSERT INTO icp_diagram_revisions
                            (revision_id,diagram_id,revision_no,schema_version,
                             canonical_graph,canonical_hash,review_status,
                             projection_status,conflicts,validation_report,conflict_count)
                        VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,'not_projected',%s::jsonb,%s::jsonb,%s)
                        RETURNING *
                        """,
                        (
                            revision_id,
                            diagram_id,
                            revision_no,
                            schema_version,
                            self._json(dict(merge_result.graph)),
                            merge_result.canonical_hash,
                            merge_result.review_status,
                            self._json(conflicts),
                            self._json(validation_report),
                            merge_result.conflict_count,
                        ),
                    )
                    revision = self._row_to_dict(cursor, cursor.fetchone())

                for candidate in all_candidates:
                    candidate_id = str(candidate["candidate_id"])
                    role = self._merge_role(
                        candidate_id,
                        merge_result,
                        isolated_candidate_ids=isolated_ids,
                    )
                    cursor.execute(
                        """
                        INSERT INTO icp_diagram_revision_candidates
                            (revision_id,candidate_id,merge_role)
                        VALUES (%s,%s,%s)
                        ON CONFLICT (revision_id,candidate_id) DO UPDATE SET
                            merge_role=EXCLUDED.merge_role
                        """,
                        (revision_id, candidate_id, role),
                    )
                    cursor.execute(
                        "UPDATE icp_diagram_candidates SET merge_status=%s,updated_at=CURRENT_TIMESTAMP WHERE candidate_id=%s",
                        (role, candidate_id),
                    )
                revision["reused"] = reused
                revision["candidate_ids"] = sorted(str(item["candidate_id"]) for item in candidates)
            self._commit()
            return revision
        except Exception:
            self._rollback()
            raise

    def get_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.*, d.doc_id, d.diagram_type, d.diagram_key, d.title,
                           (d.active_revision_id=r.revision_id) AS is_active
                    FROM icp_diagram_revisions r
                    JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                    WHERE r.revision_id=%s
                    """,
                    (revision_id,),
                )
                row = cursor.fetchone()
                result = self._row_to_dict(cursor, row) if row else None
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def list_revisions(self, diagram_id: str) -> List[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.*, (d.active_revision_id=r.revision_id) AS is_active
                FROM icp_diagram_revisions r
                JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                WHERE r.diagram_id=%s
                ORDER BY r.revision_no DESC
                """,
                (diagram_id,),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def get_revision_candidates(self, revision_id: str) -> List[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.*, rc.merge_role
                FROM icp_diagram_revision_candidates rc
                JOIN icp_diagram_candidates c ON c.candidate_id=rc.candidate_id
                WHERE rc.revision_id=%s
                ORDER BY c.created_at,c.candidate_id
                """,
                (revision_id,),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def auto_validate_revision(self, revision_id: str) -> Dict[str, Any]:
        """Apply the versioned unattended validator to the latest revision.

        This operation never writes ``review_status='confirmed'`` and never
        populates reviewer attribution.  Human review state remains available
        only for backward compatibility; automatic decisions live in the
        separate validation columns and report.
        """

        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.*, d.diagram_type, d.lifecycle_status,
                           doc.status AS document_status, latest.latest_revision_no
                    FROM icp_diagram_revisions r
                    JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                    JOIN icp_documents doc ON doc.doc_id=d.doc_id
                    JOIN LATERAL (
                        SELECT MAX(candidate.revision_no) AS latest_revision_no
                        FROM icp_diagram_revisions candidate
                        WHERE candidate.diagram_id=r.diagram_id
                    ) latest ON TRUE
                    WHERE r.revision_id=%s
                    FOR UPDATE OF r,d
                    """,
                    (revision_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramNotFoundError(f"revision not found: {revision_id}")
                current = self._row_to_dict(cursor, row)
                if str(current.get("lifecycle_status") or "") != "active":
                    raise DiagramStateError("archived diagram revision may not be auto-validated")
                if int(current["revision_no"]) != int(current["latest_revision_no"]):
                    raise DiagramStateError("only the latest diagram revision may be auto-validated")

                if current.get("review_status") == REVIEW_CONFIRMED:
                    cursor.execute(
                        """
                        UPDATE icp_diagram_revisions SET
                            validation_status='manual_confirmed',
                            validation_tier='human_review',
                            validation_method='manual',
                            validator_version=NULL,
                            validated_at=COALESCE(reviewed_at,CURRENT_TIMESTAMP),
                            quarantine_reason=NULL,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE revision_id=%s
                        RETURNING *
                        """,
                        (revision_id,),
                    )
                    result = self._row_to_dict(cursor, cursor.fetchone())
                    self._commit()
                    return result
                if current.get("review_status") == REVIEW_REJECTED:
                    cursor.execute(
                        """
                        UPDATE icp_diagram_revisions SET
                            validation_status='quarantined',
                            validation_tier='none', validation_method='manual',
                            validated_at=COALESCE(reviewed_at,CURRENT_TIMESTAMP),
                            quarantine_reason=COALESCE(NULLIF(review_note,''),'人工驳回'),
                            updated_at=CURRENT_TIMESTAMP
                        WHERE revision_id=%s
                        RETURNING *
                        """,
                        (revision_id,),
                    )
                    result = self._row_to_dict(cursor, cursor.fetchone())
                    self._commit()
                    return result

                cursor.execute(
                    """
                    SELECT c.*, rc.merge_role
                    FROM icp_diagram_revision_candidates rc
                    JOIN icp_diagram_candidates c ON c.candidate_id=rc.candidate_id
                    WHERE rc.revision_id=%s
                    ORDER BY c.created_at,c.candidate_id
                    """,
                    (revision_id,),
                )
                candidates = self._rows_to_dicts(cursor, cursor.fetchall())
                decision = evaluate_automatic_validation(
                    diagram_type=str(current.get("diagram_type") or ""),
                    graph=current.get("canonical_graph") or {},
                    candidates=candidates,
                    merge_conflicts=current.get("conflicts") or [],
                )
                previous_report = current.get("validation_report") or {}
                if not isinstance(previous_report, Mapping):
                    previous_report = {}
                report = {
                    **dict(previous_report),
                    "automatic_validation": decision.as_report(),
                    "canonical_hash": current.get("canonical_hash"),
                    "candidate_ids": [str(item.get("candidate_id") or "") for item in candidates],
                }
                quarantine_reason = "；".join(decision.reasons)[:4000]
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        validation_status=%s, validation_tier=%s,
                        validation_method=%s, validator_version=%s,
                        validation_report=%s::jsonb,
                        validated_at=CURRENT_TIMESTAMP,
                        quarantine_reason=%s,
                        lock_version=lock_version+1,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE revision_id=%s AND lock_version=%s
                    RETURNING *
                    """,
                    (
                        decision.status,
                        decision.tier,
                        decision.method,
                        decision.policy_version,
                        self._json(report),
                        quarantine_reason or None,
                        revision_id,
                        current["lock_version"],
                    ),
                )
                updated = cursor.fetchone()
                if not updated:
                    raise DiagramConcurrencyError("revision changed during automatic validation")
                result = self._row_to_dict(cursor, updated)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def auto_validate_document(
        self,
        doc_id: str,
        *,
        diagram_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate every latest diagram revision independently.

        A quarantined diagram does not block safe diagrams in the same
        document.  The returned IDs are ready for an idempotent projection
        attempt; no manual decision is required.
        """

        dtype = self._validate_diagram_type(diagram_type) if diagram_type else None
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for diagram in self.list_diagrams_by_document(doc_id):
            if dtype and str(diagram.get("diagram_type") or "") != dtype:
                continue
            revisions = self.list_revisions(str(diagram["diagram_id"]))
            if not revisions:
                continue
            revision_id = str(revisions[0]["revision_id"])
            try:
                results.append(self.auto_validate_revision(revision_id))
            except Exception as exc:
                errors.append({"revision_id": revision_id, "reason": str(exc)[:1000]})

        status_counts: Dict[str, int] = {}
        for item in results:
            status = str(item.get("validation_status") or VALIDATION_PENDING)
            status_counts[status] = status_counts.get(status, 0) + 1
        projectable_ids = [
            str(item["revision_id"])
            for item in results
            if revision_is_projectable(item)
            and str(item.get("projection_status") or "") != PROJECTION_PUBLISHED
        ]
        return {
            "doc_id": doc_id,
            "diagram_type": dtype or "",
            "policy_version": POLICY_VERSION,
            "validation_status_counts": status_counts,
            "projectable_revision_ids": projectable_ids,
            "quarantined_revision_ids": [
                str(item["revision_id"])
                for item in results
                if item.get("validation_status") == VALIDATION_QUARANTINED
            ],
            "results": results,
            "errors": errors,
        }

    def _reviewable_revision_for_update(
        self,
        cursor: Any,
        revision_id: str,
    ) -> Dict[str, Any]:
        """Lock and validate the complete authority boundary for human review."""

        cursor.execute(
            """
            SELECT r.*, d.diagram_type, d.lifecycle_status,
                   doc.status AS document_status, latest.latest_revision_no
            FROM icp_diagram_revisions r
            JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
            JOIN icp_documents doc ON doc.doc_id=d.doc_id
            JOIN LATERAL (
                SELECT MAX(candidate.revision_no) AS latest_revision_no
                FROM icp_diagram_revisions candidate
                WHERE candidate.diagram_id=r.diagram_id
            ) latest ON TRUE
            WHERE r.revision_id=%s
            FOR UPDATE OF doc,d,r
            """,
            (revision_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise DiagramNotFoundError(f"revision not found: {revision_id}")
        current = self._row_to_dict(cursor, row)
        if str(current.get("diagram_type") or "") not in _REVIEWABLE_DIAGRAM_TYPES:
            raise DiagramStateError("only FTA/RBD diagram revisions may be reviewed")
        if str(current.get("document_status") or "") not in _REVIEWABLE_DOCUMENT_STATUSES:
            raise DiagramStateError(
                "diagram review requires document status kg_pending or kg_done"
            )
        if str(current.get("lifecycle_status") or "") != "active":
            raise DiagramStateError("archived diagram revisions may not be reviewed")
        if int(current["revision_no"]) != int(current["latest_revision_no"]):
            raise DiagramStateError("only the latest diagram revision may be reviewed")
        return current

    def update_draft_revision(
        self,
        revision_id: str,
        canonical_graph: Mapping[str, Any],
        *,
        expected_lock_version: int,
        validation_report: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replace a review draft after an optional advanced correction."""

        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                current = self._reviewable_revision_for_update(cursor, revision_id)
                if current["review_status"] not in {REVIEW_PENDING, REVIEW_CONFLICTED}:
                    raise DiagramStateError("confirmed/rejected revisions are immutable")
                if int(current["lock_version"]) != int(expected_lock_version):
                    raise DiagramConcurrencyError("revision lock_version has changed")
                normalized = normalize_graph(
                    canonical_graph,
                    str(current["diagram_id"]),
                    diagram_type=str(current["diagram_type"]),
                )
                conflicts = detect_graph_conflicts(normalized)
                status = REVIEW_CONFLICTED if any(item.blocking for item in conflicts) else REVIEW_PENDING
                digest = canonical_graph_hash(normalized)
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        canonical_graph=%s::jsonb, canonical_hash=%s,
                        review_status=%s, conflicts=%s::jsonb,
                        validation_report=%s::jsonb, conflict_count=%s,
                        validation_status='pending', validation_tier='none',
                        validation_method='automatic', validator_version=NULL,
                        validated_at=NULL, quarantine_reason=NULL,
                        lock_version=lock_version+1, updated_at=CURRENT_TIMESTAMP
                    WHERE revision_id=%s AND lock_version=%s
                      AND review_status IN ('pending','conflicted')
                    RETURNING *
                    """,
                    (
                        self._json(normalized),
                        digest,
                        status,
                        self._json([item.as_dict() for item in conflicts]),
                        self._json(dict(validation_report or {})),
                        sum(1 for item in conflicts if item.blocking),
                        revision_id,
                        expected_lock_version,
                    ),
                )
                updated = cursor.fetchone()
                if not updated:
                    raise DiagramConcurrencyError("revision changed during update")
                result = self._row_to_dict(cursor, updated)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    @staticmethod
    def _reviewer(value: str) -> str:
        reviewer = clean_text(value)
        if not reviewer:
            raise ValueError("reviewed_by is required for an explicit review decision")
        return reviewer

    def confirm_revision(
        self,
        revision_id: str,
        *,
        reviewed_by: str,
        review_note: str = "",
        expected_lock_version: int,
    ) -> Dict[str, Any]:
        """Explicitly confirm one conflict-free draft; never called by extraction."""

        self.init_tables()
        reviewer = self._reviewer(reviewed_by)
        if expected_lock_version is None:
            raise ValueError("expected_lock_version is required for confirmation")
        try:
            with self.connection.cursor() as cursor:
                current = self._reviewable_revision_for_update(cursor, revision_id)
                if current["review_status"] != REVIEW_PENDING:
                    raise DiagramStateError("only a pending revision can be confirmed")
                if int(current.get("conflict_count") or 0) != 0:
                    raise DiagramStateError("a conflicted revision cannot be confirmed")
                if int(current["lock_version"]) != int(expected_lock_version):
                    raise DiagramConcurrencyError("revision lock_version has changed")
                graph_conflicts = detect_graph_conflicts(current.get("canonical_graph") or {})
                if any(item.blocking for item in graph_conflicts):
                    raise DiagramStateError("canonical graph failed structural validation")
                lock_version = int(current["lock_version"])
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        review_status='confirmed', reviewed_by=%s,
                        reviewed_at=CURRENT_TIMESTAMP, review_note=%s,
                        validation_status='manual_confirmed',
                        validation_tier='human_review', validation_method='manual',
                        validator_version=NULL, validated_at=CURRENT_TIMESTAMP,
                        quarantine_reason=NULL,
                        lock_version=lock_version+1, updated_at=CURRENT_TIMESTAMP
                    WHERE revision_id=%s AND review_status='pending'
                      AND conflict_count=0 AND lock_version=%s
                    RETURNING *
                    """,
                    (reviewer, clean_text(review_note), revision_id, lock_version),
                )
                updated = cursor.fetchone()
                if not updated:
                    raise DiagramConcurrencyError("revision changed during confirmation")
                result = self._row_to_dict(cursor, updated)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def reject_revision(
        self,
        revision_id: str,
        *,
        reviewed_by: str,
        review_note: str,
        expected_lock_version: int,
    ) -> Dict[str, Any]:
        self.init_tables()
        reviewer = self._reviewer(reviewed_by)
        if expected_lock_version is None:
            raise ValueError("expected_lock_version is required for rejection")
        note = clean_text(review_note)
        if not note:
            raise ValueError("review_note is required when rejecting a revision")
        try:
            with self.connection.cursor() as cursor:
                current = self._reviewable_revision_for_update(cursor, revision_id)
                if current["review_status"] not in {REVIEW_PENDING, REVIEW_CONFLICTED}:
                    raise DiagramStateError("only pending/conflicted revisions can be rejected")
                if int(current["lock_version"]) != int(expected_lock_version):
                    raise DiagramConcurrencyError("revision lock_version has changed")
                lock_version = int(current["lock_version"])
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        review_status='rejected', reviewed_by=%s,
                        reviewed_at=CURRENT_TIMESTAMP, review_note=%s,
                        validation_status='quarantined', validation_tier='none',
                        validation_method='manual', validator_version=NULL,
                        validated_at=CURRENT_TIMESTAMP, quarantine_reason=%s,
                        lock_version=lock_version+1, updated_at=CURRENT_TIMESTAMP
                    WHERE revision_id=%s AND review_status IN ('pending','conflicted')
                      AND lock_version=%s
                    RETURNING *
                    """,
                    (reviewer, note, note, revision_id, lock_version),
                )
                updated = cursor.fetchone()
                if not updated:
                    raise DiagramConcurrencyError("revision changed during rejection")
                result = self._row_to_dict(cursor, updated)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def get_active_revision(self, diagram_id: str) -> Optional[Dict[str, Any]]:
        """Return a human- or machine-validated published active revision."""

        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.*, d.doc_id, d.diagram_type, d.diagram_key, d.title,
                       TRUE AS is_active
                FROM icp_diagrams d
                JOIN icp_documents doc ON doc.doc_id=d.doc_id
                JOIN icp_diagram_revisions r
                  ON r.revision_id=d.active_revision_id AND r.diagram_id=d.diagram_id
                WHERE d.diagram_id=%s
                  AND d.lifecycle_status='active'
                  AND doc.status IN ('indexed','kg_pending','kg_done')
                  AND (
                        r.review_status='confirmed'
                        OR r.validation_status IN ('auto_validated','provisional','manual_confirmed')
                  )
                  AND r.projection_status='published'
                """,
                (diagram_id,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None

    def list_active_revisions(
        self,
        *,
        doc_id: Optional[str] = None,
        diagram_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Authoritative input for an idempotent HugeGraph rebuild."""

        self.init_tables()
        clauses = [
            "(r.review_status='confirmed' OR r.validation_status IN ('auto_validated','provisional','manual_confirmed'))",
            "r.projection_status='published'",
            "d.active_revision_id=r.revision_id",
            "d.lifecycle_status='active'",
            "doc.status IN ('indexed','kg_pending','kg_done')",
        ]
        params: List[Any] = []
        if doc_id:
            clauses.append("d.doc_id=%s")
            params.append(doc_id)
        if diagram_type:
            clauses.append("d.diagram_type=%s")
            params.append(self._validate_diagram_type(diagram_type))
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT r.*, d.doc_id, d.diagram_type, d.diagram_key, d.title,
                       TRUE AS is_active
                FROM icp_diagrams d
                JOIN icp_documents doc ON doc.doc_id=d.doc_id
                JOIN icp_diagram_revisions r
                  ON r.revision_id=d.active_revision_id AND r.diagram_id=d.diagram_id
                WHERE {' AND '.join(clauses)}
                ORDER BY d.diagram_id
                """,
                tuple(params),
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def start_projection(
        self,
        revision_id: str,
        *,
        idempotency_key: str,
        projector_version: str,
        expected_vertex_count: int,
        expected_edge_count: int,
    ) -> Dict[str, Any]:
        """Acquire the latest confirmed revision for deterministic projection.

        A retry with the same idempotency key may resume a running attempt.  A
        completed attempt is returned only when the same revision is still the
        published active revision; callers can then verify the physical graph
        without writing it again.
        """

        self.init_tables()
        key = clean_text(idempotency_key)
        version = clean_text(projector_version)
        if not key or not version:
            raise ValueError("idempotency_key and projector_version are required")
        if expected_vertex_count < 0 or expected_edge_count < 0:
            raise ValueError("expected projection counts must be non-negative")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.*, d.active_revision_id,d.lifecycle_status,
                           latest.latest_revision_no
                    FROM icp_diagram_revisions r
                    JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                    JOIN LATERAL (
                        SELECT MAX(candidate.revision_no) AS latest_revision_no
                        FROM icp_diagram_revisions candidate
                        WHERE candidate.diagram_id=r.diagram_id
                    ) latest ON TRUE
                    WHERE r.revision_id=%s FOR UPDATE OF r,d
                    """,
                    (revision_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramNotFoundError(f"revision not found: {revision_id}")
                revision = self._row_to_dict(cursor, row)
                if revision.get("lifecycle_status", "active") != "active":
                    raise DiagramStateError("archived diagram revision may not be projected")
                if not revision_is_projectable(revision):
                    raise DiagramStateError("only a validated revision may be projected")
                if int(revision["revision_no"]) != int(revision["latest_revision_no"]):
                    raise DiagramStateError("only the latest diagram revision may be projected")

                # Lock order is always diagram/revision -> run.  Activation and
                # failure use the same order, avoiding retry/activation deadlocks.
                cursor.execute(
                    "SELECT * FROM icp_graph_projection_runs WHERE idempotency_key=%s FOR UPDATE",
                    (key,),
                )
                existing_run_row = cursor.fetchone()
                existing_run = (
                    self._row_to_dict(cursor, existing_run_row)
                    if existing_run_row
                    else None
                )
                if existing_run:
                    if str(existing_run["revision_id"]) != revision_id:
                        raise DiagramStateError("idempotency_key belongs to another revision")
                    if clean_text(existing_run.get("canonical_hash")) != clean_text(
                        revision.get("canonical_hash")
                    ):
                        raise DiagramStateError("projection run canonical hash no longer matches")
                    if clean_text(existing_run.get("projector_version")) != version:
                        raise DiagramStateError("projection run uses another projector version")
                    if (
                        int(existing_run["expected_vertex_count"]) != expected_vertex_count
                        or int(existing_run["expected_edge_count"]) != expected_edge_count
                    ):
                        raise DiagramStateError("projection retry expected counts changed")
                    if existing_run["status"] == "succeeded":
                        if not (
                            revision["projection_status"] == PROJECTION_PUBLISHED
                            and str(revision.get("active_revision_id") or "") == revision_id
                        ):
                            raise DiagramStateError(
                                "succeeded projection run is not the active published revision"
                            )
                        existing_run["reused"] = True
                        existing_run["already_active"] = True
                        self._commit()
                        return existing_run
                    if existing_run["status"] == "running":
                        if revision["projection_status"] != PROJECTION_PUBLISHING:
                            raise DiagramStateError(
                                "running projection has no matching publishing revision"
                            )
                        existing_run["reused"] = True
                        existing_run["already_active"] = False
                        self._commit()
                        return existing_run

                if revision.get("active_revision_id") == revision_id:
                    raise DiagramStateError("active revision has no reusable succeeded run")
                if revision["projection_status"] not in {
                    PROJECTION_NOT_PROJECTED,
                    PROJECTION_FAILED,
                    PROJECTION_STALE,
                }:
                    raise DiagramStateError(
                        f"revision is not projectable from {revision['projection_status']}"
                    )
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        projection_status='publishing', lock_version=lock_version+1,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE revision_id=%s
                      AND (review_status='confirmed' OR validation_status IN ('auto_validated','provisional','manual_confirmed'))
                      AND projection_status IN ('not_projected','failed','stale')
                    RETURNING revision_id
                    """,
                    (revision_id,),
                )
                if not cursor.fetchone():
                    raise DiagramConcurrencyError("revision changed while acquiring projection")
                if existing_run:
                    cursor.execute(
                        """
                        UPDATE icp_graph_projection_runs SET
                            status='running', canonical_hash=%s, projector_version=%s,
                            expected_vertex_count=%s, expected_edge_count=%s,
                            actual_vertex_count=NULL, actual_edge_count=NULL,
                            manifest='{}'::jsonb, error=NULL, completed_at=NULL,
                            attempt_count=attempt_count+1,
                            started_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                        WHERE idempotency_key=%s RETURNING *
                        """,
                        (
                            revision["canonical_hash"],
                            version,
                            expected_vertex_count,
                            expected_edge_count,
                            key,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO icp_graph_projection_runs
                            (run_id,revision_id,idempotency_key,canonical_hash,
                             projector_version,status,expected_vertex_count,expected_edge_count)
                        VALUES (%s,%s,%s,%s,%s,'running',%s,%s)
                        RETURNING *
                        """,
                        (
                            self._id_factory(),
                            revision_id,
                            key,
                            revision["canonical_hash"],
                            version,
                            expected_vertex_count,
                            expected_edge_count,
                        ),
                    )
                run = self._row_to_dict(cursor, cursor.fetchone())
                run["reused"] = bool(existing_run)
                run["already_active"] = False
            self._commit()
            return run
        except Exception:
            self._rollback()
            raise

    def activate_projection(
        self,
        revision_id: str,
        *,
        run_id: str,
        manifest: Mapping[str, Any],
        actual_vertex_count: int,
        actual_edge_count: int,
    ) -> Dict[str, Any]:
        """Atomically expose a verified projection and stale the prior revision."""

        self.init_tables()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.*, d.active_revision_id,d.lifecycle_status,
                           latest.latest_revision_no
                    FROM icp_diagram_revisions r
                    JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                    JOIN LATERAL (
                        SELECT MAX(candidate.revision_no) AS latest_revision_no
                        FROM icp_diagram_revisions candidate
                        WHERE candidate.diagram_id=r.diagram_id
                    ) latest ON TRUE
                    WHERE r.revision_id=%s FOR UPDATE OF r,d
                    """,
                    (revision_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramNotFoundError(f"revision not found: {revision_id}")
                revision = self._row_to_dict(cursor, row)
                if revision.get("lifecycle_status", "active") != "active":
                    raise DiagramStateError("diagram was archived during projection")
                if not revision_is_projectable(revision):
                    raise DiagramStateError("projection target is no longer validated")
                if int(revision["revision_no"]) != int(revision["latest_revision_no"]):
                    raise DiagramStateError("a newer diagram revision exists; refusing activation")
                if revision["projection_status"] != PROJECTION_PUBLISHING:
                    raise DiagramStateError("projection target is not in publishing state")
                cursor.execute(
                    "SELECT * FROM icp_graph_projection_runs WHERE run_id=%s AND revision_id=%s FOR UPDATE",
                    (run_id, revision_id),
                )
                run_row = cursor.fetchone()
                if not run_row:
                    raise DiagramNotFoundError(f"projection run not found: {run_id}")
                run = self._row_to_dict(cursor, run_row)
                if run["status"] != "running":
                    raise DiagramStateError("projection run is not running")
                if clean_text(run.get("canonical_hash")) != clean_text(
                    revision.get("canonical_hash")
                ):
                    raise DiagramStateError("projection run canonical hash mismatch")
                if actual_vertex_count != int(run["expected_vertex_count"]) or actual_edge_count != int(
                    run["expected_edge_count"]
                ):
                    raise DiagramStateError("verified projection counts do not match the manifest")
                manifest_data = dict(manifest or {})
                supplied_manifest_hash = clean_text(manifest_data.pop("manifest_hash", ""))
                if not supplied_manifest_hash or supplied_manifest_hash != content_hash(manifest_data):
                    raise DiagramStateError("projection manifest hash is missing or invalid")
                if clean_text(manifest_data.get("revision_id")) != revision_id:
                    raise DiagramStateError("projection manifest revision mismatch")
                if clean_text(manifest_data.get("canonical_hash")) != clean_text(
                    revision.get("canonical_hash")
                ):
                    raise DiagramStateError("projection manifest canonical hash mismatch")
                vertex_ids = manifest_data.get("vertex_ids")
                edge_ids = manifest_data.get("edge_ids")
                if not isinstance(vertex_ids, list) or not isinstance(edge_ids, list):
                    raise DiagramStateError("projection manifest element ids are required")
                if len(set(map(str, vertex_ids))) != int(run["expected_vertex_count"]):
                    raise DiagramStateError("projection manifest vertex ids do not match expected count")
                if len(set(map(str, edge_ids))) != int(run["expected_edge_count"]):
                    raise DiagramStateError("projection manifest edge ids do not match expected count")
                old_revision_id = revision.get("active_revision_id")
                if old_revision_id and old_revision_id != revision_id:
                    cursor.execute(
                        """
                        UPDATE icp_diagram_revisions SET
                            projection_status='stale', updated_at=CURRENT_TIMESTAMP,
                            lock_version=lock_version+1
                        WHERE revision_id=%s AND diagram_id=%s
                          AND projection_status='published'
                        """,
                        (old_revision_id, revision["diagram_id"]),
                    )
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        projection_status='published', published_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP, lock_version=lock_version+1
                    WHERE revision_id=%s
                      AND (review_status='confirmed' OR validation_status IN ('auto_validated','provisional','manual_confirmed'))
                      AND projection_status='publishing'
                    """,
                    (revision_id,),
                )
                if cursor.rowcount != 1:
                    raise DiagramConcurrencyError("revision changed during activation")
                cursor.execute(
                    """
                    UPDATE icp_diagrams SET active_revision_id=%s,updated_at=CURRENT_TIMESTAMP
                    WHERE diagram_id=%s
                    """,
                    (revision_id, revision["diagram_id"]),
                )
                cursor.execute(
                    """
                    UPDATE icp_graph_projection_runs SET
                        status='succeeded',actual_vertex_count=%s,actual_edge_count=%s,
                        manifest=%s::jsonb,error=NULL,completed_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE run_id=%s AND status='running'
                    RETURNING *
                    """,
                    (
                        actual_vertex_count,
                        actual_edge_count,
                        self._json(dict(manifest)),
                        run_id,
                    ),
                )
                completed = cursor.fetchone()
                if not completed:
                    raise DiagramConcurrencyError("projection run changed during activation")
                result = self._row_to_dict(cursor, completed)
                result["active_revision_id"] = revision_id
                result["superseded_revision_id"] = old_revision_id
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def invalidate_active_projection(
        self,
        revision_id: str,
        *,
        run_id: str,
        error: str,
    ) -> Dict[str, Any]:
        """Fail closed when an active HugeGraph projection no longer reads back.

        A previously succeeded run can become physically incomplete after an
        operator action or storage fault.  Clearing the active pointer and
        marking both revision and run failed in one transaction prevents reads
        from treating that projection as valid.  ``start_projection`` can then
        reuse the same idempotency key to rebuild the immutable revision.
        """

        self.init_tables()
        message = clean_text(error) or "active projection failed verification"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT r.*, d.active_revision_id, d.lifecycle_status
                    FROM icp_diagram_revisions r
                    JOIN icp_diagrams d ON d.diagram_id=r.diagram_id
                    WHERE r.revision_id=%s FOR UPDATE OF r,d
                    """,
                    (revision_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramNotFoundError(f"revision not found: {revision_id}")
                revision = self._row_to_dict(cursor, row)
                if not (
                    revision.get("lifecycle_status", "active") == "active"
                    and revision_is_projectable(revision)
                    and revision.get("projection_status") == PROJECTION_PUBLISHED
                    and str(revision.get("active_revision_id") or "") == revision_id
                ):
                    raise DiagramStateError(
                        "only a validated published active projection may be invalidated"
                    )
                cursor.execute(
                    """
                    SELECT * FROM icp_graph_projection_runs
                    WHERE run_id=%s AND revision_id=%s FOR UPDATE
                    """,
                    (run_id, revision_id),
                )
                run_row = cursor.fetchone()
                if not run_row:
                    raise DiagramNotFoundError(f"projection run not found: {run_id}")
                run = self._row_to_dict(cursor, run_row)
                if run.get("status") != "succeeded":
                    raise DiagramStateError("only a succeeded projection run may be invalidated")

                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        projection_status='failed', updated_at=CURRENT_TIMESTAMP,
                        lock_version=lock_version+1
                    WHERE revision_id=%s
                      AND (review_status='confirmed' OR validation_status IN ('auto_validated','provisional','manual_confirmed'))
                      AND projection_status='published'
                    """,
                    (revision_id,),
                )
                if cursor.rowcount != 1:
                    raise DiagramConcurrencyError("projection changed during invalidation")
                cursor.execute(
                    """
                    UPDATE icp_diagrams SET
                        active_revision_id=NULL, updated_at=CURRENT_TIMESTAMP
                    WHERE diagram_id=%s AND active_revision_id=%s
                    """,
                    (revision["diagram_id"], revision_id),
                )
                if cursor.rowcount != 1:
                    raise DiagramConcurrencyError("active revision changed during invalidation")
                cursor.execute(
                    """
                    UPDATE icp_graph_projection_runs SET
                        status='failed', error=%s, completed_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE run_id=%s AND revision_id=%s AND status='succeeded'
                    RETURNING *
                    """,
                    (message, run_id, revision_id),
                )
                invalidated = cursor.fetchone()
                if not invalidated:
                    raise DiagramConcurrencyError("projection run changed during invalidation")
                result = self._row_to_dict(cursor, invalidated)
                result["active_revision_id"] = None
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def fail_projection(self, revision_id: str, *, run_id: str, error: str) -> Dict[str, Any]:
        self.init_tables()
        message = clean_text(error) or "projection failed"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE icp_diagram_revisions SET
                        projection_status='failed',updated_at=CURRENT_TIMESTAMP,
                        lock_version=lock_version+1
                    WHERE revision_id=%s
                      AND (review_status='confirmed' OR validation_status IN ('auto_validated','provisional','manual_confirmed'))
                      AND projection_status='publishing'
                    RETURNING revision_id
                    """,
                    (revision_id,),
                )
                if not cursor.fetchone():
                    raise DiagramStateError("projection target is not an active publishing attempt")
                cursor.execute(
                    """
                    UPDATE icp_graph_projection_runs SET
                        status='failed',error=%s,completed_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE run_id=%s AND revision_id=%s AND status='running'
                    RETURNING *
                    """,
                    (message, run_id, revision_id),
                )
                row = cursor.fetchone()
                if not row:
                    raise DiagramStateError("projection run is not active")
                result = self._row_to_dict(cursor, row)
            self._commit()
            return result
        except Exception:
            self._rollback()
            raise

    def get_projection_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        self.init_tables()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM icp_graph_projection_runs WHERE run_id=%s", (run_id,))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None


# Lazy singleton for application wiring.  Constructing it has no network or DB
# side effect; the existing connection is resolved on first repository method.
diagram_repository = DiagramRepository()


__all__ = [
    "DiagramConcurrencyError",
    "DiagramNotFoundError",
    "DiagramRepository",
    "DiagramRepositoryError",
    "DiagramStateError",
    "diagram_repository",
]
