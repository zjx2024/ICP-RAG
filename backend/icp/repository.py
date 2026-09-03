"""ICP 领域结构化数据仓储。

该模块只复用现有 PostgreSQL 连接，不改变通用知识库与 Dify 接口。
用于保存 Function Locator 与故障历史的结构化结果，并向 ICP 专用工具提供只读查询。
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)


class IcpRepository:
    def __init__(self) -> None:
        self._initialized = False

    def init_tables(self) -> None:
        """创建 ICP 专用结构表，幂等执行。"""
        if self._initialized:
            return
        try:
            with postgres_client.conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS icp_functions (
                        id               BIGSERIAL PRIMARY KEY,
                        doc_id           VARCHAR(36) NOT NULL,
                        function_id      VARCHAR(100) NOT NULL,
                        system_name      VARCHAR(200),
                        subsystem_name   VARCHAR(200),
                        module_id        VARCHAR(100),
                        module_name      VARCHAR(200),
                        parent_function_id VARCHAR(100),
                        function_name    VARCHAR(300) NOT NULL,
                        description      TEXT,
                        aliases          JSONB DEFAULT '[]'::jsonb,
                        inputs           JSONB DEFAULT '[]'::jsonb,
                        outputs          JSONB DEFAULT '[]'::jsonb,
                        failure_symptoms TEXT,
                        source_ref       VARCHAR(500),
                        raw_data         JSONB NOT NULL DEFAULT '{}'::jsonb,
                        normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, function_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_doc ON icp_functions(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_name ON icp_functions(function_name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_module ON icp_functions(module_name)")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS module_id VARCHAR(100)")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS parent_function_id VARCHAR(100)")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS inputs JSONB DEFAULT '[]'::jsonb")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS outputs JSONB DEFAULT '[]'::jsonb")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")
                cur.execute("ALTER TABLE icp_functions ADD COLUMN IF NOT EXISTS normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS icp_components (
                        id               BIGSERIAL PRIMARY KEY,
                        doc_id           VARCHAR(36) NOT NULL,
                        component_id     VARCHAR(100) NOT NULL,
                        parent_component_id VARCHAR(100),
                        component_type   VARCHAR(32) NOT NULL,
                        name             VARCHAR(300) NOT NULL,
                        abbreviation     VARCHAR(100),
                        description      TEXT,
                        source_ref       VARCHAR(1024),
                        raw_data         JSONB NOT NULL DEFAULT '{}'::jsonb,
                        normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, component_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_components_doc ON icp_components(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_components_name ON icp_components(name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_components_parent ON icp_components(doc_id, parent_component_id)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS icp_interfaces (
                        id                  BIGSERIAL PRIMARY KEY,
                        doc_id              VARCHAR(36) NOT NULL,
                        interface_id        VARCHAR(160) NOT NULL,
                        source_component_id VARCHAR(100),
                        target_component_id VARCHAR(100),
                        relation_type       VARCHAR(50) NOT NULL,
                        signal_name         VARCHAR(300),
                        direction           VARCHAR(30),
                        protocol            VARCHAR(200),
                        description         TEXT,
                        source_ref          VARCHAR(1024),
                        raw_data            JSONB NOT NULL DEFAULT '{}'::jsonb,
                        normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, interface_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_interfaces_doc ON icp_interfaces(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_interfaces_source ON icp_interfaces(doc_id, source_component_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_interfaces_target ON icp_interfaces(doc_id, target_component_id)")
                cur.execute("ALTER TABLE icp_interfaces ADD COLUMN IF NOT EXISTS protocol VARCHAR(200)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fault_history_cases (
                        id                BIGSERIAL PRIMARY KEY,
                        doc_id            VARCHAR(36) NOT NULL,
                        case_id           VARCHAR(100) NOT NULL,
                        occurred_at       VARCHAR(200),
                        personnel         VARCHAR(300),
                        device_type       VARCHAR(300),
                        fault_symptom     TEXT NOT NULL,
                        related_module    VARCHAR(300),
                        related_function  VARCHAR(300),
                        diagnosis_method  TEXT,
                        root_cause        TEXT,
                        mission_phase     VARCHAR(300),
                        final_solution    TEXT,
                        verification_result TEXT,
                        preventive_action TEXT,
                        hmc_code          VARCHAR(100),
                        fmeca_code        VARCHAR(100),
                        fta_ref           VARCHAR(200),
                        rbd_ref           VARCHAR(200),
                        mfl_code          VARCHAR(100),
                        source_ref        VARCHAR(500),
                        document_links    JSONB NOT NULL DEFAULT '{}'::jsonb,
                        raw_data          JSONB NOT NULL DEFAULT '{}'::jsonb,
                        normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, case_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_doc ON fault_history_cases(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_case ON fault_history_cases(case_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_hmc ON fault_history_cases(hmc_code)")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS mission_phase VARCHAR(300)")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS related_function VARCHAR(300)")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS verification_result TEXT")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS rbd_ref VARCHAR(200)")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS document_links JSONB NOT NULL DEFAULT '{}'::jsonb")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")
                cur.execute("ALTER TABLE fault_history_cases ADD COLUMN IF NOT EXISTS normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fault_history_steps (
                        id         BIGSERIAL PRIMARY KEY,
                        case_pk    BIGINT NOT NULL REFERENCES fault_history_cases(id) ON DELETE CASCADE,
                        step_no    INTEGER NOT NULL,
                        content    TEXT NOT NULL,
                        source_ref VARCHAR(1024),
                        raw_data   JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (case_pk, step_no)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_steps_case ON fault_history_steps(case_pk)")
                cur.execute("ALTER TABLE fault_history_steps ADD COLUMN IF NOT EXISTS source_ref VARCHAR(1024)")
                cur.execute("ALTER TABLE fault_history_steps ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")

                # 旧删除流程最终都会删除 icp_documents。通过数据库触发器清理新增领域表，
                # 避免在通用删除服务中引入 ICP 特例，并确保失败重试仍然幂等。
                cur.execute(
                    """
                    CREATE OR REPLACE FUNCTION cleanup_icp_domain_rows()
                    RETURNS trigger AS $$
                    BEGIN
                        DELETE FROM icp_functions WHERE doc_id = OLD.doc_id;
                        DELETE FROM icp_components WHERE doc_id = OLD.doc_id;
                        DELETE FROM icp_interfaces WHERE doc_id = OLD.doc_id;
                        DELETE FROM fault_history_cases WHERE doc_id = OLD.doc_id;
                        RETURN OLD;
                    END;
                    $$ LANGUAGE plpgsql
                    """
                )
                cur.execute("DROP TRIGGER IF EXISTS trg_cleanup_icp_domain_rows ON icp_documents")
                cur.execute(
                    """
                    CREATE TRIGGER trg_cleanup_icp_domain_rows
                    AFTER DELETE ON icp_documents
                    FOR EACH ROW EXECUTE FUNCTION cleanup_icp_domain_rows()
                    """
                )
            postgres_client.conn.commit()
            self._initialized = True
        except Exception as exc:
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass
            logger.warning("ICP domain table initialization failed: %s", exc)

    @staticmethod
    def _row_to_dict(cur, row) -> Dict[str, Any]:
        if row is None:
            return {}
        return {desc[0]: value for desc, value in zip(cur.description, row)}

    @staticmethod
    def _rows_to_dicts(cur, rows) -> List[Dict[str, Any]]:
        columns = [desc[0] for desc in cur.description]
        return [{name: value for name, value in zip(columns, row)} for row in rows]

    def upsert_function(self, doc_id: str, item: Dict[str, Any]) -> int:
        self.init_tables()
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [v.strip() for v in aliases.replace("；", ";").split(";") if v.strip()]
        function_id = str(item.get("function_id") or item.get("code") or item.get("function_name") or "").strip()
        function_name = str(item.get("function_name") or item.get("name") or function_id).strip()
        if not function_id or not function_name:
            raise ValueError("function_id/function_name 不能为空")
        inputs = item.get("inputs") or []
        outputs = item.get("outputs") or []
        if isinstance(inputs, str):
            inputs = [value.strip() for value in inputs.replace("；", ";").split(";") if value.strip()]
        if isinstance(outputs, str):
            outputs = [value.strip() for value in outputs.replace("；", ";").split(";") if value.strip()]
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icp_functions
                    (doc_id, function_id, system_name, subsystem_name, module_id, module_name,
                     parent_function_id, function_name, description, aliases, inputs, outputs,
                     failure_symptoms, source_ref, raw_data, normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, function_id) DO UPDATE SET
                    system_name=EXCLUDED.system_name,
                    subsystem_name=EXCLUDED.subsystem_name,
                    module_id=EXCLUDED.module_id,
                    module_name=EXCLUDED.module_name,
                    parent_function_id=EXCLUDED.parent_function_id,
                    function_name=EXCLUDED.function_name,
                    description=EXCLUDED.description,
                    aliases=EXCLUDED.aliases,
                    inputs=EXCLUDED.inputs,
                    outputs=EXCLUDED.outputs,
                    failure_symptoms=EXCLUDED.failure_symptoms,
                    source_ref=EXCLUDED.source_ref,
                    raw_data=EXCLUDED.raw_data,
                    normalization_warnings=EXCLUDED.normalization_warnings,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    doc_id,
                    function_id,
                    item.get("system_name"),
                    item.get("subsystem_name"),
                    item.get("module_id"),
                    item.get("module_name"),
                    item.get("parent_function_id"),
                    function_name,
                    item.get("description"),
                    json.dumps(aliases, ensure_ascii=False),
                    json.dumps(inputs, ensure_ascii=False),
                    json.dumps(outputs, ensure_ascii=False),
                    item.get("failure_symptoms"),
                    item.get("source_ref"),
                    json.dumps(item.get("raw_data") or {}, ensure_ascii=False),
                    json.dumps(item.get("normalization_warnings") or [], ensure_ascii=False),
                ),
            )
            row_id = int(cur.fetchone()[0])
        postgres_client.conn.commit()
        return row_id

    def upsert_component(self, doc_id: str, item: Dict[str, Any]) -> int:
        """保存系统/子系统/模块权威记录，不从自由段落推测父子关系。"""
        self.init_tables()
        component_id = str(item.get("component_id") or item.get("id") or "").strip()
        name = str(item.get("name") or item.get("component_name") or "").strip()
        component_type = str(item.get("component_type") or "").strip().lower()
        if not component_id or not name or component_type not in {"system", "subsystem", "module"}:
            raise ValueError("component_id/name/component_type(system|subsystem|module) 不能为空")
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icp_components
                    (doc_id, component_id, parent_component_id, component_type, name,
                     abbreviation, description, source_ref, raw_data, normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, component_id) DO UPDATE SET
                    parent_component_id=EXCLUDED.parent_component_id,
                    component_type=EXCLUDED.component_type,
                    name=EXCLUDED.name,
                    abbreviation=EXCLUDED.abbreviation,
                    description=EXCLUDED.description,
                    source_ref=EXCLUDED.source_ref,
                    raw_data=EXCLUDED.raw_data,
                    normalization_warnings=EXCLUDED.normalization_warnings,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    doc_id, component_id, item.get("parent_component_id"), component_type, name,
                    item.get("abbreviation"), item.get("description"), item.get("source_ref"),
                    json.dumps(item.get("raw_data") or {}, ensure_ascii=False),
                    json.dumps(item.get("normalization_warnings") or [], ensure_ascii=False),
                ),
            )
            row_id = int(cur.fetchone()[0])
        postgres_client.conn.commit()
        return row_id

    def upsert_interface(self, doc_id: str, item: Dict[str, Any]) -> int:
        """保存表格中明确声明的接口/数据流，不接受相似度推断边。"""
        self.init_tables()
        interface_id = str(item.get("interface_id") or "").strip()
        relation_type = str(item.get("relation_type") or "connects_to").strip()
        if not interface_id:
            raise ValueError("interface_id 不能为空")
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icp_interfaces
                    (doc_id, interface_id, source_component_id, target_component_id,
                     relation_type, signal_name, direction, description, source_ref,
                     protocol, raw_data, normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, interface_id) DO UPDATE SET
                    source_component_id=EXCLUDED.source_component_id,
                    target_component_id=EXCLUDED.target_component_id,
                    relation_type=EXCLUDED.relation_type,
                    signal_name=EXCLUDED.signal_name,
                    direction=EXCLUDED.direction,
                    protocol=EXCLUDED.protocol,
                    description=EXCLUDED.description,
                    source_ref=EXCLUDED.source_ref,
                    raw_data=EXCLUDED.raw_data,
                    normalization_warnings=EXCLUDED.normalization_warnings,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    doc_id, interface_id, item.get("source_component_id"),
                    item.get("target_component_id"), relation_type, item.get("signal_name"),
                    item.get("direction"), item.get("description"), item.get("source_ref"),
                    item.get("protocol"),
                    json.dumps(item.get("raw_data") or {}, ensure_ascii=False),
                    json.dumps(item.get("normalization_warnings") or [], ensure_ascii=False),
                ),
            )
            row_id = int(cur.fetchone()[0])
        postgres_client.conn.commit()
        return row_id

    def upsert_history_case(self, doc_id: str, item: Dict[str, Any], steps: Optional[Iterable[Any]] = None) -> int:
        self.init_tables()
        case_id = str(item.get("case_id") or item.get("fault_id") or "").strip()
        symptom = str(item.get("fault_symptom") or item.get("symptom") or "").strip()
        if not case_id or not symptom:
            raise ValueError("case_id/fault_symptom 不能为空")
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fault_history_cases
                    (doc_id, case_id, occurred_at, personnel, device_type, fault_symptom,
                     related_module, related_function, diagnosis_method, root_cause, mission_phase, final_solution,
                     verification_result, preventive_action, hmc_code, fmeca_code, fta_ref,
                     rbd_ref, mfl_code, source_ref, document_links, raw_data,
                     normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, case_id) DO UPDATE SET
                    occurred_at=EXCLUDED.occurred_at,
                    personnel=EXCLUDED.personnel,
                    device_type=EXCLUDED.device_type,
                    fault_symptom=EXCLUDED.fault_symptom,
                    related_module=EXCLUDED.related_module,
                    related_function=EXCLUDED.related_function,
                    diagnosis_method=EXCLUDED.diagnosis_method,
                    root_cause=EXCLUDED.root_cause,
                    mission_phase=EXCLUDED.mission_phase,
                    final_solution=EXCLUDED.final_solution,
                    verification_result=EXCLUDED.verification_result,
                    preventive_action=EXCLUDED.preventive_action,
                    hmc_code=EXCLUDED.hmc_code,
                    fmeca_code=EXCLUDED.fmeca_code,
                    fta_ref=EXCLUDED.fta_ref,
                    rbd_ref=EXCLUDED.rbd_ref,
                    mfl_code=EXCLUDED.mfl_code,
                    source_ref=EXCLUDED.source_ref,
                    document_links=EXCLUDED.document_links,
                    raw_data=EXCLUDED.raw_data,
                    normalization_warnings=EXCLUDED.normalization_warnings,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    doc_id,
                    case_id,
                    item.get("occurred_at"),
                    item.get("personnel"),
                    item.get("device_type"),
                    symptom,
                    item.get("related_module"),
                    item.get("related_function"),
                    item.get("diagnosis_method"),
                    item.get("root_cause"),
                    item.get("mission_phase"),
                    item.get("final_solution"),
                    item.get("verification_result"),
                    item.get("preventive_action"),
                    item.get("hmc_code"),
                    item.get("fmeca_code"),
                    item.get("fta_ref"),
                    item.get("rbd_ref"),
                    item.get("mfl_code"),
                    item.get("source_ref"),
                    json.dumps(item.get("document_links") or {}, ensure_ascii=False),
                    json.dumps(item.get("raw_data") or {}, ensure_ascii=False),
                    json.dumps(item.get("normalization_warnings") or [], ensure_ascii=False),
                ),
            )
            case_pk = int(cur.fetchone()[0])
            if steps is not None:
                cur.execute("DELETE FROM fault_history_steps WHERE case_pk=%s", (case_pk,))
                used_step_numbers = set()
                for idx, content in enumerate(steps, start=1):
                    if isinstance(content, dict):
                        text = str(content.get("content") or "").strip()
                        original_step_no = int(content.get("step_no") or idx)
                        source_ref = str(content.get("source_ref") or "").strip()
                        raw_data = {**content, "original_step_no": original_step_no}
                    else:
                        text = str(content or "").strip()
                        original_step_no = idx
                        source_ref = ""
                        raw_data = {"content": text}
                    if not text:
                        continue
                    step_no = original_step_no
                    if step_no <= 0 or step_no in used_step_numbers:
                        step_no = idx
                        while step_no in used_step_numbers:
                            step_no += 1
                    used_step_numbers.add(step_no)
                    cur.execute(
                        """INSERT INTO fault_history_steps
                           (case_pk, step_no, content, source_ref, raw_data)
                           VALUES (%s,%s,%s,%s,%s::jsonb)""",
                        (case_pk, step_no, text, source_ref,
                         json.dumps(raw_data, ensure_ascii=False)),
                    )
        postgres_client.conn.commit()
        return case_pk

    def search_functions(self, terms: Iterable[str], limit: int = 8) -> List[Dict[str, Any]]:
        self.init_tables()
        clean = [str(v).strip() for v in terms if str(v).strip()]
        if not clean:
            return []
        conditions: List[str] = []
        params: List[Any] = []
        for term in clean[:8]:
            pattern = f"%{term}%"
            conditions.append(
                "(f.function_id ILIKE %s OR f.function_name ILIKE %s OR f.description ILIKE %s OR f.system_name ILIKE %s "
                "OR f.subsystem_name ILIKE %s OR f.module_name ILIKE %s OR f.failure_symptoms ILIKE %s "
                "OR f.aliases::text ILIKE %s)"
            )
            params.extend([pattern] * 8)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT f.* FROM icp_functions f "
                "JOIN icp_documents d ON d.doc_id=f.doc_id "
                f"WHERE d.status IN ('indexed','kg_pending','kg_done') AND "
                f"({' OR '.join(conditions)}) "
                "ORDER BY f.updated_at DESC LIMIT %s",
                tuple(params),
            )
            return self._rows_to_dicts(cur, cur.fetchall())

    def list_collection_scope_candidates(self) -> List[Dict[str, Any]]:
        """Return every active Function Locator identity eligible for scoping.

        Complete-collection questions must not decide that a name is unique
        from a relevance-limited search window.  This unbounded identity-only
        query is intentionally used only for that workflow.  It returns no
        descriptions or free text, and includes document applicability so
        duplicate versions/configurations remain visible as ambiguity instead
        of being silently hidden by ``LIMIT`` ordering.
        """

        self.init_tables()
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                WITH active_function_documents AS (
                    SELECT
                        doc_id,
                        metadata,
                        COUNT(*) OVER (
                            PARTITION BY
                                NULLIF(BTRIM(metadata->>'applicable_system'), ''),
                                NULLIF(BTRIM(metadata->>'applicable_configuration'), '')
                        ) AS active_scope_version_count
                    FROM icp_documents
                    WHERE doc_type='function_locator'
                      AND status IN ('indexed','kg_pending','kg_done')
                )
                SELECT
                    'component'::text AS collection_entity_kind,
                    c.doc_id,
                    c.component_id AS collection_entity_id,
                    c.name AS collection_entity_name,
                    c.abbreviation AS collection_entity_abbreviation,
                    d.metadata->>'applicable_system' AS applicable_system,
                    d.metadata->>'applicable_configuration'
                        AS applicable_configuration,
                    d.active_scope_version_count
                FROM icp_components c
                JOIN active_function_documents d ON d.doc_id=c.doc_id
                UNION ALL
                SELECT
                    'function'::text AS collection_entity_kind,
                    f.doc_id,
                    f.function_id AS collection_entity_id,
                    f.function_name AS collection_entity_name,
                    NULL::text AS collection_entity_abbreviation,
                    d.metadata->>'applicable_system' AS applicable_system,
                    d.metadata->>'applicable_configuration'
                        AS applicable_configuration,
                    d.active_scope_version_count
                FROM icp_functions f
                JOIN active_function_documents d ON d.doc_id=f.doc_id
                ORDER BY collection_entity_kind, collection_entity_id, doc_id
                """
            )
            return self._rows_to_dicts(cur, cur.fetchall())

    def search_components(self, terms: Iterable[str], limit: int = 12) -> List[Dict[str, Any]]:
        self.init_tables()
        clean = [str(value).strip() for value in terms if str(value).strip()]
        if not clean:
            return []
        conditions: List[str] = []
        params: List[Any] = []
        for term in clean[:8]:
            pattern = f"%{term}%"
            conditions.append(
                "(c.component_id ILIKE %s OR c.name ILIKE %s OR c.abbreviation ILIKE %s "
                "OR c.description ILIKE %s)"
            )
            params.extend([pattern] * 4)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT c.* FROM icp_components c "
                "JOIN icp_documents d ON d.doc_id=c.doc_id "
                f"WHERE d.status IN ('indexed','kg_pending','kg_done') AND "
                f"({' OR '.join(conditions)}) "
                "ORDER BY c.updated_at DESC LIMIT %s",
                tuple(params),
            )
            return self._rows_to_dicts(cur, cur.fetchall())

    def get_component_children(self, doc_id: str, component_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """按明确父 ID 扩展一层，不根据名称生成关系。"""
        self.init_tables()
        ids = [str(value).strip() for value in component_ids if str(value).strip()]
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM icp_components WHERE doc_id=%s AND parent_component_id IN ({placeholders}) "
                "ORDER BY component_type, component_id",
                (doc_id, *ids),
            )
            return self._rows_to_dicts(cur, cur.fetchall())

    def expand_architecture_inventory(
        self,
        seed_rows: Sequence[Mapping[str, Any]],
        *,
        limit: int = 16,
    ) -> List[Dict[str, Any]]:
        """读取命中系统的完整直接子系统清单。

        该查询只沿同一 Function Locator 文档中的明确组件 ID 父链，或用
        功能记录已经保存的系统全名精确匹配系统组件。它不会用名称相似度
        猜测父子关系。每个返回成员都携带同一父系统下的确定性成员总数，
        供回答层执行“总数与逐项枚举一致”的闭集校验。
        """

        self.init_tables()
        bounded_limit = min(max(int(limit or 0), 1), 32)
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for raw in seed_rows[:64]:
            if not isinstance(raw, Mapping):
                continue
            doc_id = str(raw.get("doc_id") or "").strip()
            if not doc_id:
                continue
            if doc_id not in grouped and len(grouped) >= 4:
                continue
            grouped.setdefault(doc_id, []).append(raw)

        results: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for doc_id, rows in grouped.items():
            if len(results) >= bounded_limit:
                break
            component_ids = list(
                dict.fromkeys(
                    str(row.get(key) or "").strip()
                    for row in rows
                    for key in (
                        "component_id",
                        "module_id",
                        "parent_component_id",
                        "source_component_id",
                        "target_component_id",
                    )
                    if str(row.get(key) or "").strip()
                )
            )[:32]
            system_names = list(
                dict.fromkeys(
                    str(row.get("system_name") or "").strip()
                    for row in rows
                    if str(row.get("system_name") or "").strip()
                )
            )[:8]
            if not component_ids and not system_names:
                continue

            remaining = bounded_limit - len(results)
            with postgres_client.conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE active_document AS (
                        SELECT doc_id FROM icp_documents
                        WHERE doc_id=%s AND status IN ('indexed','kg_pending','kg_done')
                    ),
                    ancestors AS (
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id = ANY(%s::text[])
                        UNION
                        SELECT parent.* FROM icp_components parent
                        JOIN ancestors child
                          ON parent.doc_id=child.doc_id
                         AND parent.component_id=child.parent_component_id
                    ),
                    roots AS (
                        SELECT * FROM ancestors WHERE component_type='system'
                        UNION
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_type='system' AND c.name = ANY(%s::text[])
                    )
                    SELECT c.*,
                           r.component_id AS inventory_root_id,
                           (
                               SELECT COUNT(*)
                               FROM icp_components member
                               WHERE member.doc_id=r.doc_id
                                 AND member.parent_component_id=r.component_id
                                 AND member.component_type='subsystem'
                           ) AS inventory_total
                    FROM icp_components c
                    JOIN roots r
                      ON c.doc_id=r.doc_id
                     AND (
                          c.component_id=r.component_id
                          OR (
                              c.parent_component_id=r.component_id
                              AND c.component_type='subsystem'
                          )
                     )
                    ORDER BY
                        CASE WHEN c.component_type='system' THEN 0 ELSE 1 END,
                        c.component_id
                    LIMIT %s
                    """,
                    (doc_id, component_ids, system_names, remaining),
                )
                rows_found = self._rows_to_dicts(cur, cur.fetchall())

            parents = {
                str(row.get("component_id") or ""): row
                for row in rows_found
                if str(row.get("component_type") or "") == "system"
            }
            totals = {
                str(row.get("inventory_root_id") or ""): int(
                    row.get("inventory_total") or 0
                )
                for row in rows_found
                if str(row.get("inventory_root_id") or "")
            }
            children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows_found:
                if str(row.get("component_type") or "") != "subsystem":
                    continue
                parent_id = str(row.get("parent_component_id") or "").strip()
                if parent_id in parents:
                    children_by_parent.setdefault(parent_id, []).append(row)

            for parent_id, parent in parents.items():
                children = children_by_parent.get(parent_id, [])
                inventory_count = totals.get(parent_id, 0)
                # Never label a truncated bounded query as a complete list.
                # A partial group remains available through the ordinary
                # architecture expansion, but it cannot trigger the closed-set
                # answer contract below.
                if not children or len(children) != inventory_count:
                    continue
                ordered = [parent, *children]
                for row in ordered:
                    component_id = str(row.get("component_id") or "").strip()
                    identity = (doc_id, component_id)
                    if not component_id or identity in seen:
                        continue
                    seen.add(identity)
                    item = dict(row)
                    item.pop("inventory_root_id", None)
                    item.pop("inventory_total", None)
                    item.update(
                        {
                            "content_type": "function_component",
                            # Generic closed-collection contract. Other
                            # structured repositories (for example an exact
                            # module→failure-mode query) can emit the same
                            # fields without teaching the answer layer a new
                            # document-specific completeness rule.
                            "collection_closed": True,
                            "collection_parent_id": parent_id,
                            "collection_parent_name": str(parent.get("name") or ""),
                            "collection_member_kind": "subsystem",
                            "collection_member_count": inventory_count,
                            "collection_role": (
                                "parent" if component_id == parent_id else "member"
                            ),
                            "collection_member_id": (
                                component_id if component_id != parent_id else ""
                            ),
                            "collection_member_name": (
                                str(row.get("name") or "")
                                if component_id != parent_id
                                else ""
                            ),
                            "collection_member_description": (
                                str(row.get("description") or "")
                                if component_id != parent_id
                                else ""
                            ),
                        }
                    )
                    results.append(item)
                    if len(results) >= bounded_limit:
                        return results
        return results

    def expand_fmeca_fault_mode_collection(
        self,
        doc_id: str,
        *,
        component_ids: Iterable[str] = (),
        function_ids: Iterable[str] = (),
        limit: int = 64,
    ) -> List[Dict[str, Any]]:
        """Return the authoritative FMECA failure-mode set for an exact scope.

        ``doc_id`` identifies the Function Locator document that supplied the
        seed IDs.  The scope is resolved solely from explicit stored IDs: a
        component seed includes itself and its recorded descendants, while a
        function seed includes itself and its recorded descendants.
        ``module_id`` is the only component-to-function bridge.  Ancestors and
        sibling functions are deliberately excluded from membership: they are
        useful context, but including them would make a leaf module look as if
        it owned its parent system's failure modes.  A FMECA row belongs to the
        resulting scope only when ``product_flag`` contains one of the resolved
        function or component business IDs as a complete code token; names, abbreviations and
        semantic similarity never participate.

        The method deliberately searches the active FMECA document whose
        controlled applicability metadata exactly matches the Function Locator
        document.  It exposes the resolved ID sets and applicability decision
        on every returned row for audit.  A row is marked
        ``collection_closed=True`` only when the scope is non-empty, exactly
        one applicable FMECA document exists, business member IDs are complete
        and unique, at least one authoritative FMECA row matched, and the
        requested limit did not truncate that complete matching set.  Missing
        applicability or parallel document versions therefore cannot be
        presented as an exhaustive inventory.
        """

        self.init_tables()
        scope_doc_id = str(doc_id or "").strip()
        if not scope_doc_id:
            return []

        def _clean_ids(values: Iterable[str], *, maximum: int = 64) -> List[str]:
            cleaned: List[str] = []
            seen: set[str] = set()
            for raw in values:
                value = str(raw or "").strip().upper()
                if not value or value in seen:
                    continue
                seen.add(value)
                cleaned.append(value)
                if len(cleaned) >= maximum:
                    break
            return cleaned

        seed_component_ids = _clean_ids(component_ids)
        seed_function_ids = _clean_ids(function_ids)
        if not seed_component_ids and not seed_function_ids:
            return []
        bounded_limit = min(max(int(limit or 0), 1), 256)

        # ``regexp_split_to_table`` below retains the identifier characters
        # accepted by the graph projection (letters, numbers, dot, underscore,
        # hyphen).  Joining those tokens to the resolved Function IDs prevents
        # a FUNC-53 seed from accidentally matching FUNC-530.
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE
                active_scope_document AS (
                    SELECT
                        doc_id,
                        NULLIF(BTRIM(metadata->>'applicable_system'), '')
                            AS applicable_system,
                        NULLIF(BTRIM(metadata->>'applicable_configuration'), '')
                            AS applicable_configuration
                    FROM icp_documents
                    WHERE doc_id=%s
                      AND doc_type='function_locator'
                      AND status IN ('indexed','kg_pending','kg_done')
                ),
                all_document_entity_ids AS (
                    SELECT DISTINCT UPPER(BTRIM(function_id)) AS entity_id
                    FROM icp_functions f
                    JOIN active_scope_document d ON d.doc_id=f.doc_id
                    WHERE NULLIF(BTRIM(function_id), '') IS NOT NULL
                    UNION
                    SELECT DISTINCT UPPER(BTRIM(component_id)) AS entity_id
                    FROM icp_components c
                    JOIN active_scope_document d ON d.doc_id=c.doc_id
                    WHERE NULLIF(BTRIM(component_id), '') IS NOT NULL
                ),
                direct_seed_functions AS (
                    SELECT f.*
                    FROM icp_functions f
                    JOIN active_scope_document d ON d.doc_id=f.doc_id
                    WHERE UPPER(BTRIM(f.function_id)) = ANY(%s::text[])
                ),
                direct_seed_components AS (
                    SELECT c.*
                    FROM icp_components c
                    JOIN active_scope_document d ON d.doc_id=c.doc_id
                    WHERE UPPER(BTRIM(c.component_id)) = ANY(%s::text[])
                ),
                component_seed_roots AS (
                    SELECT * FROM direct_seed_components
                ),
                declared_component_edges AS (
                    SELECT DISTINCT
                        parent.doc_id AS edge_doc_id,
                        UPPER(BTRIM(parent.component_id)) AS edge_parent_id,
                        child.id AS child_row_id
                    FROM icp_components parent
                    JOIN active_scope_document d ON d.doc_id=parent.doc_id
                    JOIN icp_components child ON child.doc_id=parent.doc_id
                    WHERE NULLIF(BTRIM(parent.component_id), '') IS NOT NULL
                      AND NULLIF(BTRIM(child.parent_component_id), '') IS NULL
                      AND LOWER(BTRIM(parent.component_type)) = 'subsystem'
                      AND LOWER(BTRIM(child.component_type)) = 'module'
                      AND NULLIF(BTRIM(child.abbreviation), '') IS NOT NULL
                      AND EXISTS (
                          SELECT 1
                          FROM regexp_split_to_table(
                              UPPER(COALESCE(
                                  parent.raw_data->'raw_values'->>'包含模块',
                                  ''
                              )),
                              '[^A-Z0-9._-]+'
                          ) AS declared(value)
                          WHERE declared.value =
                              UPPER(BTRIM(child.abbreviation))
                      )
                ),
                unique_declared_component_edges AS (
                    SELECT declared.*
                    FROM declared_component_edges declared
                    WHERE 1 = (
                        SELECT COUNT(DISTINCT candidate.edge_parent_id)
                        FROM declared_component_edges candidate
                        WHERE candidate.edge_doc_id=declared.edge_doc_id
                          AND candidate.child_row_id=declared.child_row_id
                    )
                ),
                component_edges AS (
                    SELECT DISTINCT
                        parent.doc_id AS edge_doc_id,
                        UPPER(BTRIM(parent.component_id)) AS edge_parent_id,
                        child.id AS child_row_id
                    FROM icp_components parent
                    JOIN active_scope_document d ON d.doc_id=parent.doc_id
                    JOIN icp_components child
                      ON child.doc_id=parent.doc_id
                     AND NULLIF(BTRIM(child.parent_component_id), '') IS NOT NULL
                     AND UPPER(BTRIM(child.parent_component_id)) =
                         UPPER(BTRIM(parent.component_id))
                    WHERE NULLIF(BTRIM(parent.component_id), '') IS NOT NULL
                    UNION
                    SELECT edge_doc_id, edge_parent_id, child_row_id
                    FROM unique_declared_component_edges
                ),
                component_descendants AS (
                    SELECT * FROM component_seed_roots
                    UNION
                    SELECT child.*
                    FROM component_descendants parent
                    JOIN component_edges edge
                      ON edge.edge_doc_id=parent.doc_id
                     AND edge.edge_parent_id=UPPER(BTRIM(parent.component_id))
                    JOIN icp_components child ON child.id=edge.child_row_id
                ),
                scoped_component_ids AS (
                    SELECT DISTINCT UPPER(BTRIM(component_id)) AS component_id
                    FROM component_descendants
                    WHERE NULLIF(BTRIM(component_id), '') IS NOT NULL
                ),
                component_functions AS (
                    SELECT f.*
                    FROM icp_functions f
                    JOIN active_scope_document d ON d.doc_id=f.doc_id
                    WHERE UPPER(BTRIM(f.module_id)) IN (
                        SELECT component_id FROM scoped_component_ids
                    )
                ),
                function_seed_rows AS (
                    SELECT * FROM direct_seed_functions
                    UNION
                    SELECT * FROM component_functions
                ),
                function_descendants AS (
                    SELECT * FROM function_seed_rows
                    UNION
                    SELECT child.*
                    FROM icp_functions child
                    JOIN function_descendants parent
                      ON child.doc_id=parent.doc_id
                     AND UPPER(BTRIM(child.parent_function_id)) = UPPER(BTRIM(parent.function_id))
                ),
                scoped_function_ids AS (
                    SELECT DISTINCT UPPER(BTRIM(function_id)) AS function_id
                    FROM function_descendants
                    WHERE UPPER(BTRIM(function_id))
                        ~ '^FUNC-[A-Z0-9](?:[A-Z0-9._-]*[A-Z0-9])?$'
                ),
                resolved_component_ids AS (
                    SELECT component_id FROM scoped_component_ids
                    UNION
                    SELECT DISTINCT UPPER(BTRIM(module_id))
                    FROM function_descendants
                    WHERE NULLIF(BTRIM(module_id), '') IS NOT NULL
                ),
                scope_summary AS (
                    SELECT
                        ARRAY(
                            SELECT function_id FROM scoped_function_ids
                            ORDER BY function_id
                        ) AS collection_function_ids,
                        ARRAY(
                            SELECT component_id FROM resolved_component_ids
                            ORDER BY component_id
                        ) AS collection_component_ids,
                        ARRAY(
                            SELECT DISTINCT UPPER(BTRIM(function_id))
                            FROM direct_seed_functions
                            ORDER BY UPPER(BTRIM(function_id))
                        ) AS collection_seed_function_ids,
                        ARRAY(
                            SELECT DISTINCT UPPER(BTRIM(component_id))
                            FROM direct_seed_components
                            ORDER BY UPPER(BTRIM(component_id))
                        ) AS collection_seed_component_ids,
                        (
                            SELECT function_name FROM direct_seed_functions
                            ORDER BY function_id
                            LIMIT 1
                        ) AS collection_seed_function_name,
                        (
                            SELECT name FROM direct_seed_components
                            ORDER BY component_id
                            LIMIT 1
                        ) AS collection_seed_component_name
                ),
                candidate_fmeca_documents AS (
                    SELECT
                        candidate.doc_id,
                        scope.applicable_system,
                        scope.applicable_configuration
                    FROM active_scope_document scope
                    JOIN icp_documents candidate
                      ON candidate.doc_type='fmeca'
                     AND candidate.status IN ('indexed','kg_pending','kg_done')
                     AND scope.applicable_system IS NOT NULL
                     AND scope.applicable_configuration IS NOT NULL
                     AND NULLIF(
                         BTRIM(candidate.metadata->>'applicable_system'), ''
                     ) = scope.applicable_system
                     AND NULLIF(
                         BTRIM(candidate.metadata->>'applicable_configuration'), ''
                     ) = scope.applicable_configuration
                ),
                candidate_document_summary AS (
                    SELECT
                        COUNT(*) AS collection_candidate_document_count,
                        COALESCE(
                            ARRAY_AGG(doc_id::text ORDER BY doc_id),
                            ARRAY[]::text[]
                        ) AS collection_candidate_document_ids,
                        MAX(applicable_system) AS collection_applicable_system,
                        MAX(applicable_configuration)
                            AS collection_applicable_configuration
                    FROM candidate_fmeca_documents
                ),
                candidate_fmeca_row_mapping AS (
                    SELECT
                        f.id AS fmea_id,
                        EXISTS (
                            SELECT 1
                            FROM regexp_split_to_table(
                                UPPER(COALESCE(f.product_flag, '')),
                                '[^A-Z0-9._-]+'
                            ) AS token(value)
                            JOIN all_document_entity_ids entity
                              ON token.value=entity.entity_id
                        ) AS has_canonical_entity
                    FROM fmea f
                    JOIN candidate_fmeca_documents d ON d.doc_id=f.doc_id
                ),
                candidate_mapping_summary AS (
                    SELECT
                        COUNT(*) FILTER (WHERE NOT has_canonical_entity)
                            AS collection_unmapped_row_count
                    FROM candidate_fmeca_row_mapping
                ),
                scoped_entity_ids AS (
                    SELECT function_id AS entity_id, 'function'::text AS entity_kind
                    FROM scoped_function_ids
                    UNION ALL
                    SELECT component_id AS entity_id, 'component'::text AS entity_kind
                    FROM scoped_component_ids
                ),
                matching_tokens AS (
                    SELECT f.id AS fmea_id,
                           COALESCE(
                               ARRAY_AGG(DISTINCT token.value ORDER BY token.value)
                                   FILTER (WHERE scope.entity_kind='function'),
                               ARRAY[]::text[]
                           ) AS matched_function_ids,
                           COALESCE(
                               ARRAY_AGG(DISTINCT token.value ORDER BY token.value)
                                   FILTER (WHERE scope.entity_kind='component'),
                               ARRAY[]::text[]
                           ) AS matched_component_ids
                    FROM fmea f
                    JOIN candidate_fmeca_documents d ON d.doc_id=f.doc_id
                    CROSS JOIN LATERAL regexp_split_to_table(
                        UPPER(COALESCE(f.product_flag, '')),
                        '[^A-Z0-9._-]+'
                    ) AS token(value)
                    JOIN scoped_entity_ids scope
                      ON token.value=scope.entity_id
                    GROUP BY f.id
                ),
                matched_rows AS (
                    SELECT f.*, tokens.matched_function_ids,
                           tokens.matched_component_ids
                    FROM fmea f
                    JOIN matching_tokens tokens ON tokens.fmea_id=f.id
                ),
                collection_summary AS (
                    SELECT
                        COUNT(*) AS collection_total,
                        COUNT(DISTINCT UPPER(BTRIM(code)))
                            AS collection_distinct_member_total,
                        COALESCE(
                            BOOL_AND(NULLIF(BTRIM(code), '') IS NOT NULL),
                            FALSE
                        ) AS collection_member_ids_present,
                        COALESCE(
                            BOOL_AND(NULLIF(BTRIM(fault_mode), '') IS NOT NULL),
                            FALSE
                        ) AS collection_member_names_present
                    FROM matched_rows
                )
                SELECT matched_rows.*,
                       summary.collection_function_ids,
                       summary.collection_component_ids,
                       summary.collection_seed_function_ids,
                       summary.collection_seed_component_ids,
                       summary.collection_seed_function_name,
                       summary.collection_seed_component_name,
                       collection.collection_total,
                       documents.collection_candidate_document_count,
                       documents.collection_candidate_document_ids,
                       documents.collection_applicable_system,
                       documents.collection_applicable_configuration,
                       (
                           documents.collection_candidate_document_count = 1
                           AND documents.collection_applicable_system IS NOT NULL
                           AND documents.collection_applicable_configuration IS NOT NULL
                       ) AS collection_applicability_complete,
                       (
                           collection.collection_member_ids_present
                           AND collection.collection_member_names_present
                           AND collection.collection_total =
                               collection.collection_distinct_member_total
                       ) AS collection_member_ids_unique,
                       mapping.collection_unmapped_row_count
                FROM matched_rows
                CROSS JOIN scope_summary summary
                CROSS JOIN collection_summary collection
                CROSS JOIN candidate_document_summary documents
                CROSS JOIN candidate_mapping_summary mapping
                ORDER BY matched_rows.doc_id, matched_rows.code, matched_rows.id
                LIMIT %s
                """,
                (scope_doc_id, seed_function_ids, seed_component_ids, bounded_limit),
            )
            rows = self._rows_to_dicts(cur, cur.fetchall())

        def _id_list(value: Any) -> List[str]:
            if isinstance(value, (list, tuple, set)):
                values = value
            elif value is None:
                values = ()
            else:
                values = (value,)
            result: List[str] = []
            for raw in values:
                item = str(raw or "").strip()
                if item and item not in result:
                    result.append(item)
            return result

        if not rows:
            return []

        # The SQL window count describes the full exact-match result before
        # LIMIT.  It is the only completeness signal used here: no partial
        # result can acquire the closed-collection contract by coincidence.
        try:
            total = int(rows[0].get("collection_total") or 0)
        except (TypeError, ValueError):
            total = 0
        resolved_function_ids = _id_list(rows[0].get("collection_function_ids"))
        resolved_component_ids = _id_list(rows[0].get("collection_component_ids"))
        direct_function_ids = _id_list(rows[0].get("collection_seed_function_ids"))
        direct_component_ids = _id_list(rows[0].get("collection_seed_component_ids"))
        try:
            candidate_document_count = int(
                rows[0].get("collection_candidate_document_count") or 0
            )
        except (TypeError, ValueError):
            candidate_document_count = 0

        def _as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

        applicability_complete = _as_bool(
            rows[0].get("collection_applicability_complete")
        )
        member_ids_unique = _as_bool(rows[0].get("collection_member_ids_unique"))
        try:
            unmapped_row_count = int(
                rows[0].get("collection_unmapped_row_count") or 0
            )
        except (TypeError, ValueError):
            unmapped_row_count = -1
        is_closed = bool(
            (resolved_function_ids or resolved_component_ids)
            and total > 0
            and total == len(rows)
            and total <= bounded_limit
            and applicability_complete
            and candidate_document_count == 1
            and member_ids_unique
            and unmapped_row_count == 0
        )

        # A single explicit seed is the clearest human-facing scope label.  A
        # multi-seed union still has a stable parent identity, but is labelled
        # generically so that a response never pretends it is one component.
        if len(direct_component_ids) == 1 and not direct_function_ids:
            collection_parent_id = direct_component_ids[0]
            collection_parent_name = str(
                rows[0].get("collection_seed_component_name") or collection_parent_id
            ).strip()
        elif len(direct_function_ids) == 1 and not direct_component_ids:
            collection_parent_id = direct_function_ids[0]
            collection_parent_name = str(
                rows[0].get("collection_seed_function_name") or collection_parent_id
            ).strip()
        elif len(resolved_function_ids) == 1:
            collection_parent_id = resolved_function_ids[0]
            collection_parent_name = collection_parent_id
        else:
            scope_signature = "\x1f".join(
                (scope_doc_id, *resolved_component_ids, "|", *resolved_function_ids)
            )
            collection_parent_id = (
                "fmeca_scope:" + hashlib.sha256(scope_signature.encode("utf-8")).hexdigest()[:24]
            )
            collection_parent_name = f"功能范围（{len(resolved_function_ids)} 个功能）"

        prepared: List[Dict[str, Any]] = []
        for raw_row in rows:
            row = dict(raw_row)
            row.pop("collection_total", None)
            row.update(
                {
                    "collection_closed": is_closed,
                    "collection_parent_id": collection_parent_id,
                    "collection_parent_name": collection_parent_name,
                    "collection_member_kind": "failure_mode",
                    "collection_member_count": total,
                    "collection_role": "member",
                    "collection_member_id": str(row.get("code") or "").strip(),
                    "collection_member_name": str(row.get("fault_mode") or "").strip(),
                    "collection_member_description": str(row.get("fault_cause") or "").strip(),
                    "collection_function_ids": resolved_function_ids,
                    "collection_component_ids": resolved_component_ids,
                    "collection_seed_function_ids": direct_function_ids,
                    "collection_seed_component_ids": direct_component_ids,
                    "collection_scope_doc_id": scope_doc_id,
                    "collection_candidate_document_count": candidate_document_count,
                    "collection_candidate_document_ids": _id_list(
                        row.get("collection_candidate_document_ids")
                    ),
                    "collection_applicable_system": str(
                        row.get("collection_applicable_system") or ""
                    ).strip(),
                    "collection_applicable_configuration": str(
                        row.get("collection_applicable_configuration") or ""
                    ).strip(),
                    "collection_applicability_complete": applicability_complete,
                    "collection_member_ids_unique": member_ids_unique,
                    "collection_unmapped_row_count": unmapped_row_count,
                    "collection_matched_function_ids": _id_list(
                        row.get("matched_function_ids")
                    ),
                    "collection_matched_component_ids": _id_list(
                        row.get("matched_component_ids")
                    ),
                }
            )
            prepared.append(row)
        return prepared

    def expand_function_context(
        self,
        seed_rows: Sequence[Mapping[str, Any]],
        *,
        limit: int = 24,
    ) -> List[Dict[str, Any]]:
        """沿 Function Locator 的明确 ID 补齐有界架构上下文。

        语义检索往往只命中一条功能记录，但回答数据流或层级问题还需要
        它的父功能、同一父功能下的兄弟功能、实现模块、父组件，以及与
        这些组件直接相连的接口。这里仅沿 PostgreSQL 已保存的业务 ID
        与父子/端点字段扩一圈；不读取名称相似度，也不补造缺失关系。
        """

        self.init_tables()
        bounded_limit = min(max(int(limit or 0), 1), 64)
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for raw in seed_rows[:bounded_limit]:
            if not isinstance(raw, Mapping):
                continue
            doc_id = str(raw.get("doc_id") or "").strip()
            if not doc_id:
                continue
            if doc_id not in grouped and len(grouped) >= 4:
                continue
            grouped.setdefault(doc_id, []).append(raw)

        results: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def values(rows: Sequence[Mapping[str, Any]], *keys: str) -> List[str]:
            selected: List[str] = []
            for row in rows:
                for key in keys:
                    value = str(row.get(key) or "").strip()
                    if value and value not in selected:
                        selected.append(value)
            return selected[:bounded_limit]

        for doc_id, rows in list(grouped.items())[:4]:
            remaining = bounded_limit - len(results)
            if remaining <= 0:
                break
            function_ids = values(rows, "function_id")
            parent_function_ids = values(rows, "parent_function_id")
            component_ids = values(
                rows,
                "component_id",
                "module_id",
                "parent_component_id",
                "source_component_id",
                "target_component_id",
            )
            interface_ids = values(rows, "interface_id")
            if not (function_ids or parent_function_ids or component_ids or interface_ids):
                continue

            with postgres_client.conn.cursor() as cur:
                cur.execute(
                    """
                    WITH active_document AS (
                        SELECT doc_id FROM icp_documents
                        WHERE doc_id=%s AND status IN ('indexed','kg_pending','kg_done')
                    ),
                    seed_function_family AS (
                        SELECT f.* FROM icp_functions f
                        JOIN active_document d ON d.doc_id=f.doc_id
                        WHERE f.function_id = ANY(%s::text[])
                           OR f.parent_function_id = ANY(%s::text[])
                           OR f.function_id = ANY(%s::text[])
                           OR f.parent_function_id = ANY(%s::text[])
                           OR f.module_id = ANY(%s::text[])
                    ),
                    preliminary_component_ids(component_id) AS (
                        SELECT unnest(%s::text[])
                        UNION
                        SELECT module_id FROM seed_function_family
                        WHERE NULLIF(BTRIM(module_id), '') IS NOT NULL
                    ),
                    preliminary_component_base AS (
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (SELECT component_id FROM preliminary_component_ids)
                           OR c.parent_component_id IN (SELECT component_id FROM preliminary_component_ids)
                    ),
                    preliminary_parent_ids(component_id) AS (
                        SELECT parent_component_id FROM preliminary_component_base
                        WHERE NULLIF(BTRIM(parent_component_id), '') IS NOT NULL
                    ),
                    preliminary_components AS (
                        SELECT * FROM preliminary_component_base
                        UNION
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (SELECT component_id FROM preliminary_parent_ids)
                    ),
                    component_functions AS (
                        SELECT f.* FROM icp_functions f
                        JOIN active_document d ON d.doc_id=f.doc_id
                        WHERE f.module_id IN (
                            SELECT component_id FROM preliminary_components
                        )
                    ),
                    related_function_parent_ids(function_id) AS (
                        SELECT parent_function_id FROM seed_function_family
                        WHERE NULLIF(BTRIM(parent_function_id), '') IS NOT NULL
                        UNION
                        SELECT parent_function_id FROM component_functions
                        WHERE NULLIF(BTRIM(parent_function_id), '') IS NOT NULL
                        UNION
                        SELECT unnest(%s::text[])
                    ),
                    related_functions AS (
                        SELECT * FROM seed_function_family
                        UNION
                        SELECT * FROM component_functions
                        UNION
                        SELECT f.* FROM icp_functions f
                        JOIN active_document d ON d.doc_id=f.doc_id
                        WHERE f.function_id IN (
                                  SELECT function_id FROM related_function_parent_ids
                              )
                           OR f.parent_function_id IN (
                                  SELECT function_id FROM related_function_parent_ids
                              )
                    ),
                    function_component_ids(component_id) AS (
                        SELECT component_id FROM preliminary_component_ids
                        UNION
                        SELECT module_id FROM related_functions
                        WHERE NULLIF(BTRIM(module_id), '') IS NOT NULL
                    ),
                    function_components AS (
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (SELECT component_id FROM function_component_ids)
                           OR c.parent_component_id IN (SELECT component_id FROM function_component_ids)
                    ),
                    function_component_parent_ids(component_id) AS (
                        SELECT parent_component_id FROM function_components
                        WHERE NULLIF(BTRIM(parent_component_id), '') IS NOT NULL
                    ),
                    core_components AS (
                        SELECT * FROM preliminary_components
                        UNION
                        SELECT * FROM function_components
                        UNION
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (
                            SELECT component_id FROM function_component_parent_ids
                        )
                    ),
                    adjacent_interfaces AS (
                        SELECT i.* FROM icp_interfaces i
                        JOIN active_document d ON d.doc_id=i.doc_id
                        WHERE i.interface_id = ANY(%s::text[])
                           OR i.source_component_id IN (SELECT component_id FROM core_components)
                           OR i.target_component_id IN (SELECT component_id FROM core_components)
                    ),
                    endpoint_components AS (
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (
                            SELECT source_component_id FROM adjacent_interfaces
                            UNION
                            SELECT target_component_id FROM adjacent_interfaces
                        )
                    ),
                    endpoint_parent_ids(component_id) AS (
                        SELECT parent_component_id FROM endpoint_components
                        WHERE NULLIF(BTRIM(parent_component_id), '') IS NOT NULL
                    ),
                    all_components AS (
                        SELECT * FROM core_components
                        UNION
                        SELECT * FROM endpoint_components
                        UNION
                        SELECT c.* FROM icp_components c
                        JOIN active_document d ON d.doc_id=c.doc_id
                        WHERE c.component_id IN (SELECT component_id FROM endpoint_parent_ids)
                    ),
                    combined AS (
                        SELECT 0 AS kind_order, c.component_id AS business_id,
                               'function_component'::text AS content_type, to_jsonb(c) AS payload
                        FROM all_components c
                        UNION ALL
                        SELECT 1, i.interface_id, 'function_interface', to_jsonb(i)
                        FROM adjacent_interfaces i
                        UNION ALL
                        SELECT 2, f.function_id, 'function_record', to_jsonb(f)
                        FROM related_functions f
                    )
                    SELECT content_type, payload
                    FROM combined
                    ORDER BY kind_order, business_id
                    LIMIT %s
                    """,
                    (
                        doc_id,
                        function_ids,
                        function_ids,
                        parent_function_ids,
                        parent_function_ids,
                        component_ids,
                        component_ids,
                        parent_function_ids,
                        interface_ids,
                        remaining,
                    ),
                )
                for content_type, payload in cur.fetchall():
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except Exception:
                            continue
                    if not isinstance(payload, dict):
                        continue
                    item = dict(payload)
                    item["content_type"] = str(content_type)
                    business_id = str(
                        item.get("function_id")
                        or item.get("component_id")
                        or item.get("interface_id")
                        or ""
                    )
                    identity = (doc_id, str(content_type), business_id)
                    if not business_id or identity in seen:
                        continue
                    seen.add(identity)
                    results.append(item)
                    if len(results) >= bounded_limit:
                        return results
        return results

    def search_interfaces(self, terms: Iterable[str], limit: int = 12) -> List[Dict[str, Any]]:
        """查询原文明确声明的接口；不根据名称或相似度补造连边。"""
        self.init_tables()
        clean = [str(value).strip() for value in terms if str(value).strip()]
        if not clean:
            return []
        conditions: List[str] = []
        params: List[Any] = []
        for term in clean[:8]:
            pattern = f"%{term}%"
            conditions.append(
                "(i.interface_id ILIKE %s OR i.source_component_id ILIKE %s "
                "OR i.target_component_id ILIKE %s OR i.signal_name ILIKE %s "
                "OR i.description ILIKE %s)"
            )
            params.extend([pattern] * 5)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT i.* FROM icp_interfaces i "
                "JOIN icp_documents d ON d.doc_id=i.doc_id "
                f"WHERE d.status IN ('indexed','kg_pending','kg_done') AND "
                f"({' OR '.join(conditions)}) "
                "ORDER BY i.updated_at DESC LIMIT %s",
                tuple(params),
            )
            return self._rows_to_dicts(cur, cur.fetchall())

    def search_history(self, terms: Iterable[str], limit: int = 8) -> List[Dict[str, Any]]:
        self.init_tables()
        clean = [str(v).strip() for v in terms if str(v).strip()]
        if not clean:
            return []
        conditions: List[str] = []
        params: List[Any] = []
        for term in clean[:8]:
            pattern = f"%{term}%"
            conditions.append(
                "(h.case_id ILIKE %s OR h.device_type ILIKE %s OR h.fault_symptom ILIKE %s "
                "OR h.related_module ILIKE %s OR h.related_function ILIKE %s "
                "OR h.diagnosis_method ILIKE %s OR h.root_cause ILIKE %s "
                "OR h.final_solution ILIKE %s OR h.hmc_code ILIKE %s OR h.fmeca_code ILIKE %s "
                "OR h.fta_ref ILIKE %s OR h.rbd_ref ILIKE %s OR h.mfl_code ILIKE %s "
                "OR h.occurred_at ILIKE %s OR h.personnel ILIKE %s OR h.mission_phase ILIKE %s "
                "OR h.verification_result ILIKE %s OR h.preventive_action ILIKE %s)"
            )
            params.extend([pattern] * 18)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT h.* FROM fault_history_cases h "
                "JOIN icp_documents d ON d.doc_id=h.doc_id "
                f"WHERE d.status IN ('indexed','kg_pending','kg_done') AND "
                f"({' OR '.join(conditions)}) "
                "ORDER BY h.updated_at DESC LIMIT %s",
                tuple(params),
            )
            rows = self._rows_to_dicts(cur, cur.fetchall())
            for row in rows:
                cur.execute(
                    "SELECT step_no, content FROM fault_history_steps WHERE case_pk=%s ORDER BY step_no",
                    (row["id"],),
                )
                row["steps"] = [item[1] for item in cur.fetchall()]
            return rows

    def get_records_by_storage_locators(
        self,
        doc_type: str,
        locators: Iterable[Dict[str, Any]],
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """按 OpenSearch 保存的 PostgreSQL 主键回查当前权威记录。

        OpenSearch 只负责找到候选；这里同时核对 ``id`` 与 ``doc_id``，
        避免把索引中的标题、相似名称或过期正文当成结构化真源。表名和
        字段映射均来自固定白名单，用户输入不会进入 SQL 标识符。
        """

        table_by_type = {
            "hmc": "fault_hmc",
            "fmeca": "fmea",
            "mfl": "mfl",
            "fault_history": "fault_history_cases",
        }
        if str(doc_type or "").strip() == "function_locator":
            return self._get_function_records_by_storage_locators(
                locators,
                limit=limit,
            )
        table = table_by_type.get(str(doc_type or "").strip())
        if not table:
            return []

        bounded: List[tuple[int, str]] = []
        seen = set()
        for raw in locators:
            if not isinstance(raw, dict):
                continue
            raw_id = str(raw.get("pg_id_kwd") or raw.get("pg_id") or "").strip()
            doc_id = str(raw.get("doc_id") or "").strip()
            try:
                storage_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            identity = (storage_id, doc_id)
            if storage_id <= 0 or not doc_id or identity in seen:
                continue
            seen.add(identity)
            bounded.append(identity)
            if len(bounded) >= max(0, int(limit)):
                break
        if not bounded:
            return []

        conditions = " OR ".join("(t.id=%s AND t.doc_id=%s)" for _ in bounded)
        params: List[Any] = []
        for storage_id, doc_id in bounded:
            params.extend((storage_id, doc_id))
        params.append(max(0, int(limit)))

        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT t.* FROM {table} t "
                "JOIN icp_documents d ON d.doc_id=t.doc_id "
                "WHERE d.status IN ('indexed','kg_pending','kg_done') "
                f"AND ({conditions}) ORDER BY t.id LIMIT %s",
                tuple(params),
            )
            rows = self._rows_to_dicts(cur, cur.fetchall())
            if doc_type == "fault_history":
                for row in rows:
                    cur.execute(
                        "SELECT step_no, content FROM fault_history_steps "
                        "WHERE case_pk=%s ORDER BY step_no",
                        (row["id"],),
                    )
                    row["steps"] = [item[1] for item in cur.fetchall()]
            return rows

    def _get_function_records_by_storage_locators(
        self,
        locators: Iterable[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Resolve Function Locator projections through their exact source table.

        ``pg_id`` values are local to each table, so the indexed ``content_type``
        is part of the storage identity.  Unknown and free-text projection types
        are intentionally ignored; their indexed content cannot become a
        PostgreSQL fact merely because it has a numeric-looking identifier.
        """

        table_by_content_type = {
            "function_record": "icp_functions",
            "function_component": "icp_components",
            "function_interface": "icp_interfaces",
        }
        bounded: List[tuple[int, str, str]] = []
        seen = set()
        for raw in locators:
            if not isinstance(raw, dict):
                continue
            content_type = str(raw.get("content_type") or "").strip()
            if content_type not in table_by_content_type:
                continue
            raw_id = str(raw.get("pg_id_kwd") or raw.get("pg_id") or "").strip()
            doc_id = str(raw.get("doc_id") or "").strip()
            try:
                storage_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            identity = (storage_id, doc_id, content_type)
            if storage_id <= 0 or not doc_id or identity in seen:
                continue
            seen.add(identity)
            bounded.append(identity)
            if len(bounded) >= max(0, int(limit)):
                break
        if not bounded:
            return []

        rows_by_identity: Dict[tuple[int, str, str], Dict[str, Any]] = {}
        with postgres_client.conn.cursor() as cur:
            for content_type, table in table_by_content_type.items():
                group = [item for item in bounded if item[2] == content_type]
                if not group:
                    continue
                conditions = " OR ".join("(t.id=%s AND t.doc_id=%s)" for _ in group)
                params: List[Any] = []
                for storage_id, doc_id, _ in group:
                    params.extend((storage_id, doc_id))
                cur.execute(
                    f"SELECT t.* FROM {table} t "
                    "JOIN icp_documents d ON d.doc_id=t.doc_id "
                    "WHERE d.status IN ('indexed','kg_pending','kg_done') "
                    f"AND ({conditions}) ORDER BY t.id",
                    tuple(params),
                )
                for row in self._rows_to_dicts(cur, cur.fetchall()):
                    item = dict(row)
                    item["content_type"] = content_type
                    rows_by_identity[(int(item["id"]), str(item["doc_id"]), content_type)] = item

        # Preserve semantic retrieval order while returning only identities
        # that were re-authorized by the active-document join above.
        return [rows_by_identity[key] for key in bounded if key in rows_by_identity]

    def get_functions_by_doc(self, doc_id: str, page: int = 1, size: int = 50, keyword: str = "") -> Dict[str, Any]:
        self.init_tables()
        where = ["doc_id=%s"]
        params: List[Any] = [doc_id]
        if keyword:
            where.append(
                "(function_id ILIKE %s OR function_name ILIKE %s OR system_name ILIKE %s "
                "OR subsystem_name ILIKE %s OR module_name ILIKE %s OR description ILIKE %s "
                "OR failure_symptoms ILIKE %s OR aliases::text ILIKE %s)"
            )
            params.extend([f"%{keyword}%"] * 8)
        where_sql = " AND ".join(where)
        offset = max(page - 1, 0) * size
        with postgres_client.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM icp_functions WHERE {where_sql}", tuple(params))
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT * FROM icp_functions WHERE {where_sql} ORDER BY id DESC LIMIT %s OFFSET %s",
                tuple(params + [size, offset]),
            )
            rows = self._rows_to_dicts(cur, cur.fetchall())
        return {"total": total, "page": page, "size": size, "rows": rows}

    def get_history_by_doc(self, doc_id: str, page: int = 1, size: int = 50, keyword: str = "") -> Dict[str, Any]:
        self.init_tables()
        where = ["doc_id=%s"]
        params: List[Any] = [doc_id]
        if keyword:
            where.append(
                "(case_id ILIKE %s OR occurred_at ILIKE %s OR device_type ILIKE %s "
                "OR fault_symptom ILIKE %s OR related_module ILIKE %s "
                "OR related_function ILIKE %s OR root_cause ILIKE %s "
                "OR final_solution ILIKE %s OR hmc_code ILIKE %s OR fmeca_code ILIKE %s "
                "OR fta_ref ILIKE %s OR rbd_ref ILIKE %s OR mfl_code ILIKE %s)"
            )
            params.extend([f"%{keyword}%"] * 13)
        where_sql = " AND ".join(where)
        offset = max(page - 1, 0) * size
        with postgres_client.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fault_history_cases WHERE {where_sql}", tuple(params))
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT * FROM fault_history_cases WHERE {where_sql} ORDER BY id DESC LIMIT %s OFFSET %s",
                tuple(params + [size, offset]),
            )
            rows = self._rows_to_dicts(cur, cur.fetchall())
            for row in rows:
                cur.execute(
                    "SELECT step_no, content FROM fault_history_steps WHERE case_pk=%s ORDER BY step_no",
                    (row["id"],),
                )
                row["steps"] = [value[1] for value in cur.fetchall()]
        return {"total": total, "page": page, "size": size, "rows": rows}

    def delete_by_doc(self, doc_id: str) -> None:
        self.init_tables()
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM icp_functions WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM icp_components WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM icp_interfaces WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM fault_history_cases WHERE doc_id=%s", (doc_id,))
        postgres_client.conn.commit()

    def clear_function_snapshot(self, doc_id: str) -> None:
        """Replace only deterministic Function Locator truth rows for one document."""
        self.init_tables()
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM icp_interfaces WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM icp_functions WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM icp_components WHERE doc_id=%s", (doc_id,))
        postgres_client.conn.commit()

    def clear_history_snapshot(self, doc_id: str) -> None:
        self.init_tables()
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM fault_history_cases WHERE doc_id=%s", (doc_id,))
        postgres_client.conn.commit()



icp_repository = IcpRepository()
