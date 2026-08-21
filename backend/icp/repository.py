"""ICP 领域结构化数据仓储。

该模块只复用现有 PostgreSQL 连接，不改变通用知识库与 Dify 接口。
用于保存 Function Locator 与故障历史的结构化结果，并向 ICP 专用工具提供只读查询。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

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
                        module_name      VARCHAR(200),
                        function_name    VARCHAR(300) NOT NULL,
                        description      TEXT,
                        aliases          JSONB DEFAULT '[]'::jsonb,
                        failure_symptoms TEXT,
                        source_ref       VARCHAR(500),
                        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, function_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_doc ON icp_functions(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_name ON icp_functions(function_name)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_functions_module ON icp_functions(module_name)")

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
                        diagnosis_method  TEXT,
                        root_cause        TEXT,
                        final_solution    TEXT,
                        preventive_action TEXT,
                        hmc_code          VARCHAR(100),
                        fmeca_code        VARCHAR(100),
                        fta_ref           VARCHAR(200),
                        mfl_code          VARCHAR(100),
                        source_ref        VARCHAR(500),
                        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (doc_id, case_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_doc ON fault_history_cases(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_case ON fault_history_cases(case_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_hmc ON fault_history_cases(hmc_code)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fault_history_steps (
                        id         BIGSERIAL PRIMARY KEY,
                        case_pk    BIGINT NOT NULL REFERENCES fault_history_cases(id) ON DELETE CASCADE,
                        step_no    INTEGER NOT NULL,
                        content    TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (case_pk, step_no)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_fault_history_steps_case ON fault_history_steps(case_pk)")
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
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icp_functions
                    (doc_id, function_id, system_name, subsystem_name, module_name,
                     function_name, description, aliases, failure_symptoms, source_ref)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
                ON CONFLICT (doc_id, function_id) DO UPDATE SET
                    system_name=EXCLUDED.system_name,
                    subsystem_name=EXCLUDED.subsystem_name,
                    module_name=EXCLUDED.module_name,
                    function_name=EXCLUDED.function_name,
                    description=EXCLUDED.description,
                    aliases=EXCLUDED.aliases,
                    failure_symptoms=EXCLUDED.failure_symptoms,
                    source_ref=EXCLUDED.source_ref,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    doc_id,
                    function_id,
                    item.get("system_name"),
                    item.get("subsystem_name"),
                    item.get("module_name"),
                    function_name,
                    item.get("description"),
                    json.dumps(aliases, ensure_ascii=False),
                    item.get("failure_symptoms"),
                    item.get("source_ref"),
                ),
            )
            row_id = int(cur.fetchone()[0])
        postgres_client.conn.commit()
        return row_id

    def upsert_history_case(self, doc_id: str, item: Dict[str, Any], steps: Optional[Iterable[str]] = None) -> int:
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
                     related_module, diagnosis_method, root_cause, final_solution,
                     preventive_action, hmc_code, fmeca_code, fta_ref, mfl_code, source_ref)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (doc_id, case_id) DO UPDATE SET
                    occurred_at=EXCLUDED.occurred_at,
                    personnel=EXCLUDED.personnel,
                    device_type=EXCLUDED.device_type,
                    fault_symptom=EXCLUDED.fault_symptom,
                    related_module=EXCLUDED.related_module,
                    diagnosis_method=EXCLUDED.diagnosis_method,
                    root_cause=EXCLUDED.root_cause,
                    final_solution=EXCLUDED.final_solution,
                    preventive_action=EXCLUDED.preventive_action,
                    hmc_code=EXCLUDED.hmc_code,
                    fmeca_code=EXCLUDED.fmeca_code,
                    fta_ref=EXCLUDED.fta_ref,
                    mfl_code=EXCLUDED.mfl_code,
                    source_ref=EXCLUDED.source_ref,
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
                    item.get("diagnosis_method"),
                    item.get("root_cause"),
                    item.get("final_solution"),
                    item.get("preventive_action"),
                    item.get("hmc_code"),
                    item.get("fmeca_code"),
                    item.get("fta_ref"),
                    item.get("mfl_code"),
                    item.get("source_ref"),
                ),
            )
            case_pk = int(cur.fetchone()[0])
            if steps is not None:
                cur.execute("DELETE FROM fault_history_steps WHERE case_pk=%s", (case_pk,))
                for idx, content in enumerate(steps, start=1):
                    text = str(content or "").strip()
                    if not text:
                        continue
                    cur.execute(
                        "INSERT INTO fault_history_steps(case_pk, step_no, content) VALUES (%s,%s,%s)",
                        (case_pk, idx, text),
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
                "(function_name ILIKE %s OR description ILIKE %s OR system_name ILIKE %s "
                "OR subsystem_name ILIKE %s OR module_name ILIKE %s OR failure_symptoms ILIKE %s "
                "OR aliases::text ILIKE %s)"
            )
            params.extend([pattern] * 7)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM icp_functions WHERE {' OR '.join(conditions)} ORDER BY updated_at DESC LIMIT %s",
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
                "(case_id ILIKE %s OR device_type ILIKE %s OR fault_symptom ILIKE %s "
                "OR related_module ILIKE %s OR diagnosis_method ILIKE %s OR root_cause ILIKE %s "
                "OR final_solution ILIKE %s OR hmc_code ILIKE %s OR fmeca_code ILIKE %s "
                "OR fta_ref ILIKE %s OR mfl_code ILIKE %s)"
            )
            params.extend([pattern] * 11)
        params.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM fault_history_cases WHERE {' OR '.join(conditions)} "
                "ORDER BY updated_at DESC LIMIT %s",
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

    def delete_by_doc(self, doc_id: str) -> None:
        self.init_tables()
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM icp_functions WHERE doc_id=%s", (doc_id,))
            cur.execute("DELETE FROM fault_history_cases WHERE doc_id=%s", (doc_id,))
        postgres_client.conn.commit()


icp_repository = IcpRepository()
