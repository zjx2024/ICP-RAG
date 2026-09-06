import psycopg2
import hashlib
import json
import logging
from contextlib import contextmanager
from backend.config.config import config

logger = logging.getLogger(__name__)

class PostgresClient:
    def __init__(self):
        self.conn = None
        self.is_mock = False
        self._name_cache = {}  # (kb_id, doc_id) -> (kb_name, doc_name) 快照缓存，避免每次插日志都查库
        self._connect()

    def conn_is_mock(self) -> bool:
        """显式查询是否为内存 Mock 连接（替代按类名字符串判断）。

        STRICT_DEPENDENCIES=false 且真实连接失败时回退到 MockPostgres，调用方
        需要区分“真实库”与“内存替身”来决定就绪状态与降级策略。
        """
        return bool(self.is_mock) or (
            self.conn is not None and self.conn.__class__.__name__ == "MockPostgres"
        )

    @contextmanager
    def read_cursor(self):
        """执行并收口只读事务，避免 SELECT 后连接长期 idle in transaction。"""
        try:
            with self.conn.cursor() as cur:
                yield cur
            self.conn.commit()
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise

    def _get_names(self, kb_id, doc_id):
        """获取知识库/文档名称用于日志留痕。优先用缓存，缓存未命中查库并写入缓存。
        文档/知识库被删除后查不到时回落到 None，调用方应优先传入已知快照名。"""
        key = (kb_id, doc_id)
        if key in self._name_cache:
            return self._name_cache[key]
        kb_name, doc_name = None, None
        try:
            with self.read_cursor() as cur:
                if kb_id:
                    cur.execute("SELECT name FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
                    r = cur.fetchone()
                    if r:
                        kb_name = r[0]
                if doc_id:
                    cur.execute("SELECT file_name FROM documents WHERE doc_id = %s", (doc_id,))
                    r = cur.fetchone()
                    if r:
                        doc_name = r[0]
        except Exception:
            pass
        self._name_cache[key] = (kb_name, doc_name)
        return kb_name, doc_name

    @staticmethod
    def _with_storage_key_alias(data):
        """为旧 ``minio_path`` 返回值补充中性的 ``storage_key`` 别名。

        数据库物理列和现有 SQL 暂时保持不变；别名只存在于 Python 返回字典中，
        使新代码不再依赖具体存储后端名称。即使旧值为 ``None`` 或空字符串也保留
        同值别名，避免调用方需要判断返回结构是否变化。
        """
        if isinstance(data, dict) and "minio_path" in data:
            data.setdefault("storage_key", data["minio_path"])
        return data

    @staticmethod
    def _row_to_dict(cur, row):
        """用游标列名把一行转成 dict，避免依赖列顺序（新增列时不会错位）。
        mock 模式下 description 为 None，行本身就是 dict 时直接返回，否则按位置兜底。"""
        if getattr(cur, "description", None):
            data = {desc[0]: value for desc, value in zip(cur.description, row)}
            return PostgresClient._with_storage_key_alias(data)
        if isinstance(row, dict):
            return PostgresClient._with_storage_key_alias(dict(row))
        return {}
    
    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
                dbname=config.POSTGRES_DB
            )
            self._create_tables()
        except Exception as exc:
            logger.error("PostgreSQL 初始化失败: %s", exc)
            if config.STRICT_DEPENDENCIES:
                raise
            self.conn = MockPostgres()
            self.is_mock = True
    
    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    kb_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id VARCHAR(36) PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL,
                    status VARCHAR(32) DEFAULT 'uploaded',
                    minio_path VARCHAR(512),
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
                )
            """)

            # GraphRAG 社区报告的权威事实源。OpenSearch 仅保存可重建的
            # 检索投影，因此报告本文、文档归属和 stale 状态必须先落 PG。
            cur.execute("""
                CREATE TABLE IF NOT EXISTS community_reports (
                    projection_id VARCHAR(255) PRIMARY KEY,
                    report_id VARCHAR(128) NOT NULL,
                    community_id VARCHAR(128),
                    kb_id VARCHAR(36) NOT NULL,
                    doc_id VARCHAR(36),
                    doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
                    report JSONB NOT NULL DEFAULT '{}'::jsonb,
                    stale_kwd VARCHAR(32),
                    stale_reason TEXT,
                    stale_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (kb_id, report_id),
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_community_reports_kb ON community_reports (kb_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_community_reports_doc ON community_reports (doc_id)")

            try:
                cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS delete_step VARCHAR(32)")
            except Exception:
                self.conn.rollback()

            try:
                cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE")
            except Exception:
                self.conn.rollback()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id VARCHAR(128) PRIMARY KEY,
                    doc_id VARCHAR(36) NOT NULL,
                    section VARCHAR(255),
                    title VARCHAR(512),
                    content TEXT,
                    content_type VARCHAR(32) DEFAULT 'text',
                    chunk_index INTEGER DEFAULT 0,
                    selected_for_kg BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_images (
                    image_id VARCHAR(128) PRIMARY KEY,
                    doc_id VARCHAR(36) NOT NULL,
                    chunk_id VARCHAR(128),
                    title VARCHAR(512),
                    description TEXT,
                    minio_path VARCHAR(512),
                    image_type VARCHAR(32) DEFAULT 'normal',
                    image_url VARCHAR(1024),
                    mime_type VARCHAR(128),
                    source_ref VARCHAR(1024),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- doc_id may belong to documents or icp_documents.
                    FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
                )
            """)
            
            try:
                cur.execute("ALTER TABLE document_images ADD COLUMN IF NOT EXISTS image_url VARCHAR(1024)")
                cur.execute("ALTER TABLE document_images ADD COLUMN IF NOT EXISTS mime_type VARCHAR(128)")
                cur.execute("ALTER TABLE document_images ADD COLUMN IF NOT EXISTS source_ref VARCHAR(1024)")
            except Exception:
                self.conn.rollback()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_tables (
                    table_id VARCHAR(128) PRIMARY KEY,
                    doc_id VARCHAR(36) NOT NULL,
                    chunk_id VARCHAR(128),
                    title VARCHAR(512),
                    description TEXT,
                    meta_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- doc_id may belong to documents or icp_documents.
                    FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
                )
            """)
            
            # 旧通用链路的故障树结构化数据；七类 ICP 专用链路使用自动校验投影。
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_fault_trees (
                    id VARCHAR(36) PRIMARY KEY,
                    doc_id VARCHAR(36) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL,
                    chunk_id VARCHAR(128),
                    image_id VARCHAR(128),
                    image_title VARCHAR(512),
                    graph JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # These media/graph tables are shared by the legacy knowledge-base
            # documents and the ICP type-driven documents.  A PostgreSQL FK
            # cannot target either of two owner tables, so old single-table
            # constraints must be removed.  Deletion is explicit and atomic in
            # the repository methods until artifact_refs lands in phase 2.
            for table_name in ("document_images", "document_tables", "document_fault_trees"):
                cur.execute(
                    f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_doc_id_fkey"
                )
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_models (
                    model_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    model_type VARCHAR(30) NOT NULL DEFAULT 'llm',
                    provider VARCHAR(50) NOT NULL,
                    api_key TEXT,
                    api_url VARCHAR(512),
                    model_name VARCHAR(100),
                    base_url VARCHAR(512),
                    temperature FLOAT DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 262144,
                    top_p FLOAT DEFAULT 0.1,
                    timeout INTEGER DEFAULT 3000,
                    is_default BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (name)
                )
            """)
            
            cur.execute("""
                ALTER TABLE llm_models ADD COLUMN IF NOT EXISTS model_type VARCHAR(30) DEFAULT 'llm'
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_parse_logs (
                    log_id VARCHAR(64) PRIMARY KEY,
                    task_id VARCHAR(36) NOT NULL,
                    doc_id VARCHAR(36) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL,
                    block_key VARCHAR(255),
                    block_title VARCHAR(512),
                    log_level VARCHAR(20) DEFAULT 'INFO',
                    log_type VARCHAR(50) DEFAULT 'general',
                    message TEXT NOT NULL,
                    block_count INTEGER DEFAULT 0,
                    image_count INTEGER DEFAULT 0,
                    table_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    progress INTEGER DEFAULT 0,
                    status VARCHAR(32) DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    kb_name VARCHAR(255),
                    doc_name VARCHAR(255)
                )
            """)

            # 添加 kb_name/doc_name 快照列（已存在表兼容）
            try:
                cur.execute("ALTER TABLE document_parse_logs ADD COLUMN IF NOT EXISTS kb_name VARCHAR(255)")
                cur.execute("ALTER TABLE document_parse_logs ADD COLUMN IF NOT EXISTS doc_name VARCHAR(255)")
            except Exception:
                self.conn.rollback()

            # 迁移：完全移除 doc_id 外键约束，保留解析日志即使文档被删除
            # 先尝试各种可能的约束名
            fk_names = [
                'document_parse_logs_doc_id_fkey',
                'document_parse_logs_docid_fkey',
            ]
            for fk_name in fk_names:
                try:
                    cur.execute(f"ALTER TABLE document_parse_logs DROP CONSTRAINT IF EXISTS {fk_name}")
                    logger.info(f"Dropped foreign key {fk_name} from document_parse_logs")
                except Exception as e:
                    logger.warning(f"Failed to drop constraint {fk_name}: {str(e)}")
                    self.conn.rollback()

            # 迁移：移除 kb_id 外键约束，保留日志即使知识库被删除（日志留痕）
            kb_fk_names = [
                'document_parse_logs_kb_id_fkey',
                'document_parse_logs_kbid_fkey',
            ]
            for fk_name in kb_fk_names:
                try:
                    cur.execute(f"ALTER TABLE document_parse_logs DROP CONSTRAINT IF EXISTS {fk_name}")
                    logger.info(f"Dropped foreign key {fk_name} from document_parse_logs")
                except Exception as e:
                    logger.warning(f"Failed to drop constraint {fk_name}: {str(e)}")
                    self.conn.rollback()

            # 验证约束已移除
            cur.execute("""
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = 'document_parse_logs' AND constraint_type = 'FOREIGN KEY'
            """)
            remaining = cur.fetchall()
            if remaining:
                logger.warning(f"Remaining FK constraints on document_parse_logs: {remaining}")
            else:
                logger.info("All FK constraints removed from document_parse_logs")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_parse_logs_task_id ON document_parse_logs(task_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_parse_logs_doc_id ON document_parse_logs(doc_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_parse_logs_kb_id ON document_parse_logs(kb_id)
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_generation_logs (
                    log_id VARCHAR(64) PRIMARY KEY,
                    task_id VARCHAR(36) NOT NULL,
                    doc_id VARCHAR(36) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL,
                    chunk_id VARCHAR(128),
                    chunk_index INTEGER DEFAULT 0,
                    chunk_title VARCHAR(512),
                    log_level VARCHAR(20) DEFAULT 'INFO',
                    log_type VARCHAR(50) DEFAULT 'general',
                    message TEXT NOT NULL,
                    entity_count INTEGER DEFAULT 0,
                    relation_count INTEGER DEFAULT 0,
                    community_count INTEGER DEFAULT 0,
                    entities JSONB,
                    relations JSONB,
                    progress INTEGER DEFAULT 0,
                    status VARCHAR(32) DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    kb_name VARCHAR(255),
                    doc_name VARCHAR(255)
                )
            """)

            # 添加 kb_name/doc_name 快照列（已存在表兼容）
            try:
                cur.execute("ALTER TABLE kg_generation_logs ADD COLUMN IF NOT EXISTS kb_name VARCHAR(255)")
                cur.execute("ALTER TABLE kg_generation_logs ADD COLUMN IF NOT EXISTS doc_name VARCHAR(255)")
            except Exception:
                self.conn.rollback()

            # 迁移：完全移除 doc_id 外键约束，保留 KG 日志即使文档被删除
            fk_names = [
                'kg_generation_logs_doc_id_fkey',
                'kg_generation_logs_docid_fkey',
            ]
            for fk_name in fk_names:
                try:
                    cur.execute(f"ALTER TABLE kg_generation_logs DROP CONSTRAINT IF EXISTS {fk_name}")
                    logger.info(f"Dropped foreign key {fk_name} from kg_generation_logs")
                except Exception as e:
                    logger.warning(f"Failed to drop constraint {fk_name}: {str(e)}")
                    self.conn.rollback()

            # 迁移：移除 kb_id 外键约束，保留日志即使知识库被删除（日志留痕）
            kb_fk_names = [
                'kg_generation_logs_kb_id_fkey',
                'kg_generation_logs_kbid_fkey',
            ]
            for fk_name in kb_fk_names:
                try:
                    cur.execute(f"ALTER TABLE kg_generation_logs DROP CONSTRAINT IF EXISTS {fk_name}")
                    logger.info(f"Dropped foreign key {fk_name} from kg_generation_logs")
                except Exception as e:
                    logger.warning(f"Failed to drop constraint {fk_name}: {str(e)}")
                    self.conn.rollback()

            # 验证约束已移除
            cur.execute("""
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = 'kg_generation_logs' AND constraint_type = 'FOREIGN KEY'
            """)
            remaining = cur.fetchall()
            if remaining:
                logger.warning(f"Remaining FK constraints on kg_generation_logs: {remaining}")
            else:
                logger.info("All FK constraints removed from kg_generation_logs")
            
            try:
                cur.execute("ALTER TABLE kg_generation_logs ADD COLUMN IF NOT EXISTS community_count INTEGER DEFAULT 0")
            except Exception:
                pass
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_logs_task_id ON kg_generation_logs(task_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_logs_doc_id ON kg_generation_logs(doc_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_kg_logs_kb_id ON kg_generation_logs(kb_id)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS deletion_logs (
                    task_id VARCHAR(36) NOT NULL,
                    doc_id VARCHAR(36) NOT NULL,
                    kb_id VARCHAR(36) NOT NULL,
                    step VARCHAR(32) NOT NULL,
                    item_type VARCHAR(32) NOT NULL,
                    item_id VARCHAR(512),
                    item_detail JSONB,
                    status VARCHAR(16) DEFAULT 'info',
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    kb_name VARCHAR(255),
                    doc_name VARCHAR(255)
                )
            """)
            # 添加 kb_name/doc_name 快照列（已存在表兼容）
            try:
                cur.execute("ALTER TABLE deletion_logs ADD COLUMN IF NOT EXISTS kb_name VARCHAR(255)")
                cur.execute("ALTER TABLE deletion_logs ADD COLUMN IF NOT EXISTS doc_name VARCHAR(255)")
            except Exception:
                self.conn.rollback()
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_del_logs_task_id ON deletion_logs(task_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_del_logs_doc_id ON deletion_logs(doc_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_del_logs_kb_id ON deletion_logs(kb_id)
            """)

            # ===== ICP 类型驱动文档体系（新建表，不影响现有知识库功能）=====
            # 文档类型注册表（前端类型卡片数据源）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_types (
                    doc_type      VARCHAR(32) PRIMARY KEY,
                    name          VARCHAR(64) NOT NULL,
                    description   TEXT,
                    data_form     VARCHAR(16) NOT NULL,
                    input_formats VARCHAR(128) NOT NULL,
                    parser_key    VARCHAR(64) NOT NULL,
                    pg_source     VARCHAR(64),
                    es_doc_type   VARCHAR(32) NOT NULL,
                    needs_kg      BOOLEAN DEFAULT FALSE,
                    is_active     BOOLEAN DEFAULT TRUE,
                    sort_order    INTEGER DEFAULT 0,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 类型文档主表（doc_type 即顶层组织单位，替代知识库概念）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS icp_documents (
                    doc_id      VARCHAR(36) PRIMARY KEY,
                    doc_type    VARCHAR(32) NOT NULL,
                    file_name   VARCHAR(255) NOT NULL,
                    minio_path  VARCHAR(512),
                    status      VARCHAR(32) DEFAULT 'uploaded',
                    parse_count INTEGER DEFAULT 0,
                    last_result JSONB,
                    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("ALTER TABLE icp_documents ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_docs_type ON icp_documents (doc_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_icp_docs_status ON icp_documents (status)")

            # HMC 码查询真源表。业务码会在不同文档版本中重复，因此唯一性必须
            # 包含 doc_id；跨版本选择由适用范围和文档元数据显式完成。
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fault_hmc (
                    id            BIGSERIAL PRIMARY KEY,
                    doc_id        VARCHAR(36) NOT NULL,
                    variant_id    VARCHAR(64) NOT NULL,
                    hmc_code      VARCHAR(20) NOT NULL,
                    lcn           VARCHAR(10) DEFAULT '**',
                    hamc          VARCHAR(10) DEFAULT '**',
                    hsc           VARCHAR(10) DEFAULT '**',
                    fault_desc    TEXT NOT NULL,
                    repair_method TEXT NOT NULL,
                    source_ref    VARCHAR(1024),
                    raw_data      JSONB NOT NULL DEFAULT '{}'::jsonb,
                    normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (doc_id, variant_id)
                )
            """)
            cur.execute("ALTER TABLE fault_hmc ADD COLUMN IF NOT EXISTS variant_id VARCHAR(64)")
            cur.execute("ALTER TABLE fault_hmc ADD COLUMN IF NOT EXISTS source_ref VARCHAR(1024)")
            cur.execute("ALTER TABLE fault_hmc ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("ALTER TABLE fault_hmc ADD COLUMN IF NOT EXISTS normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE fault_hmc ALTER COLUMN hmc_code TYPE VARCHAR(100)")
            cur.execute("ALTER TABLE fault_hmc ALTER COLUMN lcn TYPE VARCHAR(50)")
            cur.execute("ALTER TABLE fault_hmc ALTER COLUMN hamc TYPE VARCHAR(50)")
            cur.execute("ALTER TABLE fault_hmc ALTER COLUMN hsc TYPE VARCHAR(50)")
            # Existing installations used one row per code and therefore lost
            # independently worded source variants.  Seed migrated rows once;
            # new parses supply a source-stable SHA-256 variant id.
            cur.execute(
                """UPDATE fault_hmc SET variant_id = md5(concat_ws(E'\\x1f',
                       doc_id, hmc_code, lcn, hamc, hsc,
                       COALESCE(source_ref, ''), id::text))
                   WHERE variant_id IS NULL OR variant_id = ''"""
            )
            cur.execute("ALTER TABLE fault_hmc ALTER COLUMN variant_id SET NOT NULL")
            cur.execute("ALTER TABLE fault_hmc DROP CONSTRAINT IF EXISTS fault_hmc_hmc_code_lcn_hamc_hsc_key")
            cur.execute("ALTER TABLE fault_hmc DROP CONSTRAINT IF EXISTS fault_hmc_doc_id_hmc_code_lcn_hamc_hsc_key")
            cur.execute("DROP INDEX IF EXISTS uq_fault_hmc_doc_code")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_fault_hmc_doc_variant ON fault_hmc (doc_id, variant_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hmc_doc_code ON fault_hmc (doc_id, hmc_code)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hmc_3code ON fault_hmc (lcn, hamc, hsc)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hmc_doc ON fault_hmc (doc_id)")

            # FMECA 真源表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fmea (
                    id                BIGSERIAL PRIMARY KEY,
                    doc_id            VARCHAR(36) NOT NULL,
                    code              VARCHAR(50) NOT NULL,
                    product_flag      VARCHAR(100),
                    func_management   VARCHAR(100),
                    fault_mode        TEXT NOT NULL,
                    fault_cause       TEXT,
                    mission_phase     VARCHAR(100),
                    local_impact      TEXT,
                    final_impact      TEXT,
                    severity          VARCHAR(20),
                    severity_raw      VARCHAR(100),
                    severity_normalized VARCHAR(100),
                    probability       NUMERIC(30,18),
                    probability_raw   VARCHAR(100),
                    probability_level VARCHAR(50),
                    detect_method     TEXT,
                    design_improve    TEXT,
                    compensate_action TEXT,
                    source_ref        VARCHAR(1024),
                    raw_data          JSONB NOT NULL DEFAULT '{}'::jsonb,
                    normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (doc_id, code)
                )
            """)
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS severity_raw VARCHAR(100)")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS severity_normalized VARCHAR(100)")
            cur.execute("ALTER TABLE fmea ALTER COLUMN probability TYPE NUMERIC(30,18)")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS probability_raw VARCHAR(100)")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS probability_level VARCHAR(50)")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS source_ref VARCHAR(1024)")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("ALTER TABLE fmea ADD COLUMN IF NOT EXISTS normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE fmea ALTER COLUMN code TYPE VARCHAR(100)")
            cur.execute("ALTER TABLE fmea ALTER COLUMN mission_phase TYPE VARCHAR(300)")
            cur.execute("ALTER TABLE fmea DROP CONSTRAINT IF EXISTS fmea_code_key")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_fmea_doc_code ON fmea (doc_id, code)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fmea_severity ON fmea (severity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fmea_doc ON fmea (doc_id)")

            # MFL 真源表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mfl (
                    id               BIGSERIAL PRIMARY KEY,
                    doc_id           VARCHAR(36) NOT NULL,
                    mfl_code         VARCHAR(50) NOT NULL,
                    module_fault     TEXT,
                    sur_fault        TEXT,
                    subcircuit_fault TEXT,
                    api_reportable   BOOLEAN DEFAULT FALSE,
                    api_reportable_raw VARCHAR(100),
                    api_name         VARCHAR(200),
                    api_method       VARCHAR(200),
                    api_usage        TEXT,
                    platform_test    BOOLEAN DEFAULT FALSE,
                    platform_test_raw VARCHAR(100),
                    platform_test_status VARCHAR(50),
                    maintenance_action TEXT,
                    source_ref       VARCHAR(1024),
                    raw_data         JSONB NOT NULL DEFAULT '{}'::jsonb,
                    normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (doc_id, mfl_code)
                )
            """)
            # 历史 schema 的 api_method 只有 20 字符，真实调用说明可能更长。
            cur.execute("ALTER TABLE mfl ALTER COLUMN api_method TYPE VARCHAR(200)")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS api_reportable_raw VARCHAR(100)")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS platform_test_raw VARCHAR(100)")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS platform_test_status VARCHAR(50)")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS maintenance_action TEXT")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS source_ref VARCHAR(1024)")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS raw_data JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("ALTER TABLE mfl ADD COLUMN IF NOT EXISTS normalization_warnings JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE mfl DROP CONSTRAINT IF EXISTS mfl_mfl_code_key")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_mfl_doc_code ON mfl (doc_id, mfl_code)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mfl_api_name ON mfl (api_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mfl_reportable ON mfl (api_reportable)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mfl_doc ON mfl (doc_id)")

            # 类型注册表初始化（幂等，存在即跳过）
            cur.execute("SELECT COUNT(*) FROM document_types")
            if cur.fetchone()[0] == 0:
                type_rows = [
                    ('hmc',   'HMC 码查询',     '故障码/维修码查询：按 HMC/LCN/HAMC/HSC 组合定位故障现象与维修方法', 'table', '.xlsx,.xls,.docx', 'excel_row', 'fault_hmc', 'hmc', False, 1),
                    ('fmeca', 'FMECA 故障影响', '故障模式与影响分析：严酷度/概率/检测方法与改进措施', 'table', '.xlsx,.xls,.docx', 'excel_row', 'fmea', 'fmeca', False, 2),
                    ('mfl',   'MFL 清单',       '模块故障清单：故障模式与 API 上报映射', 'table', '.xlsx,.xls,.docx', 'excel_row', 'mfl', 'mfl', False, 3),
                    ('fta',   'FTA 故障树',     '故障树顶事件查询：图片识别为节点/边三元组并生成知识图谱', 'image', '.docx,.png,.jpg', 'fault_tree', 'document_fault_trees', 'fta', True, 4),
                    ('rbd',   'RBD 可靠性框图', '可靠性框图：模块节点与串并联逻辑', 'image', '.docx,.png,.jpg', 'fault_tree', 'document_fault_trees', 'rbd', True, 5),
                    ('fault_history', '故障历史', '历史故障记录：分块向量化检索', 'text', '.docx,.txt', 'text_chunk', 'document_chunks', 'fault_history', False, 6),
                    ('function_locator', '功能定位', '设备功能定位：功能到模块/子系统的映射说明', 'text', '.docx,.txt', 'text_chunk', 'document_chunks', 'function_locator', False, 7),
                ]
                for r in type_rows:
                    cur.execute("""
                        INSERT INTO document_types (doc_type, name, description, data_form, input_formats, parser_key, pg_source, es_doc_type, needs_kg, sort_order)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, r)

            self.conn.commit()
    
    def insert_document(self, doc_id, file_name, kb_id, minio_path):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (doc_id, file_name, kb_id, minio_path)
                VALUES (%s, %s, %s, %s)
            """, (doc_id, file_name, kb_id, minio_path))
            self.conn.commit()
    
    def create_knowledge_base(self, kb_id, name, description=None):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO knowledge_bases (kb_id, name, description)
                VALUES (%s, %s, %s)
            """, (kb_id, name, description))
            self.conn.commit()
    
    def get_knowledge_base(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
            row = cur.fetchone()
            if row:
                return {
                    "kb_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "updated_at": row[4]
                }
            return None
    
    def list_knowledge_bases(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [{
                "kb_id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            } for row in rows]

    @staticmethod
    def _community_report_dict(row):
        if not row:
            return None
        fields = (
            "projection_id", "report_id", "community_id", "kb_id", "doc_id",
            "doc_ids", "entities", "report", "stale_kwd", "stale_reason",
            "stale_at", "created_at", "updated_at",
        )
        result = dict(zip(fields, row))
        for field in ("doc_ids", "entities", "report"):
            if isinstance(result.get(field), str):
                try:
                    result[field] = json.loads(result[field])
                except (TypeError, ValueError):
                    result[field] = [] if field != "report" else {}
        for field in ("stale_at", "created_at", "updated_at"):
            value = result.get(field)
            if value is not None and hasattr(value, "isoformat"):
                result[field] = value.isoformat()
        # 兼容旧检索客户端的命名，路由层响应不需要改版。
        result["es_doc_id"] = result["projection_id"]
        return result

    def upsert_community_report(self, projection_id, data):
        """幂等写入社区报告真源，必须在写检索投影之前调用。"""
        doc_ids = data.get("doc_ids") or ([data.get("doc_id")] if data.get("doc_id") else [])
        doc_ids = [str(value) for value in doc_ids if value]
        doc_id = data.get("doc_id") or (doc_ids[0] if doc_ids else "")
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO community_reports (
                    projection_id, report_id, community_id, kb_id, doc_id,
                    doc_ids, entities, report, stale_kwd, stale_reason, stale_at
                ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)
                ON CONFLICT (projection_id) DO UPDATE SET
                    report_id = EXCLUDED.report_id,
                    community_id = EXCLUDED.community_id,
                    kb_id = EXCLUDED.kb_id,
                    doc_id = EXCLUDED.doc_id,
                    doc_ids = EXCLUDED.doc_ids,
                    entities = EXCLUDED.entities,
                    report = EXCLUDED.report,
                    stale_kwd = EXCLUDED.stale_kwd,
                    stale_reason = EXCLUDED.stale_reason,
                    stale_at = EXCLUDED.stale_at,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                projection_id,
                data.get("report_id", ""),
                data.get("community_id", ""),
                data.get("kb_id", ""),
                doc_id,
                json.dumps(doc_ids, ensure_ascii=False),
                json.dumps(data.get("entities") or [], ensure_ascii=False),
                json.dumps(data.get("report") or {}, ensure_ascii=False),
                data.get("stale_kwd"),
                data.get("stale_reason"),
                data.get("stale_at"),
            ))
        self.conn.commit()
        return projection_id

    def get_community_reports(self, kb_id, doc_id=None):
        sql = """
            SELECT projection_id, report_id, community_id, kb_id, doc_id,
                   doc_ids, entities, report, stale_kwd, stale_reason, stale_at,
                   created_at, updated_at
            FROM community_reports WHERE kb_id = %s
        """
        params = [kb_id]
        if doc_id:
            sql += " AND doc_ids @> %s::jsonb"
            params.append(json.dumps([doc_id]))
        sql += " ORDER BY created_at DESC"
        with self.conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return [self._community_report_dict(row) for row in cur.fetchall()]

    def delete_community_report(self, projection_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM community_reports WHERE projection_id = %s", (projection_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def delete_community_reports_by_kb(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM community_reports WHERE kb_id = %s", (kb_id,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted

    def mark_community_report_stale(self, projection_id, doc_ids, stale_kwd, reason, stale_at):
        doc_ids = [str(value) for value in (doc_ids or []) if value]
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE community_reports
                SET doc_ids = %s::jsonb,
                    doc_id = %s,
                    stale_kwd = %s,
                    stale_reason = %s,
                    stale_at = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE projection_id = %s
            """, (
                json.dumps(doc_ids, ensure_ascii=False),
                doc_ids[0] if doc_ids else "",
                stale_kwd,
                reason,
                stale_at,
                projection_id,
            ))
            updated = cur.rowcount > 0
        self.conn.commit()
        return updated

    def kb_exists(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
            return cur.fetchone() is not None

    def doc_exists(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.fetchone() is not None
    
    def delete_knowledge_base(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def delete_knowledge_base_with_documents(self, kb_id):
        """Atomically remove a legacy KB and projections whose FK is polymorphic."""
        try:
            with self.conn.cursor() as cur:
                owner_query = "SELECT doc_id FROM documents WHERE kb_id = %s"
                cur.execute(f"DELETE FROM document_fault_trees WHERE doc_id IN ({owner_query})", (kb_id,))
                cur.execute(f"DELETE FROM document_tables WHERE doc_id IN ({owner_query})", (kb_id,))
                cur.execute(f"DELETE FROM document_images WHERE doc_id IN ({owner_query})", (kb_id,))
                cur.execute(f"DELETE FROM document_chunks WHERE doc_id IN ({owner_query})", (kb_id,))
                cur.execute("DELETE FROM knowledge_bases WHERE kb_id = %s", (kb_id,))
                deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted
        except Exception:
            self.conn.rollback()
            raise

    def update_knowledge_base(self, kb_id, name=None, description=None):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE knowledge_bases
                SET name = COALESCE(%s, name),
                    description = COALESCE(%s, description),
                    updated_at = CURRENT_TIMESTAMP
                WHERE kb_id = %s
            """, (name, description, kb_id))
            self.conn.commit()
        return cur.rowcount > 0
    
    def update_document_status(self, doc_id, status):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = %s
                """, (status, doc_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def set_delete_step(self, doc_id, step):
        """记录文档删除流程当前完成的步骤，便于失败后从断点续跑。

        step 取值: None(未开始) / edges_done / vertices_done / media_done /
        chunks_done / logs_done / es_done / file_done / done
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents SET delete_step = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = %s
                """, (step, doc_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def get_delete_step(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT delete_step FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def list_documents_by_status(self, status):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, file_name, kb_id, status, minio_path, total_chunks,
                       delete_step, created_at, updated_at
                FROM documents WHERE status = %s ORDER BY updated_at
            """, (status,))
            rows = cur.fetchall()
            return [self._with_storage_key_alias({
                "doc_id": row[0],
                "file_name": row[1],
                "kb_id": row[2],
                "status": row[3],
                "minio_path": row[4],
                "total_chunks": row[5],
                "delete_step": row[6],
                "created_at": row[7],
                "updated_at": row[8]
            }) for row in rows]
    
    def update_document_chunks(self, doc_id, total_chunks):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET total_chunks = %s, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s
            """, (total_chunks, doc_id))
            self.conn.commit()
    
    def get_document(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            if row:
                return self._with_storage_key_alias({
                    "doc_id": row[0],
                    "file_name": row[1],
                    "kb_id": row[2],
                    "status": row[3],
                    "minio_path": row[4],
                    "total_chunks": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "delete_step": row[8] if len(row) > 8 else None,
                    "is_deleted": row[9] if len(row) > 9 else False
                })
            return None
    
    def get_documents_by_kb(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE kb_id = %s ORDER BY created_at DESC", (kb_id,))
            rows = cur.fetchall()
            return [self._with_storage_key_alias({
                "doc_id": row[0],
                "file_name": row[1],
                "kb_id": row[2],
                "status": row[3],
                "minio_path": row[4],
                "total_chunks": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "delete_step": row[8] if len(row) > 8 else None,
                "is_deleted": row[9] if len(row) > 9 else False
            }) for row in rows]
    
    def delete_document(self, doc_id):
        """硬删除文档记录"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def insert_deletion_log(self, task_id, doc_id, kb_id, step, item_type, item_id=None, item_detail=None, status='info', message=None,
                            kb_name=None, doc_name=None):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO deletion_logs (task_id, doc_id, kb_id, step, item_type, item_id, item_detail, status, message, kb_name, doc_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (task_id, doc_id, kb_id, step, item_type, item_id, json.dumps(item_detail) if item_detail else None, status, message, kb_name, doc_name))
            self.conn.commit()
            logger.debug(f"Inserted deletion log: task_id={task_id}, step={step}, item_type={item_type}, status={status}")
        except Exception as e:
            logger.error(f"Failed to insert deletion log: task_id={task_id}, step={step}, error={str(e)}")
            self.conn.rollback()

    def _deletion_log_dict(self, cur, row):
        d = self._row_to_dict(cur, row)
        # item_detail 是 JSONB，解析成对象；created_at 转字符串便于前端
        if d.get("item_detail") and isinstance(d["item_detail"], str):
            try:
                d["item_detail"] = json.loads(d["item_detail"])
            except Exception:
                pass
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else d["created_at"]
        return d

    def get_deletion_logs_by_task(self, task_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM deletion_logs WHERE task_id = %s ORDER BY created_at
                """, (task_id,))
                return [self._deletion_log_dict(cur, row) for row in cur.fetchall()]
        except Exception:
            return []

    def get_deletion_logs_by_doc(self, doc_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM deletion_logs WHERE doc_id = %s ORDER BY created_at DESC
                """, (doc_id,))
                return [self._deletion_log_dict(cur, row) for row in cur.fetchall()]
        except Exception:
            return []

    def get_deletion_logs_by_kb(self, kb_id, limit=500):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM deletion_logs WHERE kb_id = %s ORDER BY created_at DESC LIMIT %s
                """, (kb_id, limit))
                return [self._deletion_log_dict(cur, row) for row in cur.fetchall()]
        except Exception:
            return []

    def get_latest_deletion_tasks(self, kb_id=None, limit=20):
        try:
            with self.conn.cursor() as cur:
                base_select = """
                    WITH task_logs AS (
                        SELECT
                            task_id,
                            doc_id,
                            kb_id,
                            status,
                            message,
                            created_at,
                            kb_name,
                            doc_name,
                            ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY created_at DESC) as rn,
                            MIN(created_at) OVER (PARTITION BY task_id) as first_time
                        FROM deletion_logs
                """
                if kb_id:
                    sql = base_select + " WHERE kb_id = %s\n                    )\n" + """
                        SELECT
                            task_id,
                            doc_id,
                            kb_id,
                            status as last_status,
                            message as last_message,
                            created_at as last_time,
                            first_time,
                            kb_name,
                            doc_name
                        FROM task_logs
                        WHERE rn = 1
                        ORDER BY last_time DESC
                        LIMIT %s
                    """
                    params = (kb_id, limit)
                else:
                    sql = base_select + "\n                    )\n" + """
                        SELECT
                            task_id,
                            doc_id,
                            kb_id,
                            status as last_status,
                            message as last_message,
                            created_at as last_time,
                            first_time,
                            kb_name,
                            doc_name
                        FROM task_logs
                        WHERE rn = 1
                        ORDER BY last_time DESC
                        LIMIT %s
                    """
                    params = (limit,)
                cur.execute(sql, params)
                rows = cur.fetchall()
                logger.info(f"Found {len(rows)} deletion tasks for kb_id={kb_id}")
                return [{
                    "task_id": row[0],
                    "doc_id": row[1],
                    "kb_id": row[2],
                    "last_status": row[3],
                    "last_message": row[4],
                    "last_time": row[5].isoformat() if row[5] else None,
                    "first_time": row[6].isoformat() if row[6] else None,
                    "kb_name": row[7],
                    "doc_name": row[8]
                } for row in rows]
        except Exception as e:
            logger.error(f"Failed to get deletion tasks: {str(e)}")
            return []

    def delete_deletion_logs_by_task(self, task_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM deletion_logs WHERE task_id = %s", (task_id,))
            self.conn.commit()
            return cur.rowcount
        except Exception:
            self.conn.rollback()
            return 0
    
    def insert_chunk(self, chunk_id, doc_id, section, title, content, content_type='text', chunk_index=0):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_chunks (chunk_id, doc_id, section, title, content, content_type, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (chunk_id, doc_id, section, title, content, content_type, chunk_index))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def insert_image(
        self,
        image_id,
        doc_id,
        chunk_id,
        title,
        description,
        minio_path,
        image_type='normal',
        image_url=None,
        mime_type=None,
        source_ref=None,
    ):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_images
                        (image_id, doc_id, chunk_id, title, description, minio_path,
                         image_type, image_url, mime_type, source_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO UPDATE SET
                        doc_id=EXCLUDED.doc_id,
                        chunk_id=EXCLUDED.chunk_id,
                        title=EXCLUDED.title,
                        description=EXCLUDED.description,
                        minio_path=EXCLUDED.minio_path,
                        image_type=EXCLUDED.image_type,
                        image_url=EXCLUDED.image_url,
                        mime_type=EXCLUDED.mime_type,
                        source_ref=EXCLUDED.source_ref
                """, (
                    image_id, doc_id, chunk_id, title, description, minio_path,
                    image_type, image_url, mime_type, source_ref,
                ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def insert_document_fault_tree(self, fault_tree_id, doc_id, kb_id, chunk_id, image_id, image_title, graph):
        """保存解析阶段 vision 提取的故障树结构化数据（不写 HugeGraph，待手动生成图谱时入库）"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_fault_trees (id, doc_id, kb_id, chunk_id, image_id, image_title, graph)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (fault_tree_id, doc_id, kb_id, chunk_id, image_id, image_title, graph))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_document_fault_trees(self, doc_id):
        """读取文档解析阶段保存的故障树结构化数据"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, doc_id, kb_id, chunk_id, image_id, image_title, graph
                    FROM document_fault_trees
                    WHERE doc_id = %s
                    ORDER BY created_at
                """, (doc_id,))
                cols = ["fault_tree_id", "doc_id", "kb_id", "chunk_id", "image_id", "image_title", "graph"]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            raise e
    
    
    def insert_table(self, table_id, doc_id, chunk_id, title, description, meta_data):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_tables (table_id, doc_id, chunk_id, title, description, meta_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (table_id, doc_id, chunk_id, title, description, meta_data))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def insert_model(self, model_id, name, model_type='llm', provider=None, api_key=None, api_url=None, 
                    model_name=None, base_url=None, temperature=0.7, max_tokens=262144,
                    top_p=0.1, timeout=3000, is_default=False, is_active=True, description=None):
        try:
            with self.conn.cursor() as cur:
                if is_default:
                    cur.execute("UPDATE llm_models SET is_default = FALSE WHERE is_default = TRUE")
                
                cur.execute("""
                    INSERT INTO llm_models (model_id, name, model_type, provider, api_key, api_url, 
                                           model_name, base_url, temperature, max_tokens, 
                                           top_p, timeout, is_default, is_active, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (model_id, name, model_type, provider, api_key, api_url, model_name, base_url,
                      temperature, max_tokens, top_p, timeout, is_default, is_active, description))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def update_model(self, model_id, **kwargs):
        try:
            with self.conn.cursor() as cur:
                if 'is_default' in kwargs and kwargs['is_default']:
                    cur.execute("UPDATE llm_models SET is_default = FALSE WHERE is_default = TRUE")
                
                set_clause = ", ".join([f"{k} = %s" for k in kwargs.keys()])
                params = list(kwargs.values()) + [model_id]
                
                cur.execute(f"""
                    UPDATE llm_models SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE model_id = %s
                """, params)
            self.conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def set_default_model_by_type(self, model_id, model_type):
        try:
            with self.conn.cursor() as cur:
                cur.execute("UPDATE llm_models SET is_default = FALSE WHERE model_type = %s AND is_default = TRUE", (model_type,))
                cur.execute("UPDATE llm_models SET is_default = TRUE, updated_at = CURRENT_TIMESTAMP WHERE model_id = %s", (model_id,))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def delete_model(self, model_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM llm_models WHERE model_id = %s", (model_id,))
            self.conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def get_model(self, model_id):
        with self.read_cursor() as cur:
            cur.execute("SELECT * FROM llm_models WHERE model_id = %s", (model_id,))
            row = cur.fetchone()
            if row:
                return {
                    "model_id": row[0],
                    "name": row[1],
                    "model_type": row[2],
                    "provider": row[3],
                    "api_key": row[4],
                    "api_url": row[5],
                    "model_name": row[6],
                    "base_url": row[7],
                    "temperature": row[8],
                    "max_tokens": row[9],
                    "top_p": row[10],
                    "timeout": row[11],
                    "is_default": row[12],
                    "is_active": row[13],
                    "description": row[14],
                    "created_at": row[15],
                    "updated_at": row[16]
                }
            return None
    
    def list_models(self):
        with self.read_cursor() as cur:
            cur.execute("SELECT * FROM llm_models WHERE is_active = TRUE ORDER BY is_default DESC, created_at DESC")
            rows = cur.fetchall()
            return [{
                "model_id": row[0],
                "name": row[1],
                "model_type": row[2],
                "provider": row[3],
                "api_key": row[4],
                "api_url": row[5],
                "model_name": row[6],
                "base_url": row[7],
                "temperature": row[8],
                "max_tokens": row[9],
                "top_p": row[10],
                "timeout": row[11],
                "is_default": row[12],
                "is_active": row[13],
                "description": row[14],
                "created_at": row[15],
                "updated_at": row[16]
            } for row in rows]
    
    def get_default_model(self):
        with self.read_cursor() as cur:
            cur.execute("SELECT * FROM llm_models WHERE is_default = TRUE AND is_active = TRUE LIMIT 1")
            row = cur.fetchone()
            if row:
                return {
                    "model_id": row[0],
                    "name": row[1],
                    "model_type": row[2],
                    "provider": row[3],
                    "api_key": row[4],
                    "api_url": row[5],
                    "model_name": row[6],
                    "base_url": row[7],
                    "temperature": row[8],
                    "max_tokens": row[9],
                    "top_p": row[10],
                    "timeout": row[11],
                    "is_default": row[12],
                    "is_active": row[13],
                    "description": row[14],
                    "created_at": row[15],
                    "updated_at": row[16]
                }
            return None
    
    def get_default_model_by_type(self, model_type):
        with self.read_cursor() as cur:
            cur.execute("""
                SELECT * FROM llm_models 
                WHERE model_type = %s AND is_default = TRUE AND is_active = TRUE 
                LIMIT 1
            """, (model_type,))
            row = cur.fetchone()
            if row:
                return {
                    "model_id": row[0],
                    "name": row[1],
                    "model_type": row[2],
                    "provider": row[3],
                    "api_key": row[4],
                    "api_url": row[5],
                    "model_name": row[6],
                    "base_url": row[7],
                    "temperature": row[8],
                    "max_tokens": row[9],
                    "top_p": row[10],
                    "timeout": row[11],
                    "is_default": row[12],
                    "is_active": row[13],
                    "description": row[14],
                    "created_at": row[15],
                    "updated_at": row[16]
                }
            return None
    
    def list_models_by_type(self, model_type):
        with self.read_cursor() as cur:
            cur.execute("""
                SELECT * FROM llm_models 
                WHERE model_type = %s AND is_active = TRUE 
                ORDER BY is_default DESC, created_at DESC
            """, (model_type,))
            rows = cur.fetchall()
            return [{
                "model_id": row[0],
                "name": row[1],
                "model_type": row[2],
                "provider": row[3],
                "api_key": row[4],
                "api_url": row[5],
                "model_name": row[6],
                "base_url": row[7],
                "temperature": row[8],
                "max_tokens": row[9],
                "top_p": row[10],
                "timeout": row[11],
                "is_default": row[12],
                "is_active": row[13],
                "description": row[14],
                "created_at": row[15],
                "updated_at": row[16]
            } for row in rows]
    
    def get_chunks_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM document_chunks WHERE doc_id = %s ORDER BY chunk_index", (doc_id,))
            rows = cur.fetchall()
            chunks = []
            for row in rows:
                chunk = {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "section": row[2],
                    "title": row[3],
                    "content": row[4],
                    "content_type": row[5],
                    "chunk_index": row[6],
                    "selected_for_kg": row[7],
                    "created_at": row[8],
                    "images": [],
                    "tables": []
                }
                chunks.append(chunk)
            
            for chunk in chunks:
                cur.execute(
                    """SELECT image_id, doc_id, chunk_id, title, description,
                              minio_path, image_type, image_url, mime_type,
                              source_ref, created_at
                       FROM document_images WHERE chunk_id = %s""",
                    (chunk["chunk_id"],),
                )
                image_rows = cur.fetchall()
                chunk["images"] = [self._with_storage_key_alias({
                    "image_id": r[0],
                    "doc_id": r[1],
                    "chunk_id": r[2],
                    "title": r[3],
                    "description": r[4],
                    "minio_path": r[5],
                    "image_type": r[6],
                    "image_url": r[7],
                    "mime_type": r[8],
                    "source_ref": r[9],
                    "created_at": r[10],
                }) for r in image_rows]
                
                cur.execute("SELECT * FROM document_tables WHERE chunk_id = %s", (chunk["chunk_id"],))
                table_rows = cur.fetchall()
                chunk["tables"] = [{
                    "table_id": r[0],
                    "doc_id": r[1],
                    "chunk_id": r[2],
                    "title": r[3],
                    "description": r[4],
                    "meta_data": r[5],
                    "created_at": r[6]
                } for r in table_rows]
            
            return chunks
    
    def update_chunk_selection(self, chunk_id, selected):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE document_chunks SET selected_for_kg = %s, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = %s
            """, (selected, chunk_id))
            self.conn.commit()
    
    def get_selected_chunks(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM document_chunks WHERE doc_id = %s AND selected_for_kg = TRUE
                ORDER BY chunk_index
            """, (doc_id,))
            rows = cur.fetchall()
            return [{
                "chunk_id": row[0],
                "doc_id": row[1],
                "section": row[2],
                "title": row[3],
                "content": row[4],
                "content_type": row[5],
                "chunk_index": row[6],
                "selected_for_kg": row[7],
                "created_at": row[8]
            } for row in rows]
    
    def delete_chunks_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
            self.conn.commit()
        return cur.rowcount > 0
    
    def get_images_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT image_id, doc_id, chunk_id, title, description,
                          minio_path, image_type, image_url, mime_type,
                          source_ref, created_at
                   FROM document_images WHERE doc_id = %s""",
                (doc_id,),
            )
            rows = cur.fetchall()
            return [self._with_storage_key_alias({
                "image_id": row[0],
                "doc_id": row[1],
                "chunk_id": row[2],
                "title": row[3],
                "description": row[4],
                "minio_path": row[5],
                "image_type": row[6],
                "image_url": row[7],
                "mime_type": row[8],
                "source_ref": row[9],
                "created_at": row[10],
            }) for row in rows]
    
    def get_all_image_docs(self, kb_id=None):
        with self.conn.cursor() as cur:
            if kb_id:
                cur.execute("""
                    SELECT d.doc_id, d.file_name, COUNT(i.image_id) as image_count
                    FROM documents d
                    JOIN document_images i ON d.doc_id = i.doc_id
                    WHERE d.kb_id = %s
                    GROUP BY d.doc_id, d.file_name
                    ORDER BY d.doc_id
                """, (kb_id,))
            else:
                cur.execute("""
                    SELECT d.doc_id, d.file_name, COUNT(i.image_id) as image_count
                    FROM documents d
                    JOIN document_images i ON d.doc_id = i.doc_id
                    GROUP BY d.doc_id, d.file_name
                    ORDER BY d.doc_id
                """)
            rows = cur.fetchall()
            return [{
                "doc_id": row[0],
                "file_name": row[1],
                "image_count": row[2]
            } for row in rows]
    
    def get_tables_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM document_tables WHERE doc_id = %s", (doc_id,))
            rows = cur.fetchall()
            return [{
                "table_id": row[0],
                "doc_id": row[1],
                "chunk_id": row[2],
                "title": row[3],
                "description": row[4],
                "meta_data": row[5],
                "created_at": row[6]
            } for row in rows]

    def insert_kg_log(self, log_id, task_id, doc_id, kb_id, chunk_id=None, chunk_index=0, chunk_title=None, 
                      log_level='INFO', log_type='general', message='', entity_count=0, relation_count=0,
                      community_count=0,
                      entities=None, relations=None, progress=0, status='running',
                      kb_name=None, doc_name=None):
        import json
        # 名字快照：调用方未传则按缓存/查库补全，确保文档删除后日志仍带来源名
        if kb_name is None or doc_name is None:
            cached_kb, cached_doc = self._get_names(kb_id, doc_id)
            if kb_name is None:
                kb_name = cached_kb
            if doc_name is None:
                doc_name = cached_doc
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO kg_generation_logs
                    (log_id, task_id, doc_id, kb_id, chunk_id, chunk_index, chunk_title,
                     log_level, log_type, message, entity_count, relation_count, community_count,
                     entities, relations, progress, status, kb_name, doc_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (log_id, task_id, doc_id, kb_id, chunk_id, chunk_index, chunk_title,
                      log_level, log_type, message, entity_count, relation_count, community_count,
                      json.dumps(entities) if entities else None,
                      json.dumps(relations) if relations else None,
                      progress, status, kb_name, doc_name))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"insert_kg_log failed (task={task_id}): {e}", flush=True)

    def get_kg_logs_by_task(self, task_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM kg_generation_logs WHERE task_id = %s ORDER BY created_at
            """, (task_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_kg_logs_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM kg_generation_logs WHERE doc_id = %s ORDER BY created_at DESC
            """, (doc_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_kg_logs_by_kb(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM kg_generation_logs WHERE kb_id = %s ORDER BY created_at DESC
            """, (kb_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_kg_log_summary_by_task(self, task_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_logs,
                    SUM(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as error_count,
                    SUM(entity_count) as total_entities,
                    SUM(relation_count) as total_relations,
                    MAX(progress) as max_progress,
                    MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                    MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM kg_generation_logs WHERE task_id = %s
            """, (task_id,))
            row = cur.fetchone()
            if row:
                return {
                    "task_id": task_id,
                    "total_logs": row[0],
                    "error_count": row[1],
                    "total_entities": row[2],
                    "total_relations": row[3],
                    "max_progress": row[4],
                    "is_finished": bool(row[5]),
                    "is_cancelled": bool(row[6]),
                    "start_time": row[7],
                    "end_time": row[8]
                }
            return None

    def insert_parse_log(self, log_id, task_id, doc_id, kb_id, block_key=None, block_title=None,
                        log_level='INFO', log_type='general', message='', block_count=0,
                        image_count=0, table_count=0, chunk_count=0, progress=0, status='running',
                        kb_name=None, doc_name=None):
        # 名字快照：调用方未传则按缓存/查库补全，确保文档删除后日志仍带来源名
        if kb_name is None or doc_name is None:
            cached_kb, cached_doc = self._get_names(kb_id, doc_id)
            if kb_name is None:
                kb_name = cached_kb
            if doc_name is None:
                doc_name = cached_doc
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_parse_logs
                    (log_id, task_id, doc_id, kb_id, block_key, block_title,
                     log_level, log_type, message, block_count, image_count,
                     table_count, chunk_count, progress, status, kb_name, doc_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (log_id, task_id, doc_id, kb_id, block_key, block_title,
                      log_level, log_type, message, block_count, image_count,
                      table_count, chunk_count, progress, status, kb_name, doc_name))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"insert_parse_log failed (task={task_id}): {e}", flush=True)

    def get_parse_logs_by_task(self, task_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM document_parse_logs WHERE task_id = %s ORDER BY created_at
            """, (task_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_parse_logs_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM document_parse_logs WHERE doc_id = %s ORDER BY created_at DESC
            """, (doc_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_parse_logs_by_kb(self, kb_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM document_parse_logs WHERE kb_id = %s ORDER BY created_at DESC
            """, (kb_id,))
            return [self._row_to_dict(cur, row) for row in cur.fetchall()]

    def get_parse_log_summary_by_task(self, task_id):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_logs,
                    SUM(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as error_count,
                    MAX(block_count) as max_block_count,
                    MAX(image_count) as max_image_count,
                    MAX(table_count) as max_table_count,
                    MAX(chunk_count) as max_chunk_count,
                    MAX(progress) as max_progress,
                    MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                    MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM document_parse_logs WHERE task_id = %s
            """, (task_id,))
            row = cur.fetchone()
            if row:
                return {
                    "task_id": task_id,
                    "total_logs": row[0],
                    "error_count": row[1],
                    "max_block_count": row[2],
                    "max_image_count": row[3],
                    "max_table_count": row[4],
                    "max_chunk_count": row[5],
                    "max_progress": row[6],
                    "is_finished": bool(row[7]),
                    "is_cancelled": bool(row[8]),
                    "start_time": row[9],
                    "end_time": row[10]
                }
            return None

    def get_all_task_summaries(self, kb_id=None):
        with self.conn.cursor() as cur:
            if kb_id:
                cur.execute("""
                    SELECT 'parse' as task_type, task_id, doc_id, kb_id,
                           MAX(progress) as max_progress,
                           MAX(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as has_error,
                           MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                           MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                           MIN(created_at) as start_time,
                           MAX(created_at) as end_time,
                           MAX(kb_name) as kb_name,
                           MAX(doc_name) as doc_name
                    FROM document_parse_logs
                    WHERE kb_id = %s
                    GROUP BY task_id, doc_id, kb_id
                    UNION ALL
                    SELECT 'kg' as task_type, task_id, doc_id, kb_id,
                           MAX(progress) as max_progress,
                           MAX(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as has_error,
                           MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                           MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                           MIN(created_at) as start_time,
                           MAX(created_at) as end_time,
                           MAX(kb_name) as kb_name,
                           MAX(doc_name) as doc_name
                    FROM kg_generation_logs
                    WHERE kb_id = %s
                    GROUP BY task_id, doc_id, kb_id
                    ORDER BY start_time DESC
                """, (kb_id, kb_id))
            else:
                cur.execute("""
                    SELECT 'parse' as task_type, task_id, doc_id, kb_id,
                           MAX(progress) as max_progress,
                           MAX(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as has_error,
                           MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                           MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                           MIN(created_at) as start_time,
                           MAX(created_at) as end_time,
                           MAX(kb_name) as kb_name,
                           MAX(doc_name) as doc_name
                    FROM document_parse_logs
                    GROUP BY task_id, doc_id, kb_id
                    UNION ALL
                    SELECT 'kg' as task_type, task_id, doc_id, kb_id,
                           MAX(progress) as max_progress,
                           MAX(CASE WHEN log_level = 'ERROR' THEN 1 ELSE 0 END) as has_error,
                           MAX(CASE WHEN log_type IN ('finish', 'cancel', 'error') THEN 1 ELSE 0 END) as is_finished,
                           MAX(CASE WHEN log_type = 'cancel' THEN 1 ELSE 0 END) as is_cancelled,
                           MIN(created_at) as start_time,
                           MAX(created_at) as end_time,
                           MAX(kb_name) as kb_name,
                           MAX(doc_name) as doc_name
                    FROM kg_generation_logs
                    GROUP BY task_id, doc_id, kb_id
                    ORDER BY start_time DESC
                """)
            rows = cur.fetchall()

            summaries = []
            for row in rows:
                status = 'running'
                if row[6]:  # is_finished (finish/cancel/error)
                    if row[7]:  # is_cancelled
                        status = 'cancelled'
                    elif row[5]:  # has_error
                        status = 'failed'
                    else:
                        status = 'completed'
                elif row[4] >= 100:
                    status = 'completed'

                summaries.append({
                    "task_type": row[0],
                    "task_id": row[1],
                    "doc_id": row[2],
                    "kb_id": row[3],
                    "progress": row[4],
                    "has_error": bool(row[5]),
                    "is_finished": bool(row[6]),
                    "is_cancelled": bool(row[7]),
                    "start_time": row[8],
                    "end_time": row[9],
                    "kb_name": row[10],
                    "doc_name": row[11],
                    "status": status
                })

            return summaries

    def delete_kg_logs_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM kg_generation_logs WHERE doc_id = %s", (doc_id,))
            self.conn.commit()
            return cur.rowcount

    def delete_parse_logs_by_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM document_parse_logs WHERE doc_id = %s", (doc_id,))
            self.conn.commit()
            return cur.rowcount

    # ===================== ICP 类型驱动文档体系 =====================

    def list_document_types(self):
        """返回类型注册表（含每类文档数与索引记录统计）。"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM document_types WHERE is_active = TRUE ORDER BY sort_order")
            types = [self._row_to_dict(cur, r) for r in cur.fetchall()]
        for t in types:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM icp_documents WHERE doc_type = %s", (t["doc_type"],))
            t["doc_count"] = cur.fetchone()[0]
            if t["doc_type"] in ('hmc', 'fmeca', 'mfl'):
                cur.execute(f"SELECT COUNT(*) FROM {t['pg_source']}")
                t["record_count"] = cur.fetchone()[0]
            elif t["doc_type"] in ('fta', 'rbd'):
                cur.execute("SELECT COUNT(*) FROM document_fault_trees WHERE kb_id = %s", (t["doc_type"],))
                t["record_count"] = cur.fetchone()[0]
            else:
                cur.execute("SELECT COUNT(*) FROM document_chunks WHERE section = %s", (t["doc_type"],))
                t["record_count"] = cur.fetchone()[0]
        return types

    def get_document_type(self, doc_type):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM document_types WHERE doc_type = %s", (doc_type,))
            row = cur.fetchone()
            return self._row_to_dict(cur, row) if row else None

    def insert_icp_document(self, doc_id, doc_type, file_name, minio_path):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO icp_documents (doc_id, doc_type, file_name, minio_path)
                VALUES (%s, %s, %s, %s)
            """, (doc_id, doc_type, file_name, minio_path))
            self.conn.commit()

    def get_icp_document(self, doc_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM icp_documents WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                result = self._row_to_dict(cur, row) if row else None
            # psycopg opens a transaction for SELECT as well.  This method is
            # used by graph reauthorization; leaving it idle would retain an
            # ACCESS SHARE lock and block a concurrent idempotent schema check.
            self.conn.commit()
            return result
        except Exception:
            self.conn.rollback()
            raise

    def list_icp_documents(self, doc_type, status=None):
        with self.conn.cursor() as cur:
            if status:
                cur.execute("SELECT * FROM icp_documents WHERE doc_type = %s AND status = %s ORDER BY created_at DESC",
                            (doc_type, status))
            else:
                cur.execute("SELECT * FROM icp_documents WHERE doc_type = %s ORDER BY created_at DESC", (doc_type,))
            return [self._row_to_dict(cur, r) for r in cur.fetchall()]

    def update_icp_document_status(self, doc_id, status):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE icp_documents SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE doc_id = %s",
                        (status, doc_id))
            self.conn.commit()

    def transition_icp_document_status(self, doc_id, allowed_statuses, new_status):
        """Atomically acquire an ICP document for parse/generate/delete work."""
        allowed = tuple(allowed_statuses)
        if not allowed:
            return False
        placeholders = ", ".join(["%s"] * len(allowed))
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE icp_documents
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = %s AND status IN ({placeholders})
                    """,
                    (new_status, doc_id, *allowed),
                )
                changed = cur.rowcount > 0
            self.conn.commit()
            return changed
        except Exception:
            self.conn.rollback()
            raise

    def update_icp_document_result(self, doc_id, last_result):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE icp_documents SET last_result = %s::jsonb, parse_count = parse_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s
            """, (json.dumps(last_result, ensure_ascii=False), doc_id))
            self.conn.commit()

    def patch_icp_document_result(self, doc_id, result_patch):
        """Merge asynchronous follow-up state without counting a new parse."""

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE icp_documents
                SET last_result = COALESCE(last_result, '{}'::jsonb) || %s::jsonb,
                    updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s
                """,
                (json.dumps(result_patch or {}, ensure_ascii=False), doc_id),
            )
            self.conn.commit()

    def update_icp_document_metadata(self, doc_id, metadata):
        """Persist controlled cover metadata before creating search projections."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE icp_documents
                SET metadata = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s
                """,
                (json.dumps(metadata or {}, ensure_ascii=False), doc_id),
            )
            self.conn.commit()

    def delete_icp_document(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM icp_documents WHERE doc_id = %s", (doc_id,))
            self.conn.commit()

    def delete_icp_document_with_sources(self, doc_id):
        """Atomically delete one ICP master row and all PostgreSQL projections."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM fault_hmc WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM fmea WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM mfl WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_fault_trees WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_tables WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_images WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM icp_documents WHERE doc_id = %s", (doc_id,))
                deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted
        except Exception:
            self.conn.rollback()
            raise

    def upsert_hmc(self, doc_id, row):
        """插入/更新 HMC 行，返回行 id（供检索投影 pg_id_kwd 回链真源表）。"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fault_hmc
                    (doc_id, variant_id, hmc_code, lcn, hamc, hsc, fault_desc, repair_method,
                     source_ref, raw_data, normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, variant_id) DO UPDATE SET
                    hmc_code = EXCLUDED.hmc_code,
                    lcn = EXCLUDED.lcn,
                    hamc = EXCLUDED.hamc,
                    hsc = EXCLUDED.hsc,
                    fault_desc = EXCLUDED.fault_desc,
                    repair_method = EXCLUDED.repair_method,
                    source_ref = EXCLUDED.source_ref,
                    raw_data = EXCLUDED.raw_data,
                    normalization_warnings = EXCLUDED.normalization_warnings,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (doc_id, row.get("variant_id") or hashlib.sha256(
                      f"hmc\x1f{row.get('hmc_code', '')}\x1f{row.get('source_ref', '')}".encode("utf-8")
                  ).hexdigest()[:32],
                  row.get("hmc_code"), row.get("lcn", "**"), row.get("hamc", "**"),
                  row.get("hsc", "**"), row.get("fault_desc", ""), row.get("repair_method", ""),
                  row.get("source_ref"), json.dumps(row.get("raw_data") or {}, ensure_ascii=False),
                  json.dumps(row.get("normalization_warnings") or [], ensure_ascii=False)))
            row_id = cur.fetchone()[0]
            self.conn.commit()
            return row_id

    def upsert_fmea(self, doc_id, row):
        """插入/更新 FMECA 行，返回行 id（供检索投影 pg_id_kwd 回链真源表）。"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fmea
                    (doc_id, code, product_flag, func_management, fault_mode, fault_cause,
                     mission_phase, local_impact, final_impact, severity, severity_raw,
                     severity_normalized,
                     probability, probability_raw, probability_level, detect_method,
                     design_improve, compensate_action, source_ref, raw_data,
                     normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, code) DO UPDATE SET
                    product_flag = EXCLUDED.product_flag,
                    func_management = EXCLUDED.func_management, fault_mode = EXCLUDED.fault_mode,
                    fault_cause = EXCLUDED.fault_cause, mission_phase = EXCLUDED.mission_phase,
                    local_impact = EXCLUDED.local_impact, final_impact = EXCLUDED.final_impact,
                    severity = EXCLUDED.severity, severity_raw = EXCLUDED.severity_raw,
                    severity_normalized = EXCLUDED.severity_normalized,
                    probability = EXCLUDED.probability, probability_raw = EXCLUDED.probability_raw,
                    probability_level = EXCLUDED.probability_level,
                    detect_method = EXCLUDED.detect_method, design_improve = EXCLUDED.design_improve,
                    compensate_action = EXCLUDED.compensate_action,
                    source_ref = EXCLUDED.source_ref, raw_data = EXCLUDED.raw_data,
                    normalization_warnings = EXCLUDED.normalization_warnings,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (doc_id, row.get("code"), row.get("product_flag"), row.get("func_management"),
                  row.get("fault_mode", ""), row.get("fault_cause"), row.get("mission_phase"),
                  row.get("local_impact"), row.get("final_impact"), row.get("severity"),
                  row.get("severity_raw"), row.get("severity_normalized", row.get("severity")),
                  (row.get("probability_value")
                   if row.get("probability_value") is not None else row.get("probability")),
                  row.get("probability_raw", row.get("probability")),
                  row.get("probability_level") or row.get("probability_class"),
                  row.get("detect_method"), row.get("design_improve"), row.get("compensate_action"),
                  row.get("source_ref"), json.dumps(row.get("raw_data") or {}, ensure_ascii=False),
                  json.dumps(row.get("normalization_warnings") or [], ensure_ascii=False)))
            row_id = cur.fetchone()[0]
            self.conn.commit()
            return row_id

    def upsert_mfl(self, doc_id, row):
        """插入/更新 MFL 行，返回行 id（供检索投影 pg_id_kwd 回链真源表）。"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO mfl
                    (doc_id, mfl_code, module_fault, sur_fault, subcircuit_fault,
                     api_reportable, api_reportable_raw, api_name, api_method, api_usage,
                     platform_test, platform_test_raw, platform_test_status,
                     maintenance_action, source_ref, raw_data, normalization_warnings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (doc_id, mfl_code) DO UPDATE SET
                    module_fault = EXCLUDED.module_fault,
                    sur_fault = EXCLUDED.sur_fault, subcircuit_fault = EXCLUDED.subcircuit_fault,
                    api_reportable = EXCLUDED.api_reportable,
                    api_reportable_raw = EXCLUDED.api_reportable_raw, api_name = EXCLUDED.api_name,
                    api_method = EXCLUDED.api_method, api_usage = EXCLUDED.api_usage,
                    platform_test = EXCLUDED.platform_test,
                    platform_test_raw = EXCLUDED.platform_test_raw,
                    platform_test_status = EXCLUDED.platform_test_status,
                    maintenance_action = EXCLUDED.maintenance_action,
                    source_ref = EXCLUDED.source_ref, raw_data = EXCLUDED.raw_data,
                    normalization_warnings = EXCLUDED.normalization_warnings,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (doc_id, row.get("mfl_code"), row.get("module_fault"), row.get("sur_fault"),
                  row.get("subcircuit_fault"), row.get("api_reportable", False),
                  row.get("api_reportable_raw"), row.get("api_name"), row.get("api_method"),
                  row.get("api_usage"), row.get("platform_test", False),
                  row.get("platform_test_raw"), row.get("platform_test_status"),
                  row.get("maintenance_action") or row.get("repair_action"), row.get("source_ref"),
                  json.dumps(row.get("raw_data") or {}, ensure_ascii=False),
                  json.dumps(row.get("normalization_warnings") or [], ensure_ascii=False)))
            row_id = cur.fetchone()[0]
            self.conn.commit()
            return row_id

    def get_hmc_rows(self, doc_id=None, page=1, size=50, filters=None):
        return self._query_rows("fault_hmc", doc_id, page, size, filters)

    def get_fmea_rows(self, doc_id=None, page=1, size=50, filters=None):
        return self._query_rows("fmea", doc_id, page, size, filters)

    def get_mfl_rows(self, doc_id=None, page=1, size=50, filters=None):
        return self._query_rows("mfl", doc_id, page, size, filters)

    _ROW_SEARCH_COLS = {
        "fault_hmc": ["hmc_code", "lcn", "hamc", "hsc", "fault_desc", "repair_method"],
        "fmea": ["code", "product_flag", "func_management", "fault_mode", "fault_cause",
                 "mission_phase", "local_impact", "final_impact", "severity", "severity_raw",
                 "probability_raw", "probability_level", "detect_method",
                 "design_improve", "compensate_action"],
        "mfl": ["mfl_code", "module_fault", "sur_fault", "subcircuit_fault",
                "api_name", "api_method", "api_usage", "platform_test_status",
                "maintenance_action"],
    }

    def _query_rows(self, table, doc_id, page, size, filters):
        """真源表行查询（doc_id 过滤 + 列过滤 + keyword 全列检索 + 分页）。
        filters 为 {列名: 值}，列名须在表结构内；keyword 按文本列 ILIKE。"""
        if table not in self._ROW_SEARCH_COLS:
            # table 目前只由本类三个方法以硬编码调用，但 SQL 中表名直接
            # 拼接，必须先经白名单校验，防止未来暴露给外部输入时成为注入面。
            raise ValueError(f"非法真源表名: {table!r}")
        where = []
        params = []
        if doc_id:
            where.append("doc_id = %s")
            params.append(doc_id)
        if filters:
            for col, val in filters.items():
                if val in (None, ""):
                    continue
                if col == "keyword":
                    # 列名来自白名单（_ROW_SEARCH_COLS），安全直接拼接；值走参数绑定
                    cols = self._ROW_SEARCH_COLS.get(table, [])
                    like = " OR ".join([f"CAST({c} AS TEXT) ILIKE '%%' || %s || '%%'" for c in cols])
                    if like:
                        params.extend([val] * len(cols))
                        where.append(f"({like})")
                elif col in self._ROW_SEARCH_COLS.get(table, []):
                    where.append(f"{col} = %s")
                    params.append(val)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        offset = (page - 1) * size
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table} {where_sql}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT * FROM {table} {where_sql} ORDER BY id DESC LIMIT %s OFFSET %s",
                params + [size, offset])
            rows = [self._row_to_dict(cur, r) for r in cur.fetchall()]
        return {"total": total, "page": page, "size": size, "rows": rows}

    def delete_source_rows_by_doc(self, doc_id):
        """删除真源表与从属数据（删除文档流程用）。"""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM fault_hmc WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM fmea WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM mfl WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM document_fault_trees WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM document_tables WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM document_images WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
            self.conn.commit()

    def clear_icp_authoritative_rows(self, doc_id, doc_type):
        """Clear one table-backed document snapshot before a deterministic reparse.

        The table name is selected from a closed mapping; callers cannot inject
        SQL identifiers.  Raw artifacts remain untouched and can always rebuild
        this snapshot if the following parse fails.
        """
        table = {
            "hmc": "fault_hmc",
            "fmeca": "fmea",
            "mfl": "mfl",
        }.get(str(doc_type))
        if not table:
            raise ValueError(f"不支持清理的表格文档类型: {doc_type}")
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted

class MockPostgres:
    def __init__(self):
        self.knowledge_bases = []
        self.documents = []
        self.chunks = []
        self.images = []
        self.tables = []
        self.models = []
        self.kg_logs = []
        self.parse_logs = []
        self.deletion_logs = []
        self.community_reports = []
    
    def cursor(self):
        return MockCursor(self)
    
    def commit(self):
        pass

    def rollback(self):
        pass

class MockCursor:
    def __init__(self, db):
        self.db = db
        self.rowcount = 0
        self._result = []
        self.description = None  # 真实游标才有列描述；mock 模式下为 None
    
    def execute(self, query, params=None):
        self.rowcount = 0
        self._result = []
        
        if "INSERT INTO community_reports" in query:
            projection_id = params[0]
            data = {
                "projection_id": projection_id,
                "report_id": params[1],
                "community_id": params[2],
                "kb_id": params[3],
                "doc_id": params[4],
                "doc_ids": json.loads(params[5]),
                "entities": json.loads(params[6]),
                "report": json.loads(params[7]),
                "stale_kwd": params[8],
                "stale_reason": params[9],
                "stale_at": params[10],
                "created_at": "2024-01-01 00:00:00",
                "updated_at": "2024-01-01 00:00:00",
            }
            existing = next(
                (item for item in self.db.community_reports if item["projection_id"] == projection_id),
                None,
            )
            if existing:
                existing.update(data)
            else:
                self.db.community_reports.append(data)
            self.rowcount = 1

        elif "SELECT projection_id, report_id, community_id" in query and "FROM community_reports" in query:
            kb_id = params[0]
            doc_id = json.loads(params[1])[0] if len(params) > 1 else None
            for item in self.db.community_reports:
                if item["kb_id"] != kb_id or (doc_id and doc_id not in item.get("doc_ids", [])):
                    continue
                self._result.append(tuple(item.get(field) for field in (
                    "projection_id", "report_id", "community_id", "kb_id", "doc_id",
                    "doc_ids", "entities", "report", "stale_kwd", "stale_reason",
                    "stale_at", "created_at", "updated_at",
                )))

        elif "UPDATE community_reports" in query:
            projection_id = params[5]
            for item in self.db.community_reports:
                if item["projection_id"] == projection_id:
                    item.update({
                        "doc_ids": json.loads(params[0]),
                        "doc_id": params[1],
                        "stale_kwd": params[2],
                        "stale_reason": params[3],
                        "stale_at": params[4],
                    })
                    self.rowcount = 1
                    break

        elif "DELETE FROM community_reports WHERE projection_id" in query:
            original_count = len(self.db.community_reports)
            self.db.community_reports = [
                item for item in self.db.community_reports if item["projection_id"] != params[0]
            ]
            self.rowcount = original_count - len(self.db.community_reports)

        elif "DELETE FROM community_reports WHERE kb_id" in query:
            original_count = len(self.db.community_reports)
            self.db.community_reports = [
                item for item in self.db.community_reports if item["kb_id"] != params[0]
            ]
            self.rowcount = original_count - len(self.db.community_reports)

        elif "INSERT INTO knowledge_bases" in query:
            self.db.knowledge_bases.append({
                "kb_id": params[0],
                "name": params[1],
                "description": params[2],
                "created_at": "2024-01-01 00:00:00",
                "updated_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "INSERT INTO documents" in query:
            self.db.documents.append({
                "doc_id": params[0],
                "file_name": params[1],
                "kb_id": params[2],
                "minio_path": params[3],
                "status": "uploaded",
                "total_chunks": 0,
                "delete_step": None,
                "created_at": "2024-01-01 00:00:00",
                "updated_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "INSERT INTO document_chunks" in query:
            doc = next((d for d in self.db.documents if d["doc_id"] == params[1]), None)
            kb_id = doc["kb_id"] if doc else ""
            self.db.chunks.append({
                "chunk_id": params[0],
                "doc_id": params[1],
                "kb_id": kb_id,
                "section": params[2],
                "title": params[3],
                "content": params[4],
                "content_type": params[5],
                "chunk_index": params[6],
                "selected_for_kg": False,
                "created_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "INSERT INTO document_images" in query:
            chunk = next((c for c in self.db.chunks if c["chunk_id"] == params[2]), None)
            kb_id = chunk["kb_id"] if chunk else ""
            self.db.images.append({
                "image_id": params[0],
                "doc_id": params[1],
                "chunk_id": params[2],
                "kb_id": kb_id,
                "title": params[3],
                "description": params[4],
                "minio_path": params[5],
                "image_type": params[6],
                "image_url": params[7] if len(params) > 7 else None,
                "mime_type": params[8] if len(params) > 8 else None,
                "source_ref": params[9] if len(params) > 9 else None,
                "created_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "INSERT INTO document_tables" in query:
            chunk = next((c for c in self.db.chunks if c["chunk_id"] == params[2]), None)
            kb_id = chunk["kb_id"] if chunk else ""
            self.db.tables.append({
                "table_id": params[0],
                "doc_id": params[1],
                "chunk_id": params[2],
                "kb_id": kb_id,
                "title": params[3],
                "description": params[4],
                "meta_data": params[5],
                "created_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "INSERT INTO llm_models" in query:
            if params[12]:
                for m in self.db.models:
                    m["is_default"] = False
            self.db.models.append({
                "model_id": params[0],
                "name": params[1],
                "model_type": params[2],
                "provider": params[3],
                "api_key": params[4],
                "api_url": params[5],
                "model_name": params[6],
                "base_url": params[7],
                "temperature": params[8],
                "max_tokens": params[9],
                "top_p": params[10],
                "timeout": params[11],
                "is_default": params[12],
                "is_active": params[13],
                "description": params[14],
                "created_at": "2024-01-01 00:00:00",
                "updated_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1
        
        elif "UPDATE llm_models SET is_default = FALSE WHERE model_type" in query:
            model_type = params[0]
            for m in self.db.models:
                if m.get("model_type", "llm") == model_type:
                    m["is_default"] = False
                    self.rowcount += 1
        
        elif "UPDATE llm_models SET is_default = FALSE" in query:
            for m in self.db.models:
                m["is_default"] = False
            self.rowcount = len(self.db.models)
        
        elif "UPDATE llm_models SET" in query:
            model_id = params[-1]
            for m in self.db.models:
                if m["model_id"] == model_id:
                    m.update(dict(zip(["name", "model_type", "provider", "api_key", "api_url", "model_name", 
                                       "base_url", "temperature", "max_tokens", "top_p", 
                                       "timeout", "is_default", "is_active", "description"], params[:-1])))
                    m["updated_at"] = "2024-01-01 00:00:00"
                    self.rowcount = 1
                    break
        
        elif "DELETE FROM llm_models" in query:
            original_count = len(self.db.models)
            self.db.models = [m for m in self.db.models if m["model_id"] != params[0]]
            self.rowcount = original_count - len(self.db.models)
        
        elif "SELECT * FROM llm_models WHERE model_id" in query:
            for m in self.db.models:
                if m["model_id"] == params[0]:
                    self._result.append((
                        m["model_id"], m["name"], m.get("model_type", "llm"), m["provider"],
                        m["api_key"], m["api_url"], m["model_name"], m["base_url"],
                        m["temperature"], m["max_tokens"], m["top_p"], m["timeout"],
                        m["is_default"], m["is_active"], m["description"], m["created_at"], m["updated_at"]
                    ))
                    break
        
        elif "SELECT * FROM llm_models WHERE is_active = TRUE ORDER BY is_default DESC" in query:
            for m in self.db.models:
                if m["is_active"]:
                    self._result.append((
                        m["model_id"], m["name"], m.get("model_type", "llm"), m["provider"],
                        m["api_key"], m["api_url"], m["model_name"], m["base_url"],
                        m["temperature"], m["max_tokens"], m["top_p"], m["timeout"],
                        m["is_default"], m["is_active"], m["description"], m["created_at"], m["updated_at"]
                    ))
        
        elif "SELECT * FROM llm_models WHERE is_default = TRUE AND is_active = TRUE" in query:
            for m in self.db.models:
                if m["is_default"] and m["is_active"]:
                    self._result.append((
                        m["model_id"], m["name"], m.get("model_type", "llm"), m["provider"],
                        m["api_key"], m["api_url"], m["model_name"], m["base_url"],
                        m["temperature"], m["max_tokens"], m["top_p"], m["timeout"],
                        m["is_default"], m["is_active"], m["description"], m["created_at"], m["updated_at"]
                    ))
                    break
        
        elif "SELECT * FROM llm_models WHERE model_type" in query:
            model_type = params[0]
            for m in self.db.models:
                if m.get("model_type", "llm") == model_type and m["is_active"]:
                    if "is_default = TRUE" in query:
                        if m["is_default"]:
                            self._result.append((
                                m["model_id"], m["name"], m.get("model_type", "llm"), m["provider"],
                                m["api_key"], m["api_url"], m["model_name"], m["base_url"],
                                m["temperature"], m["max_tokens"], m["top_p"], m["timeout"],
                                m["is_default"], m["is_active"], m["description"], m["created_at"], m["updated_at"]
                            ))
                            break
                    else:
                        self._result.append((
                            m["model_id"], m["name"], m.get("model_type", "llm"), m["provider"],
                            m["api_key"], m["api_url"], m["model_name"], m["base_url"],
                            m["temperature"], m["max_tokens"], m["top_p"], m["timeout"],
                            m["is_default"], m["is_active"], m["description"], m["created_at"], m["updated_at"]
                        ))
        
        elif "UPDATE documents SET status" in query:
            for doc in self.db.documents:
                if doc["doc_id"] == params[1]:
                    doc["status"] = params[0]
                    self.rowcount = 1
                    break

        elif "UPDATE documents SET delete_step" in query:
            for doc in self.db.documents:
                if doc["doc_id"] == params[1]:
                    doc["delete_step"] = params[0]
                    self.rowcount = 1
                    break
        
        elif "UPDATE documents SET total_chunks" in query:
            for doc in self.db.documents:
                if doc["doc_id"] == params[1]:
                    doc["total_chunks"] = params[0]
                    self.rowcount = 1
                    break
        
        elif "UPDATE document_chunks SET selected_for_kg" in query:
            for chunk in self.db.chunks:
                if chunk["chunk_id"] == params[1]:
                    chunk["selected_for_kg"] = params[0]
                    self.rowcount = 1
                    break
        
        elif "SELECT * FROM knowledge_bases WHERE kb_id" in query:
            for kb in self.db.knowledge_bases:
                if kb["kb_id"] == params[0]:
                    self._result.append((
                        kb["kb_id"], kb["name"], kb["description"],
                        kb["created_at"], kb["updated_at"]
                    ))
                    break
        
        elif "SELECT * FROM knowledge_bases ORDER BY created_at DESC" in query:
            for kb in self.db.knowledge_bases:
                self._result.append((
                    kb["kb_id"], kb["name"], kb["description"],
                    kb["created_at"], kb["updated_at"]
                ))
        
        elif "SELECT * FROM documents WHERE doc_id" in query:
            for doc in self.db.documents:
                if doc["doc_id"] == params[0]:
                    self._result.append((
                        doc["doc_id"], doc["file_name"], doc["kb_id"],
                        doc["status"], doc["minio_path"], doc["total_chunks"],
                        doc["created_at"], doc["updated_at"]
                    ))
                    break
        
        elif "SELECT * FROM documents WHERE kb_id" in query:
            for doc in self.db.documents:
                if doc["kb_id"] == params[0]:
                    self._result.append((
                        doc["doc_id"], doc["file_name"], doc["kb_id"],
                        doc["status"], doc["minio_path"], doc["total_chunks"],
                        doc["created_at"], doc["updated_at"]
                    ))

        elif "SELECT delete_step FROM documents WHERE doc_id" in query:
            for doc in self.db.documents:
                if doc["doc_id"] == params[0]:
                    self._result.append((doc.get("delete_step"),))
                    break

        elif "FROM documents WHERE status" in query:
            for doc in self.db.documents:
                if doc["status"] == params[0]:
                    self._result.append((
                        doc["doc_id"], doc["file_name"], doc["kb_id"],
                        doc["status"], doc["minio_path"], doc["total_chunks"],
                        doc.get("delete_step"), doc["created_at"], doc["updated_at"]
                    ))
        
        elif "SELECT * FROM document_chunks WHERE doc_id" in query:
            for chunk in self.db.chunks:
                if chunk["doc_id"] == params[0]:
                    self._result.append((
                        chunk["chunk_id"], chunk["doc_id"], chunk["section"], chunk["title"],
                        chunk["content"], chunk["content_type"], chunk["chunk_index"],
                        chunk["selected_for_kg"], chunk["created_at"]
                    ))
        
        elif "SELECT * FROM document_chunks WHERE doc_id = %s AND selected_for_kg = TRUE" in query:
            for chunk in self.db.chunks:
                if chunk["doc_id"] == params[0] and chunk["selected_for_kg"]:
                    self._result.append((
                        chunk["chunk_id"], chunk["doc_id"], chunk["section"], chunk["title"],
                        chunk["content"], chunk["content_type"], chunk["chunk_index"],
                        chunk["selected_for_kg"], chunk["created_at"]
                    ))
        
        elif "FROM document_images WHERE chunk_id" in query:
            for img in self.db.images:
                if img["chunk_id"] == params[0]:
                    self._result.append((
                        img["image_id"], img["doc_id"], img["chunk_id"], img["title"],
                        img["description"], img["minio_path"], img["image_type"],
                        img.get("image_url"), img.get("mime_type"), img.get("source_ref"),
                        img["created_at"],
                    ))
        
        elif "SELECT * FROM document_tables WHERE chunk_id" in query:
            for tbl in self.db.tables:
                if tbl["chunk_id"] == params[0]:
                    self._result.append((
                        tbl["table_id"], tbl["doc_id"], tbl["chunk_id"], tbl["title"],
                        tbl["description"], tbl["meta_data"], tbl["created_at"]
                    ))
        
        elif "DELETE FROM knowledge_bases" in query:
            original_count = len(self.db.knowledge_bases)
            self.db.knowledge_bases = [kb for kb in self.db.knowledge_bases if kb["kb_id"] != params[0]]
            self.db.documents = [d for d in self.db.documents if d["kb_id"] != params[0]]
            self.db.chunks = [c for c in self.db.chunks if c["kb_id"] != params[0]]
            self.db.images = [i for i in self.db.images if i["kb_id"] != params[0]]
            self.db.tables = [t for t in self.db.tables if t["kb_id"] != params[0]]
            self.rowcount = original_count - len(self.db.knowledge_bases)
        
        elif "DELETE FROM documents" in query:
            original_count = len(self.db.documents)
            self.db.documents = [d for d in self.db.documents if d["doc_id"] != params[0]]
            self.db.chunks = [c for c in self.db.chunks if c["doc_id"] != params[0]]
            self.db.images = [i for i in self.db.images if i["doc_id"] != params[0]]
            self.db.tables = [t for t in self.db.tables if t["doc_id"] != params[0]]
            self.rowcount = original_count - len(self.db.documents)
        
        elif "DELETE FROM document_chunks" in query:
            original_count = len(self.db.chunks)
            self.db.chunks = [c for c in self.db.chunks if c["doc_id"] != params[0]]
            self.rowcount = original_count - len(self.db.chunks)
        
        elif "DELETE FROM kg_generation_logs" in query:
            original_count = len(self.db.kg_logs)
            self.db.kg_logs = [l for l in self.db.kg_logs if l.get("doc_id") != params[0]]
            self.rowcount = original_count - len(self.db.kg_logs)
        
        elif "DELETE FROM document_parse_logs" in query:
            original_count = len(self.db.parse_logs)
            self.db.parse_logs = [l for l in self.db.parse_logs if l.get("doc_id") != params[0]]
            self.rowcount = original_count - len(self.db.parse_logs)
        
        elif "FROM document_images WHERE doc_id" in query:
            for img in self.db.images:
                if img["doc_id"] == params[0]:
                    self._result.append((
                        img["image_id"], img["doc_id"], img["chunk_id"], img["title"],
                        img["description"], img["minio_path"], img["image_type"],
                        img.get("image_url"), img.get("mime_type"), img.get("source_ref"),
                        img["created_at"],
                    ))
        
        elif "SELECT * FROM document_tables WHERE doc_id" in query:
            for tbl in self.db.tables:
                if tbl["doc_id"] == params[0]:
                    self._result.append((
                        tbl["table_id"], tbl["doc_id"], tbl["chunk_id"], tbl["title"],
                        tbl["description"], tbl["meta_data"], tbl["created_at"]
                    ))

        elif "INSERT INTO deletion_logs" in query:
            self.db.deletion_logs.append({
                "id": len(self.db.deletion_logs) + 1,
                "task_id": params[0],
                "doc_id": params[1],
                "kb_id": params[2],
                "step": params[3],
                "item_type": params[4],
                "item_id": params[5],
                "item_detail": json.loads(params[6]) if params[6] else None,
                "status": params[7],
                "message": params[8],
                "kb_name": params[9] if len(params) > 9 else None,
                "doc_name": params[10] if len(params) > 10 else None,
                "created_at": "2024-01-01 00:00:00"
            })
            self.rowcount = 1

        elif "SELECT * FROM deletion_logs WHERE task_id" in query:
            for log in self.db.deletion_logs:
                if log["task_id"] == params[0]:
                    self._result.append((
                        log.get("id"), log["task_id"], log["doc_id"], log["kb_id"],
                        log["step"], log["item_type"], log["item_id"],
                        json.dumps(log["item_detail"]) if log["item_detail"] else None,
                        log["status"], log["message"], log["created_at"],
                        log.get("kb_name"), log.get("doc_name")
                    ))

        elif "SELECT * FROM deletion_logs WHERE doc_id" in query:
            for log in self.db.deletion_logs:
                if log["doc_id"] == params[0]:
                    self._result.append((
                        log.get("id"), log["task_id"], log["doc_id"], log["kb_id"],
                        log["step"], log["item_type"], log["item_id"],
                        json.dumps(log["item_detail"]) if log["item_detail"] else None,
                        log["status"], log["message"], log["created_at"],
                        log.get("kb_name"), log.get("doc_name")
                    ))

        elif "SELECT * FROM deletion_logs WHERE kb_id" in query:
            for log in self.db.deletion_logs:
                if log["kb_id"] == params[0]:
                    self._result.append((
                        log.get("id"), log["task_id"], log["doc_id"], log["kb_id"],
                        log["step"], log["item_type"], log["item_id"],
                        json.dumps(log["item_detail"]) if log["item_detail"] else None,
                        log["status"], log["message"], log["created_at"],
                        log.get("kb_name"), log.get("doc_name")
                    ))

        elif "SELECT DISTINCT ON (task_id)" in query or "FROM deletion_logs" in query:
            # 简化实现：返回所有删除日志的 task 信息
            seen = set()
            for log in reversed(self.db.deletion_logs):
                if log["task_id"] not in seen:
                    seen.add(log["task_id"])
                    self._result.append((
                        log["task_id"], log["doc_id"], log["kb_id"],
                        log["status"], log["message"],
                        log["created_at"], log["created_at"]
                    ))

        elif "DELETE FROM deletion_logs" in query:
            original_count = len(self.db.deletion_logs)
            self.db.deletion_logs = [l for l in self.db.deletion_logs if l["task_id"] != params[0]]
            self.rowcount = original_count - len(self.db.deletion_logs)
    
    def fetchone(self):
        return self._result[0] if self._result else None
    
    def fetchall(self):
        return self._result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

postgres_client = PostgresClient()
