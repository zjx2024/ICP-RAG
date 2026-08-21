"""Agent 专属数据库操作层。

设计原则：
- 新建 4 张表：agent_models / agent_configs / agent_sessions / agent_messages
- 不修改任何现有表（llm_models 等）
- 通过 agent_models.model_id 只读引用现有 llm_models（不加 FK 约束，应用层校验）
- 复用现有 postgres_client 连接（同一个 PG 实例，不同表）
"""
import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from backend.storage.postgres_client import postgres_client

logger = logging.getLogger(__name__)


class AgentDB:
    """Agent 专属数据库操作（独立于现有 postgres_client 的业务方法）"""

    def __init__(self):
        self._initialized = False

    def init_tables(self):
        """建表（幂等）。复用现有 PG 连接，新建 4 张表，不动现有表。"""
        if self._initialized:
            return
        try:
            with postgres_client.conn.cursor() as cur:
                # 1. Agent 模型角色映射表（引用 llm_models，不加 FK 约束避免删模型影响 Agent）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_models (
                        id            SERIAL PRIMARY KEY,
                        agent_id      VARCHAR(36) NOT NULL,
                        role          VARCHAR(32) NOT NULL,
                        model_id      VARCHAR(36) NOT NULL,
                        is_default    BOOLEAN DEFAULT FALSE,
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (agent_id, role)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_models_agent ON agent_models(agent_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_models_role ON agent_models(role)")

                # 2. Agent 配置表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_configs (
                        agent_id          VARCHAR(36) PRIMARY KEY,
                        name              VARCHAR(100) NOT NULL,
                        description       TEXT,
                        system_prompt     TEXT NOT NULL DEFAULT '',
                        kb_id             VARCHAR(36),
                        enabled_tools     JSONB DEFAULT '["vector_search","kg_search","sql_query"]',
                        temperature       FLOAT DEFAULT 0.7,
                        top_p             FLOAT DEFAULT 0.9,
                        max_tokens        INTEGER DEFAULT 2048,
                        max_reflections   INTEGER DEFAULT 3,
                        retrieval_top_k   INTEGER DEFAULT 5,
                        summary_threshold INTEGER DEFAULT 12,
                        keep_recent_turns INTEGER DEFAULT 6,
                        streaming         BOOLEAN DEFAULT TRUE,
                        memory_enabled    BOOLEAN DEFAULT TRUE,
                        is_active         BOOLEAN DEFAULT TRUE,
                        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # 迁移：老表补 memory_enabled 列（幂等）
                try:
                    cur.execute("ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS memory_enabled BOOLEAN DEFAULT TRUE")
                    postgres_client.conn.commit()
                except Exception:
                    try:
                        postgres_client.conn.rollback()
                    except Exception:
                        pass

                # 3. 检索参数表（知识图谱检索/数据库查询参数，与 agent_configs 分离，避免修改现有表）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_retrieval_params (
                        agent_id   VARCHAR(36) PRIMARY KEY,
                        kg_top_k   INTEGER DEFAULT 5,
                        sql_limit  INTEGER DEFAULT 5,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (agent_id) REFERENCES agent_configs(agent_id) ON DELETE CASCADE
                    )
                """)

                # 3.5 知识库关联表（一个 Agent 可关联多个知识库，新建表不修改 agent_configs）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_kb_relations (
                        agent_id   VARCHAR(36) NOT NULL,
                        kb_id      VARCHAR(36) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (agent_id, kb_id),
                        FOREIGN KEY (agent_id) REFERENCES agent_configs(agent_id) ON DELETE CASCADE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_kb_relations_kb ON agent_kb_relations(kb_id)")

                # 4. 会话表（id 即 LangGraph thread_id）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        id            VARCHAR(36) PRIMARY KEY,
                        agent_id      VARCHAR(36) NOT NULL,
                        user_id       VARCHAR(64) DEFAULT 'default',
                        title         VARCHAR(64) DEFAULT '新会话',
                        pinned        BOOLEAN DEFAULT FALSE,
                        archived      BOOLEAN DEFAULT FALSE,
                        message_count INTEGER DEFAULT 0,
                        last_message  TEXT,
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (agent_id) REFERENCES agent_configs(agent_id) ON DELETE CASCADE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent ON agent_sessions(agent_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_user ON agent_sessions(user_id)")

                # 4. 消息表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_messages (
                        id              VARCHAR(36) PRIMARY KEY,
                        session_id      VARCHAR(36) NOT NULL,
                        role            VARCHAR(16) NOT NULL,
                        content         TEXT,
                        tool_calls      JSONB,
                        citations       JSONB,
                        thinking_events JSONB,
                        feedback        VARCHAR(8),
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id)")

                # 迁移：已有表加 thinking_events 列（幂等）
                try:
                    cur.execute("""
                        ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS thinking_events JSONB
                    """)
                    postgres_client.conn.commit()
                except Exception as _e:
                    try:
                        postgres_client.conn.rollback()
                    except Exception:
                        pass

                postgres_client.conn.commit()
                self._initialized = True
                logger.info("Agent tables initialized (agent_models/agent_configs/agent_sessions/agent_messages)")
        except Exception as e:
            logger.error(f"Agent tables init failed: {e}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass

    # ═════════════════════════════════════════════
    # Agent 配置 CRUD
    # ═════════════════════════════════════════════
    def create_agent(self, name: str, system_prompt: str = "", description: str = "",
                     kb_id: str = None, kb_ids: List[str] = None, enabled_tools: List[str] = None,
                     chat_model_id: str = None, embedding_model_id: str = None,
                     rerank_model_id: str = None,
                     temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048,
                     max_reflections: int = 3, retrieval_top_k: int = 5,
                     summary_threshold: int = 12, keep_recent_turns: int = 6,
                     streaming: bool = True, memory_enabled: bool = True,
                     kg_top_k: int = 5, sql_limit: int = 5) -> Dict:
        agent_id = str(uuid.uuid4())
        tools_json = json.dumps(enabled_tools or ["vector_search", "kg_search", "sql_query"], ensure_ascii=False)
        # 多知识库：kb_ids 优先（主库取第一个，兼容 agent_configs.kb_id 旧字段）
        if isinstance(kb_ids, str):
            kb_ids = [kb_ids]
        kb_ids = [k for k in (kb_ids or []) if k]
        main_kb = (kb_ids[0] if kb_ids else None) or kb_id
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_configs
                    (agent_id, name, description, system_prompt, kb_id, enabled_tools,
                     temperature, top_p, max_tokens, max_reflections, retrieval_top_k,
                     summary_threshold, keep_recent_turns, streaming, memory_enabled)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (agent_id, name, description, system_prompt, main_kb, tools_json,
                  temperature, top_p, max_tokens, max_reflections, retrieval_top_k,
                  summary_threshold, keep_recent_turns, streaming, memory_enabled))

            # 多知识库关联写入独立表
            for kb in kb_ids:
                cur.execute("""
                    INSERT INTO agent_kb_relations (agent_id, kb_id) VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (agent_id, kb))

            # 知识图谱/数据库检索参数写入独立表（避免修改 agent_configs 结构）
            cur.execute("""
                INSERT INTO agent_retrieval_params (agent_id, kg_top_k, sql_limit)
                VALUES (%s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE
                    SET kg_top_k = EXCLUDED.kg_top_k, sql_limit = EXCLUDED.sql_limit,
                        updated_at = CURRENT_TIMESTAMP
            """, (agent_id, kg_top_k, sql_limit))
            postgres_client.conn.commit()

        # 绑定 chat 模型到 plan/reflect/answer 三个角色
        self._bind_chat_model_for_agent(agent_id, chat_model_id)
        # 绑定 embedding 模型到 embedding 角色（vector_search 用）
        self._bind_embedding_model_for_agent(agent_id, embedding_model_id)
        # 绑定 rerank 模型到 rerank 角色（可选，未配置则不启用重排序）
        self._bind_rerank_model_for_agent(agent_id, rerank_model_id)
        return self.get_agent(agent_id)

    def _bind_chat_model_for_agent(self, agent_id: str, chat_model_id: str = None):
        """为智能体绑定 chat 模型到 plan/reflect/answer 三个角色。

        优先级：
        1. 用户在创建表单中选择的 chat_model_id（需校验 model_type='llm'）
        2. llm_models 表中 model_type='llm' AND is_default=TRUE 的模型
        3. llm_models 表中 model_type='llm' AND is_active=TRUE 的第一个
        绑定失败不影响智能体创建（仍可走回退逻辑）。
        """
        try:
            with postgres_client.conn.cursor() as cur:
                final_model_id = None

                # 1. 用户选择的模型（校验类型必须是 llm，防止误传 embedding 模型）
                if chat_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_id = %s AND model_type = 'llm' AND is_active = TRUE
                    """, (chat_model_id,))
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]
                    else:
                        logger.warning(f"用户指定的 chat_model_id={chat_model_id} 不存在或非 llm 类型，回退到默认")

                # 2. 默认 LLM 模型
                if not final_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_type = 'llm' AND is_default = TRUE AND is_active = TRUE
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]

                # 3. 任意活跃 LLM 模型
                if not final_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_type = 'llm' AND is_active = TRUE
                        ORDER BY created_at ASC LIMIT 1
                    """)
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]

                if not final_model_id:
                    logger.warning(f"未找到 model_type='llm' 的活跃模型 (agent={agent_id})")
                    return

                for role in ("plan", "reflect", "answer"):
                    cur.execute("""
                        INSERT INTO agent_models (agent_id, role, model_id, is_default)
                        VALUES (%s, %s, %s, TRUE)
                        ON CONFLICT (agent_id, role) DO UPDATE
                            SET model_id = EXCLUDED.model_id, is_default = TRUE
                    """, (agent_id, role, final_model_id))
                postgres_client.conn.commit()
                logger.info(f"绑定 chat 模型 {final_model_id} 到 plan/reflect/answer (agent={agent_id})")
        except Exception as e:
            logger.warning(f"绑定 chat 模型失败 (agent={agent_id}): {e}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass

    def _bind_embedding_model_for_agent(self, agent_id: str, embedding_model_id: str = None):
        """为智能体绑定 embedding 模型到 embedding 角色（vector_search 用）。

        优先级：
        1. 用户在创建表单中选择的 embedding_model_id（需校验 model_type='embedding'）
        2. llm_models 表中 model_type='embedding' AND is_default=TRUE 的模型
        3. llm_models 表中 model_type='embedding' AND is_active=TRUE 的第一个
        绑定失败不影响智能体创建。
        """
        try:
            with postgres_client.conn.cursor() as cur:
                final_model_id = None

                # 1. 用户选择的模型（校验类型必须是 embedding）
                if embedding_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_id = %s AND model_type = 'embedding' AND is_active = TRUE
                    """, (embedding_model_id,))
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]
                    else:
                        logger.warning(f"用户指定的 embedding_model_id={embedding_model_id} 不存在或非 embedding 类型，回退到默认")

                # 2. 默认 embedding 模型
                if not final_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_type = 'embedding' AND is_default = TRUE AND is_active = TRUE
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]

                # 3. 任意活跃 embedding 模型
                if not final_model_id:
                    cur.execute("""
                        SELECT model_id FROM llm_models
                        WHERE model_type = 'embedding' AND is_active = TRUE
                        ORDER BY created_at ASC LIMIT 1
                    """)
                    row = cur.fetchone()
                    if row:
                        final_model_id = row[0]

                if not final_model_id:
                    logger.warning(f"未找到 model_type='embedding' 的活跃模型 (agent={agent_id})")
                    return

                cur.execute("""
                    INSERT INTO agent_models (agent_id, role, model_id, is_default)
                    VALUES (%s, 'embedding', %s, TRUE)
                    ON CONFLICT (agent_id, role) DO UPDATE
                        SET model_id = EXCLUDED.model_id, is_default = TRUE
                """, (agent_id, final_model_id))
                postgres_client.conn.commit()
                logger.info(f"绑定 embedding 模型 {final_model_id} 到 embedding 角色 (agent={agent_id})")
        except Exception as e:
            logger.warning(f"绑定 embedding 模型失败 (agent={agent_id}): {e}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass

    def _bind_rerank_model_for_agent(self, agent_id: str, rerank_model_id: str = None):
        """为智能体绑定 rerank 模型到 rerank 角色（可选）。

        rerank 是可选项：仅当用户显式选择了 rerank 模型时绑定，
        未选择时不绑定、不回退默认（即不使用重排序）。
        需校验 model_type='rerank'，防止误传其他类型模型。
        """
        if not rerank_model_id:
            return
        try:
            with postgres_client.conn.cursor() as cur:
                cur.execute("""
                    SELECT model_id FROM llm_models
                    WHERE model_id = %s AND model_type = 'rerank' AND is_active = TRUE
                """, (rerank_model_id,))
                row = cur.fetchone()
                if not row:
                    logger.warning(f"用户指定的 rerank_model_id={rerank_model_id} 不存在或非 rerank 类型，跳过绑定")
                    return

                cur.execute("""
                    INSERT INTO agent_models (agent_id, role, model_id, is_default)
                    VALUES (%s, 'rerank', %s, TRUE)
                    ON CONFLICT (agent_id, role) DO UPDATE
                        SET model_id = EXCLUDED.model_id, is_default = TRUE
                """, (agent_id, rerank_model_id))
                postgres_client.conn.commit()
                logger.info(f"绑定 rerank 模型 {rerank_model_id} 到 rerank 角色 (agent={agent_id})")
        except Exception as e:
            logger.warning(f"绑定 rerank 模型失败 (agent={agent_id}): {e}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_configs WHERE agent_id = %s", (agent_id,))
            row = cur.fetchone()
            if not row:
                return None
            agent = self._merge_retrieval_params(cur, self._row_to_agent(cur, row))
            self._merge_kb_ids(cur, agent)
            return agent

    def _merge_kb_ids(self, cur, agent: Dict) -> Dict:
        """合并关联知识库：关联表优先，其次兼容 agent_configs.kb_id 旧字段"""
        cur.execute("SELECT kb_id FROM agent_kb_relations WHERE agent_id = %s ORDER BY created_at",
                    (agent.get("agent_id"),))
        rows = cur.fetchall()
        kb_ids = [r[0] for r in rows]
        if not kb_ids and agent.get("kb_id"):
            kb_ids = [agent["kb_id"]]
        agent["kb_ids"] = kb_ids
        return agent

    def list_agents(self, active_only: bool = False) -> List[Dict]:
        with postgres_client.conn.cursor() as cur:
            if active_only:
                cur.execute("SELECT * FROM agent_configs WHERE is_active = TRUE ORDER BY created_at DESC")
            else:
                cur.execute("SELECT * FROM agent_configs ORDER BY created_at DESC")
            rows = cur.fetchall()
            agents = [self._row_to_agent(cur, r) for r in rows]
            if not agents:
                return agents
            # 一次性拉取所有检索参数，合并到各 agent
            cur.execute("SELECT agent_id, kg_top_k, sql_limit FROM agent_retrieval_params")
            params_map = {r[0]: r for r in cur.fetchall()}
            for a in agents:
                p = params_map.get(a.get("agent_id"))
                if p:
                    a["kg_top_k"] = p[1]
                    a["sql_limit"] = p[2]
            # 一次性拉取所有知识库关联，合并到各 agent
            cur.execute("SELECT agent_id, kb_id FROM agent_kb_relations ORDER BY created_at")
            rel_map: Dict[str, List[str]] = {}
            for r in cur.fetchall():
                rel_map.setdefault(r[0], []).append(r[1])
            for a in agents:
                kbs = rel_map.get(a.get("agent_id"))
                if not kbs and a.get("kb_id"):
                    kbs = [a["kb_id"]]
                a["kb_ids"] = kbs or []
            return agents

    @staticmethod
    def _merge_retrieval_params(cur, agent: Dict) -> Dict:
        """合并 agent_retrieval_params 中的知识图谱/数据库检索参数到 agent dict"""
        cur.execute("SELECT kg_top_k, sql_limit FROM agent_retrieval_params WHERE agent_id = %s", (agent.get("agent_id"),))
        row = cur.fetchone()
        agent["kg_top_k"] = row[0] if row else 5
        agent["sql_limit"] = row[1] if row else 5
        return agent

    def update_agent(self, agent_id: str, **kwargs) -> Optional[Dict]:
        if not kwargs:
            return self.get_agent(agent_id)
        allowed = {"name", "description", "system_prompt", "kb_id", "enabled_tools",
                   "temperature", "top_p", "max_tokens", "max_reflections",
                   "retrieval_top_k", "summary_threshold", "keep_recent_turns",
                   "streaming", "memory_enabled", "is_active"}
        # 多知识库关联（kb_ids）独立处理：写入 agent_kb_relations 表
        kb_ids = kwargs.pop("kb_ids", None)
        if isinstance(kb_ids, str):
            kb_ids = [kb_ids]
        has_kb_ids = kb_ids is not None
        # 检索参数（kg_top_k/sql_limit）写入独立表
        retrieval_params = {}
        for k in ("kg_top_k", "sql_limit"):
            if k in kwargs:
                retrieval_params[k] = kwargs.pop(k)
        sets, vals = [], []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "enabled_tools" and isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            sets.append(f"{k} = %s")
            vals.append(v)
        with postgres_client.conn.cursor() as cur:
            # 更新知识库关联：先删后插（幂等）
            if has_kb_ids:
                kb_ids = [k for k in (kb_ids or []) if k]
                cur.execute("DELETE FROM agent_kb_relations WHERE agent_id = %s", (agent_id,))
                for kb in kb_ids:
                    cur.execute("""
                        INSERT INTO agent_kb_relations (agent_id, kb_id) VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, (agent_id, kb))
                # 主库同步为第一个（兼容 agent_configs.kb_id 旧字段/旧逻辑）
                main_kb = kb_ids[0] if kb_ids else None
                cur.execute("UPDATE agent_configs SET kb_id = %s, updated_at = CURRENT_TIMESTAMP WHERE agent_id = %s",
                            (main_kb, agent_id))
            if sets:
                sets.append("updated_at = CURRENT_TIMESTAMP")
                vals.append(agent_id)
                cur.execute(f"UPDATE agent_configs SET {', '.join(sets)} WHERE agent_id = %s", vals)
            if retrieval_params:
                cur.execute("""
                    INSERT INTO agent_retrieval_params (agent_id, kg_top_k, sql_limit)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (agent_id) DO UPDATE
                        SET kg_top_k = EXCLUDED.kg_top_k, sql_limit = EXCLUDED.sql_limit,
                            updated_at = CURRENT_TIMESTAMP
                """, (agent_id,
                      retrieval_params.get("kg_top_k", 5),
                      retrieval_params.get("sql_limit", 5)))
            postgres_client.conn.commit()
        return self.get_agent(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM agent_configs WHERE agent_id = %s", (agent_id,))
            postgres_client.conn.commit()
            return cur.rowcount > 0

    def _row_to_agent(self, cur, row) -> Dict:
        d = {desc[0]: v for desc, v in zip(cur.description, row)}
        if isinstance(d.get("enabled_tools"), str):
            try:
                d["enabled_tools"] = json.loads(d["enabled_tools"])
            except Exception:
                d["enabled_tools"] = []
        return d

    # ═════════════════════════════════════════════
    # Agent 模型角色 CRUD（引用 llm_models，不修改原表）
    # ═════════════════════════════════════════════
    def bind_model(self, agent_id: str, role: str, model_id: str, is_default: bool = False) -> Dict:
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_models (agent_id, role, model_id, is_default)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (agent_id, role) DO UPDATE
                    SET model_id = EXCLUDED.model_id, is_default = EXCLUDED.is_default
            """, (agent_id, role, model_id, is_default))
            postgres_client.conn.commit()
        return {"agent_id": agent_id, "role": role, "model_id": model_id, "is_default": is_default}

    def list_agent_models(self, agent_id: str) -> List[Dict]:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT id, agent_id, role, model_id, is_default, created_at FROM agent_models WHERE agent_id = %s", (agent_id,))
            return [self._row_to_model(cur, r) for r in cur.fetchall()]

    def get_model_id_by_role(self, agent_id: str, role: str) -> Optional[str]:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT model_id FROM agent_models WHERE agent_id = %s AND role = %s", (agent_id, role))
            row = cur.fetchone()
            return row[0] if row else None

    def unbind_model(self, agent_id: str, role: str) -> bool:
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM agent_models WHERE agent_id = %s AND role = %s", (agent_id, role))
            postgres_client.conn.commit()
            return cur.rowcount > 0

    def _row_to_model(self, cur, row) -> Dict:
        return {desc[0]: v for desc, v in zip(cur.description, row)}

    # ═════════════════════════════════════════════
    # 会话 CRUD
    # ═════════════════════════════════════════════
    def create_session(self, agent_id: str, user_id: str = "default", title: str = "新会话") -> Dict:
        session_id = str(uuid.uuid4())
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_sessions (id, agent_id, user_id, title)
                VALUES (%s, %s, %s, %s)
            """, (session_id, agent_id, user_id, title))
            postgres_client.conn.commit()
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict]:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_sessions WHERE id = %s", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {desc[0]: v for desc, v in zip(cur.description, row)}

    def list_sessions(self, agent_id: str = None, user_id: str = None,
                      archived: bool = False, search: str = None, limit: int = 100) -> List[Dict]:
        sql = "SELECT * FROM agent_sessions WHERE archived = %s"
        vals = [archived]
        if agent_id:
            sql += " AND agent_id = %s"
            vals.append(agent_id)
        if user_id:
            sql += " AND user_id = %s"
            vals.append(user_id)
        if search:
            sql += " AND title ILIKE %s"
            vals.append(f"%{search}%")
        sql += " ORDER BY pinned DESC, updated_at DESC LIMIT %s"
        vals.append(limit)
        with postgres_client.conn.cursor() as cur:
            cur.execute(sql, vals)
            return [{desc[0]: v for desc, v in zip(cur.description, r)} for r in cur.fetchall()]

    def update_session(self, session_id: str, **kwargs) -> Optional[Dict]:
        allowed = {"title", "pinned", "archived"}
        sets, vals = [], []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            sets.append(f"{k} = %s")
            vals.append(v)
        if not sets:
            return self.get_session(session_id)
        sets.append("updated_at = CURRENT_TIMESTAMP")
        vals.append(session_id)
        with postgres_client.conn.cursor() as cur:
            cur.execute(f"UPDATE agent_sessions SET {', '.join(sets)} WHERE id = %s", vals)
            postgres_client.conn.commit()
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        with postgres_client.conn.cursor() as cur:
            cur.execute("DELETE FROM agent_sessions WHERE id = %s", (session_id,))
            postgres_client.conn.commit()
            return cur.rowcount > 0

    def touch_session(self, session_id: str, last_message: str = None, count_delta: int = 1):
        """更新会话的最后消息和消息计数"""
        try:
            with postgres_client.conn.cursor() as cur:
                if last_message:
                    cur.execute("""
                        UPDATE agent_sessions
                        SET message_count = message_count + %s,
                            last_message = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (count_delta, last_message[:500], session_id))
                else:
                    cur.execute("""
                        UPDATE agent_sessions
                        SET message_count = message_count + %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (count_delta, session_id))
                postgres_client.conn.commit()
        except Exception as e:
            logger.error(f"touch_session failed: {e}")
            try:
                postgres_client.conn.rollback()
            except Exception:
                pass

    # ═════════════════════════════════════════════
    # 消息 CRUD
    # ═════════════════════════════════════════════
    def add_message(self, session_id: str, role: str, content: str = "",
                    tool_calls: List = None, citations: List = None,
                    thinking_events: List = None) -> Dict:
        msg_id = str(uuid.uuid4())
        tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        ci_json = json.dumps(citations, ensure_ascii=False) if citations else None
        te_json = json.dumps(thinking_events, ensure_ascii=False) if thinking_events else None
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_messages (id, session_id, role, content, tool_calls, citations, thinking_events)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (msg_id, session_id, role, content, tc_json, ci_json, te_json))
            postgres_client.conn.commit()
        # 更新会话计数
        preview = content[:200] if content else ""
        self.touch_session(session_id, preview, 1)
        return {"id": msg_id, "session_id": session_id, "role": role, "content": content,
                "thinking_events": thinking_events}

    def list_messages(self, session_id: str, limit: int = 100) -> List[Dict]:
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                SELECT id, session_id, role, content, tool_calls, citations, thinking_events, feedback, created_at
                FROM agent_messages WHERE session_id = %s ORDER BY created_at ASC LIMIT %s
            """, (session_id, limit))
            rows = cur.fetchall()
            result = []
            for r in rows:
                d = {desc[0]: v for desc, v in zip(cur.description, r)}
                for k in ("tool_calls", "citations", "thinking_events"):
                    if isinstance(d.get(k), str):
                        try:
                            d[k] = json.loads(d[k])
                        except Exception:
                            d[k] = None
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else str(d["created_at"])
                result.append(d)
            return result

    def update_feedback(self, message_id: str, feedback: str) -> bool:
        with postgres_client.conn.cursor() as cur:
            cur.execute("UPDATE agent_messages SET feedback = %s WHERE id = %s", (feedback, message_id))
            postgres_client.conn.commit()
            return cur.rowcount > 0

    def list_messages_for_state(self, session_id: str, limit: int = 20) -> List[Dict]:
        """取消息用于 Agent 状态上下文（简化格式）"""
        msgs = self.list_messages(session_id, limit)
        return [{"role": m["role"], "content": m.get("content", "")} for m in msgs if m.get("content")]


# 全局单例
agent_db = AgentDB()
