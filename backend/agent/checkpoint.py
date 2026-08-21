"""LangGraph Checkpoint 与 Store 初始化（复用现有 PG 连接）。

设计原则：
- 复用现有 PostgreSQL（同 config 中的连接参数）
- AsyncPostgresSaver / AsyncPostgresStore 自动建表（checkpoints / store 等），与现有表无冲突
- 失败时降级为 MemorySaver，不影响 Agent 基本运行

注意（langgraph-checkpoint-postgres 3.x API）：
- 对话链路使用 graph.astream（异步），必须用 Async 版本（sync PostgresSaver 不实现 aget_tuple）
- from_conn_string() 返回的是异步上下文管理器（AsyncIterator[实例]），不是实例本身
- 必须用 __aenter__() 取出实例并保持上下文存活，否则连接池被关闭、实例不可用
"""
import logging

from backend.config.config import config

logger = logging.getLogger(__name__)

PG_URL = (
    f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
    f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
)

# 全局实例（懒加载）
_checkpointer = None
_store = None
# 上下文管理器：必须保持存活，否则连接池被关闭导致实例不可用
_checkpointer_ctx = None
_store_ctx = None


async def get_checkpointer():
    """获取 LangGraph Checkpointer（复用现有 PG，异步版本）"""
    global _checkpointer, _checkpointer_ctx
    if _checkpointer is not None:
        return _checkpointer
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        # 3.x：from_conn_string 返回异步上下文管理器，需 __aenter__() 取实例并保持存活
        _checkpointer_ctx = AsyncPostgresSaver.from_conn_string(PG_URL)
        _checkpointer = await _checkpointer_ctx.__aenter__()
        # setup() 创建必要的表（checkpoints/checkpoint_writes/checkpoint_blobs）
        await _checkpointer.setup()
        logger.info("AsyncPostgresSaver 初始化成功（复用现有 PG）")
        return _checkpointer
    except Exception as e:
        logger.warning(f"AsyncPostgresSaver 初始化失败，降级为 MemorySaver: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        return _checkpointer


async def get_store():
    """获取 LangGraph Store（复用现有 PG，异步版本，用于长期记忆）"""
    global _store, _store_ctx
    if _store is not None:
        return _store
    try:
        from langgraph.store.postgres.aio import AsyncPostgresStore
        # 3.x：from_conn_string 返回异步上下文管理器，需 __aenter__() 取实例并保持存活
        _store_ctx = AsyncPostgresStore.from_conn_string(PG_URL)
        _store = await _store_ctx.__aenter__()
        await _store.setup()
        logger.info("AsyncPostgresStore 初始化成功（复用现有 PG）")
        return _store
    except Exception as e:
        logger.warning(f"AsyncPostgresStore 初始化失败，长期记忆将不可用: {e}")
        return None


def get_pg_url() -> str:
    return PG_URL
