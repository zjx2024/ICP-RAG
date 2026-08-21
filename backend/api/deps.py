"""依赖注入。

当前数据访问仍以同步 postgres_client（psycopg2）为准；
get_db 提供与既有代码一致的连接上下文。未来引入异步 SQLAlchemy session 时，
再在此扩展 get_async_db，存量路由不受影响。
"""
from typing import Iterator

from backend.storage.postgres_client import postgres_client


def get_db() -> Iterator:
    """返回现有 postgres_client（单一实现来源）。"""
    yield postgres_client
