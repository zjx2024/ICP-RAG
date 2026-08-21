"""数据访问层：基于现有同步 postgres_client（psycopg2）封装。

按文档 3.2 说明，存量 postgres_client 保持可用；新增 Repository 以同步方式
复用同一连接，业务代码通过 asyncio.to_thread 调用避免阻塞事件循环。
"""
