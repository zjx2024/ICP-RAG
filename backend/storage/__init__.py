# 同步基础设施客户端（保留自 app/storage，作为单一实现来源）
# 重构原则：不复制逻辑，仅 re-export，确保"功能不受影响"
from backend.storage.es_client import es_client  # noqa: F401
from backend.storage.hugegraph_client import hugegraph_client  # noqa: F401
from backend.storage.minio_client import minio_client  # noqa: F401
from backend.storage.postgres_client import postgres_client  # noqa: F401
from backend.storage.deletion_logger import DeletionLogger  # noqa: F401
