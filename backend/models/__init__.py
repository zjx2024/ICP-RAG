"""SQLAlchemy ORM 模型：映射现有表（只映射、不迁移、不重建）。

按重构红线，数据库结构一律不动；此处仅提供类型化访问入口，
供 services/repositories 层新代码使用。存量 psycopg2 直连路径不受影响。
"""
from backend.models.knowledge_base import KnowledgeBase  # noqa: F401
from backend.models.document import Document, DocumentChunk  # noqa: F401
from backend.models.llm_model import LLMModel  # noqa: F401
from backend.models.base import Base  # noqa: F401
