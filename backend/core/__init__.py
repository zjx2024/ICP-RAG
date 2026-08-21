"""核心层：re-export app.core 现有实现 + 新增通用工具。

- app.core 中的业务实现（config/task_manager/llm_service/model_config/embedding/rerank/retrieval/parser）
  继续作为单一实现来源，backend 仅做引用，避免复制逻辑导致行为漂移。
- exceptions/response/db 为新增通用骨架，供未来新代码使用，不影响存量。
"""
from backend.config.config import config  # noqa: F401
from backend.core.task_manager import TaskManager, task_manager  # noqa: F401
from backend.core.llm_service import llm_service  # noqa: F401
from backend.core.model_config import model_config_service  # noqa: F401
from backend.core.embedding import embedding_service  # noqa: F401
from backend.core.rerank import rerank_service  # noqa: F401
from backend.core.retrieval import knowledge_retriever, RetrievalMode  # noqa: F401
from backend.core.document_parser import DocumentParser  # noqa: F401
from backend.core.unified_parser import (  # noqa: F401
    TxtParser,
    MarkdownParser,
    MarkdownElementExtractor,
    ExcelParser,
    PptParser,
    HtmlParser,
    JsonParser,
    PdfParser,
)

from backend.core.exceptions import BusinessError  # noqa: F401
from backend.core.response import ok, fail  # noqa: F401
