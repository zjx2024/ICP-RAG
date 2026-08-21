"""服务层：业务逻辑归属地（已从路由文件物理迁入）。

- model_service：LLM / Embedding 模型解析与校验
- document_service：文档后台解析任务（含 llm_semaphore）
- kg_service：知识图谱生成后台任务
- deletion_service：删除级联编排（委托 app 迁移前的既有实现，逻辑同源）
"""
from backend.services.model_service import (  # noqa: F401
    resolve_llm_model_config,
    require_embedding_model_configured,
)
from backend.services.document_service import (  # noqa: F401
    llm_semaphore,
    convert_fault_tree_to_hugegraph,
    parse_document_background,
)
from backend.services.kg_service import generate_kg_background  # noqa: F401
from backend.services.deletion_service import (  # noqa: F401
    DeletionService,
    deletion_service,
)
