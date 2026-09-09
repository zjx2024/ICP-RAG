"""服务层：业务逻辑归属地（已从路由文件物理迁入）。

- model_service：LLM / Embedding 模型解析与校验
- document_service：ICP 七类文档后台解析任务
- kg_service：ICP 七类文档图谱投影后台任务
"""
from backend.services.model_service import (  # noqa: F401
    resolve_llm_model_config,
    require_embedding_model_configured,
)
from backend.services.document_service import (  # noqa: F401
    parse_icp_document_background,
)
from backend.services.kg_service import (  # noqa: F401
    generate_icp_graph_background,
    generate_icp_kg_background,
)
