"""删除服务：复用 app 现有删除实现（不复制逻辑）。

删除级联（8 步 DELETE_STEP_ORDER 断点续删、社区报告处理、删除日志留痕）
属于高复杂度存量机制，按重构红线原样保留在 app.api.routers.documents /
app.api.routers._deletion_helpers 中，此处仅做编排委托，确保行为完全一致。
"""
from typing import Any, Dict


class DeletionService:
    """删除相关业务编排入口。"""

    async def delete_document_graph_data(self, kb_id: str, doc_id: str) -> Dict[str, Any]:
        """删除文档关联的图谱数据（实体/关系/社区报告），保留文档本身。"""
        from backend.api.routers.documents import delete_document_graph_data as _impl

        return await _impl(kb_id, doc_id)

    async def delete_document(self, kb_id: str, doc_id: str) -> Dict[str, Any]:
        """完整删除文档（8 步级联，可断点续删）。"""
        from backend.api.routers.documents import delete_document as _impl

        return await _impl(kb_id, doc_id)

    async def delete_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """删除整个知识库。"""
        from backend.api.routers.documents import delete_knowledge_base as _impl

        return await _impl(kb_id)


deletion_service = DeletionService()
