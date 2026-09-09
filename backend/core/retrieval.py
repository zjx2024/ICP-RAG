"""统一混合检索入口（backend 无关）。

通用知识库时代的 KG/社区报告/RAG 融合等模式已随旧链路退役；
ICP 七类诊断链路只需要 pipeline 驱动的关键词+向量混合召回。
"""
import logging
from typing import Dict, List, Optional, Any

from backend.core.retrieval_pipeline import RetrievalOutcome, build_retrieval_pipeline

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = build_retrieval_pipeline()
        return self._pipeline

    @staticmethod
    def _filters(kb_id: str = "", kb_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        values = [value for value in (kb_ids or []) if value]
        if values:
            return {"kb_id": values}
        return {"kb_id": kb_id} if kb_id else None

    def search_hybrid_with_info(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        embedding_model_id: Optional[str] = None,
        kb_ids: Optional[List[str]] = None,
        agent_id: str = "",
        rerank_model_id: str = "",
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalOutcome:
        """Unified backend-neutral retrieval including full-candidate rerank."""
        return self.pipeline.retrieve(
            query=query,
            top_k=top_k,
            filters=filters if filters is not None else self._filters(kb_id, kb_ids),
            embedding_model_id=embedding_model_id,
            agent_id=agent_id,
            rerank_model_id=rerank_model_id,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )

    def search_hybrid(self, kb_id: str, query: str, top_k: int = 10, **kwargs) -> List[Dict]:
        return self.search_hybrid_with_info(kb_id, query, top_k, **kwargs).results


knowledge_retriever = KnowledgeRetriever()
