"""Legacy ``vector_search`` Agent node backed by the unified retrieval pipeline."""
import logging
import time
from typing import Dict

from backend.core.retrieval import knowledge_retriever

logger = logging.getLogger(__name__)


async def vector_search_node(state) -> Dict:
    """Run keyword+dense recall and rerank without pre-rerank truncation."""

    query = state.get("rewritten_query") or state.get("query", "")
    kb_id = state.get("kb_id", "")
    kb_ids = state.get("kb_ids") or ([kb_id] if kb_id else [])
    agent_id = state.get("agent_id", "")
    rerank_model_id = state.get("rerank_model_id", "")
    embedding_model_id = state.get("embedding_model_id", "")
    top_k = max(1, int(state.get("retrieval_top_k", 5) or 5))

    if not embedding_model_id and agent_id:
        try:
            from backend.agent.db import agent_db

            embedding_model_id = agent_db.get_model_id_by_role(agent_id, "embedding") or ""
        except Exception as exc:
            logger.warning("读取 Agent embedding 绑定失败: %s", exc)

    started = time.time()
    try:
        outcome = knowledge_retriever.search_hybrid_with_info(
            kb_id=kb_id,
            kb_ids=kb_ids,
            query=query,
            top_k=top_k,
            keyword_weight=0.4,
            vector_weight=0.6,
            embedding_model_id=embedding_model_id or None,
            agent_id=agent_id,
            rerank_model_id=rerank_model_id,
        )
        results = outcome.results
        retrieval_info = outcome.info()
    except Exception as exc:
        logger.exception("vector_search 统一检索失败")
        results = []
        retrieval_info = {
            "recall_count": 0,
            "deduplicated_count": 0,
            "reranked": False,
            "rerank_model": "",
            "rerank_source": "",
            "rerank_error": None,
            "fallback": f"统一检索失败: {exc}",
            "recall_fallbacks": [],
        }

    results_preview = []
    for result in results[:30]:
        item = {
            "title": (result.get("title") or result.get("section") or result.get("doc_name") or "").strip()[:80],
            "doc_name": (result.get("doc_name") or "").strip()[:60],
            "snippet": (result.get("content") or "").strip()[:200],
            "score": round(result.get("score", 0), 3)
            if isinstance(result.get("score"), (int, float))
            else None,
        }
        if result.get("rerank_score") is not None:
            item["rerank_score"] = round(result["rerank_score"], 3)
        results_preview.append(item)

    tool_call = {
        "tool": "vector_search",
        "query": query,
        "count": len(results),
        "duration": round(time.time() - started, 2),
        "params": {
            "kb_id": kb_id,
            "kb_ids": kb_ids,
            "top_k": top_k,
            "embedding_model_id": embedding_model_id or None,
        },
        "results_preview": results_preview,
        **retrieval_info,
    }
    if retrieval_info["reranked"]:
        logger.info(
            "[rerank] agent=%s model=%s source=%s candidates=%s returned=%s",
            agent_id,
            retrieval_info["rerank_model"],
            retrieval_info["rerank_source"],
            retrieval_info["deduplicated_count"],
            len(results),
        )
    elif retrieval_info.get("fallback"):
        logger.warning("[rerank] agent=%s fallback=%s", agent_id, retrieval_info["fallback"])

    return {"vector_results": results, "tool_calls": [tool_call]}
