"""vector_search 工具节点。

设计原则：薄包装，只读复用现有 retrieval.search_es_hybrid，不修改原文件。
集成 rerank：若 Agent 配置了 rerank 模型，则对召回结果做重排序后再返回。
"""
import time
import logging
from typing import Dict

from backend.core.retrieval import knowledge_retriever  # 只读引用，不修改
from backend.core.rerank import rerank_service  # 重排序服务（可选启用）
from backend.agent.model_role_service import agent_model_role_service

logger = logging.getLogger(__name__)


def _get_rerank_model(state) -> Dict:
    """获取 rerank 模型：优先 state 中显式指定的 rerank_model_id，其次 agent_models 绑定。

    未配置返回 None（不使用重排序）。
    """
    agent_id = state.get("agent_id", "")
    rerank_model_id = state.get("rerank_model_id", "")

    if rerank_model_id:
        try:
            from backend.storage.postgres_client import postgres_client
            model = postgres_client.get_model(rerank_model_id)
            if model and model.get("is_active", True) and model.get("model_type") == "rerank":
                return model
            logger.warning(f"指定 rerank_model_id={rerank_model_id} 不存在、未激活或非 rerank 类型")
        except Exception as e:
            logger.warning(f"获取指定 rerank 模型失败: {e}")

    if agent_id:
        try:
            return agent_model_role_service.get_rerank_model(agent_id)
        except Exception as e:
            logger.warning(f"获取 Agent 绑定的 rerank 模型失败 (agent={agent_id}): {e}")
    return None


async def vector_search_node(state) -> Dict:
    """ES 向量+关键词混合检索（复用现有 retrieval）"""
    query = state.get("rewritten_query") or state.get("query", "")
    kb_id = state.get("kb_id", "")
    # 多库关联：kb_ids 优先（可能多个知识库），无则回退 kb_id 单库
    kb_ids = state.get("kb_ids") or ([kb_id] if kb_id else [])
    agent_id = state.get("agent_id", "")
    embedding_model_id = state.get("embedding_model_id", "")
    # 向量检索返回条数：读取 agent 配置的 retrieval_top_k
    top_k = int(state.get("retrieval_top_k", 5) or 5)

    # 如果前端没传 embedding_model_id，尝试从 agent_models 表取 agent 绑定的 embedding 模型
    if not embedding_model_id and agent_id:
        try:
            from backend.agent.db import agent_db
            bound_id = agent_db.get_model_id_by_role(agent_id, "embedding")
            if bound_id:
                embedding_model_id = bound_id
        except Exception:
            pass

    # 获取 rerank 模型（可选）：配置了才启用重排序
    rerank_model = _get_rerank_model(state)
    # 多召回候选：同一文档可能被重复索引产生相同 chunk，需先召回更多再去重，确保去重后仍够 top_k 条
    recall_k = top_k * 3

    start = time.time()
    try:
        # 只读调用现有 retrieval 方法（不改原文件）
        results = knowledge_retriever.search_es_hybrid(
            kb_id=kb_id,
            kb_ids=kb_ids,
            query=query,
            top_k=recall_k,
            keyword_weight=0.4,
            vector_weight=0.6,
            embedding_model_id=embedding_model_id or None,
        )
    except Exception as e:
        logger.error(f"vector_search 调用失败: {e}")
        results = []

    # 去除重复内容：同一文档可能被重复索引（多个 doc_id 的相同 chunk），避免检索结果大量重复
    unique_results = []
    seen_contents = set()
    for r in results:
        c = (r.get("content") or "")[:120]
        if c in seen_contents:
            continue
        seen_contents.add(c)
        unique_results.append(r)
    # 去重后仍按配置的 top_k 截断（保证返回条数与 retrieval_top_k 一致）
    results = unique_results[:top_k]

    # 重排序：用 rerank 模型对召回结果重新打分排序，取前 top_k
    reranked = False
    rerank_error = None
    if rerank_model and results:
        texts = [r.get("content") or r.get("title") or "" for r in results]
        try:
            scored = rerank_service.rerank(query, texts, rerank_model, top_n=top_k)
            if scored:
                reordered = []
                for idx, score in scored:
                    item = results[idx]
                    # 保留原始 ES 分数，重排分数单独记录
                    item["original_score"] = item.get("score")
                    item["score"] = round(score, 4)
                    item["rerank_score"] = round(score, 4)
                    reordered.append(item)
                results = reordered
                reranked = True
            else:
                rerank_error = "rerank 返回空结果，保持原排序"
        except Exception as e:
            rerank_error = f"rerank 调用失败: {e}"
            logger.error(rerank_error)

    duration = round(time.time() - start, 2)
    # 构建结果预览：全部去重后的结果都返回给前端，由前端在滚动框中展示
    results_preview = []
    for r in results[:30]:
        item = {
            "title": (r.get("title") or r.get("section") or r.get("doc_name") or "").strip()[:80],
            "doc_name": (r.get("doc_name") or "").strip()[:60],
            "snippet": (r.get("content") or "").strip()[:200],
            "score": round(r.get("score", 0), 3) if isinstance(r.get("score"), (int, float)) else None,
        }
        if reranked and r.get("rerank_score") is not None:
            item["rerank_score"] = round(r["rerank_score"], 3)
        results_preview.append(item)
    tool_call = {
        "tool": "vector_search",
        "query": query,
        "count": len(results),
        "duration": duration,
        "params": {"kb_id": kb_id, "kb_ids": kb_ids, "top_k": top_k, "embedding_model_id": embedding_model_id or None},
        "results_preview": results_preview,
    }
    # 重排序信息始终随 tool_call 返回（是否配置/是否生效由前端思考面板展示）
    tool_call["reranked"] = reranked
    tool_call["rerank_model"] = (rerank_model.get("name") or rerank_model.get("model_name") or "") if rerank_model else ""
    if rerank_error:
        tool_call["rerank_error"] = rerank_error
    # 打印重排序使用情况，便于排查"是否真的用了 rerank"
    if rerank_model:
        if reranked:
            scores = [round(r.get("rerank_score", 0), 4) for r in results]
            logger.info(
                f"[rerank] agent={agent_id} ✓ 已使用重排序: model={tool_call['rerank_model']}, "
                f"重排后 top{len(results)} 分数={scores}"
            )
        else:
            logger.warning(
                f"[rerank] agent={agent_id} ✗ 配置了重排序模型({tool_call['rerank_model']}) 但未生效: "
                f"{rerank_error or '无候选结果'}"
            )
    else:
        logger.info(f"[rerank] agent={agent_id} 未配置 rerank 模型，向量检索不重排序")
    return {
        "vector_results": results,
        "tool_calls": [tool_call],
    }
