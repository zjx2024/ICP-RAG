"""Backend-neutral retrieval and rerank pipeline.

The search backend is deliberately injected behind ``SearchStore``.  This keeps
the ordering rules (recall, fusion, deduplication and rerank) independent from
OpenSearch transport details and makes the failure paths deterministic to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievalOutcome:
    """Results plus auditable information about recall and rerank fallback."""

    results: List[Dict[str, Any]]
    recall_count: int = 0
    deduplicated_count: int = 0
    reranked: bool = False
    rerank_model: str = ""
    rerank_source: str = ""
    rerank_error: Optional[str] = None
    fallback: Optional[str] = None
    recall_fallbacks: List[str] = field(default_factory=list)

    def info(self) -> Dict[str, Any]:
        return {
            "recall_count": self.recall_count,
            "deduplicated_count": self.deduplicated_count,
            "reranked": self.reranked,
            "rerank_model": self.rerank_model,
            "rerank_source": self.rerank_source,
            "rerank_error": self.rerank_error,
            "fallback": self.fallback,
            "recall_fallbacks": list(self.recall_fallbacks),
        }


def _hit_parts(hit: Any) -> Tuple[str, float, Dict[str, Any]]:
    """Accept the public SearchHit object and lightweight test doubles."""

    if isinstance(hit, dict):
        document = dict(hit.get("document") or hit)
        hit_id = str(hit.get("id") or document.get("chunk_id") or document.get("id") or "")
        raw_score = hit.get("score", document.get("score", 0.0))
    else:
        document = dict(getattr(hit, "document", {}) or {})
        hit_id = str(getattr(hit, "id", "") or document.get("chunk_id") or document.get("id") or "")
        raw_score = getattr(hit, "score", document.get("score", 0.0))
    try:
        score = float(raw_score or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return hit_id, score, document


def _content_key(item: Dict[str, Any]) -> Tuple[str, str]:
    """Deduplicate exact normalized content, retaining distinct empty chunks."""

    projection_role = str(item.get("projection_role_kwd") or "")
    diagram_id = str(item.get("diagram_id_kwd") or "")
    if projection_role == "diagram_locator" and diagram_id:
        return "diagram", diagram_id
    business_id = str(item.get("business_id_kwd") or "")
    doc_id = str(item.get("doc_id") or "")
    doc_type = str(item.get("doc_type_kwd") or item.get("kb_id") or "")
    if projection_role == "diagram_locator" and doc_id and business_id:
        return "diagram_business", f"{doc_type}\x1f{doc_id}\x1f{business_id}"
    content = " ".join(str(item.get("content") or "").split())
    if content:
        return "content", content
    identity = item.get("chunk_id") or item.get("id") or item.get("doc_id")
    return "id", str(identity or id(item))


class RetrievalPipeline:
    """Run dense + keyword recall, RRF fusion, dedupe and optional rerank."""

    def __init__(
        self,
        search_backend: Any,
        embed: Callable[..., Sequence[float]],
        rerank: Callable[..., Optional[List[Tuple[int, float]]]],
        resolve_rerank_model: Callable[[str, str], Tuple[Optional[Dict[str, Any]], str]],
        *,
        rrf_k: int = 60,
        recall_multiplier: int = 4,
    ):
        self.search_backend = search_backend
        self.embed = embed
        self.rerank = rerank
        self.resolve_rerank_model = resolve_rerank_model
        self.rrf_k = rrf_k
        self.recall_multiplier = max(3, min(int(recall_multiplier), 5))

    @staticmethod
    def _call_embed(embed: Callable[..., Sequence[float]], query: str, model_id: Optional[str]) -> Sequence[float]:
        try:
            return embed(query, model_id=model_id)
        except TypeError:
            # Test doubles and older embedding providers may only accept text.
            return embed(query)

    def _fuse(
        self,
        keyword_hits: Iterable[Any],
        vector_hits: Iterable[Any],
        keyword_weight: float,
        vector_weight: float,
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        order = 0
        for source, weight, hits in (
            ("keyword", keyword_weight, keyword_hits),
            ("vector", vector_weight, vector_hits),
        ):
            for rank, hit in enumerate(hits, start=1):
                hit_id, backend_score, document = _hit_parts(hit)
                key = hit_id or f"{source}:{rank}:{document.get('content', '')}"
                if key not in fused:
                    item = dict(document)
                    item.setdefault("chunk_id", hit_id)
                    item["backend_score"] = backend_score
                    fused[key] = {
                        "item": item,
                        "score": 0.0,
                        "sources": [],
                        "order": order,
                    }
                    order += 1
                fused[key]["score"] += float(weight) / (self.rrf_k + rank)
                fused[key]["sources"].append(source)

        ranked = sorted(fused.values(), key=lambda value: (-value["score"], value["order"]))
        results: List[Dict[str, Any]] = []
        for value in ranked:
            item = value["item"]
            item["score"] = value["score"]
            item["retrieval_mode"] = "hybrid"
            item["sources"] = value["sources"]
            results.append(item)
        return results

    @staticmethod
    def _deduplicate(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen: Dict[Tuple[str, str], int] = {}
        for item in items:
            key = _content_key(item)
            if key in seen:
                retained = unique[seen[key]]
                evidence = {
                    field: item.get(field)
                    for field in (
                        "chunk_id", "doc_id", "source_ref", "image_url", "doc_name",
                        "content_type", "doc_type_kwd", "diagram_id_kwd",
                        "business_id_kwd", "pg_id_kwd", "projection_role_kwd",
                    )
                    if item.get(field)
                }
                if evidence:
                    duplicates = retained.setdefault("duplicate_evidence", [])
                    if evidence not in duplicates:
                        duplicates.append(evidence)
                continue
            seen[key] = len(unique)
            unique.append(item)
        return unique

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        embedding_model_id: Optional[str] = None,
        agent_id: str = "",
        rerank_model_id: str = "",
        keyword_weight: float = 0.5,
        vector_weight: float = 0.5,
        recall_multiplier: Optional[int] = None,
    ) -> RetrievalOutcome:
        top_k = max(1, int(top_k))
        multiplier = self.recall_multiplier if recall_multiplier is None else max(3, min(int(recall_multiplier), 5))
        recall_k = top_k * multiplier
        fallbacks: List[str] = []

        try:
            keyword_weight = float(keyword_weight)
            vector_weight = float(vector_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("检索融合权重必须是数字") from exc
        if (
            not math.isfinite(keyword_weight)
            or not math.isfinite(vector_weight)
            or keyword_weight < 0
            or vector_weight < 0
            or keyword_weight + vector_weight <= 0
        ):
            raise ValueError("检索融合权重必须有限、非负且至少有一项大于 0")

        try:
            keyword_hits = self.search_backend.keyword_search(query, filters=filters, top_k=recall_k) or []
        except Exception as exc:
            logger.warning("Keyword recall failed: %s", exc)
            keyword_hits = []
            fallbacks.append(f"关键词召回失败: {exc}")

        vector: Sequence[float] = []
        try:
            vector = self._call_embed(self.embed, query, embedding_model_id)
        except Exception as exc:
            logger.warning("Dense embedding failed: %s", exc)
            fallbacks.append(f"向量生成失败，使用关键词候选: {exc}")

        try:
            has_vector = vector is not None and len(vector) > 0 and any(
                float(value) != 0.0 for value in vector
            )
        except (TypeError, ValueError) as exc:
            has_vector = False
            fallbacks.append(f"向量格式无效，使用关键词候选: {exc}")

        if has_vector:
            try:
                vector_hits = self.search_backend.vector_search(vector, filters=filters, top_k=recall_k) or []
            except Exception as exc:
                logger.warning("Dense recall failed: %s", exc)
                vector_hits = []
                fallbacks.append(f"向量召回失败，使用关键词候选: {exc}")
        else:
            vector_hits = []
            if not any(
                message.startswith(("向量生成失败", "向量格式无效"))
                for message in fallbacks
            ):
                fallbacks.append("向量为空或全零，使用关键词候选")

        fused = self._fuse(keyword_hits, vector_hits, keyword_weight, vector_weight)
        unique = self._deduplicate(fused)
        outcome = RetrievalOutcome(
            results=[],
            recall_count=len(fused),
            deduplicated_count=len(unique),
            recall_fallbacks=fallbacks,
        )

        try:
            model, source = self.resolve_rerank_model(agent_id, rerank_model_id)
        except Exception as exc:
            model, source = None, ""
            outcome.rerank_error = f"rerank 模型解析失败: {exc}"
            outcome.fallback = "rerank 模型解析失败，已保留融合召回排序"

        if model and unique:
            outcome.rerank_model = str(model.get("name") or model.get("model_name") or model.get("model_id") or "")
            outcome.rerank_source = source
            texts = [str(item.get("content") or item.get("title") or "") for item in unique]
            try:
                # Crucial: the complete deduplicated candidate set is reranked;
                # top_k is applied only after this call.
                scored = self.rerank(query, texts, model, top_n=len(unique))
                if not scored:
                    raise RuntimeError("rerank 返回空结果")
                reordered: List[Dict[str, Any]] = []
                used = set()
                for index, score in scored:
                    if not isinstance(index, int) or index < 0 or index >= len(unique) or index in used:
                        continue
                    item = dict(unique[index])
                    item["original_score"] = item.get("score")
                    item["rerank_score"] = float(score)
                    item["score"] = float(score)
                    item["retrieval_mode"] = "hybrid_rerank"
                    reordered.append(item)
                    used.add(index)
                if not reordered:
                    raise RuntimeError("rerank 未返回有效候选索引")
                # Some providers return fewer rows than requested.  Preserve all
                # candidates before the final slice so fallback remains lossless.
                reordered.extend(dict(item) for index, item in enumerate(unique) if index not in used)
                unique = reordered
                outcome.reranked = True
            except Exception as exc:
                outcome.rerank_error = f"rerank 调用失败: {exc}"
                outcome.fallback = "rerank 失败，已保留融合召回排序"
                logger.warning("%s", outcome.rerank_error)
        elif not model:
            if outcome.rerank_error is None:
                outcome.fallback = "未配置可用 rerank 模型，已保留融合召回排序"

        outcome.results = unique[:top_k]
        return outcome


def build_retrieval_pipeline() -> RetrievalPipeline:
    """Build the production pipeline lazily to avoid import-time cycles."""

    from backend.agent.model_role_service import agent_model_role_service
    from backend.core.embedding import embedding_service
    from backend.core.rerank import rerank_service
    from backend.storage.search_store import search_store

    return RetrievalPipeline(
        search_backend=search_store,
        embed=embedding_service.embed,
        rerank=rerank_service.rerank,
        resolve_rerank_model=agent_model_role_service.resolve_rerank_model,
    )
