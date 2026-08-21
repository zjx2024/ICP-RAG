"""Rerank 重排序服务。

设计原则：
- 只读引用 llm_models 表中的 rerank 模型配置（model_name/base_url/api_url/api_key）
- 兼容多种 rerank API 格式：
  1. OpenAI 兼容格式（llama.cpp --rerank / Xinference / 各类网关）:
     POST {base_url}/v1/rerank
     {"model": ..., "query": ..., "documents": [...], "top_n": ...}
     → {"results": [{"index": i, "relevance_score": x}]}
  2. Ollama 格式（社区 build 的 /api/rerank）:
     POST {base_url}/api/rerank
     {"model": ..., "query": ..., "documents": [...]}
     → {"results": [{"index": i, "relevance_score": x}]}
  3. TEI（text-embeddings-inference）格式:
     POST {base_url}/rerank
     {"query": ..., "texts": [...], "top_n": ...}
     → [{"index": i, "score": x}]
- 调用失败时静默返回 None，由调用方决定是否跳过重排序（不影响主流程）
"""
import logging
import requests
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RerankService:
    """文档重排序服务"""

    def rerank(self, query: str, documents: List[str], model_config: Dict,
               top_n: Optional[int] = None) -> Optional[List[Tuple[int, float]]]:
        """对候选文档重排序。

        Args:
            query: 检索 query
            documents: 候选文档文本列表
            model_config: llm_models 表中的 rerank 模型配置
            top_n: 返回前 top_n 个（不传则全量返回）

        Returns:
            [(index, score), ...] 按相关性降序；失败返回 None
        """
        if not query or not documents or not model_config:
            return None

        model_name = model_config.get("model_name")
        base_url = model_config.get("base_url", "")
        api_url = model_config.get("api_url", "")
        api_key = model_config.get("api_key", "")
        timeout = model_config.get("timeout") or 30

        if not model_name:
            logger.warning("rerank 模型缺少 model_name")
            return None

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # DashScope/百炼原生 rerank：OpenAI 兼容模式（/compatible-mode/v1）不支持
        # /v1/rerank，必须走原生端点 /api/v1/services/rerank/text-rerank/text-rerank
        provider = (model_config.get("provider") or "").lower()
        if provider == "qwen" or "dashscope" in base_url.lower():
            parsed = self._dashscope_rerank(query, documents, model_name, base_url, api_key, top_n, timeout)
            if parsed is not None:
                return parsed

        # 依次尝试候选 URL（api_url 优先，其次常见路径）
        candidate_urls = []
        if api_url:
            candidate_urls.append(api_url)
        if base_url:
            candidate_urls.append(f"{base_url.rstrip('/')}/v1/rerank")
            candidate_urls.append(f"{base_url.rstrip('/')}/api/rerank")
            candidate_urls.append(f"{base_url.rstrip('/')}/rerank")

        # 依次尝试候选请求体（OpenAI 格式优先，TEI 格式兜底）
        bodies = []
        if top_n:
            bodies.append({"model": model_name, "query": query, "documents": documents, "top_n": top_n})
        else:
            bodies.append({"model": model_name, "query": query, "documents": documents})
        bodies.append({"query": query, "texts": documents, "top_n": top_n or len(documents)})

        last_error = None
        for url in candidate_urls:
            for body in bodies:
                try:
                    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
                    resp.raise_for_status()
                    parsed = self._parse_response(resp.json())
                    if parsed is not None:
                        return parsed
                except Exception as e:
                    last_error = e
                    continue

        logger.error(f"rerank 调用失败: {last_error}")
        return None

    def _parse_response(self, data) -> Optional[List[Tuple[int, float]]]:
        """解析 rerank 响应，兼容两种格式。"""
        # 格式 1: {"results": [{"index": i, "relevance_score": x}]}
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list):
                out = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    score = item.get("relevance_score", item.get("score"))
                    if isinstance(idx, int) and isinstance(score, (int, float)):
                        out.append((idx, float(score)))
                if out:
                    out.sort(key=lambda x: x[1], reverse=True)
                    return out
        # 格式 2: [{"index": i, "score": x}]
        if isinstance(data, list):
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                score = item.get("score", item.get("relevance_score"))
                if isinstance(idx, int) and isinstance(score, (int, float)):
                    out.append((idx, float(score)))
            if out:
                out.sort(key=lambda x: x[1], reverse=True)
                return out
        return None

    def _dashscope_rerank(self, query: str, documents: List[str], model_name: str,
                          base_url: str, api_key: str, top_n: Optional[int],
                          timeout: int) -> Optional[List[Tuple[int, float]]]:
        """DashScope/百炼原生 rerank API。

        端点: POST {base_url}/services/rerank/text-rerank/text-rerank
        请求体（嵌套 input/parameters 结构）:
            {"model": ..., "input": {"query": ..., "documents": [...]}, "parameters": {"top_n": n}}
        响应: {"output": {"results": [{"index": i, "relevance_score": x}]}}
        """
        b = (base_url or "").rstrip("/")
        if b.endswith("/services/rerank"):
            url = f"{b}/text-rerank/text-rerank"
        elif "/services/rerank/" in b:
            url = b
        elif b.endswith("/api/v1"):
            url = f"{b}/services/rerank/text-rerank/text-rerank"
        else:
            return None
        body = {"model": model_name, "input": {"query": query, "documents": documents}}
        if top_n:
            body["parameters"] = {"top_n": top_n}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            results = (data.get("output") or {}).get("results") or []
            out = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                score = item.get("relevance_score")
                if isinstance(idx, int) and isinstance(score, (int, float)):
                    out.append((idx, float(score)))
            if out:
                out.sort(key=lambda x: x[1], reverse=True)
                return out
        except Exception as e:
            logger.error(f"DashScope rerank 调用失败: {e}")
        return None


rerank_service = RerankService()
