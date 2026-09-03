"""Rerank client with lossless long-candidate batching.

The local BGE service limits one document to 20,000 characters, all documents
to 500,000 characters and a request body to 1 MB.  Defaults below deliberately
leave headroom.  ``rerank_*`` model fields or same-name uppercase environment
variables may lower the limits further.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)


class RerankService:
    """Document reranking service compatible with common local APIs."""

    _HARD_MAX_DOCUMENT_CHARS = 20_000
    _HARD_MAX_TOTAL_CHARS = 500_000
    _HARD_MAX_BODY_BYTES = 1_000_000

    _DEFAULT_MAX_DOCUMENT_CHARS = 18_000
    _DEFAULT_MAX_TOTAL_CHARS = 420_000
    _DEFAULT_MAX_BODY_BYTES = 850_000
    _DEFAULT_BATCH_SIZE = 64
    _DEFAULT_CHUNK_OVERLAP_CHARS = 256
    _DEFAULT_QUERY_MAX_CHARS = 8_000

    def rerank(
        self,
        query: str,
        documents: List[str],
        model_config: Dict,
        top_n: Optional[int] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """Rerank every candidate and return original indexes in global order.

        Long documents are scored as overlapping chunks.  Every batch requests
        all scores; a document receives the maximum of its chunk scores.  The
        final ``top_n`` slice is applied only after scores from every batch have
        been mapped back to the original candidate indexes.
        """
        if not query or not documents or not model_config:
            return None

        model_name = str(model_config.get("model_name") or "").strip()
        base_url = str(model_config.get("base_url") or "").strip()
        api_url = str(model_config.get("api_url") or "").strip()
        api_key = str(model_config.get("api_key") or "")
        timeout = model_config.get("timeout") or 30
        if not model_name:
            logger.warning("rerank 模型缺少 model_name")
            return None

        limits = self._request_limits(model_config)
        original_query = str(query)
        normalized_query = original_query[: limits["query_max_chars"]]
        if len(original_query) > len(normalized_query):
            logger.info(
                "rerank query 已规范化: original_chars=%s sent_chars=%s",
                len(original_query),
                len(normalized_query),
            )

        chunk_rows = self._chunk_documents(
            [str(document or "") for document in documents],
            max_chars=limits["max_document_chars"],
            overlap=limits["chunk_overlap_chars"],
        )
        schemas = self._request_schemas(model_config, base_url)
        try:
            batches = self._build_batches(
                chunk_rows,
                query=normalized_query,
                model_name=model_name,
                schemas=schemas,
                max_batch_size=limits["batch_size"],
                max_total_chars=limits["max_total_chars"],
                max_body_bytes=limits["max_body_bytes"],
            )
        except ValueError as exc:
            logger.error("rerank 请求规范化失败: %s", exc)
            return None

        candidate_urls = self._candidate_urls(model_config, base_url, api_url)
        if not candidate_urls:
            logger.error("rerank 调用失败: 未配置 api_url 或 base_url")
            return None

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        selected: Optional[Tuple[str, str]] = None
        scores_by_document: Dict[int, float] = {}
        for batch_number, batch in enumerate(batches, start=1):
            parsed, selected, failures = self._request_batch(
                batch,
                query=normalized_query,
                model_name=model_name,
                candidate_urls=candidate_urls,
                schemas=schemas,
                headers=headers,
                timeout=timeout,
                selected=selected,
            )
            if parsed is None:
                details = " | ".join(
                    f"batch {batch_number}/{len(batches)} {failure}"
                    for failure in failures
                )
                logger.error(
                    "rerank 调用失败（api_url 优先；候选=%s，片段=%s，批次=%s）: %s",
                    len(documents),
                    len(chunk_rows),
                    len(batches),
                    details,
                )
                return None

            for local_index, score in parsed:
                document_index = batch[local_index][0]
                previous = scores_by_document.get(document_index)
                if previous is None or score > previous:
                    scores_by_document[document_index] = score

        missing = sorted(set(range(len(documents))) - set(scores_by_document))
        if missing:
            logger.error("rerank 响应缺少原候选分数: indexes=%s", missing[:20])
            return None

        result = sorted(scores_by_document.items(), key=lambda item: (-item[1], item[0]))
        if top_n is not None and int(top_n) > 0:
            result = result[: int(top_n)]
        logger.info(
            "rerank 完成: candidates=%s chunks=%s batches=%s endpoint=%s schema=%s",
            len(documents),
            len(chunk_rows),
            len(batches),
            selected[0] if selected else "",
            selected[1] if selected else "",
        )
        return result

    def _request_limits(self, model_config: Dict) -> Dict[str, int]:
        max_document_chars = self._limit_value(
            model_config,
            "rerank_max_document_chars",
            self._DEFAULT_MAX_DOCUMENT_CHARS,
            ceiling=self._HARD_MAX_DOCUMENT_CHARS,
        )
        return {
            "max_document_chars": max_document_chars,
            "max_total_chars": self._limit_value(
                model_config,
                "rerank_max_total_chars",
                self._DEFAULT_MAX_TOTAL_CHARS,
                ceiling=self._HARD_MAX_TOTAL_CHARS,
            ),
            "max_body_bytes": self._limit_value(
                model_config,
                "rerank_max_body_bytes",
                self._DEFAULT_MAX_BODY_BYTES,
                ceiling=self._HARD_MAX_BODY_BYTES,
            ),
            "batch_size": self._limit_value(
                model_config, "rerank_batch_size", self._DEFAULT_BATCH_SIZE, ceiling=512
            ),
            "query_max_chars": self._limit_value(
                model_config,
                "rerank_query_max_chars",
                self._DEFAULT_QUERY_MAX_CHARS,
                ceiling=self._HARD_MAX_DOCUMENT_CHARS,
            ),
            "chunk_overlap_chars": self._limit_value(
                model_config,
                "rerank_chunk_overlap_chars",
                self._DEFAULT_CHUNK_OVERLAP_CHARS,
                minimum=0,
                ceiling=max(0, max_document_chars - 1),
            ),
        }

    @staticmethod
    def _limit_value(
        model_config: Dict,
        name: str,
        default: int,
        *,
        minimum: int = 1,
        ceiling: Optional[int] = None,
    ) -> int:
        raw = model_config.get(name)
        if raw in (None, ""):
            raw = os.getenv(name.upper(), str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("忽略无效 rerank 限制 %s=%r，使用默认值 %s", name, raw, default)
            value = default
        value = max(minimum, value)
        return min(value, ceiling) if ceiling is not None else value

    @staticmethod
    def _chunk_documents(
        documents: Sequence[str], *, max_chars: int, overlap: int
    ) -> List[Tuple[int, str]]:
        rows: List[Tuple[int, str]] = []
        step = max(1, max_chars - overlap)
        for document_index, document in enumerate(documents):
            if not document:
                rows.append((document_index, ""))
                continue
            start = 0
            while start < len(document):
                rows.append((document_index, document[start : start + max_chars]))
                if start + max_chars >= len(document):
                    break
                start += step
        return rows

    def _build_batches(
        self,
        rows: Sequence[Tuple[int, str]],
        *,
        query: str,
        model_name: str,
        schemas: Sequence[str],
        max_batch_size: int,
        max_total_chars: int,
        max_body_bytes: int,
    ) -> List[List[Tuple[int, str]]]:
        batches: List[List[Tuple[int, str]]] = []
        current: List[Tuple[int, str]] = []
        current_chars = 0
        for row in rows:
            proposed = current + [row]
            proposed_chars = current_chars + len(row[1])
            exceeds = (
                len(proposed) > max_batch_size
                or proposed_chars > max_total_chars
                or self._largest_body_size(query, proposed, model_name, schemas) > max_body_bytes
            )
            if exceeds and current:
                batches.append(current)
                current = [row]
                current_chars = len(row[1])
            else:
                current = proposed
                current_chars = proposed_chars
            if (
                len(current) > max_batch_size
                or current_chars > max_total_chars
                or self._largest_body_size(query, current, model_name, schemas) > max_body_bytes
            ):
                raise ValueError(
                    "单个规范化片段仍超过请求上限；请调低 "
                    "rerank_max_document_chars 或提高 body 上限"
                )
        if current:
            batches.append(current)
        return batches

    def _largest_body_size(
        self,
        query: str,
        batch: Sequence[Tuple[int, str]],
        model_name: str,
        schemas: Sequence[str],
    ) -> int:
        return max(
            len(
                # requests.post(json=...) serializes with ensure_ascii=True and
                # the default separators.  Measure that representation rather
                # than raw UTF-8 text; Chinese may otherwise be underestimated
                # by roughly two times because it becomes ``\\uXXXX``.
                json.dumps(
                    self._build_body(schema, query, batch, model_name),
                    allow_nan=False,
                ).encode("utf-8")
            )
            for schema in schemas
        )

    @staticmethod
    def _candidate_urls(model_config: Dict, base_url: str, api_url: str) -> List[str]:
        provider = str(model_config.get("provider") or "").lower()
        candidates: List[str] = [api_url] if api_url else []
        base = base_url.rstrip("/")
        if base:
            if provider == "qwen" or "dashscope" in base.lower():
                if base.endswith("/services/rerank"):
                    candidates.append(f"{base}/text-rerank/text-rerank")
                elif "/services/rerank/" in base:
                    candidates.append(base)
                elif base.endswith("/api/v1"):
                    candidates.append(f"{base}/services/rerank/text-rerank/text-rerank")
            candidates.extend(
                (f"{base}/v1/rerank", f"{base}/api/rerank", f"{base}/rerank")
            )
        return list(dict.fromkeys(url for url in candidates if url))

    @staticmethod
    def _request_schemas(model_config: Dict, base_url: str) -> List[str]:
        provider = str(model_config.get("provider") or "").lower()
        if provider == "qwen" or "dashscope" in base_url.lower():
            return ["dashscope", "openai", "tei"]
        return ["openai", "tei"]

    @staticmethod
    def _build_body(
        schema: str,
        query: str,
        batch: Sequence[Tuple[int, str]],
        model_name: str,
    ) -> Dict[str, Any]:
        documents = [text for _document_index, text in batch]
        top_n = len(documents)
        if schema == "tei":
            return {"query": query, "texts": documents, "top_n": top_n}
        if schema == "dashscope":
            return {
                "model": model_name,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n},
            }
        return {
            "model": model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

    def _request_batch(
        self,
        batch: Sequence[Tuple[int, str]],
        *,
        query: str,
        model_name: str,
        candidate_urls: Sequence[str],
        schemas: Sequence[str],
        headers: Dict[str, str],
        timeout: Any,
        selected: Optional[Tuple[str, str]],
    ) -> Tuple[
        Optional[List[Tuple[int, float]]],
        Optional[Tuple[str, str]],
        List[str],
    ]:
        attempts: List[Tuple[str, str]] = [selected] if selected else []
        attempts.extend((url, schema) for url in candidate_urls for schema in schemas)
        attempts = list(dict.fromkeys(attempts))
        failures: List[str] = []
        for url, schema in attempts:
            body = self._build_body(schema, query, batch, model_name)
            try:
                response = requests.post(url, json=body, headers=headers, timeout=timeout)
            except Exception as exc:
                failures.append(f"{url} [{schema}] network_error={exc}")
                continue
            status = int(getattr(response, "status_code", 0) or 0)
            if status < 200 or status >= 300:
                failures.append(
                    f"{url} [{schema}] HTTP {status}: {self._response_detail(response)}"
                )
                continue
            try:
                parsed = self._parse_response(response.json())
            except Exception as exc:
                failures.append(f"{url} [{schema}] invalid_json={exc}")
                continue
            if parsed is None:
                failures.append(f"{url} [{schema}] 响应中没有有效 rerank 结果")
                continue
            by_index: Dict[int, float] = {}
            for local_index, score in parsed:
                if 0 <= local_index < len(batch):
                    previous = by_index.get(local_index)
                    if previous is None or score > previous:
                        by_index[local_index] = score
            missing = sorted(set(range(len(batch))) - set(by_index))
            if missing:
                failures.append(
                    f"{url} [{schema}] 响应缺少 {len(missing)}/{len(batch)} 个片段索引"
                )
                continue
            return sorted(by_index.items()), (url, schema), failures
        return None, selected, failures

    @staticmethod
    def _response_detail(response: Any, max_chars: int = 400) -> str:
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            try:
                text = json.dumps(response.json(), ensure_ascii=False)
            except Exception:
                text = "无响应正文"
        return " ".join(text.split())[:max_chars]

    def _parse_response(self, data: Any) -> Optional[List[Tuple[int, float]]]:
        """Parse OpenAI/Ollama, TEI and DashScope rerank responses."""
        if isinstance(data, dict) and isinstance(data.get("output"), dict):
            data = data["output"]
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return self._parse_result_items(data["results"])
        if isinstance(data, list):
            return self._parse_result_items(data)
        return None

    @staticmethod
    def _parse_result_items(items: Sequence[Any]) -> Optional[List[Tuple[int, float]]]:
        out: List[Tuple[int, float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(index, int) and isinstance(score, (int, float)):
                out.append((index, float(score)))
        if not out:
            return None
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out


rerank_service = RerankService()
