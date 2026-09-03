import json
import re
import unittest
from unittest.mock import patch

from backend.core.rerank import RerankService


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RerankRequestLimitTests(unittest.TestCase):
    def setUp(self):
        self.service = RerankService()

    def test_long_candidates_are_chunked_batched_and_globally_ranked(self):
        documents = ["doc-0 " + ("x" * 95) + " BEST_AT_TAIL"]
        documents.extend(f"doc-{index} candidate text" for index in range(1, 77))
        calls = []

        def fake_post(url, json=None, **_kwargs):
            calls.append((url, json))
            results = []
            for local_index, text in enumerate(json["documents"]):
                if "BEST_AT_TAIL" in text:
                    score = 0.999
                else:
                    match = re.search(r"doc-(\d+)", text)
                    score = (int(match.group(1)) / 1000) if match else 0.001
                results.append({"index": local_index, "relevance_score": score})
            return _Response(payload={"results": list(reversed(results))})

        config = {
            "model_name": "bge-reranker-v2-m3",
            "api_url": "http://reranker:8012/v1/rerank",
            "base_url": "http://reranker:8012",
            "rerank_max_document_chars": 40,
            "rerank_chunk_overlap_chars": 20,
            "rerank_max_total_chars": 100,
            "rerank_max_body_bytes": 700,
            "rerank_batch_size": 7,
        }
        with patch("backend.core.rerank.requests.post", side_effect=fake_post):
            result = self.service.rerank("反复重启的原因", documents, config)

        self.assertIsNotNone(result)
        self.assertEqual(77, len(result))
        self.assertEqual(set(range(77)), {index for index, _score in result})
        self.assertEqual(0, result[0][0])  # best evidence is in the long document tail
        self.assertGreater(len(calls), 1)
        for url, body in calls:
            self.assertEqual("http://reranker:8012/v1/rerank", url)
            sent = body["documents"]
            self.assertEqual(len(sent), body["top_n"])
            self.assertLessEqual(len(sent), 7)
            self.assertTrue(all(len(document) <= 40 for document in sent))
            self.assertLessEqual(sum(map(len, sent)), 100)
            encoded = json.dumps(body, allow_nan=False).encode("utf-8")
            self.assertLessEqual(len(encoded), 700)

    def test_chunking_preserves_entire_document_content(self):
        original = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rows = self.service._chunk_documents([original], max_chars=10, overlap=2)
        chunks = [text for index, text in rows if index == 0]
        rebuilt = chunks[0] + "".join(chunk[2:] for chunk in chunks[1:])
        self.assertEqual(original, rebuilt)

    def test_top_n_is_applied_after_all_batches(self):
        documents = [f"doc-{index}" for index in range(12)]
        seen = []

        def fake_post(_url, json=None, **_kwargs):
            seen.extend(json["documents"])
            results = [
                {
                    "index": local_index,
                    "score": float(text.split("-")[1]),
                }
                for local_index, text in enumerate(json["documents"])
            ]
            return _Response(payload=results)

        config = {
            "model_name": "bge-reranker",
            "api_url": "http://reranker/v1/rerank",
            "rerank_batch_size": 4,
        }
        with patch("backend.core.rerank.requests.post", side_effect=fake_post):
            result = self.service.rerank("query", documents, config, top_n=3)

        self.assertEqual(documents, seen)
        self.assertEqual([(11, 11.0), (10, 10.0), (9, 9.0)], result)

    def test_error_log_keeps_preferred_endpoint_failure_details(self):
        def fake_post(url, **_kwargs):
            if url == "http://reranker:8012/v1/rerank":
                return _Response(
                    status_code=413,
                    text='{"detail":"request body exceeds 1MB"}',
                )
            return _Response(status_code=404, text="not found")

        config = {
            "model_name": "bge-reranker-v2-m3",
            "api_url": "http://reranker:8012/v1/rerank",
            "base_url": "http://reranker:8012",
        }
        with self.assertLogs("backend.core.rerank", level="ERROR") as captured:
            with patch("backend.core.rerank.requests.post", side_effect=fake_post):
                result = self.service.rerank("query", ["candidate"], config)

        self.assertIsNone(result)
        log_text = "\n".join(captured.output)
        self.assertIn("http://reranker:8012/v1/rerank [openai] HTTP 413", log_text)
        self.assertIn("request body exceeds 1MB", log_text)
        self.assertIn("http://reranker:8012/rerank [tei] HTTP 404", log_text)

    def test_partial_batch_response_is_rejected(self):
        config = {
            "model_name": "bge-reranker",
            "api_url": "http://reranker/v1/rerank",
        }
        incomplete = _Response(
            payload={"results": [{"index": 0, "relevance_score": 0.8}]}
        )
        with patch("backend.core.rerank.requests.post", return_value=incomplete):
            result = self.service.rerank("query", ["one", "two"], config)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
