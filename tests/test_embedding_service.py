import unittest
from unittest.mock import Mock, patch

from backend.core.embedding import EmbeddingService


class EmbeddingServiceTests(unittest.TestCase):
    def test_embed_many_uses_ollama_batch_api_and_checks_dimension(self):
        service = EmbeddingService()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [[0.1] * 1024, [0.2] * 1024]}

        with patch.object(
            service,
            "_resolve_model_config",
            return_value={"model": "bge-m3:latest", "base_url": "http://ollama:11434"},
        ), patch("backend.core.embedding.requests.post", return_value=response) as post:
            vectors = service.embed_many(["一", "二"])

        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 1024 for vector in vectors))
        post.assert_called_once_with(
            "http://ollama:11434/api/embed",
            json={"model": "bge-m3:latest", "input": ["一", "二"]},
            timeout=60,
        )

    def test_embed_many_rejects_wrong_dimension(self):
        service = EmbeddingService()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [[0.1] * 3]}

        with patch.object(
            service,
            "_resolve_model_config",
            return_value={"model": "bge-m3:latest", "base_url": "http://ollama:11434"},
        ), patch("backend.core.embedding.requests.post", return_value=response):
            with self.assertRaisesRegex(ValueError, "维度不匹配"):
                service.embed("文本")


if __name__ == "__main__":
    unittest.main()
