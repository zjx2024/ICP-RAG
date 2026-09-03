import logging
import requests
from backend.config.config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.dim = config.EMBEDDING_DIM
    
    def _get_model_config(self):
        """获取 embedding 模型配置：仅从 llm_models 表读取，无 config 回退。"""
        # Keep the service import-safe for health tooling and isolated tests.  The
        # model repository opens PostgreSQL, so load it only when a model must be
        # resolved for an actual embedding request.
        from backend.core.model_config import model_config_service

        model = model_config_service.get_embedding_model()
        if model:
            model_name = model.get("model_name")
            base_url = model.get("base_url")
            if not model_name or not base_url:
                logger.error("embedding 模型缺少 model_name 或 base_url")
                return None
            return {
                "model": model_name,
                "base_url": base_url.rstrip("/"),
                "model_id": model.get("model_id"),
            }
        return None

    def _resolve_model_config(self, model_id=None):
        if not model_id:
            model_config = self._get_model_config()
            if not model_config:
                raise ValueError("未配置默认 embedding 模型，请在模型管理页面添加")
            return model_config

        from backend.storage.postgres_client import postgres_client

        model = postgres_client.get_model(model_id)
        if not model or not model.get("is_active", True):
            raise ValueError(f"指定 embedding 模型 {model_id} 不存在或未激活")
        model_name = model.get("model_name")
        base_url = model.get("base_url")
        if not model_name or not base_url:
            raise ValueError(f"指定 embedding 模型 {model_id} 缺少 model_name 或 base_url")
        return {
            "model": model_name,
            "base_url": base_url.rstrip("/"),
            "model_id": model_id,
        }

    def _validate(self, vector):
        if not isinstance(vector, list) or len(vector) != self.dim:
            actual = len(vector) if isinstance(vector, list) else 0
            raise ValueError(f"Embedding 维度不匹配：期望 {self.dim}，实际 {actual}")
        return [float(value) for value in vector]

    def embed_many(self, texts, model_id=None):
        """Use Ollama's batch API and enforce the OpenSearch vector schema."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding 输入必须是字符串列表")

        try:
            model_config = self._resolve_model_config(model_id)
            response = requests.post(
                f"{model_config['base_url']}/api/embed",
                json={"model": model_config["model"], "input": texts},
                timeout=60,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError("Ollama embedding 响应数量与请求不一致")
            return [self._validate(vector) for vector in embeddings]
        except ValueError as e:
            logger.error(f"Embedding配置错误: {e}")
            raise
        except Exception as e:
            logger.error(f"Embedding service error: {e}")
            raise

    def embed(self, text, model_id=None):
        """获取单条 BGE-M3 向量；底层复用批量接口。"""
        return self.embed_many([text], model_id=model_id)[0]

embedding_service = EmbeddingService()
