import logging
import requests
from backend.config.config import config
from backend.core.model_config import model_config_service

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.dim = config.EMBEDDING_DIM
    
    def _get_model_config(self):
        """获取 embedding 模型配置：仅从 llm_models 表读取，无 config 回退。"""
        model = model_config_service.get_embedding_model()
        if model:
            model_name = model.get("model_name")
            base_url = model.get("base_url")
            if not model_name or not base_url:
                logger.error("embedding 模型缺少 model_name 或 base_url")
                return None
            return {"model": model_name, "base_url": base_url}
        return None
    
    def embed(self, text, model_id=None):
        """获取向量：必须有 DB 配置的 embedding 模型，否则报错。"""
        try:
            if model_id:
                from backend.storage.postgres_client import postgres_client
                model = postgres_client.get_model(model_id)
                if model and model.get("is_active", True):
                    model_name = model.get("model_name")
                    base_url = model.get("base_url")
                    if model_name and base_url:
                        model_config = {"model": model_name, "base_url": base_url}
                    else:
                        raise ValueError(f"指定 embedding 模型 {model_id} 缺少 model_name 或 base_url")
                else:
                    raise ValueError(f"指定 embedding 模型 {model_id} 不存在或未激活")
            else:
                model_config = self._get_model_config()
                if not model_config:
                    raise ValueError("未配置默认 embedding 模型，请在模型管理页面添加")
            
            response = requests.post(
                f"{model_config['base_url']}/api/embeddings",
                json={"model": model_config['model'], "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except ValueError as e:
            logger.error(f"Embedding配置错误: {e}")
            raise
        except Exception as e:
            logger.error(f"Embedding service error: {e}")
            raise

embedding_service = EmbeddingService()
