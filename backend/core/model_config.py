from backend.storage.postgres_client import postgres_client

class ModelConfigService:
    def __init__(self):
        self._models = {}
    
    def _get_default_model_from_db(self, model_type: str = None):
        if model_type:
            model = postgres_client.get_default_model_by_type(model_type)
        else:
            model = postgres_client.get_default_model()
        if model:
            return model
        return None
    
    def get_model_by_type(self, model_type: str):
        if model_type not in self._models or self._models[model_type] is None:
            fresh = self._get_default_model_from_db(model_type)
            if fresh is not None:
                self._models[model_type] = fresh
            else:
                self._models[model_type] = None
        return self._models.get(model_type)
    
    def get_llm_model(self):
        return self.get_model_by_type("llm")
    
    def get_embedding_model(self):
        return self.get_model_by_type("embedding")
    
    def get_rerank_model(self):
        return self.get_model_by_type("rerank")
    
    def get_model_by_id(self, model_id: str):
        return postgres_client.get_model(model_id)
    
    def get_current_config(self):
        chat_model = self.get_llm_model()
        embedding_model = self.get_embedding_model()
        rerank_model = self.get_rerank_model()
        
        return {
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "config_status": {
                "chat_configured": chat_model is not None,
                "embedding_configured": embedding_model is not None,
                "ready_for_ai": chat_model is not None and embedding_model is not None
            }
        }
    
    def refresh(self):
        self._models = {}

model_config_service = ModelConfigService()
