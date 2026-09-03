"""Agent 模型角色服务。

设计原则：
- 不修改现有 model_config.py
- 新建独立服务，查 agent_models 表 + 只读引用 llm_models
- 通过 postgres_client.get_model() 只读引用现有 llm_models（不修改原表）
"""
import logging
from typing import Dict, Optional

from backend.storage.postgres_client import postgres_client
from backend.core.model_config import model_config_service  # 只读引用，不修改
from backend.agent.db import agent_db

logger = logging.getLogger(__name__)


class AgentModelRoleService:
    """按 (agent_id, role) 取模型配置，只读引用现有 llm_models"""

    def get_model_by_role(self, agent_id: str, role: str) -> Optional[Dict]:
        """取模型配置

        优先级:
        1. agent_models 表中 (agent_id, role) 指定的模型（只读引用 llm_models）
        2. 回退到现有默认 LLM（model_config_service.get_llm_model()，不改原方法）
        """
        # 1. 查 agent_models 表拿 model_id
        try:
            model_id = agent_db.get_model_id_by_role(agent_id, role)
        except Exception as e:
            logger.warning(f"查询 agent_models 失败 (agent={agent_id}, role={role}): {e}")
            model_id = None

        if model_id:
            # 只读引用现有 llm_models（不改原表）
            try:
                model = postgres_client.get_model(model_id)
                if model and model.get("is_active", True):
                    return model
                logger.warning(f"agent_models 引用的模型 {model_id} 不存在或未激活")
            except Exception as e:
                logger.warning(f"只读引用 llm_models 失败 (model_id={model_id}): {e}")

        # 2. 回退到现有默认 LLM（只读调用，不改原方法）
        return model_config_service.get_llm_model()

    @staticmethod
    def _valid_rerank_model(model: Optional[Dict]) -> Optional[Dict]:
        if model and model.get("is_active", True) and model.get("model_type") == "rerank":
            return model
        return None

    def resolve_rerank_model(
        self,
        agent_id: str = "",
        rerank_model_id: str = "",
    ) -> tuple[Optional[Dict], str]:
        """Resolve rerank model in a stable, auditable order.

        Priority is an explicitly requested model, the Agent role binding, then
        PostgreSQL's active default rerank model.  The source string is returned
        with the model so retrieval diagnostics can show which fallback won.
        """
        if rerank_model_id:
            try:
                model = self._valid_rerank_model(postgres_client.get_model(rerank_model_id))
                if model:
                    return model, "explicit"
                logger.warning("指定 rerank 模型 %s 不存在、未激活或类型错误", rerank_model_id)
            except Exception as e:
                logger.warning("读取指定 rerank 模型失败 (model_id=%s): %s", rerank_model_id, e)

        model_id = None
        if agent_id:
            try:
                model_id = agent_db.get_model_id_by_role(agent_id, "rerank")
            except Exception as e:
                logger.warning(f"查询 agent_models rerank 失败 (agent={agent_id}): {e}")

        if model_id:
            try:
                model = self._valid_rerank_model(postgres_client.get_model(model_id))
                if model:
                    return model, "agent_binding"
                logger.warning(f"agent_models 引用的 rerank 模型 {model_id} 不存在、未激活或非 rerank 类型")
            except Exception as e:
                logger.warning(f"只读引用 llm_models rerank 失败 (model_id={model_id}): {e}")

        try:
            model = self._valid_rerank_model(postgres_client.get_default_model_by_type("rerank"))
            if model:
                return model, "postgres_default"
        except Exception as e:
            logger.warning("读取 PostgreSQL 默认 rerank 模型失败: %s", e)
        return None, ""

    def get_rerank_model(self, agent_id: str = "") -> Optional[Dict]:
        """获取 Agent 绑定模型；未绑定时回退 PostgreSQL 默认 rerank。"""
        model, _source = self.resolve_rerank_model(agent_id=agent_id)
        return model

    @staticmethod
    def apply_agent_llm_params(state: Dict, model: Optional[Dict]) -> Optional[Dict]:
        """用智能体编辑表单配置的模型采样参数覆盖 llm_models 中的默认值。

        temperature/top_p/max_tokens 优先级：智能体配置 > 模型默认配置 > config.py 全局默认。
        """
        if not model:
            return model
        model = dict(model)
        for key in ("temperature", "top_p", "max_tokens"):
            value = state.get(key)
            if value is not None:
                model[key] = value
        return model


agent_model_role_service = AgentModelRoleService()
