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

    def get_rerank_model(self, agent_id: str) -> Optional[Dict]:
        """获取 Agent 绑定的 rerank 模型。

        与 LLM 角色不同：rerank 是"可选项"，未配置时返回 None（不使用重排序），
        不回退到默认模型。仅当 agent_models 表显式绑定了 rerank 模型时启用。
        """
        try:
            model_id = agent_db.get_model_id_by_role(agent_id, "rerank")
        except Exception as e:
            logger.warning(f"查询 agent_models rerank 失败 (agent={agent_id}): {e}")
            return None

        if not model_id:
            return None

        try:
            model = postgres_client.get_model(model_id)
            if model and model.get("is_active", True) and model.get("model_type") == "rerank":
                return model
            logger.warning(f"agent_models 引用的 rerank 模型 {model_id} 不存在、未激活或非 rerank 类型")
        except Exception as e:
            logger.warning(f"只读引用 llm_models rerank 失败 (model_id={model_id}): {e}")
        return None

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
