"""模型服务：LLM / Embedding 模型解析与校验（从 documents.py 路由抽取）。

保持原实现与 HTTPException 行为完全一致，仅调整归属层。
"""
from typing import Dict, Optional

from fastapi import HTTPException

from backend.storage.postgres_client import postgres_client

def resolve_llm_model_config(model_id: Optional[str]) -> Optional[Dict]:
    """解析/校验用户选择的 Chat(LLM) 模型，返回 model_config dict。

    - model_id 为空：返回系统默认 LLM 模型配置；不存在则返回 None
    - model_id 非空：校验模型存在且 model_type='llm'，否则抛 HTTPException(400)
    所有 LLM 使用场景统一走 llm_models 表，杜绝硬编码默认回退。
    """
    if model_id:
        model = postgres_client.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        if model.get("model_type", "llm") != "llm":
            raise HTTPException(
                status_code=400,
                detail=f"所选模型「{model.get('name')}」类型为 {model.get('model_type')}，"
                       f"解析/图谱生成需要 Chat(LLM) 类型模型，请到模型管理选择正确的模型类型。"
            )
        return {
            "provider": model["provider"],
            "api_key": model["api_key"],
            "api_url": model["api_url"],
            "model_name": model["model_name"],
            "base_url": model["base_url"],
            "temperature": model["temperature"],
            "max_tokens": model["max_tokens"],
            "top_p": model["top_p"],
            "timeout": model["timeout"],
        }
    # 未指定模型 → 走 llm_models 表的默认 LLM
    default_llm = postgres_client.get_default_model_by_type("llm")
    if not default_llm:
        raise HTTPException(
            status_code=400,
            detail="未配置默认 Chat(LLM) 模型，无法执行该操作。请到「模型管理」页面添加 model_type=llm 的模型并设为默认。"
        )
    return {
        "provider": default_llm["provider"],
        "api_key": default_llm["api_key"],
        "api_url": default_llm["api_url"],
        "model_name": default_llm["model_name"],
        "base_url": default_llm["base_url"],
        "temperature": default_llm["temperature"],
        "max_tokens": default_llm["max_tokens"],
        "top_p": default_llm["top_p"],
        "timeout": default_llm["timeout"],
    }


def require_embedding_model_configured():
    """预检查系统已配置可用的 Embedding 模型（向量入库必需）。

    未配置则抛 HTTPException(400)，提示用户到模型管理添加 embedding 类型模型。
    """
    default_embed = postgres_client.get_default_model_by_type("embedding")
    if default_embed:
        return
    embed_models = postgres_client.list_models_by_type("embedding")
    if embed_models:
        return
    raise HTTPException(
        status_code=400,
        detail="未配置 Embedding 模型，文档解析需要向量化入库。请到「模型管理」页面添加 model_type=embedding 的模型并设为默认。"
    )
