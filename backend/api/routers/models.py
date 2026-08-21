from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional, List, Dict
import time as _time
import requests
from backend.storage.postgres_client import postgres_client
from backend.core.model_config import model_config_service
from backend.core.llm_service import llm_service

router = APIRouter(prefix="/models", tags=["models"])

class ModelCreate(BaseModel):
    name: str
    model_type: str = "llm"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 262144
    top_p: float = 0.1
    timeout: int = 3000
    is_default: bool = False
    is_active: bool = True
    description: Optional[str] = None

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    model_type: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    timeout: Optional[int] = None
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None
    description: Optional[str] = None

class DefaultModelConfig(BaseModel):
    chat_model_id: Optional[str] = None
    embedding_model_id: Optional[str] = None
    rerank_model_id: Optional[str] = None

MODEL_TYPES = ["llm", "embedding", "rerank", "vision"]

@router.post("/")
def create_model(model: ModelCreate):
    if model.model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"无效的模型类型: {model.model_type}，可选类型: {MODEL_TYPES}")
    
    model_id = str(uuid4())
    try:
        postgres_client.insert_model(
            model_id=model_id,
            name=model.name,
            model_type=model.model_type,
            provider=model.provider,
            api_key=model.api_key,
            api_url=model.api_url,
            model_name=model.model_name,
            base_url=model.base_url,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
            timeout=model.timeout,
            is_default=model.is_default,
            is_active=model.is_active,
            description=model.description
        )
        return {"model_id": model_id, "name": model.name, "model_type": model.model_type, "message": "模型创建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建模型失败: {str(e)}")

@router.get("")
@router.get("/")
def list_models(model_type: Optional[str] = None):
    try:
        if model_type:
            models = postgres_client.list_models_by_type(model_type)
        else:
            models = postgres_client.list_models()
        for model in models:
            if model.get("api_key"):
                model["api_key"] = "******"
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@router.get("/types")
def get_model_types():
    return {
        "types": [
            {"value": "llm", "label": "LLM对话模型", "description": "用于对话、文本生成等任务"},
            {"value": "embedding", "label": "Embedding模型", "description": "用于文本向量化、相似度计算"},
            {"value": "rerank", "label": "重排序模型", "description": "用于检索结果重排序"},
            {"value": "vision", "label": "视觉模型", "description": "用于图像分析、理解"}
        ]
    }

@router.get("/default")
def get_default_model():
    try:
        model = postgres_client.get_default_model()
        if not model:
            raise HTTPException(status_code=404, detail="没有默认模型")
        if model.get("api_key"):
            model["api_key"] = "******"
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取默认模型失败: {str(e)}")

@router.get("/default/{model_type}")
def get_default_model_by_type(model_type: str):
    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"无效的模型类型: {model_type}")
    
    try:
        model = postgres_client.get_default_model_by_type(model_type)
        if not model:
            raise HTTPException(status_code=404, detail=f"没有默认的{model_type}模型")
        if model.get("api_key"):
            model["api_key"] = "******"
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取默认{model_type}模型失败: {str(e)}")

class SetDefaultRequest(BaseModel):
    model_type: str
    model_id: str

@router.post("/set-default")
def set_default_model(request: SetDefaultRequest):
    model_type = request.model_type
    model_id = request.model_id
    
    if model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"无效的模型类型: {model_type}")
    
    try:
        model = postgres_client.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        if model.get("model_type") != model_type:
            raise HTTPException(status_code=400, detail=f"模型类型不匹配，该模型是{model.get('model_type')}类型")
        
        postgres_client.set_default_model_by_type(model_id, model_type)
        model_config_service.refresh()
        llm_service.refresh()
        return {"message": f"已设置为默认{model_type}模型"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置默认模型失败: {str(e)}")

@router.get("/current-config")
def get_current_model_config():
    try:
        model_config_service.refresh()
        
        def mask_sensitive(data):
            if not data:
                return None
            result = data.copy()
            if result.get("api_key"):
                result["api_key"] = "******"
            return result
        
        chat_model = postgres_client.get_default_model_by_type("llm")
        embedding_model = postgres_client.get_default_model_by_type("embedding")
        rerank_model = postgres_client.get_default_model_by_type("rerank")
        
        return {
            "chat_model": mask_sensitive(chat_model),
            "embedding_model": mask_sensitive(embedding_model),
            "rerank_model": mask_sensitive(rerank_model),
            "config_status": {
                "chat_configured": chat_model is not None,
                "embedding_configured": embedding_model is not None,
                "ready_for_ai": chat_model is not None and embedding_model is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取当前模型配置失败: {str(e)}")

@router.post("/{model_id}/test")
def test_model_connection(model_id: str):
    """测试指定模型的连接可用性：按模型实际类型（chat/vision/embedding/rerank）发最小请求验证。

    类型判定：模型名特征优先（rerank/embedding/vision），model_type 兜底——
    避免历史数据中 rerank 模型被标成 llm 后误走 chat 测试方式。
    """
    try:
        model = postgres_client.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")

        base_url = (model.get("base_url") or "").rstrip("/")
        api_url = model.get("api_url") or ""
        model_name = model.get("model_name") or ""
        provider = model.get("provider") or ""
        api_key = model.get("api_key") or ""
        # 测试连接用更短超时（连接 5s / 读取 8s），避免地址不可达时用户等待过久
        timeout = (5, 8)
        mtype = model.get("model_type") or "llm"

        if not model_name or not (base_url or api_url):
            return {"ok": False, "message": "缺少 model_name 或 base_url/api_url 配置，请先在编辑中完善"}

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start = _time.time()
        last_err = "未知错误"

        # ---------- 1. 类型判定：模型名特征优先，model_type 兜底 ----------
        name_lower = model_name.lower()
        if "rerank" in name_lower or mtype == "rerank":
            kind = "rerank"
        elif mtype == "embedding" or "embed" in name_lower:
            kind = "embedding"
        elif mtype == "vision" or "vision" in name_lower or "-vl" in name_lower or "-vl-" in name_lower:
            kind = "vision"
        else:
            kind = "chat"

        def _ok(extra=""):
            ms = int((_time.time() - start) * 1000)
            return {"ok": True, "message": f"连接成功（{model_name}，{kind}）{extra}，耗时 {ms}ms", "latency_ms": ms}

        def _try(urls, bodies):
            """对每个候选 URL 依次尝试所有请求体，任一 2xx 即成功。返回 (ok, err)。"""
            err = ""
            for u in urls:
                for b in bodies:
                    try:
                        r = requests.post(u, json=b, headers=headers, timeout=timeout)
                        r.raise_for_status()
                        return True, u
                    except Exception as e:
                        err = f"{u}: {e}"
            return False, err

        def _fail(kind_name, err):
            return {"ok": False, "message": f"连接失败（{kind_name}）：{err}"}

        if kind == "chat":
            # chat：OpenAI 兼容 chat completions（qwen/dashscope、Ollama 兼容端点均适用），
            # 另加 Ollama 原生 /api/chat、DashScope 原生 input/messages 格式兜底
            candidates = []
            if api_url:
                candidates.append(api_url)
            if base_url:
                candidates.append(f"{base_url}/v1/chat/completions")
                candidates.append(f"{base_url}/chat/completions")
                candidates.append(f"{base_url}/api/chat")
            elif provider == "qwen":
                # 未填 base_url 的 qwen 模型，补 dashscope compatible-mode 端点
                candidates.append("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            openai_body = {"model": model_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
            ollama_body = {"model": model_name, "messages": [{"role": "user", "content": "ping"}], "stream": False}
            dashscope_body = {"model": model_name, "input": {"messages": [{"role": "user", "content": "ping"}]}, "parameters": {"max_tokens": 5}}
            ok, err = _try(candidates, [openai_body, ollama_body, dashscope_body])
            if ok:
                return _ok()
            return _fail(kind, err)

        if kind == "vision":
            # vision：先发送一张 1x1 透明 PNG 图片（OpenAI image_url / Ollama images 两种格式），
            # 真正验证视觉模型可用；图片失败再退化为纯文本 chat，确认服务可达
            candidates = []
            if api_url:
                candidates.append(api_url)
            if base_url:
                candidates.append(f"{base_url}/v1/chat/completions")
                candidates.append(f"{base_url}/chat/completions")
                candidates.append(f"{base_url}/api/chat")
            elif provider == "qwen":
                candidates.append("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            vision_openai_body = {
                "model": model_name,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "ping"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}],
                "max_tokens": 5,
            }
            vision_ollama_body = {"model": model_name, "messages": [{"role": "user", "content": "ping", "images": [img_b64]}], "stream": False}
            ok, err = _try(candidates, [vision_openai_body, vision_ollama_body])
            if ok:
                return _ok("（图片验证）")
            openai_body = {"model": model_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
            ollama_body = {"model": model_name, "messages": [{"role": "user", "content": "ping"}], "stream": False}
            ok, err = _try(candidates, [openai_body, ollama_body])
            if ok:
                return _ok("（文字验证）")
            return _fail(kind, err)

        if kind == "embedding":
            candidates = []
            if api_url:
                candidates.append(api_url)
            if base_url:
                candidates.append(f"{base_url}/api/embeddings")
                candidates.append(f"{base_url}/v1/embeddings")
                candidates.append(f"{base_url}/embeddings")
            body = {"model": model_name, "input": "ping"}
            ok, err = _try(candidates, [body])
            if ok:
                return _ok()
            return _fail(kind, err)

        if kind == "rerank":
            candidates = []
            if api_url:
                candidates.append(api_url)
            if base_url:
                candidates.append(f"{base_url}/v1/rerank")
                candidates.append(f"{base_url}/api/rerank")
                candidates.append(f"{base_url}/rerank")
            openai_body = {"model": model_name, "query": "ping", "documents": ["ping"]}
            tei_body = {"query": "ping", "texts": ["ping"]}
            # DashScope/百炼：OpenAI 兼容模式无 /v1/rerank，必须走原生 rerank 端点
            dashscope_body = {"model": model_name, "input": {"query": "ping", "documents": ["ping"]}, "parameters": {"top_n": 1}}
            if provider == "qwen" and base_url:
                candidates.insert(0, f"{base_url}/services/rerank/text-rerank/text-rerank")
            ok, err = _try(candidates, [dashscope_body, openai_body, tei_body])
            if ok:
                return _ok()
            # Ollama 服务器未暴露 rerank API（/api/rerank、/v1/rerank 均 404）时，
            # 退化为 /api/tags 模型存在性校验：连接测试通过，但注明实际重排序不可用
            if provider == "ollama" and base_url:
                try:
                    r = requests.get(f"{base_url}/api/tags", timeout=(5, 8))
                    if r.status_code == 200:
                        names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
                        if model_name in names:
                            ms = int((_time.time() - start) * 1000)
                            return {
                                "ok": True,
                                "message": f"连接成功（{model_name}，rerank）：模型已部署在服务器，但服务器未提供 rerank API（/api/rerank、/v1/rerank 均不可用），实际重排序功能不可用。耗时 {ms}ms",
                                "latency_ms": ms,
                            }
                except Exception:
                    pass
            return _fail(kind, err)

        return {"ok": False, "message": f"不支持的模型类型: {mtype}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试连接失败: {str(e)}")

@router.get("/{model_id}")
def get_model(model_id: str):
    try:
        model = postgres_client.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        # 详情接口返回真实 api_key，供编辑弹窗"点击眼睛查看"；
        # 列表(list_models)/current-config 等接口仍掩码显示
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型失败: {str(e)}")

@router.put("/{model_id}")
def update_model(model_id: str, model: ModelUpdate):
    try:
        update_data = {k: v for k, v in model.dict().items() if v is not None}
        # api_key 为空字符串或掩码值（'******'）时视为"不修改"，避免编辑保存时把已存密钥覆盖为空
        if "api_key" in update_data and (not update_data["api_key"] or update_data["api_key"] == "******"):
            del update_data["api_key"]
        if not update_data:
            raise HTTPException(status_code=400, detail="没有提供更新字段")
        
        success = postgres_client.update_model(model_id, **update_data)
        if not success:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        if update_data.get("is_default") or update_data.get("model_type"):
            model_config_service.refresh()
            llm_service.refresh()
        
        return {"message": "模型更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新模型失败: {str(e)}")

@router.post("/refresh")
def refresh_models():
    try:
        model_config_service.refresh()
        llm_service.refresh()
        return {"message": "模型配置已刷新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新模型配置失败: {str(e)}")

@router.delete("/{model_id}")
def delete_model(model_id: str):
    try:
        success = postgres_client.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        model_config_service.refresh()
        llm_service.refresh()
        return {"message": "模型删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")
