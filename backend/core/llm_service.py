import os
import re
import json
import base64
import time
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from backend.config.config import config
from backend.core.model_config import model_config_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMStreamError(RuntimeError):
    """流式模型在完整响应形成前失败；调用方必须丢弃已收集的半截文本。"""

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning("langchain_ollama not installed")


def _build_ollama_chat(
    *,
    base_url: str,
    model_name: str,
    temperature: Any = None,
    max_tokens: Any = None,
    timeout: Any = None,
    top_p: Any = None,
    num_predict: Any = None,
    reasoning: Any = None,
    response_format: Any = None,
):
    """Construct ChatOllama with the option names supported by this SDK.

    ``max_tokens``, ``timeout`` and ``model_kwargs`` are silently ignored by
    current ``langchain-ollama`` releases.  Translate the persisted model
    fields to ``num_predict``, direct sampling fields and Ollama client kwargs
    so parser-specific limits are effective rather than merely documented.
    """

    if not HAS_LANGCHAIN:
        raise RuntimeError("langchain_ollama不可用")
    kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "model": model_name,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    prediction_limit = num_predict if num_predict is not None else max_tokens
    if prediction_limit is not None:
        prediction_limit = int(prediction_limit)
        if prediction_limit <= 0:
            raise ValueError("Ollama num_predict/max_tokens 必须为正数")
        kwargs["num_predict"] = prediction_limit
    if timeout is not None:
        request_timeout = float(timeout)
        if request_timeout <= 0:
            raise ValueError("Ollama timeout 必须为正数")
        kwargs["client_kwargs"] = {"timeout": request_timeout}
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    if response_format is not None:
        kwargs["format"] = response_format
    return ChatOllama(**kwargs)


class LLMClient:
    def __init__(self):
        self.config = config
        self.chat_model = None
        self.provider = None
        self._init_model()

    def _init_model(self):
        """初始化 LLM 模型：仅从 llm_models 表读取，无配置则不初始化。"""
        model = model_config_service.get_llm_model()
        if model:
            self.provider = model.get("provider")
            if self.provider == "ollama":
                self._init_ollama_model(
                    base_url=model.get("base_url"),
                    model_name=model.get("model_name"),
                    temperature=model.get("temperature"),
                    max_tokens=model.get("max_tokens"),
                    timeout=model.get("timeout"),
                    top_p=model.get("top_p")
                )
            elif self.provider == "qwen":
                self._init_qwen_model()
            else:
                logger.error(f"未知模型提供者: {self.provider}")
        else:
            logger.error("[LLM未配置] llm_models 表中无默认 LLM 模型，所有 LLM 功能不可用。请在模型管理页面添加模型。")
            self.provider = None

    def _init_ollama_model(self, base_url=None, model_name=None, temperature=None, 
                          max_tokens=None, timeout=None, top_p=None):
        """初始化 Ollama 模型：参数必须从 DB 传入，无 config 回退。"""
        if not HAS_LANGCHAIN:
            logger.warning("langchain_ollama not available")
            return
        if not base_url or not model_name:
            logger.error("Ollama 模型缺少必要的 base_url 或 model_name 参数")
            return
        
        try:
            self.chat_model = _build_ollama_chat(
                base_url=base_url,
                model_name=model_name,
                temperature=temperature if temperature is not None else config.MODEL_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else config.MODEL_MAX_TOKENS,
                timeout=timeout if timeout is not None else config.MODEL_TIMEOUT,
                top_p=top_p if top_p is not None else config.MODEL_TOP_P,
            )
            logger.info(f"Ollama模型初始化成功: {model_name}, URL: {base_url}")
        except Exception as e:
            logger.error(f"Ollama模型初始化失败: {e}")

    def _init_qwen_model(self):
        """初始化 Qwen 模型：仅从 DB 读取 llm 类型模型，无 config 回退。"""
        model = model_config_service.get_llm_model()
        if not model:
            logger.error("[LLM未配置] llm_models 表中无默认 LLM 模型")
            return
        if model.get("provider") != "qwen":
            logger.error(f"默认 LLM 模型 provider 不是 qwen: {model.get('provider')}")
            return
        api_key = model.get("api_key")
        if not api_key:
            logger.error("Qwen 模型缺少 api_key")
            return
        model_name = model.get("model_name")
        logger.info(f"Qwen API模型初始化成功: {model_name}")

    def _call_model(self, messages: List[Dict], model_config: Optional[Dict] = None) -> str:
        """统一 LLM 调用入口：必须有 model_config 或已初始化的 chat_model。"""
        if model_config:
            provider = model_config.get("provider", self.provider)
            if provider == "ollama":
                return self._call_ollama_with_config(messages, model_config)
            elif provider == "qwen":
                return self._call_qwen_with_config(messages, model_config)
            else:
                return "[LLM不可用] 未知 provider"
        else:
            if self.provider == "ollama" and self.chat_model:
                return self._call_ollama(messages)
            elif self.provider == "qwen":
                return self._call_qwen(messages)
            else:
                return "[LLM未配置] 请在模型管理页面添加对话模型后重试"

    def call_model(
        self,
        messages: List[Dict],
        model_config: Optional[Dict] = None,
        *,
        template_name: str = "",
    ) -> str:
        """对外公开的 LLM 同步调用入口（tracing 统一挂钩点）。

        外部模块一律通过本入口调用，不再直接访问私有 ``_call_model``。模板名
        供 B1 全链路 tracing 记录“哪个提示模板产生了这次调用”；语义与
        ``_call_model`` 完全一致。
        """
        started = time.time()
        ok = False
        response = ""
        try:
            response = self._call_model(messages, model_config)
            text = str(response or "").strip()
            ok = bool(text) and not text.startswith(("[LLM", "[AI调用失败]"))
            return response
        finally:
            elapsed_ms = (time.time() - started) * 1000
            logger.debug(
                "call_model finished template=%s elapsed_ms=%.1f",
                template_name or "<default>",
                elapsed_ms,
            )
            try:
                from backend.agent.tracing import record_llm

                record_llm(
                    template_name=template_name,
                    duration_ms=elapsed_ms,
                    ok=ok,
                    model_name=str((model_config or {}).get("model_name") or ""),
                    response_chars=len(str(response or "")) if ok else 0,
                )
            except Exception as exc:  # tracing 失败不得影响调用本身
                logger.debug("trace record_llm failed: %s", exc)

    async def stream_model(self, messages: List[Dict], model_config: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """流式调用 LLM，逐 token yield 输出内容。

        - 有 model_config：按配置走对应 provider 的流式接口
        - 无 model_config：走默认 chat_model（仅支持 Ollama）

        B1 tracing：整段流式生成计为一次 LLM 调用，正常耗尽或异常退出均
        记录；中途被消费方主动中断时在生成器被 close 时兜底记录。
        """
        started = time.time()
        response_chars = 0
        ok = False
        try:
            if model_config:
                provider = model_config.get("provider", self.provider)
                if provider == "ollama":
                    async for chunk in self._stream_ollama_with_config(messages, model_config):
                        response_chars += len(chunk or "")
                        yield chunk
                elif provider == "qwen":
                    async for chunk in self._stream_qwen_with_config(messages, model_config):
                        response_chars += len(chunk or "")
                        yield chunk
                else:
                    raise LLMStreamError("未知模型 provider")
            else:
                if self.provider == "ollama" and self.chat_model:
                    async for chunk in self._stream_ollama(messages):
                        response_chars += len(chunk or "")
                        yield chunk
                else:
                    raise LLMStreamError("未配置可用的流式对话模型")
            ok = True
        finally:
            elapsed_ms = (time.time() - started) * 1000
            try:
                from backend.agent.tracing import record_llm

                record_llm(
                    template_name="<stream>",
                    duration_ms=elapsed_ms,
                    ok=ok,
                    model_name=str((model_config or {}).get("model_name") or ""),
                    response_chars=response_chars,
                )
            except Exception as exc:  # tracing 失败不得影响流式本身
                logger.debug("trace stream_model failed: %s", exc)

    async def _stream_ollama(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """异步流式调用：必须用 astream + async for，避免同步 for 阻塞事件循环
        （实测同步 for 会等全部生成完才一次性返回，导致首包延迟=总耗时）"""
        if not HAS_LANGCHAIN or not self.chat_model:
            raise LLMStreamError("未配置可用的 Ollama 流式模型")
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
        try:
            async for chunk in self.chat_model.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama流式调用失败: {e}")
            raise LLMStreamError(f"Ollama 流式调用失败: {e}") from e

    async def _stream_ollama_with_config(self, messages: List[Dict], model_config: Dict) -> AsyncGenerator[str, None]:
        """异步流式调用：使用 astream 而非 stream，避免同步阻塞事件循环。"""
        if not HAS_LANGCHAIN:
            raise LLMStreamError("langchain_ollama 不可用")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model_name")
        if not base_url or not model_name:
            raise LLMStreamError("Ollama 模型缺少 base_url 或 model_name")
        temperature = model_config.get("temperature", config.MODEL_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", config.MODEL_MAX_TOKENS)
        timeout = model_config.get("timeout", config.MODEL_TIMEOUT)
        top_p = model_config.get("top_p", config.MODEL_TOP_P)
        try:
            chat_model = _build_ollama_chat(
                base_url=base_url, model_name=model_name,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, top_p=top_p,
                num_predict=model_config.get("num_predict"),
                reasoning=model_config.get("reasoning"),
                response_format=model_config.get("format"),
            )
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            async for chunk in chat_model.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama流式调用失败: {e}")
            raise LLMStreamError(f"Ollama 流式调用失败: {e}") from e

    async def _stream_qwen_with_config(self, messages: List[Dict], model_config: Dict) -> AsyncGenerator[str, None]:
        """Qwen 异步流式调用：用 httpx.AsyncClient 替代同步 requests，
        避免 iter_lines 阻塞事件循环（同步 requests 会等全部生成完才返回所有行）。"""
        import httpx
        api_key = model_config.get("api_key")
        if not api_key:
            raise LLMStreamError("Qwen 模型缺少 api_key")
        api_url = model_config.get("api_url")
        model_name = model_config.get("model_name")
        if not api_url or not model_name:
            raise LLMStreamError("Qwen 模型缺少 api_url 或 model_name")
        temperature = model_config.get("temperature", config.MODEL_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", config.MODEL_MAX_TOKENS)
        top_p = model_config.get("top_p", config.MODEL_TOP_P)
        timeout = model_config.get("timeout", config.MODEL_TIMEOUT)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        qwen_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        payload = {"model": model_name, "messages": qwen_messages, "temperature": temperature,
                   "max_tokens": max_tokens, "top_p": top_p, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", api_url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Qwen流式调用失败: {e}")
            raise LLMStreamError(f"Qwen 流式调用失败: {e}") from e

    def _call_ollama(self, messages: List[Dict]) -> str:
        if not self.chat_model:
            return "[LLM未配置] 请在模型管理页面添加对话模型后重试"
        
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_messages.append(HumanMessage(content=msg["content"]))
        
        try:
            response = self.chat_model.invoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error(f"Ollama模型调用失败: {e}")
            return f"[AI调用失败] {str(e)}"

    def _call_qwen(self, messages: List[Dict]) -> str:
        """Qwen 调用：仅从 DB 获取 llm 类型模型配置，无 config 回退。"""
        model = model_config_service.get_llm_model()
        if model and model.get("provider") == "qwen":
            return self._call_qwen_with_config(messages, model)
        return "[LLM未配置] 请在模型管理页面添加对话模型后重试"

    def _call_ollama_with_config(self, messages: List[Dict], model_config: Dict) -> str:
        """带配置的 Ollama 调用：必须有 model_name 和 base_url，无 config 回退。"""
        if not HAS_LANGCHAIN:
            return "[langchain_ollama不可用]"
        
        base_url = model_config.get("base_url")
        model_name = model_config.get("model_name")
        if not base_url or not model_name:
            return "[LLM未配置] 缺少 base_url 或 model_name"
        temperature = model_config.get("temperature", config.MODEL_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", config.MODEL_MAX_TOKENS)
        timeout = model_config.get("timeout", config.MODEL_TIMEOUT)
        top_p = model_config.get("top_p", config.MODEL_TOP_P)
        
        try:
            chat_model = _build_ollama_chat(
                base_url=base_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                top_p=top_p,
                num_predict=model_config.get("num_predict"),
                reasoning=model_config.get("reasoning"),
                response_format=model_config.get("format"),
            )
            
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            invoke_options: Dict[str, Any] = {}
            if "stream" in model_config:
                invoke_options["stream"] = bool(model_config["stream"])
            response = chat_model.invoke(langchain_messages, **invoke_options)
            return response.content
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            return f"[AI调用失败] {str(e)}"

    def _call_qwen_with_config(self, messages: List[Dict], model_config: Dict) -> str:
        """带配置的 Qwen 调用：必须有 api_key、api_url、model_name，无 config 回退。"""
        api_key = model_config.get("api_key")
        if not api_key:
            return "[LLM未配置] 缺少 api_key"
        
        api_url = model_config.get("api_url")
        model_name = model_config.get("model_name")
        if not api_url or not model_name:
            return "[LLM未配置] 缺少 api_url 或 model_name"
        
        temperature = model_config.get("temperature", config.MODEL_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", config.MODEL_MAX_TOKENS)
        top_p = model_config.get("top_p", config.MODEL_TOP_P)
        timeout = model_config.get("timeout", config.MODEL_TIMEOUT)
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            qwen_messages = []
            for msg in messages:
                qwen_messages.append({"role": msg["role"], "content": msg["content"]})
            
            payload = {
                "model": model_name,
                "messages": qwen_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "output" in result and "choices" in result["output"]:
                return result["output"]["choices"][0]["message"]["content"]
            elif "result" in result:
                return result["result"]
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Qwen API调用失败: {e}")
            return f"[AI调用失败] {str(e)}"

    def get_image_mime_type(self, image_path: str) -> str:
        try:
            with open(image_path, 'rb') as f:
                header = f.read(512)
            
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif header.startswith(b'\xff\xd8\xff'):
                return 'image/jpeg'
            elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return 'image/gif'
            elif header.startswith(b'RIFF') and b'WEBP' in header[8:12]:
                return 'image/webp'
            elif header.startswith(b'\x42\x4D'):
                return 'image/bmp'
            elif header.startswith((b'II*\x00', b'MM\x00*')):
                return 'image/tiff'
            elif header.startswith(b'\xd7\xcd\xc6\x9a') or header.startswith(b'\x01\x00\x09\x00'):
                return 'image/wmf'
            elif len(header) >= 44 and header[40:44] == b' EMF':
                return 'image/emf'
            elif b'<svg' in header.lstrip().lower():
                return 'image/svg+xml'
            else:
                # Never label unknown engineering drawing bytes as PNG.  The
                # caller can retain the artifact and explicitly skip vision.
                return 'application/octet-stream'
        except Exception as e:
            logger.error(f"检测图片格式失败: {e}")
            return 'application/octet-stream'

    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败: {e}")
            return ""

    def analyze_image(self, image_path: str, image_title: str = "",
                      section: str = "", context_text: str = "", 
                      model_config: Optional[Dict] = None) -> Dict:
        if not os.path.exists(image_path):
            return {
                "description": f"[错误] 图片文件不存在: {image_path}",
                "image_type": "unknown",
                "confidence": 0.0,
                "normal_image_detail": None,
                "fault_tree_graph": None
            }

        context_section = f"""
## 图片上下文信息

### 章节信息
- 章节名称: {section}
- 图片标题: {image_title}

### 章节文本内容（用于理解图片背景）
{context_text[:3000] if context_text else '无相关上下文文本'}
"""

        system_prompt = """你是航空标准文档分析专家，专门分析DO-160（机载设备环境条件和测试程序）标准文档中的图片。

你需要分析图片并判断它是否为故障树（Fault Tree）图片。

故障树的特征：
1. 有树状结构，根节点是故障，向下分支为子故障或原因
2. 包含逻辑门（AND门、OR门、XOR门），通常用图形符号表示
3. 节点之间有因果关系连线

**重要：对于非故障树的普通图片，你必须提供详细、完整、专业的技术解析，禁止用一两句话敷衍了事。**

普通图片类型包括但不限于：
- 零件机械图（机械部件、结构件、装配图）
- 仪器仪表图（面板、显示屏、控制界面）
- 函数/曲线图（性能曲线、特性图、响应曲线）
- 流程图（工艺流程图、工作流程图）
- 原理图（电路原理图、系统原理图）
- 数据图（柱状图、饼图、折线图）
- 结构框图（系统架构图、功能框图）
- 示意图（安装示意图、操作示意图）

输出格式必须是严格的JSON：

**故障树图片输出格式：**
{
  "image_type": "fault_tree",
  "description": "图片的详细描述（2-3句话概括主要内容）",
  "confidence": 0.0-1.0之间的置信度,
  "fault_tree_graph": {
    "root_fault_id": "fault_001",
    "metadata": {"title": "故障树标题"},
    "vertices": [
      {"id": "fault_001", "label": "Fault", "properties": {"name": "故障名称", "description": "故障描述"}}
    ],
    "edges": [
      {"id": "edge_001", "label": "causes", "source": "fault_001", "target": "cause_001", "properties": {"gate_type": "or", "gate_group": "group_1"}}
    ]
  },
  "normal_image_detail": null
}

**普通图片输出格式（必须详细）：**
{
  "image_type": "normal",
  "confidence": 0.0-1.0,
  "description": "详细的图片解析报告，至少200字，包含图片类型识别、内容描述、技术元素、关系分析、关键参数、技术意义等",
  "normal_image_detail": {
    "image_type_category": "零件图/器械图/函数图/流程图/原理图/数据图/结构框图/示意图/其他",
    "main_content": "图片主要内容的详细描述（100字以上）",
    "technical_elements": ["技术元素1", "技术元素2", "..."],
    "relationships": "图中展示的关系或连接的详细说明",
    "labels_and_annotations": "图中的标签、标注、参数说明的完整解读",
    "data_or_measurements": "图中的数据、测量值、规格参数的提取",
    "function_or_purpose": "图中组件或系统的功能/目的说明",
    "key_insights": "从图片中可以得出的关键技术见解（80字以上）",
    "do160_relevance": "与DO-160标准的关联分析"
  },
  "fault_tree_graph": null
}

注意：确保输出是有效的JSON格式，不要有其他解释文字。"""

        user_prompt = f"""请详细分析这张图片。

{context_section}

**请结合以上章节上下文信息来理解这张图片。**

图片标题：{image_title}
所属章节：{section}

请严格按照要求输出JSON格式。"""

        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return {
                "description": "[错误] 图片编码失败",
                "image_type": "unknown",
                "confidence": 0.0,
                "normal_image_detail": None,
                "fault_tree_graph": None
            }

        mime_type = self.get_image_mime_type(image_path)

        message_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
        ]

        response_text = self._call_model([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ], model_config)

        return self._parse_image_response(response_text)

    def _parse_image_response(self, response_text: str) -> Dict:
        try:
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            data = json.loads(json_text)
            return {
                "description": data.get("description", ""),
                "image_type": data.get("image_type", "unknown"),
                "confidence": data.get("confidence", 0.5),
                "normal_image_detail": data.get("normal_image_detail"),
                "fault_tree_graph": data.get("fault_tree_graph")
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始响应: {response_text[:500]}")
            return {
                "description": f"[AI解析失败] {response_text[:200]}",
                "image_type": "unknown",
                "confidence": 0.0,
                "normal_image_detail": None,
                "fault_tree_graph": None
            }
        except Exception as e:
            logger.error(f"解析AI响应时出错: {e}")
            return {
                "description": f"[解析错误] {str(e)}",
                "image_type": "unknown",
                "confidence": 0.0,
                "normal_image_detail": None,
                "fault_tree_graph": None
            }

    def analyze_table(self, table_markdown: str, table_title: str = "", 
                      section: str = "", context_text: str = "",
                      model_config: Optional[Dict] = None) -> str:
        prompt = f"""请详细分析以下表格。

表格标题：{table_title}
所属章节：{section}

表格内容：
{table_markdown}

**章节上下文（用于理解表格背景）：**
{context_text[:2000] if context_text else '无相关上下文信息'}

请提供详细的分析报告（不少于200字），包含：
1. 表格结构和组织方式
2. 各列/行数据的含义
3. 数据规律和模式识别
4. 关键数据点和高亮信息
5. 数据之间的关联性
6. 在DO-160标准中的实际应用价值
7. 结合章节上下文解释表格与文档的关系
8. 总结表格的核心要点

请用专业、详细的语言分析。"""

        return self._call_model([
            {"role": "system", "content": "你是航空标准文档分析专家。"},
            {"role": "user", "content": prompt}
        ], model_config)

    def extract_entities_and_relations(self, text: str, model_config: Optional[Dict] = None) -> Dict:
        system_prompt = """你是一个知识图谱构建专家，擅长从文档中提取实体和关系。

请从给定的文本中提取实体和关系，输出严格的JSON格式。

实体类型：
- organization: 组织、机构、公司、协会、大学、研究院等
- person: 人物、专家、工程师等
- geo: 地理位置、城市、区域等
- event: 事件、活动、会议、测试等
- category: 类别、分类、类型等
- fault: 故障、问题、缺陷等
- cause: 原因、因素、条件等
- action: 动作、措施、解决方案等
- component: 组件、设备、零件、系统等

输出格式：
{
  "entities": [
    {
      "id": "唯一标识符",
      "name": "实体名称",
      "type": "实体类型",
      "description": "实体描述"
    }
  ],
  "relations": [
    {
      "id": "唯一标识符",
      "source": "源实体名称（必须与entities中的name字段完全匹配）",
      "target": "目标实体名称（必须与entities中的name字段完全匹配）",
      "label": "关系类型",
      "description": "关系描述",
      "weight": 1-10,
      "keywords": ["关键词1", "关键词2"]
    }
  ]
}

重要规则：
1. relations中的source和target字段必须使用实体的name值，而不是实体的id值
2. source和target必须与entities数组中某个实体的name字段完全一致
3. 如果找不到对应的实体名称，则该关系将被忽略

关系类型包括：
- causes: 导致
- has_component: 包含组件
- belongs_to: 属于
- related_to: 相关
- part_of: 部分
- affects: 影响
- requires: 需要
- produces: 产生
- measures: 测量
- controls: 控制

请确保输出有效的JSON格式。"""

        user_prompt = f"""请从以下文本中提取实体和关系：

{text[:5000]}"""

        response_text = self._call_model([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], model_config)

        return self._parse_entity_response(response_text)

    def _parse_entity_response(self, response_text: str) -> Dict:
        try:
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            data = json.loads(json_text)
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            
            id_to_name = {}
            entity_names = []
            for entity in entities:
                entity_id = entity.get("id", "")
                entity_name = entity.get("name", "")
                if entity_id and entity_name:
                    id_to_name[entity_id] = entity_name
                if entity_name:
                    entity_names.append(entity_name)
            
            for relation in relations:
                source = relation.get("source", "")
                target = relation.get("target", "")
                
                if source in id_to_name:
                    relation["source"] = id_to_name[source]
                if target in id_to_name:
                    relation["target"] = id_to_name[target]
                
                if "weight" not in relation:
                    properties = relation.get("properties", {})
                    if isinstance(properties, dict) and "confidence" in properties:
                        relation["weight"] = int(properties["confidence"] * 10)
                    else:
                        relation["weight"] = 5
            
            print(f"[LLM DEBUG] 解析到 {len(entities)} 个实体, {len(relations)} 个关系")
            if relations:
                print(f"[LLM DEBUG] 关系示例: {relations[:2]}")
            print(f"[LLM DEBUG] 实体名称列表: {entity_names}")
            
            return {
                "entities": entities,
                "relations": relations
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始响应: {response_text[:500]}")
            return {"entities": [], "relations": []}
        except Exception as e:
            logger.error(f"解析实体响应时出错: {e}")
            return {"entities": [], "relations": []}

    def extract_entities_and_relations_classic(self, text: str, model_config: Optional[Dict] = None) -> str:
        system_prompt = """---Goal---
Given a text document and a list of entity types, identify all entities of those types 
from the text and all relationships among the identified entities.

---Entity Types---
organization, person, geo, event, category, fault, cause, action, component

---Steps---
1. Identify all entities. Format each entity as:
("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all relationships. Format each relationship as:
("relationship"<|><source_entity_name><|><target_entity_name><|><relationship_description><|><keywords><|><weight>)

3. Use ## as the list delimiter.
4. For relationships, use the exact entity names identified in step 1, NOT entity IDs.

---Text---
{input_text}

---Output---"""

        user_prompt = system_prompt.replace("{input_text}", text[:5000])

        response_text = self._call_model([
            {"role": "user", "content": user_prompt}
        ], model_config)

        return response_text.strip()

    def generate_text(self, prompt: str, system_prompt: str = "", model_config: Optional[Dict] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._call_model(messages, model_config)

    def refresh(self):
        self.chat_model = None
        self.provider = None
        self._init_model()
        logger.info("LLM模型已刷新")

    def test_connection(self) -> bool:
        if not self.chat_model:
            return False
        try:
            self._call_model([{"role": "user", "content": "回复'ok'"}])
            logger.info("LLM模型连接成功")
            return True
        except Exception as e:
            logger.error(f"LLM连接失败: {e}")
            return False

    def generate_community_report(self, prompt: str, model_config: Optional[Dict] = None) -> str:
        return self._call_model([{"role": "user", "content": prompt}], model_config)


llm_service = LLMClient()
