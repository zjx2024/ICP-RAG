"""ICP 文档图像解析。

替代旧 DO-160 通用图片提示词，针对 FTA、RBD 和系统架构图输出统一节点/边结构。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from backend.core.llm_service import llm_service

logger = logging.getLogger(__name__)

_VISION_SUPPORTED_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
)


def _schema_instruction(doc_type: str) -> str:
    if doc_type == "fta":
        return """
图类型是 FTA 故障树。请识别：顶事件、中间事件、基本事件、AND/OR/XOR 门和连线方向。
vertices.label 只能使用 TopEvent、IntermediateEvent、BasicEvent、Gate；
edges 从上层事件/逻辑门指向下层原因，properties 中保留 gate_type 和 gate_group。
metadata 必须包含 top_event_name；无法确认的字段留空，不要编造概率。
"""
    if doc_type == "rbd":
        return """
图类型是 RBD 可靠性框图。请识别：START、END、功能块、串联路径、并联分支、冗余单元和旁路。
vertices.label 只能使用 Start、End、ReliabilityBlock；
edges 表示成功路径方向，properties 中保留 connection_type（series/parallel/redundant/bypass）与 branch_id。
metadata 尽可能提取 success_criteria、mission_time、reliability_parameters；没有参数时留空，禁止猜数值。
"""
    return """
图类型是 ICP 系统架构/功能图。请识别：系统、子系统、模块、功能、接口、数据流。
vertices.label 只能使用 System、Subsystem、Module、Function、Interface；
edges.label 使用 contains、implements、connects_to、data_flow、depends_on 等清晰关系。
metadata 尽可能给出系统名称和图用途；不要把视觉布局误当作工程关系。
"""


def _parse_json(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    if not value.startswith("{"):
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start:end + 1]
    try:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("diagram result is not an object")
        return data
    except Exception as exc:
        logger.warning("ICP diagram JSON parse failed: %s", exc)
        return {
            "diagram_type": "unknown",
            "description": f"[图像解析失败] {text[:500]}",
            "confidence": 0.0,
            "metadata": {},
            "vertices": [],
            "edges": [],
        }


def analyze_icp_diagram(
    image_path: str,
    doc_type: str,
    title: str = "",
    context_text: str = "",
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {
            "diagram_type": doc_type,
            "description": f"[错误] 图片不存在: {image_path}",
            "confidence": 0.0,
            "metadata": {},
            "vertices": [],
            "edges": [],
        }

    mime_type = llm_service.get_image_mime_type(image_path)
    if mime_type not in _VISION_SUPPORTED_MIME_TYPES:
        return {
            "diagram_type": doc_type,
            "description": f"[跳过视觉解析] 当前视觉链不支持 MIME: {mime_type}",
            "confidence": 0.0,
            "metadata": {
                "mime_type": mime_type,
                "vision_skipped": True,
                "skip_reason": "unsupported_image_mime",
            },
            "vertices": [],
            "edges": [],
        }
    image_base64 = llm_service.encode_image_to_base64(image_path)
    if not image_base64:
        return {
            "diagram_type": doc_type,
            "description": "[错误] 图片编码失败",
            "confidence": 0.0,
            "metadata": {},
            "vertices": [],
            "edges": [],
        }

    system_prompt = f"""你是航空电子系统可靠性与维修知识工程专家，正在解析 ICP 故障诊断知识库的原始 Word 文档。
{_schema_instruction(doc_type)}

输出必须是严格 JSON：
{{
  "diagram_type": "{doc_type}",
  "description": "基于图中可见内容的详细说明",
  "confidence": 0.0,
  "root_id": "顶事件/起点ID，可空",
  "metadata": {{}},
  "vertices": [
    {{"id": "v1", "label": "节点类型", "properties": {{"name": "节点名称", "description": "节点说明"}}}}
  ],
  "edges": [
    {{"id": "e1", "label": "关系", "source": "v1", "target": "v2", "properties": {{}}}}
  ]
}}

要求：
- 只提取图中能够确认的信息；模糊或不可读内容明确标记 unknown。
- 节点 ID 在本图内唯一，边引用的节点必须存在。
- 图题、邻近段落与图中标签冲突时，以图中可见标签为主，并在 description 中说明。
- 不输出 JSON 之外的文字。
"""

    user_prompt = f"""文档类型：{doc_type}
图片标题：{title or '未提供'}
邻近段落：
{context_text[:4000] if context_text else '无'}

请解析图片并输出统一图结构。"""

    content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
    ]
    # Diagram extraction is a bounded structured task, not an open-ended
    # reasoning conversation.  In particular Qwen thinking can otherwise run
    # for minutes before emitting JSON on CPU-only Ollama deployments.
    vision_model_config = dict(model_config or {})
    vision_model_config.update(
        {
            "temperature": 0.0,
            "top_p": 0.1,
            "num_predict": 2048,
            "reasoning": False,
            "format": "json",
            "stream": False,
            "timeout": min(float(vision_model_config.get("timeout") or 120), 120.0),
        }
    )
    response = llm_service.call_model(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        vision_model_config,
    )
    result = _parse_json(response)
    result.setdefault("diagram_type", doc_type)
    result.setdefault("description", "")
    result.setdefault("confidence", 0.0)
    result.setdefault("metadata", {})
    result.setdefault("vertices", [])
    result.setdefault("edges", [])

    vertex_ids = {str(v.get("id")) for v in result["vertices"] if isinstance(v, dict) and v.get("id")}
    valid_edges = []
    for edge in result["edges"]:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("source")) in vertex_ids and str(edge.get("target")) in vertex_ids:
            valid_edges.append(edge)
    result["edges"] = valid_edges
    return result
