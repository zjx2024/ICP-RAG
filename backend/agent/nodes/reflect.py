"""Reflect 节点：反思评估检索结果是否充分。

设计原则：
- 传入实际检索内容摘要（非仅数量），让 LLM 做智能判断
- 规则兜底：检测到用户需求未满足时强制重试
- 跟踪已调用工具，避免无限循环
"""
import json
import logging
from typing import Dict, List

from backend.core.llm_service import llm_service  # 只读引用，不修改
from backend.agent.model_role_service import agent_model_role_service

logger = logging.getLogger(__name__)

REFLECT_PROMPT = """评估检索结果是否充分回答用户问题。

用户问题: {query}
已调用工具: {tried_tools}

当前检索结果摘要:
{result_summary}

判断规则:
1. 如果用户想看图片/照片，但结果中没有 image 类型数据 → 需要 retry_table（sql_query 可查图片）
2. 如果用户想看表格/数据表/原始数据，但结果中没有 table 类型数据或 table_data → 需要 retry_table
3. 如果用户问具体实体关系但知识图谱无结果 → 需要 retry_graph
4. 如果文档检索结果与问题不相关或太少 → 需要 retry_document
5. 如果已有足够信息回答问题 → done

输出严格的 JSON（不要有其他文字）:
{{
  "need_more": false,
  "next_action": "done",
  "reasoning": "判断理由"
}}

next_action 取值:
- retry_document: 重新做向量检索
- retry_graph: 重新做知识图谱检索
- retry_table: 查询数据库（图片/表格）
- done: 信息充分，可以生成答案
"""


def _parse_json_response(text: str) -> dict:
    """解析 LLM 返回的 JSON（容错处理）"""
    try:
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start >= 0 and end > start:
                s = s[start:end + 1]
        return json.loads(s)
    except Exception as e:
        logger.warning(f"Reflect JSON 解析失败: {e}, 原始: {text[:200]}")
        return {"need_more": False, "next_action": "done", "reasoning": "解析失败，默认完成"}


def _get_tried_tools(state) -> List[str]:
    """从 tool_calls 记录中提取已调用的工具列表"""
    tried = []
    for tc in state.get("tool_calls", []):
        tool = tc.get("tool", "")
        if tool and tool not in tried:
            tried.append(tool)
    return tried


def _build_result_summary(state) -> str:
    """构建检索结果摘要（传给 LLM 做判断）"""
    parts = []

    vec_results = state.get("vector_results", [])
    if vec_results:
        titles = [r.get("title", "")[:50] for r in vec_results[:3]]
        parts.append(f"向量检索: {len(vec_results)} 条, 标题示例: {titles}")
    else:
        parts.append("向量检索: 0 条")

    kg_summary = state.get("kg_summary", "")
    if kg_summary:
        parts.append(f"知识图谱: 有摘要 ({len(kg_summary)} 字), 摘要前100字: {kg_summary[:100]}")
    else:
        parts.append("知识图谱: 无结果")

    sql_results = state.get("sql_results", [])
    if sql_results:
        types = [r.get("type", "") for r in sql_results]
        image_count = types.count("image")
        table_count = types.count("table")
        details = []
        if image_count:
            details.append(f"{image_count} 张图片")
        if table_count:
            details.append(f"{table_count} 个表格")
            # 列出表格标题
            table_titles = [r.get("title", "")[:50] for r in sql_results if r.get("type") == "table"]
            details.append(f"表格标题: {table_titles}")
        parts.append(f"数据库: {len(sql_results)} 条 ({', '.join(details) if details else '无图片/表格'})")
    else:
        parts.append("数据库: 0 条（无图片、无表格）")

    return "\n".join(parts)


def _rule_based_fallback(state, tried_tools: List[str]) -> dict:
    """规则兜底：即使 LLM 说 done，也检测明显遗漏"""
    # 同时检查原始 query 和 rewritten_query（rewritten 可能丢失关键词）
    original_query = (state.get("query") or "").lower()
    rewritten_query = (state.get("rewritten_query") or "").lower()
    combined = f"{original_query} {rewritten_query}"
    sql_results = state.get("sql_results", [])

    # 检查是否有图片/表格类型的结果
    has_images = any(r.get("type") == "image" for r in sql_results)
    has_tables = any(r.get("type") == "table" for r in sql_results)
    sql_tried = "sql_query" in tried_tools

    # 用户想看图片但没有图片结果，且 sql_query 还没调过
    image_keywords = ("图片", "image", "放映", "照片", "截图", "看图")
    want_images = any(kw in combined for kw in image_keywords)
    if want_images and not has_images and not sql_tried:
        return {"need_more": True, "next_action": "retry_table",
                "reasoning": "规则兜底: 用户想看图片但无图片结果，调用 sql_query 查询"}

    # 用户想看表格但没有表格结果，且 sql_query 还没调过
    table_keywords = ("表格", "数据表", "参数表", "标准表", "原始数据", "table", "查看表", "看看表")
    want_tables = any(kw in combined for kw in table_keywords)
    if want_tables and not has_tables and not sql_tried:
        return {"need_more": True, "next_action": "retry_table",
                "reasoning": "规则兜底: 用户想看表格但无表格结果，调用 sql_query 查询"}

    # 向量检索结果中提到"表"但数据库没有表格结果
    vec_results = state.get("vector_results", [])
    if vec_results and not has_tables and not sql_tried:
        vec_text = " ".join([r.get("content", "")[:200] for r in vec_results[:3]]).lower()
        if any(kw in vec_text for kw in ("表格", "数据表", "参数表", "标准表", "table")):
            return {"need_more": True, "next_action": "retry_table",
                    "reasoning": "规则兜底: 向量检索结果提及表格，但数据库未查到表格数据，补充 sql_query"}

    # 诊断类问题（为什么/原因/导致/故障）：图谱因果链是关键信息源，
    # 若图谱还未检索且无结果，补充一次知识图谱检索
    diag_keywords = ("为什么", "怎么回事", "原因", "导致", "由于", "造成", "故障",
                     "如何解决", "怎么解决", "是什么原因")
    want_diag = any(kw in combined for kw in diag_keywords)
    kg_tried = "kg_search" in tried_tools
    kg_summary = state.get("kg_summary", "")
    if want_diag and not kg_tried and not kg_summary:
        return {"need_more": True, "next_action": "retry_graph",
                "reasoning": "规则兜底: 诊断类问题但知识图谱未检索/无结果，补充图谱检索以获取因果链"}

    return None


async def reflect_node(state, config=None) -> Dict:
    """Reflect 节点：评估检索结果是否充分"""
    iteration = state.get("iteration", 0)
    # 最大反思轮次：读取 agent 配置的 max_reflections
    max_reflections = int(state.get("max_reflections", 3) or 3)

    # 超过最大反思次数，直接结束
    if iteration >= max_reflections:
        return {
            "need_more": False,
            "next_action": "done",
            "reflection": f"已达到最大反思次数 {max_reflections}，结束",
        }

    tried_tools = _get_tried_tools(state)

    agent_id = state.get("agent_id", "")
    chat_model_id = state.get("chat_model_id", "")
    if chat_model_id:
        from backend.storage.postgres_client import postgres_client
        model = postgres_client.get_model(chat_model_id)
    else:
        model = agent_model_role_service.get_model_by_role(agent_id, "reflect") if agent_id else None
    # 用智能体配置的模型采样参数覆盖模型默认值
    model = agent_model_role_service.apply_agent_llm_params(state, model)

    result_summary = _build_result_summary(state)

    prompt = REFLECT_PROMPT.format(
        query=state.get("query", ""),
        tried_tools=", ".join(tried_tools) if tried_tools else "无",
        result_summary=result_summary,
    )

    stream_queue = None
    if config and isinstance(config, dict):
        stream_queue = config.get("configurable", {}).get("stream_queue")

    try:
        # 流式调用：逐 token 推送到前端思考面板
        if stream_queue is not None:
            response = ""
            async for token in llm_service.stream_model(
                [{"role": "user", "content": prompt}], model
            ):
                response += token
                try:
                    await stream_queue.put({"type": "thinking_delta", "source": "reflect", "content": token})
                except Exception:
                    pass
        else:
            response = llm_service._call_model(
                [{"role": "user", "content": prompt}], model
            )
        parsed = _parse_json_response(response)
    except Exception as e:
        logger.error(f"Reflect 节点调用失败: {e}")
        parsed = {"need_more": False, "next_action": "done", "reasoning": "调用失败，默认完成"}

    # 规则兜底：LLM 说 done 但实际有遗漏
    if not parsed.get("need_more", False):
        fallback = _rule_based_fallback(state, tried_tools)
        if fallback:
            logger.info(f"Reflect 规则兜底触发: {fallback['reasoning']}")
            return {
                "need_more": True,
                "next_action": fallback["next_action"],
                "reflection": fallback["reasoning"],
                "result_summary": result_summary,
                "tried_tools": tried_tools,
            }

    # 防止重复调用已试过的工具（避免无限循环）
    next_action = parsed.get("next_action", "done")
    tool_map = {"retry_document": "vector_search", "retry_graph": "kg_search", "retry_table": "sql_query"}
    target_tool = tool_map.get(next_action, "")
    if target_tool and target_tool in tried_tools:
        # 该工具已经调过，不再重复，直接回答
        logger.info(f"Reflect: {next_action} 对应工具 {target_tool} 已调用过，直接回答")
        return {
            "need_more": False,
            "next_action": "done",
            "reflection": f"工具 {target_tool} 已调用过，不再重复",
            "result_summary": result_summary,
            "tried_tools": tried_tools,
        }

    return {
        "need_more": parsed.get("need_more", False),
        "next_action": next_action,
        "reflection": parsed.get("reasoning", ""),
        "result_summary": result_summary,
        "tried_tools": tried_tools,
    }
