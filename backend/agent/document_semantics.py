"""Shared source ontology derived from the seven-type product registry.

This teaches meanings to models; it never matches user vocabulary or grants
a tool, record identity, cross-document relation, or calculation.
"""
from backend.icp.document_types import DOCUMENT_TYPES


def document_semantics_prompt() -> str:
    catalog = "\n".join(f"- {row[0]}｜{row[1]}：{row[2]}" for row in DOCUMENT_TYPES)
    return ("【产品资料类型与语义】\n" + catalog
        + "\n资料名称代表不同来源，内容相似也不可互称或代替。按本轮所问资料理解来源，不能沿用上轮来源名称。"
          "接口查询/调用与实验开展状态是不同信息目标。字段都有值不等于回答了当前信息目标。"
          "没有验证跨资料身份关联，只能说明该关联未确认，不能推断另一资料没有功能、处理方法或接口。")
