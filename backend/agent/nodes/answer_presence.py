"""共享「字段在场断言」原语（answer.py 拆分的批 1）。

把散布在 answer.py 各域（history / target / fault）中的重复归一化与
“值是否在答案中出现”的判定逻辑收敛到这里。本模块只依赖标准库，
不 import answer.py / answer_contract / evidence 等节点模块，保证可独立
单测且移动函数时不产生循环依赖。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Set


def normalize_presence_text(value: Any) -> str:
    """NFKC + 去非字母数字，用于“源字段是否在答案中再次出现”的判定。

    history / target / fault 域的在场断言此前各自重复实现同一归一化
    （``re.sub(r"[^a-z0-9\\u4e00-\\u9fff]+", "", NFKC(value).casefold())``）。
    统一入口后，任一域的判定口径保持一致。
    """
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized)


def normalized_substring_present(
    source: str,
    normalized_answer: str,
    *,
    min_len: int = 6,
) -> bool:
    """source 归一化后是否是 normalized_answer 的子串。

    归一化已去过标点空白，因此这里的子串判定是“语义内容命中”，
    但仅接受足够长的片段（默认 ≥6 字符），避免单字被误判为事实在场。
    """
    source_normalized = normalize_presence_text(source)
    if not source_normalized:
        return False
    return len(source_normalized) >= min_len and source_normalized in normalized_answer


# 语义分词停用词：与 answer.py 原 _SEMANTIC_TOKEN_STOPWORDS 一致。
_SEMANTIC_TOKEN_STOPWORDS = {
    "系统",
    "模块",
    "故障",
    "异常",
    "文档",
    "记录",
    "资料",
    "当前",
    "可能",
    "主要",
    "相关",
}

# 内部绑定上下文标记：answer.py 在这些标记内写入确定性绑定内容，
# 语义分词必须剔除，避免机器标记参与“在场”判定。
_INTERNAL_BINDING_CONTEXTS = (
    "【历史案例上下文】",
    "【FTA最小割集回答】",
)


def semantic_tokens(value: Any) -> Set[str]:
    """提取答案/字段的可比语义 token（词 + 中文二元组），供重叠度判定。

    与 answer.py 原 _semantic_tokens 行为一致：先 NFKC+小写并去掉
    内部绑定上下文标记，再取英文 token 与中文连续片段及其二元组，
    过滤停用词。
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for marker in _INTERNAL_BINDING_CONTEXTS:
        text = text.replace(marker, "")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._*-]{1,}|[\u4e00-\u9fff]{2,}", text)
        if token not in _SEMANTIC_TOKEN_STOPWORDS
    }
    for run in re.findall(r"[\u4e00-\u9fff]{3,}", text):
        tokens.update(
            run[index:index + 2]
            for index in range(len(run) - 1)
            if run[index:index + 2] not in _SEMANTIC_TOKEN_STOPWORDS
        )
    return tokens


def approved_id_set(values: Iterable[Any], *, key: str = "evidence_id") -> Set[str]:
    """从批准集合提取去空的 ID 集合（收敛各域重复的 ``approved_ids = {...}``）。

    values 可以是证据 dict 列表（此时取 ``key`` 字段），也可以是字符串列表
    /可迭代值。过滤空串并保持无副作用。
    """
    result: Set[str] = set()
    for value in values or []:
        if isinstance(value, Mapping):
            raw = value.get(key)
        else:
            raw = value
        cleaned = str(raw or "").strip()
        if cleaned:
            result.add(cleaned)
    return result