"""诊断会话标题的共享规则。

标题是导航元数据，不需要额外占用一次 LLM 调用；稳定的文本
归一化也使断线重试得到相同结果。
"""

import re


DEFAULT_SESSION_TITLE = "新诊断会话"
LEGACY_DEFAULT_SESSION_TITLE = "新会话"
DEFAULT_SESSION_TITLES = {
    DEFAULT_SESSION_TITLE,
    LEGACY_DEFAULT_SESSION_TITLE,
}
AUTO_TITLE_MAX_LENGTH = 32


def make_session_title(content: str) -> str:
    """把首问压缩为适合侧边栏的稳定标题。"""

    title = re.sub(r"\s+", " ", str(content or "")).strip()
    title = re.sub(r"^(?:#{1,6}|>|[-*+])\s+", "", title).strip()
    title = title.strip("\"'“”‘’")
    title = title.rstrip("。！？!?;；")
    if len(title) <= AUTO_TITLE_MAX_LENGTH:
        return title
    shortened = title[: AUTO_TITLE_MAX_LENGTH - 1].rstrip("，,。！？!?:：;； ")
    return f"{shortened}…" if shortened else DEFAULT_SESSION_TITLE
