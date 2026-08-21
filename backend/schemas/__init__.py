"""Pydantic DTO 层。

存量路由的请求/响应模型继续定义在 app.api.routers.* 内（保持契约不变）；
此目录放置共享、复用的 DTO，供未来新增接口使用。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """通用消息 DTO。"""

    role: str
    content: str


class PageResult(BaseModel):
    """通用分页 DTO。"""

    items: List[Dict[str, Any]]
    total: int
    page: int = 1
    page_size: int = 20
