"""统一响应结构（新增骨架，供未来新代码使用，不影响存量响应）。"""
from typing import Any


def ok(data: Any = None, message: str = "success") -> dict:
    return {"code": 0, "message": message, "data": data}


def fail(code: int, message: str, data: Any = None) -> dict:
    return {"code": code, "message": message, "data": data}
