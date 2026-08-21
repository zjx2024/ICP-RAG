"""统一业务异常。"""
from typing import Optional


class BusinessError(Exception):
    """业务错误：由全局异常处理器转为统一响应结构。"""

    def __init__(self, message: str, code: int = 400, detail: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail


class NotFoundError(BusinessError):
    def __init__(self, message: str = "资源不存在", code: int = 404):
        super().__init__(message, code=code)


class ValidationError(BusinessError):
    def __init__(self, message: str = "参数错误", code: int = 422):
        super().__init__(message, code=code)
