"""ICP 类型驱动解析架构的公共入口。

``ParserRegistry`` 会导入所有带数据库依赖的领域解析器。将它惰性加载，
可以让 DocumentIR、表头识别和可靠性算法等纯模块在离线环境中独立使用和
测试；业务代码显式导入 ``ParserRegistry`` 时仍会完成原有的注册动作。
"""
from backend.core.parsers.base import Parser, ParseContext, ParseResult

__all__ = ["Parser", "ParseContext", "ParseResult", "ParserRegistry"]


def __getattr__(name: str):
    if name == "ParserRegistry":
        from backend.core.parsers.registry import ParserRegistry

        return ParserRegistry
    raise AttributeError(name)
