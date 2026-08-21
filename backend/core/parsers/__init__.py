"""ICP 类型驱动解析架构：解析器注册表与三类解析器。"""
from backend.core.parsers.base import Parser, ParseContext, ParseResult
from backend.core.parsers.registry import ParserRegistry  # noqa: F401  导入即触发各解析器注册

__all__ = ["Parser", "ParseContext", "ParseResult", "ParserRegistry"]
